"""
Ours / Ours+History: 에이전트 판단 + 오케스트레이터 종합
"""
import json
import logging
import os
import threading

from src.llm_client import call_llm, parse_score, parse_batch_scores, parse_ranking
from prompts.templates import (
    AGENT_JUDGE_SINGLE_PROMPT,
    AGENT_JUDGE_BATCH_PROMPT,
    ORCHESTRATOR_RATING_PROMPT,
    ORCHESTRATOR_RATING_WITH_HISTORY_PROMPT,
    ORCHESTRATOR_RANKING_PROMPT,
    ORCHESTRATOR_RANKING_WITH_HISTORY_PROMPT,
)
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)

# ── History 디버그 로깅 설정 ──────────────────────────
# orchestrate_rating()에 history가 실제로 다르게 들어가는지 육안 확인용.
# 유저별로 최초 1회만 기록하고, 총 DEBUG_HISTORY_MAX_SAMPLES명까지만 저장한다.
_DEBUG_HISTORY_PATH = os.path.join(OUTPUT_DIR, "predictions", "debug_history_samples.txt")
_DEBUG_HISTORY_MAX_SAMPLES = 5
_debug_history_lock = threading.Lock()
_debug_history_seen_users: set = set()


def _log_history_sample(user_weights: dict, history: list[str], prompt: str) -> None:
    """
    history가 포함된 orchestrator 프롬프트를 유저별로 최초 1회, 최대
    _DEBUG_HISTORY_MAX_SAMPLES명까지 파일에 이어쓰기로 저장한다.
    user_id 자체는 orchestrate_rating() 시그니처에 없으므로, 호출부(run_pilot.py)에서
    구분 가능하도록 history 리스트의 내용(제목들)을 식별자로 사용해 중복을 판단한다.
    """
    # history 리스트를 간단한 식별 키로 사용 (동일 유저는 같은 history를 반환하므로 충분)
    history_key = tuple(history[:3])  # 앞 3개 제목만으로도 유저 구분에 충분
    with _debug_history_lock:
        if history_key in _debug_history_seen_users:
            return
        if len(_debug_history_seen_users) >= _DEBUG_HISTORY_MAX_SAMPLES:
            return
        _debug_history_seen_users.add(history_key)

        os.makedirs(os.path.dirname(_DEBUG_HISTORY_PATH), exist_ok=True)
        with open(_DEBUG_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"[sample #{len(_debug_history_seen_users)}]\n")
            f.write(f"user_weights: {user_weights}\n")
            f.write(f"history (first 3 shown as key): {history_key}\n")
            f.write("-" * 70 + "\n")
            f.write("FULL PROMPT SENT TO ORCHESTRATOR:\n")
            f.write(prompt + "\n")
            f.write("=" * 70 + "\n\n")


# ── Agent Judge ──────────────────────────────────────

def agent_judge_single(
    agent_system_prompt: str,
    item_info: dict,
) -> dict:
    """
    한 에이전트가 아이템 1개를 평가 (레이팅 예측 트랙용).
    item_info: {"title": str, "genres": list[str]}
    반환: {"score": float, "reason": str}

    NOTE: 유저 정보를 전혀 받지 않음. System Prompt에 페르소나가 고정되어 있고 item_info만 봄.
    """
    user_prompt = AGENT_JUDGE_SINGLE_PROMPT.format(
        title=item_info["title"],
        genres=", ".join(item_info["genres"]),
    )

    response = call_llm(
        system_prompt=agent_system_prompt,
        user_prompt=user_prompt,
    )

    # JSON 파싱 시도
    try:
        # JSON 부분 추출
        import re
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            score = float(parsed.get("score", 3.0))
            score = max(1.0, min(5.0, score))
            reason = str(parsed.get("reason", ""))
            return {"score": score, "reason": reason}
    except (json.JSONDecodeError, ValueError):
        pass

    # JSON 파싱 실패 시 점수만이라도 추출
    try:
        score = parse_score(response)
        return {"score": score, "reason": response[:100]}
    except ValueError:
        logger.warning("agent_judge_single: Could not parse response: %s", response[:200])
        return {"score": 3.0, "reason": "parse_failure"}


def agent_judge_batch(
    agent_system_prompt: str,
    items: list[dict],
) -> list[dict]:
    """
    한 에이전트가 후보 20개를 한 번에 평가 (랭킹 트랙용, 배치 호출로 비용 절감).
    items: [{"item_id": int, "title": str, "genres": list[str]}, ...] 20개
    반환: [{"item_id": int, "score": float}, ...] 20개

    NOTE: 유저 정보를 전혀 받지 않음. LLM 출력 파싱 실패 시 1회 재시도.
    """
    n = len(items)
    items_text = "\n".join(
        f"  {item['item_id']}: {item['title']} (장르: {', '.join(item['genres'])})"
        for item in items
    )
    expected_ids = [item["item_id"] for item in items]

    user_prompt = AGENT_JUDGE_BATCH_PROMPT.format(
        n=n,
        items_text=items_text,
    )

    # 최대 2번 시도 (1회 실패 시 재시도)
    for attempt in range(2):
        response = call_llm(
            system_prompt=agent_system_prompt,
            user_prompt=user_prompt,
        )
        try:
            scores = parse_batch_scores(response, expected_ids)
            return [{"item_id": iid, "score": scores[iid]} for iid in expected_ids]
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            if attempt == 0:
                logger.warning("agent_judge_batch: Parse failed (attempt 1), retrying. Error: %s", e)
            else:
                logger.error("agent_judge_batch: Parse failed after retry. Error: %s", e)
                # 파싱 실패 시 모든 아이템에 중립 점수 3.0 부여
                return [{"item_id": iid, "score": 3.0} for iid in expected_ids]

    # 도달 불가능하지만 안전을 위해
    return [{"item_id": iid, "score": 3.0} for iid in expected_ids]


# ── Orchestrator ─────────────────────────────────────

def orchestrate_rating(
    agent_outputs: dict,
    user_weights: dict,
    history: list[str] | None,
) -> float:
    """
    레이팅 예측 트랙: LLM 호출 버전.
    agent_outputs: {factor_id: {"score": float, "reason": str}}
    user_weights: {factor_id: float}
    history: include_history=True일 때만 최근 시청 이력 제목 리스트, 아니면 None.

    반환: 최종 예측 평점 (float, 1.0~5.0 범위로 clip)
    """
    weights_text = ", ".join(f"{fid}: {w:.3f}" for fid, w in user_weights.items())
    agent_outputs_text = "\n".join(
        f"  {fid}: 점수 {out['score']:.1f} (이유: {out['reason']})"
        for fid, out in agent_outputs.items()
    )

    if history is not None:
        history_text = "\n".join(f"  - {title}" for title in history)
        prompt = ORCHESTRATOR_RATING_WITH_HISTORY_PROMPT.format(
            weights_text=weights_text,
            agent_outputs_text=agent_outputs_text,
            history_text=history_text,
        )
        # ★ 디버그: history가 유저마다 실제로 다르게 들어가는지 샘플 저장
        _log_history_sample(user_weights, history, prompt)
    else:
        prompt = ORCHESTRATOR_RATING_PROMPT.format(
            weights_text=weights_text,
            agent_outputs_text=agent_outputs_text,
        )

    response = call_llm(
        system_prompt="You are a recommendation orchestrator that combines multiple agent judgments.",
        user_prompt=prompt,
    )

    try:
        score = parse_score(response)
    except ValueError:
        # fallback: 가중합으로 계산
        score = sum(
            user_weights.get(fid, 0) * out["score"]
            for fid, out in agent_outputs.items()
        )
        logger.warning("orchestrate_rating: Could not parse LLM response, using weighted sum. Response: %s",
                        response[:200])

    return float(max(1.0, min(5.0, score)))


def orchestrate_ranking(
    agent_outputs_per_item: dict,
    user_weights: dict,
    candidate_items: list[dict],
    history: list[str] | None,
) -> list[int]:
    """
    랭킹 트랙: LLM 기반 listwise 랭킹.
    agent_outputs_per_item: {item_id: {factor_id: score}}
    candidate_items: [{"item_id": int, "title": str, "genres": list[str]}, ...]
    history: include_history=True일 때만 최근 시청 이력 제목 리스트, 아니면 None.

    반환: item_id를 LLM이 매긴 선호 순서대로 정렬한 리스트 (길이 20)

    NOTE: 이전에는 후보마다 개별 호출(pointwise)로 LLM에게 예상 평점을 물어 정렬했으나,
    같은 history가 매 호출마다 반복 주입되면서 후보별 변별력이 사라지는 문제가 있어
    (모든 후보 점수가 비슷하게 수렴 → 동점 발생) 후보 전체를 한 프롬프트에 넣고
    한 번에 순위를 매기는 listwise 방식으로 전환했다. 호출 수도 후보 수(20)에서 1로 줄어든다.
    """
    weights_text = ", ".join(f"{fid}: {w:.3f}" for fid, w in user_weights.items())
    items_text = "\n".join(
        f"  {c['title']} (장르: {', '.join(c['genres'])}) — " + ", ".join(
            f"{fid}: {agent_outputs_per_item[c['item_id']].get(fid, 3.0):.1f}"
            for fid in user_weights
        )
        for c in candidate_items
    )
    candidate_titles = {c["item_id"]: c["title"] for c in candidate_items}

    if history is not None:
        history_text = "\n".join(f"  - {title}" for title in history)
        prompt = ORCHESTRATOR_RANKING_WITH_HISTORY_PROMPT.format(
            weights_text=weights_text,
            n=len(candidate_items),
            items_text=items_text,
            history_text=history_text,
        )
    else:
        prompt = ORCHESTRATOR_RANKING_PROMPT.format(
            weights_text=weights_text,
            n=len(candidate_items),
            items_text=items_text,
        )

    response = call_llm(
        system_prompt="You are a recommendation orchestrator that combines multiple agent judgments to rank candidates.",
        user_prompt=prompt,
    )

    return parse_ranking(response, candidate_titles)