"""
ADAP Pilot Experiment — LLM API 호출 공통 래퍼
OpenRouter (OpenAI 호환 API)를 통해 모든 LLM 호출을 단일화합니다.
"""
import os
import re
import json
import time
import logging
import difflib

import httpx
from openai import OpenAI

from config import OPENROUTER_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES

logger = logging.getLogger(__name__)

# ── OpenRouter 클라이언트 (OpenAI SDK 호환) ──────────
# 학교 네트워크 SSL 인증서 검증 문제 우회 (verify=False)
_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    http_client=httpx.Client(verify=False),
)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    OpenRouter(OpenAI 호환 chat.completions API)로 LLM 호출.
    MAX_RETRIES까지 지수 백오프 재시도 (429/5xx 에러 시).
    반환: LLM 응답 원문 텍스트.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status in (429, 500, 502, 503, 504) or status is None:
                wait = 2 ** attempt
                logger.warning(
                    "LLM call attempt %d/%d failed (%s). Retrying in %ds…",
                    attempt, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ── 응답 파싱 유틸리티 ──────────────────────────────
def parse_score(text: str) -> float:
    """
    응답 텍스트에서 1.0~5.0 사이 숫자를 정규식으로 추출.
    여러 개 발견되면 첫 번째 값 사용. 못 찾으면 ValueError.
    """
    # 소수점 포함 숫자를 추출한 뒤 1.0~5.0 범위의 첫 번째 값을 반환
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    for n in numbers:
        val = float(n)
        if 1.0 <= val <= 5.0:
            return val
    raise ValueError(f"Could not parse a score in [1.0, 5.0] from: {text!r}")


def parse_batch_scores(text: str, expected_ids: list[int]) -> dict[int, float]:
    """
    JSON 형식 응답을 파싱: [{"item_id": <id>, "score": <float>}, ...]
    expected_ids와 개수/id가 불일치하면 ParseError(ValueError).
    """
    # JSON 배열 부분만 추출 (응답에 부가 텍스트가 섞일 수 있으므로)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response: {text!r}")
    data = json.loads(match.group())

    result: dict[int, float] = {}
    for item in data:
        item_id = int(item["item_id"])
        score = float(item["score"])
        result[item_id] = max(1.0, min(5.0, score))

    # 검증: 기대한 id 집합과 일치하는지
    if set(result.keys()) != set(expected_ids):
        raise ValueError(
            f"ID mismatch. Expected {sorted(expected_ids)}, got {sorted(result.keys())}"
        )
    return result


def parse_ranking(text: str, candidate_titles: dict[int, str]) -> list[int]:
    """
    LLM이 출력한 순위 텍스트(제목 리스트)를 candidate_titles와 매칭하여
    movieId 순서 리스트로 변환.
    정확히 매칭 안 되면 difflib.SequenceMatcher로 fuzzy matching.
    끝까지 매칭 안 된 항목은 리스트 맨 뒤에 원래 순서대로 채움(누락 방지).
    """
    # 후보 제목 → movieId 역매핑
    title_to_id: dict[str, int] = {title: mid for mid, title in candidate_titles.items()}
    all_titles = list(title_to_id.keys())

    # 응답에서 줄 단위로 제목 추출 (번호 접두사 제거)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        # "1. Title", "1) Title", "1: Title" 등의 접두사 제거
        cleaned_line = re.sub(r"^\d+[\.\)\:\-]\s*", "", line).strip()
        if cleaned_line:
            cleaned.append(cleaned_line)

    matched_ids: list[int] = []
    remaining_ids = set(candidate_titles.keys())
    parse_failure_count = 0

    for query in cleaned:
        if not remaining_ids:
            break
        # 1) 정확 매칭
        if query in title_to_id and title_to_id[query] in remaining_ids:
            mid = title_to_id[query]
            matched_ids.append(mid)
            remaining_ids.discard(mid)
            continue

        # 2) Fuzzy 매칭 — 남은 후보 중에서만 탐색
        remaining_titles = {candidate_titles[mid]: mid for mid in remaining_ids}
        best_match = difflib.get_close_matches(query, remaining_titles.keys(), n=1, cutoff=0.4)
        if best_match:
            mid = remaining_titles[best_match[0]]
            matched_ids.append(mid)
            remaining_ids.discard(mid)
        else:
            parse_failure_count += 1

    if parse_failure_count > 0:
        logger.warning(
            "parse_ranking: %d lines could not be matched to any candidate title.",
            parse_failure_count,
        )

    # 누락된 항목을 원래 순서(candidate_titles 순)대로 뒤에 채움
    for mid in candidate_titles:
        if mid in remaining_ids:
            matched_ids.append(mid)

    return matched_ids