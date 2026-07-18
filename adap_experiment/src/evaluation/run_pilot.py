"""
ADAP Pilot Experiment — 전체 파이프라인 실행 + 결과 집계

순서:
1. 데이터 로딩 + 시간순 분할
2. Feature Engineering
3. N_PILOT_USERS명 무작위 샘플링
4. 후보셋 생성
5. 통계 베이스라인 모델 학습 (DEBUG_SKIP_OTHER_METHODS=True면 건너뜀)
6. FACTOR_METHODS 루프: Factor Discovery → Agent Synthesis → Weight Estimation
7. 레이팅 예측 트랙 (베이스라인 1회 + Ours method별) — ThreadPoolExecutor로 병렬화, 체크포인트/재개 지원
8. 랭킹 트랙 (DEBUG_SKIP_OTHER_METHODS=True면 전체 건너뜀) — 동일하게 병렬화/체크포인트 지원
9. 지표 계산 + 결과 저장
"""
import os
import sys
import json
import logging
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트가 sys.path에 있도록 보장
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    RAW_DATA_DIR, OUTPUT_DIR,
    K_FACTORS, N_PILOT_USERS, RANDOM_SEED, HISTORY_N,
    FACTOR_METHODS,
    DEBUG_SKIP_OTHER_METHODS,   # ★ 추가: 디버깅용 스킵 플래그
    MAX_WORKERS, CHECKPOINT_INTERVAL,  # ★ 추가: 병렬화 / 체크포인트 파라미터
)
from src.data_prep.load_movielens import load_ratings, load_movies
from src.data_prep.temporal_split import temporal_split
from src.data_prep.feature_engineering import (
    compute_item_popularity,
    build_user_feature_matrix,
    build_item_feature_matrix,
)
from src.data_prep.candidate_set_builder import build_candidate_sets
from src.factor.factor_discovery import run_factor_discovery, interpret_factors
from src.factor.weight_estimation import compute_user_weights
from src.agents.agent_synthesis import synthesize_all_agents
from src.agents.orchestrator import (
    agent_judge_single,
    agent_judge_batch,
    orchestrate_rating,
    orchestrate_ranking,
)
from src.baselines.stat_baseline import UserKNNModel, mf_train, mf_predict, rank_by_stat_method
from src.baselines.llm_direct import (
    baseline_predict_rating,
    baseline_rank,
    baseline_predict_rating_with_history,
    baseline_rank_with_history,
    chatgpt_direct_predict_rating,
    chatgpt_direct_rank,
)
from src.evaluation.metrics import mae, rmse, recall_at_k, ndcg_at_k
from src.llm_client import get_rate_limit_error_count

# ── 로깅 설정 ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── 체크포인트 파일 경로 ─────────────────────────────
CHECKPOINT_RATING_PATH = os.path.join(OUTPUT_DIR, "predictions", "_checkpoint_rating.jsonl")
CHECKPOINT_RANKING_PATH = os.path.join(OUTPUT_DIR, "predictions", "_checkpoint_ranking.jsonl")


def _get_item_info(movie_id: int, movies_df: pd.DataFrame) -> dict:
    """movieId → {"title": str, "genres": list[str]}"""
    row = movies_df[movies_df["movieId"] == movie_id]
    if row.empty:
        return {"title": f"Unknown({movie_id})", "genres": []}
    r = row.iloc[0]
    return {"title": r["title"], "genres": r["genres"]}


def _get_user_history(user_id: int, train_df: pd.DataFrame, movies_df: pd.DataFrame, n: int = HISTORY_N) -> list[dict]:
    """유저의 train 데이터 중 최근 n개 이력을 [{"title": str, "rating": float}, ...] 형태로 반환."""
    user_rows = train_df[train_df["userId"] == user_id].sort_values("timestamp", ascending=False).head(n)
    result = []
    for _, row in user_rows.iterrows():
        info = _get_item_info(int(row["movieId"]), movies_df)
        result.append({"title": info["title"], "rating": float(row["rating"])})
    return result


def _get_user_history_titles(user_id: int, train_df: pd.DataFrame, movies_df: pd.DataFrame, n: int = HISTORY_N) -> list[str]:
    """유저의 train 데이터 중 최근 n개 이력 제목 리스트."""
    history = _get_user_history(user_id, train_df, movies_df, n)
    return [f"{h['title']} ({h['rating']:.1f}점)" for h in history]


# ══════════════════════════════════════════════════════
# 체크포인트 유틸리티
# ══════════════════════════════════════════════════════

def _load_checkpoint(path: str) -> list[dict]:
    """체크포인트 jsonl 파일을 읽어서 레코드 리스트로 반환. 파일이 없으면 빈 리스트."""
    records: list[dict] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Checkpoint line could not be parsed, skipping: %s", line[:100])
    return records


class CheckpointWriter:
    """
    처리 단위(행/유저)가 완료될 때마다 add()로 결과를 버퍼에 쌓고,
    CHECKPOINT_INTERVAL 단위마다 디스크에 append + flush + fsync 한다.
    스레드 세이프.
    """

    def __init__(self, path: str, interval: int):
        self.path = path
        self.interval = interval
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._units_since_flush = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add(self, records: list[dict]) -> None:
        """records: 완료된 처리 단위 1개(행 또는 유저)에서 생성된 체크포인트 레코드들."""
        with self._lock:
            self._buffer.extend(records)
            self._units_since_flush += 1
            if self._units_since_flush >= self.interval:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buffer:
            with open(self.path, "a", encoding="utf-8") as f:
                for rec in self._buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._buffer = []
        self._units_since_flush = 0


def _mark_checkpoint_done(path: str) -> None:
    """파이프라인이 정상 완료되면 체크포인트 파일을 _done.jsonl로 보관(선택 사항)."""
    if not os.path.exists(path):
        return
    done_path = path[:-len(".jsonl")] + "_done.jsonl"
    try:
        if os.path.exists(done_path):
            os.remove(done_path)
        os.rename(path, done_path)
    except OSError as e:
        logger.warning("Could not rename checkpoint file %s -> %s: %s", path, done_path, e)


# ══════════════════════════════════════════════════════
# 에이전트 병렬 호출 헬퍼 (K_FACTORS개를 동시에 호출)
# ══════════════════════════════════════════════════════

def _judge_factors_single(agent_prompts: dict, factor_ids: list, item_info: dict,
                           agent_executor: ThreadPoolExecutor) -> dict:
    futures = {
        agent_executor.submit(agent_judge_single, agent_prompts[fid], item_info): fid
        for fid in factor_ids
    }
    outputs: dict = {}
    for fut in as_completed(futures):
        fid = futures[fut]
        outputs[fid] = fut.result()
    return outputs


def _judge_factors_batch(agent_prompts: dict, factor_ids: list, candidate_items: list,
                          agent_executor: ThreadPoolExecutor) -> dict:
    futures = {
        agent_executor.submit(agent_judge_batch, agent_prompts[fid], candidate_items): fid
        for fid in factor_ids
    }
    results: dict = {}
    for fut in as_completed(futures):
        fid = futures[fut]
        results[fid] = fut.result()
    return results


# ══════════════════════════════════════════════════════
# 8a. 레이팅 트랙 — 행 1개 처리 단위
# ══════════════════════════════════════════════════════

def _process_rating_row(row, *, done_methods: set, train_df: pd.DataFrame, movies_df: pd.DataFrame,
                         user_feature_matrix: pd.DataFrame, userknn_model, mf_model,
                         factor_results: dict, agent_executor: ThreadPoolExecutor):
    """
    한 (userId, movieId) 테스트 행에 대해 모든 방법론 결과를 계산.
    done_methods에 이미 있는 method는 재계산하지 않고 건너뜀(재개용).
    반환: (uid, mid, records) — records: [{"method", "pred", "actual", "failed"}, ...]
    """
    uid = int(row["userId"])
    mid = int(row["movieId"])
    actual = float(row["rating"])
    item_info = _get_item_info(mid, movies_df)
    records: list[dict] = []

    if not DEBUG_SKIP_OTHER_METHODS:
        # --- UserKNN (1회) ---
        if "UserKNN" not in done_methods:
            pred = userknn_model.predict(uid, mid)
            records.append({"method": "UserKNN", "pred": pred, "actual": actual, "failed": False})

        # --- MF (1회) ---
        if "MF" not in done_methods:
            pred = mf_predict(mf_model, uid, mid)
            records.append({"method": "MF", "pred": pred, "actual": actual, "failed": False})

        # --- Baseline 19dim→LLM (1회) ---
        if "Baseline" not in done_methods:
            try:
                uf = user_feature_matrix.loc[uid] if uid in user_feature_matrix.index else pd.Series(dtype=float)
                pred = baseline_predict_rating(uf, item_info)
                records.append({"method": "Baseline", "pred": pred, "actual": actual, "failed": False})
            except (ValueError, Exception) as e:
                records.append({"method": "Baseline", "pred": None, "actual": actual, "failed": True})
                logger.warning("Baseline rating parse failure for user=%d item=%d: %s", uid, mid, e)

        # --- Baseline+History: 19dim 피처 + 최근 시청 이력 → LLM (1회) ---
        if "Baseline+History" not in done_methods:
            try:
                uf = user_feature_matrix.loc[uid] if uid in user_feature_matrix.index else pd.Series(dtype=float)
                history = _get_user_history(uid, train_df, movies_df)
                pred = baseline_predict_rating_with_history(uf, history, item_info)
                records.append({"method": "Baseline+History", "pred": pred, "actual": actual, "failed": False})
            except (ValueError, Exception) as e:
                records.append({"method": "Baseline+History", "pred": None, "actual": actual, "failed": True})
                logger.warning("Baseline+History rating parse failure for user=%d item=%d: %s", uid, mid, e)

        # --- ChatGPT-Direct (1회) ---
        if "ChatGPT-Direct" not in done_methods:
            try:
                history = _get_user_history(uid, train_df, movies_df)
                pred = chatgpt_direct_predict_rating(history, item_info)
                records.append({"method": "ChatGPT-Direct", "pred": pred, "actual": actual, "failed": False})
            except (ValueError, Exception) as e:
                records.append({"method": "ChatGPT-Direct", "pred": None, "actual": actual, "failed": True})
                logger.warning("ChatGPT-Direct rating parse failure for user=%d item=%d: %s", uid, mid, e)

    # --- Ours / Ours+History: method별로 각각 실행 (항상 실행됨) ---
    for method in FACTOR_METHODS:
        tag = method.upper()
        fr = factor_results[method]
        factor_ids = fr["factor_ids"]
        agent_prompts = fr["agent_prompts"]
        user_weights_df = fr["user_weights_df"]

        ours_name = f"Ours ({tag})"
        ours_hist_name = f"Ours+History ({tag})"
        need_ours = ours_name not in done_methods
        need_ours_hist = ours_hist_name not in done_methods
        if not need_ours and not need_ours_hist:
            continue

        agent_outputs = None
        uw = {fid: float(user_weights_df.loc[uid, fid]) for fid in factor_ids} \
            if uid in user_weights_df.index else {fid: 1.0 / len(factor_ids) for fid in factor_ids}

        if need_ours:
            try:
                agent_outputs = _judge_factors_single(agent_prompts, factor_ids, item_info, agent_executor)
                pred = orchestrate_rating(agent_outputs, uw, history=None)
                records.append({"method": ours_name, "pred": pred, "actual": actual, "failed": False})
            except Exception as e:
                records.append({"method": ours_name, "pred": None, "actual": actual, "failed": True})
                logger.warning("Ours (%s) rating failure for user=%d item=%d: %s", tag, uid, mid, e)

        if need_ours_hist:
            try:
                if agent_outputs is None:
                    agent_outputs = _judge_factors_single(agent_prompts, factor_ids, item_info, agent_executor)
                hist_titles = _get_user_history_titles(uid, train_df, movies_df)
                pred = orchestrate_rating(agent_outputs, uw, history=hist_titles)
                records.append({"method": ours_hist_name, "pred": pred, "actual": actual, "failed": False})
            except Exception as e:
                records.append({"method": ours_hist_name, "pred": None, "actual": actual, "failed": True})
                logger.warning("Ours+History (%s) rating failure for user=%d item=%d: %s", tag, uid, mid, e)

    return uid, mid, records


# ══════════════════════════════════════════════════════
# 8b. 랭킹 트랙 — 유저 1명 처리 단위
# ══════════════════════════════════════════════════════

def _process_ranking_user(uid: int, *, done_methods: set, candidate_sets: dict, train_df: pd.DataFrame,
                           movies_df: pd.DataFrame, user_feature_matrix: pd.DataFrame,
                           userknn_model, mf_model, factor_results: dict,
                           agent_executor: ThreadPoolExecutor):
    """
    한 유저에 대해 모든 방법론의 랭킹 결과(recall@10, ndcg@10)를 계산.
    반환: (uid, records) — records: [{"method", "recall", "ndcg", "failed"}, ...]
    """
    cs = candidate_sets[uid]
    positive_id = cs["positive"]
    candidate_ids = cs["candidates"]

    candidate_items = []
    for cid in candidate_ids:
        info = _get_item_info(cid, movies_df)
        candidate_items.append({"item_id": cid, "title": info["title"], "genres": info["genres"]})

    records: list[dict] = []

    # --- UserKNN (1회) ---
    if "UserKNN" not in done_methods:
        ranked = rank_by_stat_method(userknn_model.predict, uid, candidate_ids)
        records.append({
            "method": "UserKNN",
            "recall": recall_at_k(ranked, positive_id),
            "ndcg": ndcg_at_k(ranked, positive_id),
            "failed": False,
        })

    # --- MF (1회) ---
    if "MF" not in done_methods:
        ranked = rank_by_stat_method(lambda u, i: mf_predict(mf_model, u, i), uid, candidate_ids)
        records.append({
            "method": "MF",
            "recall": recall_at_k(ranked, positive_id),
            "ndcg": ndcg_at_k(ranked, positive_id),
            "failed": False,
        })

    # --- Baseline 19dim→LLM (1회, 배치) ---
    if "Baseline" not in done_methods:
        try:
            uf = user_feature_matrix.loc[uid] if uid in user_feature_matrix.index else pd.Series(dtype=float)
            ranked = baseline_rank(uf, candidate_items)
            records.append({
                "method": "Baseline",
                "recall": recall_at_k(ranked, positive_id),
                "ndcg": ndcg_at_k(ranked, positive_id),
                "failed": False,
            })
        except Exception as e:
            records.append({"method": "Baseline", "recall": None, "ndcg": None, "failed": True})
            logger.warning("Baseline ranking failure for user=%d: %s", uid, e)

    # --- Baseline+History: 19dim 피처 + 최근 시청 이력 → LLM (1회, 배치) ---
    if "Baseline+History" not in done_methods:
        try:
            uf = user_feature_matrix.loc[uid] if uid in user_feature_matrix.index else pd.Series(dtype=float)
            history = _get_user_history(uid, train_df, movies_df)
            ranked = baseline_rank_with_history(uf, history, candidate_items)
            records.append({
                "method": "Baseline+History",
                "recall": recall_at_k(ranked, positive_id),
                "ndcg": ndcg_at_k(ranked, positive_id),
                "failed": False,
            })
        except Exception as e:
            records.append({"method": "Baseline+History", "recall": None, "ndcg": None, "failed": True})
            logger.warning("Baseline+History ranking failure for user=%d: %s", uid, e)

    # --- ChatGPT-Direct (1회, 배치) ---
    if "ChatGPT-Direct" not in done_methods:
        try:
            history = _get_user_history(uid, train_df, movies_df)
            ranked = chatgpt_direct_rank(history, candidate_items)
            records.append({
                "method": "ChatGPT-Direct",
                "recall": recall_at_k(ranked, positive_id),
                "ndcg": ndcg_at_k(ranked, positive_id),
                "failed": False,
            })
        except Exception as e:
            records.append({"method": "ChatGPT-Direct", "recall": None, "ndcg": None, "failed": True})
            logger.warning("ChatGPT-Direct ranking failure for user=%d: %s", uid, e)

    # --- Ours & Ours+History: method별로 각각 실행 ---
    for method in FACTOR_METHODS:
        tag = method.upper()
        fr = factor_results[method]
        factor_ids = fr["factor_ids"]
        agent_prompts = fr["agent_prompts"]
        user_weights_df = fr["user_weights_df"]

        ours_name = f"Ours ({tag})"
        ours_hist_name = f"Ours+History ({tag})"
        need_ours = ours_name not in done_methods
        need_ours_hist = ours_hist_name not in done_methods
        if not need_ours and not need_ours_hist:
            continue

        # 에이전트 배치 판단은 Ours/Ours+History 둘 중 하나만 필요해도 공통으로 필요하므로 항상 계산
        try:
            agent_batch_results = _judge_factors_batch(agent_prompts, factor_ids, candidate_items, agent_executor)

            agent_outputs_per_item: dict[int, dict[str, float]] = {cid: {} for cid in candidate_ids}
            for fid, batch in agent_batch_results.items():
                for item in batch:
                    agent_outputs_per_item[item["item_id"]][fid] = item["score"]

            uw = {fid: float(user_weights_df.loc[uid, fid]) for fid in factor_ids} \
                if uid in user_weights_df.index else {fid: 1.0 / len(factor_ids) for fid in factor_ids}
        except Exception as e:
            if need_ours:
                records.append({"method": ours_name, "recall": None, "ndcg": None, "failed": True})
            if need_ours_hist:
                records.append({"method": ours_hist_name, "recall": None, "ndcg": None, "failed": True})
            logger.warning("Ours/Ours+History (%s) agent judging failure for user=%d: %s", tag, uid, e)
            continue

        # Ours (no history) — listwise 랭킹 (후보 전체를 한 프롬프트에 넣고 LLM이 한 번에 순위 매김)
        if need_ours:
            try:
                ranked = orchestrate_ranking(agent_outputs_per_item, uw, candidate_items, history=None)
                records.append({
                    "method": ours_name,
                    "recall": recall_at_k(ranked, positive_id),
                    "ndcg": ndcg_at_k(ranked, positive_id),
                    "failed": False,
                })
            except Exception as e:
                records.append({"method": ours_name, "recall": None, "ndcg": None, "failed": True})
                logger.warning("Ours (%s) ranking failure for user=%d: %s", tag, uid, e)

        # Ours+History
        if need_ours_hist:
            try:
                hist_titles = _get_user_history_titles(uid, train_df, movies_df)
                ranked = orchestrate_ranking(agent_outputs_per_item, uw, candidate_items, history=hist_titles)
                records.append({
                    "method": ours_hist_name,
                    "recall": recall_at_k(ranked, positive_id),
                    "ndcg": ndcg_at_k(ranked, positive_id),
                    "failed": False,
                })
            except Exception as e:
                records.append({"method": ours_hist_name, "recall": None, "ndcg": None, "failed": True})
                logger.warning("Ours+History (%s) ranking failure for user=%d: %s", tag, uid, e)

    return uid, records


# ══════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ADAP Pilot Experiment — Pipeline Start")
    if DEBUG_SKIP_OTHER_METHODS:
        logger.info("★ DEBUG_SKIP_OTHER_METHODS=True — "
                     "UserKNN/MF/Baseline/Baseline+History/ChatGPT-Direct 및 랭킹 트랙 전체를 건너뜁니다.")
    logger.info("MAX_WORKERS=%d, CHECKPOINT_INTERVAL=%d", MAX_WORKERS, CHECKPOINT_INTERVAL)
    logger.info("=" * 60)

    # ── 1. 데이터 로딩 ────────────────────────────────
    logger.info("[Step 1] Loading MovieLens-1M data…")
    ratings_df = load_ratings(os.path.join(RAW_DATA_DIR, "ratings.dat"))
    movies_df = load_movies(os.path.join(RAW_DATA_DIR, "movies.dat"))
    logger.info("  Ratings: %d rows, Movies: %d rows", len(ratings_df), len(movies_df))

    # ── 2. 시간순 분할 ────────────────────────────────
    logger.info("[Step 2] Temporal train/test split…")
    train_df, test_df = temporal_split(ratings_df)

    # ── 3. Feature Engineering ────────────────────────
    logger.info("[Step 3] Feature engineering…")
    item_popularity = compute_item_popularity(train_df)
    user_feature_matrix = build_user_feature_matrix(train_df, movies_df, item_popularity)
    item_feature_matrix = build_item_feature_matrix(movies_df, item_popularity)

    # ── 4. 파일럿 유저 샘플링 ─────────────────────────
    logger.info("[Step 4] Sampling %d pilot users…", N_PILOT_USERS)
    rng = np.random.RandomState(RANDOM_SEED)
    all_test_users = test_df["userId"].unique()
    if len(all_test_users) <= N_PILOT_USERS:
        pilot_users = all_test_users.tolist()
    else:
        pilot_users = rng.choice(all_test_users, size=N_PILOT_USERS, replace=False).tolist()
    pilot_users = [int(u) for u in pilot_users]
    logger.info("  Sampled %d pilot users", len(pilot_users))

    # 파일럿 유저만의 train/test 서브셋
    pilot_train = train_df[train_df["userId"].isin(pilot_users)]
    pilot_test = test_df[test_df["userId"].isin(pilot_users)]

    # ── 5. 후보셋 생성 ────────────────────────────────
    logger.info("[Step 5] Building candidate sets…")
    candidate_sets = build_candidate_sets(pilot_train, pilot_test, movies_df, seed=RANDOM_SEED)
    valid_users = [u for u in pilot_users if u in candidate_sets]
    logger.info("  Valid users (with candidate sets): %d", len(valid_users))

    # ── 6. Factor Discovery + Agent Synthesis + Weight Estimation ──
    #    FACTOR_METHODS 루프: method별로 결과를 별도 보관
    logger.info("[Step 6] Factor discovery / Agent synthesis / Weight estimation "
                "for methods: %s", FACTOR_METHODS)

    factor_results: dict[str, dict] = {}  # {"nmf": {...}, "fa": {...}}

    for method in FACTOR_METHODS:
        logger.info("  ── method=%s ──", method.upper())
        loading_matrix, user_scores = run_factor_discovery(
            user_feature_matrix, k=K_FACTORS, method=method,
        )
        interpreted = interpret_factors(loading_matrix, top_n=3)
        agent_prompts = synthesize_all_agents(
            loading_matrix, interpreted, method=method,
        )  # 캐시 파일명에 method 포함
        user_weights_df = compute_user_weights(
            user_feature_matrix, loading_matrix, method=method,
        )

        factor_results[method] = {
            "loading_matrix": loading_matrix,
            "agent_prompts": agent_prompts,
            "user_weights_df": user_weights_df,
            "factor_ids": list(agent_prompts.keys()),
        }

    # ── 7. 통계 베이스라인 모델 학습 ─────────────────
    userknn_model = None
    mf_model = None
    if not DEBUG_SKIP_OTHER_METHODS:
        logger.info("[Step 7] Training statistical baseline models…")
        userknn_model = UserKNNModel(train_df, k_neighbors=20)
        mf_model = mf_train(train_df)
    else:
        logger.info("[Step 7] Skipped (DEBUG_SKIP_OTHER_METHODS=True)")

    # ── 8. 방법론별 평가 ─────────────────────────────
    logger.info("[Step 8] Running methods on both tracks…")

    # 결과 저장용
    results_records: list[dict] = []
    preds_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    # 에이전트 판단 + 랭킹 트랙 pointwise 오케스트레이터(LLM) 호출용 공유 풀
    # (행/유저 단위 풀과는 별도 풀이라 서로 블로킹해도 데드락이 나지 않음)
    agent_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # ─────────────────────────────────────────────────
    # 8a. 레이팅 예측 트랙
    # ─────────────────────────────────────────────────
    logger.info("── Rating Prediction Track ──")

    # 대상: valid_users의 test_df 내 모든 (userId, movieId, rating) 행
    rating_test_rows = pilot_test[pilot_test["userId"].isin(valid_users)].copy()
    logger.info("  Rating track: %d test rows across %d users",
                len(rating_test_rows), rating_test_rows["userId"].nunique())

    # 베이스라인 5개 (1번만 실행, DEBUG_SKIP_OTHER_METHODS=True면 항상 빈 상태로 유지)
    baseline_methods_rating: dict[str, dict] = {
        "UserKNN":          {"preds": [], "actuals": [], "parse_failures": 0},
        "MF":               {"preds": [], "actuals": [], "parse_failures": 0},
        "Baseline":         {"preds": [], "actuals": [], "parse_failures": 0},
        "Baseline+History": {"preds": [], "actuals": [], "parse_failures": 0},
        "ChatGPT-Direct":   {"preds": [], "actuals": [], "parse_failures": 0},
    }

    # Ours 변형 (method별로 생성)
    ours_methods_rating: dict[str, dict] = {}
    for method in FACTOR_METHODS:
        tag = method.upper()
        ours_methods_rating[f"Ours ({tag})"] = {"preds": [], "actuals": [], "parse_failures": 0}
        ours_methods_rating[f"Ours+History ({tag})"] = {"preds": [], "actuals": [], "parse_failures": 0}

    methods_rating = {**baseline_methods_rating, **ours_methods_rating}

    # 이번 실행에서 유효한 method 이름 집합 (DEBUG_SKIP_OTHER_METHODS 반영)
    expected_rating_methods = set()
    if not DEBUG_SKIP_OTHER_METHODS:
        expected_rating_methods |= {"UserKNN", "MF", "Baseline", "Baseline+History", "ChatGPT-Direct"}
    for method in FACTOR_METHODS:
        tag = method.upper()
        expected_rating_methods.add(f"Ours ({tag})")
        expected_rating_methods.add(f"Ours+History ({tag})")

    # ── 재개(resume): 체크포인트에서 이미 성공한 결과를 불러와 집계에 반영 ──
    existing_rating_records = _load_checkpoint(CHECKPOINT_RATING_PATH)
    done_by_row: dict[tuple, set] = defaultdict(set)
    for rec in existing_rating_records:
        key = (rec["user_id"], rec["movie_id"])
        done_by_row[key].add(rec["method"])
        if rec["method"] in methods_rating:
            methods_rating[rec["method"]]["preds"].append(rec["pred"])
            methods_rating[rec["method"]]["actuals"].append(rec["actual"])
    if existing_rating_records:
        logger.info("  Resumed %d existing rating checkpoint records "
                     "(%d rows already have some results).",
                     len(existing_rating_records), len(done_by_row))

    rating_checkpoint_writer = CheckpointWriter(CHECKPOINT_RATING_PATH, CHECKPOINT_INTERVAL)
    rating_lock = threading.Lock()

    rows_to_process = []
    for _, row in rating_test_rows.iterrows():
        key = (int(row["userId"]), int(row["movieId"]))
        if done_by_row.get(key, set()) >= expected_rating_methods:
            continue  # 이 행은 이미 모든 방법론이 완료됨
        rows_to_process.append(row)

    logger.info("  %d/%d rows remaining to process (resume-aware).",
                len(rows_to_process), len(rating_test_rows))

    if rows_to_process:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _process_rating_row, row,
                    done_methods=done_by_row.get((int(row["userId"]), int(row["movieId"])), set()),
                    train_df=train_df, movies_df=movies_df,
                    user_feature_matrix=user_feature_matrix,
                    userknn_model=userknn_model, mf_model=mf_model,
                    factor_results=factor_results, agent_executor=agent_executor,
                ): row
                for row in rows_to_process
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Rating Track"):
                uid, mid, records = fut.result()
                new_checkpoint_records = []
                with rating_lock:
                    for rec in records:
                        method_name = rec["method"]
                        if rec["failed"]:
                            methods_rating[method_name]["parse_failures"] += 1
                        else:
                            methods_rating[method_name]["preds"].append(rec["pred"])
                            methods_rating[method_name]["actuals"].append(rec["actual"])
                            new_checkpoint_records.append({
                                "user_id": uid, "movie_id": mid,
                                "method": method_name,
                                "pred": rec["pred"], "actual": rec["actual"],
                            })
                rating_checkpoint_writer.add(new_checkpoint_records)

    rating_checkpoint_writer.flush()

    # 레이팅 트랙 지표 계산 — 베이스라인(스킵 시 빈 값) + Ours 변형 합산
    for method_name, data in methods_rating.items():
        if data["preds"]:
            results_records.append({
                "method": method_name,
                "track": "rating",
                "n_users": rating_test_rows["userId"].nunique(),
                "n_predictions": len(data["preds"]),
                "mae": mae(data["preds"], data["actuals"]),
                "rmse": rmse(data["preds"], data["actuals"]),
                "recall_at_10": "",
                "ndcg_at_10": "",
                "parse_failures": data["parse_failures"],
            })
            # 예측 결과 저장 (내용이 있을 때만)
            pred_df = pd.DataFrame({"pred": data["preds"], "actual": data["actuals"]})
            pred_df.to_csv(os.path.join(preds_dir, f"{method_name}_rating.csv"), index=False)

    # ─────────────────────────────────────────────────
    # 8b. 랭킹 트랙 (DEBUG_SKIP_OTHER_METHODS=True면 전체 건너뜀)
    # ─────────────────────────────────────────────────
    logger.info("── Ranking Track ──")

    if DEBUG_SKIP_OTHER_METHODS:
        logger.info("  DEBUG_SKIP_OTHER_METHODS=True — Ranking Track 전체를 건너뜁니다.")
    else:
        # 베이스라인 5개 (1번만 실행)
        baseline_methods_ranking: dict[str, dict] = {
            "UserKNN":          {"recall": [], "ndcg": [], "parse_failures": 0},
            "MF":               {"recall": [], "ndcg": [], "parse_failures": 0},
            "Baseline":         {"recall": [], "ndcg": [], "parse_failures": 0},
            "Baseline+History": {"recall": [], "ndcg": [], "parse_failures": 0},
            "ChatGPT-Direct":   {"recall": [], "ndcg": [], "parse_failures": 0},
        }

        # Ours 변형 (method별)
        ours_methods_ranking: dict[str, dict] = {}
        for method in FACTOR_METHODS:
            tag = method.upper()
            ours_methods_ranking[f"Ours ({tag})"] = {"recall": [], "ndcg": [], "parse_failures": 0}
            ours_methods_ranking[f"Ours+History ({tag})"] = {"recall": [], "ndcg": [], "parse_failures": 0}

        methods_ranking = {**baseline_methods_ranking, **ours_methods_ranking}

        expected_ranking_methods = {"UserKNN", "MF", "Baseline", "Baseline+History", "ChatGPT-Direct"}
        for method in FACTOR_METHODS:
            tag = method.upper()
            expected_ranking_methods.add(f"Ours ({tag})")
            expected_ranking_methods.add(f"Ours+History ({tag})")

        # ── 재개(resume) ──
        existing_ranking_records = _load_checkpoint(CHECKPOINT_RANKING_PATH)
        done_by_user: dict[int, set] = defaultdict(set)
        for rec in existing_ranking_records:
            done_by_user[rec["user_id"]].add(rec["method"])
            if rec["method"] in methods_ranking:
                methods_ranking[rec["method"]]["recall"].append(rec["recall"])
                methods_ranking[rec["method"]]["ndcg"].append(rec["ndcg"])
        if existing_ranking_records:
            logger.info("  Resumed %d existing ranking checkpoint records "
                         "(%d users already have some results).",
                         len(existing_ranking_records), len(done_by_user))

        ranking_checkpoint_writer = CheckpointWriter(CHECKPOINT_RANKING_PATH, CHECKPOINT_INTERVAL)
        ranking_lock = threading.Lock()

        users_to_process = [
            uid for uid in valid_users
            if done_by_user.get(uid, set()) < expected_ranking_methods
        ]
        logger.info("  %d/%d users remaining to process (resume-aware).",
                    len(users_to_process), len(valid_users))

        if users_to_process:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _process_ranking_user, uid,
                        done_methods=done_by_user.get(uid, set()),
                        candidate_sets=candidate_sets, train_df=train_df, movies_df=movies_df,
                        user_feature_matrix=user_feature_matrix,
                        userknn_model=userknn_model, mf_model=mf_model,
                        factor_results=factor_results, agent_executor=agent_executor,
                    ): uid
                    for uid in users_to_process
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Ranking Track"):
                    uid, records = fut.result()
                    new_checkpoint_records = []
                    with ranking_lock:
                        for rec in records:
                            method_name = rec["method"]
                            if rec["failed"]:
                                methods_ranking[method_name]["parse_failures"] += 1
                            else:
                                methods_ranking[method_name]["recall"].append(rec["recall"])
                                methods_ranking[method_name]["ndcg"].append(rec["ndcg"])
                                new_checkpoint_records.append({
                                    "user_id": uid, "method": method_name,
                                    "recall": rec["recall"], "ndcg": rec["ndcg"],
                                })
                    ranking_checkpoint_writer.add(new_checkpoint_records)

        ranking_checkpoint_writer.flush()

        # 랭킹 트랙 지표 계산 — 베이스라인 + Ours 변형 합산
        for method_name, data in methods_ranking.items():
            if data["recall"]:
                results_records.append({
                    "method": method_name,
                    "track": "ranking",
                    "n_users": len(valid_users),
                    "n_predictions": len(data["recall"]),
                    "mae": "",
                    "rmse": "",
                    "recall_at_10": np.mean(data["recall"]),
                    "ndcg_at_10": np.mean(data["ndcg"]),
                    "parse_failures": data["parse_failures"],
                })
                # 랭킹 결과 저장
                ranking_df = pd.DataFrame({"recall@10": data["recall"], "ndcg@10": data["ndcg"]})
                ranking_df.to_csv(os.path.join(preds_dir, f"{method_name}_ranking.csv"), index=False)

    agent_executor.shutdown(wait=True)

    # ── 9. 최종 결과 저장 ────────────────────────────
    logger.info("[Step 9] Saving final results…")
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(results_records)
    results_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    # 마크다운 표 생성
    md_lines = ["# ADAP Pilot Experiment — Results Summary\n"]
    if DEBUG_SKIP_OTHER_METHODS:
        md_lines.append("**(DEBUG_SKIP_OTHER_METHODS=True — Ours/Ours+History 레이팅 트랙만 실행됨)**\n")
    md_lines.append("| Method | Track | N Users | N Preds | MAE | RMSE | Recall@10 | NDCG@10 | Parse Failures |")
    md_lines.append("|--------|-------|---------|---------|-----|------|-----------|---------|----------------|")
    for _, r in results_df.iterrows():
        md_lines.append(
            f"| {r['method']} | {r['track']} | {r['n_users']} | {r['n_predictions']} | "
            f"{r['mae'] if r['mae'] != '' else '-'} | {r['rmse'] if r['rmse'] != '' else '-'} | "
            f"{r['recall_at_10'] if r['recall_at_10'] != '' else '-'} | "
            f"{r['ndcg_at_10'] if r['ndcg_at_10'] != '' else '-'} | {r['parse_failures']} |"
        )
    md_text = "\n".join(md_lines)
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_text)

    # 파이프라인이 끝까지 정상 완료되었으므로 체크포인트 파일을 _done으로 보관
    _mark_checkpoint_done(CHECKPOINT_RATING_PATH)
    _mark_checkpoint_done(CHECKPOINT_RANKING_PATH)

    n_429 = get_rate_limit_error_count()
    if n_429 > 0:
        logger.info("총 429 에러 %d회 발생 (MAX_WORKERS를 낮추는 것을 고려하세요)", n_429)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", results_dir)
    logger.info("=" * 60)

    print("\n" + md_text)


if __name__ == "__main__":
    main()
