"""
ADAP Pilot Experiment — 전체 파이프라인 실행 + 결과 집계

순서:
1. 데이터 로딩 + 시간순 분할
2. Feature Engineering
3. N_PILOT_USERS명 무작위 샘플링
4. 후보셋 생성
5. 통계 베이스라인 모델 학습 (DEBUG_SKIP_OTHER_METHODS=True면 건너뜀)
6. FACTOR_METHODS 루프: Factor Discovery → Agent Synthesis → Weight Estimation
7. 레이팅 예측 트랙 (베이스라인 1회 + Ours method별)
8. 랭킹 트랙 (DEBUG_SKIP_OTHER_METHODS=True면 전체 건너뜀)
9. 지표 계산 + 결과 저장
"""
import os
import sys
import json
import logging
import time

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
    chatgpt_direct_predict_rating,
    chatgpt_direct_rank,
)
from src.evaluation.metrics import mae, rmse, recall_at_k, ndcg_at_k

# ── 로깅 설정 ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
# 메인 파이프라인
# ══════════════════════════════════════════════════════

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ADAP Pilot Experiment — Pipeline Start")
    if DEBUG_SKIP_OTHER_METHODS:
        logger.info("★ DEBUG_SKIP_OTHER_METHODS=True — "
                     "UserKNN/MF/Baseline/ChatGPT-Direct 및 랭킹 트랙 전체를 건너뜁니다.")
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

    # ─────────────────────────────────────────────────
    # 8a. 레이팅 예측 트랙
    # ─────────────────────────────────────────────────
    logger.info("── Rating Prediction Track ──")

    # 대상: valid_users의 test_df 내 모든 (userId, movieId, rating) 행
    rating_test_rows = pilot_test[pilot_test["userId"].isin(valid_users)].copy()
    logger.info("  Rating track: %d test rows across %d users",
                len(rating_test_rows), rating_test_rows["userId"].nunique())

    # 베이스라인 4개 (1번만 실행, DEBUG_SKIP_OTHER_METHODS=True면 항상 빈 상태로 유지)
    baseline_methods_rating: dict[str, dict] = {
        "UserKNN":          {"preds": [], "actuals": [], "parse_failures": 0},
        "MF":               {"preds": [], "actuals": [], "parse_failures": 0},
        "Baseline":         {"preds": [], "actuals": [], "parse_failures": 0},
        "ChatGPT-Direct":   {"preds": [], "actuals": [], "parse_failures": 0},
    }

    # Ours 변형 (method별로 생성)
    ours_methods_rating: dict[str, dict] = {}
    for method in FACTOR_METHODS:
        tag = method.upper()
        ours_methods_rating[f"Ours ({tag})"] = {"preds": [], "actuals": [], "parse_failures": 0}
        ours_methods_rating[f"Ours+History ({tag})"] = {"preds": [], "actuals": [], "parse_failures": 0}

    for _, row in tqdm(rating_test_rows.iterrows(), total=len(rating_test_rows), desc="Rating Track"):
        uid = int(row["userId"])
        mid = int(row["movieId"])
        actual = float(row["rating"])
        item_info = _get_item_info(mid, movies_df)

        # ── UserKNN / MF / Baseline / ChatGPT-Direct: 디버깅 중에는 건너뜀 ──
        if not DEBUG_SKIP_OTHER_METHODS:
            # --- UserKNN (1회) ---
            pred = userknn_model.predict(uid, mid)
            baseline_methods_rating["UserKNN"]["preds"].append(pred)
            baseline_methods_rating["UserKNN"]["actuals"].append(actual)

            # --- MF (1회) ---
            pred = mf_predict(mf_model, uid, mid)
            baseline_methods_rating["MF"]["preds"].append(pred)
            baseline_methods_rating["MF"]["actuals"].append(actual)

            # --- Baseline 19dim→LLM (1회) ---
            try:
                uf = user_feature_matrix.loc[uid] if uid in user_feature_matrix.index else pd.Series(dtype=float)
                pred = baseline_predict_rating(uf, item_info)
                baseline_methods_rating["Baseline"]["preds"].append(pred)
                baseline_methods_rating["Baseline"]["actuals"].append(actual)
            except (ValueError, Exception) as e:
                baseline_methods_rating["Baseline"]["parse_failures"] += 1
                logger.warning("Baseline rating parse failure for user=%d item=%d: %s", uid, mid, e)

            # --- ChatGPT-Direct (1회) ---
            try:
                history = _get_user_history(uid, train_df, movies_df)
                pred = chatgpt_direct_predict_rating(history, item_info)
                baseline_methods_rating["ChatGPT-Direct"]["preds"].append(pred)
                baseline_methods_rating["ChatGPT-Direct"]["actuals"].append(actual)
            except (ValueError, Exception) as e:
                baseline_methods_rating["ChatGPT-Direct"]["parse_failures"] += 1
                logger.warning("ChatGPT-Direct rating parse failure for user=%d item=%d: %s", uid, mid, e)

        # --- Ours / Ours+History: method별로 각각 실행 (항상 실행됨) ---
        for method in FACTOR_METHODS:
            tag = method.upper()
            fr = factor_results[method]
            factor_ids = fr["factor_ids"]
            agent_prompts = fr["agent_prompts"]
            user_weights_df = fr["user_weights_df"]

            # Ours (include_history=False)
            try:
                agent_outputs = {}
                for fid in factor_ids:
                    agent_outputs[fid] = agent_judge_single(agent_prompts[fid], item_info)
                uw = {fid: float(user_weights_df.loc[uid, fid]) for fid in factor_ids} \
                    if uid in user_weights_df.index else {fid: 1.0 / len(factor_ids) for fid in factor_ids}
                pred = orchestrate_rating(agent_outputs, uw, history=None)
                ours_methods_rating[f"Ours ({tag})"]["preds"].append(pred)
                ours_methods_rating[f"Ours ({tag})"]["actuals"].append(actual)
            except Exception as e:
                ours_methods_rating[f"Ours ({tag})"]["parse_failures"] += 1
                logger.warning("Ours (%s) rating failure for user=%d item=%d: %s", tag, uid, mid, e)

            # Ours+History
            try:
                hist_titles = _get_user_history_titles(uid, train_df, movies_df)
                pred = orchestrate_rating(agent_outputs, uw, history=hist_titles)
                ours_methods_rating[f"Ours+History ({tag})"]["preds"].append(pred)
                ours_methods_rating[f"Ours+History ({tag})"]["actuals"].append(actual)
            except Exception as e:
                ours_methods_rating[f"Ours+History ({tag})"]["parse_failures"] += 1
                logger.warning("Ours+History (%s) rating failure for user=%d item=%d: %s", tag, uid, mid, e)

    # 레이팅 트랙 지표 계산 — 베이스라인(스킵 시 빈 값) + Ours 변형 합산
    all_rating_methods = {**baseline_methods_rating, **ours_methods_rating}
    for method_name, data in all_rating_methods.items():
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
        # 베이스라인 4개 (1번만 실행)
        baseline_methods_ranking: dict[str, dict] = {
            "UserKNN":          {"recall": [], "ndcg": [], "parse_failures": 0},
            "MF":               {"recall": [], "ndcg": [], "parse_failures": 0},
            "Baseline":         {"recall": [], "ndcg": [], "parse_failures": 0},
            "ChatGPT-Direct":   {"recall": [], "ndcg": [], "parse_failures": 0},
        }

        # Ours 변형 (method별)
        ours_methods_ranking: dict[str, dict] = {}
        for method in FACTOR_METHODS:
            tag = method.upper()
            ours_methods_ranking[f"Ours ({tag})"] = {"recall": [], "ndcg": [], "parse_failures": 0}
            ours_methods_ranking[f"Ours+History ({tag})"] = {"recall": [], "ndcg": [], "parse_failures": 0}

        for uid in tqdm(valid_users, desc="Ranking Track"):
            cs = candidate_sets[uid]
            positive_id = cs["positive"]
            candidate_ids = cs["candidates"]

            # 후보 아이템 정보
            candidate_items = []
            candidate_titles_map = {}
            for cid in candidate_ids:
                info = _get_item_info(cid, movies_df)
                candidate_items.append({"item_id": cid, "title": info["title"], "genres": info["genres"]})
                candidate_titles_map[cid] = info["title"]

            # --- UserKNN (1회) ---
            ranked = rank_by_stat_method(userknn_model.predict, uid, candidate_ids)
            baseline_methods_ranking["UserKNN"]["recall"].append(recall_at_k(ranked, positive_id))
            baseline_methods_ranking["UserKNN"]["ndcg"].append(ndcg_at_k(ranked, positive_id))

            # --- MF (1회) ---
            ranked = rank_by_stat_method(lambda u, i: mf_predict(mf_model, u, i), uid, candidate_ids)
            baseline_methods_ranking["MF"]["recall"].append(recall_at_k(ranked, positive_id))
            baseline_methods_ranking["MF"]["ndcg"].append(ndcg_at_k(ranked, positive_id))

            # --- Baseline 19dim→LLM (1회, 배치) ---
            try:
                uf = user_feature_matrix.loc[uid] if uid in user_feature_matrix.index else pd.Series(dtype=float)
                ranked = baseline_rank(uf, candidate_items)
                baseline_methods_ranking["Baseline"]["recall"].append(recall_at_k(ranked, positive_id))
                baseline_methods_ranking["Baseline"]["ndcg"].append(ndcg_at_k(ranked, positive_id))
            except Exception as e:
                baseline_methods_ranking["Baseline"]["parse_failures"] += 1
                logger.warning("Baseline ranking failure for user=%d: %s", uid, e)

            # --- ChatGPT-Direct (1회, 배치) ---
            try:
                history = _get_user_history(uid, train_df, movies_df)
                ranked = chatgpt_direct_rank(history, candidate_items)
                baseline_methods_ranking["ChatGPT-Direct"]["recall"].append(recall_at_k(ranked, positive_id))
                baseline_methods_ranking["ChatGPT-Direct"]["ndcg"].append(ndcg_at_k(ranked, positive_id))
            except Exception as e:
                baseline_methods_ranking["ChatGPT-Direct"]["parse_failures"] += 1
                logger.warning("ChatGPT-Direct ranking failure for user=%d: %s", uid, e)

            # --- Ours & Ours+History: method별로 각각 실행 ---
            for method in FACTOR_METHODS:
                tag = method.upper()
                fr = factor_results[method]
                factor_ids = fr["factor_ids"]
                agent_prompts = fr["agent_prompts"]
                user_weights_df = fr["user_weights_df"]

                try:
                    # 각 에이전트가 후보 20개를 배치 평가
                    agent_batch_results: dict[str, list[dict]] = {}
                    for fid in factor_ids:
                        agent_batch_results[fid] = agent_judge_batch(agent_prompts[fid], candidate_items)

                    # item_id 기준으로 재구성: {item_id: {factor_id: score}}
                    agent_outputs_per_item: dict[int, dict[str, float]] = {}
                    for cid in candidate_ids:
                        agent_outputs_per_item[cid] = {}
                    for fid, batch in agent_batch_results.items():
                        for item in batch:
                            agent_outputs_per_item[item["item_id"]][fid] = item["score"]

                    uw = {fid: float(user_weights_df.loc[uid, fid]) for fid in factor_ids} \
                        if uid in user_weights_df.index else {fid: 1.0 / len(factor_ids) for fid in factor_ids}

                    # Ours (no history)
                    ranked = orchestrate_ranking(agent_outputs_per_item, uw, history=None)
                    ours_methods_ranking[f"Ours ({tag})"]["recall"].append(recall_at_k(ranked, positive_id))
                    ours_methods_ranking[f"Ours ({tag})"]["ndcg"].append(ndcg_at_k(ranked, positive_id))

                    # Ours+History
                    hist_titles = _get_user_history_titles(uid, train_df, movies_df)
                    ranked = orchestrate_ranking(agent_outputs_per_item, uw, history=hist_titles)
                    ours_methods_ranking[f"Ours+History ({tag})"]["recall"].append(recall_at_k(ranked, positive_id))
                    ours_methods_ranking[f"Ours+History ({tag})"]["ndcg"].append(ndcg_at_k(ranked, positive_id))

                except Exception as e:
                    ours_methods_ranking[f"Ours ({tag})"]["parse_failures"] += 1
                    ours_methods_ranking[f"Ours+History ({tag})"]["parse_failures"] += 1
                    logger.warning("Ours/Ours+History (%s) ranking failure for user=%d: %s", tag, uid, e)

        # 랭킹 트랙 지표 계산 — 베이스라인 + Ours 변형 합산
        all_ranking_methods = {**baseline_methods_ranking, **ours_methods_ranking}
        for method_name, data in all_ranking_methods.items():
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

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", results_dir)
    logger.info("=" * 60)

    print("\n" + md_text)


if __name__ == "__main__":
    main()