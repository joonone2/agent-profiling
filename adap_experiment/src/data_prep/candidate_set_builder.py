"""
유저별 20개 후보(정답 1 + 오답 19) 생성
"""
import os
import json
import logging

import numpy as np
import pandas as pd

from config import POSITIVE_RATING_THRESHOLD, N_NEGATIVES, OUTPUT_DIR

logger = logging.getLogger(__name__)


def build_candidate_sets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_negatives: int = N_NEGATIVES,
    seed: int = 42,
) -> dict:
    """
    각 유저에 대해:
      1) positive: test_df에서 해당 유저의 rating >= POSITIVE_RATING_THRESHOLD인 항목 중
         timestamp가 가장 늦은 1개의 movieId. 없으면 그 유저는 스킵.
      2) negatives: 그 유저가 train+test 어디에서도 평가한 적 없는 movieId 중 n_negatives개 무작위 샘플
      3) candidates: [positive] + negatives 를 무작위 셔플한 movieId 리스트 (길이 20)

    반환: {userId: {"positive": movieId, "negatives": [...], "candidates": [...]}}
    """
    rng = np.random.RandomState(seed)
    all_movie_ids = set(movies_df["movieId"].values)

    # 유저별 평가한 영화 집합
    rated_by_user: dict[int, set[int]] = {}
    for uid in set(train_df["userId"]).union(set(test_df["userId"])):
        rated_train = set(train_df[train_df["userId"] == uid]["movieId"].values)
        rated_test = set(test_df[test_df["userId"] == uid]["movieId"].values)
        rated_by_user[uid] = rated_train | rated_test

    result: dict = {}

    for uid in test_df["userId"].unique():
        user_test = test_df[test_df["userId"] == uid]
        # positive: rating >= threshold, timestamp가 가장 늦은 것
        positive_candidates = user_test[user_test["rating"] >= POSITIVE_RATING_THRESHOLD]
        if positive_candidates.empty:
            continue
        positive_row = positive_candidates.sort_values("timestamp").iloc[-1]
        positive_id = int(positive_row["movieId"])

        # negatives: 전혀 평가 안 한 영화 중 샘플
        unrated = list(all_movie_ids - rated_by_user.get(uid, set()))
        if len(unrated) < n_negatives:
            logger.warning("User %d: not enough unrated movies (%d < %d), skipping.",
                           uid, len(unrated), n_negatives)
            continue

        negatives = rng.choice(unrated, size=n_negatives, replace=False).tolist()
        negatives = [int(x) for x in negatives]

        # candidates: positive + negatives, 셔플
        candidates = [positive_id] + negatives
        rng.shuffle(candidates)

        result[int(uid)] = {
            "positive": positive_id,
            "negatives": negatives,
            "candidates": [int(c) for c in candidates],
        }

    # 저장
    preds_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    save_path = os.path.join(preds_dir, "candidate_sets.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Candidate sets built for %d users. Saved to %s", len(result), save_path)
    return result
