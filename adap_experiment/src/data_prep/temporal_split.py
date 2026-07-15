"""
유저별 시간순 Train/Test 분할
"""
import os
import logging
import pandas as pd

from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def temporal_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    유저별로 timestamp 오름차순 정렬 후, 뒤에서 test_ratio 비율만큼을 test로 분리.
    각 유저마다 최소 1개는 train에 남도록 보장.
    반환: (train_df, test_df)
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, group in ratings_df.groupby("userId"):
        group_sorted = group.sort_values("timestamp")
        n = len(group_sorted)
        n_test = max(0, int(n * test_ratio))

        # 최소 1개는 train에 남기기
        if n_test >= n:
            n_test = n - 1

        if n_test <= 0:
            train_parts.append(group_sorted)
        else:
            train_parts.append(group_sorted.iloc[:-n_test])
            test_parts.append(group_sorted.iloc[-n_test:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(
        columns=ratings_df.columns
    )

    # 저장
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)

    logger.info(
        "Temporal split complete: %d train rows, %d test rows",
        len(train_df), len(test_df),
    )
    return train_df, test_df
