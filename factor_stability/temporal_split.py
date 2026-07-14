# =============================================================================
# temporal_split.py
# 유저별 과거 시청 이력을 시간순으로 80% Train, 20% Test로 분할합니다.
# Train: 프로파일 생성용
# Test: 추천 평가(정답 확인)용
# =============================================================================

import pandas as pd

def temporal_split(ratings: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    유저별로 timestamp 기준 정렬 후 시간순 분할.
    
    Args:
        ratings (pd.DataFrame): 최소 user_id, timestamp 컬럼을 포함하는 데이터프레임
        train_ratio (float): Train 셋 비율 (기본 0.8)
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train_log, test_log
    """
    # 유저와 시간순으로 정렬
    df_sorted = ratings.sort_values(by=["user_id", "timestamp"]).copy()
    
    # 유저별 순위와 전체 길이 계산
    df_sorted["rank"] = df_sorted.groupby("user_id").cumcount()
    df_sorted["total"] = df_sorted.groupby("user_id")["user_id"].transform("count")
    
    # Train / Test 마스크 생성
    train_mask = df_sorted["rank"] < (df_sorted["total"] * train_ratio)
    
    train_log = df_sorted[train_mask].drop(columns=["rank", "total"]).copy()
    test_log = df_sorted[~train_mask].drop(columns=["rank", "total"]).copy()
    
    return train_log, test_log
