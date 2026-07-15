"""
User×Feature(19-dim), Item×Feature(19-dim) 행렬 생성
"""
import os
import logging

import numpy as np
import pandas as pd

from config import GENRES, FEATURE_NAMES, OUTPUT_DIR

logger = logging.getLogger(__name__)


def compute_item_popularity(ratings_df: pd.DataFrame) -> pd.Series:
    """
    movieId -> popularity percentile (0~1), rating count 기준.
    """
    counts = ratings_df.groupby("movieId").size()
    # percentile rank: 0~1 범위
    popularity = counts.rank(pct=True)
    popularity.name = "popularity_bias"
    return popularity


def build_user_feature_matrix(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    item_popularity: pd.Series,
) -> pd.DataFrame:
    """
    index: userId, columns: FEATURE_NAMES (19개)

    장르별 피처 (18개): 해당 유저가 train 데이터에서 그 장르의 영화들에 준 평점의 평균.
        한 번도 평가 안 한 장르는 전체 유저 평균(global mean rating)으로 대체.
    popularity_bias (1개): 유저가 평가한 영화들의 인기도 백분위 평균값.
    """
    global_mean_rating = train_df["rating"].mean()

    # 영화별 장르 매핑 (movieId → set of genres)
    movie_genres: dict[int, set[str]] = {}
    for _, row in movies_df.iterrows():
        movie_genres[int(row["movieId"])] = set(row["genres"])

    user_ids = train_df["userId"].unique()
    records: list[dict] = []

    for uid in user_ids:
        user_rows = train_df[train_df["userId"] == uid]
        feature: dict[str, float] = {}

        # 장르별 평균 평점
        for genre in GENRES:
            genre_ratings = []
            for _, r in user_rows.iterrows():
                mid = int(r["movieId"])
                if mid in movie_genres and genre in movie_genres[mid]:
                    genre_ratings.append(r["rating"])
            if genre_ratings:
                feature[genre] = float(np.mean(genre_ratings))
            else:
                feature[genre] = global_mean_rating  # imputation

        # popularity_bias: 유저가 평가한 영화들의 인기도 백분위 평균
        user_movie_ids = user_rows["movieId"].values
        pop_values = item_popularity.reindex(user_movie_ids).dropna().values
        feature["popularity_bias"] = float(np.mean(pop_values)) if len(pop_values) > 0 else 0.5

        feature["userId"] = uid
        records.append(feature)

    result = pd.DataFrame(records).set_index("userId")
    # FEATURE_NAMES 순서를 강제
    result = result[FEATURE_NAMES]

    # 저장
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    result.to_csv(os.path.join(factors_dir, "user_features.csv"))
    logger.info("User feature matrix: %s", result.shape)
    return result


def build_item_feature_vector(
    movie_id: int,
    movies_df: pd.DataFrame,
    item_popularity: pd.Series,
) -> np.ndarray:
    """
    19-dim: 장르 multi-hot (해당 장르면 1, 아니면 0, 정규화 없음) + popularity_bias (해당 영화의 인기도 백분위)
    """
    row = movies_df[movies_df["movieId"] == movie_id]
    if row.empty:
        return np.zeros(len(FEATURE_NAMES))

    genres_list = row.iloc[0]["genres"]
    vec = []
    for genre in GENRES:
        vec.append(1.0 if genre in genres_list else 0.0)

    # popularity_bias
    pop = item_popularity.get(movie_id, 0.5)
    vec.append(float(pop))
    return np.array(vec)


def build_item_feature_matrix(
    movies_df: pd.DataFrame,
    item_popularity: pd.Series,
) -> pd.DataFrame:
    """
    전체 영화에 대해 build_item_feature_vector를 적용한 행렬.
    index: movieId, columns: FEATURE_NAMES
    """
    records = []
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        vec = build_item_feature_vector(mid, movies_df, item_popularity)
        records.append({"movieId": mid, **dict(zip(FEATURE_NAMES, vec))})

    result = pd.DataFrame(records).set_index("movieId")

    # 저장
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    result.to_csv(os.path.join(factors_dir, "item_features.csv"))
    logger.info("Item feature matrix: %s", result.shape)
    return result
