"""
User×Feature(19-dim), Item×Feature(19-dim) 행렬 생성
"""
import os
import logging

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


def _build_genre_multihot(movies_df: pd.DataFrame) -> pd.DataFrame:
    """movieId -> GENRES 순서의 0/1 multi-hot 행렬 (movies_df에 없는 장르 조합도 GENRES 컬럼으로 강제)."""
    exploded = movies_df[["movieId", "genres"]].explode("genres")
    dummies = pd.crosstab(exploded["movieId"], exploded["genres"])
    dummies = dummies.reindex(columns=GENRES, fill_value=0)
    dummies = dummies.reindex(index=movies_df["movieId"], fill_value=0)
    return dummies.astype(float)


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

    NOTE: 원래는 유저×장르×rating을 파이썬 삼중 루프(.iterrows())로 순회했으나,
    동일한 계산을 merge+groupby로 벡터화하여 대체함 (결과는 동일, 속도만 개선).
    """
    global_mean_rating = train_df["rating"].mean()
    user_ids = train_df["userId"].unique()

    genre_dummies = _build_genre_multihot(movies_df)

    # rating 행마다 해당 영화의 장르 multi-hot을 붙임 (movies_df에 없는 movieId는 전부 0으로 처리 → 원본의 "mid in movie_genres" 체크와 동일)
    ratings_with_genres = train_df[["userId", "movieId", "rating"]].merge(
        genre_dummies, left_on="movieId", right_index=True, how="left",
    )
    ratings_with_genres[GENRES] = ratings_with_genres[GENRES].fillna(0.0)

    # 유저별 장르 평균 평점 = (해당 장르 영화에 준 평점의 합) / (해당 장르 영화 평가 개수)
    genre_rating_sum = ratings_with_genres[GENRES].multiply(
        ratings_with_genres["rating"], axis=0
    ).groupby(ratings_with_genres["userId"]).sum()
    genre_count = ratings_with_genres[GENRES].groupby(ratings_with_genres["userId"]).sum()
    genre_mean = (genre_rating_sum / genre_count).reindex(user_ids)
    genre_mean = genre_mean.fillna(global_mean_rating)  # 한 번도 평가 안 한 장르는 전체 평균으로 대체(imputation)

    # popularity_bias: 유저가 평가한 영화들의 인기도 백분위 평균
    pop_per_row = train_df["movieId"].map(item_popularity)
    popularity_bias = pop_per_row.groupby(train_df["userId"]).mean().reindex(user_ids)
    popularity_bias = popularity_bias.fillna(0.5)

    result = genre_mean.copy()
    result["popularity_bias"] = popularity_bias
    result.index.name = "userId"
    # FEATURE_NAMES 순서를 강제
    result = result[FEATURE_NAMES]

    # 저장
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    result.to_csv(os.path.join(factors_dir, "user_features.csv"))
    logger.info("User feature matrix: %s", result.shape)
    return result


def build_item_feature_matrix(
    movies_df: pd.DataFrame,
    item_popularity: pd.Series,
) -> pd.DataFrame:
    """
    19-dim: 장르 multi-hot (해당 장르면 1, 아니면 0) + popularity_bias (해당 영화의 인기도 백분위).
    index: movieId, columns: FEATURE_NAMES

    NOTE: 원래는 영화마다 movies_df를 다시 필터링(movies_df[movies_df["movieId"]==mid])하는
    O(영화수^2) 루프였으나, 동일한 계산을 벡터화하여 대체함 (결과는 동일, 속도만 개선).
    """
    genre_dummies = _build_genre_multihot(movies_df)  # index=movieId, columns=GENRES (0.0/1.0)

    result = genre_dummies.copy()
    result.index.name = "movieId"
    result["popularity_bias"] = item_popularity.reindex(result.index).fillna(0.5)
    result = result[FEATURE_NAMES]

    # 저장
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    result.to_csv(os.path.join(factors_dir, "item_features.csv"))
    logger.info("Item feature matrix: %s", result.shape)
    return result
