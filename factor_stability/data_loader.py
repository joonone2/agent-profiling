# =============================================================================
# data_loader.py
# MovieLens 데이터 로드 → k-core 필터 → feature_log / validation_log 분리
#
# 설계 원칙:
#   - feature_log  : 피처 테이블 생성에만 사용
#   - validation_log : 외부 타당성 검증에만 사용 (축 추출에 절대 사용 안 함)
#     → 순환논리 방지의 출발점
# =============================================================================

import pandas as pd
from pathlib import Path
from config import DATA_DIR, MIN_RATINGS_PER_USER


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    ratings.dat 로드.
    컬럼: user_id, item_id, rating, timestamp
    """
    path = data_dir / "ratings.dat"
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    movies.dat 로드.
    컬럼: item_id, title, genres (파이프 구분 다중 장르)
    """
    path = data_dir / "movies.dat"
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    users.dat 로드.
    컬럼: user_id, gender, age, occupation, zip_code
    외부 타당성 검증용 메타데이터.
    """
    path = data_dir / "users.dat"
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    return df


def apply_kcore(ratings: pd.DataFrame, min_ratings: int = MIN_RATINGS_PER_USER) -> pd.DataFrame:
    """
    k-core 필터: 평점 수가 min_ratings 미만인 유저 제거.
    기법 간 공정한 출발선을 위해 모든 기법이 동일한 유저 집합을 씀.
    """
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered = ratings[ratings["user_id"].isin(valid_users)].copy()
    print(f"[data_loader] k-core({min_ratings}) 적용: "
          f"{ratings['user_id'].nunique()} → {filtered['user_id'].nunique()} 유저, "
          f"{len(ratings)} → {len(filtered)} 평점")
    return filtered


def compute_item_popularity(ratings: pd.DataFrame) -> pd.Series:
    """
    아이템별 인기도 = 해당 아이템을 평가한 유저 수.
    피처 테이블(popularity_bias)과 검증용 독립 신호 모두에 사용.
    반환: Series(index=item_id, values=평가 유저 수)
    """
    return ratings.groupby("item_id")["user_id"].count().rename("item_pop")


def split_feature_validation(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
) -> dict:
    """
    데이터를 두 용도로 분리.

    반환 dict:
      feature_log  : ratings + movies join (피처 엔지니어링용)
      validation_log: ratings + movies + users join (외부 타당성용)
                      ★ 축 추출에 사용 금지
      item_pop     : 아이템 인기도 Series (양쪽에서 공통 참조)
      users        : 유저 메타데이터 DataFrame
    """
    item_pop = compute_item_popularity(ratings)

    # feature_log: 평점 + 영화 장르
    feature_log = ratings.merge(movies[["item_id", "genres"]], on="item_id", how="left")
    feature_log["item_pop"] = feature_log["item_id"].map(item_pop)

    # validation_log: feature_log + 유저 메타데이터
    validation_log = feature_log.merge(users, on="user_id", how="left")

    print(f"[data_loader] feature_log: {len(feature_log)} rows, "
          f"{feature_log['user_id'].nunique()} users")
    print(f"[data_loader] validation_log: {len(validation_log)} rows "
          f"(users 메타데이터 포함)")

    return {
        "feature_log":   feature_log,
        "validation_log": validation_log,
        "item_pop":      item_pop,
        "users":         users,
    }


def load_all(data_dir: Path = DATA_DIR, subsample: float = 1.0, seed: int = 42) -> dict:
    """
    전체 로딩 파이프라인 진입점.

    Args:
        data_dir   : MovieLens 데이터 폴더 (config.DATA_DIR)
        subsample  : 유저 서브샘플 비율 (볼륨 곡선 실험용; 1.0이면 전체)
        seed       : 서브샘플 랜덤 시드

    Returns:
        split_feature_validation()의 반환 dict
    """
    ratings = load_ratings(data_dir)
    movies  = load_movies(data_dir)
    users   = load_users(data_dir)

    ratings = apply_kcore(ratings)

    # 볼륨 곡선 실험: 유저 단위로 서브샘플
    if subsample < 1.0:
        all_users = ratings["user_id"].unique()
        n = max(1, int(len(all_users) * subsample))
        rng = __import__("numpy").random.RandomState(seed)
        sampled_users = rng.choice(all_users, size=n, replace=False)
        ratings = ratings[ratings["user_id"].isin(sampled_users)].copy()
        users   = users[users["user_id"].isin(sampled_users)].copy()
        print(f"[data_loader] 서브샘플({subsample:.0%}): {n} 유저")

    return split_feature_validation(ratings, movies, users)