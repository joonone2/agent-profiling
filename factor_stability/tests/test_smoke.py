# =============================================================================
# tests/test_smoke.py
# 작은 가짜 데이터로 파이프라인 배선이 에러 없이 도는지 확인.
# MovieLens 없이도 즉시 실행 가능.
# 실행: pytest tests/test_smoke.py -v
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# 가짜 데이터 생성
# -----------------------------------------------------------------------------

def make_fake_data(n_users=80, n_items=50, n_ratings=800, seed=0):
    """MovieLens 구조를 흉내낸 가짜 데이터."""
    rng = np.random.RandomState(seed)
    user_ids = np.arange(1, n_users + 1)
    item_ids = np.arange(1, n_items + 1)

    ratings = pd.DataFrame({
        "user_id":   rng.choice(user_ids, n_ratings),
        "item_id":   rng.choice(item_ids, n_ratings),
        "rating":    rng.choice([1,2,3,4,5], n_ratings).astype(float),
        "timestamp": rng.randint(1e8, 1e9, n_ratings),
    }).drop_duplicates(["user_id","item_id"])

    genres_pool = ["Action", "Drama", "Sci-Fi", "Comedy", "Thriller"]
    movies = pd.DataFrame({
        "item_id": item_ids,
        "title":   [f"Movie_{i}" for i in item_ids],
        "genres":  ["|".join(rng.choice(genres_pool, 2, replace=False))
                    for _ in item_ids],
    })

    users = pd.DataFrame({
        "user_id":    user_ids,
        "gender":     rng.choice(["M","F"], n_users),
        "age":        rng.choice([18,25,35,45,56], n_users),
        "occupation": rng.randint(0, 21, n_users),
        "zip_code":   ["00000"] * n_users,
    })

    return ratings, movies, users


# -----------------------------------------------------------------------------
# 테스트
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_data():
    return make_fake_data()


@pytest.fixture(scope="module")
def feature_table(fake_data):
    from data_loader import apply_kcore, split_feature_validation
    from features import build_feature_table

    ratings, movies, users = fake_data
    # k-core 낮춰서 가짜 데이터도 통과
    ratings = apply_kcore(ratings, min_ratings=3)
    data = split_feature_validation(ratings, movies, users)
    return build_feature_table(data["feature_log"]), data


def test_feature_table_shape(feature_table):
    """피처 테이블이 만들어지고 NaN이 없어야 함."""
    ft, _ = feature_table
    assert ft.shape[0] > 0, "유저 없음"
    assert ft.shape[1] > 0, "피처 없음"
    assert ft.isnull().sum().sum() == 0, "NaN 존재"


def test_preprocessing(feature_table):
    """세 기법 전처리 출력이 올바른 범위인지."""
    from features import get_preprocessed
    ft, _ = feature_table
    prep = get_preprocessed(ft)

    # NMF: 0~1
    assert prep["nmf"].min() >= -1e-6, "NMF 전처리에 음수"
    assert prep["nmf"].max() <= 1 + 1e-6, "NMF 전처리 최댓값 초과"

    # FA: 표준화 (평균≈0)
    assert abs(prep["fa"].mean()) < 0.5, "FA 표준화 이상"


def test_nmf_extractor(feature_table):
    """NMF가 loading(K×피처)과 user_scores(n_users×K)를 반환하는지."""
    from features import get_preprocessed
    from extractors import extract_nmf
    ft, _ = feature_table
    prep = get_preprocessed(ft)
    res = extract_nmf(prep["nmf"], prep["feature_names"], k=3)

    assert res.type == "matrix"
    assert res.loadings.shape == (3, len(prep["feature_names"]))
    assert res.user_scores.shape[1] == 3
    assert res.explained_variance is not None


def test_fa_extractor(feature_table):
    """FA가 loading과 user_scores를 반환하는지."""
    from features import get_preprocessed
    from extractors import extract_fa
    ft, _ = feature_table
    prep = get_preprocessed(ft)
    res = extract_fa(prep["fa"], prep["feature_names"], k=3)

    assert res.type == "matrix"
    assert res.loadings.shape == (3, len(prep["feature_names"]))
    assert res.user_scores.shape[1] == 3


def test_stability_nmf(feature_table):
    """NMF split-half 안정성이 0~1 범위인지."""
    from features import get_preprocessed
    from extractors import extract_nmf
    from stability import measure_stability_matrix
    ft, _ = feature_table
    prep = get_preprocessed(ft)
    fn = prep["feature_names"]

    def nmf_fn(X):
        return extract_nmf(X, fn, k=3)

    res = measure_stability_matrix(prep["nmf"], nmf_fn, "NMF", n_seeds=3)
    assert 0.0 <= res.mean <= 1.0, f"안정성 범위 이상: {res.mean}"


def test_interpret(feature_table):
    """해석가능성 표가 기법당 K행을 포함하는지."""
    from features import get_preprocessed
    from extractors import extract_nmf, extract_fa
    from interpret import interpret_all

    ft, _ = feature_table
    prep = get_preprocessed(ft)
    fn = prep["feature_names"]

    results = {
        "NMF": extract_nmf(prep["nmf"], fn, k=3),
        "FA":  extract_fa(prep["fa"],  fn, k=3),
    }
    df = interpret_all(results)
    # NMF 3축 + FA 3축 = 6행
    assert len(df[df["method"] == "NMF"]) == 3
    assert len(df[df["method"] == "FA"])  == 3


def test_validity_runs(feature_table):
    """외부 타당성 함수가 에러 없이 실행되고 DataFrame을 반환하는지."""
    from data_loader import apply_kcore, split_feature_validation
    from features import build_feature_table, get_preprocessed
    from extractors import extract_nmf, extract_fa
    from validity import validate_all

    ratings, movies, users = make_fake_data()
    ratings = apply_kcore(ratings, min_ratings=3)
    data    = split_feature_validation(ratings, movies, users)
    ft      = build_feature_table(data["feature_log"])
    prep    = get_preprocessed(ft)
    fn      = prep["feature_names"]

    results = {
        "NMF": extract_nmf(prep["nmf"], fn, k=3),
        "FA":  extract_fa(prep["fa"],  fn, k=3),
    }
    df = validate_all(
        results,
        prep["user_ids"],
        data["validation_log"],
        data["users"],
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0