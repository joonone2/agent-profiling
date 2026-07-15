"""
MovieLens-1M 원본 데이터 로딩 + 파싱
"""
import pandas as pd


def load_ratings(path: str) -> pd.DataFrame:
    """
    ratings.dat 파일 로딩.
    columns: userId, movieId, rating, timestamp
    """
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    return df


def load_movies(path: str) -> pd.DataFrame:
    """
    movies.dat 파일 로딩.
    columns: movieId, title, genres (list[str] 형태로 파싱)
    """
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    # genres 컬럼을 '|' 구분 문자열에서 list[str]로 변환
    df["genres"] = df["genres"].apply(lambda x: x.split("|"))
    return df


def load_users(path: str) -> pd.DataFrame:
    """
    users.dat 파일 로딩 (이번 실험에서는 사용하지 않지만 로딩만 제공).
    columns: userId, gender, age, occupation, zipCode
    """
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["userId", "gender", "age", "occupation", "zipCode"],
        engine="python",
        encoding="latin-1",
    )
    return df
