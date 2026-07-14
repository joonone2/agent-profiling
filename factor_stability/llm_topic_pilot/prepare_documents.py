# =============================================================================
# prepare_documents.py
# TopicGPT 파일럿 실험용 입력 데이터 준비.
#
# ★ "user_id 한 명 = 문서 한 개"로 취급. 문서 내용은 그 유저가 실제로 평점을
#   매긴 영화 전체(제목 + 장르 + 평점), 평점 내림차순 정렬.
#
# ★ 기존 factor_stability 프로젝트의 data_loader.py를 그대로 재사용
#   (k-core 필터 등 동일한 전처리 기준 유지).
#
# ★ API 호출 전에, 문서별 길이(토큰 추정치)를 먼저 진단해서 콘솔에 보여줌.
#   이 스크립트는 "잘라내기(truncation)"를 하지 않음 — 실험 취지가
#   "절충안 없이 전체 시청 목록으로 제대로 해보자"는 것이었기 때문.
#   다만 헤비 유저의 문서가 지나치게 길면 비용/시간에 영향이 크므로,
#   실행 전 미리 확인할 수 있게 경고만 표시함.
#
# 출력: data/input/movies_100.jsonl  ({"id": "...", "text": "..."} 형식)
# =============================================================================

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 상위 factor_stability 폴더의 모듈을 재사용
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from config import DATA_DIR as _DATA_DIR_RAW  # noqa: E402
from data_loader import load_ratings, load_movies, apply_kcore  # noqa: E402

# config.py의 DATA_DIR("data/ml-1m")는 상대경로라서, 실행 위치(현재 작업
# 디렉토리)에 따라 다르게 해석됨. llm_topic_pilot/ 폴더에서 실행해도
# 항상 factor_stability/data/ml-1m을 가리키도록, PARENT_DIR 기준으로
# 절대경로를 다시 계산함. (PARENT_DIR / 절대경로)는 pathlib 특성상
# 절대경로가 이미 있으면 그걸 그대로 쓰므로, config.py를 나중에
# 절대경로로 바꿔도 안전하게 동작함.
DATA_DIR = PARENT_DIR / _DATA_DIR_RAW

N_USERS = 100
SAMPLE_SEED = 42  # 재현성을 위한 고정 시드 (프로젝트 전반의 BASE_SEED 관례와 일치)
OUT_PATH = Path(__file__).resolve().parent / "data" / "input" / f"movies_{N_USERS}.jsonl"

# 토큰 수 추정 대략치 (영어 기준 word count * 1.3). 정확한 tiktoken 계산이 아니라
# "실행 전 감을 잡기 위한" 대략치임.
TOKENS_PER_WORD = 1.3
WARN_TOKEN_THRESHOLD = 3000  # 이 이상이면 콘솔에 경고 표시 (자르지는 않음)


def build_user_document(user_ratings: pd.DataFrame) -> str:
    """
    한 유저의 평점 기록 전체를, 평점 내림차순으로 정렬한 텍스트로 변환.
    예: "Toy Story (Animation, Children's, Comedy): 5/5\n..."
    """
    sorted_ratings = user_ratings.sort_values("rating", ascending=False)
    lines = []
    for _, row in sorted_ratings.iterrows():
        genres = row["genres"].replace("|", ", ")
        lines.append(f"{row['title']} ({genres}): {row['rating']}/5")
    return "\n".join(lines)


def main():
    print("[prepare_documents] 원본 로그 로딩...")
    ratings = apply_kcore(load_ratings(DATA_DIR))
    movies = load_movies(DATA_DIR)

    merged = ratings.merge(movies[["item_id", "title", "genres"]], on="item_id", how="left")

    unique_users = merged["user_id"].unique()
    print(f"[prepare_documents] k-core 필터 후 전체 유저 수: {len(unique_users)}")

    rng = np.random.RandomState(SAMPLE_SEED)
    sampled_users = rng.choice(unique_users, size=min(N_USERS, len(unique_users)), replace=False)
    print(f"[prepare_documents] 무작위 샘플링: {len(sampled_users)}명 (seed={SAMPLE_SEED})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc_lengths = []
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for uid in sampled_users:
            user_ratings = merged[merged["user_id"] == uid]
            text = build_user_document(user_ratings)
            n_movies = len(user_ratings)
            n_words = len(text.split())
            est_tokens = int(n_words * TOKENS_PER_WORD)
            doc_lengths.append((uid, n_movies, est_tokens))

            record = {"id": f"user_{uid}", "text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[prepare_documents] 저장 완료: {OUT_PATH}")

    # ---- 실행 전 진단 (API 호출 전에 반드시 확인) ----
    lengths_df = pd.DataFrame(doc_lengths, columns=["user_id", "n_movies", "est_tokens"])
    print("\n[prepare_documents] 문서 길이 진단 (API 호출 전 확인용, 대략치):")
    print(f"  평점 개수: 최소 {lengths_df['n_movies'].min()}, "
          f"평균 {lengths_df['n_movies'].mean():.0f}, "
          f"최대 {lengths_df['n_movies'].max()}")
    print(f"  추정 토큰: 최소 {lengths_df['est_tokens'].min()}, "
          f"평균 {lengths_df['est_tokens'].mean():.0f}, "
          f"최대 {lengths_df['est_tokens'].max()}")

    long_docs = lengths_df[lengths_df["est_tokens"] > WARN_TOKEN_THRESHOLD]
    if len(long_docs) > 0:
        print(f"\n  경고: {len(long_docs)}명의 문서가 추정 토큰 {WARN_TOKEN_THRESHOLD}을 초과함.")
        print(f"  (자르지 않고 그대로 둠 — 비용/시간에 영향을 줄 수 있으니 실행 전 확인 필요)")
        print(long_docs.to_string(index=False))
    else:
        print(f"\n  모든 문서가 추정 토큰 {WARN_TOKEN_THRESHOLD} 이하.")


if __name__ == "__main__":
    main()