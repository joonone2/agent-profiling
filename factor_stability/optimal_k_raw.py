"""
Raw feature 공간(19개 feature)에서 k-means에 쓸 최적 K를
elbow(inertia)와 silhouette score로 탐색하는 스크립트.

전제:
- results/user_feature_table.csv 에 6040명 x feature 컬럼이 있다고 가정
- 식별자 컬럼(user_id 등)은 EXCLUDE_COLS에 넣어서 제외

결측 처리:
- listwise deletion 사용 (기존 visualize_phase0.py의
  cluster_raw_feature_space()와 동일한 방식). 결측이 하나라도 있는 유저는
  이 진단에서 제외됨. median/mean imputation은 쓰지 않음 — 결측 위치에
  "가짜 평균적 취향"을 주입해서 실제로는 무관한 유저들이 인위적으로
  가까워 보이는 문제가 있기 때문 (features.py의 결측 처리 철학과 동일).
- 주의: listwise deletion을 하면 남는 표본이 무작위가 아니라
  "여러 장르를 폭넓게 본" 유저 쪽으로 치우칠 수 있음. 이 스크립트의
  결과는 그 편향된 부분표본에 대한 답으로 해석해야 함(전체 6040명에
  대한 답이 아님).

전처리:
- StandardScaler를 기본으로 사용 (k-means는 스케일에 민감하므로 필수)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ── 설정 (실제 프로젝트 경로/컬럼명에 맞게 수정) ──────────────────────
# 스크립트 파일이 있는 위치를 기준으로 절대경로를 잡음 (cwd가 어디든 상관없이 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_CSV = os.path.join(RESULTS_DIR, "user_feature_table.csv")
EXCLUDE_COLS = ["user_id"]           # feature가 아닌 식별자 컬럼들
K_RANGE = range(2, 11)                # 탐색할 K 범위
RANDOM_STATE = 42
OUT_CSV = os.path.join(RESULTS_DIR, "optimal_k_raw_features.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "optimal_k_raw_features.png")

print(f"스크립트 위치: {BASE_DIR}")
print(f"입력 파일: {FEATURE_CSV}")
print(f"출력 폴더: {RESULTS_DIR}\n")

# ── 데이터 로드 ──────────────────────────────────────────────────────
df = pd.read_csv(FEATURE_CSV)
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
n_total = len(df)

# ★ listwise deletion — 기존 cluster_raw_feature_space()와 동일한 방식.
#   median/mean imputation은 사용하지 않음: 결측 위치에 "가짜 평균적 취향"을
#   주입해서 실제로는 무관한 유저들을 인위적으로 가깝게 만드는 문제가 있음
#   (features.py 설계 철학과 동일한 이유로 기각).
complete = df.dropna(subset=feature_cols)
n_kept = len(complete)
print(f"listwise deletion: {n_total}명 중 {n_kept}명 유지 "
      f"({n_kept/n_total:.1%}), 전체 {len(feature_cols)}개 feature 결측 없음 기준")
print("주의: 남는 표본은 무작위가 아니라 '여러 장르를 폭넓게 본' 유저 쪽으로 "
      "치우쳐 있을 수 있음 (기존 cluster_raw_feature_space와 동일한 한계)\n")

X = complete[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── K별 inertia / silhouette 계산 ───────────────────────────────────
records = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(
        X_scaled, labels,
        sample_size=min(5000, len(X_scaled)),
        random_state=RANDOM_STATE,
    )
    records.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    print(f"K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

result_df = pd.DataFrame(records)
result_df.to_csv(OUT_CSV, index=False)

# ── 시각화 ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(result_df["k"], result_df["inertia"], marker="o")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow method")

axes[1].plot(result_df["k"], result_df["silhouette"], marker="o", color="darkorange")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette method")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)

print(f"\n결과 저장: {OUT_CSV}, {OUT_PNG}")