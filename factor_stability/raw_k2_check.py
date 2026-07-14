"""
axis_space_rawclustered_{method}.png를 raw_cluster K=2 (elbow/silhouette
최적값)로 다시 그리는 스크립트.

★ 기존 실험(축 추출, K=3 behavioral 결과, stability/validity 등)은
  전혀 건드리지 않음 — 색칠 그림 하나만 새로 만드는 용도.
★ 새 로직을 짜지 않고 visualize_phase0.py에 있는
  cluster_raw_feature_space() / plot_axis_space_raw_clustered()를
  그대로 재사용함 (원본 로직과 어긋나지 않게 하기 위함).
★ 기존 K=3 버전 파일을 덮어쓰지 않도록 별도 폴더(raw_k2_check/)에 저장.

실행 위치: factor_stability/ 폴더 안에서
    python raw_k2_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_phase0 import (
    resolve_io_dir,
    load_results,
    cluster_raw_feature_space,
    plot_axis_space_raw_clustered,
    DEFAULT_K,
)

RAW_K = 2  # optimal_k_raw.py의 elbow/silhouette 결과

io_dir = resolve_io_dir(DEFAULT_K)
print(f"기존 결과 읽는 위치: {io_dir}")

scores, feats = load_results(io_dir)

out_dir = io_dir / "raw_k2_check"
out_dir.mkdir(exist_ok=True)
print(f"새 그림 저장 위치: {out_dir}\n")

# 기존 함수 그대로 재사용, n_clusters만 2로 지정
raw_cluster_df = cluster_raw_feature_space(feats, n_clusters=RAW_K)

for method in ["FA", "NMF", "SHAP"]:
    plot_axis_space_raw_clustered(scores, feats, method, out_dir, raw_cluster_df)

print(f"\n완료. raw_cluster K={RAW_K} 버전:")
for method in ["FA", "NMF", "SHAP"]:
    print(f"  - {out_dir}/axis_space_rawclustered_{method}.png")
print(f"\n기존 K=3 버전과 비교: {io_dir}/axis_space_rawclustered_{{method}}.png")