"""
통계적 베이스라인: UserKNN, MF (LLM 미사용)
"""
import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from surprise import SVD, Dataset, Reader

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# UserKNN
# ══════════════════════════════════════════════════════

class UserKNNModel:
    """유저×아이템 평점 행렬 기반 UserKNN 모델."""

    def __init__(self, train_df: pd.DataFrame, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.train_df = train_df
        self.global_mean = train_df["rating"].mean()

        # 유저/아이템 인덱스 매핑
        self.user_ids = sorted(train_df["userId"].unique())
        self.item_ids = sorted(train_df["movieId"].unique())
        self.user_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_idx = {iid: i for i, iid in enumerate(self.item_ids)}

        # 유저별 평균
        self.user_means = train_df.groupby("userId")["rating"].mean().to_dict()

        # 유저×아이템 행렬 (밀집)
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        self.matrix = np.full((n_users, n_items), np.nan)
        for _, row in train_df.iterrows():
            ui = self.user_idx.get(int(row["userId"]))
            ii = self.item_idx.get(int(row["movieId"]))
            if ui is not None and ii is not None:
                self.matrix[ui, ii] = row["rating"]

        # 평균 중심화 행렬 (NaN → 0 처리, 코사인 유사도 계산용)
        self.matrix_centered = self.matrix.copy()
        for i, uid in enumerate(self.user_ids):
            mean_r = self.user_means.get(uid, self.global_mean)
            mask = ~np.isnan(self.matrix_centered[i])
            self.matrix_centered[i, mask] -= mean_r
        self.matrix_centered = np.nan_to_num(self.matrix_centered, nan=0.0)

        # 유저 간 코사인 유사도
        self.sim_matrix = sklearn_cosine(self.matrix_centered)

    def predict(self, user_id: int, item_id: int) -> float:
        ui = self.user_idx.get(user_id)
        ii = self.item_idx.get(item_id)

        if ui is None or ii is None:
            return self.global_mean

        # 해당 아이템을 평가한 유저들 찾기
        rated_mask = ~np.isnan(self.matrix[:, ii])
        if not rated_mask.any():
            return self.global_mean

        # 유사도 상위 k_neighbors명
        sims = self.sim_matrix[ui].copy()
        sims[ui] = -1  # 자기 자신 제외
        sims[~rated_mask] = -1  # 해당 아이템 미평가 유저 제외

        top_k_idx = np.argsort(sims)[-self.k_neighbors:][::-1]
        top_k_idx = top_k_idx[sims[top_k_idx] > 0]  # 양의 유사도만

        if len(top_k_idx) == 0:
            return self.global_mean

        user_mean = self.user_means.get(user_id, self.global_mean)
        num = 0.0
        den = 0.0
        for ni in top_k_idx:
            neighbor_uid = self.user_ids[ni]
            neighbor_mean = self.user_means.get(neighbor_uid, self.global_mean)
            s = sims[ni]
            r = self.matrix[ni, ii]
            num += s * (r - neighbor_mean)
            den += abs(s)

        if den == 0:
            return self.global_mean

        pred = user_mean + num / den
        return float(max(1.0, min(5.0, pred)))


def userknn_predict(train_df: pd.DataFrame, user_id: int, item_id: int, k_neighbors: int = 20) -> float:
    """
    편의 함수 — 매 호출마다 모델을 새로 만들면 비효율적이므로,
    run_pilot.py에서는 UserKNNModel 인스턴스를 미리 만들어 .predict()를 직접 호출할 것을 권장.
    이 함수는 인터페이스 호환성을 위해 남겨둠.
    """
    model = UserKNNModel(train_df, k_neighbors)
    return model.predict(user_id, item_id)


# ══════════════════════════════════════════════════════
# MF (surprise.SVD)
# ══════════════════════════════════════════════════════

def mf_train(
    train_df: pd.DataFrame,
    n_factors: int = 20,
    n_epochs: int = 20,
    lr: float = 0.01,
    reg: float = 0.02,
):
    """
    surprise 라이브러리의 SVD를 사용한 Matrix Factorization.
    반환: 학습된 surprise SVD 모델 객체.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        train_df[["userId", "movieId", "rating"]], reader
    )
    trainset = data.build_full_trainset()

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr,
        reg_all=reg,
        random_state=42,
    )
    model.fit(trainset)
    logger.info("MF (SVD) training complete. n_factors=%d, n_epochs=%d", n_factors, n_epochs)
    return model


def mf_predict(model, user_id: int, item_id: int) -> float:
    """surprise SVD 모델로 평점 예측."""
    pred = model.predict(user_id, item_id)
    return float(max(1.0, min(5.0, pred.est)))


# ══════════════════════════════════════════════════════
# 공통 유틸: 후보 랭킹
# ══════════════════════════════════════════════════════

def rank_by_stat_method(predict_fn, user_id: int, candidate_ids: list[int]) -> list[int]:
    """
    candidate_ids 각각에 predict_fn 적용 후 점수 내림차순 정렬해서 반환.
    predict_fn signature: (user_id, item_id) -> float
    """
    scores = [(cid, predict_fn(user_id, cid)) for cid in candidate_ids]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scores]
