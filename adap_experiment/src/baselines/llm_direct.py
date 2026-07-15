"""
LLM 기반 베이스라인: Baseline(19dim→LLM), ChatGPT-Direct(이력→LLM)
"""
import logging

from src.llm_client import call_llm, parse_score, parse_ranking
from prompts.templates import (
    BASELINE_RATING_PROMPT,
    BASELINE_RANKING_PROMPT,
    CHATGPT_DIRECT_RATING_PROMPT,
    CHATGPT_DIRECT_RANKING_PROMPT,
)
from config import FEATURE_NAMES

import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# Baseline (19-dim user feature → LLM)
# ══════════════════════════════════════════════════════

def _format_user_features(user_feature_vector: pd.Series) -> str:
    """유저 피처 벡터를 'Action: 4.2, Comedy: 3.1, ...' 형태 텍스트로 변환."""
    parts = []
    for feat in FEATURE_NAMES:
        val = user_feature_vector.get(feat, 0.0)
        parts.append(f"{feat}: {val:.2f}")
    return ", ".join(parts)


def baseline_predict_rating(
    user_feature_vector: pd.Series,
    item_info: dict,
) -> float:
    """
    BASELINE_RATING_PROMPT 사용.
    item_info: {"title": str, "genres": list[str]}
    """
    feature_text = _format_user_features(user_feature_vector)

    prompt = BASELINE_RATING_PROMPT.format(
        feature_text=feature_text,
        title=item_info["title"],
        genres=", ".join(item_info["genres"]),
    )

    response = call_llm(
        system_prompt="You are a movie recommendation system.",
        user_prompt=prompt,
    )

    try:
        return parse_score(response)
    except ValueError:
        logger.warning("baseline_predict_rating: parse failed. Response: %s", response[:200])
        raise


def baseline_rank(
    user_feature_vector: pd.Series,
    candidates: list[dict],
) -> list[int]:
    """
    BASELINE_RANKING_PROMPT 사용, 후보 20개 배치로 한 번에 질의.
    candidates: [{"item_id": int, "title": str, "genres": list[str]}, ...]
    """
    feature_text = _format_user_features(user_feature_vector)
    candidates_text = "\n".join(
        f"  {c['title']} (장르: {', '.join(c['genres'])})"
        for c in candidates
    )
    candidate_titles = {c["item_id"]: c["title"] for c in candidates}

    prompt = BASELINE_RANKING_PROMPT.format(
        feature_text=feature_text,
        n=len(candidates),
        candidates_text=candidates_text,
    )

    response = call_llm(
        system_prompt="You are a movie recommendation system.",
        user_prompt=prompt,
    )

    return parse_ranking(response, candidate_titles)


# ══════════════════════════════════════════════════════
# ChatGPT-Direct (이력 → LLM)
# ══════════════════════════════════════════════════════

def _format_history(user_history: list[dict]) -> str:
    """이력을 '제목 (평점: X.X)' 형태 리스트 텍스트로 변환."""
    return "\n".join(
        f"  - {h['title']} (평점: {h['rating']:.1f})"
        for h in user_history
    )


def chatgpt_direct_predict_rating(
    user_history: list[dict],
    item_info: dict,
) -> float:
    """
    user_history: [{"title": str, "rating": float}, ...] (최근 HISTORY_N개)
    CHATGPT_DIRECT_RATING_PROMPT 사용.
    """
    history_text = _format_history(user_history)

    prompt = CHATGPT_DIRECT_RATING_PROMPT.format(
        history_text=history_text,
        title=item_info["title"],
        genres=", ".join(item_info["genres"]),
    )

    response = call_llm(
        system_prompt="You are a movie recommendation system.",
        user_prompt=prompt,
    )

    try:
        return parse_score(response)
    except ValueError:
        logger.warning("chatgpt_direct_predict_rating: parse failed. Response: %s", response[:200])
        raise


def chatgpt_direct_rank(
    user_history: list[dict],
    candidates: list[dict],
) -> list[int]:
    """
    CHATGPT_DIRECT_RANKING_PROMPT 사용, 후보 20개 배치로 한 번에 질의.
    """
    history_text = _format_history(user_history)
    candidates_text = "\n".join(
        f"  {c['title']} (장르: {', '.join(c['genres'])})"
        for c in candidates
    )
    candidate_titles = {c["item_id"]: c["title"] for c in candidates}

    prompt = CHATGPT_DIRECT_RANKING_PROMPT.format(
        history_text=history_text,
        n=len(candidates),
        candidates_text=candidates_text,
    )

    response = call_llm(
        system_prompt="You are a movie recommendation system.",
        user_prompt=prompt,
    )

    return parse_ranking(response, candidate_titles)
