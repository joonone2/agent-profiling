"""
Factor Loading → LLM → K개 Agent System Prompt 생성 (1회성)
"""
import os
import json
import logging

import pandas as pd

from src.llm_client import call_llm
from prompts.templates import AGENT_SYNTHESIS_PROMPT
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def synthesize_agent(
    factor_id: str,
    top_features: list[tuple[str, float]],
) -> str:
    """
    AGENT_SYNTHESIS_PROMPT를 사용해 LLM 호출.
    top_features: interpret_factors()의 출력 [(feature_name, loading_value), ...]
    반환: 생성된 System Prompt 문자열
    """
    loadings_text = "\n".join(
        f"  {name}: {value:.4f}" for name, value in top_features
    )

    prompt = AGENT_SYNTHESIS_PROMPT.format(
        factor_id=factor_id,
        loadings_text=loadings_text,
    )

    system_prompt = call_llm(
        system_prompt="You are a helpful assistant that creates movie critic personas.",
        user_prompt=prompt,
    )

    logger.info("Synthesized agent for %s: %s…", factor_id, system_prompt[:80])
    return system_prompt


def synthesize_all_agents(
    loading_matrix: pd.DataFrame,
    interpreted: dict,
    method: str | None = None,
) -> dict[str, str]:
    """
    K개 Factor에 대해 각각 1번씩만 LLM 호출하여 System Prompt 생성.
    결과를 캐싱(파일 저장)하여 재사용.

    Args:
        method: "nmf" 또는 "fa" — 캐시 파일명을 method별로 구분.
                None이면 기존 파일명(agent_system_prompts.json) 사용.

    반환: {factor_id: system_prompt_str}
    """
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    suffix = f"_{method}" if method else ""
    cache_path = os.path.join(factors_dir, f"agent_system_prompts{suffix}.json")

    # 캐시가 이미 있으면 로드해서 재사용
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if set(cached.keys()) == set(interpreted.keys()):
            logger.info("Agent system prompts loaded from cache: %s", cache_path)
            return cached
        logger.info("Cache key mismatch, regenerating agent prompts.")

    result: dict[str, str] = {}
    for factor_id, top_features in interpreted.items():
        system_prompt = synthesize_agent(factor_id, top_features)
        result[factor_id] = system_prompt

    # 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("All %d agent system prompts synthesized and saved to %s.", len(result), cache_path)
    return result

