"""
ADAP Pilot Experiment — Global Configuration
"""
import os
from dotenv import load_dotenv, find_dotenv

# .env 파일을 현재 위치에서 상위로 탐색하며 자동 로드
load_dotenv(find_dotenv())

# ── 경로 ──────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ── 실험 파라미터 ─────────────────────────────────────
K_FACTORS = 3                       # Factor(=에이전트) 개수
N_NEGATIVES = 19                    # 후보셋 오답 개수 (정답 1 + 오답 19 = 20)
N_PILOT_USERS = 100                   # 파일럿 샘플 유저 수 (디버깅 중이므로 5로 축소)
TEST_RATIO = 0.2                    # 유저별 시간순 분할 비율
RANDOM_SEED = 42
HISTORY_N = 20                      # ChatGPT-Direct / Ours+History에 제공할 최근 이력 개수
POSITIVE_RATING_THRESHOLD = 4.0     # 후보셋 정답 판정 기준
FACTOR_METHODS = ["nmf", "fa"]      # NMF/FA 이중 실행: 각 method별로 Ours 변형이 생성됨

# ── 디버깅용: 이력 프롬프트 검증 시 나머지 방법론/랭킹 트랙 건너뛰기 ──
# True로 두면 UserKNN/MF/Baseline/ChatGPT-Direct 및 랭킹 트랙 전체를 건너뛰고
# Ours/Ours+History의 레이팅 트랙만 실행한다. 검증이 끝나면 False로 되돌릴 것.
DEBUG_SKIP_OTHER_METHODS = False

# ── LLM 설정 (OpenRouter — OpenAI 호환) ──────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"   # OpenRouter 모델 슬러그
LLM_TEMPERATURE = 0.0                      # 재현성을 위해 기본 0
MAX_RETRIES = 3

# ── 장르 피처 목록 (MovieLens-1M 기준 18개 장르) ─────
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# 19번째 피처: popularity_bias
FEATURE_NAMES = GENRES + ["popularity_bias"]  # 총 19개, 순서 고정


