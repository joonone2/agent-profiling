# =============================================================================
# run_pipeline.py
# TopicGPT 4단계 파이프라인 실행 (100명 파일럿).
#
# 사전 준비:
#   1. prepare_documents.py를 먼저 실행해서 data/input/movies_100.jsonl 생성
#   2. 환경변수 설정 (OpenRouter 사용):
#        export OPENAI_API_KEY="sk-or-v1-..."      (OpenRouter 키)
#        export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
#   3. pip install topicgpt_python (이미 완료됨)
#
# 모델: openai/gpt-4o-mini (OpenRouter 경유)
#
# 4단계:
#   1) generate_topic_lvl1 — 템플릿(취향 유형) 생성
#   2) refine_topics       — 비슷한 유형 병합, 드문 유형 제거
#   3) assign_topics       — 유저를 유형에 배정 + 근거 인용
#   4) correct_topics      — 배정 결과에서 환각(근거 없는 배정) 교정
#
# ★ seed_1.md는 prompt/seed_1.md (우리 도메인용, 기존 FA/NMF 축과 안 겹치는
#   예시로 새로 작성한 것)를 사용 — TopicGPT 원본 seed가 아님.
# =============================================================================

import argparse
import os

# ★ sentence-transformers(내부적으로 huggingface_hub 사용)가 학교/회사 네트워크의
#   SSL 인터셉션 때문에 모델 다운로드에 실패하는 문제 우회.
#
#   huggingface_hub의 "SSL 검증을 끄는 공식 API"는 버전마다 이름이 바뀜
#   (구버전: configure_http_backend + requests.Session /
#    특정 버전: set_client_factory + httpx.Client 등) — 실제로 두 방식
#   다 시도했으나 설치된 huggingface_hub 버전에 따라 함수 자체가 없어서
#   계속 깨짐. 그래서 특정 API를 쫓는 대신, httpx.Client 자체를 낮은
#   레벨에서 패치해서 "누가 만들든(huggingface_hub 내부 포함) 기본값이
#   verify=False가 되도록" 강제함. httpx 자체의 생성자 시그니처는 안정적
#   이라 huggingface_hub 버전 변화에 영향을 안 받음.
#
#   반드시 "from topicgpt_python import ..." 보다 먼저 실행되어야 함 —
#   topicgpt_python의 generation_1.py/assignment.py/correction.py가
#   모듈 최상단(import 시점)에서 바로 SentenceTransformer를 로드하기 때문.
import warnings
import httpx
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# httpx 패치 (huggingface_hub 최신 버전)
_original_httpx_client_init = httpx.Client.__init__


def _patched_httpx_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    _original_httpx_client_init(self, *args, **kwargs)


httpx.Client.__init__ = _patched_httpx_client_init

# requests 패치 (huggingface_hub 구버전 또는 다른 라이브러리 대비 이중 안전장치)
_original_requests_request = requests.Session.request


def _patched_requests_request(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    return _original_requests_request(self, *args, **kwargs)


requests.Session.request = _patched_requests_request

from pathlib import Path
from dotenv import load_dotenv
from topicgpt_python import (
    generate_topic_lvl1,
    refine_topics,
    assign_topics,
    correct_topics,
)

from prepare_documents import N_USERS  # noqa: E402  (파일명을 자동으로 일치시키기 위해 재사용)

BASE_DIR = Path(__file__).resolve().parent

# .env 파일(BASE_DIR/.env)에서 OPENAI_API_KEY, OPENAI_BASE_URL을 읽어와
# 환경변수로 등록. 이미 셸에서 export한 환경변수가 있으면 그게 우선됨
# (override=False가 기본값이라, .env는 "없을 때만 채워주는" 역할).
load_dotenv(BASE_DIR / ".env")

API = "openai"                    # topicgpt_python 내부적으로 OPENAI_BASE_URL을 봄 -> OpenRouter로 라우팅
MODEL = "openai/gpt-4o-mini"      # OpenRouter 모델 ID 표기법

DATA_FILE = BASE_DIR / "data" / "input" / f"movies_{N_USERS}.jsonl"

PROMPT_DIR = BASE_DIR / "prompt"
SEED_FILE = PROMPT_DIR / "seed_1.md"
GENERATION_PROMPT = PROMPT_DIR / "generation_1.txt"
REFINEMENT_PROMPT = PROMPT_DIR / "refinement.txt"
ASSIGNMENT_PROMPT = PROMPT_DIR / "assignment.txt"
CORRECTION_PROMPT = PROMPT_DIR / "correction.txt"

OUT_DIR = BASE_DIR / "data" / "output" / f"n{N_USERS}"


def check_env():
    """실행 전 환경변수 확인 (없으면 바로 에러로 알려줌, API 낭비 방지)."""
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("OPENAI_BASE_URL"):
        missing.append("OPENAI_BASE_URL")
    if missing:
        raise EnvironmentError(
            f"환경변수 누락: {missing}. "
            f"{BASE_DIR / '.env'} 파일에 다음 두 줄이 있는지 확인하세요:\n"
            f'  OPENAI_API_KEY=sk-or-v1-...\n'
            f'  OPENAI_BASE_URL=https://openrouter.ai/api/v1'
        )
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} 없음. 먼저 prepare_documents.py를 실행하세요."
        )
    print(f"[run_pipeline] 환경변수 확인 완료. BASE_URL={os.environ['OPENAI_BASE_URL']}")


def main(skip_generation: bool = False, skip_refine: bool = False):
    check_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generation_out = OUT_DIR / "generation_1.jsonl"
    topics_lvl1 = OUT_DIR / "topics_lvl1.md"

    refined_out = OUT_DIR / "generation_1_refined.jsonl"   # 유저별 갱신 응답 (JSONL)
    topics_refined = OUT_DIR / "topics_refined.md"          # 정제된 유형 목록 (마크다운)
    mapping_file = OUT_DIR / "topic_mapping.txt"

    assignment_out = OUT_DIR / "assignment.jsonl"
    corrected_out = OUT_DIR / "assignment_corrected.jsonl"

    # skip_refine이면 skip_generation도 자연히 참(1~2단계 전부 건너뜀).
    # 정제 결과(topics_refined.md)가 이미 확정돼 있고, 배정 프롬프트
    # (assignment.txt)만 바꿔서 3단계부터 다시 돌리고 싶을 때 사용.
    if skip_refine:
        skip_generation = True

    # ---- 1) 템플릿(취향 유형) 생성 ----
    if skip_generation:
        if not (generation_out.exists() and topics_lvl1.exists()):
            raise FileNotFoundError(
                f"--skip-generation을 쓰려면 {generation_out}, {topics_lvl1}이 "
                f"이미 있어야 합니다. (이전 실행 결과를 그대로 재사용)"
            )
        print(f"\n[run_pipeline] 1/4 — 건너뜀 (기존 결과 재사용): {topics_lvl1}")
    else:
        print("\n[run_pipeline] 1/4 — 취향 유형 생성 중...")
        generate_topic_lvl1(
            api=API,
            model=MODEL,
            data=str(DATA_FILE),
            prompt_file=str(GENERATION_PROMPT),
            seed_file=str(SEED_FILE),
            out_file=str(generation_out),
            topic_file=str(topics_lvl1),
            verbose=True,
        )
        print(f"[run_pipeline] 1단계 완료: {topics_lvl1}")

    # ---- 2) 정제 (비슷한 유형 병합, 드문 유형 제거) ----
    if skip_refine:
        if not topics_refined.exists():
            raise FileNotFoundError(
                f"--skip-refine을 쓰려면 {topics_refined}이 이미 있어야 합니다. "
                f"(이전 정제 결과를 그대로 재사용, assignment.txt만 바꿔서 3단계부터 재시작)"
            )
        print(f"\n[run_pipeline] 2/4 — 건너뜀 (기존 정제 결과 재사용): {topics_refined}")
    else:
        # 이전 정제 결과(빈 매핑 등)가 남아있으면 이번 재실행과 섞이지 않도록 제거.
        # refine_topics는 mapping_file이 있으면 그 내용을 이어받아 시작하는데,
        # 이전 실행이 항상 빈 매핑({})을 만들었으므로 이어받을 이유가 없음.
        if mapping_file.exists():
            mapping_file.unlink()
            print(f"[run_pipeline] 이전 매핑 파일 제거: {mapping_file}")

        print("\n[run_pipeline] 2/4 — 유형 정제 중...")
        refine_topics(
            api=API,
            model=MODEL,
            prompt_file=str(REFINEMENT_PROMPT),
            generation_file=str(generation_out),
            topic_file=str(topics_lvl1),
            out_file=str(topics_refined),       # 정제된 유형 목록이 저장됨
            updated_file=str(refined_out),      # 유저별 갱신 JSONL이 저장됨
            verbose=True,
            remove=True,
            mapping_file=str(mapping_file),
        )
        print(f"[run_pipeline] 2단계 완료: {topics_refined}")

    # ---- 3) 배정 (유저 -> 유형, 근거 인용 포함) ----
    print("\n[run_pipeline] 3/4 — 유저별 유형 배정 중...")
    assign_topics(
        api=API,
        model=MODEL,
        data=str(DATA_FILE),
        prompt_file=str(ASSIGNMENT_PROMPT),
        out_file=str(assignment_out),
        topic_file=str(topics_refined),
        verbose=True,
    )
    print(f"[run_pipeline] 3단계 완료: {assignment_out}")

    # ---- 4) 교정 (근거 없는 배정 재확인) ----
    print("\n[run_pipeline] 4/4 — 배정 결과 교정 중...")
    correct_topics(
        api=API,
        model=MODEL,
        data_path=str(assignment_out),
        prompt_path=str(CORRECTION_PROMPT),
        topic_path=str(topics_refined),
        output_path=str(corrected_out),
        verbose=True,
    )
    print(f"[run_pipeline] 4단계 완료: {corrected_out}")

    print("\n[run_pipeline] 전체 파이프라인 완료.")
    print(f"  최종 유형 목록: {topics_refined}")
    print(f"  최종 배정 결과(근거 포함): {corrected_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TopicGPT 4단계 파이프라인 실행")
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="1단계(생성)를 건너뛰고, 이미 있는 generation_1.jsonl/topics_lvl1.md를 "
             "재사용해 2단계(정제)부터 실행.",
    )
    parser.add_argument(
        "--skip-refine", action="store_true",
        help="1~2단계(생성·정제)를 모두 건너뛰고, 이미 확정된 topics_refined.md를 "
             "재사용해 3단계(배정)부터 실행. assignment.txt만 바꿔서 배정 방식을 "
             "다시 시도할 때(예: 희소 선택 -> 전체 유형 밀집 점수) 사용.",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation, skip_refine=args.skip_refine)