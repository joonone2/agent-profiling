"""
ADAP Pilot Experiment — 단일 진입점
python run_all.py 한 줄로 전체 파이프라인이 순서대로 실행됩니다.
"""
import sys
import os

# 프로젝트 루트를 sys.path에 추가하여 절대 import가 동작하도록 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.run_pilot import main

if __name__ == "__main__":
    main()
