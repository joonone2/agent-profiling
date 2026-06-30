# factor_stability — Phase 0 실험 파이프라인

행동 축 추출의 안정성·해석가능성·외부 타당성을 검증하는 연구용 파이프라인.
NMF / Factor Analysis / SHAP 세 기법을 비교하고, 데이터 볼륨 곡선으로
"데이터가 많을수록 축이 안정적으로 수렴한다"는 핵심 가설을 검증합니다.

---

## 1. 데이터 준비

### MovieLens-1M (권장)

```
https://files.grouplens.org/datasets/movielens/ml-1m.zip
```

다운로드 후 압축 해제 → 아래 구조로 두세요:

```
factor_stability/
└── data/
    └── ml-1m/
        ├── ratings.dat
        ├── movies.dat
        └── users.dat
```

### MovieLens-100K (빠른 테스트용)

```
https://files.grouplens.org/datasets/movielens/ml-100k.zip
```

100K는 파일 구조가 다릅니다 — data_loader.py의 구분자/컬럼명을 확인하세요.
1M 권장.

---

## 2. config.py 수정

```python
# config.py 상단
DATA_DIR = Path("data/ml-1m")   # ← 본인 경로로 수정
```

나머지 상수(K, N_SEEDS, 임계값 등)는 기본값으로 실행 가능합니다.

---

## 3. 설치

```bash
pip install -r requirements.txt
```

---

## 4. 배선 확인 (먼저 실행)

MovieLens 없이도 가짜 데이터로 파이프라인 전체가 에러 없이 도는지 확인:

```bash
pytest tests/test_smoke.py -v
```

모든 테스트가 PASSED이면 배선이 정상입니다.

---

## 5. 실행

```bash
python run_phase0.py
```

볼륨 곡선 때문에 전체 실행 시간이 길 수 있습니다 (1M 기준 수십 분).
빠른 확인이 필요하면 config.py에서:

```python
VOLUME_FRACTIONS = [0.1, 1.0]   # 2개만
N_SEEDS = 5                      # 반복 줄이기
```

---

## 6. 산출물 (results/ 폴더)

| 파일                         | 내용                                                      |
| ---------------------------- | --------------------------------------------------------- |
| `stability_comparison.csv` | 기법별 재현성(평균±표준편차), 측정지표, 축 자동생성 여부 |
| `axis_interpretation.csv`  | 기법별 K=3 축 상위 피처 + loading + 희소도 + 설명분산     |
| `external_validity.csv`    | 축 점수 vs 외부 신호 상관 (유의미성 검증)                 |
| `volume_curve.csv`         | 볼륨 곡선 원데이터                                        |
| `volume_curve.png`         | 볼륨 곡선 그림 (메인)                                     |
| `method_comparison.png`    | 세 기법 재현성 비교 막대                                  |

콘솔에 GO/NO-GO 종합 판정이 출력됩니다.

---

## 7. 파일 구조

```
factor_stability/
├── config.py          # 모든 상수·경로 (여기만 수정)
├── data_loader.py     # 데이터 로드 + k-core + feature/validation 분리
├── features.py        # 피처 테이블 + 기법별 전처리
├── extractors.py      # NMF/FA/SHAP 동일 인터페이스
├── stability.py       # split-half 재현성 측정
├── interpret.py       # 해석가능성 + 노이즈 축
├── validity.py        # 외부 타당성 (순환논리 방지)
├── volume_curve.py    # 볼륨 곡선 실험
├── plots.py           # 그림 생성
├── run_phase0.py      # 전체 실행 진입점
├── requirements.txt
├── README.md
├── data/              # MovieLens 데이터 (직접 다운로드)
├── results/           # 산출물 자동 생성
└── tests/
    └── test_smoke.py  # 가짜 데이터 배선 테스트
```

---

## 8. GO/NO-GO 기준

안정성 하나만으로 판정하지 않습니다. 네 가지 종합:

| 조건                                            | 판정         |
| ----------------------------------------------- | ------------ |
| 안정성≥0.85 + 이름붙음 + 외부상관 + 노이즈없음 | ✅ GO        |
| 안정적이나 외부 타당성 약함                     | ⚠️ CAUTION |
| 안정성 0.70~0.85                                | 🔶 WEAK      |
| 안정성 <0.70                                    | ❌ REVIEW    |

---

## 9. 흔한 문제

**NMF에 음수 에러**: features.py의 MinMax 전처리가 NMF 전에 적용됐는지 확인.
**SHAP 메모리 부족**: config.py의 `SHAP_TRAIN_SAMPLE`을 0.5로 줄이세요.
**factor_analyzer import 에러**: `pip install factor_analyzer` 별도 설치 필요.
**한글 폰트 깨짐**: plots.py의 그림 라벨은 의도적으로 영어입니다.