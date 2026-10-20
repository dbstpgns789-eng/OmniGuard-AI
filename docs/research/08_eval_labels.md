# 08. 사람 판단 정답 + 적은 영상에서의 평가 방법 (Aside, 2026-10-19)

## 핵심
- **정상 영상 2분으로는 헛정지율을 거의 주장할 수 없다.** 헛정지 0건이어도 95% 상한이 분당 약 1.5건(시간당 약 90건, rule of three). 시간당 1건 미만을 주장하려면 0건 기준으로도 정상 영상 약 3시간 필요. 정상 영상을 늘리거나 사고 영상 안의 "두 평가자 모두 정상" 구간을 노출 시간에 더한다.
- **같은 사고에서 나온 crop들은 한 그룹으로 묶어야 한다.** 안 묶으면 영상 하나씩 빼기에서 누수, 재현율 분모가 부풀려짐 (빨간 상의 확대본과 iSafetyBench 넓은 화면은 같은 사고).
- **정답 정의를 바꾸면 결론이 뒤집힌 것은 정상적인 신호.** 결론을 정답 하나로 보고하지 말고 여러 정의(접촉 시점, A, B, 교집합, 합집합)에 대한 민감도 분석으로.
- 사람끼리 일치도는 원래 높지 않다(전문가 EEG 발작 판독 κ 0.58). **사람 대 사람 일치도를 "천장"으로 같이 보고**하는 것이 적은 데이터에서 가장 설득력 있다.

## 출처
### 평가자 간 일치도
- **S1 Halford 2015, ICU EEG 발작 판독 일치도** https://pmc.ncbi.nlm.nih.gov/articles/PMC4439396 — 전문가 8명. 시계열 라벨은 인접 프레임끼리 상관이 강해 κ의 독립성 가정 위반 → 1초 이진 시퀀스의 0/1 구간을 순열해 우연 일치를 경험적으로. 이벤트 일치: 최소 1초 겹치면 일치(FEA), 겹친 시간 Z/(X+Y−Z)(FEDA). **발작 κ 0.58**, 인증 받은 평가자 쌍이 더 높음 → 프레임 κ + 위험 구간 단위 일치 같이, 우연 일치는 순열·순환 이동으로
- **S2 Hallgren 2012 튜토리얼 + Krippendorff 기준** https://pmc.ncbi.nlm.nih.gov/articles/PMC3402032 — α ≥ 0.800 신뢰, 0.667~0.800 잠정 결론만, 0.667 미만 폐기. **같은 1~5점 데이터에서 nominal α 0.451, ordinal α 0.887** → 4단계 라벨은 반드시 ordinal, 합격선 사전 등록
- **S3 Feinstein & Cicchetti 1990, 높은 일치·낮은 κ 역설** — 관찰 일치 0.95, 기대 우연 0.932면 κ 0.26 → 우리 영상은 대부분 "정상"이라 전체 κ가 왜곡. **활성 구간(첫 비정상 5초 전 ~ 접촉)**에서 따로 계산하고 원시 일치율·±1단계 일치율도
- **S4 Dreem 수면 데이터셋 2020** https://arxiv.org/abs/1911.03221 — 다수결 합의, 사람 평가자 각각을 나머지의 합의와 비교. 모델 F1 89.9% vs 사람 평균 86.8% → 평가자 2명이면 "B를 시스템처럼 놓고 A 기준으로 같은 지표"를 사람 천장으로

### 구간·이벤트 재현율, 리드타임, 헛정지 예산
- **S5 Tatbul 2018 (NeurIPS) 시계열 정밀도·재현율** https://arxiv.org/abs/1803.03639 — 구간 재현율 = α×존재 + (1−α)×겹침, 조기 대응이 중요하면 앞쪽 가중. 도구 IntelLabs/TSAD-Evaluator → 1차 α=1(존재), 2차 앞쪽 가중 겹침
- **S6 SzCORE 2024 + timescoring** https://github.com/esl-epfl/timescoring — 시작 전 허용 30초, 종료 후 60초, 90초 미만 간격 병합. 24시간당 오경보율. **사용자는 24시간당 오경보 수를 수용성의 가장 중요한 기준으로 봄** → 허용창·병합·시간당 비율 구조를 초 단위로 줄여 사전 등록
- **S7 DSTA 2021, 교통사고 예측** https://ar5iv.labs.arxiv.org/html/2106.10197 — mTTA, TTA@R80. **데이터셋을 교차하면 AP 급락(DAD→CCD 35.99%)** → "한 영상에 맞춘 임계값이 안 옮겨진다"는 우리 경험의 문헌 근거
- **S8 Accident Anticipation 2025** https://arxiv.org/html/2510.22260v1 — 오경보율이 허용 범위일 때만 재현율·TTA 보고. **기존 3초 넘는 TTA는 이상 징후 훨씬 이전의 오경보를 조기 예측으로 잘못 센 것** → 헛정지 예산을 먼저 고정, 너무 이른 경보는 헛정지로
- **S9 ISO 13855 해설** — 요구 리드타임 = 파이프라인 지연 + 디바운스 + 기계 정지 시간

### 적은 영상에서의 보고
- **S10 Varoquaux 2018, 작은 표본의 교차검증 오차** — 표본 100개에서도 ±10%, 폴드 간 표준오차는 과소추정 → 5개 영상 LOVO의 "평균 ± SE"는 무의미, 폴드별 값을 전부 표로
- **S11 Hanley 1983, 0건의 해석** — n번 중 0건이면 95% 상한 약 3/n. 비율에는 Garwood 정확 구간. **정상 2분 헛정지 0건 → 상한 분당 1.5건(양측 1.84건)** → "헛정지 없음"이라고 쓰면 안 되고 상한과 노출 시간을 같이
- **S12 Brown, Cai, DasGupta 2001** — n<40이면 Wilson·Jeffreys 권장, Clopper-Pearson은 보수적. **4/5 탐지 시 CP 95% CI ≈ [0.28, 0.99], 5/5여도 하한 ≈ 0.48**
- **S13 Field & Welsh 2007, 군집 부트스트랩** — 재표집 단위는 프레임이 아니라 영상(사고 그룹). 군집 5개면 이론이 성립 안 하니 참고용
- **S14 Wong 2021 (JAMA IM), 상용 패혈증 모델 외부 검증** — 개발사 AUC 0.76~0.83 → 외부 0.63, 패혈증 67% 놓침, 입원 18%에 경보 → **개발 데이터에서 고른 임계값을 고정하고 새 데이터에 한 번만 적용**, 경보 부담도 보고

### 4단계 비대칭 평가
- **S15 Elkan 2001 비용 민감 학습, Gaudette 2009 서열 분류** — 비용행렬은 결정 단계에서 임계값으로 반영, 서열 분류에서 MAE가 견고 → 4×4 비용행렬 사전 등록, 과소(위험한데 낮게)와 과잉(한 단계 더 조심) 분리

## 공식과 코드 (Aside 제공, 직접 실행 안 해 봄)
```python
import numpy as np, krippendorff
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2
LEVELS = [0, 1, 2, 3]; FPS = 5   # 0.2초 간격

def onset(x, lvl):
    idx = np.flatnonzero(np.asarray(x) >= lvl); return idx[0] if len(idx) else None

def agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    out = dict(exact=np.mean(a == b), within1=np.mean(np.abs(a - b) <= 1),
               kappa_lin=cohen_kappa_score(a, b, labels=LEVELS, weights="linear"),
               kappa_quad=cohen_kappa_score(a, b, labels=LEVELS, weights="quadratic"),
               alpha_ord=krippendorff.alpha(reliability_data=np.vstack([a, b]).astype(float),
                                            level_of_measurement="ordinal", value_domain=LEVELS),
               cm=confusion_matrix(a, b, labels=LEVELS))
    for lvl in (1, 3):
        oa, ob = onset(a, lvl), onset(b, lvl)
        out[f"donset_ge{lvl}_s"] = None if oa is None or ob is None else (ob - oa) / FPS
    return out

def shift_null(a, b, stat, n=2000, min_shift=25, seed=0):   # 자기상관 고려 우연 일치
    rng = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b); obs = stat(a, b)
    null = [stat(a, np.roll(b, rng.integers(min_shift, len(b) - min_shift))) for _ in range(n)]
    return obs, float(np.mean(np.array(null) >= obs))

proportion_confint(4, 5, alpha=0.05, method="beta")    # 약 (0.284, 0.995)
def poisson_rate_ci(k, exposure_min, conf=0.95):          # Garwood
    a = 1 - conf; lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2; hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo / exposure_min, hi / exposure_min           # poisson_rate_ci(0, 2.0) -> (0, 약 1.84/분)

C = np.array([[0, 1, 2, 3], [3, 0, 1, 2], [6, 3, 0, 1], [20, 10, 4, 0]])  # 행=실제, 열=예측, 예시값
```
- 리드타임 L = 접촉 시각 − 경보 시각, 사람 대비 마진 M = 사람 정지 시각 − 경보 시각
- 유효 경보창 = [사람 정지 − τ_early, 접촉]. 창보다 이른 경보는 헛정지
- 두 평가자 합치기(Aside 제안): **3구역 정답** — 둘 다 정지면 "반드시 잡아야 함", 한 명만 정지면 "논쟁 구간"(헛정지로 안 셈), 둘 다 정상이면 "합의 정상"(여기서 울리면 헛정지). 서열 지표는 min(A,B)를 관대한, max(A,B)를 엄격한 정답으로 두고 "시스템 단계가 [min, max] 안에 든 비율"
- 허용 수준(사전 등록): 활성 구간 α_ordinal ≥ 0.667 잠정, ≥ 0.800 신뢰. 새 4단계 주관 판단은 0.6~0.8이 현실적. 정지 시작 차이 중앙값 ≤ 0.6초(제안). 미달이면 라벨 기준(루브릭)을 고쳐 다시 라벨링

## 이번 주 평가 프로토콜 (사고 5개, 평가자 2명, 정상 약 2분)
| # | 지표 | 계산 | 합격선(제안) |
|---|---|---|---|
| 1 | 평가자 일치도 | 활성 구간 α_ordinal, 선형 κ, ±1단계 일치율, 정지·경고 시작 차이 | α ≥ 0.667, 정지 시작 차이 중앙값 ≤ 0.6초 |
| 2 | 사고 이벤트 재현율 | 유효 창 안에서 정지가 울린 사고 그룹 / 5, CP 95% CI | 보고만 (n=5) |
| 3 | 리드타임 | 접촉 − 경보, 사람 대비 마진. 영상별·**최솟값**·중앙값 | 최솟값 ≥ 요구 리드타임 |
| 4 | 헛정지율 | 병합 경보 수 / 합의 정상 노출(분), Garwood CI. 정상 클립과 사고 영상 정상 구간 분리 | 예: ≤ 0.5/분 (점추정과 상한 모두) |
| 5 | 헛정지 예산 운영점 | 학습 폴드에서 예산 안 재현율 최대 임계값 → 테스트 영상 | 폴드 간 임계값 변동 폭도 |
| 6 | TTA@R80 | 재현율 ≥ 4/5 지점의 평균 리드타임 | 보고만 |
| 7 | 서열 오류 | 4×4 혼동행렬, 과소율·과잉율, 치명적 놓침(정지인데 경고 이하), 기대 비용, 사람 범위 적중률 | 치명적 놓침 0 |
| 8 | 사람 천장 | B를 시스템으로 놓고 A 기준 #2~#7 (반대도) | 시스템과 나란히 |
| 9 | 정답 민감도 | #2~#7을 정답 = {접촉, A, B, A∩B, A∪B}로 각각 | 모든 정답에서 유지될 때만 "주장" |

**순서:** ① B 라벨 공개 전 사전 등록(위 표, τ_early, 병합 간격, 헛정지 예산, 비용행렬, 그룹 정의, 임계값 그리드)을 1페이지로 쓰고 git 커밋으로 고정 ② 현재 시스템을 v1으로 동결, B 라벨에 **한 번만** 평가 ③ 일치도 계산, 미달이면 루브릭 보완 ④ 3구역·경계 정답 생성 ⑤ LOVO + in-sample 차이 ⑥ 사람 천장, 정답 민감도 표
**보고서에 꼭:** 영상별 표(영상, 그룹, A/B 정지 시작, 차이, 접촉, 첫 경보, 리드타임, 마진, 적중, 이른 헛정지, κ, α), 집계 수치에는 정확 구간과 n("재현율 4/5 (CP 95% CI 0.28~0.99)", "헛정지 0건 / 2.0분 (95% 상한 1.84/분)"), LOVO와 in-sample 나란히, 한계(사고 5건은 성능 주장이 아니라 방법론 검증과 실패 분석, 정상 영상 30분 이상 확보 권장)

## 확인 못 한 것
ISO 13855 원문, TTA@R80 최초 제안 논문, Gaudette·Cardoso 세부 수치, Elkan 행/열 규약, krippendorff.bootstrap_ci, 코드 실행. 3구역 정답·사람 범위 적중률·순환 이동·비용행렬 값·합격선은 Aside 제안.

## Claude 메모
- **채우 님 판단표가 오기 전에 사전 등록 문서를 먼저 커밋해야 한다.** 판단표를 본 뒤에 기준을 정하면 이 프로토콜의 의미가 사라진다. 세훈과 상의할 1순위.
- 실험 12·13·14의 "분당 헛정지"는 노출 2분이라 상한이 매우 넓다. 발표에서 "0.5/분"을 쓸 때는 신뢰구간을 같이 써야 한다.
- 정상 영상 확보: Safe and Unsafe Behaviours의 나머지 클립(주제 05)으로 30분 이상 만들 수 있다.
