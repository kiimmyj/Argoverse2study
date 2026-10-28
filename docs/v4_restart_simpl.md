# v4 종료와 SIMPL 기반 재시작

> 2026-10-28. v4의 마지막 실험(100 epoch 최선구성) 결과와, 그에 따라 base architecture를 교체하기로 한 결정을 정리했음.
> 1 epoch 실측은 진행 중이며 결과가 나오면 5절에 추가함.

---

## 1. 결론

1. **구성요소를 쌓는 방향이 막혔음** — 개별로 효과가 확인된 네 가지(smoothing · data cleansing · lane heading alignment · neighborhood vehicles)를 모두 적용하고 epoch을 3배로 늘렸더니 **기준보다 +0.045 나빠졌음**(5 seeds 전부)
2. **구조적 병목을 측정했음** — 후보 경로 집합이 ground truth를 감싸는 정도가 이미 1 m 수준이라, loss·입력 tuning으로는 목표(minADE6 ≤ 1.0)에 닿지 않음
3. **base architecture를 공개 SOTA로 교체하기로 했음** — 조사 결과 **SIMPL**(MIT, AV2 지원, 1.9M)을 선택했음
4. v4의 원래 의도인 **(a, h) 입출력과 θ = h − k 표현은 유지**하되, 기준 차로 k를 후보 경로 열거에서 **최근접 차로**로 단순화했음

---

## 2. 100 epoch 최선구성 — 실패

### 2.1 설정

| | |
| --- | --- |
| 입력 | ah3 (Gaussian σ=0.25 s smoothing + 속도 가중 heading blend + flip gate), **10 Hz** |
| 전처리 | cleansed (lane 거리 point-to-segment 수정 + focal yaw 반전 교정) |
| penalty | coordinate penalty 1.0 |
| auxiliary | lane heading alignment 3.0 |
| 입력 추가 | neighborhood vehicles radius 30 m, **2 Hz** |
| 학습 | 100 epoch cosine, 5 seeds, parameter 556,153 |

### 2.2 결과

| condition | 평균 minADE6 | seed별 | range | 1.50 이상 |
| --- | --- | --- | --- | --- |
| 기준 (coordinate penalty만, 30 epoch) | **1.3591** | 1.359 / 1.356 / 1.362 | 0.006 | 0판 |
| 최선구성 (위 전부, 100 epoch) | **1.4038** | 1.393 / 1.423 / 1.413 / 1.406 / 1.383 | 0.039 | 0판 |

1. **+0.045 나빠졌고 5 seeds 전부 기준보다 나쁨** — seed 잡음이 아님
2. 개별로 확인됐던 효과(cleansed −0.043, lane-yaw의 lane change −0.069)가 **합치면 더해지지 않았음.** 상쇄되거나 서로 간섭함
3. 1.50 부근으로 이탈하는 optimization 실패는 **0판**이었음(5 seeds, range 0.039) — 이 구성에서는 그 불안정이 나타나지 않았음

### 2.3 한계 — 원인을 분리하지 못했음

1. 30 epoch 대조판을 함께 돌리도록 설계했으나 **실행되지 않았음**
2. 원인은 본 세션의 **감시 스크립트 버그**임 — 첫 epoch 성공을 `001/100` 패턴으로 찾았는데 실제 로그는 `01/100`이라, 정상 통과(325초)했음에도 실패로 읽고 체인을 중단시켰음
3. 따라서 "구성요소를 쌓은 탓"과 "100 epoch cosine 탓"이 **분리되지 않음**. 다만 두 경우 모두 결론(이 방향으로는 목표에 못 감)은 바뀌지 않음

---

## 3. 왜 구조를 바꾸는가

### 3.1 후보 경로 집합의 기하적 한계

mode를 후보 경로(candidate route)로 정의하는 현재 구조에서, **경로 집합 자체가 ground truth를 얼마나 잘 감싸는지**를 측정했음 (val 24,988, seed 0).

| | 오라클 경로 적중률 | top-1 경로–GT 거리 | **오라클 경로–GT 거리** |
| --- | --- | --- | --- |
| 기준 (coordinate penalty 1.0) | 36.4% | 1.523 m | **1.079 m** |
| + 중심선 smoothing | 47.0% | 1.533 m | 1.081 m |
| + route history | 33.0% | 1.532 m | 1.079 m |

1. **후보 중 가장 좋은 경로조차 ground truth에서 평균 1.079 m 떨어져 있음.** 현재 minADE6(1.31~1.36)과 같은 자릿수임
2. d를 예측해 중심선에서 벗어날 수 있으므로 엄밀한 하한은 아니나, **경로 집합이 병목임을 시사함**
3. 중심선 smoothing은 오라클 경로 적중률을 10.6 pp 올렸으나 **거리는 그대로임** — 거의 동률인 후보 사이의 교체라 실익이 없었음

### 3.2 개별 구성요소의 효과 크기

| 변경 | paired Δ minADE6 | 판정 |
| --- | --- | --- |
| action → coordinate penalty | 0.000 (cost 없이 violation 0.29×) | 채택 |
| data cleansing | −0.043 | 3 seeds 부호 일치 |
| lane heading alignment 3.0 | +0.017 (전체) / **−0.069 (lane change)** | lane change에서만 |
| route history | −0.04 (수렴한 seed 기준) | 불안정 |
| neighborhood vehicles 30 m | −0.028 | 검출 불가 |
| smoothing (ah3) | −0.053 | 검출 불가 |
| downsampling 2 Hz | **+0.072** | 3 seeds 모두 **나쁨** |

1. 개별 효과가 **0.005~0.08 규모**임. 목표까지 필요한 0.31을 이 방향으로 메울 수 없음
2. 그리고 2절에서 확인했듯 **합쳐도 더해지지 않음**

---

## 4. 새 base architecture 선정

### 4.1 조사 결과 — 실제로 fork 가능한 것은 셋뿐

| model | AV2 test minADE6 | license | 재현 가능 val | 공개 학습 코드 |
| --- | --- | --- | --- | --- |
| QCNet | 0.65 | Apache-2.0 | 0.72 | 있음 — **단일 GPU 불가** |
| DeMo | 0.61 | **없음** | 0.645 | 있음 |
| Forecast-MAE | 0.709 | **없음** | 0.721 | 있음 |
| **SIMPL** | 0.72 | **MIT** | 0.777 | 있음 |
| ProphNet / GANet / ADAPT | 0.68 / 0.728 / 0.694 | — | — | **공개 코드 없음** |

1. **ProphNet · GANet · ADAPT는 코드가 공개된 적이 없음.** GANet 저장소는 "Code will be released soon"만 3년째이고 ADAPT는 프로젝트 웹페이지뿐임
2. **QCNet은 compute로 탈락** — 학습에 GPU memory 약 160 GB가 필요하고(8×3090 권장), 저자도 원인을 찾지 못한 memory leak이 있어 4×RTX 4090(24 GB)에서 3 epoch만에 OOM이 보고됨
3. EDA는 Waymo 전용, HPNet은 AV1 전용, HPTR은 4×2080Ti로 5일 이상 + 비상업 license임

### 4.2 SIMPL을 선택한 이유 — 정확도가 아니라 risk

1. **컴파일 확장이 없음** — 순수 PyTorch + `nn.MultiheadAttention`임. 본 머신의 GPU가 **sm_120 (Blackwell)**인데, Forecast-MAE의 NATTEN과 DeMo의 mamba-ssm은 이 아키텍처에서 빌드된다는 확인이 없음. 현재 환경(torch 2.13+cu130, sm_120 포함)에서 SIMPL은 그대로 동작함
2. **셋 중 유일하게 license가 있음**(MIT)
3. **decoder가 이미 output parameterization을 switch로 갖고 있음** — `param_out ∈ {bezier, monomial, none}`에 기저 행렬이 붙어 있어, **(a, Δθ) + Frenet integrator가 네 번째 분기로 들어감.** 전면 수정이 아님
4. **입력 교체가 2곳**이고 **재전처리가 필요 없음** — 캐시에 heading(`TRAJS_ANG_OBS`)과 velocity가 이미 있어 `a`는 속력의 1차 차분임
5. `yaw_loss`가 이미 있어 heading supervision이 존재함

### 4.3 알려진 risk

1. **재현 정확도가 셋 중 가장 낮음** — 저자가 논문 checkpoint를 분실해 재학습한 공개 checkpoint가 val **0.777**임(논문 0.72). DeMo(0.645)보다 0.13 나쁨
2. **작은 batch에서 성능이 크게 떨어짐** — 파생 모델(DGFNet) 저자가 단일 3090 batch 16에서 맞춘 LR schedule을 공개해 두었으므로 그것을 출발점으로 삼음
3. **AV2 test 제출 코드가 없음** — val 평가만 가능함
4. 공개된 학습 시간이 전부 multi-GPU임(8×3090 11시간). 단일 GPU 실측이 필요함 — 5절

---

## 5. 설계 결정과 1 epoch 실측

### 5.1 확정된 설계

| 항목 | 결정 | 의미 |
| --- | --- | --- |
| (a, h)·θ 적용 범위 | **입력판과 입출력판을 둘 다 만들어 비교** | ① 입력만 (a,h) + SOTA decoder ② 출력도 (a, Δθ) + Frenet |
| 기준 차로 k | **최근접 차로** | 후보 경로 열거가 사라짐 — v4 복잡도의 주원인이었음 |
| 기존 v4 자산 | **전부 새로** | 별도 디렉터리, 평가 harness도 새로 작성함 |

1. k를 최근접 차로로 바꾸면 route 열거기 · route_sd0 · sub-mode · 경로 중복제거가 전부 불필요해짐
2. mode는 SOTA 방식(anchor/proposal)으로 정의되고, **θ는 표현일 뿐 mode 축이 아니게 됨**
3. 3.1절에서 측정한 "오라클 경로 1.079 m" 병목도 함께 사라짐
4. **대가** — off-lane penalty가 공짜가 아니게 됨. v4에서는 d가 상태 변수라 hinge 하나로 끝났으나, 최근접 차로 기준이면 매 step 다시 계산해야 함. 1차에서는 off-lane penalty를 제외하고 시작함

### 5.2 1 epoch 실측 (진행 중)

1. 공개된 학습 시간이 전부 multi-GPU라 **단일 GPU 기준 epoch 시간을 모름**
2. 세 후보 모두 scenario당 pickle 하나를 읽는 구조이고, DeMo에서 8 GPU 9시간 vs 2 GPU 10시간이 보고된 바 있어 **IO bound일 가능성이 높음** — 단일 GPU 시간이 단순 N배보다 훨씬 나을 수 있음
3. 따라서 commit 전에 **전처리 + 1 epoch 실측**을 먼저 함
4. 환경은 기존 `av2` conda 환경을 그대로 사용함 (torch 2.13.0+cu130, sm_120 지원 확인, av2-api 설치됨)
5. 적용한 수정은 Python 3 호환 2건뿐임 — `from fractions import gcd` → `math.gcd`, `import imp` → `importlib`

**(실측 결과는 여기에 추가 예정)**
