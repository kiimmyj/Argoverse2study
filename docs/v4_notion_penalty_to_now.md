# v4 Progress Report — Previous Action Item 대응

> 기간 2026-09-16 ~ 09-30 · dataset Argoverse 2 (train 199,908 / val 24,988)
> baseline: L4 · jitter penalty 1.0 · 30 epoch cosine LR · seed 0~2
> 모든 비교는 **paired-seed**(같은 seed끼리 짝지어 차이를 봄) 기준임

---

## 1. Penalty — 꼭 필요한가, 필요하다면 어떻게

### 1.1 결론

1. **필요함.** 단 accuracy cost가 있음 — minADE6 +0.058 m (약 4%) 증가, 대신 kinematic violation rate가 17배 감소함
2. **현재 형태는 부적절함.** penalty가 누르는 변수가 실제 trajectory의 heading 변화를 0.3%만 설명함
3. **개선안을 구현했음** — penalty target을 model output에서 **predicted coordinate**로 교체, 3 seeds 학습 중임

### 1.2 필요성 검증 — paired-seed A/B

| condition | minADE6 (last 5 epochs) | violation rate (>7.3°/step) | jitter | off-band |
| --- | --- | --- | --- | --- |
| no penalty (seed 0 / 1) | **1.342 / 1.363** | 17.45% / 19.39% | 1.78 / 1.90 | 0.47 / 0.49 |
| penalty 1.0 (seed 0 / 1) | 1.407 / 1.415 | **1.05% / 1.14%** | **0.003** | 0.53 / 0.52 |
| paired difference | **+0.058 m** | −17배 | −600배 | 차이 없음 |

1. 두 seed에서 부호가 일치해 effect가 확인됨
2. 기존에 보고한 accuracy cost(0.106 m)를 **0.058 m로 정정했음** — 기존 값은 seed variance와 섞여 있었음
3. penalty를 제거하면 predicted trajectory의 약 20%가 실제 차량이 낼 수 없는 yaw rate를 가짐

### 1.3 metric 결함 — self-referential measurement

| condition | model output 기준 | predicted coordinate 복원 기준 |
| --- | --- | --- |
| penalty 1.0 · 30 epoch | 0.005% | **0.735%** |
| penalty 0.1 | 0.001% | **1.324%** (1,191배 차이) |
| ground truth label | — | 0.351% |

1. violation rate를 **model이 출력한 각도**에서 측정하고 있었음 — penalty가 직접 최소화하는 변수임
2. term별 제거 실험으로 기여도를 분리했음

| removed term | violation rate (baseline 0.735%) |
| --- | --- |
| integrator geometry correction | **0.186%** (−75%) |
| route curvature | 0.423% (−42%) |
| **model output (Δθ)** | **0.733% (−0.3%)** |

3. model output을 전부 제거해도 violation rate가 0.3%만 변함
4. 원인 — internal heading `h = k(s) + θ`가 model이 실제로 생성하는 trajectory의 heading과 다름

![](figures/v4/feasibility/m1_2_self_vs_xy.png)

### 1.4 Literature review

1. kinematic integrator/control output으로 violation을 0%로 만든 사례 3편은 **smoothness loss term이 없음** (DKM, PTNet, TPK)
2. 단 "smooth output ≠ accurate heading"이 반복 보고됨 — Bézier basis를 써도 heading error는 별도 loss로 처리해야 했음 (SIMPL: minFYE6 0.297 → 0.076)
3. 동일한 self-reference 문제를 지적한 논문 3편 확인 — MultiPath++ (model output 기준 0.00% ↔ coordinate 기준 1.22%), Greer YawLoss, Grad-CAPS
4. smoothness evaluation의 사실상 표준은 모두 **coordinate 기반**임 (WOSAC angular velocity distribution, curvature, turning radius)
5. seed variance를 다룬 STEP(2025)은 "작은 성능 차이는 training noise로 설명되는 경우가 많다"고 보고함 — 본 프로젝트 관측과 일치함

### 1.5 개선안 — coordinate-based penalty

1. predicted coordinate에서 복원한 heading의 step 변화가 7.3°를 초과하는 분량만 hinge로 penalize함
2. unit test 통과 — straight 0, 임계 내 turn(7°/step) 0, stationary 0, zigzag(20°/step)만 penalty 발생함
3. 3 seeds 학습 대기 중임

---

## 2. Preprocessing = smoothing + downsampling

### 2.1 결론

1. **downsampling은 현행 유지함** — 순서(smoothing → sampling)와 RAMP_SKIP=4가 측정상 타당했음
2. **기존 smoothing은 no-op였음** — filter cutoff가 2.38 Hz라 2 Hz sampling의 anti-aliasing 역할을 전혀 못 했음
3. **새 smoothing은 input quality를 개선했으나 model accuracy 이득은 검출되지 않았음** → 채택 보류함

### 2.2 기존 방식 무효 확인

| metric (2 Hz input, straight scenarios) | no smoothing | 기존 SG(5,2) |
| --- | --- | --- |
| heading jitter p90 | 3.23 °/s | **3.23 °/s** |
| zigzag ratio | 36.7% | 36.6% |
| velocity-field course angle과의 차 | 1.04° | 1.05° |

### 2.3 Data characterization

1. signal과 noise가 교차하는 frequency는 **0.6~1.0 Hz**였음 — stationary vehicle(pure noise)과 moving vehicle의 position spectrum이 그 부근에서 만남
2. position noise 구성 — low-frequency drift σ 13.5 cm + white noise σ 0.77 cm + heavy tail(stationary step의 4.02%가 1 m/s 초과)
3. AV2 position은 dataset 단계에서 이미 smoothing되어 있었음 → 본 작업은 second-pass smoothing에 해당함

![](figures/v4/smoothing/char_2_spectrum.png)

### 2.4 변경 내용

| item | 변경 | 근거 |
| --- | --- | --- |
| position smoothing | Gaussian σ = 0.25 s (cutoff 0.53 Hz), 경계는 2nd-order polynomial extrapolation | signal이 우세한 대역만 통과시킴. 1st-order 경계 처리는 window 끝 yaw rate를 0.33배로 감쇠시킴 |
| heading | speed-weighted circular blend (crossover 3 m/s) | 기존 hard switch(1 m/s)가 step당 ±150 °/s의 discontinuity를 생성했음 |
| 180° flip correction | 실제 이동 구간에서만 판정 | stationary vehicle은 position noise가 판정을 결정했음 |
| downsampling | 변경 없음 (smoothing → 5-step sampling) | downsampling만으로 heading jitter p90이 6.04 → 3.23 °/s로 감소함 |

### 2.5 결과 — before vs after

| metric | 기존 | 변경 후 |
| --- | --- | --- |
| straight heading jitter p90 | 2.70 °/s | **2.22 °/s** (−18%) |
| jerk p90 | 9.75 m/s³ | **7.61 m/s³** (−22%) |
| turn preservation ratio | 1.022 | **1.022** (감쇠 없음) |
| model accuracy (3 seeds, paired) | — | **−0.053 ± 0.069 m, 부호 불일치 → 검출 불가** |

1. input quality는 개선됐으나 accuracy 개선은 seed variance와 구분되지 않았음
2. smoothing의 효과는 noise가 큰 **neighborhood vehicle input**에서 재검증할 예정임

![](figures/v4/smoothing/panel_straight_1.png)

---

## 3. Neighborhood vehicles

### 3.1 결론

1. 요청대로 **radius 30 m** 단순 방식으로 구현 완료했음
2. seed 0 결과는 +0.026 m(소폭 악화) — 3 seeds를 채워 판정 예정임

### 3.2 구현

1. input — radius 30 m 내 vehicle의 past 5 s를 0.5 s 간격으로, channel 5개(relative position 2, sin/cos heading, observation mask)
2. encoder — vehicle별 MLP embedding 후 focal trajectory feature를 query로 attention pooling함
3. parameter 450,681 → 556,153
4. 기존 neighbor cache를 재사용했음(가까운 32대·past 5 s·focal frame·미래 미사용), past position 포함본을 추가로 생성했음

### 3.3 평가 계획

1. 전체 minADE6로는 효과가 보이지 않을 가능성이 큼 — lane change 분석에서 "prediction window 내 신규 시작 309건"을 별도 평가군으로 지정했음

---

## 4. U-turn / Lane change

### 4.1 결론

1. **U-turn은 제외함** — val 73건(0.29%)으로 희소함
2. **lane change는 제시된 두 가설 모두 기각됨**

| hypothesis | 판정 | 근거 |
| --- | --- | --- |
| (1) smoothing으로 해결됨 | 기각 | input heading error는 7.3° → 1.9°로 개선되나 lane change metric은 seed range 내에서 변화 없음 |
| (2) rule(1.75 m band) 추가 | 기각 | band는 binding constraint가 아님. 3.6 m 적용군 463건 hit rate 12.5% vs 1.75 m 적용군 159건 11.9% |

### 4.2 실제 원인

1. **lateral mode axis 부재** — 동일 route의 sub-mode 간 종방향 endpoint spread 11.32 m, 횡방향 0.31 m (실패의 55.7%)
2. **candidate route coverage 부족** — ground truth를 ±1.75 m 내에 담는 route가 존재하는 비율 3.7%
3. **관측 불가능 구간** — lane change의 49%가 prediction window에서 신규 시작함
4. model은 initial residual angle을 정상적으로 수신하고도 ground truth보다 3배 빠르게 0으로 수렴시킴

### 4.3 규모와 처방

1. lane change 627건(2.51%), U-turn 73건(0.29%)
2. lane change를 완전히 해결해도 전체 minADE6 개선 상한은 **−0.015 m**로, seed range(0.105)의 1/7임 — accuracy metric으로는 관측되지 않는 문제임
3. 원인 ①에 대한 처방으로 동일 route의 2·3번째 mode를 ±lane width(3.42 m)로 유도하는 auxiliary loss를 구현했음, 3 seeds 대기 중임

---

## 5. Data cleansing — 폐기 vs 보정

### 5.1 결론

1. **폐기 대상은 거의 없음** — 결함 2건을 수정하면 폐기 대상이 train의 0.17%(약 340건)로 감소함
2. pedestrian·cyclist는 폐기 시 오히려 성능이 저하됨 (val minADE 1.490 → 1.522)

### 5.2 전수 집계 (train + val 224,896건)

| item | ratio | 해석 |
| --- | --- | --- |
| no candidate route | 3.9% | 81.5%가 pedestrian·cyclist. vehicle만 0.78% |
| off-road | 4.6% | 99.7%가 pedestrian·cyclist |
| coverage failure | 14.4% | turn·lane change가 대부분이라 폐기 불가 |
| our heading의 급격한 변화 | 0.79% | AV2 원본 0.01% — roughness는 preprocessing에서 유입된 것임 |

### 5.3 발견 결함 2건 (수정 완료)

1. **candidate lane distance를 centerline vertex 기준으로 측정하고 있었음**
   - vehicle "no route" 사례의 91%가 실제로는 lane polyline 위에 있었음(중앙 0.49 m)
   - vertex 간격이 5 m를 초과하는 구간에서 lane 위 차량이 search radius 밖으로 배제됐음
   - point-to-segment distance로 수정했음
2. **AV2 body heading이 반전된 scenario 18건의 minADE6이 15.07 m였음** (전체 1.49)
   - normalization frame과 candidate route filter가 동일 각도를 사용해 scene 전체가 반전됨
   - 이동 구간 기준 판정으로 교정했음

### 5.4 폐기 불가 항목

1. window edge ramp (첫/끝 0.5 s의 position-derived speed가 실제의 약 절반)
2. stationary track의 heading 미정의 (33.4%)
3. slip angle

---

## 6. Visualization framework

### 6.1 결론

1. agent 기반 자동화 framework를 정식화했음 — 정성 평가 산출물이 재현 가능해졌음
2. 요청하신 **scenario grouping + ID 기록**을 산출물 규격에 반영했음

### 6.2 구성

1. agent 정의 파일에 필수 요건 11가지를 명시했음 — time series, 상태별 통계, scenario별 지표, loss 분해, grouped small multiples, unit 통일, step marker, 대표 scenario, 자체 검증, 특성 표, decision tree
2. 표준 작업 A(checkpoint 분석: dump → cases → stats → gallery), 표준 작업 B(epoch별 학습 방향)로 고정했음
3. 모든 분석에서 **독립 검토 agent**가 핵심 수치를 재계산함 — 지금까지 3회 수행, 지적 사항 60여 건 반영했음
4. scenario ID는 유형별로 묶어 json에 저장함 — 예: lane change 분석의 `id_groups.json`(실패 유형 A/B/C/D × 시작 시점 12칸)

---

## 7. 비교 방법론 — seed variance (신규 발견)

### 7.1 문제

1. 동일 설정을 seed만 바꿔 실행한 결과가 1.407 / 1.415 / **1.547**이었음 (range 0.140)
2. 기존에 "effect"로 보고하던 차이(0.02~0.11)보다 큼
3. seed 2는 training failure가 아님 — train loss가 가장 낮았음. 더 낮은 training loss가 더 낮은 generalization으로 이어진 사례임

### 7.2 해법 — paired-seed comparison

| comparison | paired difference | seed별 부호 | 판정 |
| --- | --- | --- | --- |
| 10 Hz → 2 Hz input | +0.113 ± 0.016 | 일치 | **2 Hz가 유의하게 나쁨** |
| no penalty → penalty 1.0 | +0.058 | 일치 | **accuracy cost 확인** |
| smoothing 적용 | −0.053 ± 0.069 | 불일치 | 검출 불가 |

### 7.3 적용 규칙

1. 단일 seed 차이가 0.15 m 미만이면 effect로 보고하지 않음
2. 신규 주장은 seed 3개 이상, 평균과 range를 함께 기재함
3. best epoch 단일 값으로 비교하지 않고 last-5-epoch 평균을 사용함

---

## 8. 진행 중 실험

| experiment | 판정 대상 | 상태 |
| --- | --- | --- |
| no penalty × 3 seeds | penalty의 accuracy cost | 2 seeds 완료 (+0.058 m) |
| neighborhood vehicles × 3 seeds | neighbor input 효과 | 1 seed 완료 (+0.026 m) |
| coordinate-based penalty × 3 seeds | penalty target 교체 효과 | 대기 |
| cleansed preprocessing × 3 seeds | 결함 2건 수정 효과 | 대기 |
| lateral mode axis × 3 seeds | lane change | 대기 |

### 실행 환경 메모

1. cache는 preprocessing source의 hash로 키가 결정됨 — 학습 대기 중 `src/`를 수정하면 cache miss로 즉시 실패함. 실제로 이 문제로 13시간의 GPU idle이 발생했음
2. 대응 — 남은 12판을 **단일 script**로 재구성했음. batch 3판씩 병렬, 앞 batch 종료 후 다음 batch 시작, 중간에 cache를 다시 생성함(약 12분)
3. 학습 진행 중에는 `src/` 수정을 중단함

---

## 9. Summary

1. jitter penalty는 accuracy cost +0.058 m로 kinematic violation rate를 17배 낮춤 — 유지하되 형태를 교체할 근거를 확보했음
2. 기존 violation metric은 penalty가 최소화하는 변수를 그대로 측정하는 self-referential metric이었음
3. coordinate 기준으로 보면 violation의 주항은 model output이 아니라 **integrator geometry**였음 (기여 75%)
4. preprocessing smoothing은 기존 방식이 무효였고, 신규 방식은 input quality를 개선했으나 accuracy 이득은 검출되지 않았음
5. lane change 실패 원인은 smoothing도 band rule도 아닌 **lateral mode axis 부재**였음
6. data cleansing 대상은 폐기가 아니라 보정이 적절함 — 결함 2건 수정으로 폐기 대상이 0.17%까지 감소함
7. 가장 중요한 방법론적 발견은 **seed variance 0.140 m**이며, 이후 모든 비교를 paired-seed로 전환했음
