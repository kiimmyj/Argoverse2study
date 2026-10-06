# 10/7 meeting

> Previous Action Item 대응. dataset Argoverse 2 (train 199,908 / val 24,988), 30 epoch cosine, 조건당 **seed 3판**.
> 모든 비교는 **paired-seed**(같은 seed끼리 짝지어 차이를 보고 부호 일치를 판정) 기준임.
> feasibility 지표는 전부 **predicted coordinate에서 복원한 heading** 기준임 — model output 기준 아님(1.2 참조).

---

## 1. 벌점 — 꼭 필요한가, 필요하다면 어떻게

### 1.1 결론

1. **필요함** — penalty 없이 학습하면 top-1 trajectory step의 **8.47%**가 label 상한(7.3°/step)을 넘음. ground truth label은 0.35%임(**24배**)
2. **단 현재 형태는 부적절했음** — penalty target을 model output(action)에서 **predicted coordinate**로 교체했고, accuracy cost 없이 violation을 0.29배로 낮췄음
3. **2026-10-06 기본 설정을 전환 완료했음** — `--smooth-mode xy --smooth-xy 1.0`
4. 추가로 **lane heading alignment loss**를 도입했고, 지금까지 어떤 방법으로도 움직이지 않던 **lane change 평가군을 처음으로 개선했음**(1.7)

### 1.2 기존 metric의 결함 — self-referential measurement

1. violation rate를 **model이 출력한 각도**에서 측정하고 있었음 — penalty가 직접 최소화하는 변수라 penalty를 걸면 정의상 내려감
2. 같은 checkpoint를 두 기준으로 재면 **0.005% vs 0.735%** 로 갈림
3. 따라서 penalty 설정 비교에 사용할 수 없고, 모든 수치를 **predicted coordinate 기준**으로 재측정했음 — ψ = atan2(Δy, Δx), |Δψ| > 7.3°/step, 양쪽 step 속력 1 m/s 이상일 때만 셈
4. 재현 확인 — 재채점한 minADE6가 학습 로그 best와 소수점 3자리까지 일치했음

### 1.3 Literature review

1. kinematic integrator/control output으로 violation을 0%로 만든 사례 3편은 **smoothness loss term이 없음** (DKM arXiv:1908.00219, PTNet, TPK arXiv:2505.06743)
2. 동일한 self-reference 문제를 지적한 논문 확인 — **MultiPath++**(arXiv:2111.14973, 표 4: model output 기준 0.00% ↔ coordinate 기준 1.22%), Greer YawLoss, Grad-CAPS
3. smoothness evaluation의 사실상 표준은 전부 **coordinate 기반**임 (WOSAC angular velocity distribution, curvature, turning radius)
4. **현행 action penalty도 근거가 있었음** — CAPS(arXiv:2012.06644) / Grad-CAPS(arXiv:2407.04315)의 RL action smoothness regularization임. 즉 이번 변경은 "근거 없음 → 있음"이 아니라 **두 문헌 갈래 중 선택을 바꾼 것**임
5. 채택한 형태는 Greer et al., *Lane Heading Auxiliary Loss*(arXiv:2011.06679)의 **YawLoss** 구성을 따름 — ① predicted coordinate 2점의 arctan ② tolerance hinge ③ 모든 live mode 적용

### 1.4 측정 결과 (조건당 3 seeds)

| condition | minADE6 | seed range | violation (top-1) | paired Δ vs no penalty |
| --- | --- | --- | --- | --- |
| no penalty | 1.359 | 0.035 | 8.47% | — |
| **coordinate penalty 1.0** | **1.359** | **0.006** | 2.28% | **+0.000 (부호 불일치)** |
| coordinate penalty 3.0 | 1.442 | 0.137 | 1.13% | +0.083 (일치) |
| action penalty 1.0 (기존) | 1.440 | 0.105 | 0.76% | +0.081 (일치) |
| action + coordinate | 1.459 | 0.107 | 0.79% | +0.100 (일치) |
| ground truth label | — | — | 0.35% | — |

1. **8.47% → 2.28% 구간은 accuracy cost 없이 얻어짐** — paired 차이가 seed별 +0.018 / −0.003 / −0.014로 부호가 갈림
2. **2.28% 아래로 내리는 것부터 비용 발생** — target이 action이든 coordinate든 약 +0.08 m로 동일함. cost는 target이 아니라 **penalty 강도**의 함수임
3. seed range가 penalty 강도에 비례함 — 7장 참조

### 1.5 회전량별 분해 — 비용이 어디에서 발생하나

ground truth의 총 heading 변화량으로 val을 3구간으로 나눠 재측정했음(직진 15,711 / 완만 5,256 / 회전 4,021건).

| \|Δψ\| median [°/step] | 직진 <5° | 완만 5–30° | 회전 ≥30° |
| --- | --- | --- | --- |
| ground truth label | 0.12 | 0.35 | **1.10** |
| action penalty 1.0 (기존) | 0.06 | 0.09 | **0.16** |
| coordinate penalty 1.0 | 1.69 | 1.62 | 2.11 |
| coordinate 3.0 | 0.38 | 0.55 | 0.89 |
| **coordinate 1.0 + lane-yaw 3.0** | 0.73 | 0.95 | **1.34** |

1. **penalty의 accuracy cost는 turn에 집중됨** — action penalty의 paired 차이가 직진 +0.055 / 완만 +0.093 / **회전 +0.164 m**로 3배임. jitter만 억제하는 것이 아니라 **실제 회전까지 억제하고 있음**
2. ground truth는 직진 0.12° → 회전 1.10°로 **9배 증가**함. action penalty는 2.7배에 그쳐 turn 구조를 재현하지 못함
3. **lane-yaw가 turn에서 ground truth에 가장 근접함**(1.34 vs 1.10)

### 1.6 lane heading alignment loss (신규)

1. predicted coordinate의 heading을 그 mode가 주행하는 **candidate route의 tangent angle**에 맞추는 hinge loss임
2. tolerance 15°는 ground truth 분포에서 결정했음 — |ψ_gt − lane tangent|의 p95(route가 3 m 이내일 때). 전체 p50 1.11° / p90 10.41° / p99 90.39°
3. ground truth 자체의 tail이 두꺼워(p99 90°) **|d| ≤ 3 m gate와 hinge ratio clamp**를 함께 적용했음

| condition | minADE6 | violation (top-1) | 비용 |
| --- | --- | --- | --- |
| coordinate penalty 1.0 | 1.359 | 2.28% | 기준 |
| **+ lane-yaw 1.0** | **1.372** | **1.33%** | **+0.013 m** |
| **+ lane-yaw 3.0** | **1.376** | **1.28%** | **+0.017 m** |
| coordinate penalty 3.0 (비교) | 1.442 | 1.13% | +0.083 m |

1. violation을 2.28% → 1.33%로 낮추는 cost가 **0.013 m**임. 같은 폭을 penalty weight로 사면 **0.083 m로 6배 비쌈**
2. 즉 **penalty를 강화하는 것보다 "어디로 돌아야 하는지"를 지시하는 쪽이 효율적임**
3. 전체 accuracy 기준으로는 weight 1.0이 knee point임(1.372 vs 1.376). 단 lane change에서는 다름(1.7)

### 1.7 lane change 전용 평가군 — 처음으로 일관된 개선이 나왔음

lane change는 val의 2.51%라 전체 지표로는 판정되지 않음(4.4 참조). 전용 평가군에서 재측정했음.

| condition | minADE6 (627건) | paired Δ | seed별 부호 | violation |
| --- | --- | --- | --- | --- |
| coordinate penalty 1.0 (기준) | 2.001 | — | — | 2.27% |
| + lane-yaw 1.0 | 1.985 | −0.016 | 불일치 | 0.89% |
| **+ lane-yaw 3.0** | **1.932** | **−0.069** | **일치 (−0.064 / −0.054 / −0.090)** | **0.78%** |
| coordinate penalty 3.0 | 1.991 | −0.011 | 불일치 | 0.57% |
| action penalty 1.0 (기존) | 2.011 | +0.010 | 불일치 | 0.25% |

1. **lane-yaw 3.0이 3 seeds 모두 개선 방향임** — smoothing · band rule · neighborhood vehicles · lateral mode axis가 전부 검출 불가였던 문제에서 **처음으로 일관된 효과가 확인됨**
2. 전체 비용은 +0.017 m임 — lane change에서 −0.069 m를 얻는 교환임
3. **방향성이 반대인 두 종류의 loss가 구분됨**

| 평가군 | coordinate 3.0 (smoothness 강화) | lane-yaw 3.0 (방향 지시) |
| --- | --- | --- |
| lane change 627건 | −0.011 (불일치) | **−0.069 (일치)** |
| prediction window 내 신규 1,083건 | **+0.061 (일치, 악화)** | −0.007 (불일치) |
| U-turn 73건 | **+0.201 (일치, 악화)** | +0.016 (불일치) |

4. smoothness penalty를 강화하면 lane change와 U-turn이 **악화됨** — 횡방향 운동을 같이 억제하기 때문임. lane-yaw는 악화시키지 않음 — lane change 중에는 그 mode의 route가 목표 차로이므로 정렬이 오히려 도움이 됨
5. 사전에 기록했던 위험("lane-yaw가 lane change를 역방향으로 억제할 수 있음")은 **실현되지 않았고 반대 방향이었음**

---

## 2. Preprocessing = smoothing + downsampling

### 2.1 Downsampling

1. 10 Hz → 2 Hz는 paired **+0.113 ± 0.016 m**(3 seeds 부호 일치)로 **유의하게 나쁨**
2. 입력 step 수가 1/5로 줄어 가속·heading 변화를 놓치는 것으로 해석함
3. **10 Hz를 유지하기로 했음**

### 2.2 Smoothing — 기존 방식이 무효였음

1. 기존 Savitzky–Golay(5,2)는 **출력이 입력과 동일한 no-op**이었음 — 2차 다항식을 5점에 적합하면 중앙값이 그대로 나옴
2. 교체 구현 — Gaussian σ=0.25 s(−3 dB 0.53 Hz) + 2nd-order boundary extrapolation, 속도 가중 circular heading blend(crossover 3 m/s), flip 판정 gate(속력 ≥2 m/s · 이동 ≥5 m), 초기 ramp 4 step 제외
3. **heading vs time 일관성은 개선됨** — 직진 시나리오 input heading error p99가 7.3° → 1.9°
4. **그러나 accuracy 이득은 검출되지 않았음** — paired −0.053 ± 0.069, 3 seeds 부호 불일치
5. before/after plot은 `panel_straight_*.png`에 있음 (직진 시나리오 heading vs time)

---

## 3. Neighborhood vehicles

1. 요청대로 **radius 30 m** 단순 방식으로 구현했음
2. input — radius 30 m 내 vehicle의 past 5 s를 0.5 s 간격, channel 5개(relative position 2, sin/cos heading, observation mask)
3. encoder — vehicle별 MLP embedding 후 focal trajectory feature를 query로 attention pooling. parameter 450,681 → 556,153
4. **3 seeds 결과 — 전체 지표로는 효과가 검출되지 않음**

| 평가군 | paired Δ minADE6 | seed별 부호 | 판정 |
| --- | --- | --- | --- |
| 전체 (24,988) | −0.028 ± 0.048 | +0.019 / −0.026 / −0.076 | 불일치 — 검출 불가 |
| lane change (627) | −0.041 | +0.009 / −0.102 / −0.028 | 불일치 — 검출 불가 |

5. 단 이 판은 **action penalty 위에서 학습**했고 현재 baseline은 coordinate penalty임 — 공정 비교를 위해 **같은 penalty 위에서 재학습 예정**임

---

## 4. U턴 / 차로변경

### 4.1 U턴

1. val 73건(**0.29%**)으로 희소함 — 지시대로 **제외**함

### 4.2 차로변경 — 제시된 두 가설 모두 기각됨

| hypothesis | 판정 | 근거 |
| --- | --- | --- |
| (1) smoothing을 잘 하면 heading tracking으로 해결됨 | **기각** | input heading error는 7.3° → 1.9°로 개선되나 lane change metric은 seed range 내에서 변화 없음 |
| (2) rule(1.75 m band)을 섞음 | **기각** | band가 binding constraint가 아님 — 3.6 m 적용군 463건 hit rate 12.5% vs 1.75 m 적용군 159건 11.9% |

### 4.3 실제 원인

1. **lateral mode axis 부재** — 동일 route의 sub-mode 간 종방향 endpoint spread 11.32 m, **횡방향 0.31 m**(실패의 55.7%)
2. **candidate route coverage 부족** — ground truth를 ±1.75 m 내에 담는 route 존재 비율 3.7%
3. **관측 불가능 구간** — lane change의 **49%**가 prediction window에서 신규 시작함

### 4.4 처방과 결과

1. 원인 ①에 대해 동일 route의 2·3번째 mode를 ±lane width(3.42 m)로 유도하는 auxiliary loss를 구현했음
2. 3 seeds 결과 — 전체 −0.013 ± 0.060, lane change 627건 −0.010, **둘 다 부호 불일치로 검출 불가**
3. 규모상 한계 — lane change 627건(2.51%)을 완전히 해결해도 전체 minADE6 개선 상한이 **−0.015 m**임. **전체 metric으로는 관측되지 않는 문제**이므로 전용 평가군으로만 판정 가능함
4. 부작용 점검 — alive 6 modes 기준 violation이 1.72% → 2.98%로 증가했으나 top-1은 0.76% → 0.78%로 불변임. 즉 횡으로 벌어진 보조 mode에서만 증가했음
5. **해결된 경로는 1.7의 lane heading alignment loss임** — lane change 627건에서 −0.069 m(3 seeds 부호 일치)

---

## 5. v4 데이터 cleansing — 폐기 vs 보정

### 5.1 결론 — 보정이 적절함

1. 지시대로 **전수 집계**했음 (train + val 224,896건)
2. 결함 2건을 수정하면 **폐기 대상이 train의 0.17%(약 340건)로 감소**함 — 폐기할 규모가 아님
3. pedestrian·cyclist는 폐기 시 오히려 성능 저하됨 (val minADE 1.490 → 1.522)

### 5.2 발견한 결함 2건 (수정 완료)

1. **lane 거리 계산 오류** — vertex 거리로 재고 있었음. point-to-segment 거리로 교체했음
2. **focal yaw 반전** — heading이 180° 뒤집힌 scenario를 교정했음. 추가로 align_ref에 gate를 걸었음

### 5.3 재학습 결과

1. 수정 preprocessing으로 3 seeds 재학습 — paired **−0.043 ± 0.049 m**(−0.003 / −0.028 / −0.098)
2. 크기는 작으나 **3 seeds 모두 개선 방향**이므로 보정이 유효함을 확인했음

---

## 6. Visualization framework

1. 전용 agent를 정의했음 (`.claude/agents/v4-visualizer.md`) — checkpoint 분석 / epoch별 학습 방향 분석 2가지 표준 작업
2. 지시하신 **필수 요건 11가지**를 agent 정의에 명시했음 (시계열, 상황별 통계, 묶음 그래프, 단위 통일, step마다 점, 대표 시나리오, 독립 검증, 특성 표, decision tree)
3. **시나리오 유형별 묶음 + 번호 기록을 구현했음** — 분류별 scenario_id를 JSON으로 저장함

| group | 정의 | 건수 |
| --- | --- | --- |
| D2_횡이동 | 기하 신뢰 & \|Δd6\| ≥ 2.5 m & \|Δh6\| < 30° | 627 |
| S_예측중새로 | prediction window에서 신규 시작 | 1,083 |
| S_관측중시작 / S_중간 | 관측 구간에서 시작 / 중간 | 506 / 387 |
| UT_U턴 | U-turn | 73 |
| D1 / D3 / D4 / U | 분류 / 지도차로 / 횡이동∩지도 / 합집합 | 496 / 1,771 / 422 / 1,976 |

4. 저장한 ID 목록으로 **임의의 부분집합에서 판을 비교하는 도구**를 만들었음 — scenario별 지표를 저장해 두고 group별로 paired 차이를 집계함. 1.7 · 3 · 4장의 전용 평가군 수치가 이 도구의 결과임

---

## 7. 비교 방법론 — seed variance (신규 발견)

1. 동일 설정을 seed만 바꿔 실행한 결과가 1.407 / 1.415 / **1.547**이었음(range 0.140). 기존에 "effect"로 보고하던 차이(0.02~0.11)보다 큼
2. seed 2는 training failure가 아님 — train loss가 **가장 낮았음**. 더 낮은 training loss가 더 낮은 generalization으로 이어진 사례임
3. **정정 — seed variance는 조건에 의존함**

| condition | minADE6 seed range |
| --- | --- |
| coordinate penalty 1.0 | **0.006** |
| no penalty | 0.035 |
| action penalty 1.0 | 0.105 |
| coordinate penalty 3.0 | 0.137 |

4. seed range가 **penalty 강도와 함께 커짐**. 즉 "seed noise 0.140"은 dataset 고유 성질이 아니라 **strong penalty 조건의 optimization 불안정**이었음
5. 부수 효과로 penalty를 coordinate 1.0으로 바꾸면 **재현성도 개선됨**(0.140 → 0.006)
6. **적용 규칙** — ① 1차 기준은 paired 차이의 **부호 일치 여부**임 ② 고정 임계("0.15 m 미만은 보고 안 함")는 사용하지 않고 조건별 range를 함께 기재함 ③ 신규 주장은 seed 3개 이상 ④ 효과가 특정 scenario에 국한될 변경은 전체 지표로 판정하지 않고 **전용 평가군**을 둠

---

## 8. 실험 현황

조건당 3 seeds · 30 epoch cosine. 총 **30판** 완료함.

| experiment | 결과 |
| --- | --- |
| no penalty | 완료 — +0.081 m, violation 8.47% |
| coordinate penalty 1.0 | 완료 — **cost 검출 불가, violation 0.29×** → **기본 설정 채택** |
| coordinate penalty 3.0 / action+coordinate | 완료 — +0.083 / +0.100 m |
| lane heading alignment 1.0 | 완료 — +0.013 m에 violation 1.33% |
| **lane heading alignment 3.0** | 완료 — +0.017 m, **lane change 627건에서 −0.069 m (3 seeds 일치)** |
| neighborhood vehicles / lateral mode axis | 완료 — 전체·전용 평가군 모두 검출 불가 |
| cleansed preprocessing | 완료 — −0.043 m, 3 seeds 부호 일치 |

### 다음 단계

1. 최종 penalty 설정 — 전체 accuracy 우선이면 coordinate 1.0, lane change까지 보면 **coordinate 1.0 + lane-yaw 3.0**(전체 +0.017 m에 lane change −0.069 m)
2. neighborhood vehicles · lateral mode axis를 **coordinate penalty 위에서 재학습** — 현재 비교는 baseline이 달라 공정하지 않음
3. route centerline smoothing 실험 진행 중 — 후보 경로 전체의 거칠기가 ground truth의 2.4배이고 smoothing이 42%를 제거함. **alive/top-1 violation 비**로 판정하기로 사전 등록했음
4. 남은 공통 결함 — 모든 판의 |Δψ| p99가 6.4~8.1°로 ground truth(4.50°)보다 두꺼움. hinge가 아니라 **분포 자체를 겨냥하는 term**이 필요할 수 있음

### 실행 환경 메모

1. preprocessing cache는 **source hash로 key가 결정됨** — 학습 대기 중 `src/`를 수정하면 cache miss로 queue 전체가 즉시 실패함. 실제로 3회 발생했음(13시간 GPU idle, batch 2건 전멸)
2. 대응 — queue가 도는 동안 **모든 session이 `src/` 수정을 중단**함. 불가피하면 수정을 한 commit에 묶고 직후 기본 cache를 재생성함(약 20분)
3. 출력이 동일한 수정은 cache **re-keying**으로 복구 가능함(bit-exact 확인 후 symlink + meta 갱신, 12분 절약)
4. GPU는 **3판 병렬**이 검증된 운용임

---

## 9. 최근 진행 (10/5~10/6)

### 9.1 counterfactual 해석을 정정했음

1. 09-29에 학습된 model에서 term을 제거하는 counterfactual로 "penalty가 trajectory와 거의 무관한 변수를 누른다"고 보고했음 (Δθ 제거 시 coordinate violation 0.735% → 0.733%, **−0.3%**)
2. **학습 A/B로는 반대임** — action penalty를 빼고 학습하면 violation이 0.76% → **8.47%**로 증가함
3. counterfactual은 **고정된 해 주변의 민감도**를 재는 것이고, penalty는 route 선택·속도 profile을 포함한 **해 전체를 바꾸는 방식으로** 작동함
4. 유지되는 결론은 **measurement 측면**임 — model output 기준 지표는 penalty 설정 비교에 사용할 수 없음(1.2)

### 9.2 route centerline 거칠기를 측정했음

병렬 session이 "lane tangent angle k가 평활되지 않은 계단 함수(vertex당 |Δk| p99 14.43°)라 그 거칠기가 예측 궤적에 전파된다"는 가설을 제기해 직접 측정했음. ground truth 속도로 centerline을 정확히 따라갔을 때(θ=0)의 |Δψ|를, model이 실제로 보는 보간 방식 그대로 계산했음.

| | p50 | p90 | p99 | 7.3° 초과 |
| --- | --- | --- | --- | --- |
| winner route 따라가기 | 0.00° | 0.59° | **4.79°** | **0.26%** |
| ground truth 궤적 | 0.19° | 1.60° | **4.51°** | **0.35%** |
| candidate route 전체 (평활 전) | 0.00° | 0.98° | 6.80° | 0.81% |
| candidate route 전체 (σ 1 m) | 0.00° | 1.23° | 5.43° | 0.47% |

1. **winner route는 이미 ground truth만큼 매끄러움**(p99 4.79 vs 4.51) — vertex당 14.43°와 안 맞는 이유는 integrator가 tangent **벡터**를 선형보간해 꺾임이 1.53 m 구간에 분산되기 때문임. model은 polyline의 날선 꺾임을 직접 보지 않음
2. 단 **candidate route 전체로 넓히면 ground truth의 2.4배**이고 smoothing이 그 격차의 42%를 제거함. penalty와 off-lane hinge는 winner가 아니라 **모든 live mode**에 걸리므로 이쪽이 유효한 근거임
3. 이 측정으로 원래 가설의 근거는 약화됐고, 병렬 session이 "전처리 θ가 계단을 물려받는다"는 주장을 **철회**했음

### 9.3 alive / top-1 violation 비를 판정 기준으로 도입했음

| condition | alive % | top-1 % | 비 |
| --- | --- | --- | --- |
| no penalty | 12.50 | 8.47 | 1.48 |
| coordinate penalty 1.0 | 4.09 | 2.28 | 1.80 |
| + lane-yaw 3.0 | 2.19 | 1.28 | 1.71 |
| action penalty 1.0 | 1.72 | 0.76 | 2.27 |
| lateral mode axis | 2.98 | 0.78 | **3.80** |

1. 비가 조건마다 1.48~3.80으로 크게 다름 — **penalty가 셀수록 커짐**. 거친 candidate route만으로는 설명되지 않고 **버려진 mode가 distance loss의 supervision을 받지 못하는 몫**이 섞여 있음
2. lateral mode axis가 3.80인 것이 극단 사례임 — auxiliary loss가 비승자 mode를 횡으로 밀어내 거기서만 violation이 증가했고 top-1은 불변임
3. 따라서 alive violation 단독으로는 판정할 수 없고 **비의 paired 차이**를 함께 봐야 함

### 9.4 사전 등록(pre-registration)을 도입했음

1. route centerline smoothing 실험의 판정 기준을 **학습 시작 전에 파일로 고정**했음 — 결과를 보고 기준을 고르는 것을 막기 위함임
2. 1차 기준안이 **seed 잡음보다 작아** 사용 불가였음(top-1 violation의 baseline seed range가 1.76 %p인데 기준은 ±0.1 %p였음). paired 차이의 부호 일치로 교체했음
3. 가능한 모든 결과에 대응하는 판정표로 작성했음 — "top-1 부호 갈림"을 가설 확인의 조건으로 쓰면 **null을 증거로 사용하는 것**이 되므로 별도 분기를 둠

### 9.5 병렬 session 운영

1. 2개 session이 같은 저장소에서 병렬로 작업 중임 — 본 session(penalty·lane-yaw·평가 도구)과 보조 session(route_hist·centerline smoothing)
2. **cache key 충돌로 batch 2건이 전멸했음** — 보조 session의 `src/` 수정이 원인임. 1건은 bit-exact 확인 후 re-keying으로 12분에 복구했고, 1건은 `train_v4`와 `prepare_v4`의 dataset kwargs가 어긋난 버그였음(값이 기본값이어도 **키가 한쪽에만 있으면 JSON 해시가 달라짐**)
3. 이후 규약을 합의했음 — `src/` 수정 사전 통보, 수정을 한 commit에 묶기, 직후 기본 cache 재생성, GPU 3판 병렬 유지, 문서 절 단위 분담, 커밋 시 타 session 변경 제외
4. 상호 검증이 작동했음 — 본 session의 거칠기 측정이 보조 session의 가설을 약화시켰고, 보조 session의 전체-candidate 측정이 본 session의 winner-only 측정의 범위 한계를 메웠음. 두 측정이 winner 기준에서 서로 다른 코드로 같은 값을 냈음(교차검증)
