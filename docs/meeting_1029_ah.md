# (a, h) 효용성 평가 — AV2 SOTA 모델 조사

> 2026-10-29. v1~v4를 잠정 보류하고, AV2 SOTA 모델에 **(a, h) 입력과 θ = h − k 표현**을 넣어
> 그 효용성 자체를 정량·정성으로 재기로 한 배경과, 그에 필요한 모델 조사를 정리했음.

---

## 1. 왜 방향을 바꾸는가

### 1.1 v1~v4 잠정 보류

1. v1~v4는 **공부하면서 만든 것**이라, 버전 간 구조 변경과 용어가 정리되지 않은 채 쌓였음
2. 구체적으로 — 클래스명이 v1~v3 모두 `LSTMSeq2Seq`인데 실제 seq2seq는 v1뿐임. v4의 L0~L4는 레벨을 올리면 성능이 **나빠지는 것이 설계상 정상**인 구조라 "개선/퇴행"의 의미가 버전마다 다름. 같은 "heading"이 절대각·잔차각·좌표 복원각으로 서로 다르게 쓰임
3. 그 결과 **어떤 요인이 성능에 작용하는지 분리가 안 됨** — 아래 1.2가 그 증거임
4. 따라서 v4 계열은 **잠정 보류**하고, 통제된 비교가 가능한 공개 SOTA 모델 위에서 질문 하나만 떼어 보기로 했음

### 1.2 요인이 섞여 있다는 증거

| 변경 | paired Δ minADE6 | 판정 |
| --- | --- | --- |
| action → coordinate penalty | 0.000 (cost 없이 violation 0.29×) | 채택 |
| data cleansing | −0.043 | 3 seeds 부호 일치 |
| lane heading alignment 3.0 | +0.017 전체 / **−0.069 lane change** | lane change에서만 |
| route history | −0.04 (수렴한 seed) | 불안정 (3판 중 1판 발산) |
| neighborhood vehicles 30 m | −0.028 | 검출 불가 |
| smoothing (ah3) | −0.053 | 검출 불가 |
| **downsampling 2 Hz** | **+0.072** | 3 seeds 모두 **나쁨** |

1. 개별 효과가 **0.005~0.08 규모**이고, 부호가 갈리는 것이 많아 판정 자체가 어려웠음
2. **근거가 확인된 넷을 전부 합쳐 100 epoch × 5 seeds를 돌렸더니 1.4038로, 기준(coordinate penalty만 30 epoch) 1.3591보다 +0.045 나빴음.** 5 seeds 전부 기준보다 나빴으므로 seed 잡음이 아님
3. 즉 **합치면 더해지지 않음.** 어떤 요인이 작용하는지 모르는 상태임
4. 구조적 상한도 측정됐음 — 후보 경로 집합의 **오라클 경로조차 ground truth에서 평균 1.079 m** 떨어져 있어(top-1 경로는 1.523 m) 현재 minADE6와 같은 자릿수임

### 1.3 그래서 질문을 하나로 좁혔음

1. **"(a, h) 2채널 표현이 그 자체로 쓸 만한가"** 하나만 남기고, 나머지는 검증된 SOTA 아키텍처에 맡김
2. preprocessing(smoothing + downsampling)과 data cleansing은 v4에서 이미 다뤘으므로, 남은 미지는 **표현(representation)** 임
3. θ = h − k(차로 접선각 기준 상대 heading)는 v4 방식 그대로 유지함
4. 평가는 **정량(minADE6 등) + 정성(지도 위 궤적과 (a,h) 진단 패널)** 을 같이 함

---

## 2. AV2 SOTA 모델 조사

### 2.1 비교표

| | **QCNet** | **SIMPL** | **Forecast-MAE** | **DeMo** | **HPTR** | **ProphNet** | **GANet** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Venue | CVPR'23 | ICRA'24 | ICCV'23 | NeurIPS'24 | NeurIPS'23 | CVPR'23 HL | ICRA'23 |
| **AV2 test minADE6** | **0.65** | 0.72 | 0.709 | **0.61** | 0.73 | 0.68 | 0.728 |
| Params | 7.3 M | 1.9 M (공개 ckpt 2.65 M) | 1.92 M | 5.9 M | 15.2 M | 미보고 | 미보고 |
| **입력 채널** | **4**: ‖Δp‖, ∠(Δp,h), ‖v‖, ∠(v,h) | **14**: Δp, **cos/sin h**, v, type, valid | **4**: Δp, Δ‖v‖, valid — **per-step h 없음** | **4**: 같음 — **per-step h 없음** | **68**: pose PE, v, spd, yaw rate, **acc**, size, type, step-OHE | **6**: x, y, vx, vy, sinθ, cosθ | **3** (lineage): Δp, valid |
| **공개 코드** | Apache-2.0 + ckpt | MIT + AV2 ckpt | **license 없음** | repo 이전, Mamba 빌드 필요 | **CC BY-NC** (비상업) | **없음** | **빈 저장소** |

1. **ProphNet·GANet은 코드가 공개된 적이 없음** — GANet 저장소는 1바이트 placeholder 4개가 전부이고 "Code will be released soon"이 3년째임. ProphNet은 저장소 자체가 존재하지 않음
2. QCNet은 학습에 **GPU 메모리 약 160 GB**가 필요해(8×3090 권장) 단일 24 GB로는 불가능함
3. 실제로 fork 가능한 것은 **SIMPL · Forecast-MAE · DeMo · HPTR** 넷임

### 2.2 핵심 축 — invariance 방식이 모든 것을 가름

조사에서 가장 중요한 발견임. 7개 모델이 **invariance를 어떻게 얻는가**로 두 그룹으로 갈림.

**Group A — 구조적 불변 (QCNet · SIMPL · HPTR)**
1. 절대량이 MLP에 **한 번도 들어가지 않음.** 기하는 오직 쌍별 상대 descriptor로만 들어감
2. **raw 절대 heading을 넣으면 불변성이 깨짐** — 셋 다 내부에서 이미 `h_t − h_now`를 계산하므로 **그것을** 넣어야 함
3. **θ = h − k가 이미 attention의 입력으로 들어가 있음** → θ는 대체로 중복임

**Group B — 정규화에 의한 불변 (Forecast-MAE · DeMo · ProphNet · GANet)**
1. 장면 전체를 focal agent 기준으로 **한 번 회전**시키는 방식임. 구조가 아니라 전처리로 불변성을 얻음
2. 상대 **변위**는 encode하지만 agent와 차로 사이의 상대 **각도**는 encode하지 않음
3. **θ = h − k가 진짜 새 정보임** — 효과가 있다면 여기임

### 2.3 그래서 두 가설이 서로 반대 방향임

1. **(a,h) 교체가 가장 쉬운 모델이 θ가 가장 덜 보태는 모델임.** 이 역상관이 실험 설계를 좌우함
2. 따라서 역할을 나눠야 함 — **QCNet·HPTR·SIMPL은 θ 가설의 대조군**, **Forecast-MAE·DeMo·GANet은 처치군**임
3. 한 모델에서만 재면 "θ가 안 듣는다"와 "그 구조에선 이미 있다"를 구별할 수 없음

---

## 3. 모델별 — (a, h) + θ = h − k 를 넣으면

### 3.1 QCNet

1. 입력 4채널이 **이미 전부 불변량**임 (크기 2개 + heading 기준 상대각 2개)
2. 교체는 **2줄**로 끝남(`agent_encoder` line 62, 115–119). 단 `pos_a`·`head_a`는 4개의 상대 descriptor 블록과 radius graph가 쓰므로 **남겨야 함**
3. **θ가 가장 강하게 중복됨** — `rel_orient_pl2a = orient_pl − head_s`가 곧 −θ이고, 지도를 50 step 전체에 복제해 **timestep마다** 계산해서 attention의 key/value에 넣고 있음. θ(t) 전 구간이 이미 있음
4. 출력은 이미 per-step 상대 heading을 von Mises로 예측하고 proposal이 recurrent라 (a, Δθ)와 궁합이 좋음. 비용은 per-step 독립 Laplace NLL이 적분 위치의 상관을 잘못 모델링한다는 점임
5. **단일 GPU 학습 불가**(160 GB)

### 3.2 SIMPL

1. **교체가 가장 쉬움** — heading이 이미 channel 2:4에 `(cos, sin)`으로, 그것도 per-agent 상대 프레임으로 들어 있음. 재전처리 불필요
2. θ는 **현재 step에서 중복됨** — RPE에 `cos(h−k), sin(h−k)`가 들어 있고 매 SFT layer마다 재encode됨. 논문 Table III에서 이 RPE 갱신이 가장 큰 ablation 이득이었음. 다만 **per-timestep θ(t)는 새로움**
3. 출력이 **Bezier 제어점 + 해석적 속도(hodograph)** 라 "매끄러운 곡선을 내고 운동학을 유도한다"는 설계가 이미 같음. (a, Δθ) 헤드는 이식이 아니라 형제 재파라미터화임
4. **유용한 선례** — 논문 Table IV에서 yaw loss 추가가 minADE6 0.780 → 0.783(소폭 악화), minAYE6 0.137 → 0.055. **heading 감독은 거의 공짜지만 공짜는 아님**
5. 발견 — `av2_dataset.py` line 258 `actors[..., 2:]`(주석 `# ! tmp solution`)가 **50 step 중 앞 2개를 조용히 버리고 48개를 씀**. 문서화도 ablation도 없음

### 3.3 Forecast-MAE

1. **θ 가설의 가장 깨끗한 처치군임** — agent heading과 차로 각도가 같은 PE MLP를 거쳐 **절대값으로 더해지고**, 그 뒤는 relative bias 없는 평범한 MultiheadAttention임. 차이를 **학습으로** 만들어야 함
2. 저자들도 Limitations에서 인정함 — *"relative position design [HiVT, GoReLa, QCNet]을 제대로 넣으면 더 좋아질 수 있다"*
3. **per-step heading이 아예 없음**(t=49에서 한 번만 들어감). 그래서 (a,h)는 순수한 축소가 아니라 **per-step heading을 더하면서 Δ위치를 빼는** 교환임
4. k는 바로 못 씀 — 차로 각도가 20점 폴리라인당 **현 하나**라 곡선을 따라가기엔 거침. 최근접 차로 조회가 t=49에만 있고 `.indices`를 버려 차로 identity가 날아감. 전처리 재구축 필요
5. **from scratch가 공개 baseline**(AV2 test 0.727)이라 pretraining 없이 scratch 대 scratch 비교가 가능함
6. license가 **없음**

### 3.4 DeMo

1. 교체 난이도는 SIMPL과 공동 1위이고 **재pretraining도 없음**. `x_angles`(50 step 전체, focal 상대)가 이미 batch에 들어와 있는데 per-timestep으로는 안 쓰임
2. θ는 **부분 중복** — focal agent의 현재 step에서는 `lane_angles`가 **정확히 −θ**임. 새로운 것은 θ(t), 비focal agent, 차로 **선택**임
3. 출력 head가 이미 **(mode, timestep)별**이라 (a, Δθ)가 자연스러움. 다만 **진짜 Frenet은 어려움** — lane graph가 아예 없음
4. 흔적 — `target_diff`, `target_vel_diff`(미래 위치차·Δ‖v‖)를 계산해 collate까지 해놓고 **어떤 loss에도 안 씀.** 저자들이 운동학 감독을 시도했다가 뺀 것으로 보임
5. Mamba 빌드가 sm_120(우리 GPU)에서 되는지 **미검증**

### 3.5 HPTR

1. **`a`와 `h`가 이미 그대로 있음** — `agent/acc`(m/s²), `agent/yaw_bbox`
2. θ가 **attention bias 그 자체임** — `rel_pose[...,2] = k − h`를 128차원 Fourier로 encode해 모든 agent→lane edge의 K/V에 더함
3. agent history 50 step을 PointNet이 **하나의 token으로 max-pool** 함. AV2 설정에서 temporal transformer가 아예 없음 → **시간축에서 뺄 것이 없음**
4. license가 **CC BY-NC**(비상업)이고 4×2080Ti로 5일 이상 걸림

### 3.6 ProphNet · GANet (코드 없음, 논문만)

1. **ProphNet** — heading이 native 채널이고 프레임도 맞아 표현상 가장 친화적임. 그런데 **압축된 history(t=0만)가 scene fusion에 들어가고** 전체 history는 map-free proposal 분기로만 감 → **θ(t)를 만들 기전이 없음**. per-timestep θ는 새 정보임
2. **GANet** — **θ가 전혀 중복이 아님.** 상대 정보가 `Linear(2, ·)`에 들어가는 **위치 차이뿐**이고 각도가 없음. agent heading 자체가 입력에 없음
3. 한 줄 요약 — *ProphNet은 θ의 두 조각을 다 갖고 있으나 timestep마다 짝지은 적이 없고, GANet은 첫 조각조차 없음*
4. GANet의 정규화 축이 **마지막 step 변위 방향**이라 정지 차량에서 `arctan2(0,0) = 0`으로 프레임 자체가 임의가 됨. v4에서 확인한 "정지 차량 h는 믿지 말 것"이 여기서 가장 심하게 작용함
5. 둘 다 **전체 재구현**이 필요함

---

## 4. 설계 교정 — (a, h) 2채널만으로는 부족함

조사에서 나온 두 번째 중요 발견임.

1. **7개 모델 전부가 속도를 입력에서 얻고 있음** — per-step 변위에서든 명시적 velocity 채널에서든
2. `a`를 적분하면 Δv만 복원되고 **v(t₀)가 관측 불가**임. 그런데 7개 전부 6초 지평의 **위치 공간에서 지도학습**함
3. → **(a, h, v₀) 또는 (a, h, |v|)** 로 가야 함
4. **validity 채널도 반드시 필요함** — Forecast-MAE·DeMo·SIMPL은 결측 step을 **literal 0.0**으로 채움. 2채널만 쓰면 결측 step이 "직진 등속"과 구별되지 않아, **(a,h) 효용성 실험이 데이터 품질 퇴행과 섞임**
5. 추가 비용은 파라미터 수백 개 수준임
6. 또 하나 — **AV2는 position·heading·velocity·category를 주지만 acceleration은 주지 않음.** `a`는 유도해야 하고, v4에서 확인한 **11초 창 양 끝 0.5초 속도 램프**가 경계에서 `a`를 오염시킴

---

## 5. 5 Hz 다운샘플링이 과적합을 막아 주는가

### 5.1 결론부터 — 직접 근거가 없음

1. **agent history의 *샘플링 레이트*를 고정 길이에서 바꾼 통제된 ablation은 공개 문헌에 존재하지 않음.** 4번의 독립 검색이 같은 결론에 도달했음
2. history **길이** ablation은 많고 결과도 뚜렷함. **레이트**는 비어 있음
3. 이 공백 자체가 기여가 될 수 있음

### 5.2 길이 ablation은 "뒤쪽 history가 거의 무가치"라고 말함

"Recover to Predict"(arXiv:2603.10597), **AV2 val, QCNet·DeMo**, minADE6:

| 관측 step | QCNet-Ori | QCNet-IT | DeMo-Ori | DeMo-IT |
| --- | --- | --- | --- | --- |
| 10 (1.0 s) | 0.900 | **0.741** | 0.861 | **0.675** |
| 20 (2.0 s) | 0.777 | 0.734 | 0.700 | 0.661 |
| 30 (3.0 s) | 0.752 | 0.730 | 0.671 | 0.660 |
| 40 (4.0 s) | **0.725** | 0.726 | 0.662 | 0.659 |
| 50 (5.0 s) | 0.726 | 0.726 | 0.658 | 0.658 |

1. **4초 ≈ 5초임**(QCNet 0.725 vs 0.726). AV2 history의 마지막 1초는 값이 없음
2. 짧은 관측으로 **학습한** 모델은 1초만으로 전체 history의 **97~98%** 에 도달함
3. ⚠️ `-Ori`/`-IT`의 정의는 확인하지 못했음. 표가 full length에서 정확히 일치하는 것으로 보아 `-Ori`는 전체로 학습 후 잘라서 평가한 robustness 곡선으로 보이나 **추정임**

### 5.3 실무자들이 config로 투표한 흔적 둘 (코드에서 직접 확인함)

1. **QCNet의 공식 AV2 명령이 `--num_t2m_steps 30`** 이고 `qcnet_decoder.py` line 154가 그 앞을 전부 마스킹함 → **mode query가 5초 중 앞 2초를 아예 보지 않음**
2. **SIMPL이 50 step 중 앞 2개를 버림**(`actors[..., 2:]`, `# ! tmp solution`)
3. 둘 다 ablation은 아니지만, 저자들이 앞쪽 history를 불필요하다고 판단한 증거임

### 5.4 레이트에 대해 유일하게 가까운 연구

1. "Temporal Sampling Frequency Matters"(arXiv:2605.10388, 2026-05) — **단 이것은 카메라 프레임을 subsample하는 end-to-end driving 연구이지 vectorized agent state가 아님.** 유추이지 직접 근거가 아님
2. 결과는 시사적임 — *"작은 E2E 모델은 비단조 또는 평탄한 경향을 보이고 중간 주파수에서 최고 성능"*, 반면 큰 모델(AutoVLA)은 **가장 높은 주파수에서 최고**였음
3. 제시한 기전이 **과적합이 아니라 capacity**임 — *"조밀한 샘플링이 중복 콘텐츠와 off-manifold 잡음을 더해 주행과 무관한 capacity 부담을 만든다"*
4. 이 틀을 우리 대상에 적용하면 — 효과는 **Forecast-MAE 1.9 M · SIMPL 2.6 M · DeMo 5.9 M에서 보이고, HPTR 15.2 M에서 가장 안 보여야 함**
5. 저자 결론 — *"temporal sampling frequency는 가용 최고값으로 고정할 것이 아니라 보고하고 튜닝해야 한다"*

### 5.5 반대 증거도 같이 적음

1. **SHARP**(AV2 val): 5초 context 0.64 vs **1초 context 0.76(+18.8%)**, *"입력 context가 길수록 성능이 좋아진다"*
2. **POP**: *"예측기 성능은 관측 길이와 직접 상관"*, 특히 긴 지평에서. AV2의 6초는 긴 지평임
3. **FlexiLength**(CVPR'24): 학습과 다른 관측 길이로 평가하면 *"상당한 성능 저하"*
4. **우리 자신의 실측** — v4에서 10 Hz → 2 Hz는 **3 seeds 모두 나빴음**(+0.072)

### 5.6 종합

1. 질문 "5 Hz가 과적합을 막아 주는가"에 **문헌이 답을 갖고 있지 않음.** 7개 모델 중 과적합을 논한 것은 Forecast-MAE 하나뿐이고(Table 4: 60 epoch에서 포화), 그마저 레이트와 무관함
2. **history timestep을 정규화 수단으로 떨어뜨리는 연구는 찾지 못했음.** HPTR에 `dropout_p_history` 손잡이가 있으나 HPTR 자신은 꺼 두었고 ablation한 적이 없음
3. 따라서 이번 실험에서는 **각 SOTA 모델의 기본 레이트를 따르고**, 레이트는 별도 ablation 축으로 둠
4. 만약 재게 된다면 **capacity 축과 묶어서** 재는 것이 문헌이 가리키는 방향임(작은 모델에서만 이득이 보일 것)
5. ⚠️ 리스크 — AV2 논문 자신이 *"등속 모델은 특히 성능이 나쁘다. 데이터셋이 운동학적으로 흥미로운 궤적 쪽으로 편향돼 있기 때문"* 이라고 경고함. 2채널 입력에는 위험이지만, 동시에 **(a, h) 파라미터화의 가장 좋은 근거**이기도 함 — 벤치마크가 바로 그 가속·회전 신호를 겨냥해 수집됐음

---

## 6. 적합성 평가 — 어느 모델로 갈 것인가

| 모델 | (a,h) 교체 | θ 신규성 | 단일 GPU | license | 역할 |
| --- | --- | --- | --- | --- | --- |
| **SIMPL** | **최易** | 낮음 (현재 step 중복) | ✅ 45 h/50 ep | **MIT** | **(a,h) 가설의 주 무대** |
| **Forecast-MAE** | 쉬움 | **높음** | ✅ | 없음 | **θ 가설의 처치군** |
| DeMo | 최易 | 중간 | ✅ (Mamba 빌드 미검증) | 없음 | 정확도 최고(0.61) |
| QCNet | 쉬움 | **가장 낮음(완전 중복)** | ❌ 160 GB | Apache-2.0 | θ 대조군 (참고만) |
| HPTR | 쉬움 | 낮음 | ❌ 5일+ | CC BY-NC | 제외 |
| ProphNet / GANet | 재구현 | 높음 | — | 코드 없음 | 제외 |

1. **1차는 SIMPL** — 컴파일 확장이 없어 sm_120에서 그대로 돌고, 셋 중 유일하게 license가 있으며, heading이 이미 per-step으로 들어 있어 교체가 2줄임. decoder가 출력 파라미터화 스위치를 갖고 있어 (a, Δθ) 헤드가 네 번째 분기로 들어감
2. **θ 가설까지 보려면 Forecast-MAE가 필요함** — SIMPL에서 θ가 안 들어도 "효과 없음"이 아니라 "이미 있음"일 수 있기 때문임
3. 단일 GPU 실측 — SIMPL 1 epoch **54.6분**(train 51.9 + val 2.7), 50 epoch **약 45시간**. 1 epoch 만에 val minADE6 **1.354**로 v4의 완주 기준(1.3591) 수준에 도달했음
4. ⚠️ SIMPL 재현 정확도는 셋 중 가장 낮음 — 저자가 논문 체크포인트를 분실해 재학습한 공개 ckpt가 val **0.777**(논문 0.72)임

---

## 7. 실험 계획과 비용

### 7.1 단계

| 단계 | 내용 | 비용 | 여기서 정해지는 것 |
| --- | --- | --- | --- |
| 0 | 배포 체크포인트 평가 | **3분** | 파이프라인 검증 + 기준점 |
| 1 | **R0** 원본 입력 학습 (seed 1) | 45 h | 우리 학습 루프가 0.78 근처를 내는가 |
| 2 | **R1** (a, h, v₀) 입력, 같은 seed | 45 h | **R1 − R0 이 핵심 수치** |
| 3 | 차이가 크면 종료, 작으면 **그때 seed 추가** | 조건부 | |
| 4 | **R2** 출력까지 (a, Δθ) + 적분기 | 45 h | 출력 교체의 대가 |
| 5 | **R3** Forecast-MAE에 θ 추가 | 45 h | θ 가설 (처치군) |

1. 2조건 × 3 seeds를 미리 돌리면 **270시간**이므로, 단계적으로 가서 애매할 때만 seed를 붙임
2. 1 epoch에 54.6분인데 **GPU 사용률 중앙 88%**, 로더 대기 **0.2%**, backward가 65%로 **연산 바운드**임. batch·workers·bf16 전부 효과 없었고 `torch.compile`은 가변 길이 입력에서 실패함 → **45시간은 바닥에 가까움**
3. 참고로 논문의 "8×3090 11시간"은 **88 GPU-시간**이므로 우리가 느린 것이 아님

### 7.2 1 seed로 효과를 가릴 수 있는가

| 차이 크기 | 판정 |
| --- | --- |
| > 0.05 | **가능** (v4에서 안정 조건의 seed 범위가 0.006~0.047이었음) |
| 0.02~0.05 | 불확실 |
| < 0.02 | **불가능** |

1. 어느 경우일지 미리 알 수 없고, **SIMPL의 seed 잡음은 측정한 적이 없음**(v4 수치를 빌려 쓰는 것임)
2. 그래서 단계적 설계로 가되, 결과를 보고할 때 **seed별 값과 "발산 판" 개수**를 항상 같이 적음

---

## 8. 정성 평가 계획

1. **시나리오 선정** — v4에서 만든 **상황 군집 8개**의 대표 사례를 씀 (C0 회전 18.1% · C1 정속 23.6% · C2 직진 가감속 18.6% · C6 급가감속 섞인 회전 7.3% 등). 회전 두 군집이 시나리오의 25.4%인데 **전체 오차의 39.9%** 를 만듦
2. **지도 그림** — 한 장에 **ground truth + 확률 1위 궤적 + 최근접 차로 중심선과 그 접선 방향**을 겹침. θ = h − k를 눈으로 읽을 수 있게 함. step마다 점을 찍음
3. **비교 배치** — **같은 시나리오를 나란히** 둠 (왼쪽 원본 입력 / 오른쪽 (a,h))
4. **(a,h) 진단 패널 4종**
   - heading vs 시간 (ground truth 대비)
   - 회전량 순/총 (ground truth 대비)
   - \|Δψ\| 분포 (중앙·p99)
   - **θ = h − k 분포**
5. 이 네 가지는 v4에서 "덜 도는가 / 몰아서 꺾는가 / 왕복이 남는가"를 가려냈던 지표와 같음

---

## 9. 다음 미팅까지

1. SIMPL 배포 체크포인트 평가로 파이프라인 검증 (3분)
2. R0 → R1 학습 (각 45시간)
3. 시각화 하네스를 SIMPL 출력에 맞춰 재작성
4. 결과를 seed별 값·발산 판 개수와 함께 보고

### 확인하지 못한 것

1. SIMPL 파라미터 수 — 논문 1.9 M vs 공개 체크포인트 실측 **2,646,481**. 원인 미상
2. Forecast-MAE의 논문 val(0.801) vs README val(0.7117) 약 10% 차이. 원인 미상
3. ProphNet의 AV2 history 길이 — 논문이 침묵함("AV1과 같은 설정"에서 2초로 추정)
4. "Recover to Predict"의 `-Ori`/`-IT` 정의 — 표에서 추정했고 원문 확인 못 함
5. GANet의 AV2 test 수치가 ensemble인지 — README의 "variant" 표현으로 보아 논문 모델과 다를 가능성
6. DeMo의 Mamba가 sm_120에서 빌드되는지
