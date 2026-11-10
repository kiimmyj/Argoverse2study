# 최신 연구 환경 변화에 대한 조사 — 자율주행 데이터셋 · 최신 SOTA 모델

> 연차보고서 PART 1 「인지 증강화 기반 동적 객체 이동 궤적 예측 모델 개발」 중 **김영진 담당(1~2쪽)** · 초안 2026-11-10
> 작년 보고서와 같은 개조식("~함 / ~임"), 소제목마다 그림·표 하나. 수치 출처는 맨 아래 「작성 메모」에 있음

---

## 1. 자율주행 데이터셋 조사

### 1.1 데이터셋의 종류와 수집 방식

- 궤적 예측 연구는 공개 데이터셋과 그 벤치마크를 중심으로 발전해 왔음
- 수집 방식에 따라 두 갈래로 나뉨
  - **차량 센서 기반** — 자율주행 차량의 라이다·카메라·레이더로 주변 객체를 검출·추적한 결과를 궤적으로 공개함. 도심 주행 로그 중 교차로·회전·차로변경처럼 예측이 어려운 구간을 골라(채굴) 구성하고, 대부분 HD 벡터 지도를 함께 제공함 (Argoverse 1·2, WOMD, nuScenes, Lyft Level 5)
  - **부감 촬영 기반** — 고층건물 카메라나 드론으로 도로를 위에서 촬영해 차량을 추적함. 가림이 없어 위치가 정밀하나 고속도로 구간에 한정되고 교차로·HD 지도가 없음 (NGSIM, highD)
- 차량 센서 기반 데이터도 추적기 출력이므로 잡음이 있음. 본 과제 실측에서 AV2 의 heading 은 진행 방향이 아니라 차체 방향이었고, 11초 창 양 끝 0.5초의 위치 차분 속력이 실제의 약 절반으로 나타남

![](figures/annual_report/fig1_datasets.png)

> 그림 1. 주요 궤적 예측 데이터셋의 공개 규모(총 주행 시간)와 관측→예측 구간 — Argoverse 1 은 전체 로그 1,006 h 중 공개된 320 h, nuScenes 는 주석 장면 기준

### 1.2 데이터 특성과 활용 과제

| 데이터셋 (공개) | 지역 | 규모 | 관측→예측 | 지도 | 주 과제 · 공식 지표 |
| --- | --- | --- | --- | --- | --- |
| Argoverse 1 (2019, CVPR) | 피츠버그·마이애미 | 5초 시퀀스 324,557개 | 2 s→3 s, 10 Hz | 차선 중심선 벡터 HD 맵 최초 공개 | 단일 에이전트 예측 · minADE/minFDE/MR/DAC |
| Argoverse 2 (2021, NeurIPS) | 미국 6개 도시 | 11초 시나리오 250,000개 (763 h) | 5 s→6 s, 10 Hz | 시나리오별 3D 벡터 HD 맵(차로 경계·그래프·횡단보도) | 단일·다중 에이전트 예측 · minADE6/minFDE6/MR6/brier-minFDE6 |
| WOMD (2021, ICCV) | 미국 6개 도시 | 20초 세그먼트 103,354개 (574 h) | 1 s→8 s, 10 Hz | 3D 벡터 맵 + 신호등 상태 | 개별·상호작용(결합) 예측, Sim Agents · Soft mAP/mAP/minADE/MR |
| nuScenes (2020, CVPR) | 보스턴·싱가포르 | 20초 장면 1,000개 | 2 s→6 s, 2 Hz | 의미 지도 11층 | 인지(검출·추적) 중심 + 예측 챌린지 · minADE5·10/MR/OffRoad |
| Lyft Level 5 (2020, CoRL) | 팔로알토 단일 노선 | 25초 장면 약 17만 (1,118 h) | 5 s 예측(기준선), 10 Hz | HD 시맨틱 맵 + 항공사진 | 대규모 학습 · 데이터 규모 효과 분석 |
| highD (2018, ITSC) | 독일 고속도로 6개 지점 | 차량 110,000대 (16.5 h) | 공식 예측 과제 없음 | 차로 주석 | 연속류 예측·안전성 검증 |
| NGSIM (2006) | 미국 고속도로 | 차량 9,206대 (1.5 h) | 공식 예측 과제 없음 | 없음 | 연속류 예측 (본 과제 3차년도 사용) |

- **예측 구간이 길어지고 규모가 커짐** — AV1 3초 → AV2 6초 → WOMD 8초, 2개 도시 → 6개 도시. 3초로 학습한 모델을 6초로 평가하면 minFDE 가 2.91 → 6.82 m 로 커져(AV2 원논문) 긴 구간 전용 설계가 필요해짐
- **지도는 벡터 HD 맵으로 표준화됨** — AV1 이 차선 중심선 벡터 맵을 처음 공개했고, AV2·WOMD 는 차로 경계·연결 그래프·횡단보도·신호등 상태까지 제공함
- **과제가 단일 → 다중 에이전트·생성형으로 확장됨** — WOMD 상호작용(결합) 예측, AV2 다중 에이전트 예측, Sim Agents(시뮬레이션 에이전트 생성). Waymo 2025 챌린지는 단일 에이전트 예측 트랙 없이 E2E 주행·시나리오 생성·Sim Agents·상호작용 예측으로 운영됨
- **지표가 확률 품질 쪽으로 이동함** — min-of-K 거리 지표는 K 를 늘리면 계속 좋아지는 한계가 있어, WOMD 는 mAP(Soft mAP), AV2 는 brier-minFDE 로 순위를 매김
- **AV2 활용 사례** — 리더보드 상위 모델(QCNet, DeMo, FutureNet-LoF), 경량 기준선(SIMPL), 지도를 쓰지 않는 공개 구현(MFTF: CRAT-Pred 기반 Transformer)까지 폭넓게 쓰임. 본 과제도 AV2 를 주 데이터로 사용함

---

## 2. 최신 SOTA 모델 조사

### 2.1 모델 계보

- **벡터 표현의 등장 (2020)** — VectorNet 이 지도와 궤적을 폴리라인 벡터 집합으로 표현했고, LaneGCN 이 차선 그래프 위 합성곱으로 지도와 에이전트를 융합함 → 래스터 이미지 방식을 대체함
- **Transformer 와 좌표 불변 표현 (2022)** — HiVT 가 에이전트마다 국소 좌표계를 둬 회전·이동 불변을 얻었고, MTR 이 학습된 의도점 64개를 쿼리로 쓰는 구조로 WOMD 에서 1위를 차지함
- **Query-centric·자기지도 사전학습 (2023)** — QCNet 이 장면 요소마다 국소 좌표계를 두고 인코딩을 재사용해(인코더 지연 82 → 13 ms) AV2 1위에 올랐고, Forecast-MAE 가 마스킹 자기지도 사전학습을 도입함
- **언어모델·생성형 (2023~2024)** — MotionLM 이 궤적을 이산 토큰으로 바꿔 다중 에이전트 미래를 언어모델처럼 생성하고, SMART 가 지도까지 토큰화해 Sim Agents 2024 에서 realism 0.7614 로 1위를 함
- **경량화·분해·상태공간 모델 (2024~2025)** — SIMPL(1.9M 파라미터 경량 기준선), DeMo(방향 의도와 시간 상태를 분리 + Mamba), Trajectory Mamba(attention 을 선택적 SSM 으로 대체), FutureNet-LoF(예측 종점에서 맥락을 다시 인코딩)

![](figures/annual_report/fig2_av2_sota.png)

> 그림 2. AV2 test 단일 에이전트 minADE6(K=6, 5 s→6 s) — 논문 보고치. 같은 해 안의 가로 위치는 겹치지 않게 벌린 것임. 점선은 2026-11-10 리더보드 1위의 값

| 모델 | 발표 | 핵심 아이디어 | minADE6 | minFDE6 | 비고 |
| --- | --- | --- | --- | --- | --- |
| QCNet | CVPR 2023 | query-centric 인코딩 + 재귀 제안·정제 | 0.65 | 1.29 | 앙상블 0.62 / 1.19 |
| ProphNet | CVPR 2023 | 앵커 기반 제안, gMLP 로 입력 통합 | 0.68 | 1.33 | 코드 미공개 |
| Forecast-MAE | ICCV 2023 | 마스킹 자기지도 사전학습 | 0.709 | 1.392 | 앙상블 0.690 / 1.338 |
| SIMPL | RA-L 2024 | 상대 자세 + 대칭 Transformer, 1.9M | 0.72 | 1.43 | 경량·공개 코드 |
| DeMo | NeurIPS 2024 | 방향 의도·시간 상태 분리 + Mamba | 0.61 | 1.17 | 앙상블 0.60 / 1.11 |
| Trajectory Mamba | CVPR 2025 | attention → 선택적 SSM, 4.54M | 0.64 | 1.24 | 앙상블 여부 미기재 |
| FutureNet-LoF | ICRA 2025 | 미래 맥락 재인코딩 + 차로 점유장 | 0.58 | 1.07 | 앙상블 |
| 리더보드 1위 IMR | 2025-09 제출 | 논문 미확인 | 0.5714 | 1.0684 | brier-minFDE6 1.6248 |

- WOMD(1 s→8 s, K=6) 대표 기록 — MTR mAP 0.4129, MTR++ mAP 0.4329, EDA Soft mAP 0.4510(이상 test), 2024 Motion Prediction 챌린지 1위 MTR v3 Soft mAP 0.4967(앙상블), 2025 상호작용 예측 1위 Parallel ModeSeq Soft mAP 0.2978

### 2.2 연구 환경 변화와 시사점

- **성능 포화와 앙상블 의존** — AV2 test minADE6 상위권이 0.57~0.65 m 에 모였고 상위 기록 다수가 앙상블임 → 단일 모델을 같은 조건에서 비교해야 차이가 드러남
- **정확도 외 평가의 부재** — 상위 모델 대부분이 도로 이탈·역주행 같은 실현가능성 지표를 보고하지 않음. Sim Agents 평가에서는 minADE 1위 모델(QCNeXt, 1.083)이 도로 이탈 우도는 최하위(0.360)였음 → 지도 규칙과 실현가능성을 함께 보는 평가가 필요함
- **재현 장벽** — 일부 상위 모델은 코드가 공개되지 않았고(ProphNet), 일부는 다수 GPU 학습을 전제로 함(QCNet) → 공개 코드·단일 GPU 로 재현 가능한 경량 모델의 가치가 커짐
- **과제의 확장** — 단일 에이전트 예측에서 다중 에이전트 결합 예측, 시뮬레이션 에이전트, E2E 주행으로 연구 범위가 넓어지는 추세임

---

## 작성 메모 (보고서 본문 아님)

**MFTF 는 논문이 아니라 GitHub 공개 저장소임.** README 제목은 "MFTF: Map Free Motion Forecasting Using Transformers", 부제 "Argoverse2 Motion Forecasting". CRAT-Pred(지도 미사용 LSTM+그래프)를 바탕으로 LSTM·self-attention 을 Transformer 인코더로 바꾼 구현이고, 결과표·저자·인용 정보가 없음. EvalAI AV2 리더보드 196개 항목에도 없음(2026-11-10 조회). 모델 담당의 "궤적 예측 모델링(CVAE, MFTF)"과 겹치므로 본문에는 'AV2 활용 사례' 한 줄만 두었음 — 표기를 맞출 것
- https://github.com/narsisn/Argoverse2_Motion_Forecasting
- 이름이 비슷한 MFTraj(IJCAI 2024)는 AV1·NGSIM·highD 평가라 "AV2 – MFTF" 와 다름

**수치 출처**

| 항목 | 출처 |
| --- | --- |
| 데이터셋 표·그림 1 | `~/paper-kb/notes/` chang2019argoverse1 · wilson2023argoverse2 · ettinger2021womd · caesar2020nuscenes · houston2020one · krajewski2018highd (NGSIM 은 highD 표 I) |
| WOMD 103,354개·6개 도시 | https://waymo.com/open/about/ (v1.3.1, 2025-10) — 원논문은 104k |
| nuScenes 2 s→6 s @2 Hz·지표 | nuscenes-devkit prediction README (예측 과제는 원논문이 아니라 후속 챌린지) |
| AV2 모델 표·그림 2 | 각 모델 노트(zhou2023qcnet 표 1 · wang2023prophnet 표 3 · cheng2023forecastmae 표 3 · zhang2024simpl 표 II · zhang2024demo 표 1 · huang2025trajectory 표 1 · wang2025futurenet 표 1) |
| 리더보드 1위 IMR | https://eval.ai/web/challenges/challenge-page/1719/leaderboard/4098 (2026-11-10 조회, brier-minFDE6 정렬) |
| WOMD 모델 기록 | shi2022mtr · shi2024mtr · lin2024eda 노트, Waymo 2024 1위 보고서(MTR v3), 2025 상호작용 1위 보고서(Parallel ModeSeq) |
| Waymo 2025 트랙 구성 | https://waymo.com/open/challenges/ |
| QCNeXt 1.083 / 0.360 | wu2024smart 노트(WOSAC 표) |
| 3 s→6 s minFDE 2.91 → 6.82 | wilson2023argoverse2 부록 표 8 |
| AV2 heading·속도 램프 | 본 과제 실측 (`docs/heading_quality.md`, `docs/v4_data_cleansing.md` §8) |

**옮길 때 주의**
- 데이터셋마다 예측 구간·K·분할이 달라(AV1 2→3 s, AV2 5→6 s, WOMD 1→8 s) **수치를 한 표에서 우열로 비교하면 안 됨**
- AV2 고유 도로 길이는 논문 2,220 km, 공식 사이트 2,110 km 로 달라 본문에서 뺐음
- 리더보드 값은 조회 시점(2026-11-10) 스냅샷임. 보고서 제출 직전에 다시 확인할 것
- 그림 원본은 `src/viz_annual_report.py` 가 그림. 수치를 고칠 때는 이 스크립트의 표를 고친 뒤 다시 그림
