# 데이터 전처리 — SIMPL 은 무엇을 미리 굽는가

> **왜 이 문서가 있나.** R0 · R1 · R0-s43 이 전부 같은 캐시를 읽는다. 그 캐시에 무엇이 들어 있고
> 무엇이 **안** 들어 있는지를 모르면, 런 사이의 차이를 엉뚱한 데로 돌리게 된다.
> 실제로 (a, h) 구현 때 여기서 두 번 걸렸다(§5).
>
> 모든 수치는 코드와 캐시를 직접 읽어 확인했다. 확인 경로는 §7 에 적었다.

---

## 1. 한눈에

| | |
| --- | --- |
| 캐시 경로 | `/data/argoverse2/simpl_features/` (하위 `train/`, `val/`) |
| 크기 | **30 GB** — train 27 GB · val 3.4 GB |
| 파일 수 | train **199,908** · val **24,988** — **시나리오당 `.pkl` 1개** (중앙 134 KB) |
| 굽는 시간 | val 4분 · train 40분 (CPU 전체 − 2 코어 병렬) |
| 출처 | SIMPL 저자 코드 원본 — **우리가 고친 것 없음** |
| 재생성 필요 | **없음.** R0 · R1 · R0-s43 모두 같은 캐시를 읽는다 |

**실행 명령** (`scripts/av2_preproc_all.sh`):

```bash
python data_av2/run_preprocess.py --mode val   --data_dir <AV2>/val/   --save_dir /data/argoverse2/simpl_features/
python data_av2/run_preprocess.py --mode train --data_dir <AV2>/train/ --save_dir /data/argoverse2/simpl_features/
```

`--obs_len 50` · `--pred_len 60` 이 기본값이고 우리는 기본값으로 구웠다.

---

## 2. 무엇이 구워져 있나

`.pkl` 은 **행 1개짜리 pandas DataFrame** 이고 열은 **정확히 여섯 개**다.

| 열 | 내용 |
| --- | --- |
| `SEQ_ID` · `CITY_NAME` | 시나리오 식별자, 도시 |
| `ORIG` · `ROT` | 정규화 기준 — **focal agent 의 t=49 위치와 방향** |
| `TRAJS` | agent 궤적 묶음 (아래) |
| `LANE_GRAPH` | 차로 그래프 (아래) |

### TRAJS — agent 당

| 키 | 모양 | 비고 |
| --- | --- | --- |
| `trajs_pos` | `[N, 110, 2]` | 위치 |
| `trajs_ang` | `[N, 110]` | AV2 차체각(yaw) |
| `trajs_vel` | `[N, 110, 2]` | 속도 |
| `trajs_type` | `[N, 110, 7]` | 원-핫 — vehicle · pedestrian · … |
| `has_flags` | `[N, 110]` | 유효성 |
| `trajs_ctrs` · `trajs_vecs` | `[N, 2]` | **현재 pose** — t=49 의 중심과 방향 |
| `trajs_tid` · `trajs_cat` | `[N]` | 트랙 id, 카테고리(focal/score/unscore/frag) |

> ⚠️ **캐시에는 110 step 이 통째로 들어 있다.** 흔히 `TRAJS_POS_OBS` 로 부르는 관측 구간은
> 캐시가 아니라 **로더가 자른 것**이다 — `av2_dataset.py:86-93` 에서 `[:, :50]` / `[:, 50:]` 로 나눈다.
> 즉 **정답(미래 60 step)도 같은 파일 안에 있다.**

**110 = 50 + 60 @ 10 Hz = 과거 5초 + 미래 6초.** 코드 근거는 `run_preprocess.py:32-33`
(`--obs_len 50`, `--pred_len 60`)이고, 현재 시점은 `ts_obs = obs_len − 1 = 49` 로 고정이다
(`av2_preprocess.py:170`). 전처리는 **t=49 에 관측되지 않은 트랙을 버린다** — 모든 agent 가
현재 시점에는 반드시 존재한다(`assert has_flags[:, 49] == 1`).

### LANE_GRAPH — 차로

세그먼트를 **길이 15 m 기준**으로 쪼개고(`SEG_LENGTH = 15.0`), 각 세그먼트를 **노드 10개**로
리샘플한다(`SEG_N_NODE = 10`).

| 키 | 모양 (예) | 내용 |
| --- | --- | --- |
| `node_ctrs` · `node_vecs` | `[L, 10, 2]` | 노드 중심·방향 (세그먼트 로컬 좌표) |
| `lane_ctrs` · `lane_vecs` | `[L, 2]` | 세그먼트 앵커 pose |
| `lane_type` | `[L, 10, 3]` | VEHICLE / BIKE / BUS |
| `intersect` | `[L, 10]` | 교차로 여부 |
| `cross_left` · `cross_right` | `[L, 10, 3]` | 차선 표시 종류 |
| `left` · `right` | `[L, 10]` | 좌·우 이웃 차로 존재 |

**규모 (val 60개 표본):** agent 중앙 **26**(3~53) · 차로 세그먼트 중앙 **98**(33~213) ·
노드 중앙 **985**(330~2,130).

> ⚠️ **차로에 반경 필터가 없다.** `static_map.vector_lane_segments` 를 **전부** 넣는다.
> 거리로 걸러지는 것은 agent 쪽뿐이고, 그것도 `unscore`/`frag` 카테고리가 지도에서
> **20 m**(`FAR_DIST_THRES`) 넘게 떨어졌을 때만이다.

---

## 3. 매 에폭 CPU 가 다시 하는 일

구워 뒀다고 CPU 일이 0 은 아니다.

1. **캐시 디스크 읽기** — 시나리오당 `.pkl` 1개 (OS 페이지 캐시가 일부 흡수)
2. **`--data_aug` 증강** — 무작위 회전·이동을 매번 새로 입힌다. 고정값이 아니라 캐시 불가
3. **⚠️ RPE 계산** — **캐시에 없다.** 아래 참고
4. **`collate_fn`** — 배치 묶기·패딩·텐서 적재
5. **R1 한정** — `ah_feat.py` 가 (a, h) 를 로더 단계에서 계산 (전처리 재생성 불필요)

> ### ⚠️ RPE 는 구워져 있지 않다
>
> 캐시 열 여섯 개에 RPE 가 없다(§2). `av2_dataset.py:142-146` 에서 **`__getitem__` 안**,
> 그것도 **증강 이후에** 매번 계산한다.
>
> **정의** (`_get_rpe`, `av2_dataset.py:183`): 장면의 모든 요소 — agent 와 차로 세그먼트를
> 한 줄로 이어 붙인 `N = N_agent + N_lane` — 에 대한 **N×N 쌍별 상대 인코딩**이고 **5채널**이다.
>
> | 채널 | 식 |
> | --- | --- |
> | `cos_a1` · `sin_a1` | 두 요소의 **방향 벡터끼리**의 상대각 |
> | `cos_a2` · `sin_a2` | i 의 방향과 **i→j 변위 벡터** 사이의 각 |
> | `d_pos` | 거리 × 2 / 100 — `radius = 100 m` 로 정규화 |
>
> 논문에 나오는 sin/cos 다중주파수 위치 인코딩(10단계 20채널)은 **`if False:` 로 꺼져 있다.**
> 실제로 쓰이는 것은 위 5채널뿐이다.
>
> 비용은 N² 로 커진다 — 중앙값 `N ≈ 26 + 98 = 124` 라 `124² × 5`. 작지 않지만 병목은 아니다.

**→ 그래도 병목이 아니다.** R0 실측 GPU 사용률 중앙 **88%**,
단계별로 backward **65.1%** · forward **28.7%** · **로더 대기 0.2%**.

---

## 4. 로더가 캐시에서 모델 입력을 만드는 과정

`simpl/av2_dataset.py` 의 `actor_gather`(agent)와 `graph_gather`(차로)가 담당한다.

**agent — 원본 14채널** (`in_actor = 14`)

| 변위 Δp | 속도 vx·vy | cos h · sin h | object type | validity |
| --- | --- | --- | --- | --- |
| 2 | 2 | 2 | 7 | 1 |

**agent — R1 의 11채널** (`SIMPL_ACTOR_FEAT=ah2`)

| a | cos h · sin h | object type | validity |
| --- | --- | --- | --- |
| 1 | 2 | 7 | 1 |

전처리 재생성 없이 **로더에서 변환**한다. 바뀌는 것은 `actor_gather` 하나와
`config/simpl_av2_cfg.py` 의 `in_actor` 뿐이다.

**차로 — 16채널** (`in_lane = 16`, `av2_dataset.py:313`)

| node_ctrs | node_vecs | intersect | lane_type | cross_left | cross_right | left | right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | 1 | 3 | 3 | 3 | 1 | 1 |

**지도 분기는 R1 에서 건드리지 않았다** — 차로 그래프도, agent 의 현재 pose 도, RPE 도 원본 그대로다.

---

## 5. ⚠️ 주의할 두 가지 (실제로 걸렸던 것)

### ① 앞 2 step 이 조용히 버려진다

`actor_gather` 끝의 `actors[..., 2:]` 에 `# ! tmp solution` 주석이 달려 있고,
**50 step 중 앞 2개를 버려 48 step** 이 된다.

(a, h) 를 만들 때는 **먼저 48 step 으로 자른 뒤 차분**해야 한다. 순서를 바꾸면
**`a[0] = v₀` 가 파괴된다** — `a` 는 가속도가 아니라 속도 증분이고 첫 값이 초기 속력이기 때문이다.

### ② 결측 step 은 0 이 아니라 "최근접 복사" 로 채워진다

`padding_traj_nn` 이 위치를 가장 가까운 유효 step 값으로 메우고 속도는 0 으로 둔다.
결과적으로 그 구간은 **Δp = 0** 이 되어 저속 게이트가 전부 발동한다 —
**패딩 step 108,983개가 100.0% 전부** 걸린다. 설계대로 동작하는 것이지 버그가 아니다.

---

## 6. v4 의 평활·cleansing 은 들어 있지 않다

의도적이다. 지금 원칙이 **"한 번에 하나만 바꾼다"** 라서, R1 에서 바꾼 것은 입력 채널 구성 하나뿐이다.
평활을 같이 넣으면 R1 − R0 차이를 가를 수 없게 된다.

R1 에 들어간 **유일한 추가 가공**은 저속 보조(`V_MIN = 0.5`, 2026-11-02 승인)이고,
전체 agent 의 **63.4%** 가 이 게이트에 걸린다 — 쪼개 보면 **focal 은 13.3% 뿐**이고
다수인 쪽은 주변 agent 다.

> ⚠️ 이것도 엄밀히는 R1 에 섞인 **두 번째 변경**이다. R1 − R0 의 교란에 대해서는
> 📘 **「R1 결과 분석」** §2 를 볼 것 — 거기엔 초기화·데이터 순서까지 **셋**이 적혀 있다.

---

## 7. 확인 경로

이 문서의 수치는 전부 아래에서 직접 읽었다. 다시 확인할 때 쓰면 된다.

| 항목 | 어디서 |
| --- | --- |
| 전처리 진입점·인자 | `data_av2/run_preprocess.py:29-36` |
| obs/pred step 수 | 같은 파일 `:32-33`, 현재 시점 `av2_preprocess.py:170` |
| 차로 세그먼트 파라미터 | `av2_preprocess.py:33-34` (`SEG_LENGTH` · `SEG_N_NODE`) |
| agent 거리 필터 | `av2_preprocess.py:31, 203` (`FAR_DIST_THRES`) |
| 캐시 열 구성 | `av2_preprocess.py:106` |
| OBS/FUT 분할 | `av2_dataset.py:86-93` |
| **RPE** | `av2_dataset.py:142-146`(호출) · `:183-214`(정의) |
| 차로 16채널 조립 | `av2_dataset.py:313-319` |
| 실행 스크립트 | `scripts/av2_preproc_all.sh` |

```bash
# 캐시 한 개를 직접 열어 보기
python -c "
import pandas as pd, glob
r = pd.read_pickle(sorted(glob.glob('/data/argoverse2/simpl_features/val/*.pkl'))[0]).iloc[0]
print(list(r.index)); print(list(r['TRAJS'].keys())); print(list(r['LANE_GRAPH'].keys()))"
```

---

## 8. 더 읽을 것

- 📘 **「R1 결과 분석」** — R1 − R0 의 교란 셋, 판정 기준, 지금까지의 수치
- 📘 **「아이디어 노트 — (a, h)」** — `a` 와 `h` 의 정확한 정의, 측정으로 뒤집힌 다섯
