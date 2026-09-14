"""
heading_decomp.py - v4 의 방향 표현  h = k + theta  를 만들고 되돌린다.

    h       진행 방향        (여기서 직접 만든다 — AV2 heading 필드를 그대로 쓰지 않는다)
    k       차로 방향각      '도로의 휘어짐'  = 차로 중심선의 접선각
    theta   차로 대비 잔차각 '차체의 휘어짐'  = wrap(h - k)

왜 나누나
---------
h 는 도시 도로 방향을 따라 360° 전체에 퍼져 있어 모델이 절대각을 외워야 한다.
theta 는 0 근처에 몰려 있고(주행 대부분이 차로와 나란함) 큰 부분은 지도가 공짜로 준다.
단위가 같아 지도 융합이 학습된 변환기 없이 **덧셈 한 번**으로 끝나는 것도 이유다.

이름 주의
---------
docs/action_design_explained.md 의 `kappa` 는 **곡률 [1/m]**, 여기 k 는 **각도 [rad]** 다.
두 식은 같은 분해의 미분/적분 관계다:   dh/ds = kappa_lane(s) + d(theta)/ds
각도 레벨(여기)로 예측하면 적분이 한 단계 줄어 잔차 편향이 거리에 비례해 커지지 않는다.

왜 heading 을 직접 만드나
-------------------------
AV2 heading 은 **차체 방향**이라 이동방향과 슬립각만큼 어긋난다. 방향만 바꿔 궤적을 다시
굴리면(--task recon) AV2 heading 은 최종 1.693 m 어긋나고, 위치차분은 정의상 0.000 m 다.

주변 차량의 '뒤집힌 라벨'은 나눠 봐야 한다 (--task flip, val 3,000 시나리오 / 차량 12.5만 트랙).
총 이동거리별 뒤집힘 비율:  0–2 m 42.4% / 2–5 m 35.3% / 5–20 m 12.3% / 20–50 m 0.7% / 50 m+ 0.1%
→ **실제로 움직인 차량의 라벨은 거의 안 뒤집혀 있다.** 크게 나오는 수치는 거의 안 움직인
트랙에서 비교 기준인 '이동방향' 자체를 못 세우는 것이고, 후진 차량도 여기 섞인다.
즉 위치차분을 쓰는 이유는 '라벨이 망가져서'가 아니라 **복원이 무손실이기 때문**이다.
다만 저속 트랙에서는 뒤집힘을 검증할 방법이 없으므로 보조 신호는 의심하고 써야 한다.

  → h 는 **위치차분**으로 만들고, 위치차분이 무의미해지는 저속·정지 구간만 AV2 heading 으로 메운다.
  → 보조로 쓸 때는 그 트랙의 주행 구간과 대조해 **180° 뒤집힘을 먼저 교정**한다 (align_ref).

  대가: theta 가 담는 것이 '차체 자세'가 아니라 '진행방향'이 된다 — 슬립각(p95 9.9°)을 버린다.

기준 차로를 무엇으로 잡나
-------------------------
  ref=point  매 시점 진행방향과 같은 쪽을 향하는 중심선 점 중 최근접. 지도만 있으면 된다.
  ref=route  예측 시작점에서 규칙이 허용하는 경로를 열거하고 그중 하나를 기준으로. (L3 방식)
5 m 안에 기준이 없으면 valid=False — 주차장·도로 밖이라 k 가 정의되지 않는 구간이고,
학습에서는 이 플래그가 곧 "여기서는 차로를 믿지 마라" 채널이 된다.

  python src/heading_decomp.py --task speed                    # 저속 보조 기준속도 실측
  python src/heading_decomp.py --task theta --ref point --limit 3000
  python src/heading_decomp.py --task theta --ref route --limit 3000
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_frame import build_routes, densify, tangents, to_frame
from lane_graph import LaneGraph, REACH_MARGIN_M
import motion_repr as mr

ROOT = Path("/data/argoverse2/motion_forecasting")
OBS_LEN = 50
DT = 0.1               # AV2 10 Hz
MAX_LAT_M = 5.0        # 이보다 멀면 '차로 없음' (lane_graph.CAND_RADIUS_M 과 같은 기준)
STEP_M = 0.5           # 중심선 리샘플 간격 — 접선각 해상도
JUMP_DEG = 30.0        # 스텝당 방향 변화 상한 (10 Hz 에서 300°/s — 차량 한계 밖)
# 이 속력 미만이면 위치차분 방향을 못 믿고 보조로 메운다. 실측(--task speed / --task recon)으로 정함:
#   0.5 m/s 미만  스텝간 방향이 7~11° 흔들림 = 70~110°/s → 차량 한계 밖, 명백한 노이즈
#   0.5~1.0       2.97°/step (30°/s) → 여전히 의심스러움
#   1.0~1.5       1.81°/step (18°/s) → 실제 저속 회전으로 설명 가능
# 복원오차(최종)는 0.5 → 0.087 m, 1.0 → 0.145 m, 2.0 → 0.310 m 로 커진다.
# 1.0 은 '물리적으로 불가능한 구간을 전부 제거하는 가장 낮은 값'이고,
# AV2 heading 을 그대로 쓸 때(1.693 m)보다 여전히 12배 정확하다.
V_MIN = 1.0


def wrap(a):
    """각도를 (-pi, pi] 로."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def guard(h, jump_deg=JUMP_DEG):
    """스텝당 |dh| 가 한계를 넘으면 직전 값 유지.

    전수 조사에서 2,573만 step 중 11건(174° 뒤집힘 포함)이 물리적으로 불가능했다.
    정상값 상한(p99.99 = 7.3°)의 4배로 자르면 이상치만 잡고 실제 회전은 안 건드린다.
    """
    out = np.array(h, dtype=np.float64)
    lim = np.radians(jump_deg)
    for t in range(1, len(out)):
        if np.isfinite(out[t]) and np.isfinite(out[t - 1]) \
                and abs(wrap(out[t] - out[t - 1])) > lim:
            out[t] = out[t - 1]
    return out


def align_ref(h_ref, h, obs=None):
    """보조 heading 의 180° 뒤집힘을 그 트랙의 이동방향으로 교정한다.

    주변 차량 트랙의 8.5% 가 뒤집혀 있다. 뒤집힘은 트랙 안에서 일정하므로
    (트랙 안 변화 0.18°) 중앙값 한 번으로 판정하면 된다.
    """
    h_ref = np.asarray(h_ref, dtype=np.float64)
    m = np.isfinite(h) & np.isfinite(h_ref)
    if obs is not None:
        m &= obs
    if m.sum() < 3:
        return h_ref, False
    if np.median(np.abs(wrap(h_ref[m] - h[m]))) > np.pi / 2:
        return wrap(h_ref + np.pi), True
    return h_ref, False


def build_heading(pos, obs=None, h_ref=None, dt=DT, v_min=V_MIN, jump_deg=JUMP_DEG,
                  fallback="av2"):
    """위치차분으로 h 를 만든다. 저속 구간은 h_ref(AV2 heading)로 메운다.

    pos (T,2), obs (T,) 관측 여부. 주변 차량은 중간이 끊기므로 관측된 이웃끼리 차분한다.
    반환: h (T,), src (T,)  0=못 정함 / 1=위치차분 / 2=보조 / 3=이웃 값 유지
    """
    pos = np.asarray(pos, dtype=np.float64)
    T = len(pos)
    obs = np.ones(T, bool) if obs is None else np.asarray(obs, bool)
    idx = np.flatnonzero(obs)
    h = np.full(T, np.nan)
    src = np.zeros(T, np.int8)

    for a, b in zip(idx[:-1], idx[1:]):
        d = pos[b] - pos[a]
        if np.linalg.norm(d) / ((b - a) * dt) >= v_min:
            h[a], src[a] = np.arctan2(d[1], d[0]), 1
    if len(idx) >= 2 and src[idx[-2]] == 1:        # 마지막 관측점은 직전 값 복제
        h[idx[-1]], src[idx[-1]] = h[idx[-2]], 1

    if h_ref is not None and fallback == "av2":
        # 튐 가드는 **보조 신호에만** 건다. 위치차분 h 는 위치 그 자체라 라벨 오류가 없고,
        # 정지 구간의 큰 방향 변화도 실제 변위 방향이다. 여기에 가드를 걸면 그 값을 얼려서
        # '위치차분이라 복원이 무손실'이라는 이득이 통째로 사라진다 (실측 0.00 → 2.34 m).
        h_ref = guard(h_ref, jump_deg)
        h_ref, _ = align_ref(h_ref, h, obs)
        fill = obs & ~np.isfinite(h) & np.isfinite(h_ref)
        h[fill], src[fill] = h_ref[fill], 2

    # fallback="hold" 면 보조 없이 가장 가까운 위치차분 값을 끌어다 쓴다 (motion_repr 방식)
    good = np.flatnonzero(np.isfinite(h))          # 남은 구멍은 가장 가까운 값으로
    if len(good):
        miss = np.flatnonzero(obs & ~np.isfinite(h))
        if len(miss):
            j = good[np.abs(miss[:, None] - good[None, :]).argmin(axis=1)]
            h[miss], src[miss] = h[j], 3
    rest = obs & ~np.isfinite(h)
    if rest.any():
        # 트랙 내내 v_min 을 못 넘긴 경우(주차 차량 등). 끌어다 쓸 위치차분 값이 없으니
        # 방향을 알려줄 수 있는 건 바운딩 박스뿐이다.
        h[rest] = 0.0 if h_ref is None else guard(np.asarray(h_ref, float), jump_deg)[rest]
        src[rest] = 2
    return h, src


def ah_features(pos_obs, head_obs, yaw0, dt=DT, v_min=V_MIN):
    """v4 입력 2채널 (a, h) 를 **관측 구간만으로** 만든다.

      a   속도 증분 [km/h] / 3   — motion_repr 의 ah0 방식. 누적합하면 속도가 되고 첫 값이 v0 다.
      h   진행 방향 [rad]        — focal 정규화 프레임, (-pi, pi] 로 wrap

    왜 이 형태인가
    --------------
    출력이 (a, h = k + θ) 이므로 입력도 같은 두 양으로 맞춘다. 입력의 h 는 **h 자체**다.
    지도(차로·후보 경로)는 모델이 따로 받으므로 k 와 θ 로 쪼개 넣을 이유가 없고,
    그렇게 해야 2채널 제약도 지킨다.

    a 에 v0 를 싣는 이유: (a, h) 만으로는 초기속도가 빠져 궤적을 복원할 수 없다(v0 를
    모르면 FDE 13.4 m). 첫 스텝이 v0 를 통째로 들고 있으면 누적합으로 속도가 무손실로
    돌아온다. 예전 ah0_2 실험(minADE6 1.406)과 같은 인코딩이다.

    h 를 AV2 heading 대신 위치차분으로 만드는 이유: AV2 heading 은 차체 방향이라 이동방향과
    슬립각만큼 어긋난다(방향만 바꿔 다시 굴리면 AV2 1.693 m / 위치차분 기반 0.145 m).

    **관측 구간만 받는다.** build_heading 은 전방차분이라 110 스텝을 다 넣으면 h[49] 가
    pos[50] — 첫 예측 대상 — 을 보고, 저속 채우기와 뒤집힘 판정도 미래를 본다.

    반환: feat (T,2) float32, h0 = 마지막 관측 스텝의 진행방향(정규화 프레임, rad).
    h0 는 적분기의 시작 잔차각 θ₀ = wrap(h0 − k(s0)) 에 쓴다. 지금의 θ₀ = −k(s0) 는
    정규화 프레임의 0 방향(= AV2 차체 방향)을 진행방향으로 간주해 슬립각만큼 어긋난다.
    """
    pos_obs = np.asarray(pos_obs, dtype=np.float64)
    m = mr.traj_to_motion(pos_obs, dt=dt, stop_ms=1.0, smooth=1)
    a_ch = m.dv_kph / mr.DEFAULT_SCALES["dv_kph"]
    h_city, _ = build_heading(pos_obs, h_ref=np.asarray(head_obs, dtype=np.float64),
                              dt=dt, v_min=v_min)
    h_n = wrap(h_city - float(yaw0))
    return np.stack([a_ch, h_n], axis=1).astype(np.float32), float(h_n[-1])


# ------------------------------------------------------------------ 차로 방향각 k
def lane_field(graph, near_xy=None, near_m=120.0, step=STEP_M):
    """모든 차로 중심선을 등간격 점으로 펴서 (점, 접선각, 교차로여부) 로 쌓는다.

    차로 단위로 최근접을 찾으면 차로 경계에서 기준이 튀므로 점 단위로 본다.
    near_xy 를 주면 그 근처 차로만 남겨 비교량을 줄인다.
    """
    P, A, I = [], [], []
    for ln in graph.lanes.values():
        c = np.asarray(ln.centerline, dtype=np.float64)[:, :2]
        if len(c) < 2:
            continue
        if near_xy is not None and np.linalg.norm(c - near_xy, axis=1).min() > near_m:
            continue
        d, _ = densify(c, step)
        if len(d) < 2:
            continue
        t = tangents(d)
        P.append(d)
        A.append(np.arctan2(t[:, 1], t[:, 0]))
        I.append(np.full(len(d), float(ln.is_intersection)))
    if not P:
        return None
    return np.concatenate(P), np.concatenate(A), np.concatenate(I)


def decompose(pos, h, field, max_lat=MAX_LAT_M):
    """h (T,) 를 k, theta 로 분해. pos (T,2) 는 city 좌표.

    반환: k, theta, valid, lat(횡거리), inter(교차로 차로인가)
    """
    P, A, I = field
    T = len(h)
    k = np.zeros(T)
    lat = np.full(T, np.inf)
    inter = np.zeros(T)
    for t in range(T):
        if not np.isfinite(h[t]):
            continue
        d2 = ((P - pos[t]) ** 2).sum(axis=1)
        # 대향차로 제외: 점 접선이 진행방향과 90° 안쪽인 점만 후보
        d2 = np.where(np.cos(A - h[t]) > 0, d2, np.inf)
        j = int(d2.argmin())
        if not np.isfinite(d2[j]):
            continue
        k[t], lat[t], inter[t] = A[j], np.sqrt(d2[j]), I[j]
    return k, wrap(h - k), lat <= max_lat, lat, inter


def route_of(graph, pos, h, obs_len=OBS_LEN):
    """예측 시작점에서 규칙이 허용하는 경로를 만들고, 실제 미래에 가장 잘 맞는 하나를 고른다.

    L3(후보 경로 = 예측 모드)가 성공했을 때 L0 가 받게 될 기준이다.
    정답 미래로 고르므로 **오라클** — theta 분포의 상한을 재는 용도다.
    """
    p0 = pos[obs_len - 1]
    h0 = np.array([np.cos(h[obs_len - 1]), np.sin(h[obs_len - 1])])
    starts = graph.candidate_lanes(p0, h0, path=pos[:obs_len])
    if not starts:
        return None
    speed = float(np.linalg.norm(pos[obs_len - 1] - pos[obs_len - 2])) / DT
    reach = graph.reachable(starts, max(20.0, speed * 6.0) + REACH_MARGIN_M)
    # v0 를 넘겨 경로가 6초 주행거리를 덮게 한다 (안 넘기면 30 m 로 잘려 궤적이 경로 끝에서
    # 얼어붙고 |d| 가 무한정 커진다 — 실측으로 시나리오의 34.5% 가 그랬다).
    # 접힘 필터는 없앴다. build_routes 가 succ-only 라 접힌 경로를 애초에 만들지 않는다.
    routes = build_routes(graph, starts, reach, v0=speed)
    if not routes:
        return None
    fut = pos[obs_len:]
    best, bm = None, np.inf
    for r in routes:
        # 전역 최근접이 아니라 to_frame 의 전진 대응으로 잰다. 전역으로 재면 되돌아온
        # 구간에 대응점이 붙어 거리가 작게 나온다.
        _, _, md = to_frame(fut, r)
        if md < bm:
            best, bm = r, md
    return best


def decompose_on_route(pos, h, route):
    """기준을 최근접 차로 점 대신 **고른 경로**로 두고 분해한다.

    경로는 방향이 하나뿐이라 대향차로 필터가 필요 없고 교차로에서 분기가 섞이지 않는다.
    대응점은 직전 점보다 앞에서만 찾는다 (lane_frame.to_frame 과 같은 이유).
    """
    P, TAN, S = route["pts"], route["tan"], route["s"]
    A = np.arctan2(TAN[:, 1], TAN[:, 0])
    T = len(h)
    k = np.zeros(T); lat = np.zeros(T); js = np.zeros(T, int)
    j_prev, win = 0, int(25.0 / 0.25)
    for t in range(T):
        lo = j_prev if t else 0
        hi = min(len(P), (j_prev + win) if t else len(P))
        d2 = ((P[lo:hi] - pos[t]) ** 2).sum(axis=1)
        j = int(d2.argmin()) + lo
        j_prev, js[t] = j, j
        k[t], lat[t] = A[j], np.sqrt(d2[j - lo])
    return k, wrap(h - k), lat, S[js]


def compose(k, theta):
    """되돌리기. h = k + theta."""
    return wrap(np.asarray(k) + np.asarray(theta))


def inter_on_route(graph, route, s_at):
    """경로 위 각 시점이 교차로 차로 구간인가."""
    seg, acc = [], 0.0
    for lid in route["lanes"]:
        c = np.asarray(graph.lanes[lid].centerline, dtype=np.float64)[:, :2]
        L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        seg.append((acc, acc + L, float(graph.lanes[lid].is_intersection)))
        acc += L
    out = np.zeros(len(s_at))
    for t, st in enumerate(s_at):
        for a, b, f in seg:
            if a <= st <= b:
                out[t] = f
                break
    return out


# ------------------------------------------------------------------ 시나리오 읽기
def focal(d):
    s = scenario_serialization.load_argoverse_scenario_parquet(
        d / f"scenario_{d.name}.parquet")
    f = next(t for t in s.tracks if t.track_id == s.focal_track_id)
    st = sorted(f.object_states, key=lambda x: x.timestep)
    return (np.array([x.position for x in st], dtype=np.float64),
            np.array([x.velocity for x in st], dtype=np.float64),
            np.array([x.heading for x in st], dtype=np.float64))


def graph_of(d):
    return LaneGraph.from_json_dict(
        json.loads((d / f"log_map_archive_{d.name}.json").read_text()))


# ------------------------------------------------------------------ task: speed
# 저속에서 위치차분 방향이 언제부터 못 쓰게 되는지 — 보조로 넘어갈 기준속도를 정한다.
SP_BINS = np.array([0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9])


def one_speed(d):
    try:
        pos, vel, hav2 = focal(d)
        if len(pos) < 3:
            return None
        dp = np.diff(pos, axis=0)
        sp = np.linalg.norm(dp, axis=1) / DT
        hm = np.arctan2(dp[:, 1], dp[:, 0])
        L = len(pos) - 2                      # 세 배열의 공통 길이
        b = np.digitize(sp[:L], SP_BINS) - 1
        # 이동방향의 스텝간 떨림 / AV2 heading 의 떨림 / 둘의 불일치
        jm = np.abs(np.degrees(wrap(hm[1:L + 1] - hm[:L])))
        ja = np.abs(np.degrees(wrap(hav2[1:L + 1] - hav2[:L])))
        dis = np.abs(np.degrees(wrap(hm[:L] - hav2[:L])))
        n = len(SP_BINS) - 1
        out = np.zeros((n, 4))
        for i in range(n):
            m = b == i
            if m.any():
                out[i] = [m.sum(), np.median(jm[m]), np.median(ja[m]), np.median(dis[m])]
        return out
    except Exception:
        return None


def run_speed(a):
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one_speed, dirs, chunksize=16) if r is not None]
    R = np.array(res)
    n = R[:, :, 0].sum(axis=0)
    print(f"\n=== [{a.split}] 저속에서 위치차분 방향이 언제 무너지나  (focal {len(res):,} 트랙) ===")
    print(f"{'속력 [m/s]':>14} {'step':>12} {'위치차분 떨림':>14} {'AV2 떨림':>12} {'둘의 불일치':>12}")
    for i in range(len(n)):
        if n[i] == 0:
            continue
        w = R[:, i, 0]
        med = lambda c: float(np.average(R[:, i, c], weights=w)) if w.sum() else np.nan
        lo, hi = SP_BINS[i], SP_BINS[i + 1]
        rng = f"{lo:.1f}–{hi:.1f}" if hi < 1e8 else f"{lo:.0f}+"
        print(f"{rng:>14} {int(n[i]):>12,} {med(1):>13.2f}° {med(2):>11.2f}° {med(3):>11.2f}°")
    print("\n떨림 = 스텝간 방향 변화의 중앙값. 실제 회전은 10 Hz 에서 1° 안쪽이라,")
    print("이보다 크면 추정 노이즈다. 위치차분 떨림이 AV2 떨림보다 커지는 구간이 보조가 필요한 곳이다.")


# ------------------------------------------------------------------ task: flip
PATH_BINS = np.array([0.0, 2.0, 5.0, 20.0, 50.0, 1e9])
# 주변 차량 라벨이 실제로 얼마나 뒤집혀 있는지, 그리고 build_heading 이 그걸 잡는지 확인.
def one_flip(d, v_min=V_MIN):
    try:
        s = scenario_serialization.load_argoverse_scenario_parquet(
            d / f"scenario_{d.name}.parquet")
        T = 110
        acc = {}
        for tr in s.tracks:
            st = sorted(tr.object_states, key=lambda x: x.timestep)
            if len(st) < 5:
                continue
            pos = np.full((T, 2), np.nan)
            hav2 = np.full(T, np.nan)
            obs = np.zeros(T, bool)
            for x in st:
                if 0 <= x.timestep < T:
                    pos[x.timestep] = x.position
                    hav2[x.timestep] = x.heading
                    obs[x.timestep] = True
            ot = str(getattr(tr, "object_type", "?")).split(".")[-1].lower()
            h, src = build_heading(pos, obs=obs, h_ref=None, v_min=v_min)
            m = obs & (src == 1)                     # 위치차분으로 정해진 구간 = 실제 이동
            # 총 이동거리로 나눠 본다. 거의 안 움직인 트랙은 '이동방향' 자체가 노이즈라
            # 절반이 뒤집힌 것으로 세진다 — 그건 라벨 오류가 아니라 판정 불가다.
            o = np.flatnonzero(obs)
            path = float(np.linalg.norm(np.diff(pos[o], axis=0), axis=1).sum()) if len(o) > 1 else 0.0
            b = int(np.searchsorted(PATH_BINS, path, side="right")) - 1
            key = (ot, max(b, 0))
            a = acc.setdefault(key, [0, 0, 0])
            a[0] += 1
            if m.sum() < 5:
                continue
            a[1] += 1
            if np.median(np.abs(wrap(hav2[m] - h[m]))) > np.pi / 2:
                a[2] += 1
        return acc
    except Exception:
        return None


def run_flip(a):
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one_flip, dirs, chunksize=16) if r]
    acc = {}
    for r in res:
        for k, v in r.items():
            cur = acc.setdefault(k, [0, 0, 0])
            for i in range(3):
                cur[i] += v[i]
    print(f"\n=== [{a.split}] AV2 heading 이 이동방향과 180° 뒤집힌 트랙  (시나리오 {len(res):,}) ===")
    VEH = {"vehicle", "bus", "truck", "truck_cab", "box_truck", "large_vehicle",
           "motorcyclist", "articulated_bus", "school_bus"}
    lbl = [f"{PATH_BINS[i]:.0f}–{PATH_BINS[i+1]:.0f} m" if PATH_BINS[i+1] < 1e8
           else f"{PATH_BINS[i]:.0f} m 이상" for i in range(len(PATH_BINS) - 1)]
    print(f"{'트랙 총 이동거리':>16} {'트랙':>10} {'판정가능':>10} {'뒤집힘':>9} {'비율':>8} {'시나리오당':>10}")
    for grp, keep in (("차량 계열", lambda t: t in VEH), ("그 외", lambda t: t not in VEH)):
        print(f"  [{grp}]")
        tot = [0, 0, 0]
        for b in range(len(PATH_BINS) - 1):
            n, m, f = [sum(acc.get((t, b), [0, 0, 0])[i] for t in
                           {k[0] for k in acc} if keep(t)) for i in range(3)]
            for i, v in enumerate((n, m, f)):
                tot[i] += v
            print(f"{lbl[b]:>16} {n:>10,} {m:>10,} {f:>9,} {f/max(m,1)*100:7.1f}% "
                  f"{f/len(res):10.2f}")
        print(f"{'합계':>16} {tot[0]:>10,} {tot[1]:>10,} {tot[2]:>9,} "
              f"{tot[2]/max(tot[1],1)*100:7.1f}% {tot[2]/len(res):10.2f}")
    print("\n위치차분으로 h 를 만들면 이 트랙들이 자동으로 바로잡힌다. 저속 보조로 AV2 를 쓸 때만")
    print("align_ref() 가 트랙별로 180° 를 되돌린다 — 안 하면 정지 구간을 통해 뒤집힘이 되돌아온다.")


# ------------------------------------------------------------------ task: recon
# 기준속도(V_MIN)를 얼마로 둘지: 저속을 AV2 로 메울수록 신호는 깨끗해지지만
# '위치차분이라 복원이 무손실'이라는 이득을 깎는다. 둘의 교환을 직접 잰다.
#   길이는 위치에서 그대로 쓰고 **방향만** 각 방식으로 바꿔 굴린다 → 오차 원인이 방향뿐이다.
V_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def rollout(pos, h):
    """방향 h 로만 궤적을 다시 굴린다. 스텝 길이는 원본에서 가져온다."""
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    d = np.stack([np.cos(h[:-1]), np.sin(h[:-1])], axis=1) * step[:, None]
    return np.concatenate([pos[:1], pos[0] + np.cumsum(d, axis=0)])


def one_recon(d):
    try:
        pos, vel, hav2 = focal(d)
        if len(pos) < 3:
            return None
        out = []
        for fb in ("av2", "hold"):
            for v in V_GRID:
                h, src = build_heading(pos, h_ref=hav2, v_min=v, fallback=fb)
                e = np.linalg.norm(rollout(pos, h) - pos, axis=1)
                out.append([e.mean(), e[-1], (src != 1).mean()])
        h = guard(hav2)                                  # 비교군: AV2 heading 그대로
        e = np.linalg.norm(rollout(pos, h) - pos, axis=1)
        out.append([e.mean(), e[-1], 1.0])
        return np.array(out)
    except Exception:
        return None


def run_recon(a):
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one_recon, dirs, chunksize=16) if r is not None]
    R = np.array(res)
    print(f"\n=== [{a.split}] 방향만 바꿔 궤적 복원  (focal {len(res):,} 트랙) ===")
    print("스텝 길이는 원본 그대로 — 오차는 전부 '방향을 무엇으로 만들었나' 때문이다.\n")
    print(f"{'h 만드는 법':>26} {'평균오차':>10} {'최종오차':>10} {'p95 최종':>10} {'보조로 메운 비율':>15}")
    for b, fb in enumerate(("AV2 heading", "직전 방향 유지")):
        for i, v in enumerate(V_GRID):
            j = b * len(V_GRID) + i
            if v == 0:
                if b:
                    continue
                name = "위치차분만 (보조 없음)"
            else:
                name = f"<{v} m/s 를 {fb}"
            print(f"{name:>26} {R[:,j,0].mean():9.3f} m {R[:,j,1].mean():9.3f} m "
                  f"{np.percentile(R[:,j,1],95):9.3f} m {R[:,j,2].mean()*100:14.1f}%")
    print(f"{'AV2 heading 그대로':>26} {R[:,-1,0].mean():9.3f} m {R[:,-1,1].mean():9.3f} m "
          f"{np.percentile(R[:,-1,1],95):9.3f} m {100.0:14.1f}%")


# ------------------------------------------------------------------ task: theta
BINS = np.concatenate([np.arange(0, 60, 0.25), np.arange(60, 181, 1.0)])


def one_theta(d, ref="point", src_av2=False):
    try:
        pos, vel, hav2 = focal(d)
        if len(pos) < OBS_LEN + 2:
            return None
        if src_av2:
            h, src = guard(hav2), np.full(len(pos), 2, np.int8)
        else:
            h, src = build_heading(pos, h_ref=hav2)
        g = graph_of(d)
        if ref == "route":
            r = route_of(g, pos, h)
            if r is None:
                return None
            k, th, lat, s_at = decompose_on_route(pos, h, r)
            inter = inter_on_route(g, r, s_at)
            valid = np.zeros(len(h), bool)
            valid[OBS_LEN - 1:] = lat[OBS_LEN - 1:] <= MAX_LAT_M
            n_eval = len(h) - OBS_LEN + 1
        else:
            fld = lane_field(g, near_xy=pos[OBS_LEN - 1])
            if fld is None:
                return None
            k, th, valid, lat, inter = decompose(pos, h, fld)
            n_eval = len(h)
        valid &= np.isfinite(th)

        moving = np.linalg.norm(vel, axis=1) >= 0.5
        deg = np.degrees(np.abs(th))
        m = valid
        H = [np.histogram(deg[m & sel], bins=BINS)[0] for sel in
             (np.ones(len(m), bool), inter > 0, moving, ~moving)]
        a_, b_ = OBS_LEN - 1, len(h) - 1
        dh = dth = np.nan
        if valid[a_] and valid[b_]:
            dh = abs(np.degrees(wrap(h[b_] - h[a_])))
            dth = abs(np.degrees(wrap(th[b_] - th[a_])))
        return (H, int(m.sum()), n_eval, dh, dth,
                int((src == 1).sum()), int((src == 2).sum()), len(h))
    except Exception:
        return None


def pct(hist, q):
    c = np.cumsum(hist) / max(hist.sum(), 1)
    return BINS[min(int(np.searchsorted(c, q / 100.0)), len(BINS) - 2)]


def share(hist, upto_deg):
    return hist[BINS[:-1] < upto_deg].sum() / max(hist.sum(), 1) * 100


def run_theta(a):
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(
            partial(one_theta, ref=a.ref, src_av2=a.src == "av2"), dirs, chunksize=8) if r]
    HS = [np.sum([r[0][i] for r in res], axis=0) for i in range(4)]
    nv, nt = sum(r[1] for r in res), sum(r[2] for r in res)
    dh = np.array([r[3] for r in res]); dth = np.array([r[4] for r in res])
    ok = np.isfinite(dh)
    n1, n2, na = (sum(r[i] for r in res) for i in (5, 6, 7))

    print(f"\n=== [{a.split}] h={a.src}  기준={a.ref}  시나리오 {len(res):,} ===")
    if a.src == "build":
        print(f"h 출처: 위치차분 {n1/na*100:.1f}%   저속 보조(AV2) {n2/na*100:.1f}%   "
              f"이웃값 유지 {(na-n1-n2)/na*100:.1f}%")
    print(f"기준 차로 매칭 {nv:,} / {nt:,} step ({nv/nt*100:.1f}%)")
    print(f"\n|theta| = |h - k|")
    for name, h_ in zip(("전체", "교차로 차로", "주행 중", "정지 중"), HS):
        if h_.sum() == 0:
            continue
        print(f"  {name:10} n {h_.sum():>9,}   중앙 {pct(h_,50):5.2f}°  p90 {pct(h_,90):6.2f}°  "
              f"p95 {pct(h_,95):6.2f}°  p99 {pct(h_,99):7.2f}°   "
              f"±5° {share(h_,5):5.1f}%  ±10° {share(h_,10):5.1f}%  ±20° {share(h_,20):5.1f}%")
    print(f"\n6초 구간에서 따라가야 할 변화량 (n={int(ok.sum()):,})")
    print(f"  |dh|     (절대각)  중앙 {np.median(dh[ok]):6.2f}°  p90 {np.percentile(dh[ok],90):7.2f}°  "
          f"p99 {np.percentile(dh[ok],99):7.2f}°")
    print(f"  |dtheta| (잔차)    중앙 {np.median(dth[ok]):6.2f}°  p90 {np.percentile(dth[ok],90):7.2f}°  "
          f"p99 {np.percentile(dth[ok],99):7.2f}°")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="theta", choices=["theta", "speed", "recon", "flip"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ref", default="point", choices=["point", "route"])
    ap.add_argument("--src", default="build", choices=["build", "av2"],
                    help="h 를 위치차분으로 만들지(build) AV2 필드를 그대로 쓸지(av2)")
    a = ap.parse_args()
    {"speed": run_speed, "recon": run_recon, "flip": run_flip,
     "theta": run_theta}[a.task](a)


if __name__ == "__main__":
    main()
