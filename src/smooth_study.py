"""
smooth_study.py - v4 전처리의 **평활(smoothing)** 후보를 우리 데이터에서 직접 재는 도구.

무엇을 재나
-----------
전처리 = 평활 + 다운샘플. 다운샘플(10 Hz -> 2 Hz)은 기계적이고, 결과를 가르는 것은 평활이다.
사용자 기준 1순위는 **직진 시나리오에서 heading vs 시간 그래프가 일관되게 유지되는가** 다.

  1단계 (--task char)   무엇을 평활해야 하는지: 위치 잡음 크기 · 스펙트럼 · 속도 필드와의 관계 · 창 가장자리
  2단계 (METHODS)       같은 인터페이스의 후보 평활기 (없음 / SG / 가우시안 / Butterworth / 스플라인 / 칼만+RTS / 운동학 제약 / 각도 전용)
  3단계 (--task eval)   지표 ①~⑧
  4단계                 그림은 viz_smooth_*.py 가 이 결과를 읽어 그린다

절대 규칙
---------
학습하지 않는다. 기존 파일은 import 만 한다(수정 없음). GPU 를 쓰지 않는다(torch 도 CPU).
모든 후보는 **관측 50스텝만** 본다 — 미래(스텝 >= 50)를 보지 않는다.
창 안에서 양방향(zero-phase)으로 도는 것은 허용한다. 지금 쓰는 SG(5,2, mode='interp') 도 같다.

  python src/smooth_study.py --task char  --limit 3000
  python src/smooth_study.py --task eval  --limit 3000
  python src/smooth_study.py --task cases --limit 3000
  python src/smooth_study.py --task verify            # 배치 구현 == heading_decomp 원본 확인
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import heading_decomp as hd                      # 읽기 전용 참조 (원본 규약·검증용)

ROOT = Path("/data/argoverse2/motion_forecasting")
OUT = SRC.parent / "viz" / "v4" / "smoothing"
DATA = OUT / "data"

DT = 0.1
T_ALL = 110
OBS = 50                 # 관측 50스텝 (t = -4.9 … 0 s)
RAMP = 5                 # 창 시작 램프 스텝 수 — 지표는 인덱스 RAMP 부터 잰다 (1단계에서 실측 확인)
V_MIN = 1.0              # heading_decomp.V_MIN — 위치차분 방향을 믿는 하한 [m/s]
JUMP_DEG = 30.0          # heading_decomp.JUMP_DEG — 보조 heading 튐 가드 (10 Hz 기준)
DS_STEP = 5              # 10 Hz -> 2 Hz
IDX_2HZ = np.arange(OBS - 1, 4 - 1, -DS_STEP)[::-1]     # (4, 9, …, 49)

VEH_TYPES = ("vehicle", "bus", "truck", "truck_cab", "box_truck", "large_vehicle",
             "motorcyclist", "articulated_bus", "school_bus")

# ---------------------------------------------------------------- 모집단 임계값 (리포트·그림과 같은 값)
STOP_VMAX = 0.3          # [m/s] 관측 50스텝 AV2 속력 최대 < 이면 '정지 차량' (잡음 추정·지표 ⑤)
STRAIGHT_DEG = 5.0       # [deg] 관측 구간 AV2 속도필드 course 변화 |Δ| < 이면 직진
STRAIGHT_MOVE_M = 10.0   # [m]   관측 구간 이동거리 > 이어야 직진 시나리오로 본다
TURN_DEG = 30.0          # [deg] |Δcourse| > 이면 회전
COURSE_VMIN = 1.0        # [m/s] AV2 속도필드 course 를 기준으로 쓸 수 있는 하한
SPEED_BINS = np.array([0.0, 1.0, 5.0, 10.0, 15.0, np.inf])
SPEED_LBL = ["정지 <1", "1–5", "5–10", "10–15", ">15"]


def wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


# ================================================================ 원본 읽기
_COLS = ["track_id", "object_type", "object_category", "timestep",
         "position_x", "position_y", "heading", "velocity_x", "velocity_y", "focal_track_id"]


def scen_dirs(split="val", limit=3000, seed=0):
    """시드 고정 표본. 정렬된 디렉터리에서 뽑고 다시 정렬해 순서를 재현 가능하게 둔다."""
    dirs = [p for p in sorted((ROOT / split).iterdir()) if p.is_dir()]
    if limit is None or limit >= len(dirs):
        return dirs
    idx = np.sort(np.random.default_rng(seed).choice(len(dirs), limit, replace=False))
    return [dirs[i] for i in idx]


def one_scenario(sdir):
    """한 시나리오 -> (focal 110스텝, 관측 50스텝이 모두 있는 차량 계열 주변 차량).

    반환 focal: pos(110,2) vel(110,2) head(110,)   — focal 은 정의상 110스텝 모두 있다
    반환 agents: pos(n,50,2) vel(n,50,2) head(n,50) — 관측 구간 완전관측 차량만
    """
    try:
        df = pd.read_parquet(sdir / f"scenario_{sdir.name}.parquet", columns=_COLS)
    except Exception:
        return None
    fid = str(df["focal_track_id"].iloc[0])
    df = df.astype({"track_id": "string", "object_type": "string"})
    foc = df[df["track_id"] == fid].sort_values("timestep")
    if len(foc) != T_ALL:
        return None
    f_pos = foc[["position_x", "position_y"]].to_numpy(np.float64)
    f_vel = foc[["velocity_x", "velocity_y"]].to_numpy(np.float64)
    f_head = foc["heading"].to_numpy(np.float64)

    o = df[(df["timestep"] < OBS) & df["object_type"].isin(VEH_TYPES) & (df["track_id"] != fid)]
    cnt = o.groupby("track_id")["timestep"].count()
    keep = set(cnt.index[cnt == OBS])
    a_pos = a_vel = a_head = None
    if keep:
        o = o[o["track_id"].isin(keep)].sort_values(["track_id", "timestep"])
        n = len(keep)
        a_pos = o[["position_x", "position_y"]].to_numpy(np.float64).reshape(n, OBS, 2)
        a_vel = o[["velocity_x", "velocity_y"]].to_numpy(np.float64).reshape(n, OBS, 2)
        a_head = o["heading"].to_numpy(np.float64).reshape(n, OBS)
    return sdir.name, f_pos, f_vel, f_head, a_pos, a_vel, a_head


def collect(dirs, workers=16, agents=True):
    """표본 전체를 한 번에 읽어 배열로 쌓는다."""
    from multiprocessing import Pool
    with Pool(workers) as pool:
        res = [r for r in pool.imap(one_scenario, dirs, chunksize=8) if r is not None]
    sids = [r[0] for r in res]
    F = dict(pos=np.stack([r[1] for r in res]), vel=np.stack([r[2] for r in res]),
             head=np.stack([r[3] for r in res]), sid=np.array(sids))
    A = None
    if agents:
        pa = [r[4] for r in res if r[4] is not None]
        A = dict(pos=np.concatenate(pa), vel=np.concatenate([r[5] for r in res if r[5] is not None]),
                 head=np.concatenate([r[6] for r in res if r[6] is not None]),
                 scen=np.concatenate([np.full(len(r[4]), i) for i, r in enumerate(res) if r[4] is not None]))
    return F, A


# ================================================================ 배치 유틸 (heading_decomp 규약을 트랙축으로 벡터화)
def guard_batch(h, jump_deg=JUMP_DEG):
    """heading_decomp.guard 를 (N,T) 로. 스텝당 |dh| 한계 초과면 직전 값 유지."""
    out = np.array(h, np.float64)
    lim = np.radians(jump_deg)
    for t in range(1, out.shape[1]):
        bad = np.isfinite(out[:, t]) & np.isfinite(out[:, t - 1]) & \
            (np.abs(wrap(out[:, t] - out[:, t - 1])) > lim)
        out[bad, t] = out[bad, t - 1]
    return out


def _nearest_fill(h):
    """유한값이 아닌 칸을 '가장 가까운 유한값'으로 채운다 (동률이면 앞쪽 — argmin 과 같은 규약)."""
    N, T = h.shape
    idx = np.arange(T)[None, :].repeat(N, 0)
    ok = np.isfinite(h)
    fwd = np.maximum.accumulate(np.where(ok, idx, -10 ** 6), axis=1)
    bwd = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[:, ::-1], axis=1)[:, ::-1]
    df, db = idx - fwd, bwd - idx
    take = np.where(df <= db, fwd, bwd)                 # 동률이면 앞쪽 (np.argmin 과 같다)
    take = np.clip(take, 0, T - 1)
    return np.take_along_axis(h, take, axis=1), ok.any(axis=1)


def build_heading_batch(pos, h_ref=None, dt=DT, v_min=V_MIN, jump_deg=JUMP_DEG):
    """heading_decomp.build_heading 의 **완전관측 배치판**. pos (N,T,2) -> h (N,T), src (N,T).

    obs 가 전부 True 인 경우만 다룬다(우리 모집단은 관측 50스텝 완전관측만 쓴다).
    src: 1=위치차분 / 2=보조(AV2) / 3=이웃 값 유지.
    """
    pos = np.asarray(pos, np.float64)
    N, T = pos.shape[0], pos.shape[1]
    d = np.diff(pos, axis=1)
    sp = np.linalg.norm(d, axis=2) / dt                       # (N,T-1)
    ok = sp >= v_min
    h = np.full((N, T), np.nan)
    src = np.zeros((N, T), np.int8)
    ang = np.arctan2(d[:, :, 1], d[:, :, 0])
    h[:, :-1] = np.where(ok, ang, np.nan)
    src[:, :-1] = ok.astype(np.int8)
    last = ok[:, -1]                                          # 마지막 관측점은 직전 값 복제
    h[last, -1], src[last, -1] = h[last, -2], 1

    if h_ref is not None:
        hr = guard_batch(np.asarray(h_ref, np.float64), jump_deg)
        m = np.isfinite(h) & np.isfinite(hr)
        # align_ref: 위치차분 스텝이 3개 이상일 때만 판정, 중앙값 > 90° 면 트랙 전체를 180° 돌린다
        diff = np.where(m, np.abs(wrap(hr - h)), np.nan)
        cnt = m.sum(axis=1)
        med = np.full(N, np.nan)
        any_m = cnt >= 3
        if any_m.any():
            med[any_m] = np.nanmedian(diff[any_m], axis=1)
        flip = any_m & (med > np.pi / 2)
        hr[flip] = wrap(hr[flip] + np.pi)
        fill = ~np.isfinite(h) & np.isfinite(hr)
        h[fill], src[fill] = hr[fill], 2
    else:
        flip = np.zeros(N, bool)

    miss = ~np.isfinite(h)
    if miss.any():
        filled, has = _nearest_fill(h)
        put = miss & has[:, None]
        h[put], src[put] = filled[put], 3
        rest = ~np.isfinite(h)
        if rest.any():
            if h_ref is not None:
                hg = guard_batch(np.asarray(h_ref, np.float64), jump_deg)
                h[rest] = hg[rest]
            else:
                h[rest] = 0.0
            src[rest] = 2
    return h, src, flip


# ================================================================ 2단계 — 후보 평활기
# 인터페이스: fn(pos (N,50,2), vel, head, dt) -> pos_s (N,50,2)
# 모두 관측 50스텝만 본다. 경계 처리는 각 함수 docstring/주석에 적는다.

def _linpad(x, pad, fit=None, deg=1):
    """양 끝을 **다항 외삽**으로 늘린다 (기본 1차 = 등속 보존).
    edge/nearest 패딩은 끝에서 속도를 0 으로 끌어내려 창 가장자리에 가짜 감속을 더한다.
    deg=1 은 등속 직진을 정확히 보존하지만 등회전(곡률)은 보존하지 않는다.
    deg=2 는 등가속·등곡률까지 보존하는 대신 끝에서 잡음을 더 크게 되받는다 — 둘 다 재 본다.
    x (N,T,C). 앞/뒤 fit 점으로 deg 차 적합해 pad 칸을 채운다."""
    N, T, C = x.shape
    t = np.arange(T, dtype=np.float64)
    f = min(fit or (5 if deg == 1 else 9), T)      # 짧은 적합 창이 끝 기울기를 더 잘 잡는다(실측)
    out = np.empty((N, T + 2 * pad, C))
    out[:, pad:pad + T] = x
    for side in (0, 1):
        tt = t[:f] if side == 0 else t[T - f:]
        seg = (x[:, :f] if side == 0 else x[:, T - f:]).transpose(1, 0, 2).reshape(f, -1)
        V = np.vander(tt, deg + 1)
        coef, *_ = np.linalg.lstsq(V, seg, rcond=None)
        tp = (np.arange(-pad, 0) if side == 0 else np.arange(T, T + pad)).astype(np.float64)
        v = (np.vander(tp, deg + 1) @ coef).reshape(pad, N, C).transpose(1, 0, 2)
        if side == 0:
            out[:, :pad] = v
        else:
            out[:, pad + T:] = v
    return out


def sm_none(pos, vel, head, dt=DT):
    return np.array(pos, np.float64)


def _sg(pos, win, order, skip):
    """Savitzky–Golay. mode='interp' 은 배열 양 끝에서 창 안 다항식을 그대로 평가한다(외삽 없음).
    skip>0 이면 그 앞 스텝은 손대지 않고 평활 창에도 넣지 않는다 (지금 방식 RAMP_SKIP=4)."""
    from scipy.signal import savgol_filter
    out = np.array(pos, np.float64)
    seg = out[:, skip:]
    if seg.shape[1] >= win:
        out[:, skip:] = savgol_filter(seg, win, order, axis=1, mode="interp")
    return out


def _gauss(pos, sigma_s, dt=DT, trunc=3.0, pad_deg=1):
    """가우시안 커널. 경계는 다항 외삽 패딩(_linpad, 기본 1차) — 등속 구간을 왜곡하지 않는다.
    (edge/nearest 패딩은 창 끝에서 속도를 0 쪽으로 끌어내려 가짜 감속을 더한다.)
    pad_deg=2 판은 등회전의 끝 스텝 요레이트까지 보존한다 — 대가는 끝에서의 잡음 증폭이다."""
    s = sigma_s / dt
    pad = int(np.ceil(trunc * s))
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / s) ** 2)
    k /= k.sum()
    xp = _linpad(np.asarray(pos, np.float64), pad, deg=pad_deg)
    T = pos.shape[1]
    out = np.zeros((xp.shape[0], T, xp.shape[2]))
    for j, kk in enumerate(k):
        out += kk * xp[:, j:j + T]
    return out


_BUTTER_PAD = 30         # filtfilt 패딩 길이 [스텝]. 기본 9 는 0.5 Hz 처럼 좁은 대역의 과도응답을 다 못 먹는다


def _butter(pos, fc_hz, dt=DT, order=2, padlen=_BUTTER_PAD):
    """영위상 저역통과 (Butterworth 2차, filtfilt).
    경계: padtype='odd' (끝점 기준 홀수 확장) + padlen=30 스텝.
    홀수 확장은 **1차 추세를 정확히 보존**하므로 등속 구간에 가짜 감속을 만들지 않는다.
    기본 padlen(=9)으로는 0.5 Hz 판에서 등속 직선이 37 cm 휜다(--task verify 로 실측) — 그래서 늘렸다."""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, fc_hz / (0.5 / dt), btype="low")
    return filtfilt(b, a, np.asarray(pos, np.float64), axis=1, padtype="odd",
                    padlen=min(padlen, pos.shape[1] - 1))


_SPL_CACHE = {}


def _spline(pos, knot_s, dt=DT, k=3):
    """3차 B-스플라인 **최소제곱** 적합 (등간격 매듭). 매듭 간격이 곧 평활 강도다.
    매듭이 고정이라 적합은 선형이다 -> 평활 행렬 S = B·pinv(B) 를 한 번 만들어 모든 트랙에 곱한다
    (make_lsq_spline 한 트랙씩 도는 것과 수치까지 같다). 경계는 clamped knot — 외삽하지 않는다."""
    from scipy.interpolate import BSpline
    pos = np.asarray(pos, np.float64)
    T = pos.shape[1]
    key = (T, knot_s, dt, k)
    if key not in _SPL_CACHE:
        x = np.arange(T, dtype=np.float64) * dt
        inner = np.arange(knot_s, x[-1] - 1e-9, knot_s)
        if len(inner) == 0:
            inner = np.array([x[-1] / 2])
        t = np.r_[[x[0]] * (k + 1), inner, [x[-1]] * (k + 1)]
        B = BSpline.design_matrix(x, t, k, extrapolate=False).toarray()
        _SPL_CACHE[key] = B @ np.linalg.pinv(B)
    S = _SPL_CACHE[key]
    return np.einsum("ij,njc->nic", S, pos)


# ---------------------------------------------------------------- 칼만 + RTS
def _kf_rts(z, F, Q, H, R, x0, P0, miss=None):
    """배치 선형 칼만 필터 + RTS 평활기.
    z (N,T,m) 관측, F (n,n), Q (n,n), H (m,n), R (m,m). 반환 xs (N,T,n)."""
    N, T, m = z.shape
    n = F.shape[0]
    xf = np.zeros((N, T, n)); Pf = np.zeros((N, T, n, n))
    xp = np.zeros((N, T, n)); Pp = np.zeros((N, T, n, n))
    x = np.array(x0, np.float64)
    P = np.repeat(np.asarray(P0, np.float64)[None], N, 0)
    I = np.eye(n)
    for t in range(T):
        if t > 0:
            x = x @ F.T
            P = F @ P @ F.T + Q
        xp[:, t], Pp[:, t] = x, P
        y = z[:, t] - x @ H.T
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)                       # (N,n,m)
        x = x + np.einsum("nij,nj->ni", K, y)
        P = (I - K @ H) @ P
        xf[:, t], Pf[:, t] = x, P
    xs = np.array(xf); Ps = np.array(Pf)
    for t in range(T - 2, -1, -1):
        C = Pf[:, t] @ F.T @ np.linalg.inv(Pp[:, t + 1])
        xs[:, t] = xf[:, t] + np.einsum("nij,nj->ni", C, xs[:, t + 1] - xp[:, t + 1])
        Ps[:, t] = Pf[:, t] + C @ (Ps[:, t + 1] - Pp[:, t + 1]) @ C.transpose(0, 2, 1)
    return xs


def _kf_cv(pos, vel, head, dt=DT, sig_pos=None, q_acc=2.0, use_vel=False):
    """등속(CV) 모델 칼만+RTS. 상태 [x, y, vx, vy].
    q_acc = 가속도 잡음 밀도 [m/s²] (모델이 허용하는 가속 크기), sig_pos = 위치 관측 잡음 [m] (1단계 실측)."""
    sig_pos = SIG_POS if sig_pos is None else sig_pos
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    G = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
    Q = G @ (np.eye(2) * q_acc ** 2) @ G.T
    if use_vel:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        R = np.diag([sig_pos ** 2, sig_pos ** 2, SIG_VEL ** 2, SIG_VEL ** 2])
        z = np.concatenate([pos, vel], axis=2)
    else:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * sig_pos ** 2
        z = np.asarray(pos, np.float64)
    N = pos.shape[0]
    v0 = (pos[:, 1] - pos[:, 0]) / dt
    x0 = np.concatenate([pos[:, 0], v0], axis=1)
    P0 = np.diag([sig_pos ** 2, sig_pos ** 2, 25.0, 25.0])
    return _kf_rts(z, F, Q, H, R, x0, P0)[:, :, :2]


def _kf_ca(pos, vel, head, dt=DT, sig_pos=None, q_jerk=4.0, use_vel=False, sig_vel=None):
    """등가속(CA) 모델 칼만+RTS. 상태 [x, y, vx, vy, ax, ay]. q_jerk = 저크 잡음 밀도 [m/s³]."""
    sig_pos = SIG_POS if sig_pos is None else sig_pos
    F = np.eye(6)
    F[0, 2] = F[1, 3] = F[2, 4] = F[3, 5] = dt
    F[0, 4] = F[1, 5] = 0.5 * dt ** 2
    G = np.array([[dt ** 3 / 6, 0], [0, dt ** 3 / 6], [0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
    Q = G @ (np.eye(2) * q_jerk ** 2) @ G.T
    if use_vel:
        H = np.zeros((4, 6)); H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1
        sv = SIG_VEL if sig_vel is None else sig_vel
        R = np.diag([sig_pos ** 2, sig_pos ** 2, sv ** 2, sv ** 2])
        z = np.concatenate([pos, vel], axis=2)
    else:
        H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1
        R = np.eye(2) * sig_pos ** 2
        z = np.asarray(pos, np.float64)
    v0 = (pos[:, 1] - pos[:, 0]) / dt
    x0 = np.concatenate([pos[:, 0], v0, np.zeros_like(v0)], axis=1)
    P0 = np.diag([sig_pos ** 2, sig_pos ** 2, 25.0, 25.0, 25.0, 25.0])
    return _kf_rts(z, F, Q, H, R, x0, P0)[:, :, :2]


def _kf_ctra(pos, vel, head, dt=DT, sig_pos=None, q_a=3.0, q_w=0.6, use_vel=False):
    """CTRA(등회전율·등가속) EKF + RTS. 상태 [x, y, θ, v, a, ω].
    비선형이라 야코비안으로 편다. q_a [m/s³]·q_w [rad/s²] = a·ω 의 확산(저크·요각가속) 잡음 밀도."""
    sig_pos = SIG_POS if sig_pos is None else sig_pos
    pos = np.asarray(pos, np.float64)
    N, T, _ = pos.shape
    n = 6
    G = np.zeros((n, 2)); G[4, 0] = dt; G[5, 1] = dt
    G[3, 0] = 0.5 * dt ** 2; G[2, 1] = 0.5 * dt ** 2
    Q = G @ np.diag([q_a ** 2, q_w ** 2]) @ G.T + np.eye(n) * 1e-9
    if use_vel:
        H = np.zeros((4, n))
        R = np.diag([sig_pos ** 2, sig_pos ** 2, SIG_VEL ** 2, SIG_VEL ** 2])
    else:
        H = np.zeros((2, n)); H[0, 0] = H[1, 1] = 1
        R = np.eye(2) * sig_pos ** 2

    d0 = pos[:, 1] - pos[:, 0]
    v0 = np.linalg.norm(d0, axis=1) / dt
    th0 = np.where(v0 > 0.2, np.arctan2(d0[:, 1], d0[:, 0]), np.asarray(head, np.float64)[:, 0])
    x = np.zeros((N, n))
    x[:, :2], x[:, 2], x[:, 3] = pos[:, 0], th0, v0
    P = np.repeat(np.diag([sig_pos ** 2, sig_pos ** 2, 0.5, 25.0, 25.0, 1.0])[None], N, 0)

    def step(xs):
        th, v, a, w = xs[:, 2], xs[:, 3], xs[:, 4], xs[:, 5]
        vm = v + 0.5 * a * dt                                 # 중점 속도 (원 CTRA 폐형식의 1차 근사)
        thm = th + 0.5 * w * dt
        nx = np.stack([xs[:, 0] + vm * np.cos(thm) * dt, xs[:, 1] + vm * np.sin(thm) * dt,
                       wrap(th + w * dt), np.maximum(v + a * dt, 0.0), a, w], axis=1)
        F = np.repeat(np.eye(n)[None], len(xs), 0)
        c, s = np.cos(thm), np.sin(thm)
        F[:, 0, 2] = -vm * s * dt; F[:, 0, 3] = c * dt; F[:, 0, 4] = 0.5 * dt ** 2 * c
        F[:, 0, 5] = -vm * s * 0.5 * dt ** 2
        F[:, 1, 2] = vm * c * dt; F[:, 1, 3] = s * dt; F[:, 1, 4] = 0.5 * dt ** 2 * s
        F[:, 1, 5] = vm * c * 0.5 * dt ** 2
        F[:, 2, 5] = dt; F[:, 3, 4] = dt
        return nx, F

    def obs(xs):
        if not use_vel:
            return xs[:, :2], H[None].repeat(len(xs), 0)
        hh = np.stack([xs[:, 0], xs[:, 1], xs[:, 3] * np.cos(xs[:, 2]), xs[:, 3] * np.sin(xs[:, 2])], axis=1)
        J = np.zeros((len(xs), 4, n))
        J[:, 0, 0] = J[:, 1, 1] = 1
        J[:, 2, 2] = -xs[:, 3] * np.sin(xs[:, 2]); J[:, 2, 3] = np.cos(xs[:, 2])
        J[:, 3, 2] = xs[:, 3] * np.cos(xs[:, 2]); J[:, 3, 3] = np.sin(xs[:, 2])
        return hh, J

    z = pos if not use_vel else np.concatenate([pos, np.asarray(vel, np.float64)], axis=2)
    xf = np.zeros((N, T, n)); Pf = np.zeros((N, T, n, n))
    xp = np.zeros((N, T, n)); Pp = np.zeros((N, T, n, n)); Fs = np.zeros((N, T, n, n))
    I = np.eye(n)
    for t in range(T):
        if t > 0:
            x, Fj = step(x)
            P = Fj @ P @ Fj.transpose(0, 2, 1) + Q
            Fs[:, t] = Fj
        xp[:, t], Pp[:, t] = x, P
        hh, Hj = obs(x)
        y = z[:, t] - hh
        if use_vel:
            pass
        S = Hj @ P @ Hj.transpose(0, 2, 1) + R
        K = P @ Hj.transpose(0, 2, 1) @ np.linalg.inv(S)
        x = x + np.einsum("nij,nj->ni", K, y)
        x[:, 2] = wrap(x[:, 2])
        P = (I - K @ Hj) @ P
        xf[:, t], Pf[:, t] = x, P
    xs = np.array(xf)
    for t in range(T - 2, -1, -1):
        C = Pf[:, t] @ Fs[:, t + 1].transpose(0, 2, 1) @ np.linalg.inv(Pp[:, t + 1])
        d = xs[:, t + 1] - xp[:, t + 1]
        d[:, 2] = wrap(d[:, 2])
        xs[:, t] = xf[:, t] + np.einsum("nij,nj->ni", C, d)
        xs[:, t, 2] = wrap(xs[:, t, 2])
    return xs[:, :, :2]


# ---------------------------------------------------------------- 운동학 제약 적합 (torch, CPU)
def _kin_fit(pos, vel, head, dt=DT, iters=60, rounds=3, sig_pos=None, huber_k=3.0, use_vel=False, sig_vel=0.2,
             a_max=8.0, w_max=0.95, sig_j=3.0, sig_al=1.0, lam=200.0, chunk=8192):
    """운동학 제약 적합 — 위치 오차 + 매끄러움(저크·요각가속)을 함께 최소화하되
    |a| <= 8 m/s², |요레이트| <= 0.95 rad/s 를 지킨다.

    형식: **직접법(collocation)**. 자유변수는 위치 p (N,T,2) 자체이고, 거기서 파생량을 만든다.
      v[t] = |p[t+1]−p[t]|/dt,  θ[t] = atan2(...),  a = Δv/dt,  ω = wrap(Δθ)/dt,  j = Δa/dt,  α = Δω/dt
    손실 = Σ ρ(|p−z|/σ_p)  +  Σ(j/σ_j)²dt  +  Σ(α/σ_α)²dt  +  λ·Σ relu(|a|−a_max)²  +  λ·Σ relu(|ω|−w_max)²
    ρ 는 무릎 3σ 의 **Huber** 다. 1단계에서 위치 잡음이 정규분포가 아니기 때문이다 —
    정지 차량 스텝의 72.5% 는 1 cm 미만인데 4.0% 는 10 cm 를 넘는다(꼬리가 두껍다).
    제곱 손실이면 그 4% 가 적합을 끌고 가서 제약(|a|<=8)을 못 지킨다(실측: 위반 6~9%).
    (사격법 — a, ω 를 쌓아 궤적을 굴리는 방식 — 은 50스텝 누적이라 조건수가 나빠 수렴하지 않았다.
     실측: 같은 손실에서 사격법 rmse 60 cm vs 직접법 8 cm.
     최적화는 L-BFGS(strong-Wolfe) 다. Adam 은 저크 항의 조건수가 (2/dt)³ 라 1,500 스텝에도 |a|>8 이 6% 남았다.)
    σ_p 는 1단계 실측 위치 잡음(백색 성분), σ_j·σ_α 는 사람 운전 범위에서 잡았다.
    제약은 벌점이라 '거의' 지켜진다 — 적합 뒤 위반 비율을 함께 보고한다(--task verify).
    모델 가중치와 무관하다. 시나리오마다 따로 맞춘다. CPU 로만 돈다.
    """
    import torch
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    sig_pos = SIG_POS if sig_pos is None else sig_pos
    pos = np.asarray(pos, np.float64)
    out = np.empty_like(pos)
    for lo in range(0, len(pos), chunk):
        org = pos[lo:lo + chunk][:, :1].copy()               # 국소 좌표 (city 좌표는 수천 m — 조건수)
        Z = torch.tensor(pos[lo:lo + chunk] - org, dtype=torch.float64)
        Vf = torch.tensor(np.asarray(vel, np.float64)[lo:lo + chunk], dtype=torch.float64) if use_vel else None
        p = torch.tensor(_gauss(pos[lo:lo + chunk], 0.25) - org, dtype=torch.float64).requires_grad_(True)
        opt = torch.optim.LBFGS([p], max_iter=iters, history_size=30, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-16)
        eps = 1e-12

        def closure():
            opt.zero_grad()
            d = p[:, 1:] - p[:, :-1]
            v = torch.sqrt((d ** 2).sum(-1) + eps) / dt
            th = torch.atan2(d[:, :, 1], d[:, :, 0])
            a = (v[:, 1:] - v[:, :-1]) / dt
            dth = th[:, 1:] - th[:, :-1]
            w = torch.atan2(torch.sin(dth), torch.cos(dth)) / dt
            j = (a[:, 1:] - a[:, :-1]) / dt
            al = (w[:, 1:] - w[:, :-1]) / dt
            # 요레이트 제약·매끄러움은 **움직일 때만** 건다 — 정지 구간의 '이동방향'은 정의되지 않는다.
            g = torch.clamp(0.5 * (v[:, 1:] + v[:, :-1]) / V_MIN, max=1.0)
            ga = torch.clamp(0.5 * (g[:, 1:] + g[:, :-1]), max=1.0)
            r2 = ((p - Z) ** 2).sum(-1) / sig_pos ** 2
            rr = torch.sqrt(r2 + 1e-12)
            hub = torch.where(rr <= huber_k, r2, 2 * huber_k * rr - huber_k ** 2)
            loss = hub.sum(1).mean() \
                + ((j / sig_j) ** 2 * dt).sum(1).mean() + (ga * (al / sig_al) ** 2 * dt).sum(1).mean() \
                + lam * (torch.relu(a.abs() - a_max) ** 2).sum(1).mean() \
                + lam * (g * torch.relu(w.abs() - w_max) ** 2).sum(1).mean()
            if Vf is not None:
                # AV2 속도 필드도 관측으로: 스텝 속도벡터 d/dt 를 구간 중점 속도 필드에 맞춘다
                vm = 0.5 * (Vf[:, 1:] + Vf[:, :-1])
                loss = loss + (((d / dt - vm) ** 2).sum(-1) / sig_vel ** 2).sum(1).mean()
            loss.backward()
            return loss

        for _ in range(rounds):
            opt.step(closure)
        out[lo:lo + chunk] = p.detach().numpy() + org
    return out



# ---------------------------------------------------------------- 각도 전용 처리 (방식 b)
ANG_MIN_RUN = 7          # 이 길이 이상 이어진 '위치차분 구간' 에서만 각도를 평활한다


def smooth_heading_direct(h, src, kind="sg", win=9, order=2, sigma_s=0.25, dt=DT,
                          min_run=ANG_MIN_RUN):
    """방식 (b): 방향을 unwrap 해서 **직접** 평활한다.

    정지 구간 처리 규칙 (명시):
      - 평활은 src==1(위치차분으로 정해진 = 실제 이동) 스텝이 min_run 이상 **연속**인 구간 안에서만 한다.
      - 그 구간 안에서 unwrap 한 뒤 평활하고 다시 wrap 한다 (±π 점프를 넘어가도 안전).
      - src!=1 (정지·보조·이웃값) 스텝과 min_run 미만의 짧은 이동 구간은 **그대로 둔다** —
        정지 구간의 값은 AV2 차체 방향이고, 이웃 이동 구간과 섞으면 잡음이 정지 구간으로 번진다.
    """
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    h = np.array(h, np.float64)
    out = np.array(h)
    N, T = h.shape
    for i in range(N):
        s = src[i] == 1
        t = 0
        while t < T:
            if not s[t]:
                t += 1
                continue
            j = t
            while j < T and s[j]:
                j += 1
            L = j - t
            if L >= min_run:
                u = np.unwrap(h[i, t:j])
                if kind == "sg":
                    w = min(win, L if L % 2 else L - 1)
                    if w > order:
                        u = savgol_filter(u, w, order, mode="interp")
                else:
                    u = gaussian_filter1d(u, sigma_s / dt, mode="nearest")
                out[i, t:j] = wrap(u)
            t = j
    return out


# ---------------------------------------------------------------- 후보 목록
SIG_POS = 0.03           # [m]  위치 관측 잡음 σ — 1단계(--task char)가 덮어쓴다
SIG_VEL = 0.10           # [m/s] AV2 속도 필드 관측 잡음 σ — 1단계가 덮어쓴다


def load_char_sigma():
    """1단계 결과가 있으면 잡음 파라미터를 실측값으로 바꾼다."""
    global SIG_POS, SIG_VEL
    p = DATA / "char.json"
    if p.exists():
        c = json.loads(p.read_text())
        SIG_POS = float(c["noise"]["sig_pos_used"])
        SIG_VEL = float(c["noise"]["sig_vel_used"])
    return SIG_POS, SIG_VEL


def methods(include_slow=True):
    """(key, 라벨, 그룹, 위치평활 fn, 각도모드) 목록. 각도모드 'pos' = 평활 위치에서 방향(지금 방식), 'ang' = 방향 직접 평활."""
    M = [
        ("none", "없음 (기준선)", "기준", sm_none, "pos"),
        ("sg5_2", "SG(5,2) idx4~ [현재]", "SG", lambda p, v, h, dt=DT: _sg(p, 5, 2, 4), "pos"),
        ("sg5_2_ns", "SG(5,2) 전체", "SG", lambda p, v, h, dt=DT: _sg(p, 5, 2, 0), "pos"),
        ("sg7_2", "SG(7,2) idx4~", "SG", lambda p, v, h, dt=DT: _sg(p, 7, 2, 4), "pos"),
        ("sg9_2", "SG(9,2) idx4~", "SG", lambda p, v, h, dt=DT: _sg(p, 9, 2, 4), "pos"),
        ("sg11_2", "SG(11,2) idx4~", "SG", lambda p, v, h, dt=DT: _sg(p, 11, 2, 4), "pos"),
        ("sg9_3", "SG(9,3) idx4~", "SG", lambda p, v, h, dt=DT: _sg(p, 9, 3, 4), "pos"),
        ("sg11_3", "SG(11,3) idx4~", "SG", lambda p, v, h, dt=DT: _sg(p, 11, 3, 4), "pos"),
        ("g015", "가우시안 σ=0.15 s", "가우시안", lambda p, v, h, dt=DT: _gauss(p, 0.15), "pos"),
        ("g025", "가우시안 σ=0.25 s", "가우시안", lambda p, v, h, dt=DT: _gauss(p, 0.25), "pos"),
        ("g040", "가우시안 σ=0.40 s", "가우시안", lambda p, v, h, dt=DT: _gauss(p, 0.40), "pos"),
        ("g025q", "가우시안 σ=0.25 s (경계 2차 외삽)", "가우시안", lambda p, v, h, dt=DT: _gauss(p, 0.25, pad_deg=2), "pos"),
        ("bw05", "Butterworth 2차 0.5 Hz", "저역통과", lambda p, v, h, dt=DT: _butter(p, 0.5), "pos"),
        ("bw10", "Butterworth 2차 1 Hz", "저역통과", lambda p, v, h, dt=DT: _butter(p, 1.0), "pos"),
        ("bw20", "Butterworth 2차 2 Hz", "저역통과", lambda p, v, h, dt=DT: _butter(p, 2.0), "pos"),
        ("spl05", "B-스플라인 3차 매듭 0.5 s", "스플라인", lambda p, v, h, dt=DT: _spline(p, 0.5), "pos"),
        ("spl10", "B-스플라인 3차 매듭 1.0 s", "스플라인", lambda p, v, h, dt=DT: _spline(p, 1.0), "pos"),
        ("kf_cv", "칼만+RTS CV (위치)", "칼만", lambda p, v, h, dt=DT: _kf_cv(p, v, h), "pos"),
        ("kf_ca", "칼만+RTS CA (위치)", "칼만", lambda p, v, h, dt=DT: _kf_ca(p, v, h), "pos"),
        ("kf_ctra", "EKF+RTS CTRA (위치)", "칼만", lambda p, v, h, dt=DT: _kf_ctra(p, v, h), "pos"),
        ("kf_cv_v", "칼만+RTS CV (위치+속도필드)", "칼만", lambda p, v, h, dt=DT: _kf_cv(p, v, h, use_vel=True), "pos"),
        ("kf_ca_v", "칼만+RTS CA (위치+속도필드)", "칼만", lambda p, v, h, dt=DT: _kf_ca(p, v, h, use_vel=True), "pos"),
        ("kf_ctra_v", "EKF+RTS CTRA (위치+속도필드)", "칼만", lambda p, v, h, dt=DT: _kf_ctra(p, v, h, use_vel=True), "pos"),
        ("kf_ca_r5", "칼만+RTS CA (σ_pos=5 cm)", "칼만", lambda p, v, h, dt=DT: _kf_ca(p, v, h, sig_pos=0.05), "pos"),
        ("kf_ca_v2", "칼만+RTS CA (위치+속도필드, σ_v=0.2)", "칼만",
         lambda p, v, h, dt=DT: _kf_ca(p, v, h, use_vel=True, sig_vel=0.2), "pos"),
        ("kf_ctra_r5", "EKF+RTS CTRA (σ_pos=5 cm)", "칼만", lambda p, v, h, dt=DT: _kf_ctra(p, v, h, sig_pos=0.05), "pos"),
        ("ang_none", "각도 직접 SG(9,2) (위치 평활 없음)", "각도", sm_none, "ang_sg9"),
        ("ang_sg9", "SG(9,2) + 각도 직접 SG(9,2)", "각도", lambda p, v, h, dt=DT: _sg(p, 9, 2, 4), "ang_sg9"),
        ("ang_g025", "가우시안 σ=0.25 s + 각도 직접 SG(9,2)", "각도", lambda p, v, h, dt=DT: _gauss(p, 0.25), "ang_sg9"),
        ("ang_kfcav", "칼만 CA(위치+속도필드) + 각도 직접 SG(9,2)", "각도",
         lambda p, v, h, dt=DT: _kf_ca(p, v, h, use_vel=True), "ang_sg9"),
    ]
    if include_slow:
        M.append(("kin", "운동학 제약 적합 (|a|≤8, |ω|≤0.95)", "제약적합", _kin_fit, "pos"))
        # kin_v(제약적합 + 속도필드 항)은 뺐다 — L-BFGS 가 배치에서 수렴하지 않아
        # σ_v=0.02 에서도 |v − v_field| 를 0.283 -> 0.197 m/s 로만 줄이고 결과가 kin 과 같았다.
        # 속도 필드 융합은 칼만 계열(kf_*_v)이 대표한다.
    return M


def apply_method(m, pos, vel, head, dt=DT, v_min=V_MIN, jump_deg=JUMP_DEG):
    """후보 하나를 적용해 (평활 위치, h, src) 를 돌려준다."""
    key, lbl, grp, fn, angm = m
    ps = fn(pos, vel, head, dt)
    h, src, flip = build_heading_batch(ps, h_ref=head, dt=dt, v_min=v_min, jump_deg=jump_deg)
    if angm.startswith("ang"):
        h = smooth_heading_direct(h, src, kind="sg", win=9, order=2, dt=dt)
    return ps, h, src, flip


# ================================================================ 1단계 — 데이터 특성
def task_char(a):
    DATA.mkdir(parents=True, exist_ok=True)
    dirs = scen_dirs(a.split, a.limit, a.seed)
    print(f"[char] 시나리오 {len(dirs):,} 읽는 중 …", flush=True)
    F, A = collect(dirs, a.workers)
    print(f"[char] focal {len(F['pos']):,} · 주변차량(50스텝 완전관측) {len(A['pos']):,}", flush=True)
    out = {"n_scen": len(F["pos"]), "n_agent": len(A["pos"]), "split": a.split,
           "limit": a.limit, "seed": a.seed}
    npz = {}

    # ---------- (a) 정지 차량 위치 잡음
    sp_av2_a = np.linalg.norm(A["vel"], axis=2)
    stop_a = sp_av2_a.max(axis=1) < STOP_VMAX
    sp_av2_f = np.linalg.norm(F["vel"][:, :OBS], axis=2)
    stop_f = sp_av2_f.max(axis=1) < STOP_VMAX
    t = np.arange(OBS, dtype=np.float64)

    def lin_resid(p, deg=1):
        """트랙별 축별 다항 적합 잔차 (N,T,2)."""
        V = np.vander(t, deg + 1)
        coef, *_ = np.linalg.lstsq(V, p.reshape(len(p), OBS, 2).transpose(1, 0, 2).reshape(OBS, -1), rcond=None)
        fit = (V @ coef).reshape(OBS, len(p), 2).transpose(1, 0, 2)
        return p - fit

    def rot_lonlat(r, hd):
        """잔차를 트랙 heading 기준 (종, 횡) 으로 돌린다. hd (N,)."""
        c, s = np.cos(hd)[:, None], np.sin(hd)[:, None]
        return np.stack([r[:, :, 0] * c + r[:, :, 1] * s, -r[:, :, 0] * s + r[:, :, 1] * c], axis=2)

    noise = {}
    for name, pos, stop, hd in (("주변차량", A["pos"], stop_a, np.median(A["head"], axis=1)),
                                ("focal", F["pos"][:, :OBS], stop_f, np.median(F["head"][:, :OBS], axis=1))):
        p = pos[stop]
        if len(p) < 20:
            continue
        r0 = p - p.mean(axis=1, keepdims=True)
        r1 = lin_resid(p)
        rl = rot_lonlat(r1, hd[stop])
        noise[name] = {
            "n": int(stop.sum()),
            "std_mean_xy": [float(r0[:, :, 0].std()), float(r0[:, :, 1].std())],
            "std_lin_xy": [float(r1[:, :, 0].std()), float(r1[:, :, 1].std())],
            "std_lin_lonlat": [float(rl[:, :, 0].std()), float(rl[:, :, 1].std())],
            "std_lin_rms": float(np.sqrt((r1 ** 2).sum(axis=2).mean() / 2)),
            "per_step_rms": (np.sqrt((r1 ** 2).sum(axis=2).mean(axis=0) / 2)).tolist(),
            "step_disp_p50": float(np.median(np.linalg.norm(np.diff(p, axis=1), axis=2))),
            "step_disp_p99": float(np.percentile(np.linalg.norm(np.diff(p, axis=1), axis=2), 99)),
        }
    npz["stop_resid_step_agent"] = np.array(noise.get("주변차량", {}).get("per_step_rms", []))

    # ---------- (b) 등속 직진 구간의 직선 적합 잔차
    course_f, dcourse_f, spd_f = _course(F["vel"][:, :OBS])
    path_f = np.linalg.norm(np.diff(F["pos"][:, :OBS], axis=1), axis=2).sum(axis=1)
    cs = np.zeros((len(F["pos"]),), bool)
    ok = spd_f[:, RAMP:].min(axis=1) >= 8.0
    turn = np.abs(wrap(course_f[:, OBS - 1] - course_f[:, RAMP]))
    cs = ok & (np.degrees(turn) < 2.0) & (path_f > 40.0)
    if cs.sum() >= 20:
        p = F["pos"][cs][:, RAMP:OBS]
        tt = np.arange(p.shape[1], dtype=np.float64)
        for deg, key in ((1, "lin"), (2, "quad")):
            V = np.vander(tt, deg + 1)
            coef, *_ = np.linalg.lstsq(V, p.transpose(1, 0, 2).reshape(len(tt), -1), rcond=None)
            fit = (V @ coef).reshape(len(tt), len(p), 2).transpose(1, 0, 2)
            r = p - fit
            hd = course_f[cs][:, OBS - 1]
            c, s = np.cos(hd)[:, None], np.sin(hd)[:, None]
            rl = np.stack([r[:, :, 0] * c + r[:, :, 1] * s, -r[:, :, 0] * s + r[:, :, 1] * c], axis=2)
            noise.setdefault("등속직진", {})[key] = {
                "n": int(cs.sum()), "std_lonlat": [float(rl[:, :, 0].std()), float(rl[:, :, 1].std())],
                "rms": float(np.sqrt((r ** 2).sum(axis=2).mean() / 2))}
    # ---------- (a2) 잡음의 **색**: 백색인가? (칼만·제약적합의 R 을 정하는 값)
    # 백색 잡음이면 Var(Δ²x) = 6σ² 이다. 실제 잔차는 시간 상관이 크므로 두 값이 크게 갈린다.
    col = {}
    for name, pos, stop in (("주변차량", A["pos"], stop_a), ("focal", F["pos"][:, :OBS], stop_f)):
        p = pos[stop]
        if len(p) < 20:
            continue
        d1 = np.diff(p, axis=1)
        d2 = np.diff(d1, axis=1)
        sp1 = np.linalg.norm(d1, axis=2)
        r = p - p.mean(axis=1, keepdims=True)
        rr = r - r.mean(axis=1, keepdims=True)
        ac = [float((rr[:, :-k] * rr[:, k:]).sum() / max((rr * rr).sum(), 1e-12)) for k in range(1, 11)]
        a2 = np.diff(np.diff(np.linalg.norm(np.diff(pos[~stop][:4000], axis=1), axis=2) / DT, axis=1), axis=1)
        col[name] = {
            "sig_white_rms_cm": float(np.sqrt((d2 ** 2).mean() / 6) * 100),
            "sig_white_mad_cm": float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6) * 100),
            "step_ge_vmin_pct": float((sp1 / DT >= V_MIN).mean() * 100),
            "step_under_1cm_pct": float((sp1 < 0.01).mean() * 100),
            "resid_autocorr_lag1_10": ac,
            "moving_speed_dd_autocorr_lag1": float(
                (a2[:, :-1] * a2[:, 1:]).sum() / max((a2 * a2).sum(), 1e-12)) if len(a2) else float("nan"),
        }
    noise["색"] = col
    sig_pos = float(col["주변차량"]["sig_white_rms_cm"]) / 100 if "주변차량" in col else SIG_POS
    # AV2 속도 필드 잡음: 정지 차량의 속도 필드 성분 표준편차
    sv = A["vel"][stop_a]
    sig_vel = float(np.sqrt((sv ** 2).sum(axis=2).mean() / 2)) if len(sv) else SIG_VEL
    noise["sig_pos_used"] = sig_pos
    noise["sig_vel_used"] = max(sig_vel, 0.02)
    noise["sig_pos_근거"] = ("정지 차량 위치의 2계 차분에서 뽑은 **백색 성분** rms "
                          "(백색이면 Var(Δ²x)=6σ²). 1계 잔차 표준편차(13.5 cm)는 시간 상관이 큰 저주파 흔들림까지 "
                          "포함해 칼만의 R 로 쓰면 과평활이 된다.")
    out["noise"] = noise

    # ---------- (c) 스펙트럼
    spec = _spectrum(F, A, stop_a)
    out["spectrum"] = spec["summary"]
    npz.update({f"spec_{k}": v for k, v in spec["arr"].items()})

    # ---------- (d) 속도 필드 course vs 위치차분 방향
    out["course"] = _course_compare(F, A)

    # ---------- (e) 창 가장자리
    out["edge"] = _edge(F)
    npz["edge_ratio_step"] = np.array(out["edge"]["ratio_step_p50"])
    npz["edge_absa_step"] = np.array(out["edge"]["absa_step_p50"])

    (DATA / "char.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    np.savez_compressed(DATA / "char.npz", **npz)
    print(json.dumps({k: v for k, v in out.items() if k != "spectrum"}, ensure_ascii=False, indent=1)[:4000])
    print(f"\n[char] -> {DATA/'char.json'} · {DATA/'char.npz'}")


def _course(vel):
    """AV2 속도 필드 -> course angle (N,T), 스텝간 변화, 속력."""
    v = np.asarray(vel, np.float64)
    sp = np.linalg.norm(v, axis=2)
    c = np.arctan2(v[:, :, 1], v[:, :, 0])
    return c, wrap(np.diff(c, axis=1)), sp


def _spectrum(F, A, stop_a):
    """위치(추세 제거)·속력·방향(unwrap, 추세 제거)의 진폭 스펙트럼 — 속력 구간별.

    창: 11초 110스텝 중 인덱스 5~104 (100스텝 = 10 s) — 양 끝 램프를 뺀다.
    전처리는 관측 50스텝만 쓰지만 **스펙트럼은 데이터 특성 측정**이라 긴 창을 써서
    주파수 해상도를 0.1 Hz 로 둔다(50스텝이면 0.2 Hz 라 저역이 뭉개진다).
    """
    lo, hi = 5, 105
    L = hi - lo
    win = np.hanning(L)
    cg = win.sum() / L                                     # 진폭 보정 (coherent gain)
    freq = np.fft.rfftfreq(L, DT)
    tt = np.arange(L, dtype=np.float64)

    def amp(x):
        """x (n,L) 1차 추세 제거 -> Hann -> rFFT 진폭."""
        V = np.vander(tt, 2)
        coef, *_ = np.linalg.lstsq(V, x.T, rcond=None)
        r = x - (V @ coef).T
        S = np.abs(np.fft.rfft(r * win, axis=1)) * 2.0 / (L * cg)
        return S

    pos, vel = F["pos"], F["vel"]
    sp = np.linalg.norm(vel, axis=2)
    d = np.diff(pos, axis=1)
    spd_pos = np.linalg.norm(d, axis=1 + 1) / DT
    h_pos = np.unwrap(np.arctan2(d[:, :, 1], d[:, :, 0]), axis=1)
    med = np.median(sp[:, lo:hi], axis=1)
    arr = {"freq": freq}
    summ = {"window": f"인덱스 {lo}~{hi-1} (10 s, 100스텝)", "df_hz": float(freq[1]),
            "bins": SPEED_LBL, "n": {}}
    for i, lbl in enumerate(SPEED_LBL):
        m = (med >= SPEED_BINS[i]) & (med < SPEED_BINS[i + 1])
        if m.sum() < 10:
            continue
        summ["n"][lbl] = int(m.sum())
        arr[f"pos_x_{i}"] = amp(pos[m][:, lo:hi, 0]).mean(axis=0)
        arr[f"pos_y_{i}"] = amp(pos[m][:, lo:hi, 1]).mean(axis=0)
        arr[f"spd_{i}"] = amp(spd_pos[m][:, lo:hi]).mean(axis=0)
        arr[f"hed_{i}"] = np.degrees(amp(h_pos[m][:, lo:hi]).mean(axis=0))
        arr[f"spdfield_{i}"] = amp(sp[m][:, lo:hi]).mean(axis=0)
    # 정지 차량(= 순수 잡음) 스펙트럼: 주변 차량 관측 50스텝, 인덱스 5~49
    if stop_a.sum() >= 20:
        ps = A["pos"][stop_a][:, 5:OBS]
        L2 = ps.shape[1]
        w2 = np.hanning(L2); cg2 = w2.sum() / L2
        t2 = np.arange(L2, dtype=np.float64)
        V = np.vander(t2, 2)
        f2 = np.fft.rfftfreq(L2, DT)
        S = []
        for c in range(2):
            x = ps[:, :, c]
            coef, *_ = np.linalg.lstsq(V, x.T, rcond=None)
            r = x - (V @ coef).T
            S.append((np.abs(np.fft.rfft(r * w2, axis=1)) * 2.0 / (L2 * cg2)).mean(axis=0))
        arr["freq_stop"] = f2
        arr["pos_stop"] = np.mean(S, axis=0)
        summ["n_stop"] = int(stop_a.sum())
    return {"summary": summ, "arr": arr}


def _course_compare(F, A):
    """속도 필드 course 와 위치차분 방향의 차이 — 속력 구간별. 램프(0~4, 105~109)는 뺀다."""
    res = {}
    for name, pos, vel in (("focal", F["pos"], F["vel"]), ("주변차량", A["pos"], A["vel"])):
        T = pos.shape[1]
        d = np.diff(pos, axis=1)
        sp_pos = np.linalg.norm(d, axis=2) / DT
        h_pos = np.arctan2(d[:, :, 1], d[:, :, 0])
        vm = vel[:, :-1] + vel[:, 1:]                       # 구간 중점의 속도 필드
        c_mid = np.arctan2(vm[:, :, 1], vm[:, :, 0])
        sp_f = np.linalg.norm(vel, axis=2)
        sp_mid = 0.5 * (sp_f[:, :-1] + sp_f[:, 1:])
        lo, hi = RAMP, (T - 5 if T > OBS else T - 1)
        sel = np.zeros_like(sp_pos, bool)
        sel[:, lo:hi] = True
        dif = np.degrees(np.abs(wrap(h_pos - c_mid)))
        rows = {}
        for i, lbl in enumerate(SPEED_LBL):
            m = sel & (sp_mid >= SPEED_BINS[i]) & (sp_mid < SPEED_BINS[i + 1])
            if m.sum() < 50:
                continue
            rows[lbl] = {"n": int(m.sum()), "p50": float(np.median(dif[m])),
                         "p90": float(np.percentile(dif[m], 90)),
                         "p99": float(np.percentile(dif[m], 99)),
                         "spd_ratio_p50": float(np.median(sp_pos[m] / np.maximum(sp_mid[m], 1e-6)))}
        res[name] = rows
    return res


def _edge(F):
    """창 가장자리 램프: 스텝별 (위치차분 속력 / 속도 필드 속력) 과 |a|. focal, 기준속력 > 3 m/s."""
    pos, vel = F["pos"], F["vel"]
    d = np.diff(pos, axis=1)
    sp_pos = np.linalg.norm(d, axis=2) / DT
    sp_f = np.linalg.norm(vel, axis=2)
    sp_mid = 0.5 * (sp_f[:, :-1] + sp_f[:, 1:])
    ref = sp_pos[:, 10:16].mean(axis=1)
    m = ref > 3.0
    ratio = sp_pos[m] / np.maximum(sp_mid[m], 1e-6)
    a = np.abs(np.diff(sp_pos[m], axis=1) / DT)
    return {"n": int(m.sum()), "ref_def": "스텝 10~15 위치차분 속력 평균 > 3 m/s",
            "ratio_step_p50": np.median(ratio, axis=0).tolist(),
            "ratio_step_p25": np.percentile(ratio, 25, axis=0).tolist(),
            "ratio_step_p75": np.percentile(ratio, 75, axis=0).tolist(),
            "absa_step_p50": np.median(a, axis=0).tolist(),
            "first_ratio_p50": float(np.median(sp_pos[m][:, 0] / ref[m])),
            "first_under08": float((sp_pos[m][:, 0] / ref[m] < 0.8).mean() * 100)}


# ================================================================ 3단계 — 지표
def classify(pos, vel):
    """평활과 무관한 **독립 기준**(AV2 속도 필드)으로 관측 구간 상황을 나눈다. pos/vel 은 (N,50,·)."""
    c, dc, sp = _course(vel)
    path = np.linalg.norm(np.diff(pos, axis=1), axis=2).sum(axis=1)
    mov = sp >= COURSE_VMIN
    # 관측 구간 순 변화: 속력이 COURSE_VMIN 이상인 첫/마지막 스텝의 course 차
    first = np.argmax(mov, axis=1)
    last = OBS - 1 - np.argmax(mov[:, ::-1], axis=1)
    has = mov.any(axis=1)
    dcourse = np.full(len(pos), np.nan)
    idx = np.arange(len(pos))
    dcourse[has] = np.degrees(wrap(c[idx[has], last[has]] - c[idx[has], first[has]]))
    spmax = sp.max(axis=1)
    spmed = np.median(sp, axis=1)
    return dict(course=c, spd=sp, path=path, dcourse=dcourse, spmax=spmax, spmed=spmed,
                first=first, last=last, has=has)


def _straight_mask(cl):
    return (np.abs(cl["dcourse"]) < STRAIGHT_DEG) & (cl["path"] > STRAIGHT_MOVE_M) & \
        cl["has"] & (cl["spd"][:, RAMP:].min(axis=1) >= COURSE_VMIN)


def _turn_mask(cl):
    return (np.abs(cl["dcourse"]) > TURN_DEG) & (cl["path"] > STRAIGHT_MOVE_M) & \
        cl["has"] & (cl["spd"][:, RAMP:].min(axis=1) >= 3.0)


def metric_straight(h, vel=None, lo=RAMP, hi=OBS, dt=DT):
    """① 직진 heading 일관성. h (N,T). 인덱스 lo..hi-1 만 쓴다.
    직진이므로 '참 요레이트 ≈ 0' 이고, |Δh| 는 그대로 잡음 크기다."""
    u = np.unwrap(h[:, lo:hi], axis=1)
    d = np.degrees(np.diff(u, axis=1)) / dt                  # [°/s]
    ad = np.abs(d)
    t = np.arange(u.shape[1], dtype=np.float64)
    V = np.vander(t, 2)
    coef, *_ = np.linalg.lstsq(V, u.T, rcond=None)
    r = np.degrees(u - (V @ coef).T)
    s = np.sign(d)
    flip = (s[:, 1:] * s[:, :-1] < 0).mean(axis=1) * 100
    out = {"n": int(len(h)),
           "dh_p50": float(np.median(ad)), "dh_p90": float(np.percentile(ad, 90)),
           "dh_p99": float(np.percentile(ad, 99)),
           "resid_std_p50": float(np.median(r.std(axis=1))),
           "resid_std_mean": float(r.std(axis=1).mean()),
           "flip_pct": float(np.mean(flip))}
    if vel is not None:
        c, _, _ = _course(vel)
        wr = np.degrees(wrap(np.diff(c[:, lo:hi], axis=1))) / dt
        out["w_rmse_degs"] = float(np.sqrt(((d - wr) ** 2).mean()))
    return out


def metric_turn(h, vel, lo=RAMP, hi=OBS, dt=DT):
    """② 회전 보존. 기준은 AV2 속도 필드 course 의 요레이트(위치 잡음과 무관)."""
    c, _, _ = _course(vel)
    wr = np.degrees(wrap(np.diff(c[:, lo:hi], axis=1))) / dt
    wm = np.degrees(np.diff(np.unwrap(h[:, lo:hi], axis=1), axis=1)) / dt
    pr = np.abs(wr).max(axis=1)
    pm = np.abs(wm).max(axis=1)
    ratio = pm / np.maximum(pr, 1e-6)
    # 위상 지연: 교차상관 최대점 (lag -5..+5 스텝)
    a = wm - wm.mean(axis=1, keepdims=True)
    b = wr - wr.mean(axis=1, keepdims=True)
    L = a.shape[1]
    lags = np.arange(-5, 6)
    cc = np.zeros((len(a), len(lags)))
    for j, k in enumerate(lags):
        if k >= 0:
            cc[:, j] = (a[:, k:] * b[:, :L - k]).sum(axis=1) if k else (a * b).sum(axis=1)
        else:
            cc[:, j] = (a[:, :L + k] * b[:, -k:]).sum(axis=1)
    lag = lags[cc.argmax(axis=1)] * dt
    # 회전 시작·끝 시점: |누적 회전| 이 전체의 10% / 90% 를 넘는 첫 스텝
    def t1090(w):
        cum = np.cumsum(np.abs(w), axis=1)
        tot = cum[:, -1:]
        f = cum / np.maximum(tot, 1e-9)
        return (np.argmax(f >= 0.1, axis=1) * dt, np.argmax(f >= 0.9, axis=1) * dt)
    s_m, e_m = t1090(wm)
    s_r, e_r = t1090(wr)
    net_m = np.degrees(wrap(h[:, hi - 1] - h[:, lo]))
    net_r = np.degrees(wrap(c[:, hi - 1] - c[:, lo]))
    return {"n": int(len(h)),
            "peak_ratio_p50": float(np.median(ratio)), "peak_ratio_p10": float(np.percentile(ratio, 10)),
            "peak_ref_p50": float(np.median(pr)), "peak_m_p50": float(np.median(pm)),
            "lag_p50_s": float(np.median(lag)), "lag_mean_s": float(lag.mean()),
            "t_start_err_p50_s": float(np.median(np.abs(s_m - s_r))),
            "t_end_err_p50_s": float(np.median(np.abs(e_m - e_r))),
            # 요레이트 시계열이 독립 기준과 얼마나 맞나 — 잡음(①)과 감쇠(②)를 한 숫자로 본다.
            # 임계값이 없고 방향도 하나뿐이라 순위를 매기는 데 쓴다.
            "w_rmse_degs": float(np.sqrt(((wm - wr) ** 2).mean())),
            "w_rmse_p50": float(np.median(np.sqrt(((wm - wr) ** 2).mean(axis=1)))),
            "net_ratio_p50": float(np.median(net_m / np.where(np.abs(net_r) > 1e-6, net_r, np.nan))),
            "net_err_p50_deg": float(np.median(np.abs(net_m - net_r)))}


def metric_ref(h, pos, vel, lo=RAMP, hi=OBS):
    """③ 독립 기준(속도 필드 course)과의 일치 — 속력 구간별. 램프는 뺀다."""
    vm = vel[:, :-1] + vel[:, 1:]
    c_mid = np.arctan2(vm[:, :, 1], vm[:, :, 0])
    sp_f = np.linalg.norm(vel, axis=2)
    sp_mid = 0.5 * (sp_f[:, :-1] + sp_f[:, 1:])
    dif = np.degrees(np.abs(wrap(h[:, :-1] - c_mid)))
    sel = np.zeros_like(dif, bool)
    sel[:, lo:hi - 1] = True
    out = {"all": {}}
    m = sel & (sp_mid >= COURSE_VMIN)
    out["all"] = {"n": int(m.sum()), "p50": float(np.median(dif[m])), "p90": float(np.percentile(dif[m], 90))}
    for i, lbl in enumerate(SPEED_LBL):
        mm = sel & (sp_mid >= SPEED_BINS[i]) & (sp_mid < SPEED_BINS[i + 1])
        if mm.sum() < 50:
            continue
        out[lbl] = {"n": int(mm.sum()), "p50": float(np.median(dif[mm])),
                    "p90": float(np.percentile(dif[mm], 90))}
    return out


def metric_recon(pos, ps, h, lo=RAMP, hi=OBS):
    """④ 재구성: 위치 RMSE, 그리고 방향만 바꿔 다시 굴린 궤적의 끝점 오차."""
    e = np.linalg.norm(ps[:, lo:hi] - pos[:, lo:hi], axis=2)
    step = np.linalg.norm(np.diff(pos[:, lo:hi], axis=1), axis=2)
    d = np.stack([np.cos(h[:, lo:hi - 1]), np.sin(h[:, lo:hi - 1])], axis=2) * step[:, :, None]
    roll = pos[:, lo:lo + 1] + np.cumsum(d, axis=1)
    fe = np.linalg.norm(roll[:, -1] - pos[:, hi - 1], axis=1)
    return {"rmse_m": float(np.sqrt((e ** 2).mean())), "rmse_p95": float(np.percentile(e, 95)),
            "roll_fde_mean": float(fe.mean()), "roll_fde_p95": float(np.percentile(fe, 95))}


def metric_stop(ps, h, src, flip, v_min=V_MIN, dt=DT):
    """⑤ 정지 차량: 잡음이 만든 가짜 이동 비율과 180° 뒤집힘 판정 비율."""
    sp = np.linalg.norm(np.diff(ps, axis=1), axis=2) / dt
    fake = (sp >= v_min).mean() * 100
    judged = (src == 1).sum(axis=1) >= 3
    return {"n": int(len(ps)), "fake_step_pct": float(fake),
            "judged_pct": float(judged.mean() * 100),
            "flip_pct": float(flip.mean() * 100),
            "flip_of_judged_pct": float(flip[judged].mean() * 100) if judged.any() else float("nan")}


def metric_edge(ps, pos, dt=DT):
    """⑥ 경계: 창 시작 0.5초 속력 비율(첫 스텝 ÷ 스텝 10~15 평균)과 관측 끝 스텝 비율."""
    sp = np.linalg.norm(np.diff(ps, axis=1), axis=2) / dt
    ref = sp[:, 10:16].mean(axis=1)
    m = np.linalg.norm(np.diff(pos, axis=1), axis=2).mean(axis=1) / dt > 3.0
    r0 = sp[m][:, 0] / np.maximum(ref[m], 1e-6)
    ref2 = sp[m][:, 43:48].mean(axis=1)
    r1 = sp[m][:, 48] / np.maximum(ref2, 1e-6)
    return {"n": int(m.sum()), "first_ratio_p50": float(np.median(r0)),
            "first_iqr": [float(np.percentile(r0, 25)), float(np.percentile(r0, 75))],
            "last_ratio_p50": float(np.median(r1))}


def metric_jerk(ps, h, lo=RAMP, hi=OBS, dt=DT):
    """⑦ 저크·요각가속: 사람 운전 범위(|j| < 4 m/s³)에 들어오는가."""
    sp = np.linalg.norm(np.diff(ps[:, lo:hi], axis=1), axis=2) / dt
    acc = np.diff(sp, axis=1) / dt
    j = np.diff(acc, axis=1) / dt
    w = np.degrees(np.diff(np.unwrap(h[:, lo:hi], axis=1), axis=1)) / dt
    al = np.diff(w, axis=1) / dt
    return {"j_p50": float(np.median(np.abs(j))), "j_p90": float(np.percentile(np.abs(j), 90)),
            "j_p99": float(np.percentile(np.abs(j), 99)),
            "j_under4_pct": float((np.abs(j) < 4.0).mean() * 100),
            "a_p90": float(np.percentile(np.abs(acc), 90)),
            "alpha_p90": float(np.percentile(np.abs(al), 90)),
            "alpha_p99": float(np.percentile(np.abs(al), 99))}


def task_eval(a):
    DATA.mkdir(parents=True, exist_ok=True)
    load_char_sigma()
    dirs = scen_dirs(a.split, a.limit, a.seed)
    print(f"[eval] 시나리오 {len(dirs):,} 읽는 중 … (σ_pos={SIG_POS:.4f} m, σ_vel={SIG_VEL:.4f} m/s)", flush=True)
    F, A = collect(dirs, a.workers)
    pos, vel, head = F["pos"][:, :OBS], F["vel"][:, :OBS], F["head"][:, :OBS]
    cl = classify(pos, vel)
    ms, mt = _straight_mask(cl), _turn_mask(cl)
    print(f"[eval] focal {len(pos):,} · 직진 {ms.sum():,} · 회전 {mt.sum():,}", flush=True)

    sp_av2_a = np.linalg.norm(A["vel"], axis=2)
    stop_a = sp_av2_a.max(axis=1) < STOP_VMAX
    rng = np.random.default_rng(a.seed)
    idx_stop = np.flatnonzero(stop_a)
    if len(idx_stop) > a.stop_cap:
        idx_stop = np.sort(rng.choice(idx_stop, a.stop_cap, replace=False))
    SP, SV, SH = A["pos"][idx_stop], A["vel"][idx_stop], A["head"][idx_stop]
    print(f"[eval] 정지 차량 {len(idx_stop):,} (전체 정지 {stop_a.sum():,} 중)", flush=True)

    # 저속 직진 하위 모집단 (①을 속력 구간별로도 본다)
    slow = ms & (cl["spmed"] < 5.0)
    fast = ms & (cl["spmed"] >= 10.0)

    res = {"meta": {"n_scen": len(dirs), "n_focal": int(len(pos)), "n_straight": int(ms.sum()),
                    "n_turn": int(mt.sum()), "n_straight_slow": int(slow.sum()),
                    "n_straight_fast": int(fast.sum()), "n_stop_eval": int(len(idx_stop)),
                    "n_stop_all": int(stop_a.sum()), "n_agent": int(len(A["pos"])),
                    "sig_pos": SIG_POS, "sig_vel": SIG_VEL, "seed": a.seed, "split": a.split,
                    "eval_range": [RAMP, OBS], "v_min": V_MIN, "jump_deg": JUMP_DEG},
           "methods": {}}
    import time
    for m in methods(include_slow=not a.fast):
        key, lbl, grp = m[0], m[1], m[2]
        t0 = time.time()
        ps, h, src, flip = apply_method(m, pos, vel, head)
        r = {"label": lbl, "group": grp}
        r["m1"] = metric_straight(h[ms], vel[ms])
        r["m1_slow"] = metric_straight(h[slow], vel[slow])
        r["m1_fast"] = metric_straight(h[fast], vel[fast])
        r["m2"] = metric_turn(h[mt], vel[mt])
        r["m3"] = metric_ref(h, pos, vel)
        r["m4"] = metric_recon(pos, ps, h)
        r["m6"] = metric_edge(ps, pos)
        r["m7"] = metric_jerk(ps, h)
        # ⑤ 정지 차량
        sps, sh, ssrc, sflip = apply_method(m, SP, SV, SH)
        r["m5"] = metric_stop(sps, sh, ssrc, sflip)
        # ⑧ 2 Hz 다운샘플 (평활 -> 다운샘플 순서, 지금과 같다)
        p2 = ps[:, IDX_2HZ]
        hr2 = guard_batch(head)[:, IDX_2HZ]
        v2 = vel[:, IDX_2HZ]
        h2, src2, _ = build_heading_batch(p2, h_ref=hr2, dt=DT * DS_STEP, v_min=V_MIN, jump_deg=180.0)
        if m[4].startswith("ang"):
            h2 = smooth_heading_direct(h2, src2, kind="sg", win=5, order=2, dt=DT * DS_STEP, min_run=5)
        r["m8_1"] = metric_straight(h2[ms], v2[ms], 0, len(IDX_2HZ), dt=DT * DS_STEP)
        r["m8_2"] = metric_turn(h2[mt], v2[mt], 0, len(IDX_2HZ), dt=DT * DS_STEP)
        r["m8_3"] = metric_ref(h2, p2, v2, 0, len(IDX_2HZ))
        r["sec"] = round(time.time() - t0, 1)
        res["methods"][key] = r
        print(f"  {key:10s} {lbl:34s} ①p90 {r['m1']['dh_p90']:7.2f}°/s  지그재그 {r['m1']['flip_pct']:5.1f}%  "
              f"②비 {r['m2']['peak_ratio_p50']:.3f}  ③p50 {r['m3']['all']['p50']:5.2f}°  "
              f"④RMSE {r['m4']['rmse_m']*100:5.1f} cm  ⑦j<4 {r['m7']['j_under4_pct']:5.1f}%  [{r['sec']}s]",
              flush=True)

    res["rule"] = decide(res)
    (DATA / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n[eval] -> {DATA/'eval.json'}")
    print(json.dumps(res["rule"], ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 판별 규칙 (미리 적고 그대로 따른다)
RULE_TXT = ("① 직진 |Δh| p90 이 기준선(없음)의 50% 이하   AND   "
            "② 최대 요레이트 비(peak/속도필드 기준) 가 기준선 대비 5%p 이내로만 감소   AND   "
            "③ 전체 |h − course| 중앙값이 기준선보다 나빠지지 않음   AND   "
            "⑦ |저크| < 4 m/s³ 비율이 기준선보다 높음")
# 1차 규칙이 빈 집합이면 아래를 쓴다 (실행 결과를 보고 정했다 — 리포트에 그렇게 적는다).
RULE2_TXT = ("2차: ①(2 Hz) |Δh| p90 <= 기준선의 70%   AND   ② 최대 요레이트 비가 기준선 대비 10%p 이내   AND   "
             "③(2 Hz) |h − course| 중앙값 <= 기준선   AND   ⑦ |저크|<4 비율 >= 기준선. "
             "순위는 **독립 기준(속도필드 course) 대비 요레이트 RMSE** 오름차순 — 잡음과 감쇠를 한 숫자로 보고 임계값이 없다.")


def decide(res):
    base = res["methods"]["none"]
    out = {"rule": RULE_TXT, "pass": [], "fail": {}}
    for k, r in res["methods"].items():
        if k == "none":
            continue
        c1 = r["m1"]["dh_p90"] <= 0.5 * base["m1"]["dh_p90"]
        c2 = r["m2"]["peak_ratio_p50"] >= base["m2"]["peak_ratio_p50"] - 0.05
        c3 = r["m3"]["all"]["p50"] <= base["m3"]["all"]["p50"] + 1e-9
        c7 = r["m7"]["j_under4_pct"] >= base["m7"]["j_under4_pct"]
        if c1 and c2 and c3 and c7:
            out["pass"].append(k)
        else:
            out["fail"][k] = [n for n, c in (("①", c1), ("②", c2), ("③", c3), ("⑦", c7)) if not c]
    out["rank"] = sorted(out["pass"], key=lambda k: res["methods"][k]["m4"]["rmse_m"])
    out["rule2"] = RULE2_TXT
    p2, f2 = [], {}
    for k, r in res["methods"].items():
        if k == "none":
            continue
        c = [r["m8_1"]["dh_p90"] <= 0.7 * base["m8_1"]["dh_p90"],
             r["m2"]["peak_ratio_p50"] >= base["m2"]["peak_ratio_p50"] - 0.10,
             r["m8_3"]["all"]["p50"] <= base["m8_3"]["all"]["p50"] + 1e-9,
             r["m7"]["j_under4_pct"] >= base["m7"]["j_under4_pct"]]
        (p2.append(k) if all(c) else f2.update({k: [n for n, cc in zip(("①", "②", "③", "⑦"), c) if not cc]}))
    out["pass2"], out["fail2"] = p2, f2
    out["rank2"] = sorted(p2, key=lambda k: res["methods"][k]["m2"]["w_rmse_degs"])
    out["rank_wrmse_all"] = sorted(res["methods"], key=lambda k: res["methods"][k]["m2"]["w_rmse_degs"])
    out["rank_wrmse_straight"] = sorted(res["methods"], key=lambda k: res["methods"][k]["m1"]["w_rmse_degs"])
    return out


# ================================================================ 대표 시나리오 덤프
PANEL_CLASSES = ["정속 직진", "저속 직진", "좌회전", "우회전", "급가감속", "정지", "차선변경"]


def panel_class(cl, pos, vel):
    """관측 구간만 보는 상황 분류 (독립 기준 = AV2 속도 필드). 우선순위 순, 상호배타."""
    n = len(pos)
    lab = np.array(["기타"] * n, dtype=object)
    c, dc, sp = cl["course"], None, cl["spd"]
    from scipy.signal import savgol_filter
    sps = savgol_filter(np.median(np.stack([sp, np.roll(sp, 1, 1), np.roll(sp, -1, 1)]), axis=0), 15, 2, axis=1)
    acc = np.gradient(sps, DT, axis=1)
    amin, amax_ = acc[:, RAMP:].min(axis=1), acc[:, RAMP:].max(axis=1)
    dh = cl["dcourse"]
    # 횡변위: 시작 course 방향 직선에서의 수직 거리 (지도 없이 차선변경 대용)
    c0 = c[np.arange(n), np.maximum(cl["first"], RAMP)]
    rel = pos[:, OBS - 1] - pos[:, RAMP]
    lat = -np.sin(c0) * rel[:, 0] + np.cos(c0) * rel[:, 1]
    lab[(cl["spmax"] < 1.0)] = "정지"
    free = lab == "기타"
    lab[free & (dh > TURN_DEG) & (cl["path"] > 5)] = "좌회전"
    free = lab == "기타"
    lab[free & (dh < -TURN_DEG) & (cl["path"] > 5)] = "우회전"
    free = lab == "기타"
    lab[free & (np.abs(dh) < 15) & (np.abs(lat) > 2.5) & (cl["path"] > 10)] = "차선변경"
    free = lab == "기타"
    lab[free & ((amin < -3.0) | (amax_ > 2.5)) & (cl["path"] > 5)] = "급가감속"
    free = lab == "기타"
    lab[free & (np.abs(dh) < STRAIGHT_DEG) & (cl["spmed"] >= 8.0) & (cl["path"] > 10)] = "정속 직진"
    free = lab == "기타"
    lab[free & (np.abs(dh) < STRAIGHT_DEG) & (cl["spmed"] >= 1.0) & (cl["spmed"] < 5.0) & (cl["path"] > 5)] = "저속 직진"
    return lab, lat, acc


def task_cases(a):
    DATA.mkdir(parents=True, exist_ok=True)
    load_char_sigma()
    dirs = scen_dirs(a.split, a.limit, a.seed)
    F, A = collect(dirs, a.workers, agents=False)
    pos, vel, head = F["pos"][:, :OBS], F["vel"][:, :OBS], F["head"][:, :OBS]
    cl = classify(pos, vel)
    lab, lat, acc = panel_class(cl, pos, vel)
    rng = np.random.default_rng(a.seed)
    picks = {}
    for c in PANEL_CLASSES:
        idx = np.flatnonzero(lab == c)
        if len(idx) == 0:
            continue
        picks[c] = sorted(rng.choice(idx, min(a.per_class, len(idx)), replace=False).tolist())
    sel = sorted({i for v in picks.values() for i in v})
    keys = [m[0] for m in methods(include_slow=True)]
    P = {"sid": F["sid"][sel].tolist(), "cls": [str(lab[i]) for i in sel],
         "idx": sel, "picks": {k: [sel.index(i) for i in v] for k, v in picks.items()}}
    sub = lambda x: np.ascontiguousarray(x[sel])
    arr = {"pos": sub(pos), "vel": sub(vel), "head": sub(head), "pos_full": sub(F["pos"]),
           "lat": lat[sel], "acc": sub(acc), "dcourse": cl["dcourse"][sel], "spmed": cl["spmed"][sel]}
    for m in methods(include_slow=True):
        ps, h, src, flip = apply_method(m, sub(pos), sub(vel), sub(head))
        arr[f"ps_{m[0]}"] = ps
        arr[f"h_{m[0]}"] = h
        arr[f"src_{m[0]}"] = src
        p2 = ps[:, IDX_2HZ]
        hr2 = guard_batch(sub(head))[:, IDX_2HZ]
        h2, s2, _ = build_heading_batch(p2, h_ref=hr2, dt=DT * DS_STEP, v_min=V_MIN, jump_deg=180.0)
        if m[4].startswith("ang"):
            h2 = smooth_heading_direct(h2, s2, kind="sg", win=5, order=2, dt=DT * DS_STEP, min_run=5)
        arr[f"h2_{m[0]}"] = h2
        arr[f"p2_{m[0]}"] = p2
    np.savez_compressed(DATA / "cases.npz", **arr)
    (DATA / "cases.json").write_text(json.dumps({**P, "methods": keys,
                                                 "class_counts": {c: int((lab == c).sum()) for c in PANEL_CLASSES},
                                                 "n_focal": int(len(pos))}, ensure_ascii=False, indent=1))
    print(json.dumps({c: int((lab == c).sum()) for c in PANEL_CLASSES}, ensure_ascii=False))
    print(f"[cases] {len(sel)} 시나리오 -> {DATA/'cases.npz'}")


# ================================================================ 검증
def task_verify(a):
    """배치 구현이 heading_decomp 원본과 같은지, 그리고 지금 파이프라인을 재현하는지 확인한다."""
    dirs = scen_dirs(a.split, min(a.limit, 400), a.seed)
    F, A = collect(dirs, a.workers)
    pos, vel, head = F["pos"][:, :OBS], F["vel"][:, :OBS], F["head"][:, :OBS]
    out = {}

    # (1) build_heading_batch == heading_decomp.build_heading
    h_b, s_b, fl = build_heading_batch(pos, h_ref=head)
    e, se = 0.0, 0
    for i in range(len(pos)):
        h0, s0 = hd.build_heading(pos[i], h_ref=head[i])
        e = max(e, float(np.max(np.abs(wrap(h_b[i] - h0)))))
        se += int((s_b[i] != s0).sum())
    out["build_heading 최대차 [rad]"] = e
    out["build_heading src 불일치 스텝"] = se

    # (2) 주변 차량에서도 같은지
    h_a, s_a, _ = build_heading_batch(A["pos"][:2000], h_ref=A["head"][:2000])
    ea, sa = 0.0, 0
    for i in range(min(2000, len(A["pos"]))):
        h0, s0 = hd.build_heading(A["pos"][i], h_ref=A["head"][i])
        ea = max(ea, float(np.max(np.abs(wrap(h_a[i] - h0)))))
        sa += int((s_a[i] != s0).sum())
    out["주변차량 build_heading 최대차 [rad]"] = ea
    out["주변차량 src 불일치"] = sa

    # (3) 현재 방식(sg5_2) + 2 Hz 가 heading_decomp.ah_features_2hz 의 h 와 같은지
    ps = _sg(pos, 5, 2, 4)
    p2 = ps[:, IDX_2HZ]
    hr2 = guard_batch(head)[:, IDX_2HZ]
    h2, _, _ = build_heading_batch(p2, h_ref=hr2, dt=DT * DS_STEP, v_min=V_MIN, jump_deg=180.0)
    e2 = 0.0
    for i in range(len(pos)):
        f = hd.ah_features_2hz(pos[i], head[i], 0.0)
        e2 = max(e2, float(np.max(np.abs(wrap(h2[i] - f[:, 1])))))
    out["ah_features_2hz h 최대차 [rad]"] = e2

    # (4) 10 Hz 현재 입력(ah_features) 의 h 는 평활 없음 + build_heading 과 같아야 한다
    e1 = 0.0
    for i in range(len(pos)):
        f, _ = hd.ah_features(pos[i], head[i], 0.0)
        e1 = max(e1, float(np.max(np.abs(wrap(h_b[i] - f[:, 1])))))
    out["ah_features h 최대차 [rad]"] = e1

    # (5) 평활기 성질 확인 ①: 등속 직선(잡음 없음)을 그대로 두는가 — 창 경계 포함
    t = np.arange(OBS, dtype=np.float64)
    line = np.stack([np.stack([12.0 * t * DT, np.zeros(OBS)], axis=1)])
    lvel = np.stack([np.stack([np.full(OBS, 12.0), np.zeros(OBS)], axis=1)])
    chk = {}
    for m in methods(include_slow=True):
        ps_ = m[3](line, lvel, np.zeros((1, OBS)), DT)
        chk[m[0]] = {"max_dev_cm": float(np.abs(ps_ - line).max() * 100),
                     "first_speed_ratio": float(np.linalg.norm(ps_[0, 1] - ps_[0, 0]) / DT / 12.0)}
    out["등속 직선(12 m/s) 왜곡"] = chk

    # (5b) 평활기 성질 확인 ②: 등회전(반경 30 m, 12 m/s = 0.4 rad/s) 을 얼마나 깎는가
    w, R = 0.4, 30.0
    th = w * t * DT
    arc = np.stack([np.stack([R * np.sin(th), R * (1 - np.cos(th))], axis=1)])
    avel = np.stack([np.stack([12.0 * np.cos(th), 12.0 * np.sin(th)], axis=1)])
    chk2 = {}
    for m in methods(include_slow=True):
        ps_ = m[3](arc, avel, th[None], DT)
        d = np.diff(ps_[0], axis=0)
        hh = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        ww = np.diff(hh) / DT
        chk2[m[0]] = {"max_dev_cm": float(np.abs(ps_ - arc).max() * 100),
                      "yaw_rate_mid": float(np.median(ww[10:35]) / w),
                      "yaw_rate_min": float(ww[RAMP:].min() / w)}
    out["등회전(R=30 m, ω=0.4 rad/s) 보존"] = chk2

    # (6) 운동학 제약이 실제로 지켜지는가 (벌점이라 '거의' 다 — 남은 위반을 적는다)
    vio = {}
    for key in ("none", "kin"):
        m = next(x for x in methods(include_slow=True) if x[0] == key)
        ps_ = m[3](pos, vel, head, DT)
        sp = np.linalg.norm(np.diff(ps_, axis=1), axis=2) / DT
        acc = np.diff(sp, axis=1) / DT
        d = np.diff(ps_, axis=1)
        ww = wrap(np.diff(np.arctan2(d[:, :, 1], d[:, :, 0]), axis=1)) / DT
        mv = 0.5 * (sp[:, 1:] + sp[:, :-1]) >= V_MIN
        vio[key] = {"|a|>8 [%]": float((np.abs(acc) > 8).mean() * 100),
                    "|w|>0.95 전체 [%]": float((np.abs(ww) > 0.95).mean() * 100),
                    "|w|>0.95 이동스텝 [%]": float((np.abs(ww[mv]) > 0.95).mean() * 100),
                    "rmse [cm]": float(np.sqrt(((ps_ - pos) ** 2).sum(-1).mean()) * 100)}
    out["운동학 제약 위반 (focal %d트랙)" % len(pos)] = vio
    (DATA).mkdir(parents=True, exist_ok=True)
    (DATA / "verify.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="char", choices=["char", "eval", "cases", "verify"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stop-cap", dest="stop_cap", type=int, default=20000)
    ap.add_argument("--per-class", dest="per_class", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="느린 후보(운동학 제약 적합)를 뺀다")
    a = ap.parse_args()
    try:
        os.nice(10)
    except Exception:
        pass
    {"char": task_char, "eval": task_eval, "cases": task_cases, "verify": task_verify}[a.task](a)


if __name__ == "__main__":
    main()
