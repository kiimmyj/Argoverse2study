"""
viz_v4_feas.py - 측정 1~3: 복원된 xy 기준 위반율 · 저크 분포 · 액션열 주파수.

왜 다시 재나
------------
지금 쓰는 "7.3°/step 초과"는 모델이 낸 dθ 에서 잰 값이다. 흔들림 벌점이 **직접 누르는 변수**라
벌점을 걸면 당연히 내려간다. MultiPath++ (arXiv:2111.14973 표 4) 는 제어 출력이 heading 기준
비실현율을 4.10% → 0.00% 로 없애면서도 좌표 3점으로 재면 1.08% → 1.22% 로 나빠졌다고 보고한다.
그래서 여기서는 **예측 좌표에서** 운동학을 다시 만들어 잰다.

정의는 파일 맨 위 상수에 모아 둔다. 그림·json 이 같은 숫자를 쓴다.

  python src/viz_v4_feas.py                 # 계산 + 그림
  python src/viz_v4_feas.py --figs-only     # 저장된 json 으로 그림만
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

OUT = C.VIZ_ROOT / "feasibility"
DATA = OUT / "data"
DT = C.DT
FS = 1.0 / DT                       # 액션열 표본율 [Hz] — 2 Hz 판도 **출력은 10 Hz** 다 (입력만 2 Hz)

# ---------------------------------------------------------------- 임계값 (출처를 적는다)
MOVE_V = C.MOVE_V                   # 1.0 m/s — 이보다 느린 스텝은 위치차분 방향이 잡음이다 (프로젝트 관례)
LABEL_DTHETA_DEG = C.LABEL_DTHETA_DEG   # 7.3°/step — AV2 라벨 전수조사 p99.99 (기존 지표와 같은 임계)
YAWRATE_MAX = 0.95                  # rad/s   — nuPlan ego_is_comfortable
YAWACC_MAX = 1.93                   # rad/s²  — nuPlan ego_is_comfortable
R_MIN = 3.0                         # m       — DKM 최소 회전반경
JERK_LK = 0.9                       # m/s³    — LK-SDE (arXiv:2309.09317)
JERK_NUPLAN = 4.13                  # m/s³    — nuPlan 종방향 저크 상한
BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
BAND_LAB = ["0–0.5 Hz", "0.5–1 Hz", "1–2 Hz", "2–5 Hz"]
# 스텝 속력 구간 — 위반이 저속에서 나오는지 보려고 (MOVE_V 미만은 애초에 무효)
SPD_EDGES = [1.0, 2.0, 5.0, 10.0, 20.0, np.inf]
SPD_LAB = ["1–2", "2–5", "5–10", "10–20", "≥20"]
# 위반 크기 구간 [°/step]
MAG_EDGES = [LABEL_DTHETA_DEG, 15.0, 45.0, 90.0, 180.1]
MAG_LAB = ["7.3–15°", "15–45°", "45–90°", "90–180°"]


def by_speed(k, Rc=None):
    """유효 이웃 쌍을 **스텝 속력**으로 나눠 위반율을 낸다 (저속 구간 기여 확인)."""
    o2, v = k["ok2"], k["vmid"]
    d_deg = np.degrees(np.abs(k["dpsi"]))
    out = {}
    for i, lab in enumerate(SPD_LAB):
        m = o2 & (v >= SPD_EDGES[i]) & (v < SPD_EDGES[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        out[lab] = {"n": n, "share_pct": float(100.0 * n / max(int(o2.sum()), 1)),
                    "over_label_pct": float(100.0 * (d_deg[m] > LABEL_DTHETA_DEG).mean()),
                    "R_under3_pct": float(100.0 * (k["R"][m] < R_MIN).mean()),
                    "mean_step_m": float((v[m] * DT).mean()),
                    "viol_share_pct": float(100.0 * ((d_deg > LABEL_DTHETA_DEG) & m).sum()
                                            / max(((d_deg > LABEL_DTHETA_DEG) & o2).sum(), 1))}
    mag = {}
    tot = max(int(((d_deg > LABEL_DTHETA_DEG) & o2).sum()), 1)
    for i, lab in enumerate(MAG_LAB):
        m = o2 & (d_deg >= MAG_EDGES[i]) & (d_deg < MAG_EDGES[i + 1])
        mag[lab] = {"n": int(m.sum()), "share_of_viol_pct": float(100.0 * int(m.sum()) / tot),
                    "mean_step_m": float((k["vmid"][m] * DT).mean()) if m.any() else float("nan")}
    return {"by_speed": out, "by_magnitude": mag}

# 판 (표시 순서 = 벌점 세기 순). hz 는 **입력** 주기다.
TAGS = [
    ("v4_l4nw_ah2_full_s0",               "벌점없음",        0.0, "15에폭 상수", 10, C.C_RED,     "o"),
    ("v4_l0_ah2_full_sm0.1_s0",           "L0·벌점0.1",      0.1, "15에폭 상수", 10, C.C_ORANGE,  "v"),
    ("v4_l4nw_ah2_full_sm1_s0",           "L4·벌점1",        1.0, "15에폭 상수", 10, C.C_BLUE,    "s"),
    ("v4_l4nw_ah2_full_sm1_cos30_s0",     "L4·벌점1·30",     1.0, "30에폭 코사인", 10, C.C_VIOLET, "D"),
    ("v4_l4nw_ah2_2hz_full_sm1_cos30_s0", "2Hz·벌점1·30",    1.0, "30에폭 코사인", 2,  C.C_AQUA,   "^"),
]
GT_COL, GT_MK = C.INK, "*"
SUBSAMPLE = 3_000_000               # Wasserstein 계산용 표본 상한 (시드 고정)
KEEP_N = 500_000                    # dists.npz 에 저장할 표본 상한 (그림용 — 생존함수 2e-6 까지 보인다)


# ================================================================== 공용 운동학
def xy_kin(P, move_v=None):
    """좌표열 P (..., T, 2) → 위치에서 복원한 운동학. 자기참조가 없다.

    heading 은 **연속 두 점의 차분**이다. |Δp|/dt < MOVE_V 인 스텝은 방향이 잡음이라 무효로 두고,
    Δheading 은 **양쪽이 모두 유효한 이웃 쌍**에서만 센다 (직전값 유지로 메우지 않는다 —
    메우면 정지 구간의 0 이 분모만 키운다). 무효 비율은 따로 보고한다.
    """
    move_v = MOVE_V if move_v is None else move_v
    u = np.diff(P, axis=-2)                                   # (...,T-1,2)
    sp = np.linalg.norm(u, axis=-1) / DT                      # (...,T-1) [m/s]
    ok = sp >= move_v
    psi = np.arctan2(u[..., 1], u[..., 0])
    dpsi = C.wrap(psi[..., 1:] - psi[..., :-1])               # (...,T-2) [rad/step]
    ok2 = ok[..., 1:] & ok[..., :-1]
    yaw = dpsi / DT                                           # [rad/s]
    vmid = 0.5 * (sp[..., 1:] + sp[..., :-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.abs(yaw) > 1e-9, vmid / np.abs(yaw), np.inf)
    dyaw = (yaw[..., 1:] - yaw[..., :-1]) / DT                # (...,T-3) [rad/s²]
    ok3 = ok2[..., 1:] & ok2[..., :-1]
    return {"sp": sp, "ok": ok, "psi": psi, "dpsi": dpsi, "ok2": ok2,
            "yaw": yaw, "vmid": vmid, "R": R, "dyaw": dyaw, "ok3": ok3}


MV_GRID = [0.0, 0.5, 1.0, 2.0]      # MOVE_V 민감도 격자 [m/s]


def mv_sensitivity(P):
    """MOVE_V 를 바꿔 가며 |Δψ| > 7.3°/step 위반율을 다시 잰다 (임계 의존성 확인)."""
    out = {}
    for mv in MV_GRID:
        k = xy_kin(P, move_v=mv)
        o2 = k["ok2"]
        d = np.degrees(np.abs(k["dpsi"]))
        out[f"{mv:g}"] = {"n_pair": int(o2.sum()),
                          "invalid_pair_pct": float(100.0 * (~o2).mean()),
                          "over_label_pct": (float(100.0 * (d[o2] > LABEL_DTHETA_DEG).mean())
                                             if o2.any() else float("nan")),
                          "R_under3_pct": float(100.0 * (k["R"][o2] < R_MIN).mean()) if o2.any() else float("nan")}
    return out


def decompose(dpsi_deg, dh_deg, dth_deg, o2):
    """Δψ 를 세 항으로 나눈다. **가산 분해가 아니라** 각 항의 크기와 반사실 위반율이다.
      Δψ = Δθ + Δk + 기하,  Δk = Δh − Δθ,  기하 = Δψ − Δh   (부호 항등식으로는 성립)
    반사실: 한 항을 0 으로 두면 위반율이 얼마가 되는가 → 어느 항이 위반을 만드는가.
    """
    dk = dh_deg - dth_deg
    geom = dpsi_deg - dh_deg
    viol = (np.abs(dpsi_deg) > LABEL_DTHETA_DEG) & o2
    f = lambda x: float(np.abs(x)[o2].mean())
    g = lambda x: float(np.abs(x)[viol].mean()) if viol.any() else float("nan")
    cf = lambda x: float(100.0 * (np.abs(x)[o2] > LABEL_DTHETA_DEG).mean())
    return {"mean_abs_dpsi_deg": f(dpsi_deg), "mean_abs_dtheta_deg": f(dth_deg),
            "mean_abs_dk_deg": f(dk), "mean_abs_geom_deg": f(geom),
            "sum_of_parts_deg": f(dth_deg) + f(dk) + f(geom),
            "identity_max_resid_deg": float(np.abs(dth_deg + dk + geom - dpsi_deg)[o2].max()),
            "viol_mean_abs_dtheta_deg": g(dth_deg), "viol_mean_abs_dk_deg": g(dk),
            "viol_mean_abs_geom_deg": g(geom),
            "cf_drop_dtheta_over_pct": cf(dpsi_deg - dth_deg),
            "cf_drop_dk_over_pct": cf(dpsi_deg - dk),
            "cf_drop_geom_over_pct": cf(dpsi_deg - geom)}


def frenet_fold(P, H, S, D, TH):
    """프레네 접힘 — 경로 좌표에서는 앞으로 가는데 **좌표에서는 뒤로 가는** 스텝.

    궤적점은 P(s) + N(s)·d 다. 곡률 κ 인 경로에서 횡오프셋 d 인 점의 호길이 배율은 (1 − κ·d) 이고,
    |d| 가 국소 곡률반경 1/κ 를 넘으면 이 값이 **음수**가 되어 s 가 늘어도 점이 뒤로 간다.
    적분기에는 이 경우를 막는 장치가 없다 (model_v4.rollout).
    두 갈래로 센다 — 기하식 (1 − κ·d) < 0 과, 스텝 벡터를 모델 내부 진행방향 h 에 투영한 부호.
    """
    kang = H - TH                                       # 경로 접선각 k(s)
    dk = C.wrap(np.diff(kang, axis=-1))
    ds = np.diff(S, axis=-1)
    kap = np.where(np.abs(ds) > 1e-3, dk / np.where(np.abs(ds) > 1e-3, ds, 1.0), 0.0)
    kd = kap * D[..., 1:]
    fold_geo = (1.0 - kd) < 0.0
    step = np.diff(P, axis=-2)
    e = np.stack([np.cos(H[..., :-1]), np.sin(H[..., :-1])], -1)
    fold_dot = (step * e).sum(-1) < 0.0
    return {"kappa": kap, "kd": kd, "fold_geo": fold_geo, "fold_dot": fold_dot}


def circum_R(P):
    """연속 3점의 외접원 반경 (..., T-2). 세 점이 거의 일직선이면 inf."""
    p0, p1, p2 = P[..., :-2, :], P[..., 1:-1, :], P[..., 2:, :]
    a = np.linalg.norm(p1 - p0, axis=-1)
    b = np.linalg.norm(p2 - p1, axis=-1)
    c = np.linalg.norm(p2 - p0, axis=-1)
    cr = (p1 - p0)[..., 0] * (p2 - p0)[..., 1] - (p1 - p0)[..., 1] * (p2 - p0)[..., 0]
    area = 0.5 * np.abs(cr)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(area > 1e-9, a * b * c / (4.0 * np.maximum(area, 1e-12)), np.inf)
    return R


def smooth_speed(v):
    """프로젝트 표준 종방향 평활: median 5 → Savitzky–Golay(15스텝, 2차). 값과 미분을 함께."""
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    size = (1,) * (v.ndim - 1) + (C.MED_K,)
    vm = median_filter(np.asarray(v, np.float64), size=size, mode="nearest")
    vs = savgol_filter(vm, C.SG_WIN, C.SG_POLY, axis=-1, mode="interp")
    a = savgol_filter(vm, C.SG_WIN, C.SG_POLY, deriv=1, delta=DT, axis=-1, mode="interp")
    return vs, a


def jerk_pipes(P, a_action=None, v_src=None):
    """저크 [m/s³] 네 갈래. 모델·정답에 **같은 식**을 건다.
      J1 액션   : 액션 a(t) 의 스텝 차분 (모델은 aux a, 정답은 Frenet 역변환 a)
      J2 위치원시: 위치 1·2·3차 차분 (평활 없음) — 라벨 위치 잡음이 그대로 들어온다
      J3 위치평활: 위치차분 속력 → median5+SG(15,2) → a → 차분
      J4 속도채널: 속도 채널 → **같은 필터** → a → 차분
                  (모델 = 적분기 v, 정답 = AV2 velocity 필드)
    벡터 3차차분(J2v)도 함께 낸다 — 종방향만이 아니라 궤적 전체의 3계 도함수 크기다.
    """
    out = {}
    u = np.diff(P, axis=-2)
    sp = np.linalg.norm(u, axis=-1) / DT
    a_raw = np.diff(sp, axis=-1) / DT
    out["J2"] = np.diff(a_raw, axis=-1) / DT
    out["J2v"] = np.linalg.norm(P[..., 3:, :] - 3 * P[..., 2:-1, :] + 3 * P[..., 1:-2, :]
                                - P[..., :-3, :], axis=-1) / DT ** 3
    _, a_sm = smooth_speed(sp)
    out["J3"] = np.diff(a_sm, axis=-1) / DT
    if a_action is not None:
        out["J1"] = np.diff(np.asarray(a_action, np.float64), axis=-1) / DT
    if v_src is not None:
        _, a_f = smooth_speed(np.asarray(v_src, np.float64))
        out["J4"] = np.diff(a_f, axis=-1) / DT
    return out


def spectrum(x, fs=FS, window="hann", detrend="linear"):
    """진폭 스펙트럼. 창함수 Hann(coherent gain 보정), detrend = 1차 추세 제거.
    반환 f (n/2+1,), M (..., n/2+1) — M 은 편측 진폭 [x 의 단위]."""
    from scipy.signal import detrend as sp_detrend
    x = np.asarray(x, np.float64)
    n = x.shape[-1]
    if detrend == "linear":
        x = sp_detrend(x, axis=-1, type="linear")
    elif detrend == "mean":
        x = x - x.mean(-1, keepdims=True)
    if window == "hann":
        w = np.hanning(n + 2)[1:-1]
    else:
        w = np.ones(n)
    cg = w.mean()
    X = np.fft.rfft(x * w, axis=-1)
    M = 2.0 * np.abs(X) / (n * cg)
    f = np.fft.rfftfreq(n, DT)
    return f, M


def caps_sm(f, M):
    """CAPS 평활도 Sm = (2/(n·f_s))·Σ M_i f_i (arXiv:2012.06644 식 4). DC(i=0) 는 뺀다."""
    n = 2 * (len(f) - 1)
    return (2.0 / (n * FS)) * (M[..., 1:] * f[1:]).sum(-1)


def band_amp(f, M):
    """대역별 진폭 합 (..., 4) 과 전체 합 (DC 제외).

    대역은 **반개구간 (lo, hi]** 다. 예전에는 양 끝에 1e-9 여유를 줘서 0.5·1·2 Hz bin 이
    이웃 대역에 **중복**으로 들어갔다(30 bin 을 33 번 셈). 그래서 대역 합이 전체 합을 넘었다.
    """
    tot = M[..., 1:].sum(-1)
    out, used = [], 0
    for lo, hi in BANDS:
        m = (f > lo) & (f <= hi + 1e-9)
        used += int(m.sum())
        out.append(M[..., m].sum(-1))
    assert used == len(f) - 1, f"대역이 DC 제외 {len(f)-1} bin 을 정확히 덮지 않는다 (덮은 수 {used})"
    return np.stack(out, -1), tot


# ================================================================== 집계 도구
def pct(mask, sel):
    """sel 에서 mask 가 참인 비율 [%] (분모 = sel 의 원소 수)."""
    n = int(sel.sum())
    if n == 0:
        return float("nan"), 0
    return float(100.0 * mask[sel].mean()), n


def m1_stats(P, label):
    """측정 1 — 복원된 xy 기준 위반율. P (M,T,2) 는 한 모집단의 궤적들."""
    k = xy_kin(P)
    Rc = circum_R(P)
    o2 = k["ok2"]
    d_deg = np.degrees(np.abs(k["dpsi"]))
    st = {}
    st["n_traj"] = int(P.shape[0])
    st["n_step_pair"] = int(o2.sum())
    st["invalid_step_pct"] = float(100.0 * (~k["ok"]).mean())
    st["invalid_pair_pct"] = float(100.0 * (~o2).mean())
    st["over_label_pct"], _ = pct(d_deg > LABEL_DTHETA_DEG, o2)
    st["yawrate_over_pct"], _ = pct(np.abs(k["yaw"]) > YAWRATE_MAX, o2)
    st["R_under3_pct"], _ = pct(k["R"] < R_MIN, o2)
    st["R3pt_under3_pct"], _ = pct(Rc < R_MIN, o2)
    st["yawacc_over_pct"], _ = pct(np.abs(k["dyaw"]) > YAWACC_MAX, k["ok3"])
    st["dpsi_p50_deg"] = float(np.percentile(d_deg[o2], 50)) if o2.any() else float("nan")
    st["dpsi_p99_deg"] = float(np.percentile(d_deg[o2], 99)) if o2.any() else float("nan")
    st["dpsi_max_deg"] = float(d_deg[o2].max()) if o2.any() else float("nan")
    st["R_p1_m"] = float(np.percentile(k["R"][o2], 1)) if o2.any() else float("nan")
    st["scen_any_over_label_pct"] = float(100.0 * ((d_deg > LABEL_DTHETA_DEG) & o2).any(1).mean())
    st["scen_any_R3_pct"] = float(100.0 * ((k["R"] < R_MIN) & o2).any(1).mean())
    st["label"] = label
    return st, k, Rc


def jerk_stats(J, sel=None):
    """|저크| 요약 + 두 임계 위반율."""
    x = J if sel is None else J[sel]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("mean_abs", "p90", "p99", "over_0.9_pct", "over_4.13_pct", "n")}
    ax = np.abs(x)
    return {"mean_abs": float(ax.mean()), "p90": float(np.percentile(ax, 90)),
            "p99": float(np.percentile(ax, 99)),
            "over_0.9_pct": float(100.0 * (ax > JERK_LK).mean()),
            "over_4.13_pct": float(100.0 * (ax > JERK_NUPLAN).mean()), "n": int(ax.size)}


def sub(x, n=SUBSAMPLE, seed=0):
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(x.size, n, replace=False)]


def wass(a, b):
    from scipy.stats import wasserstein_distance
    a, b = sub(a), sub(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


# ================================================================== 계산 본체
def compute(tags, n_limit=None):
    import pandas as pd
    t_all = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    res = {"meta": {"git_head": C.git_head(), "dt_s": DT, "fs_hz": FS,
                    "thresholds": {"MOVE_V_mps": MOVE_V, "label_dtheta_deg": LABEL_DTHETA_DEG,
                                   "yawrate_max_radps": YAWRATE_MAX, "yawacc_max_radps2": YAWACC_MAX,
                                   "R_min_m": R_MIN, "jerk_lk_mps3": JERK_LK,
                                   "jerk_nuplan_mps3": JERK_NUPLAN, "bands_hz": BANDS},
                    "fft": {"window": "hann", "detrend": "linear", "n": C.FUT, "fs_hz": FS},
                    "runs": {}}, "tags": {}, "gt": {}}
    keep = {}

    # ---------- 정답 (라벨) 기준선: 캐시 y 를 그대로 쓴다 (모든 판이 같은 val)
    from dataset_cached import CachedV4Dataset
    ds = CachedV4Dataset(C.VAL_CACHE)
    N = len(ds) if n_limit is None else min(n_limit, len(ds))
    Y = np.asarray(ds.raw("y")[:N], np.float64)                    # (N,60,2)
    ref = C.tag_dirs(TAGS[3][0])["data"]
    df0 = pd.read_parquet(ref / "scenarios.parquet").iloc[:N]
    rz = np.load(ref / "raw.npz")
    v_fld_fut = np.asarray(rz["v_fld"][:N, C.OBS:], np.float64)     # (N,60) AV2 속도 필드 평활 전 값
    cls = df0["cls"].to_numpy()

    st_gt, k_gt, Rc_gt = m1_stats(Y, "정답 라벨")
    st_gt.update(by_speed(k_gt, Rc_gt))
    st_gt["mv_sensitivity"] = mv_sensitivity(Y)
    res["gt"]["m1"] = st_gt
    Jg = jerk_pipes(Y, v_src=v_fld_fut)
    res["gt"]["m2"] = {p: jerk_stats(Jg[p]) for p in Jg}
    keep["gt"] = {"dpsi_deg": sub(np.degrees(np.abs(k_gt["dpsi"]))[k_gt["ok2"]]),
                  "R": sub(k_gt["R"][k_gt["ok2"]]),
                  **{f"J_{p}": sub(Jg[p]) for p in Jg}}
    # 상황별 정답
    res["gt"]["m1_by_cls"] = {}
    res["gt"]["m2_by_cls"] = {}
    for cn in C.CLASSES:
        m = cls == cn
        if m.sum() < 1:
            continue
        s_, _, _ = m1_stats(Y[m], cn)
        res["gt"]["m1_by_cls"][cn] = s_
        res["gt"]["m2_by_cls"][cn] = {p: jerk_stats(Jg[p][m]) for p in ("J2", "J3", "J4")}
    print(f"[gt] 7.3°초과 {st_gt['over_label_pct']:.3f}%  R<3m {st_gt['R_under3_pct']:.3f}%  "
          f"|저크|평균 J2 {res['gt']['m2']['J2']['mean_abs']:.2f} J3 {res['gt']['m2']['J3']['mean_abs']:.3f} "
          f"J4 {res['gt']['m2']['J4']['mean_abs']:.3f} m/s³  {time.time()-t_all:.0f}s", flush=True)

    # ---------- 판별
    for tag, name, sm, sched, hz, col, mk in tags:
        t0 = time.time()
        d = C.tag_dirs(tag)["data"]
        pz = np.load(d / "pred.npz")
        df = pd.read_parquet(d / "scenarios.parquet").iloc[:N]
        meta = json.loads((d / "meta.json").read_text())
        alive = pz["alive"][:N]                                     # (N,6)
        traj = np.asarray(pz["traj"][:N], np.float64)               # (N,6,60,2)
        a_ch = np.asarray(pz["a"][:N], np.float64)
        dth = np.asarray(pz["dtheta"][:N], np.float64)
        th = np.asarray(pz["theta"][:N], np.float64)
        h_ch = np.asarray(pz["h"][:N], np.float64)
        top1 = pz["top1"][:N].astype(int)
        ar = np.arange(N)
        r = {"name": name, "smooth": sm, "sched": sched, "input_hz": hz, "color": col, "marker": mk,
             "n": int(N), "repro_ok": meta["repro"]["ok"], "val24988": meta["repro"]["got"],
             "n_alive_modes": int(alive.sum())}

        v_ch = np.asarray(pz["v"][:N], np.float64)
        s_ch = np.asarray(pz["s"][:N], np.float64)
        d_ch = np.asarray(pz["d"][:N], np.float64)
        pops = {"alive": (traj[alive], a_ch[alive], dth[alive], th[alive], h_ch[alive],
                          v_ch[alive], s_ch[alive], d_ch[alive]),
                "top1": (traj[ar, top1], a_ch[ar, top1], dth[ar, top1], th[ar, top1],
                         h_ch[ar, top1], v_ch[ar, top1], s_ch[ar, top1], d_ch[ar, top1])}
        r["m1"], r["m2"], r["m3"] = {}, {}, {}
        for pn, (P, A, D, TH, H, V, Sc, Dc) in pops.items():
            st, k, Rc = m1_stats(P, f"{name} ({pn})")
            # 자기참조 지표(기존 계기)와 같은 식: |Δθ| > 7.3°/step, 모집단은 같은 모드 집합
            dth_deg = np.degrees(np.abs(np.diff(TH, axis=-1)))
            dh_deg = np.degrees(np.abs(C.wrap(np.diff(H, axis=-1))))
            st["self_dtheta_over_pct"] = float(100.0 * (dth_deg > LABEL_DTHETA_DEG).mean())
            st["aux_dh_over_pct"] = float(100.0 * (dh_deg > LABEL_DTHETA_DEG).mean())
            # 같은 스텝 쌍(양쪽 유효)에서만 비교해야 공정하다
            o2 = k["ok2"]
            st["self_dtheta_over_pct_sameden"], _ = pct(dth_deg[..., 1:] > LABEL_DTHETA_DEG, o2)
            st["aux_dh_over_pct_sameden"], _ = pct(dh_deg[..., 1:] > LABEL_DTHETA_DEG, o2)
            # 분해: Δψ = Δθ + Δk(경로 곡률) + 기하. 가산 분해가 아니라 각 항의 크기 + 반사실이다.
            dpsi_deg = np.degrees(k["dpsi"])
            dh_sig = np.degrees(C.wrap(np.diff(H, axis=-1)))[..., 1:]
            dth_sig = np.degrees(np.diff(TH, axis=-1))[..., 1:]
            st["mean_abs_dh_deg"] = float(np.abs(dh_sig[o2]).mean())
            st.update(decompose(dpsi_deg, dh_sig, dth_sig, o2))
            st["mean_abs_geom_gap_deg"] = st["mean_abs_geom_deg"]      # 이전 이름 유지
            st["mean_abs_curv_gap_deg"] = st["mean_abs_dk_deg"]
            st["mv_sensitivity"] = mv_sensitivity(P)
            # 프레네 접힘 (적분기 기하 결함)
            fd = frenet_fold(P, H, Sc, Dc, TH)
            fg, fdt = fd["fold_geo"], fd["fold_dot"]
            st["fold_geo_pct"] = float(100.0 * fg.mean())
            st["fold_dot_pct"] = float(100.0 * fdt.mean())
            st["fold_agree_pct"] = float(100.0 * (fg == fdt).mean())
            inter, union = float((fg & fdt).sum()), float((fg | fdt).sum())
            st["fold_jaccard"] = inter / union if union > 0 else float("nan")
            st["fold_geo_over_dot"] = (st["fold_geo_pct"] / st["fold_dot_pct"]
                                       if st["fold_dot_pct"] > 0 else float("nan"))
            st["fold_scen_pct"] = float(100.0 * fdt.any(-1).mean())
            st["kd_p99"] = float(np.percentile(fd["kd"], 99))
            st["kd_max"] = float(fd["kd"].max())
            # 위반 쌍 중 접힘이 낀 비율
            viol = (np.degrees(np.abs(k["dpsi"])) > LABEL_DTHETA_DEG) & o2
            near_fold = fdt[..., :-1] | fdt[..., 1:]
            st["viol_with_fold_pct"], _ = pct(near_fold, viol) if viol.any() else (float("nan"), 0)
            st["fold_viol_share_pct"] = float(
                100.0 * (viol & near_fold).sum() / max(viol.sum(), 1))
            st.update(by_speed(k, Rc))
            r["m1"][pn] = st

            J = jerk_pipes(P, a_action=A, v_src=V)
            r["m2"][pn] = {p: jerk_stats(J[p]) for p in J}
            r["m2"][pn]["wass_vs_gt"] = {"J1_vs_gtJ2": wass(J["J1"], Jg["J2"]),
                                         "J2": wass(J["J2"], Jg["J2"]),
                                         "J3": wass(J["J3"], Jg["J3"]),
                                         "J4": wass(J["J4"], Jg["J4"]),
                                         "J2v": wass(J["J2v"], Jg["J2v"]),
                                         "J1_vs_gtJ4": wass(J["J1"], Jg["J4"]),
                                         "J3_vs_gtJ4": wass(J["J3"], Jg["J4"])}

            # 측정 3 — a(t) [m/s²] 와 잔차 요레이트 dθ/dt [°/s]
            yr = np.degrees(D) / DT
            f, Ma = spectrum(A)
            _, Mt = spectrum(yr)
            ba, ta = band_amp(f, Ma)
            bt, tt = band_amp(f, Mt)
            r["m3"][pn] = {
                "Sm_a": float(caps_sm(f, Ma).mean()), "Sm_dtheta": float(caps_sm(f, Mt).mean()),
                "band_a": (ba.mean(0)).tolist(), "band_dtheta": (bt.mean(0)).tolist(),
                "band_a_frac": (ba.mean(0) / max(ta.mean(), 1e-12)).tolist(),
                "band_dtheta_frac": (bt.mean(0) / max(tt.mean(), 1e-12)).tolist(),
                "tot_a": float(ta.mean()), "tot_dtheta": float(tt.mean()),
                "spec_a": Ma.mean(0).tolist(), "spec_dtheta": Mt.mean(0).tolist(),
                "freq": f.tolist()}
            if pn == "alive":
                keep[tag] = {"dpsi_deg": sub(np.degrees(np.abs(k["dpsi"]))[o2]),
                             "R": sub(k["R"][o2]), "kd": sub(fd["kd"]),
                             **{f"J_{p}": sub(J[p]) for p in J}}

        # 상황별
        cls_t = df["cls"].to_numpy()
        v0b = C.bin_labels(df["v0_f"].to_numpy(), "v0")
        cm = np.repeat(cls_t[:, None], 6, 1)[alive]
        P_al, A_al = traj[alive], a_ch[alive]
        J_al = jerk_pipes(P_al, a_action=A_al, v_src=v_ch[alive])
        r["m1_by_cls"], r["m2_by_cls"] = {}, {}
        fd_al = frenet_fold(P_al, h_ch[alive], s_ch[alive], d_ch[alive], th[alive])
        # top-1 모집단 (실제 예측) — 리포트 표의 기본
        P_t1 = traj[ar, top1]
        J_t1 = jerk_pipes(P_t1, a_action=a_ch[ar, top1], v_src=v_ch[ar, top1])
        fd_t1 = frenet_fold(P_t1, h_ch[ar, top1], s_ch[ar, top1], d_ch[ar, top1], th[ar, top1])
        r["m1_by_cls_top1"], r["m2_by_cls_top1"], r["m1_by_v0_top1"] = {}, {}, {}
        for cn in C.CLASSES:
            mt = cls_t == cn
            if mt.sum() >= 5:
                s1, _, _ = m1_stats(P_t1[mt], cn)
                s1["fold_dot_pct"] = float(100.0 * fd_t1["fold_dot"][mt].mean())
                s1["n_scen"] = int(mt.sum())
                r["m1_by_cls_top1"][cn] = s1
                r["m2_by_cls_top1"][cn] = {pp: jerk_stats(J_t1[pp][mt]) for pp in ("J1", "J2", "J3", "J4")}
                r["m2_by_cls_top1"][cn]["n_scen"] = int(mt.sum())
        for lb in C.BINS["v0"][1]:
            mt = v0b == lb
            if mt.sum() >= 5:
                s1, _, _ = m1_stats(P_t1[mt], lb)
                s1["n_scen"] = int(mt.sum())
                r["m1_by_v0_top1"][lb] = s1
        for cn in C.CLASSES:
            m = cm == cn
            if m.sum() < 5:
                continue
            s_, _, _ = m1_stats(P_al[m], cn)
            s_["fold_dot_pct"] = float(100.0 * fd_al["fold_dot"][m].mean())
            s_["fold_scen_pct"] = float(100.0 * fd_al["fold_dot"][m].any(-1).mean())
            r["m1_by_cls"][cn] = s_
            r["m2_by_cls"][cn] = {p: jerk_stats(J_al[p][m]) for p in ("J1", "J2", "J3", "J4")}
        # v0 구간별 (alive)
        vm = np.repeat(v0b[:, None], 6, 1)[alive]
        r["m1_by_v0"] = {}
        for lb in C.BINS["v0"][1]:
            m = vm == lb
            if m.sum() < 5:
                continue
            s_, _, _ = m1_stats(P_al[m], lb)
            s_["fold_dot_pct"] = float(100.0 * fd_al["fold_dot"][m].mean())
            r["m1_by_v0"][lb] = s_
        # 시나리오 단위 (결정 트리용): top-1 모드의 저크 p90 과 위반 여부
        Pt = traj[ar, top1]
        Jt = jerk_pipes(Pt, a_action=a_ch[ar, top1], v_src=v_ch[ar, top1])
        kt = xy_kin(Pt)
        o2t = kt["ok2"]
        scen = {
            "jerk_p90_J3": np.nanpercentile(np.where(np.isfinite(Jt["J3"]), np.abs(Jt["J3"]), np.nan),
                                            90, axis=1),
            "jerk_p90_J2": np.nanpercentile(np.where(np.isfinite(Jt["J2"]), np.abs(Jt["J2"]), np.nan),
                                            90, axis=1),
            "jerk_p90_J4": np.nanpercentile(np.where(np.isfinite(Jt["J4"]), np.abs(Jt["J4"]), np.nan),
                                            90, axis=1),
            "jerk_mean_J3": np.abs(Jt["J3"]).mean(1),
            "jerk_mean_J4": np.abs(Jt["J4"]).mean(1),
            "any_over_label": ((np.degrees(np.abs(kt["dpsi"])) > LABEL_DTHETA_DEG) & o2t).any(1),
            "any_R3": ((kt["R"] < R_MIN) & o2t).any(1),
            "n_valid_pair": o2t.sum(1),
            "fold_any": frenet_fold(Pt, h_ch[ar, top1], s_ch[ar, top1], d_ch[ar, top1],
                                    th[ar, top1])["fold_dot"].any(1),
        }
        # 프레네 접힘의 정확도 영향 (상관이지 인과가 아니다 — 접힘은 |d| 가 크게 벗어난 모드에서 생긴다)
        fold_al_any = fd_al["fold_dot"].any(-1)
        ade_al = np.asarray(pz["ade"][:N], np.float64)[alive]
        t1_fold = fd_t1["fold_dot"].any(-1)
        mn = np.asarray(pz["minade"][:N], np.float64)
        r["fold_impact"] = {
            "mode_ade_folded": float(np.nanmean(ade_al[fold_al_any])) if fold_al_any.any() else float("nan"),
            "mode_ade_clean": float(np.nanmean(ade_al[~fold_al_any])),
            "n_mode_folded": int(fold_al_any.sum()),
            "top1_fold_scen_n": int(t1_fold.sum()),
            "top1_fold_scen_pct": float(100.0 * t1_fold.mean()),
            "minade6_top1_folded": float(mn[t1_fold].mean()) if t1_fold.any() else float("nan"),
            "minade6_top1_clean": float(mn[~t1_fold].mean()),
            "minade6_all": float(mn.mean()),
            # 상한: 접힌 시나리오를 깨끗한 시나리오 평균으로 바꾸면 전체가 얼마나 내려가나
            "minade6_upper_bound_gain": float(
                mn.mean() - np.where(t1_fold, mn[~t1_fold].mean(), mn).mean()) if t1_fold.any() else 0.0,
        }
        np.savez(DATA / f"scen_{tag}.npz", **{k: np.asarray(v) for k, v in scen.items()})
        res["tags"][tag] = r
        print(f"[{tag}] alive: xy 7.3°초과 {r['m1']['alive']['over_label_pct']:.3f}% "
              f"(자기참조 {r['m1']['alive']['self_dtheta_over_pct_sameden']:.3f}%)  "
              f"R<3m {r['m1']['alive']['R_under3_pct']:.3f}%  "
              f"|저크|J2 {r['m2']['alive']['J2']['mean_abs']:.2f} J3 {r['m2']['alive']['J3']['mean_abs']:.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)
        del pz, traj, a_ch, dth, th, h_ch, v_ch, s_ch, d_ch

    # ---------- 창함수 민감도: 30에폭 ÷ 15에폭 대역비가 창 선택에 얼마나 흔들리나
    import pandas as _pd  # noqa: F401
    win_cases = [("hann", "linear"), ("hann", "none"), ("rect", "linear"), ("rect", "none")]
    pair = ("v4_l4nw_ah2_full_sm1_s0", "v4_l4nw_ah2_full_sm1_cos30_s0")
    if all(t in res["tags"] for t in pair):
        sens = {}
        cache = {}
        for t in pair:
            z2 = np.load(C.tag_dirs(t)["data"] / "pred.npz")
            al2 = z2["alive"][:N]
            cache[t] = (np.asarray(z2["a"][:N], np.float64)[al2],
                        np.degrees(np.asarray(z2["dtheta"][:N], np.float64)[al2]) / DT)
            del z2
        for w, dt_ in win_cases:
            row = {}
            for nm, ch in (("a", 0), ("dtheta", 1)):
                bb = []
                for t in pair:
                    f2, M2 = spectrum(cache[t][ch], window=w, detrend=dt_)
                    b2, _ = band_amp(f2, M2)
                    bb.append(b2.mean(0))
                row[nm] = (bb[1] / bb[0]).tolist()
                row[nm + "_n_over_1"] = int((bb[1] > bb[0]).sum())
            sens[f"{w}+{dt_}"] = row
        res["window_sensitivity"] = {"pair": list(pair), "cases": sens,
                                     "note": "값은 30에폭 코사인 ÷ 15에폭 상수의 대역 진폭비"}
        del cache
        print("[win] 창함수 민감도 " + "  ".join(
            f"{k}: a 1초과 {v['a_n_over_1']}/4, dθ {v['dtheta_n_over_1']}/4" for k, v in sens.items()),
            flush=True)

    # ---------- 정답 시나리오 단위 (트리용)
    Jg_s = jerk_pipes(Y, v_src=v_fld_fut)
    kg = xy_kin(Y)
    np.savez(DATA / "scen_gt.npz",
             jerk_p90_J3=np.percentile(np.abs(Jg_s["J3"]), 90, axis=1),
             jerk_p90_J2=np.percentile(np.abs(Jg_s["J2"]), 90, axis=1),
             jerk_p90_J4=np.percentile(np.abs(Jg_s["J4"]), 90, axis=1),
             jerk_mean_J3=np.abs(Jg_s["J3"]).mean(1), jerk_mean_J4=np.abs(Jg_s["J4"]).mean(1),
             any_over_label=((np.degrees(np.abs(kg["dpsi"])) > LABEL_DTHETA_DEG) & kg["ok2"]).any(1),
             any_R3=((kg["R"] < R_MIN) & kg["ok2"]).any(1))

    # ---------- 정답 액션열 (측정 3 기준선) — viz_v4_labelfit.py 가 만든다
    ga = DATA / "gt_actions.npz"
    if ga.exists():
        z = np.load(ga, allow_pickle=False)
        A, D = np.asarray(z["a"], np.float64), np.asarray(z["dtheta"], np.float64)
        # 저속 잡음이 심해 두 갈래로 낸다: 전체 / 움직인 시나리오(미래 최소 속력 >= MOVE_V)
        mv_all = np.asarray(z["moving"]).all(1)
        f, Ma = spectrum(A)
        _, Mt = spectrum(np.degrees(D) / DT)
        ba, ta = band_amp(f, Ma)
        bt, tt = band_amp(f, Mt)
        gj = {"n": int(len(A)), "n_moving": int(mv_all.sum()),
              "Sm_a": float(caps_sm(f, Ma).mean()), "Sm_dtheta": float(caps_sm(f, Mt).mean()),
              "band_a": ba.mean(0).tolist(), "band_dtheta": bt.mean(0).tolist(),
              "band_a_frac": (ba.mean(0) / ta.mean()).tolist(),
              "band_dtheta_frac": (bt.mean(0) / tt.mean()).tolist(),
              "tot_a": float(ta.mean()), "tot_dtheta": float(tt.mean()),
              "spec_a": Ma.mean(0).tolist(), "spec_dtheta": Mt.mean(0).tolist(), "freq": f.tolist()}
        if mv_all.any():
            gj["moving"] = {
                "Sm_a": float(caps_sm(f, Ma[mv_all]).mean()),
                "Sm_dtheta": float(caps_sm(f, Mt[mv_all]).mean()),
                "band_a": ba[mv_all].mean(0).tolist(), "band_dtheta": bt[mv_all].mean(0).tolist(),
                "band_a_frac": (ba[mv_all].mean(0) / ta[mv_all].mean()).tolist(),
                "band_dtheta_frac": (bt[mv_all].mean(0) / tt[mv_all].mean()).tolist(),
                "spec_a": Ma[mv_all].mean(0).tolist(), "spec_dtheta": Mt[mv_all].mean(0).tolist()}
        res["gt"]["m3"] = gj
        res["gt"]["m2"]["J1_gt_action"] = jerk_stats(np.diff(A, axis=-1) / DT)
        keep["gt"]["J_J1"] = sub(np.diff(A, axis=-1) / DT)
        print(f"[gt] 액션열 기준선 n={len(A)} (움직인 {int(mv_all.sum())})  "
              f"Sm_a {gj['Sm_a']:.4f}", flush=True)
    else:
        print("[gt] gt_actions.npz 가 없다 — 먼저 viz_v4_labelfit.py 를 돌려라", flush=True)

    np.savez(DATA / "dists.npz",
             **{f"{k}__{kk}": sub(vv, KEEP_N, seed=1) for k, v in keep.items() for kk, vv in v.items()})
    (DATA / "feas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"[done] compute {time.time()-t_all:.0f}s → {DATA/'feas.json'}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--figs-only", dest="figs_only", action="store_true")
    ap.add_argument("--no-figs", dest="no_figs", action="store_true")
    a = ap.parse_args()
    if not a.figs_only:
        compute(TAGS, a.limit)
    if not a.no_figs:
        import viz_v4_feas_figs as F
        F.main()
