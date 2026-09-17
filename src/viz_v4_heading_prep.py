"""
viz_v4_heading_prep.py - v4 입력 전처리 (a, h) 의 10 Hz / 2 Hz 비교 그림. 모델 결과가 아니라 **입력 자체**를 본다.

무엇을 읽나 (전부 val, 세 캐시는 정렬된 val 디렉터리 순서라 인덱스가 같다 — scenario_id.json 으로 확인)
  focal 10 Hz  C.VAL_CACHES["ah2"]      x (N,50,2) = (a, h)   heading_decomp.ah_features
  focal 2 Hz   C.VAL_CACHES["ah2_2hz"]  x (N,10,2) = (a, h)   heading_decomp.ah_features_2hz
  주변 차량    AGENTS (prep_agents_heading.py)  h·h_src (N,32,50), pos0, dist0, atype, is_av, n_cand
  원본         parquet (AV2 heading·속도 필드·과거 위치) + 지도 JSON
  a = 속도 증분 [km/h] / 3 (첫 값 = 시작 속도) → 누적합 × 3 / 3.6 = 속력 [m/s]
  h = 진행방향 [rad], focal 프레임(원점 = focal pos[49], 회전 = focal AV2 heading[49]), (−π, π] wrap

주변 차량 2 Hz 는 캐시가 없다. 관측 50스텝이 모두 있는 차량만 ah_features_2hz(pos[:50], heading[:50], θ_focal) 로
즉석 계산한다(중간이 끊긴 트랙은 그리지 않는다).

단계
  scan    원본 val 전체를 병렬(≤16)로 한 번 읽는다. 동시에 검증한다:
          ① focal 10 Hz·2 Hz 를 원본에서 다시 만들어 캐시와 대조 ② 주변 차량 캐시를 agents_one 으로 다시 만들어 대조
          ③ 주변 차량 위치차분 h 를 원본 위치로 따로 계산해 대조 ④ 그림용 2 Hz 펼침(ah2hz_detail)이 원 함수와 같은지
  picks   상황별 시나리오 1개씩 (시드 0, 규칙은 PICK_RULES) → data/picks.json
  panels  시나리오 패널 6장 → panel_<n>_<상황>.png
  dist    분포 그림 → dist_*.png, 숫자는 data/summary.json

  python src/viz_v4_heading_prep.py                    # 전부 (scan 은 결과가 있으면 건너뛴다)
  python src/viz_v4_heading_prep.py --stage scan --limit 300 --tag dev
출력: viz/v4/heading_prep/  (데이터는 data/)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C
import heading_decomp as HD
import prep_agents_heading as PA

OUT = C.VIZ_ROOT / "heading_prep"
CACHE10 = C.VAL_CACHES["ah2"]
CACHE2 = C.VAL_CACHES["ah2_2hz"]
AGENTS = Path("/data/argoverse2/cache/v4/agents_hist_val_361221aa2f2e23f3_n24988")
AG_FIELDS = ("h", "h_src", "pos0", "dist0", "atype", "is_av", "n_cand")
OBS, K = 50, PA.K
KMH = 3.0 / 3.6                                    # a 채널 1 = 3 km/h → m/s
IDX2 = np.arange(OBS - 1, HD.RAMP_SKIP - 1, -HD.DS_STEP)[::-1]     # (4, 9, …, 49)
T10 = (np.arange(OBS) - (OBS - 1)) * HD.DT        # −4.9 … 0 s
T2 = T10[IDX2]                                     # −4.5 … 0 s
TYPE_NAMES = {1: "vehicle", 2: "bus", 3: "motorcyclist", 4: "cyclist"}
TYPE_KO = {1: "승용·트럭", 2: "버스", 3: "오토바이", 4: "자전거"}
SRC_NAMES = {0: "미관측", 1: "위치차분", 2: "AV2 보조", 3: "이웃 유지"}
# 스텝 속력(AV2 속도 필드) 구간 — 0 번은 미관측 스텝
SPD_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
SPD_LAB = ["<0.5", "0.5–1", "1–2", "2–5", "5–10", "≥10"]
DIST_EDGES = [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, np.inf]
DIST_LAB = ["<10", "10–20", "20–30", "30–50", "50–100", "≥100"]
# 트랙 단위: 과거 4.9 s 관측 스텝의 AV2 속력 최대
TRK_EDGES = [0.0, 0.5, 2.0, 5.0, np.inf]
TRK_LAB = ["<0.5\n(4.9 s 내내 정지)", "0.5–2", "2–5", "≥5"]
RAMP_REF = slice(10, 16)        # 램프 기준 속력: 창 시작 후 1.0~1.6 s 의 위치차분 속력 평균 (메모리 정의와 같다)
RAMP_MIN_V = 3.0                # 기준 속력이 이보다 큰 focal 만
MOVE_V = HD.V_MIN               # 위치차분 구간 = 출처 1 (스텝 속력 ≥ 1 m/s)

# 그림 색 — 밝은 면 (dataviz 기본 팔레트, viz_v4_common)
COL10, COL2, COLAV2 = C.C_BLUE, C.C_RED, C.INK2
SRC_COL = {0: "#c3c2b7", 1: C.C_AQUA, 2: C.C_YELLOW, 3: C.C_VIOLET}
SRC_BAND = {0: "#ecebe6", 1: "#dcf2e8", 2: "#fbefcc", 3: "#e4e0f5"}
# 어두운 지도 위
M_SRC = {0: "#9aa0a8", 1: "#5ef0a8", 2: "#ffc233", 3: "#b69cff"}
M_FOCAL, M_PAST, M_2HZ, M_NB, M_NBSEL = "#35a2ff", "#f5f5f5", "#ff6b6b", "#6b707a", "#ff6fd2"
BOX = {1: (4.6, 1.9), 2: (12.0, 2.6), 3: (2.2, 0.8), 4: (1.8, 0.6)}


def wrap(a):
    return HD.wrap(a)


def deg(a):
    return np.degrees(np.asarray(a, np.float64))


# ============================================================================ 공통 계산
def focal_raw(scn):
    """dataset_lane 과 같은 dtype(float32)으로 focal 110스텝."""
    f = next(t for t in scn.tracks if t.track_id == scn.focal_track_id)
    st = sorted(f.object_states, key=lambda x: x.timestep)
    pos = np.array([x.position for x in st], dtype=np.float32)
    vel = np.array([x.velocity for x in st], dtype=np.float32)
    head = np.array([x.heading for x in st], dtype=np.float32)
    return pos, vel, head


def ah2hz_detail(pos_obs, head_obs, yaw0):
    """heading_decomp.ah_features_2hz 를 한 단계씩 펼친 것 (그림용: 평활 위치 · 출처 코드 · 속력).

    계산 순서는 원 함수와 같다. 결과 h 가 원 함수와 같은지 부르는 쪽에서 매번 대조한다.
    """
    from scipy.signal import savgol_filter
    pos_obs = np.asarray(pos_obs, dtype=np.float64)
    T = len(pos_obs)
    idx = np.arange(T - 1, HD.RAMP_SKIP - 1, -HD.DS_STEP)[::-1]
    sm = np.full_like(pos_obs, np.nan)
    sm[HD.RAMP_SKIP:] = savgol_filter(pos_obs[HD.RAMP_SKIP:], HD.SG_WINDOW, HD.SG_ORDER, axis=0, mode="interp")
    p = sm[idx]
    dt_s = HD.DT * HD.DS_STEP
    h_ref = HD.guard(np.asarray(head_obs, dtype=np.float64))[idx]
    h_city, src = HD.build_heading(p, h_ref=h_ref, dt=dt_s, v_min=HD.V_MIN, jump_deg=180.0)
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt_s
    h = wrap(h_city - float(yaw0))
    return {"idx": idx, "p": p, "h_city": h_city, "h": h, "h32": h.astype(np.float32), "src": src,
            "v": np.append(v, v[-1])}


def agent_tracks(scn, origin32):
    """prep_agents_heading.agents_one 과 같은 후보·정렬 (슬롯 → 트랙). 정렬 키가 같아 순서도 같다."""
    cands = []
    for t in scn.tracks:
        if t.track_id == scn.focal_track_id:
            continue
        code = PA.AGENT_TYPES.get(t.object_type.value)
        if code is None:
            continue
        past = {o.timestep: o for o in t.object_states if o.timestep < OBS}
        if (OBS - 1) not in past:
            continue
        p0 = np.asarray(past[OBS - 1].position, dtype=np.float32)
        cands.append((float(np.linalg.norm(p0 - origin32)), code, t.track_id == "AV", past, t.track_id))
    cands.sort(key=lambda c: c[0])
    out = []
    for d, code, av, past, tid in cands[:K]:
        P = np.full((OBS, 2), np.nan)
        H = np.full(OBS, np.nan)
        V = np.full(OBS, np.nan)
        obs = np.zeros(OBS, bool)
        for ts, o in past.items():
            P[ts] = o.position
            H[ts] = o.heading
            V[ts] = float(np.hypot(*o.velocity))
            obs[ts] = True
        out.append({"tid": tid, "code": code, "av": av, "d": d, "P": P, "H": H, "V": V, "obs": obs})
    return out, len(cands)


def jumps(h, obs=None):
    """입력값이 ±π 경계를 넘어 튄 곳: 관측된 이웃 스텝 사이의 **wrap 하지 않은** 차가 π 를 넘는 곳 (앞 스텝 인덱스)."""
    h = np.asarray(h, np.float64)
    idx = np.arange(len(h)) if obs is None else np.flatnonzero(obs)
    if len(idx) < 2:
        return np.zeros(0, int)
    d = np.abs(np.diff(h[idx]))
    return idx[:-1][d > np.pi]


def flip_decision(ref_g, h_pd, m):
    """build_heading 안 align_ref 의 판정을 다시 한다: 위치차분 스텝 m 이 3개 미만이면 −1(판정 안 함), 아니면 뒤집음 1 / 그대로 0."""
    m = np.asarray(m, bool)
    if m.sum() < 3:
        return -1
    return int(np.median(np.abs(wrap(ref_g[m] - h_pd[m]))) > np.pi / 2)


def flip_observed(ref_g, h_out, fill):
    """출력 쪽에서 본 뒤집힘 (판정 재구현과 다른 경로): AV2 로 메운 스텝의 h 가 가드된 AV2 와
    0 차이면 0, π 차이면 1, 섞이면 9, 메운 스텝이 없으면 −1."""
    fill = np.asarray(fill, bool)
    if not fill.any():
        return -1
    d = np.abs(wrap(np.asarray(h_out, np.float64)[fill] - ref_g[fill]))
    if (d < 1e-3).all():
        return 0
    if (d > np.pi - 1e-3).all():
        return 1
    return 9


# ============================================================================ 1. scan
_G = {}
AG_COLS = ["scen", "slot", "atype", "is_av", "dist0", "vmax", "v49", "path", "n_obs", "full_obs",
           "n_src1", "n_src2", "n_src3", "jump10", "jump2", "n2_av2", "flip", "h2_err",
           "flip2", "fill_dis", "n_src1_slow", "n2_src1", "lane_m", "ag_av2", "ag_10", "ag_2", "lane_dev",
           "flip_obs", "flip2_obs", "src2_last"]
LANE_IN_M = 1.0      # 주변 차량 t=0 위치가 교차로 밖 차로 중심선에서 이 안이면 '차로 안' — 그 차로 방향을 방향의 참고값으로 쓴다


def _init(sids):
    _G["sids"] = sids
    _G["x10"] = np.load(CACHE10 / "x.npy", mmap_mode="r")
    _G["x2"] = np.load(CACHE2 / "x.npy", mmap_mode="r")
    for k in ("origin", "theta"):
        _G[k] = np.load(CACHE10 / f"{k}.npy", mmap_mode="r")
    for k in AG_FIELDS:
        _G["ag_" + k] = np.load(AGENTS / f"{k}.npy", mmap_mode="r")


def scan_one(i):
    from av2.datasets.motion_forecasting import scenario_serialization as ss
    from dataset_map import _rotation_matrix
    from scipy.signal import savgol_filter
    sid = _G["sids"][i]
    sdir = C.VAL_DIR / sid
    scn = ss.load_argoverse_scenario_parquet(sdir / f"scenario_{sid}.parquet")
    pos, vel, head = focal_raw(scn)
    origin, theta = pos[OBS - 1].copy(), head[OBS - 1].copy()
    R = _rotation_matrix(-theta)
    pn = ((pos - origin) @ R.T)[:OBS]
    x10 = np.asarray(_G["x10"][i], np.float64)
    x2 = np.asarray(_G["x2"][i], np.float64)

    # ---- focal: 원본에서 다시 만들어 캐시와 대조
    f10, _ = HD.ah_features(pos[:OBS], head[:OBS], float(theta))
    f2 = HD.ah_features_2hz(pos[:OBS], head[:OBS], float(theta))
    det = ah2hz_detail(pos[:OBS], head[:OBS], float(theta))
    err = [float(np.abs(f10[:, 0] - x10[:, 0]).max()), float(np.abs(wrap(f10[:, 1] - x10[:, 1])).max()),
           float(np.abs(f2[:, 0] - x2[:, 0]).max()), float(np.abs(wrap(f2[:, 1] - x2[:, 1])).max()),
           float(np.abs(det["h32"] - f2[:, 1]).max()),         # 원 함수는 float32 로 내보낸다 → 같은 캐스트로 비트 대조
           float(max(np.abs(origin - _G["origin"][i]).max(), abs(theta - _G["theta"][i])))]
    pos64, head64 = pos[:OBS].astype(np.float64), head[:OBS].astype(np.float64)
    h10c, src10 = HD.build_heading(pos64, h_ref=head64)
    head_n = wrap(head64 - float(theta))
    vf = np.linalg.norm(vel[:OBS].astype(np.float64), axis=1)
    # 램프 비교용: 원본 위치 그대로 0.5초 속력 (인덱스 4→9), 평활을 인덱스 0 부터 했다면(가정)의 첫 표본 속력
    v2_raw0 = float(np.linalg.norm(pos64[9] - pos64[4]) / 0.5)
    sm_all = savgol_filter(pos64, HD.SG_WINDOW, HD.SG_ORDER, axis=0, mode="interp")
    v2_alt0 = float(np.linalg.norm(sm_all[9] - sm_all[4]) / 0.5)
    # 방향만 바꿔 굴린 복원 끝점 오차 (스텝 길이는 위치 그대로): AV2 heading / h — 10 Hz(4.9 s), 2 Hz(평활 점, 4.5 s)
    p2 = det["p"]
    rec = [float(np.linalg.norm(HD.rollout(pos64, HD.guard(head64))[-1] - pos64[-1])),
           float(np.linalg.norm(HD.rollout(pos64, h10c)[-1] - pos64[-1])),
           float(np.linalg.norm(HD.rollout(p2, HD.guard(head64)[IDX2])[-1] - p2[-1])),
           float(np.linalg.norm(HD.rollout(p2, det["h_city"])[-1] - p2[-1]))]

    # focal 뒤집힘 판정 (10 Hz / 2 Hz) — 판정 재구현과 출력에서 본 값
    hg = HD.guard(head64)
    fflip = [flip_decision(hg, h10c, src10 == 1),
             flip_observed(hg, x10[:, 1] + float(theta), src10 == 2),
             flip_decision(hg[IDX2], det["h_city"], det["src"] == 1),
             flip_observed(hg[IDX2], x2[:, 1] + float(theta), det["src"] == 2)]

    # 차로 대비 횡이동 (차선변경 고르기용): θ = h − k(최근접 같은 방향 중심선 접선), Σ 스텝길이·sin θ
    lane = [np.nan, np.nan, np.nan, np.nan]
    fld = None
    try:
        fld = HD.lane_field(HD.graph_of(sdir), near_xy=pos64[-1])
        if fld is not None:
            _, th, valid, _, inter = HD.decompose(pos64, h10c, fld)
            sl = np.linalg.norm(np.diff(pos64, axis=0), axis=1)
            lane = [float((sl * np.sin(th[:-1])).sum()), float(np.degrees(np.abs(th[:-1])).max()),
                    float(valid[:-1].mean()), float((inter[:-1] > 0).mean())]
    except Exception:
        pass

    # ---- 주변 차량: agents_one 으로 다시 만들어 캐시와 비트 대조
    ag = {k: np.asarray(_G["ag_" + k][i]) for k in AG_FIELDS}
    _, re = PA.agents_one(sdir, with_pos=True, loader=lambda _p: scn)
    ag_mis = [k for k in AG_FIELDS if not np.array_equal(re[k], ag[k])]
    tracks, n_cand = agent_tracks(scn, origin)
    R64 = R.astype(np.float64)
    rows = []
    H = np.zeros((4, len(SPD_EDGES), len(DIST_EDGES) - 1, 4), np.int64)   # [출처, 속력(0=미관측), 거리, 종류]
    HT = np.zeros((4, len(TRK_EDGES) - 1, 4), np.int64)                   # [출처, 트랙 최대속력, 종류] (관측 스텝)
    chk_n, chk_err, chk_spd_mis, map_mis = 0, 0.0, 0, 0
    for s, tr in enumerate(tracks):
        obs, P, Hh, V = tr["obs"], tr["P"], tr["H"], tr["V"]
        hc = ag["h"][s].astype(np.float64)
        sc = ag["h_src"][s]
        # 슬롯 → 트랙 대응 확인 (agents_one 이 구운 위치와 같은가)
        pn_s = (P.astype(np.float32) - origin) @ R.T
        if not np.array_equal(re["pos"][s][obs], pn_s[obs]):
            map_mis += 1
        # ③ 위치차분 h 독립 계산: 관측된 이웃 스텝끼리 방향 − θ_focal
        oi = np.flatnonzero(obs)
        for a, b in zip(oi[:-1], oi[1:]):
            d = P[b] - P[a]
            fast = np.linalg.norm(d) / ((b - a) * HD.DT) >= HD.V_MIN
            if fast != (sc[a] == 1):
                chk_spd_mis += 1
            if sc[a] == 1:
                chk_n += 1
                chk_err = max(chk_err, abs(float(wrap(hc[a] - (np.arctan2(d[1], d[0]) - float(theta))))))
        if len(oi) >= 2 and sc[oi[-1]] == 1:
            chk_n += 1
            chk_err = max(chk_err, abs(float(wrap(hc[oi[-1]] - hc[oi[-2]]))))
        # 걸음 히스토그램
        code = tr["code"]
        db = int(np.searchsorted(DIST_EDGES, tr["d"], side="right")) - 1
        vmax = float(np.nanmax(V[obs]))
        tb = int(np.searchsorted(TRK_EDGES, vmax, side="right")) - 1
        for t in range(OBS):
            if obs[t]:
                sb = int(np.searchsorted(SPD_EDGES, V[t], side="right"))      # 1..6
                HT[sc[t], tb, code - 1] += 1
            else:
                sb = 0
            H[sc[t], sb, db, code - 1] += 1
        path = float(np.linalg.norm(np.diff(P[oi], axis=0), axis=1).sum()) if len(oi) > 1 else 0.0
        full = bool(obs.all())
        n1 = int(((sc == 1) & obs).sum())
        # 뒤집힘 판정 (align_ref 와 같은 기준): 위치차분 스텝이 3개 이상일 때만 가능
        flip = -1
        if n1 >= 3:
            m = obs & (sc == 1)
            hg = HD.guard(Hh)
            flip = int(np.median(np.abs(wrap(hg[m] - (hc[m] + float(theta))))) > np.pi / 2)
        j10 = len(jumps(hc, obs))
        j2 = n2 = flip2 = fdis = n21 = flip2o = src2l = -1
        flipo = flip_observed(HD.guard(Hh), hc + float(theta), obs & (sc == 2))
        h2err = np.nan
        if full:
            g2 = HD.ah_features_2hz(P, Hh, float(theta))
            d2 = ah2hz_detail(P, Hh, float(theta))
            h2err = float(np.abs(d2["h32"] - g2[:, 1]).max())
            j2 = len(jumps(g2[:, 1]))
            n2 = int((d2["src"] == 2).sum())
            n21 = int((d2["src"] == 1).sum())
            # 2 Hz 쪽 뒤집힘 판정 (build_heading 안의 align_ref 와 같은 기준 — 10 Hz 에서 가드, 뽑은 뒤 대조)
            m2 = d2["src"] == 1
            ref2 = HD.guard(Hh)[IDX2]
            flip2 = flip_decision(ref2, d2["h_city"], m2)
            flip2o = flip_observed(ref2, g2[:, 1].astype(np.float64) + float(theta), d2["src"] == 2)
            src2l = int(d2["src"][-1])
            # 두 판이 모두 AV2 로 메운 표본에서 방향이 90° 넘게 갈리는가 (= 뒤집힘 판정이 갈림)
            both = (d2["src"] == 2) & (sc[IDX2] == 2)
            fdis = int((np.abs(wrap(hc[IDX2] - g2[:, 1])) > np.pi / 2)[both].sum())
        n1s = int(((sc == 1) & obs & (V < 0.5)).sum())
        # 차로 방향과 대조 (참고값): t=0 위치가 교차로 밖 차로 중심선 LANE_IN_M 안이면 그 접선 방향과 같은 쪽(±90°)을 향하는가
        lane_m, ag_av2, ag_10, ag_2, ldev = 0, -1, -1, -1, np.nan
        if fld is not None:
            PL, AL, IL = fld
            dd = ((PL - P[OBS - 1]) ** 2).sum(axis=1)
            jn = int(dd.argmin())
            if dd[jn] <= LANE_IN_M ** 2 and IL[jn] == 0:
                lane_m = 1
                kdir = AL[jn]
                ag_av2 = int(np.cos(Hh[OBS - 1] - kdir) > 0)
                ag_10 = int(np.cos(hc[OBS - 1] + float(theta) - kdir) > 0)
                ldev = float(np.degrees(abs(wrap(hc[OBS - 1] + float(theta) - kdir))))
                if full:
                    ag_2 = int(np.cos(float(g2[-1, 1]) + float(theta) - kdir) > 0)
        rows.append([i, s, code, int(tr["av"]), tr["d"], vmax, float(V[OBS - 1]), path, int(obs.sum()), int(full),
                     n1, int(((sc == 2) & obs).sum()), int(((sc == 3) & obs).sum()), j10, j2, n2, flip, h2err,
                     flip2, fdis, n1s, n21, lane_m, ag_av2, ag_10, ag_2, ldev, flipo, flip2o, src2l])
    foc = {"head_n": head_n.astype(np.float32), "src10": src10.astype(np.int8), "vf": vf.astype(np.float32),
           "pn": pn.astype(np.float32), "src2": det["src"].astype(np.int8),
           "v2x": np.array([v2_raw0, v2_alt0, det["v"][0]], np.float32), "rec": np.array(rec, np.float32),
           "err": np.array(err, np.float64), "n_cand_raw": n_cand, "lane": np.array(lane, np.float32),
           "fflip": np.array(fflip, np.int8)}
    chk = {"ag_mis": ag_mis, "map_mis": map_mis, "n": chk_n, "err": chk_err, "spd_mis": chk_spd_mis}
    return i, foc, np.array(rows, np.float64).reshape(-1, len(AG_COLS)), H, HT, chk


def flip_consistency(AGT, ff):
    """뒤집힘 판정 재구현(flip_decision)과 캐시 출력에서 본 뒤집힘(flip_observed)이 맞는가.
    판정 −1(안 함)이면 출력은 0(그대로) 이어야 하고, 0/1 이면 같아야 한다 (메운 스텝이 없으면 −1 은 비교 제외)."""
    def cmp(dec, obs):
        m = obs >= 0
        want = np.where(dec < 0, 0, dec)
        return {"n_compared": int(m.sum()), "mismatch": int((want[m] != obs[m]).sum()), "mixed_9": int((obs == 9).sum())}
    c = {k: AGT[:, AG_COLS.index(k)].astype(int) for k in ("flip", "flip_obs", "flip2", "flip2_obs")}
    return {"agents_10hz": cmp(c["flip"], c["flip_obs"]),
            "agents_2hz": cmp(c["flip2"][c["flip2_obs"] > -2], c["flip2_obs"][c["flip2_obs"] > -2]),
            "focal_10hz": cmp(ff[:, 0].astype(int), ff[:, 1].astype(int)),
            "focal_2hz": cmp(ff[:, 2].astype(int), ff[:, 3].astype(int))}


def run_scan(a, data):
    sids = json.loads((CACHE10 / "scenario_id.json").read_text())
    s2 = json.loads((CACHE2 / "scenario_id.json").read_text())
    sg = json.loads((AGENTS / "scenario_id.json").read_text())
    dirs = sorted(d.name for d in C.VAL_DIR.iterdir() if d.is_dir())
    order_ok = {"ah2_vs_ah2_2hz": sids == s2, "ah2_vs_agents": sids == sg, "ah2_vs_sorted_val_dirs": sids == dirs}
    print(f"[scan] 인덱스 정렬 {order_ok}", flush=True)
    if not all(order_ok.values()):
        raise SystemExit("캐시 순서가 다르다 — 멈춘다")
    n = len(sids) if a.limit <= 0 else min(a.limit, len(sids))
    F = {"head_n": np.zeros((n, OBS), np.float32), "src10": np.zeros((n, OBS), np.int8),
         "vf": np.zeros((n, OBS), np.float32), "pn": np.zeros((n, OBS, 2), np.float32),
         "src2": np.zeros((n, len(IDX2)), np.int8), "v2x": np.zeros((n, 3), np.float32),
         "rec": np.zeros((n, 4), np.float32), "err": np.zeros((n, 6), np.float64),
         "n_cand_raw": np.zeros(n, np.int32), "lane": np.zeros((n, 4), np.float32),
         "fflip": np.zeros((n, 4), np.int8)}
    rows, Hs, HTs = [], np.zeros((4, len(SPD_EDGES), len(DIST_EDGES) - 1, 4), np.int64), \
        np.zeros((4, len(TRK_EDGES) - 1, 4), np.int64)
    chk = {"scen_field_mismatch": 0, "fields": {}, "map_mis": 0, "n": 0, "err": 0.0, "spd_mis": 0}
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(sids,)) as pool:
        for k, (i, foc, rw, H, HT, ck) in enumerate(pool.imap(scan_one, range(n), chunksize=16)):
            for key, v in foc.items():
                F[key][i] = v
            rows.append(rw)
            Hs += H
            HTs += HT
            if ck["ag_mis"]:
                chk["scen_field_mismatch"] += 1
                for f in ck["ag_mis"]:
                    chk["fields"][f] = chk["fields"].get(f, 0) + 1
            chk["map_mis"] += ck["map_mis"]
            chk["n"] += ck["n"]
            chk["err"] = max(chk["err"], ck["err"])
            chk["spd_mis"] += ck["spd_mis"]
            if (k + 1) % 5000 == 0 or k + 1 == n:
                print(f"[scan] {k + 1:,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    AGT = np.concatenate(rows) if rows else np.zeros((0, len(AG_COLS)))
    E = F["err"]
    ver = {
        "n": n, "order": order_ok, "sec": round(time.time() - t0, 1),
        "focal_10hz_vs_cache": {"a_max_abs": float(E[:, 0].max()), "h_max_abs_rad": float(E[:, 1].max()),
                                "n_exact_both": int(((E[:, 0] == 0) & (E[:, 1] == 0)).sum())},
        "focal_2hz_vs_cache": {"a_max_abs": float(E[:, 2].max()), "h_max_abs_rad": float(E[:, 3].max()),
                               "n_exact_both": int(((E[:, 2] == 0) & (E[:, 3] == 0)).sum())},
        "focal_2hz_detail_vs_function_h_max_rad": float(E[:, 4].max()),
        "cache_origin_theta_vs_raw_max": float(E[:, 5].max()),
        "agents_one_vs_cache": {"scenarios_with_mismatch": chk["scen_field_mismatch"], "fields": chk["fields"]},
        "agents_slot_track_map_mismatch": chk["map_mis"],
        "agents_posdiff_h_independent": {"n_steps": chk["n"], "max_abs_rad": chk["err"],
                                         "src1_vs_speed_rule_mismatch": chk["spd_mis"]},
        "agents_2hz_detail_vs_function_h_max_rad": float(np.nanmax(AGT[:, AG_COLS.index("h2_err")]))
        if len(AGT) and np.isfinite(AGT[:, AG_COLS.index("h2_err")]).any() else None,
        "n_cand_raw_vs_cache_mismatch": int((F["n_cand_raw"] != np.load(AGENTS / "n_cand.npy")[:n]).sum()),
        "flip_decision_vs_output": flip_consistency(AGT, F["fflip"]),
    }
    np.savez_compressed(data / "scan.npz", AGT=AGT, H=Hs, HT=HTs, **F)
    (data / "verify.json").write_text(json.dumps(ver, indent=2, ensure_ascii=False))
    print(json.dumps(ver, indent=2, ensure_ascii=False), flush=True)
    return ver


# ============================================================================ 로드 + 파생
class Scan:
    def __init__(self, data):
        z = np.load(data / "scan.npz")
        self.__dict__.update({k: z[k] for k in z.files})
        self.n = len(self.head_n)
        self.sids = json.loads((CACHE10 / "scenario_id.json").read_text())[:self.n]
        self.x10 = np.load(CACHE10 / "x.npy")[:self.n].astype(np.float64)
        self.x2 = np.load(CACHE2 / "x.npy")[:self.n].astype(np.float64)
        self.h10, self.h2 = self.x10[:, :, 1], self.x2[:, :, 1]
        self.v10 = np.cumsum(self.x10[:, :, 0], axis=1) * KMH
        self.v2 = np.cumsum(self.x2[:, :, 0], axis=1) * KMH
        self.ag = {k: np.load(AGENTS / f"{k}.npy")[:self.n] for k in AG_FIELDS}
        self.T = {c: self.AGT[:, j] for j, c in enumerate(AG_COLS)}

    def focal_feats(self):
        """고르기에 쓰는 focal 과거 4.9 s 특징."""
        n = self.n
        mv = self.src10 == 1
        dh = np.zeros(n)
        hmax = np.zeros(n)
        for i in range(n):
            # 순 방향 변화는 AV2 heading(차체, 튐 가드)으로 잰다 — 위치차분 h 는 저속 잡음·후진으로 180° 씩 튈 수 있다
            hu = np.unwrap(HD.guard(self.head_n[i].astype(np.float64)))
            dh[i] = np.degrees(hu[-3:].mean() - hu[:3].mean())
            t = np.flatnonzero(mv[i, :OBS - 1])
            if len(t):
                hmax[i] = np.degrees(np.abs(self.h10[i, t]).max())
        slip = np.abs(wrap(self.head_n[:, :OBS - 1].astype(np.float64) - self.h10[:, :OBS - 1]))
        rev = ((slip > np.pi / 2) & mv[:, :OBS - 1]).sum(1)
        pn = self.pn.astype(np.float64)
        return {"mv_frac": mv.mean(1), "av2_frac": (self.src10 == 2).mean(1), "dh": dh, "hmax": hmax,
                "vmean": self.vf.mean(1), "vrange": self.vf.max(1) - self.vf.min(1),
                "path": np.linalg.norm(np.diff(pn, axis=1), axis=2).sum(1),
                "y_start": pn[:, :10, 1].mean(1), "y_late": np.abs(pn[:, 40:, 1]).max(1),
                "vmax": self.vf.max(1), "rev_steps": rev, "lat_lane": self.lane[:, 0], "th_lane_max": self.lane[:, 1],
                "lane_valid": self.lane[:, 2], "lane_inter": self.lane[:, 3]}


# ============================================================================ 2. picks
PICK_ORDER = ["정속 직진", "좌회전", "우회전", "저속·정지", "차선변경", "혼잡", "정지 차량 뒤집힘",
              "정지 차량 많음", "focal ±π 점프", "큰 슬립각", "램프 뚜렷"]
PICK_RULES = {
    "정속 직진": "과거 50스텝 모두 위치차분(≥1 m/s) · AV2 속력 평균 ≥ 8 m/s · 최대−최소 < 1.5 m/s · 순 방향 변화 |Δh| < 3° · max|h| < 5°",
    "좌회전": "순 방향 변화 Δh > +45° (AV2 heading 의 마지막 3 − 처음 3 스텝 평균, unwrap) · 위치차분 스텝 ≥ 80% · "
            "AV2 속력 평균 ≥ 3 m/s · 위치차분 h 가 AV2 와 90° 넘게 다른 스텝 없음",
    "우회전": "Δh < −45° · 나머지는 좌회전과 같다",
    "저속·정지": "AV2 보조 스텝 50~90% · 과거 이동거리 ≥ 3 m (위치차분과 AV2 보조가 섞인 장면)",
    "차선변경": "50스텝 모두 위치차분 · 평균 ≥ 6 m/s · |Δh| < 10° · 모든 스텝에 5 m 안 같은 방향 차로(교차로 차로 아님) · 차로 대비 횡이동 "
            "|Σ 스텝길이·sin(h − k)| 2.8~4.5 m · max|h − k| ≥ 3° (k = 최근접 중심선 접선각)",
    "혼잡": "주변 차량 후보 ≥ 40대(32대에서 잘림) · 과거 4.9 s에 2 m/s 넘게 움직인 주변 차량 ≥ 8대",
    "정지 차량 뒤집힘": "(추가 칸) 주변 차량 중 4.9 s 내내 AV2 속력 < 0.5 m/s · 관측 50스텝 전부 · t=0 에 교차로 밖 차로 중심선 1 m 안 · "
                   "AV2 heading 은 차로 방향 쪽인데 10 Hz 는 위치 잡음 스텝(≥3)으로 뒤집힘 교정이 걸려 반대 · AV2 보조 ≥ 25스텝 · "
                   "2 Hz h 는 차로 방향 쪽 · 거리 < 25 m",
    "정지 차량 많음": "(발견용 칸) 4.9 s 내내 AV2 속력 < 0.5 m/s 인 주변 차량 ≥ 15대 · 주변 차량 관측 스텝 중 AV2 보조 ≥ 80% · "
                 "N = 정지 차량 중 관측 50스텝 전부·위치차분(잡음) ≥ 3 인 가장 가까운 차",
    "focal ±π 점프": "(발견용 칸) focal 10 Hz h 에 ±π 점프 ≥ 1회 · 2 Hz h 에는 없음",
    "큰 슬립각": "(발견용 칸) 위치차분·속력 ≥ 3 m/s 스텝 중 |AV2 − h| ≥ 10° 가 10스텝 이상 · 90° 넘는 스텝 없음",
    "램프 뚜렷": "(발견용 칸) 기준 속력(1.0~1.6 s 위치차분 평균) ≥ 10 m/s · 첫 스텝 10 Hz 속력 ≤ 기준의 0.35배",
}
NB_PREF = {"정속 직진": "jump", "좌회전": "jump", "우회전": "mixed", "저속·정지": "mixed", "차선변경": "mixed", "혼잡": "jump",
           "정지 차량 뒤집힘": "flip", "정지 차량 많음": "parked", "focal ±π 점프": "jump", "큰 슬립각": "mixed",
           "램프 뚜렷": "jump"}


SLUG = {"정속 직진": "straight", "좌회전": "left_turn", "우회전": "right_turn", "저속·정지": "slow_stop",
        "차선변경": "lane_change", "혼잡": "crowded", "정지 차량 뒤집힘": "parked_flip", "정지 차량 많음": "many_stopped",
        "focal ±π 점프": "focal_pi_jump", "큰 슬립각": "large_slip", "램프 뚜렷": "strong_ramp"}


def flip_nb_ok(T):
    return ((T["vmax"] < 0.5) & (T["full_obs"] == 1) & (T["lane_m"] == 1) & (T["ag_av2"] == 1)
            & (T["ag_10"] == 0) & (T["ag_2"] == 1) & (T["flip"] == 1) & (T["n_src2"] >= 25) & (T["dist0"] < 25))


def nb_ok(T):
    """패널에 그릴 주변 차량 후보(움직이는 차): 관측 50스텝 전부(2 Hz 계산 가능) · 위치차분 스텝 ≥ 15 ·
    AV2 속력 최대 ≥ 3 m/s · 과거 이동 ≥ 10 m · 정지 잡음(AV2 속력 < 0.5 인데 위치차분) 스텝 ≤ 2."""
    return ((T["full_obs"] == 1) & (T["n_src1"] >= 15) & (T["vmax"] >= 3.0)
            & (T["path"] >= 10.0) & (T["n_src1_slow"] <= 2))


def nb_candidates(S, i):
    return np.flatnonzero((S.T["scen"] == i) & nb_ok(S.T))


def choose_nb(S, i, pref):
    T = S.T
    if pref == "flip":
        rows = np.flatnonzero((T["scen"] == i) & flip_nb_ok(T))
        return int(T["slot"][rows[int(np.argmin(T["dist0"][rows]))]]) if len(rows) else None
    if pref == "parked":
        base = (T["scen"] == i) & (T["vmax"] < 0.5) & (T["full_obs"] == 1)
        for m in (base & (T["n_src1"] >= 3), base):
            rows = np.flatnonzero(m)
            if len(rows):
                return int(T["slot"][rows[int(np.argmin(T["dist0"][rows]))]])
        return None
    rows = nb_candidates(S, i)
    if not len(rows):
        return None
    j10 = (T["jump10"][rows] > 0).astype(float)
    mixed = ((T["n_src1"][rows] >= 5) & (T["n_src2"][rows] >= 5)).astype(float)
    base = T["n_src1"][rows] / 50.0 - T["dist0"][rows] / 100.0
    score = base + (4 * j10 + 2 * mixed if pref == "jump" else 4 * mixed + 2 * j10)
    return int(T["slot"][rows[int(np.argmax(score))]])


def run_picks(S, data, seed=C.SEED):
    f = S.focal_feats()
    T = S.T
    n = S.n
    mvnb = np.bincount(T["scen"][T["vmax"] > 2.0].astype(int), minlength=n)
    has_nb = np.bincount(T["scen"][nb_ok(T)].astype(int), minlength=n) > 0
    nc = S.ag["n_cand"].astype(int)
    n_stat = np.bincount(T["scen"][T["vmax"] < 0.5].astype(int), minlength=n)
    has_parked = np.bincount(T["scen"][(T["vmax"] < 0.5) & (T["full_obs"] == 1)].astype(int), minlength=n) > 0
    hs = S.ag["h_src"]
    nb_av2_share = (hs == 2).sum((1, 2)) / np.maximum((hs > 0).sum((1, 2)), 1)
    j10 = np.array([len(jumps(h)) for h in S.h10])
    j2 = np.array([len(jumps(h)) for h in S.h2])
    slip = np.abs(wrap(S.head_n[:, :OBS - 1].astype(np.float64) - S.h10[:, :OBS - 1]))
    big_slip = ((S.src10[:, :OBS - 1] == 1) & (S.v10[:, :OBS - 1] >= 3) & (slip >= np.radians(10))).sum(1)
    ref = S.v10[:, RAMP_REF].mean(1)
    rules = {
        "정속 직진": (f["mv_frac"] == 1) & (f["vmean"] >= 8) & (f["vrange"] < 1.5) & (np.abs(f["dh"]) < 3) & (f["hmax"] < 5),
        "좌회전": (f["dh"] > 45) & (f["mv_frac"] >= 0.8) & (f["vmean"] >= 3) & (f["rev_steps"] == 0),
        "우회전": (f["dh"] < -45) & (f["mv_frac"] >= 0.8) & (f["vmean"] >= 3) & (f["rev_steps"] == 0),
        "저속·정지": (f["av2_frac"] >= 0.5) & (f["av2_frac"] <= 0.9) & (f["path"] >= 3),
        "차선변경": (f["mv_frac"] == 1) & (f["vmean"] >= 6) & (np.abs(f["dh"]) < 10) & (f["lane_valid"] == 1)
        & (f["lane_inter"] == 0)
        & (np.abs(f["lat_lane"]) >= 2.8) & (np.abs(f["lat_lane"]) <= 4.5) & (f["th_lane_max"] >= 3),
        "혼잡": (nc >= 40) & (mvnb >= 8),
        "정지 차량 뒤집힘": np.bincount(T["scen"][flip_nb_ok(T)].astype(int), minlength=n) > 0,
        "정지 차량 많음": (n_stat >= 15) & (nb_av2_share >= 0.8) & has_parked,
        "focal ±π 점프": (j10 >= 1) & (j2 == 0),
        "큰 슬립각": (big_slip >= 10) & (f["rev_steps"] == 0),
        "램프 뚜렷": (ref >= 10) & (S.v10[:, 0] <= 0.35 * ref),
    }
    rng = np.random.default_rng(seed)
    used, picks = set(), []
    for k, cls in enumerate(PICK_ORDER):
        ok = rules[cls] if NB_PREF[cls] in ("flip", "parked") else rules[cls] & has_nb
        cand = [int(i) for i in np.flatnonzero(ok) if int(i) not in used]
        if not cand:
            print(f"[picks] {cls}: 후보 없음 — 건너뛴다", flush=True)
            continue
        i = int(rng.choice(cand))
        used.add(i)
        slot = choose_nb(S, i, NB_PREF[cls])
        picks.append({"n": k + 1, "cls": cls, "idx": i, "sid": S.sids[i], "rule": PICK_RULES[cls],
                      "n_candidates": len(cand), "n_rule_only": int(rules[cls].sum()), "nb_slot": slot,
                      "nb_pref": NB_PREF[cls],
                      "focal": {"dh_deg": round(float(f["dh"][i]), 1), "vmean": round(float(f["vmean"][i]), 2),
                                "mv_frac": round(float(f["mv_frac"][i]), 3), "av2_frac": round(float(f["av2_frac"][i]), 3),
                                "path_m": round(float(f["path"][i]), 1), "y_start": round(float(f["y_start"][i]), 2),
                                "lat_lane_m": round(float(f["lat_lane"][i]), 2),
                                "th_lane_max_deg": round(float(f["th_lane_max"][i]), 1)},
                      "n_cand": int(nc[i]), "n_moving_nb": int(mvnb[i])})
    out = {"seed": seed, "population": f"val 앞 {n:,}", "picks": picks,
           "nb_rule": "관측 50스텝 전부 · 위치차분 스텝 ≥ 15 인 차량 중 점수 최대. 점수 = 위치차분 비율 − 거리/100 + "
                      "(jump 선호: 4·[10 Hz ±π 점프 있음] + 2·[위치차분·AV2 각 ≥5스텝]) / (mixed 선호: 가중치 반대)"}
    (data / "picks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for p in picks:
        print(f"[picks] {p['n']} {p['cls']:6} idx {p['idx']:5d} {p['sid']}  후보 {p['n_candidates']:,}  nb {p['nb_slot']}", flush=True)
    return out


# ============================================================================ 3. panels
def runs_of(mask):
    """참인 연속 구간 [(시작, 끝)] (끝 포함)."""
    m = np.asarray(mask, bool)
    out, s = [], None
    for t, v in enumerate(m):
        if v and s is None:
            s = t
        if not v and s is not None:
            out.append((s, t - 1))
            s = None
    if s is not None:
        out.append((s, len(m) - 1))
    return out


def shade_src(ax, src, t_axis, which=(0, 1, 2, 3), alpha=1.0, dt=HD.DT):
    """스텝 출처를 배경 띠로 (스텝 t 의 칸 = [t − dt/2, t + dt/2])."""
    for code in which:
        for a, b in runs_of(np.asarray(src) == code):
            ax.axvspan(t_axis[a] - dt / 2, t_axis[b] + dt / 2, color=SRC_BAND[code], lw=0, zorder=0, alpha=alpha)


def broken(t, y, obs=None):
    """±π 점프와 미관측에서 선을 끊는다 (NaN 삽입)."""
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64).copy()
    if obs is not None:
        y[~np.asarray(obs, bool)] = np.nan
    tt, yy = [t[0]], [y[0]]
    for k in range(1, len(t)):
        if np.isfinite(y[k]) and np.isfinite(y[k - 1]) and abs(y[k] - y[k - 1]) > np.pi:
            tt.append((t[k] + t[k - 1]) / 2)
            yy.append(np.nan)
        tt.append(t[k])
        yy.append(y[k])
    return np.array(tt), np.array(yy)


def step_xy(t, y, end, angle=True):
    """steps-post 점열: 값 y[k] 가 [t[k], t[k+1]) 에 머문다. angle=True 면 ±π 점프에서 선을 끊는다."""
    xs, ys = [], []
    for k in range(len(t)):
        t1 = t[k + 1] if k + 1 < len(t) else end
        if k and angle and (not np.isfinite(y[k]) or not np.isfinite(y[k - 1]) or abs(y[k] - y[k - 1]) > np.pi):
            xs.append(np.nan)
            ys.append(np.nan)
        elif k:
            xs.append(t[k])
            ys.append(y[k - 1])
        xs += [t[k], t1]
        ys += [y[k], y[k]]
    return np.array(xs), np.array(ys)


def top_pred_axis(ax, label="예측 시작 기준 [s]  (창 시작 0 s = −4.9 s)"):
    """창 시작 기준 가로축 위에 예측 시작 기준(−4.9 … 0 s) 눈금을 함께 단다 (필수 요건 1)."""
    sec = ax.secondary_xaxis("top", functions=(lambda x: x - 4.9, lambda x: x + 4.9))
    sec.set_xlabel(label, fontsize=8.5, color=C.INK2)
    sec.tick_params(labelsize=7.8, colors=C.MUTED)
    return sec


def ymin_span(ax, span):
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        c = (lo + hi) / 2
        ax.set_ylim(c - span / 2, c + span / 2)


def draw_arrow(ax, x, y, ang, L, color, lw=2.2, z=20):
    import viz_v4_gallery as G
    ax.annotate("", xy=(x + L * np.cos(ang), y + L * np.sin(ang)), xytext=(x, y), zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, path_effects=G._pe(lw, 1.8)))


def panel_facts(S, p):
    i, s = p["idx"], p["nb_slot"]
    src10 = S.src10[i]
    mv = (src10[:OBS - 1] == 1)
    slip = deg(wrap(S.head_n[i][:OBS - 1] - S.h10[i][:OBS - 1]))[mv]
    T = S.T
    r = np.flatnonzero((T["scen"] == i) & (T["slot"] == s))[0]
    kept = S.ag["atype"][i] > 0
    nb_src = S.ag["h_src"][i][kept]
    never = int(((nb_src == 1).sum(1) == 0).sum())
    return {"slip_med_abs": float(np.median(np.abs(slip))) if len(slip) else float("nan"),
            "slip_max_abs": float(np.abs(slip).max()) if len(slip) else float("nan"),
            "ramp_ratio": float(S.v10[i, 0] / S.vf[i, 0]) if S.vf[i, 0] >= 1.0 else None,
            "v10_0": float(S.v10[i, 0]), "vf_0": float(S.vf[i, 0]), "v2_0": float(S.v2[i, 0]),
            "jump10": int(len(jumps(S.h10[i]))), "jump2": int(len(jumps(S.h2[i]))),
            "n_kept": int(kept.sum()), "n_cand": int(S.ag["n_cand"][i]), "never_moved": never,
            "nb_row": {c: float(T[c][r]) for c in AG_COLS},
            "src10_pct": {SRC_NAMES[c]: round(float((src10 == c).mean() * 100), 1) for c in (1, 2, 3)},
            "src2_n_av2": int((S.src2[i] == 2).sum())}


def smooth_field(v):
    """AV2 속도 필드 속력 → (평활 속력, 가속도). viz_v4_dump 와 같은 규칙(median 5 → Savitzky–Golay 15스텝·2차)."""
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    vm = median_filter(np.asarray(v, np.float64), size=C.MED_K, mode="nearest")
    return (savgol_filter(vm, C.SG_WIN, C.SG_POLY, mode="interp"),
            savgol_filter(vm, C.SG_WIN, C.SG_POLY, deriv=1, delta=C.DT, mode="interp"))


def lane_theta(sdir, pos_city, h_city, fld=None):
    """θ = h − k (k = 5 m 안 같은 방향 최근접 중심선 접선, heading_decomp.decompose 'point' 기준)."""
    if fld is None:
        fld = HD.lane_field(HD.graph_of(sdir), near_xy=np.asarray(pos_city, np.float64)[-1])
    if fld is None:
        return None, None
    k, th, valid, lat, inter = HD.decompose(np.asarray(pos_city, np.float64), np.asarray(h_city, np.float64), fld)
    return {"k": k, "th": th, "valid": valid, "inter": inter > 0}, fld


def draw_panel(S, p, path):
    plt = C.setup_mpl()
    import viz_v4_gallery as G
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from av2.datasets.motion_forecasting import scenario_serialization as ss
    from dataset_map import _rotation_matrix
    i, sid, s_sel = p["idx"], p["sid"], p["nb_slot"]
    sdir = C.VAL_DIR / sid
    scn = ss.load_argoverse_scenario_parquet(sdir / f"scenario_{sid}.parquet")
    pos, vel, head = focal_raw(scn)
    origin, theta = pos[OBS - 1].copy(), head[OBS - 1].copy()
    th0 = float(theta)
    R = _rotation_matrix(-theta)
    R64 = R.astype(np.float64)
    past = ((pos - origin) @ R.T)[:OBS].astype(np.float64)
    pos64 = pos[:OBS].astype(np.float64)
    det = ah2hz_detail(pos[:OBS], head[:OBS], th0)
    p2n = (det["p"] - origin.astype(np.float64)) @ R64.T
    _, ag = PA.agents_one(sdir, with_pos=True, loader=lambda _p: scn)
    tracks, _ = agent_tracks(scn, origin)
    assert np.array_equal(ag["h"], S.ag["h"][i]) and np.array_equal(ag["h_src"], S.ag["h_src"][i])
    h10, h2, v10, v2 = S.h10[i], S.h2[i], S.v10[i], S.v2[i]
    src10, src2 = S.src10[i], S.src2[i]
    assert np.array_equal(src2, det["src"])
    head_n, vf = S.head_n[i].astype(np.float64), S.vf[i].astype(np.float64)
    fx = panel_facts(S, p)
    kept = np.flatnonzero(ag["atype"] > 0)

    # ---------------- 배치 (인치 단위)
    W, Hh = 19.5, 16.4
    top_in, bot_in = 1.7, 1.95
    fig = plt.figure(figsize=(W, Hh))
    gs = fig.add_gridspec(6, 2, width_ratios=[1.0, 1.22], height_ratios=[1.1, 0.55, 0.85, 0.95, 0.8, 1.2],
                          left=0.02, right=0.975, top=1 - top_in / Hh, bottom=bot_in / Hh, wspace=0.12, hspace=0.55)
    axm = fig.add_subplot(gs[:, 0])
    axh = fig.add_subplot(gs[0, 1])
    axd = fig.add_subplot(gs[1, 1], sharex=axh)
    axt = fig.add_subplot(gs[2, 1], sharex=axh)
    axv = fig.add_subplot(gs[3, 1], sharex=axh)
    axa = fig.add_subplot(gs[4, 1], sharex=axh)
    axn = fig.add_subplot(gs[5, 1], sharex=axh)
    ramp_span = (T10[0] - 0.05, T10[5] - 0.05)

    def ramp_hatch(ax):
        ax.axvspan(*ramp_span, facecolor="none", edgecolor=C.MUTED, hatch="///", lw=0, zorder=1)

    def two_hz_start(ax):
        ax.axvline(T2[0], color=COL2, lw=1.0, ls=":", zorder=2)

    # ---------------- 지도
    sel_pos = ag["pos"][s_sel][ag["h_src"][s_sel] > 0].astype(np.float64)
    xlim, ylim = G.view([past, sel_pos, np.zeros((1, 2))], min_span=55.0, margin=6.0)
    bb = axm.get_position()
    aspect = (bb.height * Hh) / (bb.width * W)
    cy, half = (ylim[0] + ylim[1]) / 2, (xlim[1] - xlim[0]) / 2 * aspect
    if half > (ylim[1] - ylim[0]) / 2:
        ylim = (cy - half, cy + half)
    span = xlim[1] - xlim[0]
    L = max(3.2, 0.05 * span)
    scene = C.build_scene(sid, origin, th0)
    G.draw_map_dark(axm, scene, xlim, ylim)
    SEC_IDX = np.array([9, 19, 29, 39])          # −4, −3, −2, −1 s
    for s in kept:
        o = ag["h_src"][s] > 0
        q = ag["pos"][s].astype(np.float64)
        sel = s == s_sel
        col = M_NBSEL if sel else "#aab0ba"
        for a_, b_ in runs_of(o):
            axm.plot(q[a_:b_ + 1, 0], q[a_:b_ + 1, 1], color=col, lw=2.2 if sel else 1.0,
                     alpha=1.0 if sel else 0.55, zorder=8 if sel else 6,
                     path_effects=G._pe(2.2, 1.4) if sel else None)
        si = SEC_IDX[o[SEC_IDX]]
        axm.scatter(q[si, 0], q[si, 1], s=18 if sel else 7, color=col, lw=0, zorder=8 if sel else 6,
                    alpha=1.0 if sel else 0.8)
        tr = tracks[s]
        yaw = float(wrap(tr["H"][OBS - 1] - th0))
        bl, bw = BOX[int(ag["atype"][s])]
        x0, y0 = ag["pos0"][s]
        C.draw_box(axm, x0, y0, yaw, M_NBSEL if sel else M_NB, length=bl, width=bw, zorder=9,
                   ec="white" if sel else "#1b1d22", lw=1.3 if sel else 0.6)
        draw_arrow(axm, x0, y0, float(ag["h"][s][OBS - 1]), L * (1.15 if sel else 0.9),
                   M_SRC[int(ag["h_src"][s][OBS - 1])], lw=2.4 if sel else 1.8, z=21 if sel else 19)
        if sel:
            axm.annotate("N", (x0, y0), xytext=(-12, 10), textcoords="offset points", color=M_NBSEL, fontsize=15,
                         fontweight="bold", zorder=30, path_effects=G._txt_pe(3.4))
    C.draw_box(axm, 0, 0, 0, M_FOCAL, zorder=12, ec="white", lw=1.2)
    axm.plot(past[:, 0], past[:, 1], color=M_PAST, lw=2.4, zorder=13, path_effects=G._pe(2.4))
    axm.scatter(past[:, 0], past[:, 1], s=7, color=M_PAST, lw=0, zorder=14)
    for k in range(len(IDX2)):
        filled = src2[k] == 1
        axm.scatter(*p2n[k], s=70, marker="o", facecolor=M_2HZ if filled else "none", edgecolor=M_2HZ,
                    linewidths=2.0, zorder=16)
    draw_arrow(axm, 0.0, 0.0, float(h10[OBS - 1]), L * 1.3, M_SRC[int(src10[OBS - 1])], lw=3.0, z=22)
    axm.annotate("focal", (0, 0), xytext=(8, -18), textcoords="offset points", color=M_FOCAL, fontsize=13,
                 fontweight="bold", zorder=30, path_effects=G._txt_pe(3.4))
    G.finish_axes(axm, xlim, ylim)
    axm.set_title("지도: t = 0 위치 · 방향 화살표 = 10 Hz h[49] (색 = 출처) · 과거 4.9 s", fontsize=12, color=C.INK, pad=6)
    hd = [Line2D([], [], color=M_PAST, lw=2.4, marker="o", ms=3, label="focal 과거 4.9 s (점 = 0.1 s 마다)"),
          Line2D([], [], color="none", marker="o", ms=9, mfc=M_2HZ, mec=M_2HZ, label="focal 2 Hz 표본 (평활 위치, 0.5 s 마다)"),
          Line2D([], [], color="none", marker="o", ms=9, mfc="none", mec=M_2HZ, mew=2, label="  └ 그 구간 h 가 AV2 보조"),
          Line2D([], [], color="#aab0ba", lw=1.2, marker="o", ms=3, label="주변 차량 과거 (점 = 1 s 마다)"),
          Line2D([], [], color=M_NBSEL, lw=2.4, label="N = 아래 시계열의 주변 차량"),
          Patch(facecolor=M_SRC[1], label="화살표: 위치차분 (구간 ≥1 m/s)"),
          Patch(facecolor=M_SRC[2], label="화살표: AV2 보조 (<1 m/s)")]
    if (ag["h_src"][kept][:, OBS - 1] == 3).any():
        hd.append(Patch(facecolor=M_SRC[3], label="화살표: 이웃 값 유지"))
    leg = axm.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=2, fontsize=9.3,
                     frameon=True, facecolor="#1b1d22", edgecolor="#3a3d44", framealpha=1.0, handlelength=2.2,
                     columnspacing=1.2, borderpad=0.5)
    for t in leg.get_texts():
        t.set_color("#eceef2")

    # ---------------- ① focal 진행방향 h
    ax = axh
    shade_src(ax, src10, T10, which=(2, 3))
    ax.plot(T10, deg(head_n), color=COLAV2, lw=1.3, ls=(0, (4, 2)), label="AV2 heading 원본 (차체 방향, 10 Hz)", zorder=3)
    tb, yb = broken(T10, h10)
    ax.plot(tb, deg(yb), color=COL10, lw=1.6, marker="o", ms=2.6, label="10 Hz h (위치차분, 0.1 s 구간)", zorder=4)
    xs, ys = step_xy(T2, h2, 0.0)
    ax.plot(xs, deg(ys), color=COL2, lw=2.0, zorder=5, label="2 Hz h (평활 후 0.5 s 구간)")
    for k in range(len(IDX2)):
        ax.scatter(T2[k], deg(h2[k]), s=40, facecolor=COL2 if src2[k] == 1 else "white", edgecolor=COL2,
                   linewidths=1.6, zorder=6)
    for j in jumps(h10):
        ax.axvline(T10[j] + 0.05, color=COL10, lw=0.9, ls=":", zorder=2)
    ax.set_ylabel("h [°]")
    ax.set_title("① focal 진행방향 h (0° = t=0 AV2 heading)", fontsize=11, loc="left")
    hh = ax.get_legend_handles_labels()
    extra = [Patch(facecolor=SRC_BAND[2], label="10 Hz h 가 AV2 보조인 스텝")] if (src10 == 2).any() else []
    extra += [Line2D([], [], color="none", marker="o", ms=6, mfc="white", mec=COL2, mew=1.6, label="2 Hz: AV2 보조 표본")] \
        if (src2 == 2).any() else []
    ax.legend(handles=hh[0] + extra, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.2,
              borderaxespad=0.1)
    ymin_span(ax, 8.0)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---------------- ② AV2 − h (슬립각 + 잡음)
    ax = axd
    shade_src(ax, src10, T10, which=(2, 3))
    d10 = deg(wrap(head_n - h10))
    d2 = deg(wrap(head_n[IDX2] - h2))
    ax.axhline(0, color=C.AXIS, lw=0.9)
    ax.plot(T10[:OBS - 1], d10[:OBS - 1], color=COL10, lw=1.3, marker="o", ms=2.2, label="AV2 − 10 Hz h")
    ax.plot(T2, d2, color=COL2, lw=0, marker="o", ms=6.5, label="AV2 − 2 Hz h")
    lim = float(np.nanmax(np.abs(np.concatenate([d10[:OBS - 1], d2]))))
    lim = min(max(4.0, lim * 1.2), 185.0)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("차 [°]")
    ax.set_title(f"② AV2 heading − h : 위치차분 구간 |차| 중앙 {fx['slip_med_abs']:.1f}° · 최대 {fx['slip_max_abs']:.1f}° "
                 "(슬립각 + 위치 잡음, AV2 보조 구간은 정의상 0)", fontsize=10.5, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.2, borderaxespad=0.1)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---------------- ③ 차로 대비 잔차각 θ = h − k  (+ 누적 횡이동)
    ax = axt
    lt10, fld = lane_theta(sdir, pos64, wrap(h10 + th0))
    lt2, _ = lane_theta(sdir, det["p"], wrap(h2 + th0), fld) if fld is not None else (None, None)
    lc_span = None
    cum = None
    if lt10 is not None and lt10["valid"].any():
        t10 = np.where(lt10["valid"], deg(lt10["th"]), np.nan)
        for a_, b_ in runs_of(lt10["inter"]):
            ax.axvspan(T10[a_] - 0.05, T10[b_] + 0.05, facecolor="#eeeeee", edgecolor=C.AXIS, hatch="..", lw=0, zorder=0)
        ax.axhline(0, color=C.AXIS, lw=0.9)
        ax.plot(T10[:OBS - 1], t10[:OBS - 1], color=COL10, lw=1.3, marker="o", ms=2.2, label="10 Hz θ", zorder=4)
        t2v = np.where(lt2["valid"], deg(lt2["th"]), np.nan)
        xs, ys = step_xy(T2, np.radians(t2v), 0.0)
        ax.plot(xs, deg(ys), color=COL2, lw=1.8, zorder=5, label="2 Hz θ")
        ax.scatter(T2, t2v, s=40, color=COL2, zorder=6)
        fin = t10[:OBS - 1][np.isfinite(t10[:OBS - 1])]
        tl = max(4.0, min(185.0, float(np.nanmax(np.abs(np.r_[fin, t2v[np.isfinite(t2v)]]))) * 1.15 if len(fin) else 4.0))
        ax.set_ylim(-tl, tl)
        ax2 = ax.twinx()
        sl = np.linalg.norm(np.diff(pos64, axis=0), axis=1)
        inc = np.where(lt10["valid"][:-1], sl * np.sin(lt10["th"][:-1]), 0.0)
        cum = np.r_[0.0, np.cumsum(inc)]
        sl2 = np.linalg.norm(np.diff(det["p"], axis=0), axis=1)
        inc2 = np.where(lt2["valid"][:-1], sl2 * np.sin(lt2["th"][:-1]), 0.0)
        cum2 = np.r_[0.0, np.cumsum(inc2)]
        ax2.plot(T10, cum, color=C.C_VIOLET, lw=1.4, ls=(0, (3, 1.5)), label="누적 횡이동 10 Hz")
        ax2.plot(T2, cum2 + cum[4], color=C.C_VIOLET, lw=0, marker="D", ms=5, label="누적 횡이동 2 Hz (−4.5 s 에 맞춤)")
        cl = max(1.0, float(np.abs(np.r_[cum, cum2 + cum[4]]).max()) * 1.2)
        ax2.set_ylim(-cl, cl)
        ax2.set_ylabel("차로 대비 횡이동 [m]", color=C.C_VIOLET)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        tot = cum[-1]
        if abs(tot) >= 2.5 and not lt10["inter"].any():
            a_ = int(np.argmax(np.abs(cum) >= 0.1 * abs(tot)))
            b_ = int(np.argmax(np.abs(cum) >= 0.9 * abs(tot)))
            lc_span = (T10[a_], T10[b_])
            ax.axvspan(*lc_span, facecolor="#e4e0f5", edgecolor="none", zorder=0, alpha=0.8)
        h1, l1 = ax.get_legend_handles_labels()
        h2_, l2_ = ax2.get_legend_handles_labels()
        extra = [Patch(facecolor="#eeeeee", hatch="..", edgecolor=C.AXIS, label="기준 차로가 교차로 차로")] \
            if lt10["inter"].any() else []
        if lc_span:
            extra.append(Patch(facecolor="#e4e0f5", label="차선변경 구간(추정: 누적 횡이동 10→90%)"))
        ax.legend(h1 + h2_ + extra, l1 + l2_ + [e.get_label() for e in extra], loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.0, borderaxespad=0.1)
        ttl = f"③ 차로 대비 잔차각 θ = h − k  · 4.9 s 누적 횡이동 {tot:+.1f} m"
        if lt10["inter"].any():
            ax.text(0.0, -0.04, "점무늬 = 기준이 교차로 차로 — k 가 회전 차로를 따라가 θ·누적 횡이동은 참고만",
                    transform=ax.transAxes, fontsize=8.2, color=C.INK2, va="top")
    else:
        ax.text(0.5, 0.5, "모든 스텝에서 5 m 안에 같은 방향 차로가 없다 (주차장·도로 밖) — θ 정의 안 됨",
                transform=ax.transAxes, ha="center", va="center", color=C.MUTED, fontsize=10)
        ax.set_yticks([])
        ttl = "③ 차로 대비 잔차각 θ = h − k"
    ax.set_title(ttl, fontsize=11, loc="left")
    ax.set_ylabel("θ [°]")
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---------------- ④ 속력 v
    ax = axv
    shade_src(ax, src10, T10, which=(2, 3))
    ramp_hatch(ax)
    ax.plot(T10, vf, color=COLAV2, lw=1.3, ls=(0, (4, 2)), label="AV2 속도 필드 |v| (10 Hz)", zorder=3)
    ax.plot(T10, v10, color=COL10, lw=1.6, marker="o", ms=2.6, label="10 Hz: a 누적합 (위치차분 속력)", zorder=4)
    xs, ys = step_xy(T2, v2, 0.0, angle=False)
    ax.plot(xs, ys, color=COL2, lw=2.0, zorder=5, label="2 Hz: a 누적합 (평활 위치, 0.5 s)")
    ax.scatter(T2, v2, s=40, color=COL2, zorder=6)
    two_hz_start(ax)
    top = max(float(np.nanmax(np.r_[vf, v10, v2])), 1.0)
    ax.set_ylim(0, top * 1.3)
    rtxt = (f" = {fx['ramp_ratio']:.2f}배" if fx["vf_0"] >= 1.0 else " (정지 출발 — 비율 없음)")
    ax.text(T10[0], top * 1.25, f" 창 시작 0.5 s 램프(빗금): 첫 스텝 10 Hz {fx['v10_0']:.2f} / AV2 {fx['vf_0']:.2f} m/s"
            + rtxt + "  ·  점선 = 2 Hz 첫 표본(인덱스 4, −4.5 s)", fontsize=8.6, va="top", color=C.INK2,
            bbox=dict(facecolor=C.SURF, edgecolor="none", pad=1.0, alpha=0.85))
    ax.set_ylabel("v [m/s]")
    ax.set_title("④ focal 속력 (a 누적합)", fontsize=11, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.2, borderaxespad=0.1)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---------------- ⑤ 가속도 (초당)
    ax = axa
    shade_src(ax, src10, T10, which=(2, 3))
    ramp_hatch(ax)
    a10 = S.x10[i, :, 0] * KMH / HD.DT
    a2 = S.x2[i, :, 0] * KMH / (HD.DT * HD.DS_STEP)
    _, af = smooth_field(vf)
    ax.axhline(0, color=C.AXIS, lw=0.9)
    ax.plot(T10, af, color=COLAV2, lw=1.3, ls=(0, (4, 2)), label="AV2 속도 필드 평활 미분 (10 Hz)", zorder=3)
    ax.plot(T10[1:OBS - 1], a10[1:OBS - 1], color=COL10, lw=1.3, marker="o", ms=2.6,
            label="10 Hz: a ÷ 0.1 s", zorder=4)
    ax.plot(T2[1:-1], a2[1:-1], color=COL2, lw=2.0, marker="o", ms=6.5, label="2 Hz: a ÷ 0.5 s", zorder=5)
    two_hz_start(ax)
    ref = np.r_[np.abs(a2[1:-1]), np.abs(af), np.percentile(np.abs(a10[6:OBS - 1]), 95)]
    al = float(np.clip(ref.max() * 1.35, 2.0, 12.0))
    ax.set_ylim(-al, al)
    big = np.abs(a10[1:OBS - 1]).max()
    if big > al:
        tb_ = int(np.argmax(np.abs(a10[1:OBS - 1]))) + 1
        ax.text(0.995, 0.04, f"10 Hz 최대 |a| {big:.1f} m/s² ({T10[tb_]:+.1f} s) — 축 밖", transform=ax.transAxes,
                ha="right", fontsize=8.4, color=COL10)
    ax.set_ylabel("a [m/s²]")
    ax.set_title("⑤ focal 가속도 — 두 판 모두 초당 값 (10 Hz 는 첫 값 v0·마지막 0 을 뺌)", fontsize=11, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.2, borderaxespad=0.1)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---------------- ⑥ 주변 차량 N 의 h
    ax = axn
    tr = tracks[s_sel]
    hs = ag["h_src"][s_sel]
    hn = ag["h"][s_sel].astype(np.float64)
    obs = hs > 0
    shade_src(ax, hs, T10)
    P, Hs = tr["P"], tr["H"]
    g2 = HD.ah_features_2hz(P, Hs, th0)
    dn = ah2hz_detail(P, Hs, th0)
    assert np.array_equal(dn["h32"], g2[:, 1])
    av2n = wrap(Hs - th0)
    tb, yb = broken(T10, av2n, obs)
    ax.plot(tb, deg(yb), color=COLAV2, lw=1.2, ls=(0, (4, 2)), label="AV2 heading 원본 (10 Hz)", zorder=3)
    tb, yb = broken(T10, hn, obs)
    ax.plot(tb, deg(yb), color=COL10, lw=1.6, marker="o", ms=2.6, label="10 Hz h (캐시 agents_hist)", zorder=4)
    xs, ys = step_xy(T2, g2[:, 1].astype(np.float64), 0.0)
    ax.plot(xs, deg(ys), color=COL2, lw=2.0, zorder=5, label="2 Hz h (즉석 계산: ah_features_2hz)")
    for k in range(len(IDX2)):
        ax.scatter(T2[k], deg(g2[k, 1]), s=40, facecolor=COL2 if dn["src"][k] == 1 else "white", edgecolor=COL2,
                   linewidths=1.6, zorder=6)
    j10, j2 = jumps(hn, obs), jumps(g2[:, 1])
    hg_n = HD.guard(Hs)
    m1 = obs & (hs == 1)
    med10 = float(np.degrees(np.median(np.abs(wrap(hg_n[m1] - (hn[m1] + th0)))))) if m1.sum() >= 3 else None
    m2_ = dn["src"] == 1
    med2 = float(np.degrees(np.median(np.abs(wrap(hg_n[IDX2][m2_] - dn["h_city"][m2_]))))) if m2_.sum() >= 3 else None
    nb_flip_info = {"posdiff_dirs_city_deg_10hz": [round(float(v), 2) for v in np.degrees(wrap(hn[m1] + th0))],
                    "av2_guard_city_deg_at_posdiff": [round(float(v), 2) for v in np.degrees(wrap(hg_n[m1]))],
                    "median_abs_av2_minus_posdiff_deg_10hz": med10, "median_abs_av2_minus_posdiff_deg_2hz": med2,
                    "rule": "중앙값 > 90° 이면 뒤집음 (위치차분 ≥ 3 일 때만)",
                    "vmax_exact": float(np.nanmax(tr["V"][obs]))}
    for j in j10:
        ax.axvline(T10[j] + 0.05, color=COL10, lw=1.0, ls=":", zorder=2)
    for j in j2:
        ax.axvline(T2[j] + 0.25, color=COL2, lw=1.0, ls=":", zorder=2)
    anyjump = len(j10) or len(j2) or np.nanmax(np.abs(deg(np.r_[hn[obs], g2[:, 1]]))) > 150
    if anyjump:
        ax.set_ylim(-200, 200)
        ax.set_yticks([-180, -90, 0, 90, 180])
        for yv in (-180, 180):
            ax.axhline(yv, color=C.INK2, lw=0.9, ls="--", zorder=1)
    else:
        ymin_span(ax, 8.0)
    ax.set_ylabel("h [°]")
    row = fx["nb_row"]
    fl = {1: "뒤집음", 0: "그대로", -1: "안 함(위치차분<3)"}
    ttl = (f"⑥ 주변 차량 N — {TYPE_KO[int(row['atype'])]}{' (AV)' if row['is_av'] else ''} · 거리 {row['dist0']:.1f} m · "
           f"AV2 속력 t=0 {row['v49']:.1f} / 4.9 s 최대 {row['vmax']:.2f} m/s · 출처 위치차분 {int(row['n_src1'])} / AV2 보조 "
           f"{int(row['n_src2'])} 스텝 · ±π 점프 10 Hz {len(j10)} / 2 Hz {len(j2)}회")
    mt = lambda v: f"(중앙차 {v:.1f}°)" if v is not None else ""
    ttl += f"\n   뒤집힘 판정 10 Hz {fl[int(row['flip'])]}{mt(med10)} · 2 Hz {fl[int(row['flip2'])]}{mt(med2)}"
    if row["lane_m"] == 1:
        yn = lambda v: "같은 쪽" if v == 1 else ("반대" if v == 0 else "—")
        ttl += (f"  |  t=0 차로 방향(중심선 {LANE_IN_M:g} m 안) 대비 AV2 {yn(row['ag_av2'])} · 10 Hz {yn(row['ag_10'])} · "
                f"2 Hz {yn(row['ag_2'])}")
    ax.set_title(ttl, fontsize=10.3, loc="left")
    ax.set_xlabel("시각 [s]  (0 = t=49, 예측 시작 · 10 Hz 작은 점 = 0.1 s, 2 Hz 큰 점 = 0.5 s)")
    ax.set_xlim(T10[0] - 0.12, 0.35)
    hd = ax.get_legend_handles_labels()[0]
    hd += [Patch(facecolor=SRC_BAND[c], label=f"10 Hz 출처 {SRC_NAMES[c]}") for c in (1, 2, 3, 0) if (hs == c).any()]
    hd += [Line2D([], [], color="none", marker="o", ms=6, mfc="white", mec=COL2, mew=1.6, label="2 Hz: AV2 보조 표본")]
    if len(j10) or len(j2):
        hd += [Line2D([], [], color=C.INK2, lw=1.0, ls=":", label="±π 점프 (값이 +180° ↔ −180° 로 넘어감)")]
    ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(-0.01, -0.36), ncol=4, fontsize=8.3)

    # ---------------- 제목 · 정의
    fd = p["focal"]
    y = Hh - 0.28
    fig.text(0.012, y / Hh, f"[{p['n']}. {p['cls']}]  시나리오 {sid}  —  (a, h) 입력 전처리: 10 Hz vs 2 Hz",
             fontsize=17, fontweight="bold", va="top", color=C.INK)
    fig.text(0.012, (y - 0.42) / Hh,
             f"focal 과거 4.9 s: 순 방향 변화(AV2) {fd['dh_deg']:+.0f}° · AV2 평균 속력 {fd['vmean']:.1f} m/s · 이동 {fd['path_m']:.0f} m · "
             f"10 Hz h 출처 위치차분 {fx['src10_pct']['위치차분']:.0f}% / AV2 보조 {fx['src10_pct']['AV2 보조']:.0f}% · "
             f"2 Hz 표본 중 AV2 보조 {fx['src2_n_av2']}/10 · focal ±π 점프 10 Hz {fx['jump10']} / 2 Hz {fx['jump2']}회   |   "
             f"주변 차량 {fx['n_kept']}대 (후보 {fx['n_cand']}) · 위치차분이 한 번도 없는 차량 {fx['never_moved']}대",
             fontsize=10.6, va="top", color=C.INK2)
    fig.text(0.012, (y - 0.70) / Hh, f"고른 규칙 ({p['cls']}): {p['rule']} — 후보 {p['n_candidates']:,}개 중 시드 {C.SEED} 로 1개",
             fontsize=9.6, va="top", color=C.MUTED)
    foot = [
        "좌표 = focal 정규화 (원점 = focal t=49 위치, +x = focal t=49 AV2 heading).  10 Hz (a, h) = heading_decomp.ah_features "
        f"(캐시 {CACHE10.name}),  2 Hz = ah_features_2hz (캐시 {CACHE2.name}: 관측 위치를 Savitzky–Golay 5점·2차로 인덱스 4 부터 평활 → 인덱스 4, 9, …, 49).",
        "h[t] = t → t+1 구간의 진행방향 (마지막 값은 직전 값 복제).  구간 속력 < 1 m/s 이면 AV2 heading 으로 메운다(뒤집힘 교정·튐 가드 포함).  "
        "a = 속도 증분 [km/h]/3, 첫 값 = 시작 속도.  빗금 = 창 시작 0.5 s.  θ 의 k = 5 m 안 같은 방향 최근접 중심선 접선(heading_decomp.decompose).",
        "주변 차량 10 Hz h 는 캐시 agents_hist_val (prep_agents_heading.py). 2 Hz 는 캐시가 없어 관측 50스텝이 모두 있는 차량만 ah_features_2hz 로 즉석 계산했다.",
    ]
    for k, s in enumerate(foot):
        fig.text(0.012, (0.1 + 0.24 * (len(foot) - 1 - k)) / Hh, s, fontsize=8.6, color=C.MUTED, va="bottom")
    C.savefig(fig, path, dpi=135)
    fx["file"] = str(Path(path).relative_to(C.ROOT))
    fx["nb_jumps"] = {"10hz": int(len(j10)), "2hz": int(len(j2))}
    fx["lane_cum_lat_m"] = float(cum[-1]) if cum is not None else None
    fx["lc_span_s"] = [float(v) for v in lc_span] if lc_span else None
    fx["a10_max_abs"] = float(np.abs(a10[1:OBS - 1]).max())
    fx["nb_flip"] = nb_flip_info
    fx["h10_h2_t0_diff_deg"] = float(np.degrees(abs(wrap(h10[OBS - 1] - h2[-1]))))
    fx["a2_max_abs"] = float(np.abs(a2[1:-1]).max())
    return fx


def run_panels(S, picks, data):
    out = []
    for p in picks["picks"]:
        path = OUT / f"panel_{p['n']:02d}_{SLUG[p['cls']]}.png"
        fx = draw_panel(S, p, path)
        out.append({"n": p["n"], "cls": p["cls"], "sid": p["sid"], **fx})
        print(f"[panel] {path.name}  N 점프 10/2 Hz {fx['nb_jumps']}  램프 {fx['ramp_ratio']}", flush=True)
    (data / "panel_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


# ============================================================================ 4. 분포
def pctl(x, qs=(50, 90, 95, 99, 99.9)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {f"p{q:g}": float(np.percentile(x, q)) for q in qs} if len(x) else {}


def ccdf(ax, x, color, label, ls="-", lw=1.8):
    x = np.sort(np.asarray(x, np.float64))
    n = len(x)
    y = 1.0 - np.arange(n) / n
    k = np.unique(np.r_[np.linspace(0, n - 1, 4000).astype(int), np.arange(max(0, n - 2000), n)])
    ax.plot(x[k], y[k], color=color, ls=ls, lw=lw, label=label)


def bar_labels(ax, bars, fmt="{:.1f}%", dy=0.5, fs=8.5, color=None):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=fs,
                color=color or C.INK2)


def run_dist(S, data):
    plt = C.setup_mpl()
    n = S.n
    summ = {"population": f"val {n:,} 시나리오 (focal) · 주변 차량은 같은 시나리오의 캐시 슬롯"}

    # ------------------------------------------------ A. ±π 점프 · 스텝 간 방향 변화
    j10 = np.array([len(jumps(h)) for h in S.h10])
    j2 = np.array([len(jumps(h)) for h in S.h2])
    dh10 = np.abs(wrap(np.diff(S.h10[:, :OBS - 1], axis=1)))           # t = 0..47 (마지막 복제 제외)
    dh2 = np.abs(wrap(np.diff(S.h2[:, :len(IDX2) - 1], axis=1)))       # k = 0..7
    mv10 = (S.src10[:, :OBS - 2] == 1) & (S.src10[:, 1:OBS - 1] == 1)
    mv2 = (S.src2[:, :-2] == 1) & (S.src2[:, 1:-1] == 1)
    r10 = np.degrees(dh10) / 0.1
    r2 = np.degrees(dh2) / 0.5
    sp10 = np.minimum(S.v10[:, :OBS - 2], S.v10[:, 1:OBS - 1])
    sp2 = np.minimum(S.v2[:, :-2], S.v2[:, 1:-1])
    summ["jump"] = {"10hz_scen_pct": float((j10 > 0).mean() * 100), "2hz_scen_pct": float((j2 > 0).mean() * 100),
                    "10hz_scen_n": int((j10 > 0).sum()), "2hz_scen_n": int((j2 > 0).sum()),
                    "both_n": int(((j10 > 0) & (j2 > 0)).sum()), "only10_n": int(((j10 > 0) & (j2 == 0)).sum()),
                    "only2_n": int(((j10 == 0) & (j2 > 0)).sum()),
                    "10hz_mean_count_if_any": float(j10[j10 > 0].mean()) if (j10 > 0).any() else 0.0,
                    "2hz_mean_count_if_any": float(j2[j2 > 0].mean()) if (j2 > 0).any() else 0.0,
                    "def": "입력 h 의 이웃 스텝 차(wrap 전) |Δ| > π 가 1번 이상인 시나리오"}
    SPB = [1, 2, 5, 10, np.inf]
    SPL = ["1–2", "2–5", "5–10", "≥10"]
    rate = {"all_10": pctl(r10), "all_2": pctl(r2), "mv_10": pctl(r10[mv10]), "mv_2": pctl(r2[mv2]),
            "n_all_10": int(r10.size), "n_all_2": int(r2.size), "n_mv_10": int(mv10.sum()), "n_mv_2": int(mv2.sum()),
            "by_speed": {}}
    for a, b, lab in zip(SPB[:-1], SPB[1:], SPL):
        m10 = mv10 & (sp10 >= a) & (sp10 < b)
        m2 = mv2 & (sp2 >= a) & (sp2 < b)
        rate["by_speed"][lab] = {"10": pctl(r10[m10], (50, 90, 99)), "2": pctl(r2[m2], (50, 90, 99)),
                                 "n10": int(m10.sum()), "n2": int(m2.sum())}
    summ["turn_rate_deg_s"] = rate

    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.4), gridspec_kw={"width_ratios": [0.8, 1.25, 1.25]})
    ax = axs[0]
    vals = [summ["jump"]["10hz_scen_pct"], summ["jump"]["2hz_scen_pct"]]
    bars = ax.bar(["10 Hz h", "2 Hz h"], vals, color=[COL10, COL2], width=0.55)
    bar_labels(ax, bars, "{:.2f}%", dy=max(vals) * 0.02)
    for b, c in zip(bars, [summ["jump"]["10hz_scen_n"], summ["jump"]["2hz_scen_n"]]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2, f"{c:,}개", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.75 + 0.01)
    ax.set_ylabel("시나리오 비율 [%]")
    ax.set_title("(a) focal 입력 h 에 ±π 점프가 있는 시나리오", loc="left")
    ax.text(0.98, 0.97, f"둘 다 {summ['jump']['both_n']:,} · 10 Hz 만 {summ['jump']['only10_n']:,} · "
            f"2 Hz 만 {summ['jump']['only2_n']:,}\n점프 있는 시나리오의 평균 횟수: 10 Hz {summ['jump']['10hz_mean_count_if_any']:.1f}"
            f" / 2 Hz {summ['jump']['2hz_mean_count_if_any']:.1f}\nn = {n:,}", transform=ax.transAxes, va="top",
            ha="right", fontsize=8.6, color=C.INK2)
    ax.grid(axis="x", visible=False)
    ax = axs[1]
    ccdf(ax, r10.ravel(), COL10, f"10 Hz 전체 스텝 (n={r10.size:,})")
    ccdf(ax, r2.ravel(), COL2, f"2 Hz 전체 표본 (n={r2.size:,})")
    ccdf(ax, r10[mv10], COL10, f"10 Hz 위치차분 구간 (n={int(mv10.sum()):,})", ls="--")
    ccdf(ax, r2[mv2], COL2, f"2 Hz 위치차분 구간 (n={int(mv2.sum()):,})", ls="--")
    ax.axvline(73.0, color=C.INK2, lw=0.9, ls=":")
    ax.text(76, 2e-6, "73°/s = 7.3°/0.1 s\n(focal 라벨 p99.99)", fontsize=7.8, color=C.INK2, va="bottom")
    ax.text(1850, 2e-2, "180° 뒤집힘\n= 1800°/s", fontsize=7.8, color=COL10, ha="right", va="bottom")
    ax.text(370, 5e-2, "180°/0.5 s\n= 360°/s", fontsize=7.8, color=COL2, ha="right", va="bottom")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_xlim(0, 3000)
    ax.set_ylim(1e-6, 1.2)
    ax.set_xlabel("스텝 간 방향 변화 |Δh| [°/s]  (10 Hz: ÷0.1 s, 2 Hz: ÷0.5 s)")
    ax.set_ylabel("이 값을 넘는 비율")
    ax.set_title("(b) focal 스텝 간 방향 변화 — 초당 값으로 맞춤", loc="left")
    ax.legend(loc="lower left", fontsize=8.4)
    txt = (f"위치차분 구간  p50 / p99 / p99.9 [°/s]\n"
           f"10 Hz  {rate['mv_10']['p50']:.2f} / {rate['mv_10']['p99']:.1f} / {rate['mv_10']['p99.9']:.1f}\n"
           f" 2 Hz  {rate['mv_2']['p50']:.2f} / {rate['mv_2']['p99']:.1f} / {rate['mv_2']['p99.9']:.1f}")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8.6, family="Noto Sans CJK KR",
            color=C.INK2, bbox=dict(facecolor=C.SURF, edgecolor=C.GRID))
    ax = axs[2]
    x = np.arange(len(SPL))
    for off, key, col, lab in ((-0.17, "10", COL10, "10 Hz"), (0.17, "2", COL2, "2 Hz")):
        p50 = [rate["by_speed"][l][key].get("p50", np.nan) for l in SPL]
        p90 = [rate["by_speed"][l][key].get("p90", np.nan) for l in SPL]
        p99 = [rate["by_speed"][l][key].get("p99", np.nan) for l in SPL]
        ax.vlines(x + off, p50, p99, color=col, lw=2.0, alpha=0.45)
        ax.scatter(x + off, p50, color=col, s=46, zorder=3, label=f"{lab} 중앙")
        ax.scatter(x + off, p90, color=col, s=34, marker="s", zorder=3, facecolor="white", label=f"{lab} p90")
        ax.scatter(x + off, p99, color=col, s=46, marker="^", zorder=3, label=f"{lab} p99")
    for k, l in enumerate(SPL):
        b = rate["by_speed"][l]
        ax.text(k, 1.02, f"n {b['n10']:,}\n/ {b['n2']:,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color=C.MUTED)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{l} m/s" for l in SPL])
    ax.set_xlabel("스텝 속력 (위치차분, 구간 양 끝의 작은 값)")
    ax.set_ylabel("|Δh| [°/s]")
    ax.set_title("(c) 위치차분 구간의 방향 변화 — 속력별", loc="left", pad=26)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    fig.suptitle(f"focal 입력 h: 10 Hz vs 2 Hz — ±π 점프와 스텝 간 방향 변화 (val {n:,})", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "위치차분 구간 = 구간 양 끝 스텝의 h 출처가 모두 위치차분(구간 속력 ≥ 1 m/s). 마지막 값(직전 복제)은 뺐다. "
             "10 Hz 는 스텝 48개 × 시나리오, 2 Hz 는 표본 간 8개 × 시나리오.", fontsize=8.6, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    C.savefig(fig, OUT / "dist_1_jump_turnrate.png")

    # ------------------------------------------------ B. 슬립각 · 복원
    mv = S.src10[:, :OBS - 1] == 1
    slip = np.degrees(wrap(S.head_n[:, :OBS - 1].astype(np.float64) - S.h10[:, :OBS - 1]))
    sv = S.v10[:, :OBS - 1]
    sl = slip[mv]
    SB = [1, 3, 6, 10, 15, np.inf]
    SBL = ["1–3", "3–6", "6–10", "10–15", "≥15"]
    sb = {}
    for a, b, lab in zip(SB[:-1], SB[1:], SBL):
        m = mv & (sv >= a) & (sv < b)
        sb[lab] = {**pctl(np.abs(slip[m]), (50, 90, 95, 99)), "n": int(m.sum()),
                   "gt90_pct": float((np.abs(slip[m]) > 90).mean() * 100)}
    rec = S.rec.astype(np.float64)
    summ["slip"] = {"n_steps": int(mv.sum()), "abs": pctl(np.abs(sl)), "signed_mean": float(sl.mean()),
                    "signed_median": float(np.median(sl)), "gt90_pct": float((np.abs(sl) > 90).mean() * 100),
                    "gt90_n": int((np.abs(sl) > 90).sum()),
                    "scen_with_gt90": int((np.abs(np.where(mv, slip, 0)) > 90).any(1).sum()),
                    "by_speed": sb,
                    "def": "AV2 heading − 10 Hz h (위치차분 스텝 t=0..48). 속력 = 10 Hz 위치차분 구간 속력"}
    summ["recon_final_err_m"] = {
        k: {"mean": float(rec[:, j].mean()), **pctl(rec[:, j], (50, 95, 99))}
        for j, k in enumerate(["10hz_av2", "10hz_h", "2hz_av2", "2hz_h"])}
    summ["recon_final_err_m"]["def"] = ("방향만 바꿔 과거 궤적을 다시 굴린 끝점 오차 (heading_decomp.rollout, 스텝 길이는 위치 그대로). "
                                        "10 Hz = 원본 위치 50점(4.9 s), 2 Hz = 평활 위치 10점(4.5 s). AV2 = guard 만 건 AV2 heading")
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    ax = axs[0]
    bins = np.arange(-15, 15.01, 0.25)
    cnt, _ = np.histogram(np.clip(sl, -15, 15), bins=bins)
    ax.bar(bins[:-1], cnt / len(sl) * 100, width=0.25, align="edge", color=COL10, alpha=0.85)
    for q in (5, 95):
        v = np.percentile(sl, q)
        ax.axvline(v, color=C.INK2, lw=1, ls="--")
        ax.text(v, ax.get_ylim()[1] * 0.55, f" p{q} {v:+.1f}°", fontsize=8.4, color=C.INK2, va="top")
    ax.set_xlabel("AV2 heading − 위치차분 h [°]  (양 끝 칸에 범위 밖을 모음)")
    ax.set_ylabel("스텝 비율 [%] (0.25° 칸)")
    ax.set_title("(a) focal 슬립각 + 잡음 — 위치차분 스텝", loc="left")
    a_ = summ["slip"]["abs"]
    ax.text(0.02, 0.97, f"n = {len(sl):,} 스텝\n|차| 중앙 {a_['p50']:.2f}° · p90 {a_['p90']:.2f}° · p99 {a_['p99']:.1f}°\n"
            f"|차| > 90° (후진·라벨 뒤집힘) {summ['slip']['gt90_pct']:.2f}%", transform=ax.transAxes, va="top",
            fontsize=8.6, color=C.INK2, bbox=dict(facecolor=C.SURF, edgecolor=C.GRID))
    ax = axs[1]
    x = np.arange(len(SBL))
    for key, mk, lab in (("p50", "o", "중앙"), ("p90", "s", "p90"), ("p99", "^", "p99")):
        ax.plot(x, [sb[l][key] for l in SBL], marker=mk, color=COL10, lw=1.4,
                mfc=COL10 if key != "p90" else "white", label=f"|AV2 − h| {lab}")
    for k, l in enumerate(SBL):
        ax.text(k, 1.02, f"n {sb[l]['n']:,}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.8, color=C.MUTED)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{l} m/s" for l in SBL])
    ax.set_xlabel("스텝 속력 (위치차분)")
    ax.set_ylabel("|AV2 − h| [°]")
    ax.set_title("(b) 속력별 — 저속일수록 위치 잡음이 섞인다", loc="left", pad=18)
    ax.legend(fontsize=8.4)
    ax = axs[2]
    labs = ["AV2 heading\n10 Hz", "h (위치차분)\n10 Hz", "AV2 heading\n2 Hz", "h\n2 Hz"]
    cols = [COLAV2, COL10, COLAV2, COL2]
    R_ = summ["recon_final_err_m"]
    keys = ["10hz_av2", "10hz_h", "2hz_av2", "2hz_h"]
    xs = np.arange(4)
    means = [R_[k]["mean"] for k in keys]
    p95 = [R_[k]["p95"] for k in keys]
    bars = ax.bar(xs, means, color=cols, width=0.6, alpha=0.9)
    ax.scatter(xs, p95, marker="_", s=500, color=C.INK, zorder=3, label="p95")
    for k, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, max(means[k], p95[k]) * 1.04 + 0.01,
                f"평균 {means[k]:.3f}\np95 {p95[k]:.2f}", ha="center", va="bottom", fontsize=8.2, color=C.INK2)
    ax.set_xticks(xs, labs, fontsize=8.6)
    ax.set_ylim(0, max(p95) * 1.35)
    ax.set_ylabel("끝점 오차 [m]")
    ax.set_title("(c) 방향만 바꿔 과거를 다시 굴린 끝점 오차", loc="left")
    ax.legend(loc="upper right", fontsize=8.4)
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"왜 h 를 직접 만드나 — AV2 heading(차체 방향)과 위치차분 진행방향의 차이 (val {n:,})", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "(c) 스텝 길이는 위치에서 그대로 쓰고 방향만 바꿔 누적한다 (heading_decomp.rollout). 10 Hz = 원본 위치 50점(4.9 s), "
             "2 Hz = 평활 위치 10점(4.5 s). h 는 저속(<1 m/s) 구간을 AV2 로 메우므로 0 이 아니다.", fontsize=8.6, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    C.savefig(fig, OUT / "dist_2_slip_recon.png")

    # ------------------------------------------------ C. 창 시작 램프
    ref = S.v10[:, RAMP_REF].mean(1)
    pop = ref > RAMP_MIN_V
    npop = int(pop.sum())
    NT = 30
    r10 = S.v10[pop, :NT] / ref[pop, None]
    rf = S.vf[pop, :NT].astype(np.float64) / ref[pop, None]
    q10 = np.percentile(r10, [25, 50, 75], axis=0)
    qf = np.percentile(rf, [25, 50, 75], axis=0)
    r2_0 = S.v2[pop, 0] / ref[pop]
    r2_1 = S.v2[pop, 1] / ref[pop]
    r2raw = S.v2x[pop, 0] / ref[pop]
    r2alt = S.v2x[pop, 1] / ref[pop]
    acc10 = np.abs(S.x10[pop, 1:OBS - 1, 0]) * KMH / 0.1           # t = 1..48 [m/s²]
    acc2 = np.abs(S.x2[pop, 1:len(IDX2) - 1, 0]) * KMH / 0.5       # k = 1..8
    accf = np.abs(np.diff(S.vf[pop].astype(np.float64), axis=1)) / 0.1   # t = 1..49
    v0ratio = S.x10[pop, 0, 0] * KMH / np.maximum(S.vf[pop, 0], 1e-6)
    summ["ramp"] = {
        "population": f"창 시작 후 1.0~1.6 s (스텝 10~15) 위치차분 속력 평균 > {RAMP_MIN_V} m/s 인 focal",
        "n": npop, "ref": "위치차분 속력 v10[10..15] 평균",
        "ratio10_median_t0_9": [float(v) for v in q10[1, :10]],
        "ratio_field_median_t0_9": [float(v) for v in qf[1, :10]],
        "first10_lt_0.8_pct": float((r10[:, 0] < 0.8).mean() * 100),
        "v0_10hz_over_field_t0_median": float(np.median(v0ratio)),
        "2hz_first_seg_ratio": pctl(r2_0, (25, 50, 75)), "2hz_second_seg_ratio": pctl(r2_1, (25, 50, 75)),
        "2hz_first_seg_raw_pos_ratio": pctl(r2raw, (25, 50, 75)),
        "2hz_first_seg_if_smooth_from_0_ratio": pctl(r2alt, (25, 50, 75)),
        "2hz_first_seg_detail_vs_cache_max": float(np.abs(S.v2x[:, 2] - S.v2[:, 0]).max()),
        "abs_diff_skip_vs_smooth_from_0_ratio": pctl(np.abs(r2alt - r2_0), (50, 90, 99)),
        "stationary_start": (lambda m: {
            "def": "창 시작 1.5 s 동안 AV2 속력 최대 < 0.3 m/s 인 focal — 램프가 아니라 위치 잡음이 만드는 가짜 움직임",
            "n": int(m.sum()), "v10_0": pctl(S.v10[m, 0], (50, 90, 99)),
            "v10_0_ge1_pct": float((S.v10[m, 0] >= 1.0).mean() * 100),
            "v10_t0_5_median": [float(v) for v in np.median(S.v10[m, :6], axis=0)],
            "v10_t10_14_ge1_pct": float((S.v10[m, 10:15] >= 1.0).any(1).mean() * 100),
            "v2_0": pctl(S.v2[m, 0], (50, 90, 99)), "v2_0_ge1_pct": float((S.v2[m, 0] >= 1.0).mean() * 100),
            "src10_t0_posdiff_pct": float((S.src10[m, 0] == 1).mean() * 100)})(S.vf[:, :15].max(1) < 0.3),
        "ratio10_mean_idx4_8": float(np.median(r10[:, 4:9].mean(1))),
        "acc10_median_t1_8": [float(v) for v in np.median(acc10[:, :8], axis=0)],
        "acc10_median_t1_12": [float(v) for v in np.median(acc10[:, :12], axis=0)],
        "acc10_median_t12_48": float(np.median(acc10[:, 11:])),
        "2hz_seg_iqr_width": {"first": float(np.subtract(*np.percentile(r2_0, [75, 25]))),
                              "second": float(np.subtract(*np.percentile(r2_1, [75, 25])))},
        "artifact_note": "10 Hz |a| 중앙은 0.8 s 에 1.6 으로 내려간 뒤 0.9·1.0 s 에 다시 오른다 → 인공물 구간은 창 시작 약 1.1 s",
        "acc10_median_mid_t20_48": float(np.median(acc10[:, 19:])),
        "acc2_median_k1_8": [float(v) for v in np.median(acc2, axis=0)],
        "accfield_median_t1_8": [float(v) for v in np.median(accf[:, :8], axis=0)],
    }
    fig, axs = plt.subplots(1, 2, figsize=(18.0, 5.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axs[0]
    tt = np.arange(NT) * 0.1
    ax.fill_between(tt, q10[0], q10[2], color=COL10, alpha=0.18, lw=0)
    ax.plot(tt, q10[1], color=COL10, marker="o", ms=3.5, label="10 Hz 위치차분 속력 (a 누적합) 중앙 · IQR")
    ax.fill_between(tt, qf[0], qf[2], color=COLAV2, alpha=0.12, lw=0)
    ax.plot(tt, qf[1], color=COLAV2, ls=(0, (4, 2)), label="AV2 속도 필드 중앙 · IQR")
    for seg, rr, lab, col, ls in ((0, r2_0, "2 Hz 첫 표본 (인덱스 4→9, 평활)", COL2, "-"),
                                  (1, r2_1, "2 Hz 둘째 표본 (9→14)", COL2, "-")):
        t0 = IDX2[seg] * 0.1
        q = np.percentile(rr, [25, 50, 75])
        ax.plot([t0, t0 + 0.5], [q[1], q[1]], color=col, lw=3.2, ls=ls, label=lab if seg == 0 else lab)
        ax.fill_between([t0, t0 + 0.5], [q[0]] * 2, [q[2]] * 2, color=col, alpha=0.15, lw=0)
    qa = np.percentile(r2alt, [25, 50, 75])
    ax.plot([0.4, 0.9], [qa[1]] * 2, color=C.C_VIOLET, lw=2.4, ls=(0, (2, 1.5)),
            label="(가정) 평활을 인덱스 0 부터 했다면 첫 표본")
    ax.axvspan(-0.05, 0.45, facecolor="none", edgecolor=C.MUTED, hatch="///", lw=0, zorder=0)
    ax.axvline(0.4, color=COL2, lw=1, ls=":")
    ax.axhline(1.0, color=C.AXIS, lw=0.9)
    ax.axvspan(0.45, 1.15, facecolor="#f1f0ec", edgecolor="none", zorder=0)
    ax.set_xlabel("창 시작부터 시간 [s]  (0 = 인덱스 0)")
    top_pred_axis(ax)
    ax.set_ylabel("속력 ÷ 기준 속력")
    ax.set_ylim(0.2, 1.3)
    ax.set_title("(a) 창 시작 램프 — 위치 속력은 실제의 절반에서 출발해 0.7 s 에 넘친다 (회색 = 넘침·되돌림, ~1.1 s)",
                 loc="left", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5)
    rp = summ["ramp"]
    ax.text(0.01, 0.98, f"n = {npop:,} focal (기준 = 1.0~1.6 s 위치차분 속력 평균 > {RAMP_MIN_V:g} m/s)\n"
            f"10 Hz 첫 스텝 중앙 {q10[1, 0]:.2f}배 · 첫 스텝 < 0.8배 {rp['first10_lt_0.8_pct']:.1f}%\n"
            f"2 Hz 첫 표본 중앙 {rp['2hz_first_seg_ratio']['p50']:.3f}배 (원본 위치 그대로 {rp['2hz_first_seg_raw_pos_ratio']['p50']:.3f}, "
            f"평활을 0부터 {rp['2hz_first_seg_if_smooth_from_0_ratio']['p50']:.3f})\n"
            f"2 Hz IQR 폭: 첫 표본 {rp['2hz_seg_iqr_width']['first']:.3f} (Q1 {rp['2hz_first_seg_ratio']['p25']:.2f}–Q3 "
            f"{rp['2hz_first_seg_ratio']['p75']:.2f}) vs 둘째 {rp['2hz_seg_iqr_width']['second']:.3f} → 개별 값은 흔들린다",
            transform=ax.transAxes, va="top", fontsize=8.6, color=C.INK2,
            bbox=dict(facecolor=C.SURF, edgecolor=C.GRID, alpha=0.9))
    ax = axs[1]
    ax.plot(np.arange(1, OBS - 1) * 0.1, np.median(acc10, axis=0), color=COL10, marker="o", ms=3,
            label="10 Hz a → |가속도| 중앙")
    ax.plot(IDX2[1:-1] * 0.1, np.median(acc2, axis=0), color=COL2, marker="o", ms=6, lw=2,
            label="2 Hz a → |가속도| 중앙")
    ax.plot(np.arange(1, OBS) * 0.1, np.median(accf, axis=0), color=COLAV2, ls=(0, (4, 2)),
            label="AV2 속도 필드 차분 |가속도| 중앙")
    ax.axvspan(-0.05, 0.45, facecolor="none", edgecolor=C.MUTED, hatch="///", lw=0, zorder=0)
    ax.axvspan(0.45, 1.15, facecolor="#f1f0ec", edgecolor="none", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("창 시작부터 시간 [s]")
    top_pred_axis(ax)
    ax.set_ylabel("|가속도| [m/s²]")
    ax.set_title("(b) a 채널이 담는 가짜 가속 — 0.8 s 에 내려갔다 0.9–1.0 s 에 다시 오른다 (~1.1 s)", loc="left",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"창 시작 램프: 10 Hz 입력은 창 시작 약 1.1 s 에 가짜 가속을 담는다. 2 Hz 첫 표본은 중앙값만 1.017 로 보정되고 "
                 f"개별 값은 흔들린다 (val {n:,} 중 {npop:,})",
                 fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "(a) 2 Hz 막대는 그 0.5 s 구간 속력(평활 위치 차분), 옅은 띠 = IQR. 빗금 = 인덱스 0~4, 회색 = 0.5~1.1 s. "
             "기준 속력 = 스텝 10~15 (창 시작 1.0~1.6 s) 위치차분 속력 평균. (b) 10 Hz 는 t=1..48 (첫 값은 v0, 마지막은 0), "
             "2 Hz 는 k=1..8. 속도 필드 차분은 평활하지 않은 값이다.", fontsize=8.6, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    C.savefig(fig, OUT / "dist_3_ramp.png")

    # ------------------------------------------------ D. 주변 차량 수 · 종류
    T = S.T
    atype = S.ag["atype"]
    kept = atype > 0
    nk = kept.sum(1)
    nc = S.ag["n_cand"].astype(int)
    tcount = {TYPE_NAMES[c]: int((atype == c).sum()) for c in TYPE_NAMES}
    nkept = int(kept.sum())
    summ["agents_count"] = {"kept_mean": float(nk.mean()), "cand_mean": float(nc.mean()),
                            "cand_median": float(np.median(nc)), "cand_p95": float(np.percentile(nc, 95)),
                            "cand_max": int(nc.max()), "truncated_pct": float((nc > K).mean() * 100),
                            "zero_agents_pct": float((nk == 0).mean() * 100), "n_kept": nkept,
                            "type_pct": {k: v / nkept * 100 for k, v in tcount.items()}, "type_n": tcount,
                            "av_pct_of_kept": float(S.ag["is_av"][kept].mean() * 100),
                            "av_in_scen_pct": float((S.ag["is_av"].sum(1) > 0).mean() * 100)}
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.0), gridspec_kw={"width_ratios": [1.4, 0.9, 1.1]})
    ax = axs[0]
    mx = int(np.percentile(nc, 99.5)) + 2
    bins = np.arange(0, mx + 1) - 0.5
    ax.hist(np.minimum(nc, mx), bins=bins, color=C.BLUE_RAMP[6], label="후보 (t=0 관측, 4종류)", alpha=0.8)
    ax.hist(nk, bins=bins, histtype="step", color=C.INK, lw=1.5, label="캐시에 남은 수 (≤32)")
    ax.axvline(K + 0.5, color=C.C_RED, lw=1.4, ls="--")
    ax.text(K + 1, ax.get_ylim()[1] * 0.92, f"K = 32 에서 자름\n잘린 시나리오 {summ['agents_count']['truncated_pct']:.1f}%",
            color=C.C_RED, fontsize=8.8, va="top")
    ax.set_xlabel("시나리오당 주변 차량 수 (오른쪽 끝 칸에 그 이상을 모음)")
    ax.set_ylabel("시나리오 수")
    ax.set_title(f"(a) 시나리오당 대수 — 평균 남음 {nk.mean():.1f} / 후보 {nc.mean():.1f}", loc="left")
    ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.75))
    ax = axs[1]
    names = [TYPE_KO[c] for c in TYPE_NAMES]
    pct = [tcount[TYPE_NAMES[c]] / nkept * 100 for c in TYPE_NAMES]
    bars = ax.barh(names[::-1], pct[::-1], color=C.BLUE_RAMP[7])
    ax.set_xscale("log")
    for b, c in zip(bars, [tcount[TYPE_NAMES[c]] for c in TYPE_NAMES][::-1]):
        ax.text(b.get_width() * 1.1, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}% ({c:,})", va="center",
                fontsize=8.6, color=C.INK2)
    ax.set_xlim(0.1, 600)
    ax.set_xlabel("남은 주변 차량 중 비율 [%] (로그)")
    ax.set_title(f"(b) 종류 — n = {nkept:,}대", loc="left")
    ax.text(0.98, 0.03, f"AV(자율주행 차량) = 남은 차량의 {summ['agents_count']['av_pct_of_kept']:.2f}%\n"
            f"AV 가 들어간 시나리오 {summ['agents_count']['av_in_scen_pct']:.2f}%", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.4, color=C.INK2)
    ax.grid(axis="y", visible=False)
    ax = axs[2]
    d0 = S.ag["dist0"][kept]
    ax.hist(np.minimum(d0, 150), bins=np.arange(0, 151, 5), color=C.BLUE_RAMP[5])
    for e in DIST_EDGES[1:-1]:
        ax.axvline(e, color=C.GRID, lw=0.8)
    ax.set_xlabel("t=0 focal 과의 거리 [m] (150 이상은 끝 칸)")
    ax.set_ylabel("차량 수")
    ax.set_title(f"(c) 거리 — 중앙 {np.median(d0):.1f} m · p90 {np.percentile(d0, 90):.1f} m", loc="left")
    summ["agents_count"]["dist0"] = pctl(d0, (50, 90, 99))
    fig.suptitle(f"주변 차량 캐시 (agents_hist_val, K = 32) — val {n:,}", fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.savefig(fig, OUT / "dist_4_agents_count_type.png")

    # ------------------------------------------------ E. 주변 차량 h 출처
    Hs = S.H.astype(np.float64)       # [src, spd(0=미관측), dist, type]
    HT = S.HT.astype(np.float64)      # [src, track vmax, type]
    tot = Hs.sum()
    obs_tot = Hs[:, 1:].sum()
    src_all = {SRC_NAMES[c]: float(Hs[c].sum() / tot * 100) for c in range(4)}
    src_obs = {SRC_NAMES[c]: float(Hs[c, 1:].sum() / obs_tot * 100) for c in (1, 2, 3)}
    by_spd = {SPD_LAB[b - 1]: {**{SRC_NAMES[c]: float(Hs[c, b].sum() / max(Hs[:, b].sum(), 1) * 100) for c in (1, 2, 3)},
                               "n": int(Hs[:, b].sum())} for b in range(1, len(SPD_EDGES))}
    by_dist = {DIST_LAB[d]: {**{SRC_NAMES[c]: float(Hs[c, 1:, d].sum() / max(Hs[:, 1:, d].sum(), 1) * 100) for c in (1, 2, 3)},
                             "미관측(전체 스텝 중)": float(Hs[0, 0, d].sum() / max(Hs[:, :, d].sum(), 1) * 100),
                             "n_obs": int(Hs[:, 1:, d].sum())} for d in range(len(DIST_LAB))}
    by_type = {TYPE_NAMES[t + 1]: {**{SRC_NAMES[c]: float(Hs[c, 1:, :, t].sum() / max(Hs[:, 1:, :, t].sum(), 1) * 100)
                                      for c in (1, 2, 3)}, "n_obs": int(Hs[:, 1:, :, t].sum())} for t in range(4)}
    by_trk = {TRK_LAB[k].split("\n")[0]: {**{SRC_NAMES[c]: float(HT[c, k].sum() / max(HT[:, k].sum(), 1) * 100) for c in (1, 2, 3)},
                                          "n_obs": int(HT[:, k].sum()),
                                          "share_of_all_av2_steps": float(HT[2, k].sum() / max(HT[2].sum(), 1) * 100)}
              for k in range(len(TRK_LAB))}
    # AV2 보조 스텝이 어느 속력에서 나오나
    av2_spd_share = {SPD_LAB[b - 1]: float(Hs[2, b].sum() / Hs[2, 1:].sum() * 100) for b in range(1, len(SPD_EDGES))}
    trk_vmax = T["vmax"]
    summ["agents_src"] = {"pct_of_all_kept_steps": src_all, "pct_of_observed_steps": src_obs,
                          "by_step_speed_field": by_spd, "by_dist0": by_dist, "by_type": by_type,
                          "by_track_vmax": by_trk, "av2_steps_by_step_speed_share": av2_spd_share,
                          "tracks_by_vmax_pct": {TRK_LAB[k].split("\n")[0]: float(
                              ((trk_vmax >= TRK_EDGES[k]) & (trk_vmax < TRK_EDGES[k + 1])).mean() * 100)
                              for k in range(len(TRK_LAB))},
                          "def": "스텝 속력 = 그 스텝의 AV2 속도 필드 |v|. 출처 규칙은 위치차분 구간 속력 ≥ 1 m/s 이므로 두 속력은 다를 수 있다. "
                                 "트랙 최대속력 = 과거 4.9 s 관측 스텝의 AV2 |v| 최대"}
    fig, axs = plt.subplots(1, 4, figsize=(20.0, 5.6), gridspec_kw={"width_ratios": [1.25, 1.0, 1.15, 0.85]})

    def stacked(ax, labels, fracs, ns, title, xlabel, nfmt="n {:,}"):
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for c in (1, 2, 3):
            v = np.array([f[SRC_NAMES[c]] for f in fracs])
            ax.bar(x, v, bottom=bottom, color=SRC_COL[c], width=0.72, label=SRC_NAMES[c] if c != 3 or v.sum() > 0 else None)
            for k in range(len(x)):
                if v[k] >= 7:
                    ax.text(x[k], bottom[k] + v[k] / 2, f"{v[k]:.0f}", ha="center", va="center", fontsize=8.4,
                            color="white" if c != 2 else C.INK, fontweight="bold")
            bottom += v
        for k in range(len(x)):
            ax.text(x[k], 101, nfmt.format(ns[k]), ha="center", va="bottom", fontsize=7.4, color=C.MUTED, rotation=0)
        ax.set_xticks(x, labels, fontsize=8.6)
        ax.set_ylim(0, 110)
        ax.set_ylabel("관측 스텝 중 비율 [%]")
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="x", visible=False)

    stacked(axs[0], SPD_LAB, [by_spd[l] for l in SPD_LAB], [by_spd[l]["n"] for l in SPD_LAB],
            "(a) 스텝 속력별", "그 스텝의 AV2 속력 [m/s]")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.6)
    tl = [TRK_LAB[k] for k in range(len(TRK_LAB))]
    stacked(axs[1], tl, [by_trk[l.split("\n")[0]] for l in tl], [by_trk[l.split("\n")[0]]["n_obs"] for l in tl],
            "(b) 차량의 과거 4.9 s 최대 속력별", "트랙 최대 AV2 속력 [m/s]")
    for k, l in enumerate(tl):
        axs[1].text(k, -26, f"AV2 보조의\n{by_trk[l.split(chr(10))[0]]['share_of_all_av2_steps']:.0f}%",
                    ha="center", va="top", fontsize=8.2, color=C.C_YELLOW, fontweight="bold",
                    transform=axs[1].transData, clip_on=False)
    stacked(axs[2], DIST_LAB, [by_dist[l] for l in DIST_LAB], [by_dist[l]["n_obs"] for l in DIST_LAB],
            "(c) t=0 거리별", "focal 과의 거리 [m]")
    for k, l in enumerate(DIST_LAB):
        axs[2].text(k, 107.5, f"미관측 {by_dist[l]['미관측(전체 스텝 중)']:.0f}%", ha="center", va="bottom",
                    fontsize=7.6, color=C.INK, fontweight="bold")
    axs[2].set_ylim(0, 117)
    axs[2].text(0.5, -0.2, "막대 위 '미관측' = 그 거리 구간의 전체 50스텝 중 미관측 비율", transform=axs[2].transAxes,
                ha="center", va="top", fontsize=8.2, color=C.INK2)
    tn = [TYPE_NAMES[t] for t in TYPE_NAMES]
    stacked(axs[3], [TYPE_KO[t] for t in TYPE_NAMES], [by_type[t] for t in tn], [by_type[t]["n_obs"] for t in tn],
            "(d) 종류별", "")
    fig.suptitle(f"주변 차량 h 의 출처 — 관측 스텝 {int(obs_tot):,}개 (전체 {int(tot):,} 중 미관측 {src_all['미관측']:.1f}%)",
                 fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, f"전체 슬롯 스텝 기준: 위치차분 {src_all['위치차분']:.1f}% · AV2 보조 {src_all['AV2 보조']:.1f}% · 미관측 "
             f"{src_all['미관측']:.1f}% · 이웃 유지 {src_all['이웃 유지']:.2f}%  |  관측 스텝 기준: 위치차분 {src_obs['위치차분']:.1f}% · "
             f"AV2 보조 {src_obs['AV2 보조']:.1f}%.  AV2 보조 스텝의 {av2_spd_share['<0.5']:.0f}% 가 AV2 속력 0.5 m/s 미만 스텝이다. "
             "(b) 아래 노란 숫자 = 전체 AV2 보조 스텝 중 그 차량군이 낸 몫.", fontsize=8.6, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.subplots_adjust(bottom=0.25)
    C.savefig(fig, OUT / "dist_5_agents_src.png")

    # ------------------------------------------------ F. 안 움직인 차량 · 정지 차량 · 2 Hz 계산 가능 · 점프
    never = T["n_src1"] == 0
    few = T["n_src1"] < 3
    stat = T["vmax"] < 0.5
    full = T["full_obs"] == 1
    ty = T["atype"].astype(int)
    db = np.searchsorted(DIST_EDGES, T["dist0"], side="right") - 1
    flip = T["flip"]
    movers = full & (T["n_src1"] >= 5) & (T["vmax"] >= 2.0)
    j10a = T["jump10"] > 0
    j2a = T["jump2"] > 0
    summ["agents_moved"] = {
        "n_tracks": int(len(never)), "never_posdiff_pct": float(never.mean() * 100),
        "lt3_posdiff_pct": float(few.mean() * 100),
        "stationary_pct": float(stat.mean() * 100),
        "stationary_with_posdiff_pct_of_stationary": float((stat & ~never).sum() / max(stat.sum(), 1) * 100),
        "stationary_ge3_posdiff_pct_of_stationary": float((stat & ~few).sum() / max(stat.sum(), 1) * 100),
        "posdiff_steps_at_field_speed_lt_0.5_pct": float(T["n_src1_slow"].sum() / max(T["n_src1"].sum(), 1) * 100),
        "never_by_type_pct": {TYPE_NAMES[t]: float(never[ty == t].mean() * 100) for t in TYPE_NAMES},
        "never_by_dist_pct": {DIST_LAB[d]: float(never[db == d].mean() * 100) for d in range(len(DIST_LAB))},
        "stationary_by_type_pct": {TYPE_NAMES[t]: float(stat[ty == t].mean() * 100) for t in TYPE_NAMES},
        "stationary_by_dist_pct": {DIST_LAB[d]: float(stat[db == d].mean() * 100) for d in range(len(DIST_LAB))},
        "full_obs_pct": float(full.mean() * 100),
        "full_obs_by_dist_pct": {DIST_LAB[d]: float(full[db == d].mean() * 100) for d in range(len(DIST_LAB))},
        "full_obs_by_type_pct": {TYPE_NAMES[t]: float(full[ty == t].mean() * 100) for t in TYPE_NAMES},
        "full_obs_never_posdiff_pct": float(never[full].mean() * 100),
        "jump_full_obs": {"n": int(full.sum()), "10hz_pct": float(j10a[full].mean() * 100),
                          "2hz_pct": float(j2a[full].mean() * 100)},
        "jump_movers": {"n": int(movers.sum()), "10hz_pct": float(j10a[movers].mean() * 100),
                        "2hz_pct": float(j2a[movers].mean() * 100),
                        "10hz_mean_count": float(T["jump10"][movers].mean()),
                        "2hz_mean_count": float(T["jump2"][movers].mean())},
        "jump_stationary_full": {"n": int((full & stat).sum()), "10hz_pct": float(j10a[full & stat].mean() * 100),
                                 "2hz_pct": float(j2a[full & stat].mean() * 100)},
        "2hz_av2_samples_pct_full": float(T["n2_av2"][full].sum() / (full.sum() * len(IDX2)) * 100),
        "10hz_av2_steps_pct_full": float(T["n_src2"][full].sum() / (full.sum() * OBS) * 100),
        "def": "안 움직임(위치차분 0) = 과거 4.9 s 관측 스텝 중 h 출처가 위치차분인 스텝이 0개. 정지 = 과거 4.9 s 관측 스텝의 AV2 속력 최대 < 0.5 m/s. "
               "뒤집힘 판정(align_ref)은 위치차분 스텝 ≥ 3 일 때만 한다. 2 Hz 계산 가능 = 관측 50스텝이 모두 있음. "
               "movers = 2 Hz 계산 가능 & 위치차분 스텝 ≥ 5 & AV2 속력 최대 ≥ 2 m/s",
    }
    am = summ["agents_moved"]
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    ax = axs[0]
    x = np.arange(len(DIST_LAB))
    ax.plot(x, [float(stat[db == d].mean() * 100) for d in range(len(DIST_LAB))], color=C.C_MAGENTA, marker="o",
            lw=2, label="정지 (4.9 s 내내 AV2 속력 < 0.5 m/s)")
    ax.plot(x, [am["never_by_dist_pct"][l] for l in DIST_LAB], color=C.C_YELLOW, marker="o",
            lw=2, label="위치차분 스텝 0개 (출처 1 없음)")
    ax.plot(x, [float(few[db == d].mean() * 100) for d in range(len(DIST_LAB))], color=C.C_YELLOW, marker="s",
            mfc="white", lw=1.2, ls="--", label="위치차분 < 3 스텝 (뒤집힘 판정 안 함)")
    ax.plot(x, [am["full_obs_by_dist_pct"][l] for l in DIST_LAB], color=COL2, marker="o", lw=2,
            label="관측 50스텝 전부 (2 Hz 계산 가능)")
    for d in range(len(DIST_LAB)):
        ax.text(d, 1.01, f"n {int((db == d).sum()):,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color=C.MUTED)
    ax.set_xticks(x, DIST_LAB)
    ax.set_ylim(0, 100)
    ax.set_xlabel("t=0 focal 과의 거리 [m]")
    ax.set_ylabel("차량 비율 [%]")
    ax.set_title("(a) 거리별", loc="left", pad=16)
    ax.legend(fontsize=8.2, loc="lower left")
    ax = axs[1]
    tlab = [TYPE_KO[t] for t in TYPE_NAMES]
    x = np.arange(4)
    w = 0.27
    b0 = ax.bar(x - w, [am["stationary_by_type_pct"][TYPE_NAMES[t]] for t in TYPE_NAMES], w,
                color=C.C_MAGENTA, label="정지")
    b1 = ax.bar(x, [am["never_by_type_pct"][TYPE_NAMES[t]] for t in TYPE_NAMES], w,
                color=C.C_YELLOW, label="위치차분 0개")
    b2 = ax.bar(x + w, [am["full_obs_by_type_pct"][TYPE_NAMES[t]] for t in TYPE_NAMES], w,
                color=COL2, label="2 Hz 계산 가능")
    bar_labels(ax, list(b0) + list(b1) + list(b2), "{:.0f}", dy=1, fs=7.8)
    for k, t in enumerate(TYPE_NAMES):
        ax.text(k, 1.01, f"n {int((ty == t).sum()):,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color=C.MUTED)
    ax.set_xticks(x, tlab)
    ax.set_ylim(0, 105)
    ax.set_ylabel("차량 비율 [%]")
    ax.set_title("(b) 종류별", loc="left", pad=16)
    ax.legend(fontsize=8.2, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axs[2]
    groups = [("2 Hz 계산 가능\n전체", full), ("그중 움직인 차", movers), ("그중 정지 차", full & stat)]
    x = np.arange(len(groups))
    w = 0.38
    v10 = [float(j10a[m].mean() * 100) for _, m in groups]
    v2 = [float(j2a[m].mean() * 100) for _, m in groups]
    b1 = ax.bar(x - w / 2, v10, w, color=COL10, label="10 Hz h (캐시)")
    b2 = ax.bar(x + w / 2, v2, w, color=COL2, label="2 Hz h (즉석 계산)")
    bar_labels(ax, list(b1) + list(b2), "{:.1f}%", dy=0.3)
    for k, (_, m) in enumerate(groups):
        ax.text(k, 1.01, f"n {int(m.sum()):,}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.6, color=C.MUTED)
    ax.set_xticks(x, [g for g, _ in groups], fontsize=8.6)
    ax.set_ylabel("±π 점프가 있는 차량 비율 [%]")
    ax.set_title("(c) 주변 차량 ±π 점프: 10 Hz vs 2 Hz", loc="left", pad=16)
    ax.legend(fontsize=8.4)
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"주변 차량: 정지 {am['stationary_pct']:.1f}% · 위치차분 0개 {am['never_posdiff_pct']:.1f}% · "
                 f"2 Hz 계산 가능 {am['full_obs_pct']:.1f}% (n = {am['n_tracks']:,}대)", fontsize=14, fontweight="bold",
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, f"정지 차량의 {am['stationary_with_posdiff_pct_of_stationary']:.0f}% 는 위치 잡음(0.1 s 에 10 cm 이상)으로 위치차분 스텝을 1개 이상 갖고, "
             f"{am['stationary_ge3_posdiff_pct_of_stationary']:.0f}% 는 3개 이상이라 뒤집힘 판정이 걸린다. 위치차분 스텝의 "
             f"{am['posdiff_steps_at_field_speed_lt_0.5_pct']:.1f}% 가 AV2 속력 < 0.5 m/s 스텝이다.  "
             f"2 Hz 계산 가능 차량의 AV2 보조 비율: 10 Hz 스텝 {am['10hz_av2_steps_pct_full']:.1f}% / 2 Hz 표본 {am['2hz_av2_samples_pct_full']:.1f}%.",
             fontsize=8.6, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    C.savefig(fig, OUT / "dist_6_agents_moved_2hz.png")

    # ------------------------------------------------ G. 뒤집힘 교정 · 차로 방향 대조
    PB = [0.0, 2.0, 5.0, 20.0, 50.0, np.inf]
    PBL = ["0–2", "2–5", "5–20", "20–50", "≥50"]
    pb = np.searchsorted(PB, T["path"], side="right") - 1
    f2 = T["flip2"]
    lm = T["lane_m"] == 1
    lane = {"def": f"t=0 위치가 교차로 밖 차로 중심선 {LANE_IN_M:g} m 안인 주변 차량. 그 점 접선 방향과 ±90° 안이면 '같은 쪽'. "
                   "AV2 = t=0 AV2 heading 원본, 10 Hz = 캐시 h[49], 2 Hz = 즉석 2 Hz h[9] (관측 50스텝 전부인 차량만)",
            "n_in_lane": int(lm.sum()), "by_path": {}, "by_vmax": {}}
    flipinfo = {"def": "10 Hz: 캐시 h 를 만든 build_heading 의 align_ref 와 같은 기준으로 다시 판정(위치차분 스텝 ≥ 3). "
                       "2 Hz: 관측 50스텝 전부인 차량에서 2 Hz 표본(위치차분 ≥ 3)으로 같은 판정", "by_path": {}}
    for k, l in enumerate(PBL):
        m = pb == k
        v10m = m & (flip >= 0)
        v2m = m & (f2 >= 0)
        flipinfo["by_path"][l] = {"n": int(m.sum()), "n_verif10": int(v10m.sum()),
                                  "flip10_pct": float((flip[v10m] == 1).mean() * 100) if v10m.any() else None,
                                  "n_verif2": int(v2m.sum()),
                                  "flip2_pct": float((f2[v2m] == 1).mean() * 100) if v2m.any() else None,
                                  "fill_disagree_pct_full": float((T["fill_dis"][m & full] > 0).mean() * 100) if (m & full).any() else None,
                                  "n_full": int((m & full).sum())}
        mm = m & lm
        mf = mm & (T["ag_2"] >= 0)
        lane["by_path"][l] = {"n": int(mm.sum()), "opp_av2_pct": float((T["ag_av2"][mm] == 0).mean() * 100) if mm.any() else None,
                              "opp_10_pct": float((T["ag_10"][mm] == 0).mean() * 100) if mm.any() else None,
                              "n_full": int(mf.sum()),
                              "opp_av2_pct_full": float((T["ag_av2"][mf] == 0).mean() * 100) if mf.any() else None,
                              "opp_10_pct_full": float((T["ag_10"][mf] == 0).mean() * 100) if mf.any() else None,
                              "opp_2_pct_full": float((T["ag_2"][mf] == 0).mean() * 100) if mf.any() else None}
    for name, m in (("정지 <0.5", stat), ("0.5–2", (T["vmax"] >= 0.5) & (T["vmax"] < 2)),
                    ("2–5", (T["vmax"] >= 2) & (T["vmax"] < 5)), ("≥5", T["vmax"] >= 5)):
        mm = m & lm
        lane["by_vmax"][name] = {"n": int(mm.sum()), "opp_av2_pct": float((T["ag_av2"][mm] == 0).mean() * 100),
                                 "opp_10_pct": float((T["ag_10"][mm] == 0).mean() * 100)}
    so = stat & lm & (T["ag_10"] == 0)
    s49 = S.ag["h_src"][T["scen"].astype(int), T["slot"].astype(int), OBS - 1]
    lane["stationary_opp10_breakdown"] = {
        "def": "차로 안 정지 차량 중 10 Hz h[49] 가 차로와 반대인 트랙을 t=0 출처로 나눈다",
        "n": int(so.sum()),
        "track_flip_decision_1": int((so & (flip == 1)).sum()), "track_flip_decision_not1": int((so & (flip != 1)).sum()),
        "t0_av2fill_flipped": int((so & (s49 == 2) & (flip == 1)).sum()),
        "t0_av2fill_not_flipped_av2_itself_opposite": int((so & (s49 == 2) & (flip != 1)).sum()),
        "t0_posdiff_noise": int((so & (s49 == 1)).sum()),
        "t0_posdiff_noise_on_flipped_track": int((so & (s49 == 1) & (flip == 1)).sum())}
    sf = stat & lm & (flip == 1) & (T["n_src2"] > 0)
    lane["stationary_flipped10"] = {"def": "차로 안 정지 차량 중 10 Hz 가 실제로 AV2 보조 값을 뒤집은 트랙 (판정 1 · AV2 보조 스텝 > 0)",
                                    "n": int(sf.sum()), "av2_same_pct": float((T["ag_av2"][sf] == 1).mean() * 100) if sf.any() else None,
                                    "h10_same_pct": float((T["ag_10"][sf] == 1).mean() * 100) if sf.any() else None}
    lane["all"] = {"opp_av2_pct": float((T["ag_av2"][lm] == 0).mean() * 100), "opp_10_pct": float((T["ag_10"][lm] == 0).mean() * 100)}
    lf_ = lm & (T["ag_2"] >= 0)
    lane["all_full"] = {"n": int(lf_.sum()), "opp_av2_pct": float((T["ag_av2"][lf_] == 0).mean() * 100),
                        "opp_10_pct": float((T["ag_10"][lf_] == 0).mean() * 100),
                        "opp_2_pct": float((T["ag_2"][lf_] == 0).mean() * 100)}
    flipinfo["all"] = {"n_verif10": int((flip >= 0).sum()), "flip10_pct": float((flip == 1).sum() / max((flip >= 0).sum(), 1) * 100),
                       "n_verif2": int((f2 >= 0).sum()), "flip2_pct": float((f2 == 1).sum() / max((f2 >= 0).sum(), 1) * 100),
                       "fill_disagree_full_pct": float((T["fill_dis"][full] > 0).mean() * 100),
                       "fill_disagree_full_n": int((T["fill_dis"][full] > 0).sum()),
                       "verif10_not2_full": int((full & (flip >= 0) & (f2 < 0)).sum()),
                       "verif10_not2_full_flipped10": int((full & (flip == 1) & (f2 < 0)).sum())}
    summ["agents_flip"] = flipinfo
    summ["agents_lane_check"] = lane
    VB = (("정지\n<0.5", stat), ("0.5–2", (T["vmax"] >= 0.5) & (T["vmax"] < 2)),
          ("2–5", (T["vmax"] >= 2) & (T["vmax"] < 5)), ("≥5", T["vmax"] >= 5))
    lane["by_vmax_full"] = {}
    for name, m in VB:
        mf = m & lm & (T["ag_2"] >= 0)
        lane["by_vmax_full"][name.replace("\n", " ")] = {
            "n": int(mf.sum()), "opp_av2_pct": float((T["ag_av2"][mf] == 0).mean() * 100),
            "opp_10_pct": float((T["ag_10"][mf] == 0).mean() * 100), "opp_2_pct": float((T["ag_2"][mf] == 0).mean() * 100)}
    fig, axs = plt.subplots(1, 4, figsize=(22.0, 5.4), gridspec_kw={"width_ratios": [1.1, 1.1, 0.8, 1.0]})
    x = np.arange(len(PBL))
    w = 0.38
    ax = axs[0]
    a1 = [flipinfo["by_path"][l]["flip10_pct"] or 0 for l in PBL]
    a2 = [flipinfo["by_path"][l]["flip2_pct"] or 0 for l in PBL]
    b1 = ax.bar(x - w / 2, a1, w, color=COL10, label="10 Hz 판정 (위치차분 스텝 ≥ 3)")
    b2 = ax.bar(x + w / 2, a2, w, color=COL2, label="2 Hz 판정 (위치차분 표본 ≥ 3, 50스텝 관측)")
    bar_labels(ax, list(b1) + list(b2), "{:.1f}", dy=0.6, fs=8)
    for k, l in enumerate(PBL):
        fb = flipinfo["by_path"][l]
        ax.text(k, 1.01, f"n {fb['n_verif10']:,}\n/ {fb['n_verif2']:,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.4, color=C.MUTED)
    ax.axhline(50, color=C.AXIS, lw=0.9, ls="--")
    ax.set_xticks(x, [f"{l} m" for l in PBL])
    ax.set_xlabel("과거 4.9 s 이동거리 (관측 스텝 위치 합)")
    ax.set_ylabel("'AV2 heading 이 뒤집혔다'고 판정한 비율 [%]")
    ax.set_ylim(0, 60)
    ax.set_title("(a) 뒤집힘 교정이 걸린 비율", loc="left", pad=24)
    ax.legend(fontsize=8.2, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axs[1]
    w3 = 0.27
    o_av2 = [lane["by_path"][l]["opp_av2_pct_full"] or 0 for l in PBL]
    o_10 = [lane["by_path"][l]["opp_10_pct_full"] or 0 for l in PBL]
    o_2 = [lane["by_path"][l]["opp_2_pct_full"] or 0 for l in PBL]
    bb = [ax.bar(x - w3, o_av2, w3, color=COLAV2, label="AV2 heading 원본"),
          ax.bar(x, o_10, w3, color=COL10, label="10 Hz h (캐시)"),
          ax.bar(x + w3, o_2, w3, color=COL2, label="2 Hz h (즉석)")]
    bar_labels(ax, [b for g in bb for b in g], "{:.1f}", dy=0.3, fs=7.6)
    for k, l in enumerate(PBL):
        ax.text(k, 1.01, f"n {lane['by_path'][l]['n_full']:,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color=C.MUTED)
    ax.set_xticks(x, [f"{l} m" for l in PBL])
    ax.set_xlabel("과거 4.9 s 이동거리")
    ax.set_ylabel("t=0 방향이 차로 방향과 반대인 비율 [%]")
    ax.set_title(f"(b) 차로 방향과 대조 — 차로 안({LANE_IN_M:g} m)·교차로 밖·50스텝 관측", loc="left", pad=16)
    ax.legend(fontsize=8.2, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axs[2]
    fdp = [flipinfo["by_path"][l]["fill_disagree_pct_full"] or 0 for l in PBL]
    b1 = ax.bar(x, fdp, 0.6, color=C.C_VIOLET)
    bar_labels(ax, b1, "{:.1f}%", dy=0.3)
    for k, l in enumerate(PBL):
        ax.text(k, 1.01, f"n {flipinfo['by_path'][l]['n_full']:,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color=C.MUTED)
    ax.set_xticks(x, [f"{l} m" for l in PBL])
    ax.set_xlabel("과거 4.9 s 이동거리")
    ax.set_ylabel("차량 비율 [%]")
    ax.set_title("(c) 두 판이 모두 AV2 로 메운 표본에서\n     10 Hz 와 2 Hz 방향이 90° 넘게 갈린 차량", loc="left", pad=16)
    ax.grid(axis="x", visible=False)
    ax = axs[3]
    xv = np.arange(len(VB))
    keys = [nm.replace("\n", " ") for nm, _ in VB]
    bb = [ax.bar(xv - w3, [lane["by_vmax_full"][k]["opp_av2_pct"] for k in keys], w3, color=COLAV2, label="AV2 heading 원본"),
          ax.bar(xv, [lane["by_vmax_full"][k]["opp_10_pct"] for k in keys], w3, color=COL10, label="10 Hz h"),
          ax.bar(xv + w3, [lane["by_vmax_full"][k]["opp_2_pct"] for k in keys], w3, color=COL2, label="2 Hz h")]
    bar_labels(ax, [b for g in bb for b in g], "{:.1f}", dy=0.3, fs=7.6)
    for k, key in enumerate(keys):
        ax.text(k, 1.01, f"n {lane['by_vmax_full'][key]['n']:,}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color=C.MUTED)
    ax.set_xticks(xv, [nm for nm, _ in VB])
    ax.set_xlabel("과거 4.9 s 최대 AV2 속력 [m/s]")
    ax.set_ylabel("차로 방향과 반대인 비율 [%]")
    ax.set_title("(d) (b) 와 같은 대조 — 최대 속력별\n     (정지 차는 후진이 아니라고 추정 → 차로 반대는 오류로 봄)",
                 loc="left", pad=16)
    ax.set_ylim(0, max(lane["by_vmax_full"][k]["opp_10_pct"] for k in keys) * 1.25)
    ax.legend(fontsize=8.2, loc="upper right")
    ax.grid(axis="x", visible=False)
    sfl = lane["stationary_flipped10"]
    fig.suptitle("주변 차량의 180° 뒤집힘 교정 — 거의 안 움직인 차량에서는 위치 잡음이 판정을 정한다", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, f"(b) 참고값은 차로 방향이다(주차·정차 차량이 차로와 반대로 서 있으면 틀린다). 차로 안 정지 차량 중 10 Hz 가 뒤집은 {sfl['n']:,}대"
             f"(뒤집힌 정지 차량 {int((stat & (flip == 1) & (T['n_src2'] > 0)).sum()):,}대의 "
             f"{sfl['n'] / max(int((stat & (flip == 1) & (T['n_src2'] > 0)).sum()), 1) * 100:.0f}% 만 덮는다): "
             f"AV2 원본은 {sfl['av2_same_pct']:.1f}% 가 차로 방향 쪽, 교정 뒤 10 Hz h 는 {sfl['h10_same_pct']:.1f}%.  "
             f"차로 안 전체 {lane['n_in_lane']:,}대: 반대 방향 비율 AV2 {lane['all']['opp_av2_pct']:.2f}% / 10 Hz {lane['all']['opp_10_pct']:.2f}%.",
             fontsize=8.6, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    C.savefig(fig, OUT / "dist_7_agents_flip_lane.png")

    # ------------------------------------------------ focal 도 같은 문제가 있나
    h10, s10 = S.h10, S.src10
    opp = np.abs(h10) > np.pi / 2
    has2 = (s10 == 2).any(1)
    all_opp = np.array([bool(opp[i][s10[i] == 2].all()) if has2[i] else False for i in range(n)])
    rev = np.array([bool((opp[i][s10[i] == 1]).mean() > 0.5) if (s10[i] == 1).any() else False for i in range(n)])
    o2 = np.abs(S.h2) > np.pi / 2
    all_opp2 = np.array([bool(o2[i][S.src2[i] == 2].all()) if (S.src2[i] == 2).any() else False for i in range(n)])
    summ["focal_frame_opposite"] = {
        "def": "뒤집힘 판정과 무관한 '입력 h 가 frame 0° 와 반대' 집계 (logic_tree.focal_* 의 뒤집힘 판정 개수와 정의가 다르다). "
               "all_fill_opposite = AV2 보조 스텝이 모두 |h| > 90° 인 시나리오 (AV2 원본 자체가 과거에 90° 넘게 달랐던 경우 포함). "
               "h49_opposite = t=49 입력 h 가 |h| > 90° 인 시나리오 (출처 무관)",
        "scen_with_av2_fill": int(has2.sum()), "all_fill_opposite_10hz": int(all_opp.sum()),
        "all_fill_opposite_10hz_pct": float(all_opp.mean() * 100),
        "of_which_posdiff_mostly_opposite": int((all_opp & rev).sum()),
        "all_fill_opposite_2hz": int(all_opp2.sum()),
        "steps_opposite_pct_10hz": float(opp.mean() * 100),
        "h49_opposite_10hz": int(opp[:, OBS - 1].sum()), "h9_opposite_2hz": int(o2[:, -1].sum()),
    }

    f = S.focal_feats()
    cls = fig_by_state(S, f, summ)
    np.save(data / "state.npy", np.array(cls, dtype="U8"))
    fig_ramp_by_speed(S, summ)
    fig_logic_tree(S, summ)
    fig_cart(S, summ, data)
    (data / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    return summ


# ============================================================================ 4b. 상태별 · 램프 속력별 · 결정 트리
STATE_DEF = ("과거 4.9 s 상태 (viz_v4_common 임계값을 과거 창에 적용, 앞에서 걸리면 뒤는 안 봄): "
             "정지 = AV2 속력(median5 → SG15) 최대 < 1 m/s · 좌/우회전 = AV2 heading(튐 가드) 순변화 ±30° 초과 · "
             "좌/우 차선변경 = |Δh| < 15° · 이동 ≥ 10 m · 모든 스텝 5 m 안 같은 방향 차로(교차로 차로 아님) · 차로 대비 누적 횡이동 ±2.5 m 초과 · "
             "급감속 = 평활 가속도 최소 < −3 m/s² · 급가속 = 최대 > 2.5 m/s² · 정속 = |v끝 − v시작| < 1.5 m/s 이고 v시작 > 2 m/s")


def past_states(S, f):
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    vm = median_filter(S.vf.astype(np.float64), size=(1, C.MED_K), mode="nearest")
    vs = savgol_filter(vm, C.SG_WIN, C.SG_POLY, axis=1, mode="interp")
    acc = savgol_filter(vm, C.SG_WIN, C.SG_POLY, deriv=1, delta=C.DT, axis=1, mode="interp")
    n = S.n
    cls = np.full(n, "기타", dtype=object)
    lc_ok = ((np.abs(f["dh"]) < C.LC_MAX_DEG) & (f["path"] >= C.LC_MIN_MOVE) & (f["lane_valid"] == 1)
             & (f["lane_inter"] == 0))
    conds = [("정지", vs.max(1) < C.STOP_VMAX),
             ("좌회전", f["dh"] > C.TURN_DEG), ("우회전", f["dh"] < -C.TURN_DEG),
             ("좌차선변경", lc_ok & (f["lat_lane"] > C.LC_LAT_M)), ("우차선변경", lc_ok & (f["lat_lane"] < -C.LC_LAT_M)),
             ("급감속", acc.min(1) < C.DEC_A), ("급가속", acc.max(1) > C.ACC_A),
             ("정속", (np.abs(vs[:, -1] - vs[:, 0]) < C.CONST_DV) & (vs[:, 0] > C.CONST_V0))]
    done = np.zeros(n, bool)
    for name, m in conds:
        sel = m & ~done
        cls[sel] = name
        done |= sel
    return cls


def gbar(ax, cats, v10, v2, ns, m10=None, m2=None, fmt="{:.1f}", w=0.38, log=False, lab=("10 Hz", "2 Hz"),
         mlab="p95"):
    """상태별 두 판 막대 (+ 선택 표식). n < 50 칸은 흐리게."""
    x = np.arange(len(cats))
    for off, v, m, col, lb in ((-w / 2, v10, m10, COL10, lab[0]), (w / 2, v2, m2, COL2, lab[1])):
        if v is None:
            continue
        bars = ax.bar(x + off, np.nan_to_num(v), w, color=col, label=lb)
        for k, b in enumerate(bars):
            if C.faded(ns[k]):
                b.set_alpha(0.3)
        if m is not None:
            ax.scatter(x + off, m, marker="_", s=180, color=C.INK, zorder=4, label=f"{mlab}" if off < 0 else None)
    if log:
        ax.set_yscale("log")
    labels = [f"{c}\nn {n:,}" for c, n in zip(cats, ns)]
    ax.set_xticks(x, labels, fontsize=7.6)
    for t, n in zip(ax.get_xticklabels(), ns):
        if C.faded(n):
            t.set_color(C.MUTED)
    ax.grid(axis="x", visible=False)


def fig_by_state(S, f, summ):
    plt = C.setup_mpl()
    cls = past_states(S, f)
    cats = list(C.CLASSES)
    ns = [int((cls == c).sum()) for c in cats]
    j10 = np.array([len(jumps(h)) for h in S.h10]) > 0
    j2 = np.array([len(jumps(h)) for h in S.h2]) > 0
    mv10 = (S.src10[:, :OBS - 2] == 1) & (S.src10[:, 1:OBS - 1] == 1)
    mv2 = (S.src2[:, :-2] == 1) & (S.src2[:, 1:-1] == 1)
    r10 = np.degrees(np.abs(wrap(np.diff(S.h10[:, :OBS - 1], axis=1)))) / 0.1
    r2 = np.degrees(np.abs(wrap(np.diff(S.h2[:, :-1], axis=1)))) / 0.5
    sl10 = np.degrees(np.abs(wrap(S.head_n[:, :OBS - 1].astype(np.float64) - S.h10[:, :OBS - 1])))
    ps10 = S.src10[:, :OBS - 1] == 1
    sl2 = np.degrees(np.abs(wrap(S.head_n[:, IDX2[:-1]].astype(np.float64) - S.h2[:, :-1])))
    ps2 = S.src2[:, :-1] == 1
    a10 = np.abs(S.x10[:, 6:OBS - 1, 0]) * KMH / 0.1
    a2 = np.abs(S.x2[:, 1:-1, 0]) * KMH / 0.5
    okr = S.vf[:, 0] >= 1.0
    ramp10 = np.where(okr, S.v10[:, 0] / np.maximum(S.vf[:, 0], 1e-6), np.nan)
    ramp2 = np.where(okr, S.v2[:, 0] / np.maximum(S.vf[:, 4:10].mean(1), 1e-6), np.nan)
    ff = S.fflip.astype(int)
    rows = {}
    for c in cats:
        m = cls == c
        g = lambda arr, mask: arr[m][mask[m]] if mask is not None else arr[m].ravel()
        d = {"n": int(m.sum())}
        if m.any():
            d.update({
                "av2_10_pct": float((S.src10[m] == 2).mean() * 100), "av2_2_pct": float((S.src2[m] == 2).mean() * 100),
                "jump10_pct": float(j10[m].mean() * 100), "jump2_pct": float(j2[m].mean() * 100),
                "rate10": pctl(r10[m][mv10[m]], (50, 95)), "rate2": pctl(r2[m][mv2[m]], (50, 95)),
                "slip10": pctl(sl10[m][ps10[m]], (50, 95)), "slip2": pctl(sl2[m][ps2[m]], (50, 95)),
                "rec": {k: float(S.rec[m, j].mean()) for j, k in enumerate(["10hz_av2", "10hz_h", "2hz_av2", "2hz_h"])},
                "ramp10_med": float(np.nanmedian(ramp10[m])) if np.isfinite(ramp10[m]).any() else None,
                "ramp2_med": float(np.nanmedian(ramp2[m])) if np.isfinite(ramp2[m]).any() else None,
                "n_ramp": int(np.isfinite(ramp10[m]).sum()),
                "acc10": pctl(a10[m], (50, 95)), "acc2": pctl(a2[m], (50, 95)),
                "flip10_pct": float((ff[m, 0] == 1).sum() / max((ff[m, 0] >= 0).sum(), 1) * 100),
                "flip2_pct": float((ff[m, 2] == 1).sum() / max((ff[m, 2] >= 0).sum(), 1) * 100),
                "n_flip10": int((ff[m, 0] >= 0).sum()), "n_flip2": int((ff[m, 2] >= 0).sum()),
            })
        rows[c] = d
    summ["by_state"] = {"def": STATE_DEF, "rows": rows}
    G = lambda key, sub=None: np.array([(rows[c].get(key, {}) or {}).get(sub, np.nan) if sub else
                                         (rows[c].get(key, np.nan) if rows[c].get(key) is not None else np.nan)
                                         for c in cats], float)
    fig, axs = plt.subplots(2, 4, figsize=(24.0, 10.5))
    ax = axs[0, 0]
    gbar(ax, cats, G("av2_10_pct"), G("av2_2_pct"), ns)
    ax.set_ylabel("%")
    ax.set_title("(a) focal h 가 AV2 보조인 비율\n(10 Hz 스텝 / 2 Hz 표본)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.4)
    ax = axs[0, 1]
    gbar(ax, cats, G("jump10_pct"), G("jump2_pct"), ns)
    ax.set_ylabel("시나리오 %")
    ax.set_title("(b) ±π 점프가 있는 시나리오", loc="left", fontsize=10.5)
    ax = axs[0, 2]
    gbar(ax, cats, G("rate10", "p50"), G("rate2", "p50"), ns, G("rate10", "p95"), G("rate2", "p95"), log=True)
    ax.set_ylabel("|Δh| [°/s]")
    ax.set_title("(c) 스텝 간 방향 변화 (위치차분 구간)\n막대 = 중앙, ─ = p95", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.0)
    ax = axs[0, 3]
    gbar(ax, cats, G("slip10", "p50"), G("slip2", "p50"), ns, G("slip10", "p95"), G("slip2", "p95"), log=True)
    ax.set_ylabel("|AV2 − h| [°]")
    ax.set_title("(d) 슬립각+잡음 (위치차분 스텝·표본)\n막대 = 중앙, ─ = p95", loc="left", fontsize=10.5)
    ax = axs[1, 0]
    x = np.arange(len(cats))
    ww = 0.2
    for k, (key, col, hatch, lb) in enumerate((("10hz_av2", COL10, "//", "AV2 heading · 10 Hz"),
                                                ("10hz_h", COL10, None, "h · 10 Hz"),
                                                ("2hz_av2", COL2, "//", "AV2 heading · 2 Hz"),
                                                ("2hz_h", COL2, None, "h · 2 Hz"))):
        v = np.array([rows[c]["rec"][key] if rows[c]["n"] else np.nan for c in cats])
        bars = ax.bar(x + (k - 1.5) * ww, v, ww, color="white" if hatch else col, edgecolor=col, hatch=hatch, lw=1.0,
                      label=lb)
        for kk, b in enumerate(bars):
            if C.faded(ns[kk]):
                b.set_alpha(0.3)
    ax.set_xticks(x, [f"{c}\nn {n:,}" for c, n in zip(cats, ns)], fontsize=7.6)
    ax.set_ylabel("끝점 오차 평균 [m]")
    ax.set_title("(e) 방향만 바꿔 과거를 다시 굴린 끝점 오차", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.8, ncol=2)
    ax.grid(axis="x", visible=False)
    ax = axs[1, 1]
    nr = [rows[c].get("n_ramp", 0) for c in cats]
    gbar(ax, cats, G("ramp10_med"), G("ramp2_med"), nr, lab=("10 Hz 첫 스텝 ÷ AV2 속력", "2 Hz 첫 표본 ÷ AV2 평균(4~9)"))
    ax.axhline(1.0, color=C.INK2, lw=0.9, ls="--")
    ax.set_ylabel("중앙 비율")
    ax.set_ylim(0, 1.5)
    ax.set_title("(f) 창 시작 속력 비율 (AV2 시작 속력 ≥ 1 m/s)\n아래 n = 이 조건의 수", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.0, loc="upper center", ncol=2)
    ax = axs[1, 2]
    gbar(ax, cats, G("acc10", "p50"), G("acc2", "p50"), ns, G("acc10", "p95"), G("acc2", "p95"), log=True)
    ax.set_ylabel("|a| [m/s²]")
    ax.set_title("(g) a 채널 → 초당 가속도 (램프 뺀 구간) · ─ = p95\n10 Hz 예측 시작 기준 −4.3~−0.1 s (창 시작 0.6~4.8 s)"
                 " · 2 Hz −4.0~−0.5 s", loc="left", fontsize=10.5)
    ax = axs[1, 3]
    nf = [rows[c].get("n_flip10", 0) for c in cats]
    gbar(ax, cats, G("flip10_pct"), G("flip2_pct"), nf)
    ax.set_ylabel("판정한 시나리오 중 %")
    ax.set_title("(h) focal AV2 보조 구간을 180° 뒤집은 비율\n아래 n = 10 Hz 에서 판정한 수", loc="left", fontsize=10.5)
    fig.suptitle(f"focal 입력 전처리 — 과거 4.9 s 상태별 (val {S.n:,}) · 파랑 = 10 Hz, 빨강 = 2 Hz · n < {C.MIN_N} 은 흐리게",
                 fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.004, STATE_DEF, fontsize=8.2, color=C.MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    C.savefig(fig, OUT / "dist_8_by_state.png")
    return cls


def fig_ramp_by_speed(S, summ):
    plt = C.setup_mpl()
    ref = S.v10[:, RAMP_REF].mean(1)
    bins = [(3, 6), (6, 10), (10, 15), (15, np.inf)]
    NT = 30
    tt = np.arange(NT) * 0.1
    fig, axs = plt.subplots(1, 4, figsize=(22.0, 5.2), sharey=True)
    out = {}
    for ax, (lo, hi) in zip(axs, bins):
        m = (ref > lo) & (ref <= hi)
        n = int(m.sum())
        r10 = S.v10[m, :NT] / ref[m, None]
        rf = S.vf[m, :NT].astype(np.float64) / ref[m, None]
        q10 = np.percentile(r10, [25, 50, 75], axis=0)
        qf = np.percentile(rf, [25, 50, 75], axis=0)
        a = 1.0 if n >= C.MIN_N else 0.35
        ax.fill_between(tt, q10[0], q10[2], color=COL10, alpha=0.18 * a, lw=0)
        ax.plot(tt, q10[1], color=COL10, marker="o", ms=3, alpha=a, label="10 Hz 위치차분 속력 (0.1 s)")
        ax.plot(tt, qf[1], color=COLAV2, ls=(0, (4, 2)), alpha=a, label="AV2 속도 필드")
        seg = {}
        for k in (0, 1):
            r2 = S.v2[m, k] / ref[m]
            q = np.percentile(r2, [25, 50, 75])
            t0 = IDX2[k] * 0.1
            ax.plot([t0, t0 + 0.5], [q[1]] * 2, color=COL2, lw=3.0, alpha=a, label="2 Hz 표본 (0.5 s 구간)" if k == 0 else None)
            ax.scatter([t0], [q[1]], s=60, color=COL2, zorder=5, alpha=a)
            ax.fill_between([t0, t0 + 0.5], [q[0]] * 2, [q[2]] * 2, color=COL2, alpha=0.15 * a, lw=0)
            seg[k] = float(q[1])
        ax.axvspan(-0.05, 0.45, facecolor="none", edgecolor=C.MUTED, hatch="///", lw=0, zorder=0)
        ax.axhline(1.0, color=C.AXIS, lw=0.9)
        lab = f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"> {lo:g}"
        ax.set_title(f"기준 속력 {lab} m/s  (n = {n:,})", loc="left", fontsize=11)
        ax.text(0.98, 0.04, f"10 Hz 첫 스텝 {q10[1, 0]:.2f}배\n2 Hz 첫 표본 {seg[0]:.3f}배", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=C.INK2, bbox=dict(facecolor=C.SURF, edgecolor=C.GRID))
        ax.set_xlabel("창 시작부터 [s]")
        top_pred_axis(ax, "예측 시작 기준 [s]")
        out[lab] = {"n": n, "ratio10_t0_4": [float(v) for v in q10[1, :5]], "ratio10_overshoot_max": float(q10[1].max()),
                    "ratio_field_t0": float(qf[1, 0]), "2hz_seg0": seg[0], "2hz_seg1": seg[1]}
    axs[0].set_ylabel("속력 ÷ 기준 속력 (창 시작 1.0~1.6 s 위치차분 평균)")
    axs[0].set_ylim(0.2, 1.3)
    axs[0].legend(fontsize=8.4, loc="lower center")
    fig.suptitle("창 시작 램프 — 기준 속력 구간별 (중앙 · IQR). 빗금 = 인덱스 0~4 (예측 시작 기준 −4.9~−4.5 s), 2 Hz 는 인덱스 4 에서 시작",
                 fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    C.savefig(fig, OUT / "dist_3b_ramp_by_speed.png")
    summ["ramp_by_speed"] = out


def fig_logic_tree(S, summ):
    """스텝별 h 출처가 어떻게 정해지는지 — 규칙 흐름 도식에 실제 개수를 적는다 (10 Hz / 2 Hz, 주변 차량 / focal)."""
    plt = C.setup_mpl()
    T = S.T
    ag = S.ag
    kept = ag["atype"] > 0
    hs = ag["h_src"][kept]
    tot = int(kept.sum()) * OBS
    c = {k: int((hs == k).sum()) for k in range(4)}
    dup = int((hs[:, OBS - 1] == 1).sum())
    flip, ns1, ns2 = T["flip"].astype(int), T["n_src1"].astype(int), T["n_src2"].astype(int)
    src49 = ag["h_src"][T["scen"].astype(int), T["slot"].astype(int), OBS - 1]
    lm = T["lane_m"] == 1
    f2 = T["flip2"].astype(int)
    full = T["full_obs"] == 1
    n21, n2a = T["n2_src1"].astype(int), T["n2_av2"].astype(int)
    pct = lambda v, base: f"{v:,} ({v / base * 100:.1f}%)"
    info = {}

    def lane_txt(mask, key):
        m = mask & lm
        if m.sum() < 1:
            return ""
        o = float((T[key][m] == 0).mean() * 100)
        oa = float((T["ag_av2"][m] == 0).mean() * 100)
        return f"\n── 차로 방향과 반대 ──\n이 값 {o:.1f}% · AV2 원본 {oa:.1f}%\n(t=0 이 이 갈래 · 차로 안\n트랙 {int(m.sum()):,})"

    # --- 주변 차량 10 Hz
    a1, a0, am = int(ns2[flip == 1].sum()), int(ns2[flip == 0].sum()), int(ns2[flip == -1].sum())
    am_none, am_few = int(ns2[(flip == -1) & (ns1 == 0)].sum()), int(ns2[(flip == -1) & (ns1 > 0)].sum())
    assert a1 + a0 + am == c[2], (a1, a0, am, c[2])
    info["agents_10hz"] = {"total_steps": tot, "src": c, "posdiff_dup_last": dup, "av2_flip1": a1, "av2_flip0": a0,
                           "av2_undecided": am, "av2_undecided_no_posdiff": am_none, "av2_undecided_1_2_posdiff": am_few,
                           "tracks_flip1": int(((flip == 1) & (ns2 > 0)).sum()),
                           "tracks_flip0": int(((flip == 0) & (ns2 > 0)).sum()),
                           "tracks_undecided": int(((flip == -1) & (ns2 > 0)).sum())}
    for leaf, key in ((1, "flip1"), (0, "flip0"), (-1, "undecided")):
        m = lm & (src49 == 2) & (flip == leaf)
        info["agents_10hz"][f"lane_{key}"] = {"n": int(m.sum()), "opp_10_pct": float((T["ag_10"][m] == 0).mean() * 100),
                                              "opp_av2_pct": float((T["ag_av2"][m] == 0).mean() * 100)}
    B = SRC_BAND
    L10 = lambda leaf: lane_txt((src49 == 2) & (flip == leaf), "ag_10")
    tree_a10 = {"text": f"주변 차량 슬롯 스텝 (K≤32 × 50)\nn = {tot:,}", "children": [
        ("관측 안 됨", {"text": f"미관측 → 0\n{pct(c[0], tot)}", "color": B[0]}),
        ("관측됨", {"text": f"관측 스텝\n{pct(tot - c[0], tot)}", "children": [
            ("다음 관측과의 구간 속력 ≥ 1 m/s", {"text": f"위치차분 h\n{pct(c[1], tot)}\n(그중 마지막 스텝 복제 {dup:,})",
                                              "color": B[1]}),
            ("< 1 m/s", {"text": f"AV2 heading 보조\n(튐 가드 30°/0.1 s)\n{pct(c[2], tot)}", "color": B[2], "children": [
                ("트랙 위치차분 ≥ 3", {"text": f"뒤집힘 판정\n{pct(a1 + a0, tot)}", "color": B[2], "children": [
                    ("중앙 |AV2 − h| > 90°", {"text": f"180° 뒤집어 씀\n{pct(a1, tot)}{L10(1)}", "color": "#f6d5d5"}),
                    ("≤ 90°", {"text": f"그대로\n{pct(a0, tot)}{L10(0)}", "color": B[2]})]}),
                ("< 3", {"text": f"판정 안 함\n(검증 불가)\n{pct(am, tot)}\n위치차분 0개 {am_none:,}\n1~2개 {am_few:,}{L10(-1)}",
                         "color": "#eeeeee"})]}),
        ]}),
    ]}
    # --- 주변 차량 2 Hz
    nt = len(T["scen"])
    nfull = int(full.sum())
    S2 = nfull * len(IDX2)
    p1, p2 = int(n21[full].sum()), int(n2a[full].sum())
    assert p1 + p2 == S2, (p1, p2, S2)
    b1, b0, bm = int(n2a[full & (f2 == 1)].sum()), int(n2a[full & (f2 == 0)].sum()), int(n2a[full & (f2 == -1)].sum())
    bm_none = int(n2a[full & (f2 == -1) & (n21 == 0)].sum())
    info["agents_2hz"] = {"tracks": nt, "tracks_full": nfull, "samples": S2, "posdiff": p1, "av2": p2,
                          "av2_flip1": b1, "av2_flip0": b0, "av2_undecided": bm, "av2_undecided_no_posdiff": bm_none}
    s2l = T["src2_last"].astype(int)
    for leaf, key in ((1, "flip1"), (0, "flip0"), (-1, "undecided")):
        m = lm & full & (s2l == 2) & (f2 == leaf)
        info["agents_2hz"][f"lane_{key}"] = {"n": int(m.sum()), "opp_2_pct": float((T["ag_2"][m] == 0).mean() * 100),
                                             "opp_av2_pct": float((T["ag_av2"][m] == 0).mean() * 100)}
    L2 = lambda leaf: lane_txt(full & (s2l == 2) & (f2 == leaf), "ag_2")
    tree_a2 = {"text": f"주변 차량 트랙\nn = {nt:,}", "children": [
        ("50스텝 중 빠진 관측 있음", {"text": f"2 Hz 계산 안 함\n트랙 {pct(nt - nfull, nt)}", "color": B[0]}),
        ("50스텝 모두 관측", {"text": f"평활(SG 5점·2차, 인덱스 4~49)\n→ 0.5 s 표본 10개\n트랙 {nfull:,} · 표본 {S2:,}", "children": [
            ("0.5 s 구간 속력 ≥ 1 m/s", {"text": f"위치차분 h\n표본 {pct(p1, S2)}", "color": B[1]}),
            ("< 1 m/s", {"text": f"AV2 보조\n(10 Hz 에서 가드 → 뽑기)\n표본 {pct(p2, S2)}", "color": B[2], "children": [
                ("위치차분 표본 ≥ 3", {"text": f"뒤집힘 판정\n{pct(b1 + b0, S2)}", "color": B[2], "children": [
                    ("> 90°", {"text": f"180° 뒤집어 씀\n{pct(b1, S2)}{L2(1)}", "color": "#f6d5d5"}),
                    ("≤ 90°", {"text": f"그대로\n{pct(b0, S2)}{L2(0)}", "color": B[2]})]}),
                ("< 3", {"text": f"판정 안 함\n{pct(bm, S2)}\n위치차분 0개 {bm_none:,}{L2(-1)}", "color": "#eeeeee"})]}),
        ]}),
    ]}
    # --- focal
    ff = S.fflip.astype(int)
    n = S.n
    F10 = {k: int((S.src10 == k).sum()) for k in (1, 2, 3)}
    ns1f = (S.src10 == 1).sum(1)
    av2f = (S.src10 == 2).sum(1)
    fa1, fa0, fam = int(av2f[ff[:, 0] == 1].sum()), int(av2f[ff[:, 0] == 0].sum()), int(av2f[ff[:, 0] == -1].sum())
    assert fa1 + fa0 + fam == F10[2]
    opp49 = (np.abs(S.h10[:, OBS - 1]) > np.pi / 2) & (S.src10[:, OBS - 1] == 2)
    info["focal_10hz"] = {"steps": n * OBS, "src": F10, "av2_flip1": fa1, "av2_flip0": fa0, "av2_undecided": fam,
                          "scen_flip1": int(((ff[:, 0] == 1) & (av2f > 0)).sum()),
                          "scen_h49_opposite_av2fill": int(opp49.sum()),
                          "scen_h49_opposite_av2fill_flip1": int((opp49 & (ff[:, 0] == 1)).sum())}
    tot_f = n * OBS
    tree_f10 = {"text": f"focal 스텝 (시나리오 × 50)\nn = {tot_f:,}", "children": [
        ("구간 속력 ≥ 1 m/s", {"text": f"위치차분 h\n{pct(F10[1], tot_f)}", "color": B[1]}),
        ("< 1 m/s", {"text": f"AV2 보조\n{pct(F10[2], tot_f)}", "color": B[2], "children": [
            ("위치차분 ≥ 3", {"text": f"뒤집힘 판정\n{pct(fa1 + fa0, tot_f)}", "color": B[2], "children": [
                ("> 90°", {"text": f"180° 뒤집어 씀\n{pct(fa1, tot_f)}\n시나리오 {info['focal_10hz']['scen_flip1']:,}\n"
                                   f"(t=0 입력 h 가 뒤를 향함 {info['focal_10hz']['scen_h49_opposite_av2fill_flip1']:,})",
                           "color": "#f6d5d5"}),
                ("≤ 90°", {"text": f"그대로\n{pct(fa0, tot_f)}", "color": B[2]})]}),
            ("< 3", {"text": f"판정 안 함\n{pct(fam, tot_f)}", "color": "#eeeeee"})]}),
    ]}
    S2f = n * len(IDX2)
    G1, G2 = int((S.src2 == 1).sum()), int((S.src2 == 2).sum())
    av2f2 = (S.src2 == 2).sum(1)
    ga1, ga0, gam = int(av2f2[ff[:, 2] == 1].sum()), int(av2f2[ff[:, 2] == 0].sum()), int(av2f2[ff[:, 2] == -1].sum())
    assert ga1 + ga0 + gam == G2
    info["focal_2hz"] = {"samples": S2f, "posdiff": G1, "av2": G2, "av2_flip1": ga1, "av2_flip0": ga0, "av2_undecided": gam,
                         "scen_flip1": int(((ff[:, 2] == 1) & (av2f2 > 0)).sum())}
    tree_f2 = {"text": f"focal 2 Hz 표본 (평활 → 0.5 s × 10)\nn = {S2f:,}", "children": [
        ("0.5 s 구간 속력 ≥ 1 m/s", {"text": f"위치차분 h\n{pct(G1, S2f)}", "color": B[1]}),
        ("< 1 m/s", {"text": f"AV2 보조\n{pct(G2, S2f)}", "color": B[2], "children": [
            ("위치차분 표본 ≥ 3", {"text": f"뒤집힘 판정\n{pct(ga1 + ga0, S2f)}", "color": B[2], "children": [
                ("> 90°", {"text": f"180° 뒤집어 씀\n{pct(ga1, S2f)}\n시나리오 {info['focal_2hz']['scen_flip1']:,}",
                           "color": "#f6d5d5"}),
                ("≤ 90°", {"text": f"그대로\n{pct(ga0, S2f)}", "color": B[2]})]}),
            ("< 3", {"text": f"판정 안 함\n{pct(gam, S2f)}", "color": "#eeeeee"})]}),
    ]}
    same = full & True
    base10, base2 = int(same.sum()) * OBS, int(same.sum()) * len(IDX2)
    info["agents_same_tracks_full"] = {
        "def": "관측 50스텝이 모두 있는 트랙끼리 10 Hz 와 2 Hz 비교 (슬롯 스텝 / 표본 기준)",
        "n_tracks": int(same.sum()),
        "judged10_pct": float(T["n_src2"][same & (flip >= 0)].sum() / base10 * 100),
        "flip10_pct": float(T["n_src2"][same & (flip == 1)].sum() / base10 * 100),
        "judged2_pct": float(n2a[same & (f2 >= 0)].sum() / base2 * 100),
        "flip2_pct": float(n2a[same & (f2 == 1)].sum() / base2 * 100)}
    stat = T["vmax"] < 0.5
    sf = stat & (flip == 1) & (ns2 > 0)
    info["stationary_flipped10"] = {
        "def": "정지(AV2 최대 < 0.5 m/s) 이면서 10 Hz 가 실제로 뒤집은 트랙(판정 1 · AV2 보조 스텝 > 0) — 차로 방향 대조가 덮는 비율",
        "n": int(sf.sum()), "in_lane_n": int((sf & lm).sum()), "in_lane_pct": float((sf & lm).mean() / max(sf.mean(), 1e-12) * 100)}
    hh = ag["h"]
    jfull = full & (T["jump10"] > 0)
    near = np.array([bool((np.abs(hh[int(a_), int(b_)][ag["h_src"][int(a_), int(b_)] > 0]) >= np.radians(150)).all())
                     for a_, b_ in zip(T["scen"][jfull], T["slot"][jfull])])
    info["jump10_full_near180"] = {
        "def": "관측 50스텝 전부이고 10 Hz ±π 점프가 있는 트랙 중, 관측 스텝이 모두 |h| ≥ 150° 인 비율 (나머지는 잡음 튐이 점프를 만든 것으로 추정)",
        "n": int(jfull.sum()), "all_steps_ge150_pct": float(near.mean() * 100)}
    summ["logic_tree"] = info
    fig, axs = plt.subplots(2, 2, figsize=(26.0, 18.0), gridspec_kw={"height_ratios": [1.35, 1.0]})
    for ax, tr, ttl in ((axs[0, 0], tree_a10, "주변 차량 · 10 Hz (캐시 agents_hist_val) — 슬롯 스텝 기준 비율"),
                        (axs[0, 1], tree_a2, "주변 차량 · 2 Hz (즉석 계산) — 트랙 / 표본 기준 비율"),
                        (axs[1, 0], tree_f10, "focal · 10 Hz (캐시 ah2)"),
                        (axs[1, 1], tree_f2, "focal · 2 Hz (캐시 ah2_2hz)")):
        C.draw_tree(ax, tr, fontsize=9.6, edge_fs=9.0)
        ax.set_title(ttl, loc="left", fontsize=13, fontweight="bold")
    fig.suptitle(f"h 출처 결정 흐름 — heading_decomp.build_heading 의 갈래마다 실제 개수 (val {S.n:,})",
                 fontsize=17, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.006, "뒤집힘 판정 = align_ref: 위치차분 스텝의 |가드된 AV2 − 위치차분 h| 중앙값이 90° 를 넘으면 그 트랙의 AV2 보조 값을 모두 180° 돌린다. "
             "판정 갈래의 개수는 재구현 판정이며, 캐시 출력에서 본 뒤집힘과 불일치 0 (verify.json). "
             f"'차로 반대' = t=0 위치가 교차로 밖 차로 중심선 {LANE_IN_M:g} m 안이고 t=0 값이 이 갈래(AV2 보조)인 트랙에서, 그 값이 차로 방향과 90° 넘게 다른 비율 (참고값).",
             fontsize=10, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    C.savefig(fig, OUT / "dist_9_h_logic_tree.png", dpi=120)


def fig_cart(S, summ, data):
    """분석용 얕은 결정 트리 (sklearn) — 서술용이지 인과가 아니다."""
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
    plt = C.setup_mpl()
    T = S.T
    scen = T["scen"].astype(int)
    rng = np.random.default_rng(C.SEED)
    te_s = rng.random(S.n) < 0.3
    te = te_s[scen]
    code_txt = "종류 코드 1 = 승용·트럭, 2 = 버스, 3 = 오토바이, 4 = 자전거"

    def thr_fmt(name, x):
        if name == "종류" or "수" in name:
            return f"{x:.1f}"
        return f"{x:.3g}"

    def leaf_table(est, X, y, names, root, kind):
        leaves = est.apply(X)
        df = pd.DataFrame({"leaf": leaves, "y": y, "te": te, "scen": scen, "slot": T["slot"].astype(int)})
        tr = df[~df.te].groupby("leaf").y.agg(["mean", "size"])
        tst = df[df.te].groupby("leaf").y.agg(["mean", "size"])
        rows, worst = [], 0.0
        for nd in C.tree_leaves(root):
            j = nd["id"]
            gm, gn = float(tr.loc[j, "mean"]), int(tr.loc[j, "size"])
            worst = max(worst, abs(gm - nd["value"]), abs(gn - nd["n"]))
            members = df[df.leaf == j]
            pick = members.iloc[int(rng.integers(len(members)))]
            rows.append({"leaf": int(j), "rule": " · ".join(nd["rule"]), "n_train": nd["n"], "value_train": nd["value"],
                         "groupby_train_mean": gm, "n_test": int(tst.loc[j, "size"]) if j in tst.index else 0,
                         "value_test": float(tst.loc[j, "mean"]) if j in tst.index else None,
                         "rep_sid": S.sids[int(pick.scen)], "rep_idx": int(pick.scen), "rep_slot": int(pick.slot)})
        return rows, worst

    # ---- A: AV2 보조 비율 (회귀)
    featA = {"4.9 s 최대 속력": T["vmax"], "t=0 속력": T["v49"], "t=0 거리": T["dist0"], "종류": T["atype"],
             "관측 스텝 수": T["n_obs"], "4.9 s 이동거리": T["path"]}
    XA = np.column_stack(list(featA.values()))
    yA = T["n_src2"] / np.maximum(T["n_obs"], 1)
    A = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=C.SEED).fit(XA[~te], yA[~te])
    rootA = C.sk_tree_nodes(A, list(featA), lambda t, j: float(t.value[j][0][0]),
                            lambda n, v: f"n = {n:,}\nAV2 보조 {v * 100:.1f}%", thr_fmt, 0.0, 1.0)
    rowsA, worstA = leaf_table(A, XA, yA, list(featA), rootA, "reg")
    base_te = float(np.mean((yA[te] - yA[~te].mean()) ** 2))
    infoA = {"target": "주변 차량 트랙의 AV2 보조 스텝 비율 (관측 스텝 중)", "features": list(featA), "n_train": int((~te).sum()),
             "n_test": int(te.sum()), "r2_train": float(A.score(XA[~te], yA[~te])), "r2_test": float(A.score(XA[te], yA[te])),
             "leaf_groupby_max_diff": worstA, "leaves": rowsA,
             "rules": export_text(A, feature_names=list(featA), decimals=2)}
    # ---- B: 10 Hz t=0 h 가 차로 방향과 반대 (분류)
    lm = T["lane_m"] == 1
    featB = {"4.9 s 최대 속력": T["vmax"], "4.9 s 이동거리": T["path"], "위치차분 스텝 수": T["n_src1"],
             "정지 잡음 위치차분 수": T["n_src1_slow"], "관측 스텝 수": T["n_obs"], "t=0 거리": T["dist0"], "종류": T["atype"]}
    XB = np.column_stack(list(featB.values()))
    yB = (T["ag_10"] == 0).astype(float)
    trB, teB = lm & ~te, lm & te
    Bm = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=C.SEED).fit(XB[trB], yB[trB])
    rate = lambda t, j: float(t.value[j][0][1] / t.value[j][0].sum())
    rootB = C.sk_tree_nodes(Bm, list(featB), rate, lambda n, v: f"n = {n:,}\n반대 {v * 100:.1f}%", thr_fmt, 0.0,
                            max(0.3, max(rate(Bm.tree_, j) for j in range(Bm.tree_.node_count))))
    # 잎 표는 차로 안 트랙만 (te 마스크는 전체 기준이라 부분집합으로 다시 만든다)
    sub = np.flatnonzero(lm)
    leavesB = Bm.apply(XB[sub])
    dfB = pd.DataFrame({"leaf": leavesB, "y": yB[sub], "te": te[sub], "scen": scen[sub], "slot": T["slot"][sub].astype(int)})
    trg = dfB[~dfB.te].groupby("leaf").y.agg(["mean", "size"])
    tsg = dfB[dfB.te].groupby("leaf").y.agg(["mean", "size"])
    rowsB, worstB = [], 0.0
    for nd in C.tree_leaves(rootB):
        j = nd["id"]
        worstB = max(worstB, abs(float(trg.loc[j, "mean"]) - nd["value"]), abs(int(trg.loc[j, "size"]) - nd["n"]))
        mem = dfB[dfB.leaf == j]
        pos = mem[mem.y == 1]
        pick = (pos if len(pos) else mem).iloc[int(rng.integers(len(pos) if len(pos) else len(mem)))]
        rowsB.append({"leaf": int(j), "rule": " · ".join(nd["rule"]), "n_train": nd["n"], "value_train": nd["value"],
                      "groupby_train_mean": float(trg.loc[j, "mean"]),
                      "n_test": int(tsg.loc[j, "size"]) if j in tsg.index else 0,
                      "value_test": float(tsg.loc[j, "mean"]) if j in tsg.index else None,
                      "rep_sid": S.sids[int(pick.scen)], "rep_idx": int(pick.scen), "rep_slot": int(pick.slot),
                      "rep_is_positive": bool(pick.y == 1)})
    from sklearn.metrics import roc_auc_score
    pred = Bm.predict(XB[teB])
    maj = float(max(yB[trB].mean(), 1 - yB[trB].mean()))
    maj_te = float(max(yB[teB].mean(), 1 - yB[teB].mean()))
    p_te = Bm.predict_proba(XB[teB])[:, 1]                     # 잎의 학습 비율 = 잎 통계 확률
    brier = float(np.mean((p_te - yB[teB]) ** 2))
    base_p = float(yB[trB].mean())
    brier_ref = float(np.mean((base_p - yB[teB]) ** 2))        # 기준 = 학습 양성 비율을 모든 트랙에 준 예보
    infoB = {"target": "10 Hz h[49] 가 차로 방향과 반대 (차로 안·교차로 밖 트랙)", "features": list(featB),
             "n_train": int(trB.sum()), "n_test": int(teB.sum()), "n_all": int(lm.sum()),
             "pos_rate_train": base_p, "pos_rate_test": float(yB[teB].mean()),
             "auc_test": float(roc_auc_score(yB[teB], p_te)), "brier_test": brier, "brier_ref_test": brier_ref,
             "brier_skill_test": 1.0 - brier / brier_ref,
             "acc_majority_baseline_test": maj_te,
             "note": "양성 약 7% 인 불균형 자료라 정확도는 맞지 않는 지표다. 잎 통계 확률로 AUC·Brier 기술점수를 본다. "
                     "'정지 잡음 위치차분 수' 분기는 뒤집힘 규칙의 전제조건(위치차분 ≥ 3)과 기계적으로 이어져 새 발견이 아니다",
             "acc_test": float((pred == yB[teB]).mean()), "acc_majority_baseline": maj,
             "recall_test": float(((pred == 1) & (yB[teB] == 1)).sum() / max((yB[teB] == 1).sum(), 1)),
             "precision_test": float(((pred == 1) & (yB[teB] == 1)).sum() / max((pred == 1).sum(), 1)),
             "leaf_groupby_max_diff": worstB, "leaves": rowsB,
             "rules": export_text(Bm, feature_names=list(featB), decimals=2)}
    summ["cart"] = {"note": "서술용 얕은 트리다 — 인과가 아니다. 학습/검증은 시나리오 단위 70/30 (시드 0), 깊이 3, 잎당 ≥ 200.",
                    "code": code_txt, "A_av2_ratio": infoA, "B_lane_opposite": infoB}
    (data / "tree_A_rules.txt").write_text(infoA["rules"])
    (data / "tree_B_rules.txt").write_text(infoB["rules"])
    fig, axs = plt.subplots(2, 1, figsize=(26.0, 15.0))
    C.draw_tree(axs[0], rootA, fontsize=10, edge_fs=9.5)
    axs[0].set_title(f"A. 주변 차량 한 대의 AV2 보조 비율 (회귀 · 색 = 평균) — 학습 {infoA['n_train']:,} 트랙 · "
                     f"R² 학습 {infoA['r2_train']:.3f} / 검증 {infoA['r2_test']:.3f}", loc="left", fontsize=13, fontweight="bold")
    C.draw_tree(axs[1], rootB, fontsize=10, edge_fs=9.5)
    axs[1].set_title(f"B. 10 Hz t=0 h 가 차로 방향과 반대인가 (분류 · 색 = 반대 비율) — 차로 안 {infoB['n_all']:,} 트랙 중 학습 "
                     f"{infoB['n_train']:,} / 검증 {infoB['n_test']:,} · 검증 AUC {infoB['auc_test']:.3f} · Brier 기술점수 "
                     f"{infoB['brier_skill_test']:.2f}\n   (양성 {infoB['pos_rate_test'] * 100:.1f}% 불균형 → 정확도는 부적합: 검증 정확도 "
                     f"{infoB['acc_test'] * 100:.1f}%, 검증 다수 클래스 {maj_te * 100:.2f}%.  '정지 잡음 위치차분 수' 분기는 뒤집힘 판정 "
                     f"전제(위치차분 ≥ 3)와 기계적으로 이어진다)", loc="left", fontsize=12.5, fontweight="bold")
    fig.suptitle("분석용 얕은 결정 트리 (sklearn, 깊이 3, 잎당 ≥ 200) — 서술용이지 인과가 아니다", fontsize=17,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.006, f"{code_txt}.  정지 잡음 위치차분 수 = AV2 속력 < 0.5 m/s 인데 위치차분 출처가 된 스텝 수.  "
             "노드의 n 은 학습 분할 기준.  Brier 기술점수 = 1 − Brier / (학습 양성 비율 예보의 Brier).  "
             "학습/검증 = 시나리오 단위 70/30 무작위(시드 0).  잎 값은 group-by 로 따로 집계해 대조했다(summary.json cart.*.leaf_groupby_max_diff).",
             fontsize=10, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    C.savefig(fig, OUT / "dist_10_cart_trees.png", dpi=120)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "scan", "picks", "panels", "dist"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="scan 을 val 앞 N 개로 (0 = 전체)")
    ap.add_argument("--rescan", action="store_true")
    a = ap.parse_args()
    if a.workers > 16:
        raise SystemExit("공유 머신 규칙: workers ≤ 16")
    global OUT
    if a.limit:
        OUT = OUT / f"dev_n{a.limit}"
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "scan"):
        if a.rescan or not (data / "scan.npz").exists():
            run_scan(a, data)
        else:
            print(f"[scan] 있음: {data / 'scan.npz'}")
    if a.stage == "scan":
        return
    S = Scan(data)
    if a.stage in ("all", "picks") or not (data / "picks.json").exists():
        picks = run_picks(S, data)
    else:
        picks = json.loads((data / "picks.json").read_text())
    if a.stage in ("all", "panels"):
        run_panels(S, picks, data)
    if a.stage in ("all", "dist"):
        s = run_dist(S, data)
        print(json.dumps({k: s[k] for k in ("jump", "slip", "ramp")}, ensure_ascii=False, indent=1)[:3000])
    meta = {"git_head": C.git_head(), "caches": [str(CACHE10), str(CACHE2), str(AGENTS)], "n": S.n,
            "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    (data / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
