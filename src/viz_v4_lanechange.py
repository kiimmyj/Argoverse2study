"""
viz_v4_lanechange.py - "차선변경을 왜 못 그리나" 전용 분석.

물음 (사용자, 2026-09-30)
  (1) 전처리 평활을 잘 하면 heading 을 잘 따라가니 차선변경이 자연스럽게 풀리나?
  (2) 아니면 규칙(±1.75 m 밴드)을 섞어야 하나?
  둘 다 아니면 제3 원인(후보 경로 열거 / 주변 차량 정보 없음 / 라벨 희소)인가?

단계
  scan   정답 쪽 표 — 차선변경 정의 3가지, 규모, 시작 시점(관측/예측), 밴드, 후보 경로 커버리지,
         지도 차로 인덱스 변화, U턴 빈도.  -> data/gt.npz · data/gt_groups.json
  infer  판별 태그마다 추론 — 시나리오별 지표 + 모드별 (d, 밴드, hinge) + 차선변경 부분집합의
         정답 기준 경로 투영.  -> data/pred_<tag>.npz
  figs   그림.  tree  결정 트리.

  python src/viz_v4_lanechange.py scan
  python src/viz_v4_lanechange.py infer --tags v4_l4nw_ah2_full_sm1_cos30_s0,...
  python src/viz_v4_lanechange.py figs
  python src/viz_v4_lanechange.py tree

정의·임계값은 viz_v4_common.py 를 따르고, 여기서만 쓰는 것은 아래 상수에 둔다.
학습은 하지 않는다 (추론만).
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

OUT = C.VIZ_ROOT / "lanechange"
DATA = OUT / "data"
# 정답 쪽 양(상황 분류·Δd6·정답 기준 경로)은 모델과 무관하다. 이미 구운 전수 덤프에서 읽는다.
BASE_DUMP = C.VIZ_ROOT / "v4_l4nw_ah2_full_s0" / "data"

# ---------------------------------------------------------------- 이 분석의 임계값
LC_D_M = 2.5          # [m] 정답 기준 경로 대비 6초 횡이동 |Δd6| 이 이상이면 '횡이동 있음'
LC_HEAD_DEG = 15.0    # [deg] |Δh6| 이 미만이어야 차선변경 (그 이상은 회전)
LC_HEAD_WIDE = 30.0   # [deg] 넓은 정의 (곡선 도로 위 차선변경까지)
LC_START_OBS_M = 1.0  # [m] 관측 마지막 2초에 이미 이만큼 옆으로 갔으면 '이미 시작'
LC_NEW_OBS_M = 0.5    # [m] 그 값이 이 미만이면 '예측 구간에서 새로 시작'
LC_OBS_SEC = 2.0      # [s] 관측 쪽 횡이동을 재는 창 (경로가 과거를 안 덮으므로 d 대신 진행방향으로 잰다)
UTURN_DEG = 135.0     # [deg] 6초 방향 변화 이 이상이면 U턴
HALF_W = 1.75         # [m] 기본 밴드 (lane_frame.LANE_HALF_W)
WIDE_W = 3.6          # [m] 차선변경 허용·교차로 밴드 (lane_frame.WIDE_HALF_W)
ROUTE_COVER_M = 1.75  # [m] 후보 경로가 정답을 '차로 안에' 담는 기준 (max|d| <= 이 값)
DD_HIT_FRAC = 0.5     # 모델 횡이동이 정답의 이 비율 이상이면 '차선변경을 그렸다'
MAP_MAX_HOPS = 14     # 지도 차로 인덱스 변화 판정 BFS 홉 한계
MAP_MAX_LAT = 3       # 세는 최대 횡이동 수


# ================================================================= scan (정답 쪽)
_W = {}


def _init_scan(cache_dir):
    from dataset_cached import CachedV4Dataset
    ds = CachedV4Dataset(cache_dir)
    for k in ("routes", "route_tan", "route_band", "route_len", "route_mask", "n_distinct",
              "route_fallback", "route_sd0", "y", "origin", "theta"):
        _W[k] = ds.raw(k)


def _band_at(band, length, s):
    """64점 밴드를 호길이 s 에서 선형보간 (viz_v4_dump._band_at 와 같은 식)."""
    M = band.shape[0]
    idx = np.clip(np.asarray(s, np.float64) / max(float(length), 1e-3) * (M - 1), 0.0, M - 1 - 1e-4)
    i0 = np.floor(idx).astype(int)
    f = (idx - i0)[:, None]
    return band[i0] * (1 - f) + band[np.minimum(i0 + 1, M - 1)] * f


def _lane_of(g, xy, hdir, path):
    c = g.candidate_lanes(np.asarray(xy, np.float64), np.asarray(hdir, np.float64), path=path)
    return c[0] if c else -1


def _lat_moves(g, a, b):
    """차로 a -> b 로 가는 데 필요한 최소 '옆 차로 이동' 수. 도달 못 하면 -1.
    successor 는 공짜, 지도의 left/right 이웃(규칙 게이트 없음)은 1 이다."""
    if a < 0 or b < 0:
        return -1
    if a == b:
        return 0
    best = {a: 0}
    frontier = [(a, 0, 0)]
    while frontier:
        nxt = []
        for lid, lat, hop in frontier:
            if hop >= MAP_MAX_HOPS:
                continue
            ln = g.lanes.get(lid)
            if ln is None:
                continue
            for sid_, dlat in ([(s, 0) for s in ln.successors]
                               + [(x, 1) for x in (ln.left_neighbor, ln.right_neighbor)
                                  if x is not None]):
                if sid_ not in g.lanes:
                    continue
                nl = lat + dlat
                if nl > MAP_MAX_LAT or best.get(sid_, 99) <= nl:
                    continue
                best[sid_] = nl
                nxt.append((sid_, nl, hop + 1))
        frontier = nxt
    return int(best.get(b, -1))


def scan_one(task):
    """시나리오 하나의 정답 쪽 양. 모델을 전혀 쓰지 않는다."""
    import lane_frame as lf
    from lane_graph import LaneGraph, Move
    i, sid, g_idx, pos, h, v_f = task
    pos = np.asarray(pos, np.float64)
    h = np.asarray(h, np.float64)
    v_f = np.asarray(v_f, np.float64)
    nd = max(int(_W["n_distinct"][i]), 1)
    theta = float(_W["theta"][i])
    origin = np.asarray(_W["origin"][i], np.float64)

    # --- 후보 경로별 정답 (s, d, 밴드)
    maxd_f = np.full(6, np.nan)          # 예측 구간 max|d|
    endd_f = np.full(6, np.nan)          # 마지막 5스텝 평균 d
    inband = np.zeros(6, bool)
    d_g = s_g = None
    bl_g = br_g = None
    for r in range(nd):
        rd = C.route_dict(_W["routes"][i, r], _W["route_tan"][i, r], _W["route_len"][i, r])
        s_r, d_r, _ = lf.to_frame(pos, rd)
        bd = _band_at(np.asarray(_W["route_band"][i, r], np.float64), float(_W["route_len"][i, r]), s_r)
        maxd_f[r] = np.abs(d_r[C.OBS:]).max()
        endd_f[r] = d_r[-5:].mean()
        inband[r] = bool(np.all(d_r[C.OBS:] <= bd[C.OBS:, 0]) and np.all(-d_r[C.OBS:] <= bd[C.OBS:, 1]))
        if r == g_idx:
            d_g, s_g, bl_g, br_g = d_r, s_r, bd[:, 0], bd[:, 1]
    if d_g is None:                       # 방어 (gt_route 가 nd 밖일 수 없다)
        d_g = np.zeros(len(pos)); s_g = np.zeros(len(pos))
        bl_g = np.full(len(pos), HALF_W); br_g = np.full(len(pos), HALF_W)

    # --- 관측 구간 횡이동: 경로에 의존하지 않는 방식
    # 후보 경로는 현재 위치에서 시작하므로 과거 위치의 s 가 음수인 시나리오가 D2 의 76% 다
    # (창 시작 기준). 그 구간의 d 는 '경로 시작점까지의 수직거리' 라 ±30 m 까지 튄다.
    # 그래서 관측 쪽은 d 대신 **현재 차로 방향 k(s0) 대비 진행방향**으로 잰다:
    #   lat_obs = Σ v(t)·sin(h(t) − k0)·dt,  t ∈ [−LC_OBS_SEC, 0]
    idx0 = np.clip(float(_W["route_sd0"][i, g_idx, 0]) / max(float(_W["route_len"][i, g_idx]), 1e-3)
                   * (64 - 1), 0.0, 64 - 1 - 1e-4)
    j0 = int(np.floor(idx0)); fr = idx0 - j0
    tg0 = (np.asarray(_W["route_tan"][i, g_idx], np.float64)[j0] * (1 - fr)
           + np.asarray(_W["route_tan"][i, g_idx], np.float64)[min(j0 + 1, 63)] * fr)
    k0 = float(np.arctan2(tg0[1], tg0[0]))
    n_obs = int(round(LC_OBS_SEC / C.DT))
    sl_obs = slice(C.OBS - 1 - n_obs, C.OBS)
    lat_obs = float((v_f[sl_obs] * np.sin(C.wrap(h[sl_obs] - k0)) * C.DT).sum())
    th0_true = float(C.wrap(h[C.OBS - 1] - k0))

    # --- 차선변경 진행 곡선 (110스텝 전체)
    d_a = d_g[:5].mean()                                # 창 시작 (−4.9 s)
    d_0 = d_g[C.OBS - 3:C.OBS + 2].mean()               # 예측 시작 (0 s)
    d_e = d_g[-5:].mean()                               # 창 끝 (+6 s)
    dd_full, dd_obs, dd6 = d_e - d_a, d_0 - d_a, d_e - d_0
    t10 = t90 = np.nan
    if abs(dd_full) > 1e-6:
        f = (d_g - d_a) / dd_full
        for frac, key in ((0.1, "t10"), (0.9, "t90")):
            hit = np.nonzero(f >= frac)[0]
            v = C.T_AX[hit[0]] if len(hit) else np.nan
            if key == "t10":
                t10 = v
            else:
                t90 = v

    # --- 밴드: 정답이 어디서 얼마를 벗어나나 (정답 기준 경로 위)
    over175 = float(np.maximum(np.abs(d_g[C.OBS:]) - HALF_W, 0.0).max())
    band_sign = np.where(d_g[C.OBS:] >= 0, bl_g[C.OBS:], br_g[C.OBS:])
    over_rule = float(np.maximum(np.abs(d_g[C.OBS:]) - band_sign, 0.0).max())
    over36 = float(np.maximum(np.abs(d_g[C.OBS:]) - WIDE_W, 0.0).max())
    j = int(np.argmax(np.abs(d_g[C.OBS:]))) + C.OBS     # 최대 횡오프셋 시점
    band_at_max = float(bl_g[j] if d_g[j] >= 0 else br_g[j])
    wide_frac = float((band_sign > HALF_W + 1e-3).mean())   # 예측 구간에서 3.6 이 적용된 비율

    # --- 지도 차로 인덱스
    raw = json.loads((C.VAL_DIR / sid / f"log_map_archive_{sid}.json").read_text())
    gr = LaneGraph.from_json_dict(raw, centerline="api")
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    city = pos @ R.T + origin                                  # 정규화 -> city
    hc = h + theta
    hd0 = np.array([np.cos(hc[C.OBS - 1]), np.sin(hc[C.OBS - 1])])
    hd1 = np.array([np.cos(hc[-2]), np.sin(hc[-2])])
    l0 = _lane_of(gr, city[C.OBS - 1], hd0, city[:C.OBS])
    l1 = _lane_of(gr, city[-1], hd1, city[-10:])
    n_lat = _lat_moves(gr, l0, l1)
    rule_ok = -1
    if l0 >= 0:
        mv = {m for _, m in gr.edges.get(l0, [])}
        rule_ok = int((Move.LEFT in mv) * 1 + (Move.RIGHT in mv) * 2)   # 비트: 1 좌 · 2 우
    inter0 = bool(gr.lanes[l0].is_intersection) if l0 in gr.lanes else False

    row = dict(idx=i, dd_full=dd_full, dd_obs=dd_obs, dd6_r=dd6, d0_g=d_0, d_a=d_a, d_e=d_e,
               lat_obs=lat_obs, th0_true=th0_true, k0=k0, s_obs_start=float(s_g[0]),
               s_obs_2s=float(s_g[C.OBS - 1 - n_obs]),
               t10=t10, t90=t90, over175=over175, over_rule=over_rule, over36=over36,
               band_at_max=band_at_max, wide_frac=wide_frac,
               lane0=l0, lane1=l1, n_lat_map=n_lat, rule_lc=rule_ok, inter0=inter0,
               n_cover175=int((maxd_f[:nd] <= ROUTE_COVER_M).sum()),
               n_cover36=int((maxd_f[:nd] <= WIDE_W).sum()),
               n_inband_f=int(inband[:nd].sum()), min_maxd=float(np.nanmin(maxd_f[:nd])))
    arr = dict(d_g=d_g.astype(np.float32), s_g=s_g.astype(np.float32),
               band_g=np.stack([bl_g, br_g], 1).astype(np.float32),
               maxd_f=maxd_f.astype(np.float32), endd_f=endd_f.astype(np.float32),
               inband_f=inband)
    return row, arr


def step_scan(a):
    import pandas as pd
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(BASE_DUMP / "scenarios.parquet")
    raw = np.load(BASE_DUMP / "raw.npz")
    meta = json.loads((BASE_DUMP / "meta.json").read_text())
    assert meta["val_cache"] == str(C.VAL_CACHES["ah2"]), meta["val_cache"]
    N = len(df) if a.limit is None else min(a.limit, len(df))
    sids = list(df["sid"][:N])
    pos = raw["pos"][:N].astype(np.float64)
    hh = raw["h"][:N].astype(np.float64)
    vv = raw["v_fld"][:N].astype(np.float64)
    gidx = df["gt_route"].to_numpy()[:N]

    tasks = [(i, sids[i], int(gidx[i]), pos[i], hh[i], vv[i]) for i in range(N)]
    rows, arrs = [None] * N, [None] * N
    with Pool(a.workers, initializer=_init_scan, initargs=(str(C.VAL_CACHES["ah2"]),)) as pool:
        for k, (r_, ar) in enumerate(pool.imap(scan_one, tasks, chunksize=16)):
            rows[k], arrs[k] = r_, ar
            if (k + 1) % 5000 == 0:
                print(f"[scan] {k + 1}/{N}  {time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(rows)
    A = {k: np.stack([x[k] for x in arrs]) for k in arrs[0]}

    # --- 대조: 새로 잰 Δd6 이 기존 덤프의 dd6 과 같은가 (독립 경로 재계산)
    chk = float(np.abs(G["dd6_r"].to_numpy() - df["dd6"].to_numpy()[:N]).max())
    print(f"[check] Δd6 재계산 vs 기존 덤프 최대차 {chk:.2e} m", flush=True)
    if chk > 1e-3:
        raise SystemExit("Δd6 가 기존 덤프와 다르다 — 정답 기준 경로 인덱스가 어긋났다")

    # --- 정의별 집합
    d = df.iloc[:N].reset_index(drop=True)
    lat = np.abs(G["dd6_r"].to_numpy())
    dh = np.abs(d["dh6"].to_numpy())
    geo = d["lc_ok"].to_numpy()                  # 폴백 아님 · 경로와 나란함 · max|d|<=5 · 6초 10 m 이상 이동
    defs = {}
    defs["D1_분류"] = d["cls"].isin(["좌차선변경", "우차선변경"]).to_numpy()
    defs["D2_횡이동"] = geo & (lat >= LC_D_M) & (dh < LC_HEAD_WIDE)
    defs["D3_지도차로"] = (G["n_lat_map"].to_numpy() >= 1) & geo & (dh < LC_HEAD_WIDE)
    defs["D4_횡이동_지도일치"] = defs["D2_횡이동"] & (G["n_lat_map"].to_numpy() >= 1)
    uni = defs["D1_분류"] | defs["D2_횡이동"] | defs["D3_지도차로"]
    defs["U_합집합"] = uni
    obsd = np.abs(G["lat_obs"].to_numpy())
    same = np.sign(G["lat_obs"].to_numpy()) == np.sign(G["dd6_r"].to_numpy())
    started = uni & (obsd >= LC_START_OBS_M) & same
    fresh = uni & (obsd < LC_NEW_OBS_M)
    defs["S_관측중시작"] = started
    defs["S_예측중새로"] = fresh
    defs["S_중간"] = uni & ~started & ~fresh
    defs["UT_U턴"] = (dh >= UTURN_DEG)

    groups = {k: {"n": int(v.sum()), "pct_val": round(100.0 * v.mean(), 3),
                  "sids": [sids[j] for j in np.nonzero(v)[0]]} for k, v in defs.items()}
    counts = {k: v["n"] for k, v in groups.items()}
    print("[groups] " + "  ".join(f"{k} {v}" for k, v in counts.items()), flush=True)

    np.savez(DATA / "gt.npz", sids=np.array(sids), **A,
             **{k: G[k].to_numpy() for k in G.columns},
             **{f"def_{k}": v for k, v in defs.items()})
    (DATA / "gt_groups.json").write_text(json.dumps(
        {"n_val": N, "base_dump": str(BASE_DUMP), "git_head": C.git_head(),
         "thresholds": {k: globals()[k] for k in ("LC_D_M", "LC_HEAD_DEG", "LC_HEAD_WIDE",
                                                 "LC_START_OBS_M", "LC_NEW_OBS_M", "LC_OBS_SEC", "UTURN_DEG",
                                                 "HALF_W", "WIDE_W", "ROUTE_COVER_M", "DD_HIT_FRAC")},
         "dd6_recheck_max_diff_m": chk, "counts": counts, "groups": groups},
        indent=1, ensure_ascii=False))
    print(f"[done] scan {time.time() - t0:.0f}s -> {DATA}", flush=True)


# ================================================================= infer (모델 쪽)
def step_infer(a):
    import torch
    from torch.utils.data import DataLoader
    from dataset_cached import CachedV4Dataset
    from model_v4 import A_SCALE, DTHETA_MAX
    from train_v4 import to_dev

    gt = np.load(DATA / "gt.npz", allow_pickle=True)
    sub = np.nonzero(gt["def_U_합집합"])[0]                 # 차선변경 후보 집합 (모드별 투영 대상)
    device = C.pick_device(a.device)
    tags = [t for t in a.tags.split(",") if t]
    verify = {}
    for tag in tags:
        t0 = time.time()
        model, info = C.load_model(tag, device)
        cache = C.val_cache_for(info["input"])
        ds = CachedV4Dataset(cache)
        N, K, T = len(ds), 6, C.FUT
        P = {k: np.zeros((N, K), np.float32) for k in
             ("prob", "ade", "fde", "hinge", "d0", "dend", "dmax", "band_at_dmax", "over_max",
              "smax", "off_frac", "vend", "thmax")}
        P["alive"] = np.zeros((N, K), bool)
        S = {k: np.zeros(N, np.float32) for k in ("minade", "minfde")}
        S["winner"] = np.zeros(N, np.int16)
        S["top1"] = np.zeros(N, np.int16)
        keep = {i: j for j, i in enumerate(sub)}
        TR = np.zeros((len(sub), K, T, 2), np.float32)
        DD = np.zeros((len(sub), K, T), np.float32)       # 모델 d (자기 경로 기준)
        TH = np.zeros((len(sub), K, T), np.float32)       # 잔차각 θ [rad]
        VV = np.zeros((len(sub), K, T), np.float32)       # 속력
        BD = np.zeros((len(sub), K, T, 2), np.float32)    # 밴드 (좌, 우)
        exc = live = jt = ja = 0.0
        i0 = 0
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=2):
                traj, logits, aux = model(**to_dev(b, device, "l0", True))
                y = b["y"].to(device)
                mm = b["route_mask"].to(device)
                lv = mm > 0
                B = y.size(0)
                dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
                ade_m, fde_m = dist.mean(2), dist[:, :, -1]
                big = torch.full_like(ade_m, float("inf"))
                winner = fde_m.masked_fill(mm == 0, float("inf")).argmin(1)
                over = (torch.relu(aux["d"] - aux["band"][..., 0])
                        + torch.relu(-aux["d"] - aux["band"][..., 1]))
                dmi = aux["d"].abs().argmax(2, keepdim=True)
                th = aux["theta"]
                dd = (th[:, :, 1:] - th[:, :, :-1]).abs() * 180.0 / np.pi
                ut, ua = aux["dtheta"] / DTHETA_MAX, aux["a"] / A_SCALE
                lm = mm.unsqueeze(-1)
                exc += ((dd > C.LABEL_DTHETA_DEG).float() * lm).sum().item()
                live += lm.sum().item() * dd.size(2)
                jt += (((ut[..., 1:] - ut[..., :-1]) ** 2) * lm).sum().item()
                ja += (((ua[..., 1:] - ua[..., :-1]) ** 2) * lm).sum().item()
                sl = slice(i0, i0 + B)
                cpu = lambda t: t.detach().float().cpu().numpy()
                P["prob"][sl] = cpu(torch.softmax(logits, 1))
                P["ade"][sl], P["fde"][sl] = cpu(ade_m), cpu(fde_m)
                P["hinge"][sl] = cpu(over.mean(2))
                P["off_frac"][sl] = cpu((over > 0).float().mean(2))
                P["d0"][sl] = cpu(b["route_sd0"][..., 1].to(device))
                P["dend"][sl] = cpu(aux["d"][:, :, -1])
                P["dmax"][sl] = cpu(torch.gather(aux["d"], 2, dmi).squeeze(2))
                P["band_at_dmax"][sl] = cpu(torch.gather(
                    torch.where(aux["d"] >= 0, aux["band"][..., 0], aux["band"][..., 1]), 2, dmi).squeeze(2))
                P["over_max"][sl] = cpu(over.max(2).values)
                P["smax"][sl] = cpu(aux["s"][:, :, -1])
                P["vend"][sl] = cpu(aux["v"][:, :, -1])
                P["thmax"][sl] = cpu(th.abs().max(2).values) * 180.0 / np.pi
                P["alive"][sl] = cpu(lv) > 0
                S["minade"][sl] = cpu(torch.where(lv, ade_m, big).min(1).values)
                S["minfde"][sl] = cpu(torch.where(lv, fde_m, big).min(1).values)
                S["winner"][sl] = cpu(winner).astype(np.int16)
                S["top1"][sl] = cpu(logits.argmax(1)).astype(np.int16)
                tj, dcpu = cpu(traj), cpu(aux["d"])
                thc, vc, bdc = cpu(th), cpu(aux["v"]), cpu(aux["band"])
                for bi in range(B):
                    j = keep.get(i0 + bi)
                    if j is not None:
                        TR[j], DD[j] = tj[bi], dcpu[bi]
                        TH[j], VV[j], BD[j] = thc[bi], vc[bi], bdc[bi]
                i0 += B
        assert i0 == N
        got = {"minADE6": float(S["minade"].astype(np.float64).mean()),
               "minFDE6": float(S["minfde"].astype(np.float64).mean()),
               "offlane": float(np.where(P["alive"], P["off_frac"], 0.0).sum(1).astype(np.float64).mean()),
               "dtheta_over_label_pct": 100.0 * exc / live,
               "jitter_theta": jt / live, "jitter_a": ja / live}
        exp = C.expected_scores(tag)
        _, rj = C.run_args(tag)
        logged = {"minADE6": rj["best_minADE6"], "minFDE6": rj["best_minFDE6"]}
        v = {"got": got, "compare_json": exp, "runs_json_best": logged,
             "diff_vs_runs_json": {k: got[k] - logged[k] for k in logged},
             "n": N, "cache": str(cache), "input": info["input"], "sec": time.time() - t0}
        if exp is not None:
            v["diff_vs_compare_json"] = {k: got[k] - exp[k] for k in got}
            bad = [k for k in got if abs(got[k] - exp[k]) > (2e-5 if k.startswith("jitter") else 2e-3)]
            v["compare_ok"] = not bad
            if bad:
                raise SystemExit(f"{tag}: 재현 실패 {bad} — {v}")
        if max(abs(x) for x in v["diff_vs_runs_json"].values()) > 2e-3:
            raise SystemExit(f"{tag}: runs/{tag}.json 의 best 와 다르다 — {v['diff_vs_runs_json']}")
        verify[tag] = v
        print(f"[{tag}] minADE6 {got['minADE6']:.4f} minFDE6 {got['minFDE6']:.4f} "
              f"(json {logged['minADE6']:.4f}/{logged['minFDE6']:.4f})  {time.time() - t0:.0f}s", flush=True)

        # --- 차선변경 부분집합: 모드 궤적을 **정답 기준 경로**에 투영 (모드마다 경로가 달라 비교가 안 되므로)
        t1 = time.time()
        tasks = [(int(i), TR[j]) for j, i in enumerate(sub)]
        with Pool(a.workers, initializer=_init_scan, initargs=(str(C.VAL_CACHES["ah2"]),)) as pool:
            pd_ = np.stack(pool.map(_proj_one, tasks, chunksize=8))
        print(f"[{tag}] 투영 {len(sub)}개 {time.time() - t1:.0f}s", flush=True)
        np.savez(DATA / f"pred_{tag}.npz", sub=sub, traj_sub=TR, d_sub=DD, proj_d_sub=pd_,
                 th_sub=TH, v_sub=VV, band_sub=BD, **P, **S)
        del P, S, TR, DD, TH, VV, BD, pd_, model
        if device == "cuda":
            torch.cuda.empty_cache()
    p = DATA / "infer_verify.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(verify)
    p.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    print(f"[done] infer -> {p}", flush=True)


def _proj_one(task):
    """모델 6모드 궤적 (K,60,2) 을 그 시나리오의 **정답 기준 경로**에 투영한 d (K,60)."""
    import lane_frame as lf
    i, tr = task
    g = int(_G_ROUTE[i])
    rd = C.route_dict(_W["routes"][i, g], _W["route_tan"][i, g], _W["route_len"][i, g])
    out = np.zeros(tr.shape[:2], np.float32)
    for k in range(tr.shape[0]):
        _, d_, _ = lf.to_frame(np.asarray(tr[k], np.float64), rd)
        out[k] = d_
    return out


_G_ROUTE = None


def _load_groute():
    global _G_ROUTE
    import pandas as pd
    _G_ROUTE = pd.read_parquet(BASE_DUMP / "scenarios.parquet")["gt_route"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["scan", "infer", "figs", "tree"])
    ap.add_argument("--tags", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.workers > 16:
        raise SystemExit("workers 는 16 이하 (공유 머신)")
    if a.step == "scan":
        step_scan(a)
    elif a.step == "infer":
        _load_groute()
        step_infer(a)
    else:
        import viz_v4_lanechange_figs as F
        (F.step_figs if a.step == "figs" else F.step_tree)(a)


if __name__ == "__main__":
    main()
