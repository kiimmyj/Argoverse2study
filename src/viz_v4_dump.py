"""
viz_v4_dump.py - v4 체크포인트의 val 추론 덤프 + 원본 parquet 파생량 + 정답 기반 상황 분류.

단계
  1. 추론 (캐시 val 24,988). 모드별 궤적·확률·aux, 모드별 ADE/FDE/hinge/jitter, 시나리오 손실 항.
     먼저 runs/v4_full_compare.json 의 재채점 값과 맞는지 확인한다 — 안 맞으면 멈춘다.
     손실 항은 train_v4.loss_fn / jitter 와 같은 식을 시나리오 하나에 적용한 값이고, 첫 배치에서
     배치 값이 학습 코드와 같게 모이는지 직접 대조한다.
  2. 원본 parquet (병렬 <= 16). 110스텝 위치·속도·가속도·진행방향, 앞차 거리, 후보 경로 기준 정답 (s, d),
     경로 커버리지, 상황 분류.
  3. 표 하나(scenarios.parquet) + 배열 두 묶음(pred.npz, raw.npz) + meta.json.

  python src/viz_v4_dump.py                              # 주 모델
  python src/viz_v4_dump.py --tag v4_l4nw_ah2_full_s0    # 다른 체크포인트
  python src/viz_v4_dump.py --limit 256 --out-dir /tmp/…/smoke  # 스모크 (viz/ 밖에)

정의는 viz_v4_common.py 상수에 있다. 여기서는 계산만 한다.
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

# --------------------------------------------------------------------------- 1. 추론
def run_inference(tag, device, batch, workers, limit=None, model=None, info=None, cache_dir=None):
    """model 을 주지 않으면 best 체크포인트(runs/lstm_<tag>.pth)를 연다. 에폭별 분석은 model·info 를 넘긴다.
    cache_dir 를 주지 않으면 그 판의 입력(info['input'])에 맞는 val 캐시를 쓴다 — 10 Hz 캐시를 2 Hz 모델에
    넣어도 LSTM 은 길이를 가리지 않아 오류 없이 틀린 값이 나오기 때문이다."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from dataset_cached import CachedV4Dataset
    from model_v4 import A_SCALE, DTHETA_MAX
    from train_v4 import to_dev, loss_fn, jitter, jitter_steps

    if model is None:
        model, info = C.load_model(tag, device)
    # 손실 가중치는 그 판의 학습 인자를 따른다 (주 모델: offlane 1.0 · 버려진 모드만 · smooth 1.0)
    w_off, w_sm, nonwin_only = info["offlane"], info["smooth"], info["off_nonwinner"]
    ds = CachedV4Dataset(cache_dir if cache_dir is not None else C.val_cache_for(info["input"]), limit=limit)
    N, K, T = len(ds), 6, C.FUT
    f32 = np.float32
    P = {k: np.zeros((N, K, T, 2), f32) for k in ("traj", "band")}
    P.update({k: np.zeros((N, K, T), f32) for k in ("a", "theta", "dtheta", "v", "s", "d", "h")})
    P.update({k: np.zeros((N, K), f32) for k in ("prob", "ade", "fde", "hinge", "jit", "jit_t_sum",
                                                   "jit_a_sum", "exc_cnt", "off_frac")})
    P["alive"] = np.zeros((N, K), bool)
    S = {k: np.zeros(N, f32) for k in ("minade", "minfde", "l_sl1", "l_ce", "l_hinge", "l_jit", "l_total")}
    S.update({k: np.zeros(N, np.int16) for k in ("winner", "top1")})

    checked = False
    i0 = 0
    t0 = time.time()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
            kw = to_dev(b, device, "l0", True)
            traj, logits, aux = model(**kw)
            y = b["y"].to(device)
            mm = b["route_mask"].to(device)
            lv = mm > 0
            B = y.size(0)
            dist = torch.norm(traj - y.unsqueeze(1), dim=-1)                  # (B,K,T)
            ade_m, fde_m = dist.mean(2), dist[:, :, -1]
            big = torch.full_like(ade_m, float("inf"))
            minade = torch.where(lv, ade_m, big).min(1).values
            minfde = torch.where(lv, fde_m, big).min(1).values
            # 승자 = 학습 손실의 WTA 와 같다 (살아있는 모드 중 끝점 오차 최소)
            winner = fde_m.masked_fill(mm == 0, float("inf")).argmin(1)
            idx = winner.view(B, 1, 1, 1).expand(B, 1, T, 2)
            best = traj.gather(1, idx).squeeze(1)
            sl1 = F.smooth_l1_loss(best, y, reduction="none").mean((1, 2))
            ce = F.cross_entropy(logits, winner, reduction="none")
            over = torch.relu(aux["d"] - aux["band"][..., 0]) + torch.relu(-aux["d"] - aux["band"][..., 1])
            hinge_m = over.mean(2)
            nonwin = mm * (1.0 - F.one_hot(winner, K).float()) if nonwin_only else mm
            l_hinge = (hinge_m * nonwin).sum(1) / nonwin.sum(1).clamp(min=1)
            js = jitter_steps(aux)                                              # (B,K,T-1)
            jit_m = js.mean(2)
            l_jit = (jit_m * mm).sum(1) / mm.sum(1).clamp(min=1)
            total = sl1 + ce + w_off * l_hinge + w_sm * l_jit
            if not checked:
                # 배치로 모으면 학습 코드의 값과 같아야 한다 — 시나리오 분해가 같은 식인지 확인
                ref, off_ref = loss_fn(traj, logits, y, mm, aux, w_off, nonwin_only)
                if w_sm > 0:
                    ref = ref + w_sm * jitter(aux, mm)
                mine = (sl1.mean() + ce.mean()
                        + w_off * (hinge_m * nonwin).sum() / nonwin.sum().clamp(min=1)
                        + w_sm * (jit_m * mm).sum() / mm.sum().clamp(min=1))
                diff = abs(float(ref) - float(mine))
                print(f"[loss check] train_v4 배치 손실 {float(ref):.6f} vs 시나리오 분해 재집계 "
                      f"{float(mine):.6f} (차 {diff:.2e})", flush=True)
                if diff > 1e-4:
                    raise SystemExit("손실 분해가 학습 코드와 다르다 — 멈춘다")
                checked = True
            th = aux["theta"]
            dd = (th[:, :, 1:] - th[:, :, :-1]).abs() * 180.0 / np.pi
            ut = aux["dtheta"] / DTHETA_MAX
            ua = aux["a"] / A_SCALE
            sl = slice(i0, i0 + B)
            cpu = lambda t: t.detach().float().cpu().numpy()
            P["traj"][sl] = cpu(traj)
            P["band"][sl] = cpu(aux["band"])
            for k in ("a", "theta", "dtheta", "v", "s", "d", "h"):
                P[k][sl] = cpu(aux[k])
            P["prob"][sl] = cpu(torch.softmax(logits, 1))
            P["ade"][sl], P["fde"][sl] = cpu(ade_m), cpu(fde_m)
            P["hinge"][sl], P["jit"][sl] = cpu(hinge_m), cpu(jit_m)
            P["jit_t_sum"][sl] = cpu(((ut[..., 1:] - ut[..., :-1]) ** 2).sum(2))
            P["jit_a_sum"][sl] = cpu(((ua[..., 1:] - ua[..., :-1]) ** 2).sum(2))
            P["exc_cnt"][sl] = cpu((dd > C.LABEL_DTHETA_DEG).float().sum(2))
            P["off_frac"][sl] = cpu((over > 0).float().mean(2))
            P["alive"][sl] = cpu(lv) > 0
            S["minade"][sl], S["minfde"][sl] = cpu(minade), cpu(minfde)
            S["l_sl1"][sl], S["l_ce"][sl] = cpu(sl1), cpu(ce)
            S["l_hinge"][sl], S["l_jit"][sl], S["l_total"][sl] = cpu(l_hinge), cpu(l_jit), cpu(total)
            S["winner"][sl] = cpu(winner).astype(np.int16)
            S["top1"][sl] = cpu(logits.argmax(1)).astype(np.int16)
            i0 += B
    assert i0 == N
    print(f"[infer] {N} 시나리오 {time.time() - t0:.0f}s ({device})", flush=True)
    # 죽은 슬롯은 NaN 으로 비운다 (그림·통계가 실수로 쓰지 않게)
    dead = ~P["alive"]
    for k in ("ade", "fde", "hinge", "jit", "off_frac"):
        P[k][dead] = np.nan
    return P, S, info, ds


def score_like_compare(P, S):
    """compare_v4_full.score 와 같은 집계."""
    al = P["alive"]
    live = al.sum() * (C.FUT - 1)
    return {"n": int(len(S["minade"])),
            "minADE6": float(S["minade"].astype(np.float64).mean()),
            "minFDE6": float(S["minfde"].astype(np.float64).mean()),
            "offlane": float(np.nansum(np.where(al, P["off_frac"], 0.0), axis=1).astype(np.float64).mean()),
            "dtheta_over_label_pct": float(100.0 * np.where(al, P["exc_cnt"], 0).sum() / live),
            "jitter_theta": float(np.where(al, P["jit_t_sum"], 0).astype(np.float64).sum() / live),
            "jitter_a": float(np.where(al, P["jit_a_sum"], 0).astype(np.float64).sum() / live)}


# --------------------------------------------------------------------------- 2. 원본 parquet
_W = {}


def _init_worker(cache_dir=None):
    # 여기서 읽는 필드(경로·정답·원점)는 두 val 캐시에서 같다. 그래도 추론과 같은 캐시를 받는다.
    from dataset_cached import CachedV4Dataset
    ds = CachedV4Dataset(cache_dir or C.VAL_CACHE)
    for k in ("routes", "route_tan", "route_band", "route_len", "y", "origin", "theta", "n_distinct",
              "route_fallback"):
        _W[k] = ds.raw(k)


def _smooth_speed(v):
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    vm = median_filter(np.asarray(v, np.float64), size=C.MED_K, mode="nearest")
    return (savgol_filter(vm, C.SG_WIN, C.SG_POLY, mode="interp"),
            savgol_filter(vm, C.SG_WIN, C.SG_POLY, deriv=1, delta=C.DT, mode="interp"))


def _band_at(band, length, s):
    """64점 밴드를 호길이 s 에서 선형보간 (model_v4.interp1d 와 같은 clamp)."""
    M = band.shape[0]
    idx = np.clip(np.asarray(s, np.float64) / max(float(length), 1e-3) * (M - 1), 0.0, M - 1 - 1e-4)
    i0 = np.floor(idx).astype(int)
    f = (idx - i0)[:, None]
    return band[i0] * (1 - f) + band[np.minimum(i0 + 1, M - 1)] * f


def _lead(scn, pos_n, h_u, theta, origin, R):
    """앞차 시계열. 기준선 = focal 이 실제로 달린 110스텝 경로 + 끝 접선 방향 LEAD_EXT_M 연장."""
    from heading_decomp import build_heading
    T = len(pos_n)
    seg = np.linalg.norm(np.diff(pos_n, axis=0), axis=1)
    S_t = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3])
    Pk, Sk = pos_n[keep], S_t[keep]
    L = float(S_t[-1])
    if len(Pk) >= 2 and L > 0.5:
        g = np.append(np.arange(0.0, L, C.PATH_STEP_M), L)
        path = np.stack([np.interp(g, Sk, Pk[:, 0]), np.interp(g, Sk, Pk[:, 1])], 1)
    else:
        g, path = np.array([L]), pos_n[-1:].copy()
    e = np.array([np.cos(h_u[-2]), np.sin(h_u[-2])])
    ge = L + np.arange(C.PATH_STEP_M, C.LEAD_EXT_M + 1e-6, C.PATH_STEP_M)
    G = np.concatenate([g, ge])
    PP = np.concatenate([path, pos_n[-1] + (ge - L)[:, None] * e])
    tg = np.gradient(PP, axis=0) if len(PP) > 1 else np.tile(e, (1, 1))
    nn = np.linalg.norm(tg, axis=1, keepdims=True)
    tg = np.where(nn > 1e-9, tg / np.maximum(nn, 1e-9), e)
    ang = np.arctan2(tg[:, 1], tg[:, 0])

    obs_t, obs_q, obs_h, obs_k, tracks, types = [], [], [], [], [], []
    for tr in scn.tracks:
        if tr.track_id == scn.focal_track_id or tr.object_type.value.upper() not in C.LEAD_TYPES:
            continue
        st = tr.object_states
        ts = np.array([x.timestep for x in st], int)
        q = (np.array([x.position for x in st], np.float64) - origin) @ R.T
        if len(ts) < 2 or np.min(((q[:, None, :] - PP[None, ::5, :]) ** 2).sum(-1)) > (C.LEAD_LAT_M + 8.0) ** 2:
            continue
        full = np.zeros((T, 2)); obs = np.zeros(T, bool); href = np.zeros(T)
        full[ts], obs[ts] = q, True
        href[ts] = np.array([x.heading for x in st], np.float64) - theta
        hq, _ = build_heading(full, obs=obs, h_ref=href)
        k = len(tracks)
        tracks.append(np.where(obs[:, None], full, np.nan))
        types.append(tr.object_type.value)
        obs_t.append(ts); obs_q.append(q); obs_h.append(hq[ts]); obs_k.append(np.full(len(ts), k))
    dist = np.full(T, np.nan)
    lid = np.full(T, -1, int)
    if tracks:
        ot, oq = np.concatenate(obs_t), np.concatenate(obs_q)
        oh, ok = np.concatenate(obs_h), np.concatenate(obs_k)
        st0 = S_t[ot]
        d2 = ((oq[:, None, :] - PP[None, :, :]) ** 2).sum(-1)
        win = (G[None, :] >= st0[:, None] - 2.0) & (G[None, :] <= st0[:, None] + C.LEAD_MAX_M + 5.0)
        d2 = np.where(win, d2, np.inf)
        j = d2.argmin(1)
        rel = oq - PP[j]
        lon = G[j] + (rel * tg[j]).sum(1) - st0
        lat = tg[j, 0] * rel[:, 1] - tg[j, 1] * rel[:, 0]
        dh = np.abs(C.wrap(oh - ang[j]))
        ok_m = (np.isfinite(d2[np.arange(len(j)), j]) & (lon > C.LEAD_MIN_M) & (lon <= C.LEAD_MAX_M)
                & (np.abs(lat) < C.LEAD_LAT_M) & (dh < np.radians(C.LEAD_HEAD_DEG)))
        for t_, lo, kk in zip(ot[ok_m], lon[ok_m], ok[ok_m]):
            if not (lo >= dist[t_]):          # NaN 이거나 더 가까우면 갱신
                dist[t_], lid[t_] = lo, kk
    pick = lid[C.OBS - 1]
    if pick < 0:
        fut = lid[C.OBS:][lid[C.OBS:] >= 0]
        pick = int(np.bincount(fut).argmax()) if len(fut) else -1
    track = tracks[pick] if pick >= 0 else np.full((T, 2), np.nan)
    return dist, lid, track, (types[lid[C.OBS - 1]] if lid[C.OBS - 1] >= 0 else ""), len(tracks)


def raw_one(task):
    from av2.datasets.motion_forecasting import scenario_serialization as ss
    from dataset_map import _rotation_matrix
    from heading_decomp import build_heading
    import lane_frame as lf
    i, sid, win = task
    scn = ss.load_argoverse_scenario_parquet(C.VAL_DIR / sid / f"scenario_{sid}.parquet")
    foc = next(t for t in scn.tracks if t.track_id == scn.focal_track_id)
    st = sorted(foc.object_states, key=lambda x: x.timestep)
    assert len(st) == C.OBS + C.FUT, f"{sid}: focal {len(st)} 스텝"
    pos32 = np.array([x.position for x in st], dtype=np.float32)
    vel = np.array([x.velocity for x in st], dtype=np.float64)
    head = np.array([x.heading for x in st], dtype=np.float64)
    origin = np.asarray(_W["origin"][i]); theta = float(_W["theta"][i])
    R32 = _rotation_matrix(-theta)
    pn32 = (pos32 - origin) @ R32.T
    y_err = float(np.abs(pn32[C.OBS:] - np.asarray(_W["y"][i])).max())
    # 위치 원점 검사: 캐시 origin 은 pos[49] 여야 한다
    y_err = max(y_err, float(np.abs(pos32[C.OBS - 1] - origin).max()))
    pos = pn32.astype(np.float64)
    R = R32.astype(np.float64)

    v_pos = np.linalg.norm(np.gradient(pos, axis=0), axis=1) / C.DT
    v_fld, a_fld = _smooth_speed(np.linalg.norm(vel, axis=1))
    from scipy.signal import savgol_filter
    a_pos = savgol_filter(v_pos, C.SG_WIN, C.SG_POLY, deriv=1, delta=C.DT, mode="interp")
    h, _ = build_heading(pos, h_ref=head - theta)          # 정답 분석이라 110스텝 전체를 쓴다
    h_u = np.unwrap(h)
    # Δh6 는 움직인 스텝(v_fld >= MOVE_V)만 이어 붙여 unwrap 한다 — 저속 구간의 방향 잡음이 누적되면
    # 수백 도가 나온다(실측 425°). h[109] 는 h[108] 복제라 t=49..108 을 쓴다.
    fut_t = np.arange(C.OBS - 1, C.OBS + C.FUT - 1)
    mv_t = fut_t[v_fld[fut_t] >= C.MOVE_V]
    if len(mv_t) >= 2 * C.HEAD_AVG:
        hm = np.unwrap(h[mv_t])
        dh6 = float(np.degrees(hm[-C.HEAD_AVG:].mean() - hm[:C.HEAD_AVG].mean()))
    else:
        dh6 = 0.0
    move6 = float(np.linalg.norm(np.diff(pos[C.OBS - 1:], axis=0), axis=1).sum())
    mv_fut = np.zeros(len(pos), bool)
    mv_fut[C.OBS:] = v_fld[C.OBS:] >= C.MOVE_V

    # 후보 경로 기준 정답 (s, d)
    nd = max(int(_W["n_distinct"][i]), 1)
    K = 6
    meand = np.full(K, np.nan, np.float32); maxd = np.full(K, np.nan, np.float32)
    maxth = np.full(K, np.nan, np.float32)
    inband = np.zeros(K, bool)
    sd = {}
    fu = slice(C.OBS, None)
    for r in range(nd):
        rd = C.route_dict(_W["routes"][i, r], _W["route_tan"][i, r], _W["route_len"][i, r])
        ln_r = float(_W["route_len"][i, r])
        s_r, d_r, _ = lf.to_frame(pos, rd)
        bd = _band_at(np.asarray(_W["route_band"][i, r], np.float64), ln_r, s_r)
        tk = _band_at(np.asarray(_W["route_tan"][i, r], np.float64), ln_r, s_r)
        k_r = np.arctan2(tk[:, 1], tk[:, 0])
        th_r = C.wrap(h - k_r)
        meand[r] = np.abs(d_r[fu]).mean()
        maxd[r] = np.abs(d_r[fu]).max()
        maxth[r] = np.degrees(np.abs(th_r[mv_fut]).max()) if mv_fut.any() else 0.0
        inband[r] = bool(np.all(d_r[fu] <= bd[fu, 0]) and np.all(-d_r[fu] <= bd[fu, 1]))
        sd[r] = (s_r, d_r, bd, k_r)
    # 정답 기준 경로: (밴드 안 & 나란함) > 나란함 > 밴드 안 > 전체 순으로 후보를 좁혀 mean|d| 최소.
    # '밴드 안' 만으로는 부족하다 — 지평 끝에서 꺾이는 경로도 밴드(교차로 ±3.6 m) 안에 들 수 있다.
    rr = np.arange(nd)
    aligned = maxth[:nd] < C.ROUTE_ALIGN_DEG
    ib = inband[:nd]
    for m_, gq in ((ib & aligned, "밴드 안·나란함"), (aligned, "나란함"), (ib, "밴드 안"),
                   (np.ones(nd, bool), "없음")):
        if m_.any():
            cand = rr[m_]
            break
    g = int(cand[np.nanargmin(meand[cand])])
    fallback = bool(_W["route_fallback"][i] > 0.5)
    rw = int(win) % nd
    s_w, d_w, b_w, k_w = sd[rw]
    d_g = sd[g][1]
    dd6 = float(d_g[-5:].mean() - d_g[C.OBS - 3:C.OBS + 2].mean())
    # 차선변경은 차로가 있는(폴백 아님) 나란한 기준 경로에서만 판정한다
    lc_ok = ((not fallback) and bool(aligned[g]) and (maxd[g] <= C.LC_MAX_ABS_D)
             and (move6 >= C.LC_MIN_MOVE))

    # 상황 분류 (우선순위 = C.CLASSES 순서)
    fu = slice(C.OBS - 1, None)
    vmax, amax, amin = float(v_fld[fu].max()), float(a_fld[fu].max()), float(a_fld[fu].min())
    v0f, vend = float(v_fld[C.OBS - 1]), float(v_fld[-1])
    if vmax < C.STOP_VMAX:
        cls = "정지"
    elif dh6 > C.TURN_DEG:
        cls = "좌회전"
    elif dh6 < -C.TURN_DEG:
        cls = "우회전"
    elif lc_ok and abs(dh6) < C.LC_MAX_DEG and dd6 > C.LC_LAT_M:
        cls = "좌차선변경"
    elif lc_ok and abs(dh6) < C.LC_MAX_DEG and dd6 < -C.LC_LAT_M:
        cls = "우차선변경"
    elif amin < C.DEC_A:
        cls = "급감속"
    elif amax > C.ACC_A:
        cls = "급가속"
    elif abs(vend - v0f) < C.CONST_DV and v0f > C.CONST_V0:
        cls = "정속"
    else:
        cls = "기타"

    ldist, lid, ltrack, ltype, n_cand = _lead(scn, pos, h_u, theta, origin.astype(np.float64), R)
    gap49 = float(ldist[C.OBS - 1])
    thw49 = gap49 / v0f if (np.isfinite(gap49) and v0f >= C.THW_MIN_V) else np.nan
    fut_d = ldist[C.OBS:]
    row = {"idx": i, "y_err": y_err, "cls": cls, "dh6": dh6, "dd6": dd6, "vmax_f": vmax,
           "amax_f": amax, "amin_f": amin, "v0_f": v0f, "vend_f": vend,
           "gt_route": g, "gt_route_q": gq, "coverage": bool(inband[:nd].any()),
           "n_inband": int(inband[:nd].sum()), "n_aligned": int(aligned.sum()),
           "gt_route_meand": float(meand[g]), "gt_route_maxd": float(maxd[g]), "move6": move6,
           "lc_ok": bool(lc_ok),
           "has_lead49": bool(np.isfinite(gap49)), "gap49": gap49, "thw49": thw49,
           "lead_stopped49": bool(np.isfinite(gap49) and v0f < C.THW_MIN_V),
           "has_lead_fut": bool(np.isfinite(fut_d).any()),
           "min_fut_gap": float(np.nanmin(fut_d)) if np.isfinite(fut_d).any() else np.nan,
           "lead_type49": ltype, "n_lead_cand": n_cand,
           "gt_s_end_w": float(s_w[-1]), "gt_d_end_w": float(d_w[-1])}
    arr = {"pos": pos.astype(np.float32), "v_pos": v_pos, "v_fld": v_fld, "a_fld": a_fld, "a_pos": a_pos,
           "h": h_u, "lead_dist": ldist, "lead_track": ltrack, "gt_s_w": s_w, "gt_d_w": d_w,
           "gt_band_w": b_w, "gt_k_w": k_w, "gt_d_g": d_g, "route_meand": meand, "route_maxd": maxd,
           "route_maxth": maxth, "route_inband": inband}
    return row, arr


# --------------------------------------------------------------------------- 3. 파생 지표
def derived(P, S, cache_raw):
    N, K = P["prob"].shape
    al = P["alive"]
    n_alive = al.sum(1)
    ar = np.arange(N)
    win, top = S["winner"].astype(int), S["top1"].astype(int)
    end = P["traj"][:, :, -1, :].astype(np.float64)
    spread = np.full(N, np.nan)
    ncl = np.zeros(N, int)
    for i in range(N):
        e = end[i][al[i]]
        if len(e) >= 2:
            dm = np.linalg.norm(e[:, None] - e[None], axis=-1)
            spread[i] = dm[np.triu_indices(len(e), 1)].mean()
        # 끝점 군집 수 — 확률 순서로 2.5 m(DEDUP_M) 안이면 같은 군집
        cen = []
        for m in np.argsort(-P["prob"][i]):
            if not al[i, m]:
                continue
            if all(np.linalg.norm(end[i, m] - c) > 2.5 for c in cen):
                cen.append(end[i, m])
        ncl[i] = len(cen)
    nd = np.maximum(np.asarray(cache_raw["n_distinct"]).astype(int), 1)
    slot_route = np.arange(K)[None, :] % nd[:, None]
    rprob = np.zeros((N, K))
    for r in range(K):
        rprob[:, r] = np.where(slot_route == r, P["prob"], 0.0).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(rprob > 0, rprob * np.log(rprob), 0.0), axis=1)
    jt = np.where(al, P["jit_t_sum"], 0).sum(1) / (n_alive * (C.FUT - 1))
    ja = np.where(al, P["jit_a_sum"], 0).sum(1) / (n_alive * (C.FUT - 1))
    exc = 100.0 * np.where(al, P["exc_cnt"], 0).sum(1) / (n_alive * (C.FUT - 1))
    lanes = np.asarray(cache_raw["lanes"])
    lm = np.asarray(cache_raw["lane_mask"]) > 0
    near = (np.linalg.norm(lanes, axis=-1).min(-1) < 30.0) & lm
    return {
        "top1_ade": P["ade"][ar, top], "top1_fde": P["fde"][ar, top],
        "win_ade": P["ade"][ar, win],
        "win_eq_top1": win == top, "miss": S["minfde"] > C.MISS_M,
        "offlane": np.nansum(np.where(al, P["off_frac"], 0.0), axis=1),
        "jit_theta": jt, "jit_a": ja, "exc_pct": exc, "spread": spread, "n_end_clusters": ncl,
        "n_alive": n_alive, "n_distinct": nd, "win_route": win % nd, "top1_route": top % nd,
        "eff_branches": np.exp(ent), "n_branch_p05": (rprob >= 0.05).sum(1),
        "rprob": rprob, "top1_prob": P["prob"][ar, top], "win_prob": P["prob"][ar, win],
        "win_s_end": P["s"][ar, win, -1], "win_d_end": P["d"][ar, win, -1],
        "n_lanes": lm.sum(1), "n_lanes30": near.sum(1),
        "n_reachable": np.asarray(cache_raw["n_reachable"]),
        "fallback": np.asarray(cache_raw["route_fallback"]) > 0.5,
        "v0": np.asarray(cache_raw["v0"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=C.DEFAULT_TAG)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=32, help="compare_v4_full.py 와 같게 32")
    ap.add_argument("--loader-workers", dest="lw", type=int, default=2)
    ap.add_argument("--workers", type=int, default=14, help="원본 parquet 병렬 (공유 머신 규칙: 16 이하)")
    ap.add_argument("--limit", type=int, default=None, help="스모크용 — 재현 확인은 전체일 때만 한다")
    ap.add_argument("--out-dir", dest="out_dir", default=None, help="기본 viz/v4/<tag>/data")
    ap.add_argument("--tol", type=float, default=2e-3, help="재현 허용 오차 (지표 단위)")
    a = ap.parse_args()
    if a.workers > 16:
        raise SystemExit("workers 는 16 이하 (공유 머신)")
    import pandas as pd
    t_all = time.time()
    dirs = C.tag_dirs(a.tag)
    out = Path(a.out_dir) if a.out_dir else dirs["data"]
    out.mkdir(parents=True, exist_ok=True)
    device = C.pick_device(a.device)

    # ---- 1. 추론 + 재현 확인
    P, S, info, ds = run_inference(a.tag, device, a.batch, a.lw, a.limit)
    got = score_like_compare(P, S)
    exp = C.expected_scores(a.tag)
    repro = {"got": got, "expected": exp, "ok": None}
    print("[repro] 계산: " + "  ".join(f"{k} {v:.4f}" for k, v in got.items() if k != "n"), flush=True)
    if a.limit is None and exp is not None:
        keys = ("minADE6", "minFDE6", "offlane", "dtheta_over_label_pct", "jitter_theta", "jitter_a")
        diffs = {k: abs(got[k] - exp[k]) for k in keys}
        print("[repro] 기준: " + "  ".join(f"{k} {exp[k]:.4f}" for k in keys), flush=True)
        tol = {k: a.tol * (0.01 if k.startswith("jitter") else 1.0) for k in keys}
        repro["ok"] = all(diffs[k] <= tol[k] for k in keys)
        repro["abs_diff"] = diffs
        if not repro["ok"]:
            (out / "meta_failed.json").write_text(json.dumps(repro, indent=2, ensure_ascii=False))
            raise SystemExit(f"재현 실패 — 멈춘다: {diffs}")
        print(f"[repro] 통과 (최대 차 {max(diffs.values()):.2e})", flush=True)
    elif exp is None:
        print("[repro] 기준값이 없다 (v4_full_compare.json 에 이 태그가 없음) — 확인 생략", flush=True)

    sids = ds.sids[:len(S["minade"])]
    N = len(sids)
    cache_raw = {k: ds.raw(k)[:N] for k in ("n_distinct", "n_reachable", "route_fallback", "v0",
                                           "lanes", "lane_mask", "y")}
    D = derived(P, S, cache_raw)

    # ---- 2. 원본 parquet
    t1 = time.time()
    tasks = [(i, sids[i], int(S["winner"][i])) for i in range(N)]
    rows, arrs = [None] * N, [None] * N
    with Pool(a.workers, initializer=_init_worker, initargs=(str(ds.dir),)) as pool:
        for k, (row, arr) in enumerate(pool.imap(raw_one, tasks, chunksize=16)):
            rows[k], arrs[k] = row, arr
            if (k + 1) % 5000 == 0:
                print(f"[raw] {k + 1}/{N}  {time.time() - t1:.0f}s", flush=True)
    print(f"[raw] {N} 시나리오 {time.time() - t1:.0f}s (workers {a.workers})", flush=True)
    R = {k: np.stack([x[k] for x in arrs]).astype(np.float32 if k != "route_inband" else bool)
         for k in arrs[0]}

    # ---- 3. 표
    df = pd.DataFrame(rows)
    df.insert(1, "sid", sids)
    for k, v in S.items():
        df[k] = v
    for k, v in D.items():
        if np.ndim(v) == 1:
            df[k] = v
    df["route_err"] = df["top1_route"] != df["gt_route"]
    df["win_route_err"] = df["win_route"] != df["gt_route"]
    rp = D["rprob"]
    g = df["gt_route"].to_numpy()
    df["gt_route_prob"] = rp[np.arange(N), g]
    df["gt_route_rank"] = 1 + (rp > rp[np.arange(N), g][:, None] + 1e-9).sum(1)
    df["ds_end"] = df["win_s_end"] - df["gt_s_end_w"]
    df["dd_end"] = df["win_d_end"] - df["gt_d_end_w"]
    df["pct_minade"] = df["minade"].rank(pct=True) * 100.0
    print(f"[check] 원본 -> 정규화 정답이 캐시 y 와 다른 최대값 {df['y_err'].max():.2e} m", flush=True)
    if df["y_err"].max() > 1e-3:
        raise SystemExit("원본 정답과 캐시 y 가 다르다 — 인덱스 정렬이 어긋났다")

    df.to_parquet(out / "scenarios.parquet", index=False)
    np.savez(out / "pred.npz", sids=np.array(sids), rprob=rp.astype(np.float32), **P,
             **{k: v for k, v in S.items()})
    np.savez(out / "raw.npz", sids=np.array(sids), y=np.asarray(cache_raw["y"]), **R)
    cls_counts = df["cls"].value_counts().reindex(C.CLASSES).fillna(0).astype(int).to_dict()
    meta = {
        "tag": a.tag, "model": info, "device": device, "val_cache": str(ds.dir), "n": N,
        "git_head": C.git_head(), "seed": C.SEED, "repro": repro,
        "loss_weights": {"offlane": info["offlane"], "smooth": info["smooth"],
                         "hinge": "버려진 모드만" if info["off_nonwinner"] else "살아있는 모든 모드"},
        "thresholds": {k: getattr(C, k) for k in (
            "CLASSES", "STOP_VMAX", "TURN_DEG", "LC_MAX_DEG", "LC_LAT_M", "LC_MAX_ABS_D", "LC_MIN_MOVE",
            "ROUTE_ALIGN_DEG", "MOVE_V", "DEC_A", "ACC_A", "CONST_DV",
            "CONST_V0", "MED_K", "SG_WIN", "SG_POLY", "HEAD_AVG", "LEAD_TYPES", "LEAD_LAT_M", "LEAD_MIN_M",
            "LEAD_MAX_M", "LEAD_HEAD_DEG", "LEAD_EXT_M", "PATH_STEP_M", "THW_MIN_V", "MISS_M", "MIN_N")},
        "class_counts": cls_counts,
        "sec": {"total": time.time() - t_all, "raw": time.time() - t1},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))
    print("[class] " + "  ".join(f"{k} {v}" for k, v in cls_counts.items()), flush=True)
    print(f"[done] {out}  {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
