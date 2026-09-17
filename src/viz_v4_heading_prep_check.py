"""
viz_v4_heading_prep_check.py - viz_v4_heading_prep.py 결과를 **다른 코드 경로**로 다시 계산해 대조한다 (필수 요건 9b).

무엇을 다른 길로 계산하나
  1. focal (a, h) 10 Hz / 2 Hz — pandas 로 parquet 을 직접 읽고, 위치차분·튐 가드·뒤집힘·채우기·평활·뽑기를
     heading_decomp 를 부르지 않고 새로 짠 코드로 만들어 캐시 x 와 비교 (무작위 시나리오 표본).
  2. summary.json 의 주요 숫자 — scan.npz·캐시에서 pandas group-by 로 다시 집계.
     (뒤집힘 갈래는 '재구현 판정' 대신 '캐시 출력에서 본 뒤집힘' 열로 집계해 교차 확인)
  3. 패널 제목 숫자 — 고른 시나리오를 pandas 로 다시 읽어 계산.
  4. 차로 방향 대조 — heading_decomp.lane_field 대신 지도 JSON 의 좌/우 경계 평균으로 중심선을 새로 만들어 판정 일치율.

  python src/viz_v4_heading_prep_check.py [--n 400] [--workers 16]
출력: viz/v4/heading_prep/data/check.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

OUT = C.VIZ_ROOT / "heading_prep"
DATA = OUT / "data"
C10 = C.VAL_CACHES["ah2"]
C2 = C.VAL_CACHES["ah2_2hz"]
AG = Path("/data/argoverse2/cache/v4/agents_hist_val_361221aa2f2e23f3_n24988")
TYPES = {"vehicle": 1, "bus": 2, "motorcyclist": 3, "cyclist": 4}


def wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ 1. 독립 구현
def my_guard(h, lim_deg=30.0):
    out = np.array(h, np.float64)
    lim = np.radians(lim_deg)
    for t in range(1, len(out)):
        a, b = out[t], out[t - 1]
        if np.isfinite(a) and np.isfinite(b) and abs(wrap(a - b)) > lim:
            out[t] = b
    return out


def my_heading(p, ref, dt, guard_deg):
    """위치차분(구간 속력 ≥ 1) → 부족분은 가드된 AV2 (뒤집힘 교정 포함). 전 스텝 관측 가정."""
    d = np.diff(p, axis=0)
    fast = np.hypot(d[:, 0], d[:, 1]) / dt >= 1.0
    h = np.full(len(p), np.nan)
    h[:-1][fast] = np.arctan2(d[fast, 1], d[fast, 0])
    if fast[-1]:
        h[-1] = h[-2]
    g = my_guard(ref, guard_deg) if guard_deg is not None else np.asarray(ref, np.float64)
    m = np.isfinite(h)
    if m.sum() >= 3 and np.median(np.abs(wrap(g[m] - h[m]))) > np.pi / 2:
        g = wrap(g + np.pi)
    src = np.where(m, 1, 2)
    h[~m] = g[~m]
    return h, src


def my_a(p, dt):
    v = np.hypot(*np.diff(p, axis=0).T) / dt
    v = np.r_[v, v[-1]] * 3.6
    return np.diff(v, prepend=0.0) / 3.0


def focal_from_parquet(sid):
    df = pd.read_parquet(C.VAL_DIR / sid / f"scenario_{sid}.parquet")
    fid = df["focal_track_id"].iloc[0]
    f = df[df.track_id == fid].sort_values("timestep")
    pos = f[["position_x", "position_y"]].to_numpy(np.float32)
    head = f["heading"].to_numpy(np.float32)
    vel = f[["velocity_x", "velocity_y"]].to_numpy(np.float32)
    return df, fid, pos, head, vel


def one_focal(args):
    from scipy.signal import savgol_filter
    i, sid = args
    df, fid, pos, head, _ = focal_from_parquet(sid)
    p = pos[:50].astype(np.float64)
    hd = head[:50].astype(np.float64)
    th = float(head[49])
    h, _ = my_heading(p, hd, 0.1, 30.0)
    f10 = np.stack([my_a(p, 0.1), wrap(h - th)], 1)
    idx = np.arange(49, 3, -5)[::-1]
    sm = savgol_filter(p[4:], 5, 2, axis=0, mode="interp")
    q = np.full_like(p, np.nan)
    q[4:] = sm
    q = q[idx]
    h2, _ = my_heading(q, my_guard(hd)[idx], 0.5, None)
    f2 = np.stack([my_a(q, 0.5), wrap(h2 - th)], 1)
    return i, f10, f2


# ------------------------------------------------------------------ 3. 패널 숫자
def panel_one(p, x10, agsrc, ncand):
    df, fid, pos, head, vel = focal_from_parquet(p["sid"])
    d = np.diff(pos[:50].astype(np.float64), axis=0)
    fast = np.hypot(*d.T) / 0.1 >= 1.0
    src = np.r_[np.where(fast, 1, 2), 1 if fast[-1] else 2]
    t49 = df[(df.timestep == 49) & (df.track_id != fid) & df.object_type.isin(list(TYPES))]
    kept = agsrc[(agsrc > 0).any(1)]
    return {"posdiff_pct": float((src == 1).mean() * 100), "av2_pct": float((src == 2).mean() * 100),
            "n_cand_parquet": int(len(t49)), "n_cand_cache": int(ncand),
            "never_moved": int(((kept == 1).sum(1) == 0).sum())}


# ------------------------------------------------------------------ 4. 차로 방향 (독립 중심선)
def resample(pl, n=40):
    pl = np.asarray(pl, np.float64)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
    if s[-1] <= 0:
        return np.repeat(pl[:1], n, 0)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, pl[:, 0]), np.interp(u, s, pl[:, 1])], 1)


def lane_one(args):
    i, sid, rows = args
    js = json.loads((C.VAL_DIR / sid / f"log_map_archive_{sid}.json").read_text())
    P, A, I = [], [], []
    for ln in js["lane_segments"].values():
        lb = [(q["x"], q["y"]) for q in ln["left_lane_boundary"]]
        rb = [(q["x"], q["y"]) for q in ln["right_lane_boundary"]]
        c = (resample(lb) + resample(rb)) / 2
        L = float(np.hypot(*np.diff(c, axis=0).T).sum())
        c = resample(c, max(2, int(L / 0.25) + 1))
        g = np.gradient(c, axis=0)
        P.append(c)
        A.append(np.arctan2(g[:, 1], g[:, 0]))
        I.append(np.full(len(c), bool(ln["is_intersection"])))
    P, A, I = np.concatenate(P), np.concatenate(A), np.concatenate(I)
    df, fid, pos, head, _ = focal_from_parquet(sid)
    th = float(head[49])
    t49 = df[(df.timestep == 49) & (df.track_id != fid) & df.object_type.isin(list(TYPES))].copy()
    org = pos[49]
    t49["d"] = np.hypot(t49.position_x.to_numpy(np.float32) - org[0], t49.position_y.to_numpy(np.float32) - org[1])
    t49 = t49.sort_values("d", kind="mergesort").head(32).reset_index(drop=True)
    ah = np.load(AG / "h.npy", mmap_mode="r")[i]
    out = []
    for slot, lane_m, ag_av2, ag_10 in rows:
        r = t49.iloc[int(slot)]
        dd = (P[:, 0] - r.position_x) ** 2 + (P[:, 1] - r.position_y) ** 2
        j = int(dd.argmin())
        m = int(dd[j] <= 1.0 and not I[j])
        a = int(np.cos(r.heading - A[j]) > 0) if m else -1
        b = int(np.cos(float(ah[int(slot), 49]) + th - A[j]) > 0) if m else -1
        out.append((int(lane_m), m, int(ag_av2), a, int(ag_10), b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    import viz_v4_heading_prep as V
    S = V.Scan(DATA)
    summ = json.loads((DATA / "summary.json").read_text())
    rep = {}
    rng = np.random.default_rng(1)
    pick = np.sort(rng.choice(S.n, size=min(a.n, S.n), replace=False))

    # ---- 1
    with Pool(a.workers) as pool:
        res = pool.map(one_focal, [(int(i), S.sids[i]) for i in pick], chunksize=8)
    e = np.array([[np.abs(f10[:, 0] - S.x10[i, :, 0]).max(), np.abs(wrap(f10[:, 1] - S.x10[i, :, 1])).max(),
                   np.abs(f2[:, 0] - S.x2[i, :, 0]).max(), np.abs(wrap(f2[:, 1] - S.x2[i, :, 1])).max()]
                  for i, f10, f2 in res])
    rep["1_focal_reimplementation"] = {"n": len(res), "a10_max": float(e[:, 0].max()), "h10_max_rad": float(e[:, 1].max()),
                                       "a2_max": float(e[:, 2].max()), "h2_max_rad": float(e[:, 3].max()),
                                       "n_h10_gt_1e-4": int((e[:, 1] > 1e-4).sum()), "n_h2_gt_1e-4": int((e[:, 3] > 1e-4).sum())}

    # ---- 2. 요약 숫자 (pandas)
    chk = {}

    def cmp(name, mine, theirs, tol=1e-6):
        chk[name] = {"recalc": float(mine), "summary": float(theirs), "abs_diff": float(abs(mine - theirs)),
                     "ok": bool(abs(mine - theirs) <= tol)}

    h10 = pd.DataFrame(S.x10[:, :, 1])
    h2 = pd.DataFrame(S.x2[:, :, 1])
    cmp("jump10_pct", (h10.diff(axis=1).abs() > np.pi).any(axis=1).mean() * 100, summ["jump"]["10hz_scen_pct"])
    cmp("jump2_pct", (h2.diff(axis=1).abs() > np.pi).any(axis=1).mean() * 100, summ["jump"]["2hz_scen_pct"])
    # 방향 변화 (위치차분 구간) — long 표로
    s10 = pd.DataFrame(S.src10)
    d10 = h10.iloc[:, :49].diff(axis=1).iloc[:, 1:]
    ok10 = (s10.iloc[:, :48].to_numpy() == 1) & (s10.iloc[:, 1:49].to_numpy() == 1)
    r10 = np.degrees(np.abs(wrap(d10.to_numpy()))) / 0.1
    cmp("rate10_mv_p50", np.quantile(r10[ok10], 0.5), summ["turn_rate_deg_s"]["mv_10"]["p50"], 1e-6)
    cmp("rate10_mv_p99", np.quantile(r10[ok10], 0.99), summ["turn_rate_deg_s"]["mv_10"]["p99"], 1e-6)
    d2 = h2.iloc[:, :9].diff(axis=1).iloc[:, 1:].to_numpy()
    ok2 = (S.src2[:, :8] == 1) & (S.src2[:, 1:9] == 1)
    r2 = np.degrees(np.abs(wrap(d2))) / 0.5
    cmp("rate2_mv_p50", np.quantile(r2[ok2], 0.5), summ["turn_rate_deg_s"]["mv_2"]["p50"], 1e-6)
    cmp("rate2_mv_p99", np.quantile(r2[ok2], 0.99), summ["turn_rate_deg_s"]["mv_2"]["p99"], 1e-6)
    long = pd.DataFrame({"slip": np.degrees(wrap(S.head_n[:, :49].astype(np.float64) - S.x10[:, :49, 1])).ravel(),
                         "src": S.src10[:, :49].ravel()})
    ps = long[long.src == 1].slip.abs()
    cmp("slip_abs_p50", ps.quantile(0.5), summ["slip"]["abs"]["p50"], 1e-6)
    cmp("slip_abs_p99", ps.quantile(0.99), summ["slip"]["abs"]["p99"], 1e-6)
    cmp("slip_gt90_pct", (ps > 90).mean() * 100, summ["slip"]["gt90_pct"])
    v10 = S.x10[:, :, 0].cumsum(1) * 3 / 3.6
    ref = v10[:, 10:16].mean(1)
    pop = ref > 3
    cmp("ramp_n", pop.sum(), summ["ramp"]["n"], 0)
    cmp("ramp_ratio_t0", np.median(v10[pop, 0] / ref[pop]), summ["ramp"]["ratio10_median_t0_9"][0])
    cmp("ramp_first_lt_0.8_pct", (v10[pop, 0] / ref[pop] < 0.8).mean() * 100, summ["ramp"]["first10_lt_0.8_pct"])
    v2 = S.x2[:, :, 0].cumsum(1) * 3 / 3.6
    cmp("ramp_2hz_seg0_p50", np.median(v2[pop, 0] / ref[pop]), summ["ramp"]["2hz_first_seg_ratio"]["p50"])
    cmp("recon_10hz_av2_mean", S.rec[:, 0].astype(np.float64).mean(), summ["recon_final_err_m"]["10hz_av2"]["mean"])
    # 주변 차량 출처 — 캐시에서 직접, 그리고 캐시 meta 와
    hs = np.load(AG / "h_src.npy")
    at = np.load(AG / "atype.npy")
    kept = at > 0
    sk = hs[kept]
    meta = json.loads((AG / "meta.json").read_text())["stats"]["step_src_pct_of_kept_steps"]
    for code, key, mk in ((0, "미관측", "none"), (1, "위치차분", "posdiff"), (2, "AV2 보조", "av2")):
        v = (sk == code).mean() * 100
        cmp(f"agents_src_{mk}_pct", v, summ["agents_src"]["pct_of_all_kept_steps"][key])
        cmp(f"agents_src_{mk}_pct_vs_cache_meta", v, meta[mk])
    T = pd.DataFrame(S.AGT, columns=V.AG_COLS)
    tb = pd.cut(T.vmax, [0, 0.5, 2, 5, np.inf], right=False, labels=["<0.5", "0.5–2", "2–5", "≥5"])
    share = T.groupby(tb, observed=False).n_src2.sum() / T.n_src2.sum() * 100
    for lab in ["<0.5", "0.5–2", "2–5", "≥5"]:
        cmp(f"av2_share_trackvmax_{lab}", share[lab], summ["agents_src"]["by_track_vmax"][lab]["share_of_all_av2_steps"], 1e-6)
    cmp("agents_stationary_pct", (T.vmax < 0.5).mean() * 100, summ["agents_moved"]["stationary_pct"])
    cmp("agents_never_posdiff_pct", (T.n_src1 == 0).mean() * 100, summ["agents_moved"]["never_posdiff_pct"])
    cmp("agents_full_obs_pct", (T.n_obs == 50).mean() * 100, summ["agents_moved"]["full_obs_pct"])
    ver = T[T.flip >= 0]
    cmp("flip10_pct_verifiable", (ver.flip == 1).mean() * 100, summ["agents_flip"]["all"]["flip10_pct"])
    # 뒤집힘 갈래 스텝 수 — '출력에서 본 뒤집힘' 열로 (판정 재구현과 다른 경로)
    lt = summ["logic_tree"]["agents_10hz"]
    cmp("tree_av2_flip1_steps_by_output", T[T.flip_obs == 1].n_src2.sum(), lt["av2_flip1"], 0)
    cmp("tree_av2_notflipped_steps_by_output", T[T.flip_obs == 0].n_src2.sum(), lt["av2_flip0"] + lt["av2_undecided"], 0)
    full = T[T.full_obs == 1]
    lt2 = summ["logic_tree"]["agents_2hz"]
    cmp("tree2_av2_flip1_samples_by_output", full[full.flip2_obs == 1].n2_av2.sum(), lt2["av2_flip1"], 0)
    cmp("tree2_posdiff_samples", full.n2_src1.sum(), lt2["posdiff"], 0)
    st = T[(T.vmax < 0.5) & (T.lane_m == 1) & (T.flip_obs == 1)]
    sl = summ["agents_lane_check"]["stationary_flipped10"]
    cmp("stationary_flipped_av2_same_pct", (st.ag_av2 == 1).mean() * 100, sl["av2_same_pct"])
    cmp("stationary_flipped_h10_same_pct", (st.ag_10 == 1).mean() * 100, sl["h10_same_pct"])
    ff = S.fflip.astype(int)
    lf = summ["logic_tree"]["focal_10hz"]
    av2f = (S.src10 == 2).sum(1)
    cmp("focal_tree_flip1_steps_by_output", av2f[ff[:, 1] == 1].sum(), lf["av2_flip1"], 0)
    # 상태별 (group-by)
    cls = np.load(DATA / "state.npy", allow_pickle=True)
    g = pd.DataFrame({"cls": cls, "av2": (S.src10 == 2).mean(1), "j10": (h10.diff(axis=1).abs() > np.pi).any(axis=1)})
    agg = g.groupby("cls").agg(n=("av2", "size"), av2=("av2", "mean"), j10=("j10", "mean"))
    for c, row in summ["by_state"]["rows"].items():
        if row["n"]:
            cmp(f"state_{c}_n", agg.loc[c, "n"], row["n"], 0)
            cmp(f"state_{c}_av2_pct", agg.loc[c, "av2"] * 100, row["av2_10_pct"])
            cmp(f"state_{c}_jump10_pct", agg.loc[c, "j10"] * 100, row["jump10_pct"])
    # 검토 뒤 추가된 값 — 같은 트랙 비교, 정지·뒤집힘의 차로 안 비율, 트리 B AUC (잎 group-by 확률로)
    sm = summ["logic_tree"]["agents_same_tracks_full"]
    fo = T[T.n_obs == 50]
    cmp("same_tracks_judged10_pct", fo[fo.flip >= 0].n_src2.sum() / (len(fo) * 50) * 100, sm["judged10_pct"])
    cmp("same_tracks_flip10_pct", fo[fo.flip_obs == 1].n_src2.sum() / (len(fo) * 50) * 100, sm["flip10_pct"])
    cmp("same_tracks_flip2_pct", fo[fo.flip2_obs == 1].n2_av2.sum() / (len(fo) * 10) * 100, sm["flip2_pct"])
    sf = T[(T.vmax < 0.5) & (T.flip_obs == 1)]
    cmp("stationary_flipped_in_lane_pct", (sf.lane_m == 1).mean() * 100,
        summ["logic_tree"]["stationary_flipped10"]["in_lane_pct"])
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    rngc = np.random.default_rng(C.SEED)
    te = (rngc.random(S.n) < 0.3)[T.scen.astype(int).to_numpy()]
    lm = (T.lane_m == 1).to_numpy()
    cols = ["vmax", "path", "n_src1", "n_src1_slow", "n_obs", "dist0", "atype"]
    X = T[cols].to_numpy()
    y = (T.ag_10 == 0).to_numpy().astype(int)
    est = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=C.SEED).fit(X[lm & ~te], y[lm & ~te])
    lf = pd.DataFrame({"leaf": est.apply(X), "y": y})
    rate = lf[lm & ~te].groupby("leaf").y.mean()               # 잎 통계 (학습 분할)
    pt = lf[lm & te].leaf.map(rate).to_numpy()
    yt = y[lm & te]
    b = summ["cart"]["B_lane_opposite"]
    cmp("treeB_auc_test_groupby", roc_auc_score(yt, pt), b["auc_test"], 1e-9)
    bs = np.mean((pt - yt) ** 2)
    ref_b = np.mean((y[lm & ~te].mean() - yt) ** 2)
    cmp("treeB_brier_skill_groupby", 1 - bs / ref_b, b["brier_skill_test"], 1e-9)
    cmp("treeB_majority_test", max(yt.mean(), 1 - yt.mean()), b["acc_majority_baseline_test"], 1e-12)
    rep["2_summary_recalc"] = {"note": "대부분 같은 scan.npz 를 다른 코드(pandas group-by)로 다시 집계한 것이다 — 원본부터의 독립 재현은 1·4·(감독 세션 검토) 가 맡는다",
                               "n_checks": len(chk), "n_fail": sum(not v["ok"] for v in chk.values()),
                               "fails": {k: v for k, v in chk.items() if not v["ok"]}, "all": chk}

    # ---- 3. 패널 숫자
    picks = json.loads((DATA / "picks.json").read_text())["picks"]
    facts = {f["n"]: f for f in json.loads((DATA / "panel_facts.json").read_text())}
    pr = []
    for p in picks:
        i = p["idx"]
        r = panel_one(p, S.x10[i], np.load(AG / "h_src.npy", mmap_mode="r")[i], S.ag["n_cand"][i])
        fx = facts[p["n"]]
        pr.append({"n": p["n"], "sid": p["sid"],
                   "posdiff_pct_ok": abs(r["posdiff_pct"] - fx["src10_pct"]["위치차분"]) < 0.051,
                   "n_cand_ok": r["n_cand_parquet"] == r["n_cand_cache"] == fx["n_cand"],
                   "never_moved_ok": r["never_moved"] == fx["never_moved"], **r})
    rep["3_panel_facts"] = {"all_ok": all(x["posdiff_pct_ok"] and x["n_cand_ok"] and x["never_moved_ok"] for x in pr),
                            "rows": pr}

    # ---- 4. 차로 방향 (독립 중심선)
    lp = pick[: min(len(pick), 300)]
    tasks = []
    for i in lp:
        rr = T[T.scen == i][["slot", "lane_m", "ag_av2", "ag_10"]].to_numpy(int)
        tasks.append((int(i), S.sids[i], [tuple(r) for r in rr]))
    with Pool(a.workers) as pool:
        out = [x for part in pool.map(lane_one, tasks, chunksize=4) for x in part]
    o = np.array(out)
    both = (o[:, 0] == 1) & (o[:, 1] == 1)
    rep["4_lane_independent_centerline"] = {
        "n_agents": int(len(o)), "lane_match_agree_pct": float((o[:, 0] == o[:, 1]).mean() * 100),
        "n_both_in_lane": int(both.sum()),
        "ag_av2_agree_pct": float((o[both, 2] == o[both, 3]).mean() * 100),
        "ag_10_agree_pct": float((o[both, 4] == o[both, 5]).mean() * 100),
        "opp_av2_pct_mine_vs_indep": [float((o[both, 2] == 0).mean() * 100), float((o[both, 3] == 0).mean() * 100)],
        "opp_10_pct_mine_vs_indep": [float((o[both, 4] == 0).mean() * 100), float((o[both, 5] == 0).mean() * 100)],
        "note": "독립 중심선 = 지도 JSON 좌/우 경계를 40점으로 맞춰 평균, 0.25 m 간격 재표본. 원 계산은 LaneGraph 중심선 + 0.5 m 간격"}
    (DATA / "check.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: (v if k != "2_summary_recalc" else {kk: vv for kk, vv in v.items() if kk != "all"})
                      for k, v in rep.items()}, indent=1, ensure_ascii=False, default=float)[:6000])


if __name__ == "__main__":
    main()
