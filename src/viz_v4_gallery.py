"""
viz_v4_gallery.py - v4 모델이 실제로 낸 예측 궤적을 어두운 HD 맵 위에 그린 '평균 사례' 3×3 갤러리.

고르는 규칙 (덤프 scenarios.parquet 의 시나리오별 minADE6 · minFDE6, val 24,988 전체)
  평균값 기준 (주 그림)  거리 = |ADE − 평균ADE| / σADE + |FDE − 평균FDE| / σFDE 가 가장 작은 9개.
                        평균·표준편차(ddof=0)는 전체에서 구한다. 폴백(경로 없음) 시나리오도 빼지 않는다.
                        칸 순서 = 거리 순.
  중앙값 기준 (참고)     minFDE6 오름차순(안정 정렬)의 가운데 9개 = 정렬 위치 mid−4 … mid+4 (mid = N // 2).
                        visualize_map.py 의 AVERAGE 그림과 같은 규칙이다. 다만 예전 그림
                        (pred_lanerules_avg.png · prediction_map_avg.png)은 val 앞 300개(--pool 기본값)에서
                        뽑았다 — traj_lane_rules.npz 로 9칸 minADE 가 순서까지 재현된다. 같은 모집단은 --median-pool 300.

칸 하나 = 지도(focal 정규화: t=0 위치가 원점, 진행방향 +x) + 과거 5초 + 정답 6초 + v4 예측 6모드.
  best of 6 = 끝점 오차 최소 모드(학습 WTA 의 승자와 같다). 나머지는 확률로 진하기·굵기. ▲ = 확률 1위 끝점.
  예측선은 현재 위치(원점)에서 시작하게 t=0 점을 앞에 붙여 그린다(모델 출력은 t=0.1 s 부터).
  선택: 후보 경로 중심선을 흐리게(--routes 1), v3 규칙 모델 예측을 흐린 점선으로(--v3 auto).

v3 겹치기 (--v3 auto)
  runs/lstm_lane_rules_s0.pth (raw5 5채널 · 차선+규칙 30) 를 현재 dataset_lane 으로 돌린다.
  먼저 val 앞 64개(배치 64)를 runs/traj_lane_rules.npz (학습 직후 GPU 로 뽑은 예측)와 대조해 같을 때만 겹친다.
  GPU 전용이다 — 같은 가중치·입력인데 CPU 는 최대 0.23 m 달랐다(2026-09-17, val 앞 64개). 학습이 돌면 겹치지 않는다.

비교 그림 (--compare <tag2>)
  --tag 판의 평균값 기준 9개와 같은 시나리오에서 두 판의 best of 6 과 확률 1위만 나란히 그린다.

  python src/viz_v4_gallery.py                                    # 주 모델, 평균값·중앙값 두 장
  python src/viz_v4_gallery.py --tag v4_l4nw_ah2_2hz_full_sm1_s0
  python src/viz_v4_gallery.py --rule none --compare v4_l4nw_ah2_2hz_full_sm1_s0
출력: viz/v4/<tag>/gallery/avg_mean.png · avg_median.png · compare_<a>_vs_<b>_avg_mean.png
      viz/v4/<tag>/data/gallery_picks.json (고른 ID·지표·검증·칸별 사실)
덤프가 없는 판은 먼저 표준 작업 A 1단계(python src/viz_v4_dump.py --tag <tag>)를 돌린다.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

N_PICK = 9
MIN_SPAN_M = 40.0          # 칸 시야 최소 폭
V3_TAG = "lane_rules_s0"
V3_REF = C.RUNS / "traj_lane_rules.npz"
V3_N_CHECK = 64
V3_TOL_M = 1e-3
P_FULL = 0.5               # 이 확률 이상이면 가장 진하고 굵게

# ---------------------------------------------------------------- 색 (어두운 지도)
# hdmap_render 색 계열을 한 단계 어둡게 두고, 흰 노면표시는 회색으로 낮춘다 — 흰 과거 궤적과 겹쳐 보이지 않게.
D_BG, D_DRIVE, D_LANE, D_LANE_INT = "#111215", "#24262b", "#2d2f35", "#35373e"
D_BIKE, D_CURB = "#27352f", "#41444c"
D_MARK_W, D_MARK_Y, D_MARK_B, D_CROSS = "#8d9097", "#a98f3d", "#4f86b3", "#8e9199"
TXT, TXT2, TXT3 = "#eceef2", "#b9bec7", "#8a9099"
PANEL_EDGE, LEG_BG = "#3a3d44", "#1b1d22"
OUTLINE = "#07080a"
C_PAST = "#f5f5f5"
C_GT = "#35a2ff"
C_BEST = "#62f08e"
C_OTHER = "#ffa53d"
C_TOP = "#ffe14d"
C_FOCAL = "#ff5c33"
C_V3 = "#e3909b"
C_ROUTE = "#9cc9f0"
C_A, C_B = "#62f08e", "#ff6fd2"      # 비교 그림: 첫 판(실선) · 둘째 판(점선)
LS_B = (0, (4.0, 2.2))

INPUT_DESC = {"ah2": "(a, h) 10 Hz 입력", "ah2_2hz": "(a, h) 2 Hz 입력", "raw5": "raw5 5채널 입력"}
SHORT = {"ah2": "10 Hz", "ah2_2hz": "2 Hz", "raw5": "raw5"}
FILE_LAB = {"ah2": "10hz", "ah2_2hz": "2hz", "raw5": "raw5"}
FULL_TRAIN = 199908
PAST_VIEW_S = 2.0          # 시야 계산에 넣는 과거 길이 [s] — 5 초 전부 넣으면 고속 칸에서 예측이 너무 작아진다


def fmt_w(x):
    """가중치 표기: 1 -> '1.0', 0.1 -> '0.1', 0.25 -> '0.25'."""
    s = f"{float(x or 0.0):.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def fmt_deg(x):
    v = int(round(float(x)))
    return "0°" if v == 0 else f"{v:+d}°"


# ---------------------------------------------------------------- 데이터
class Run:
    """덤프 한 판(--tag)의 표·배열·캐시 경로. 덤프를 다시 계산하지 않는다."""

    def __init__(self, tag):
        import pandas as pd
        from dataset_cached import CachedV4Dataset
        self.tag = tag
        self.dirs = C.tag_dirs(tag)
        d = self.dirs["data"]
        miss = [n for n in ("scenarios.parquet", "pred.npz", "raw.npz", "meta.json") if not (d / n).exists()]
        if miss:
            raise SystemExit(f"[gallery] {tag} 의 추론 덤프가 없다 ({', '.join(miss)} 없음 — {d}).\n"
                             f"  먼저 표준 작업 A 1단계를 돌린다:  python src/viz_v4_dump.py --tag {tag}")
        self.df = pd.read_parquet(d / "scenarios.parquet")
        self.meta = json.loads((d / "meta.json").read_text())
        pr = np.load(d / "pred.npz")
        self.traj, self.prob, self.alive = pr["traj"], pr["prob"], pr["alive"]
        self.off_frac, self.rprob, self.v = pr["off_frac"], pr["rprob"], pr["v"]
        rw = np.load(d / "raw.npz")
        self.pos, self.y = rw["pos"], rw["y"]
        sids = self.df["sid"].to_numpy()
        if not (np.array_equal(pr["sids"], sids) and np.array_equal(rw["sids"], sids)):
            raise SystemExit(f"[gallery] {tag}: pred/raw/표의 시나리오 순서가 다르다")
        self.args, self.args_src = C.run_config(tag)
        self.inp = self.args.get("input", self.meta["model"].get("input", "raw5"))
        cache = CachedV4Dataset(self.meta["val_cache"])
        if list(cache.sids[:len(sids)]) != list(sids):
            raise SystemExit(f"[gallery] {tag}: 캐시 {cache.dir} 의 순서가 덤프와 다르다")
        self.cache = {k: cache.raw(k) for k in ("routes", "route_mask", "origin", "theta")}
        self._scene = {}

    @property
    def short(self):
        return SHORT.get(self.inp, self.tag)

    def desc(self):
        a = self.args
        lvl = str(a.get("level", "l3")).upper()
        if a.get("level") == "l0" and float(a.get("offlane", 0.0)) > 0:
            lvl = "L4" + ("nw" if a.get("off_nonwinner") else "")
        n = int(a.get("limit") or 0)
        data = "전체 데이터" if n >= FULL_TRAIN else f"학습 {n:,}개"
        return (f"{lvl} · {INPUT_DESC.get(self.inp, self.inp)} · {data} · "
                f"흔들림 벌점 {fmt_w(a.get('smooth'))}")

    def scene(self, i):
        """지도 + 교차로 판정 (교차로 차로 면 안에 t=0 위치 / 정답 6초 경로가 드는가)."""
        if i not in self._scene:
            import hdmap_render as hr
            from matplotlib.path import Path as MPath
            sid = self.df["sid"].iloc[i]
            sc = C.build_scene(sid, self.cache["origin"][i], self.cache["theta"][i])
            polys = [MPath(p) for p, col in sc.lanes if col == hr.ASPHALT_INT]
            y = np.asarray(self.y[i], np.float64)
            now = any(p.contains_point((0.0, 0.0)) for p in polys)
            fut = any(p.contains_points(y).any() for p in polys)
            self._scene[i] = (sc, {"now": bool(now), "fut": bool(fut)})
        return self._scene[i]


def pick_mean(df, k=N_PICK):
    A = df["minade"].to_numpy(np.float64)
    F = df["minfde"].to_numpy(np.float64)
    st = {"n": int(len(df)), "mean_ade": float(A.mean()), "sd_ade": float(A.std()),
          "mean_fde": float(F.mean()), "sd_fde": float(F.std()),
          "median_ade": float(np.median(A)), "median_fde": float(np.median(F)), "sd_ddof": 0}
    dist = np.abs(A - st["mean_ade"]) / st["sd_ade"] + np.abs(F - st["mean_fde"]) / st["sd_fde"]
    idx = np.argsort(dist, kind="mergesort")[:k]
    return [int(i) for i in idx], dist, st


def pick_median(df, k=N_PICK, pool=0):
    F = df["minfde"].to_numpy(np.float64)
    A = df["minade"].to_numpy(np.float64)
    n = len(F) if pool <= 0 else min(pool, len(F))
    o = np.argsort(F[:n], kind="mergesort")        # visualize_map 의 list.sort 와 같은 안정 정렬
    mid = n // 2
    lo = mid - k // 2
    idx = [int(i) for i in o[lo:lo + k]]
    return idx, {"pool": n, "mid_pos0": int(mid), "sorted_pos0": [int(lo), int(lo + k - 1)],
                 "rank_from1": [int(lo + 1), int(lo + k)],
                 "median_fde": float(np.median(F[:n])), "median_ade": float(np.median(A[:n])),
                 "mean_fde": float(F[:n].mean()), "mean_ade": float(A[:n].mean())}


def path_len(xy):
    p = np.vstack([np.zeros((1, 2)), np.asarray(xy, np.float64)])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def end_err(pe_, y):
    """정답 끝 진행방향 기준 끝점 오차 [m] (종: + 앞섬 / 횡: + 좌). 방향은 정답 마지막 1초 변위."""
    y = np.asarray(y, np.float64)
    t = y[-1] - y[-11]
    if np.linalg.norm(t) < 0.5:
        t = y[-1].copy()
    nt = np.linalg.norm(t)
    if nt < 0.5:
        return float("nan"), float("nan")
    t = t / nt
    e = np.asarray(pe_, np.float64) - y[-1]
    return float(e @ t), float(t[0] * e[1] - t[1] * e[0])


def facts(run, i, inter):
    """칸 설명에 쓰는 사실 (전부 덤프 배열에서 직접 계산)."""
    r = run.df.iloc[i]
    al = run.alive[i]
    prob = run.prob[i].astype(np.float64)
    tr = run.traj[i].astype(np.float64)
    y = np.asarray(run.y[i], np.float64)
    win, top = int(r["winner"]), int(r["top1"])
    nd = int(r["n_distinct"])
    n_alive = int(al.sum())
    off = al & (np.nan_to_num(run.off_frac[i]) > 0)
    d = np.linalg.norm(tr - y[None], axis=-1)
    Lg = path_len(y)
    out = {
        "idx": int(i), "sid": r["sid"], "cls": r["cls"], "dh6_deg": round(float(r["dh6"]), 2),
        "inter_now": inter["now"], "inter_fut": inter["fut"],
        "n_distinct": nd, "fallback": bool(r["fallback"]), "n_alive": n_alive,
        "minade": round(float(r["minade"]), 4), "minfde": round(float(r["minfde"]), 4),
        "winner": win, "top1": top, "win_eq_top1": bool(win == top),
        "top1_prob": round(float(prob[top]), 3), "win_prob": round(float(prob[win]), 3),
        "probs_desc": [round(float(p), 3) for p in sorted(prob[al], reverse=True)],
        "mode_route": [int(m % max(nd, 1)) for m in range(6)],
        "route_prob": [round(float(p), 3) for p in run.rprob[i][:nd]],
        "gt_route": int(r["gt_route"]), "gt_route_q": r["gt_route_q"],
        "top1_route": int(r["top1_route"]), "win_route": int(r["win_route"]),
        "route_ok_top1": bool(not r["route_err"]), "route_ok_win": bool(not r["win_route_err"]),
        "gt_route_prob": round(float(r["gt_route_prob"]), 3), "gt_route_rank": int(r["gt_route_rank"]),
        "eff_branches": round(float(r["eff_branches"]), 2),
        "off_modes": int(off.sum()), "off_mode_ids": [int(m) for m in np.where(off)[0]],
        "top1_ade": round(float(d[top].mean()), 3), "top1_fde": round(float(d[top, -1]), 3),
        "len_gt": round(Lg, 2), "len_best": round(path_len(tr[win]), 2), "len_top1": round(path_len(tr[top]), 2),
        "v0": round(float(r["v0"]), 2), "v0_f": round(float(r["v0_f"]), 2), "vend_f": round(float(r["vend_f"]), 2),
        "amin_f": round(float(r["amin_f"]), 2), "amax_f": round(float(r["amax_f"]), 2),
        "v_end_best": round(float(run.v[i, win, -1]), 2), "v_end_top1": round(float(run.v[i, top, -1]), 2),
        "ds_end_best_frenet": round(float(r["ds_end"]), 2), "dd_end_best_frenet": round(float(r["dd_end"]), 2),
        "spread": round(float(r["spread"]), 2) if np.isfinite(r["spread"]) else None,
    }
    for key, m in (("best", win), ("top1", top)):
        lo, la = end_err(tr[m, -1], y)
        out[f"end_lon_{key}"] = round(lo, 2) if np.isfinite(lo) else None
        out[f"end_lat_{key}"] = round(la, 2) if np.isfinite(la) else None
    out["summary"] = summary_text(out)
    return out


def summary_text(f):
    """사실을 한 줄로 (해석 없이 숫자만)."""
    if f["fallback"]:
        route = "경로 없음(직진 폴백, 모드 1개)"
    elif f["n_distinct"] == 1:
        route = "분기 1개(6모드가 같은 경로·속도로 갈림)"
    else:
        route = (f"분기 {f['n_distinct']} · 정답 경로 r{f['gt_route']}({f['gt_route_q']}) 확률 {f['gt_route_prob']:.2f}"
                 f"(순위 {f['gt_route_rank']}) · 1위 경로 r{f['top1_route']} {'맞음' if f['route_ok_top1'] else '틀림'}"
                 f" · best 경로 r{f['win_route']} {'맞음' if f['route_ok_win'] else '틀림'}")
    sp = (f"6초 이동 정답 {f['len_gt']:.1f} · best {f['len_best']:.1f}({f['len_best'] - f['len_gt']:+.1f})"
          f" · 1위 {f['len_top1']:.1f}({f['len_top1'] - f['len_gt']:+.1f}) m")
    lon = (f" · 끝점 종/횡 best {f['end_lon_best']:+.1f}/{f['end_lat_best']:+.1f}, 1위 {f['end_lon_top1']:+.1f}/"
           f"{f['end_lat_top1']:+.1f} m" if f["end_lon_best"] is not None and f["end_lon_top1"] is not None else "")
    best = " = best" if f["win_eq_top1"] else " · best m{} {:.2f}".format(f["winner"], f["win_prob"])
    top3 = "/".join("{:.2f}".format(p) for p in f["probs_desc"][:3])
    pr = f"확률 1위 m{f['top1']} {f['top1_prob']:.2f}{best} · 상위 {top3}"
    return f"{route} | {sp}{lon} | {pr} | 이탈 {f['off_modes']}/{f['n_alive']}"


def metric_check(run, idxs):
    """그림에 쓴 궤적으로 minADE6/minFDE6 를 다시 계산해 표 값과 비교한다."""
    rows, worst = [], 0.0
    for i in idxs:
        d = np.linalg.norm(run.traj[i].astype(np.float64) - np.asarray(run.y[i], np.float64)[None], axis=-1)
        al = run.alive[i]
        ade, fde = float(d.mean(1)[al].min()), float(d[al, -1].min())
        ta, tf = float(run.df["minade"].iloc[i]), float(run.df["minfde"].iloc[i])
        same_txt = f"{ade:.2f}" == f"{ta:.2f}" and f"{fde:.2f}" == f"{tf:.2f}"
        worst = max(worst, abs(ade - ta), abs(fde - tf))
        rows.append({"idx": int(i), "ade_recalc": ade, "fde_recalc": fde, "ade_table": ta, "fde_table": tf,
                     "title_text_same": bool(same_txt)})
    return {"max_abs_diff_m": worst, "all_title_text_same": all(r["title_text_same"] for r in rows),
            "ok": bool(worst < 1e-4 and all(r["title_text_same"] for r in rows)), "rows": rows}


def coord_check(run, idxs):
    """과거 끝점 = 원점인가, 예측 첫 점이 원점 근처(정답 첫 점 근처)에서 시작하는가."""
    past_end = max(float(np.abs(run.pos[i, C.OBS - 1]).max()) for i in idxs)
    gt_join = max(float(np.abs(run.pos[i, C.OBS:] - run.y[i]).max()) for i in idxs)
    rows = []
    for i in idxs:
        al = run.alive[i]
        p0 = run.traj[i][al, 0].astype(np.float64)
        y0 = np.asarray(run.y[i, 0], np.float64)
        w = int(run.df["winner"].iloc[i])
        rows.append({"idx": int(i), "gt_first_step_m": round(float(np.linalg.norm(y0)), 3),
                     "pred_first_step_m_max": round(float(np.linalg.norm(p0, axis=1).max()), 3),
                     "pred_first_vs_gt_first_m_max": round(float(np.linalg.norm(p0 - y0, axis=1).max()), 3),
                     "best_first_vs_gt_first_m": round(float(np.linalg.norm(run.traj[i, w, 0] - y0)), 3)})
    return {"past_end_abs_max_m": past_end, "pos_future_vs_y_max_m": gt_join,
            "pred_first_vs_gt_first_m_max": max(r["pred_first_vs_gt_first_m_max"] for r in rows),
            "ok": bool(past_end < 1e-5 and gt_join < 1e-5
                       and max(r["pred_first_vs_gt_first_m_max"] for r in rows) < 1.0),
            "rows": rows}


# ---------------------------------------------------------------- v3 기준선
def v3_predict(run, idxs, device):
    """v3 규칙 모델(lane_rules_s0) 예측. 대조를 통과하지 못하면 (info, None)."""
    ckpt = C.RUNS / f"lstm_{V3_TAG}.pth"
    info = {"tag": V3_TAG, "ckpt": str(ckpt), "ref": str(V3_REF), "device": device}
    if not (ckpt.exists() and V3_REF.exists() and (C.RUNS / f"{V3_TAG}.json").exists()):
        info["skip"] = "기준선 가중치·기록·대조 파일 중 없는 것이 있다"
        return info, None
    if device != "cuda":
        info["skip"] = ("GPU 를 쓸 수 없다(학습 중이거나 CUDA 없음) — CPU 결과는 저장된 GPU 예측과 "
                        "최대 0.23 m 달라 겹치지 않는다")
        return info, None
    import torch
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.dataloader import default_collate
    from dataset_lane import Av2LaneRuleDataset
    from model_lane import LSTMMapRule
    t0 = time.time()
    rec = json.loads((C.RUNS / f"{V3_TAG}.json").read_text())
    sd = torch.load(ckpt, map_location="cpu")
    in_dim = int(sd["traj_encoder.weight_ih_l0"].shape[1])
    lane_in = int(sd["lane_encoder.0.weight"].shape[1])
    theta = int(rec["args"].get("theta", 0))
    info.update({"in_dim": in_dim, "lane_in": lane_in, "theta": theta, "rules": rec["args"].get("rules")})
    if in_dim != 5 or lane_in != 30 or theta or not rec["args"].get("rules"):
        info["skip"] = f"입력 형식이 raw5·규칙 30 이 아니다 (in_dim {in_dim}, lane_in {lane_in}, theta {theta})"
        return info, None
    model = LSTMMapRule(in_dim=5, lane_in=30).to(device)
    model.load_state_dict(sd)
    model.eval()
    ds = Av2LaneRuleDataset(str(C.DATA_ROOT), "val", with_rules=True)     # raw5, theta 없음
    sids = run.df["sid"].tolist()
    if [d.name for d in ds.dirs] != sids:
        info["skip"] = "dataset_lane 의 val 순서가 덤프와 다르다"
        return info, None

    def fwd(b):
        with torch.no_grad():
            t, lg = model(b["x"].to(device), b["lanes"].to(device), b["lane_mask"].to(device),
                          b["lane_feat"].to(device))
        return t.float().cpu().numpy(), torch.softmax(lg, 1).float().cpu().numpy()

    ref = np.load(V3_REF, allow_pickle=True)
    b = next(iter(DataLoader(Subset(ds, range(V3_N_CHECK)), batch_size=V3_N_CHECK, shuffle=False,
                             num_workers=8)))
    same_ids = list(b["scenario_id"]) == [str(s) for s in ref["scenario_id"][:V3_N_CHECK]]
    t_chk, _ = fwd(b)
    diff = float(np.abs(t_chk - ref["pred"][:V3_N_CHECK]).max())
    info["check"] = {"n": V3_N_CHECK, "batch": V3_N_CHECK, "same_ids": same_ids, "pred_max_abs_diff_m": diff,
                     "tol_m": V3_TOL_M, "ref_minADE6_val2000": float(ref["min_ade"].astype(np.float64).mean()),
                     "log_best_minADE6_val2000": float(rec["best_minADE6"])}
    if not same_ids or diff > V3_TOL_M:
        info["skip"] = f"저장된 v3 예측과 대조 실패 (최대 차 {diff:.3g} m, id 일치 {same_ids})"
        return info, None
    idxs = list(idxs)
    bb = default_collate([ds[i] for i in idxs])
    if list(bb["scenario_id"]) != [sids[i] for i in idxs]:
        info["skip"] = "고른 시나리오의 id 가 다르다"
        return info, None
    pred, prob = fwd(bb)
    past_diff = float(np.abs(bb["x"][:, :, :2].numpy() - run.pos[idxs, :C.OBS]).max())
    y_diff = float(np.abs(bb["y"].numpy() - run.y[idxs]).max())
    d = np.linalg.norm(pred - run.y[idxs][:, None], axis=-1)
    info["picks"] = {str(i): {"minade": round(float(d[k].mean(1).min()), 4),
                              "minfde": round(float(d[k, :, -1].min()), 4),
                              "top1_prob": round(float(prob[k].max()), 3)} for k, i in enumerate(idxs)}
    info["frame_check"] = {"past_xy_vs_v4_dump_max_m": past_diff, "y_vs_v4_dump_max_m": y_diff}
    info["note"] = (f"고른 시나리오는 배치 {len(idxs)} 로 돌렸다 — 배치 크기에 따라 cuDNN 결과가 수 mm 달라질 수 있다"
                    " (배치 1 대조 최대 2.6e-3 m)")
    info["sec"] = round(time.time() - t0, 1)
    if past_diff > 1e-4 or y_diff > 1e-4:
        info["skip"] = f"v3 입력의 좌표계가 v4 덤프와 다르다 (과거 {past_diff:.3g} · 정답 {y_diff:.3g} m)"
        return info, None
    return info, {i: pred[k] for k, i in enumerate(idxs)}


# ---------------------------------------------------------------- 그리기 도구
def dark_rc():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": D_BG, "savefig.facecolor": D_BG, "axes.facecolor": D_BG,
                         "text.color": TXT, "axes.grid": False, "axes.edgecolor": PANEL_EDGE,
                         "legend.frameon": True})


def _pe(lw, extra=1.9):
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw + extra, foreground=OUTLINE), pe.Normal()]


def _txt_pe(w=3.0):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground=OUTLINE)]


def with_origin(t):
    return np.vstack([np.zeros((1, 2)), np.asarray(t, np.float64)])


def p_style(p):
    q = min(max(float(p), 0.0) / P_FULL, 1.0)
    return 0.30 + 0.70 * q, 1.1 + 2.9 * q          # (진하기, 굵기)


def view(pts, min_span=MIN_SPAN_M, pad=0.07, margin=3.0):
    xy = np.concatenate([np.asarray(p, np.float64).reshape(-1, 2) for p in pts])
    xy = xy[np.isfinite(xy).all(1)]
    lo, hi = xy.min(0), xy.max(0)
    c = (lo + hi) / 2
    half = max((hi - lo).max() * (0.5 + pad) + margin, min_span / 2)
    return (c[0] - half, c[0] + half), (c[1] - half, c[1] + half)


def draw_map_dark(ax, scene, xlim, ylim, mark_lw=1.15):
    """hdmap_render.draw_scene 과 같은 순서(주행면 -> 차로 면 -> 횡단보도 -> 노면표시), 어두운 색."""
    import hdmap_render as hr
    from matplotlib.patches import Polygon as MplPolygon
    pad = 0.1 * (xlim[1] - xlim[0])
    box = (xlim[0] - pad, xlim[1] + pad, ylim[0] - pad, ylim[1] + pad)
    ax.set_facecolor(D_BG)
    for poly in scene.drivable:
        if hr._visible(poly, box):
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=D_DRIVE, edgecolor=D_CURB, lw=0.9, zorder=1))
    for poly, color in scene.lanes:
        if hr._visible(poly, box):
            fc = {hr.BIKE: D_BIKE, hr.ASPHALT_INT: D_LANE_INT}.get(color, D_LANE)
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=fc, edgecolor="none", zorder=2))
    for e1, e2 in scene.crossings:
        if not hr._visible(np.concatenate([e1, e2]), box):
            continue
        n = max(2, int(np.linalg.norm(e1[1] - e1[0]) / 0.9))
        t = np.linspace(0, 1, 2 * n + 1)
        p1 = e1[0] + np.outer(t, e1[1] - e1[0])
        p2 = e2[0] + np.outer(t, e2[1] - e2[0])
        for k in range(0, 2 * n, 2):
            quad = np.array([p1[k], p1[k + 1], p2[k + 1], p2[k]])
            ax.add_patch(MplPolygon(quad, closed=True, facecolor=D_CROSS, edgecolor="none", alpha=0.42, zorder=3))
    cmap = {hr.MARK_W: D_MARK_W, hr.MARK_Y: D_MARK_Y, hr.MARK_B: D_MARK_B}
    for bnd, name in scene.marks:
        if not hr._visible(bnd, box):
            continue
        for style, color, off in hr._MARK_STYLE[name]:
            line = hr._offset(bnd, off)
            for p in (hr._dashes(line) if style == "dash" else [line]):
                ax.plot(p[:, 0], p[:, 1], color=cmap.get(color, D_MARK_W), lw=mark_lw,
                        solid_capstyle="butt", zorder=4)


def scale_bar(ax, xlim, ylim):
    span = xlim[1] - xlim[0]
    length = min([m for m in (5, 10, 20, 50, 100, 200) if m >= span / 6], default=200)
    x0 = xlim[0] + 0.05 * span
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    kw = dict(color=TXT, lw=2.4, zorder=40, solid_capstyle="butt", path_effects=_pe(2.4, 1.6))
    ax.plot([x0, x0 + length], [y0, y0], **kw)
    for xe in (x0, x0 + length):
        ax.plot([xe, xe], [y0 - 0.011 * span, y0 + 0.011 * span], **kw)
    ax.text(x0 + length / 2, y0 + 0.02 * span, f"{length} m", color=TXT, fontsize=11, ha="center",
            va="bottom", zorder=40, path_effects=_txt_pe())


def finish_axes(ax, xlim, ylim):
    scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(PANEL_EDGE)


def mark_top(ax, xy, text, color, xlim, size=140):
    ax.scatter(*xy, marker="^", s=size, color=color, edgecolors=OUTLINE, linewidths=1.3, zorder=30)
    if text:
        right = xy[0] > xlim[0] + 0.72 * (xlim[1] - xlim[0])
        ax.annotate(text, xy, xytext=(-9 if right else 9, 7), textcoords="offset points", fontsize=11,
                    color=color, ha="right" if right else "left", va="bottom", zorder=31,
                    path_effects=_txt_pe(3.2))


def grid_figure(header_in, footer_in, title_lines, W=19.5, side=0.32, gap_w=0.34):
    """정사각 칸 3×3. 칸 위에 title_lines 줄의 제목 자리를 둔다. 크기는 인치로 잡는다."""
    import matplotlib.pyplot as plt
    pw = (W - 2 * side - 2 * gap_w) / 3
    title_in = 0.26 * title_lines + 0.12
    gap_h = title_in + 0.22
    H = header_in + title_in + 3 * pw + 2 * gap_h + footer_in
    fig = plt.figure(figsize=(W, H))
    axes = []
    for rr in range(3):
        for cc in range(3):
            x0 = side + cc * (pw + gap_w)
            y0 = footer_in + (2 - rr) * (pw + gap_h)
            axes.append(fig.add_axes([x0 / W, y0 / H, pw / W, pw / H]))
    return fig, axes, W, H, pw


def panel_title(ax, lines, pw_in, colors=None, sizes=None):
    """칸 위 여러 줄 제목 (아래 줄부터 쌓는다). 첫 줄은 굵게."""
    colors = colors or [TXT] + [TXT2] * (len(lines) - 1)
    sizes = sizes or [13.0] + [11.8] * (len(lines) - 1)
    y = 1.0 + 0.06 / pw_in
    for k in range(len(lines) - 1, -1, -1):
        ax.text(0.5, y, lines[k], transform=ax.transAxes, ha="center", va="bottom", fontsize=sizes[k],
                color=colors[k], fontweight="bold" if k == 0 else "normal")
        y += sizes[k] * 1.42 / 72.0 / pw_in


def header(fig, W, H, title, subs, y_top_in=0.28):
    y = H - y_top_in
    fig.text(0.5, y / H, title, ha="center", va="top", fontsize=22, color=TXT, fontweight="bold")
    y -= 0.52
    for s in subs:
        fig.text(0.5, y / H, s, ha="center", va="top", fontsize=13, color=TXT2)
        y -= 0.30
    return y


def legend(fig, W, H, y_in, handles, labels, ncol=3):
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y_in / H), ncol=ncol,
                     fontsize=12.5, frameon=True, facecolor=LEG_BG, edgecolor=PANEL_EDGE, framealpha=1.0,
                     handlelength=3.4, columnspacing=2.2, borderpad=0.7, labelspacing=0.55)
    for t in leg.get_texts():
        t.set_color(TXT)
    return leg


def footer(fig, H, lines, y0_in=0.12):
    y = y0_in + 0.27 * (len(lines) - 1)
    for s in lines:
        fig.text(0.5, y / H, s, ha="center", va="bottom", fontsize=11, color=TXT3)
        y -= 0.27


def title_line1(r, inter):
    it = "·교차로 안" if inter["now"] else ("·교차로 진입" if inter["fut"] else "")
    rt = "경로 없음(직진 폴백)" if bool(r["fallback"]) else f"경로후보 {int(r['n_distinct'])}개"
    return f"{r['sid'][:8]}  {r['cls']}({fmt_deg(r['dh6'])}){it}  {rt}"


def off_text(run, i):
    al = run.alive[i]
    k = int((al & (np.nan_to_num(run.off_frac[i]) > 0)).sum())
    return f"{k}/{int(al.sum())}"


def composition(run, idxs, inters):
    cnt = Counter(run.df["cls"].iloc[idxs])
    cls = " · ".join(f"{c} {cnt[c]}" for c in C.CLASSES if cnt.get(c))
    n_now = sum(1 for f in inters if f["now"])
    n_fut = sum(1 for f in inters if f["fut"] and not f["now"])
    n_fb = int(run.df["fallback"].iloc[idxs].sum())
    return f"9개 구성: {cls}   |   교차로 안 {n_now} · 교차로 진입 {n_fut}   |   폴백(경로 없음) {n_fb}"


def def_lines(past_view_s, compare=False):
    """그림 아래 정의 줄."""
    l1 = ("상황 = 정답 기반 분류(viz_v4_dump: 정지 > 좌·우회전(6초 방향변화 |Δh| > 30°) > 좌·우 차선변경 > 급감속 > 급가속"
          " > 정속 > 기타)  ·  교차로 안 / 진입 = t=0 위치 / 정답 6초 경로가 교차로 차로 면 안")
    l2 = ("경로후보 = 지도에서 열거한 구별되는 후보 경로 수(≤6, 6모드가 나눠 탐)  ·  best = 끝점 오차 최소 모드  ·  "
          "1위 = 확률 최대 모드  ·  (밴드) 이탈 k/6 = 60스텝 중 한 스텝이라도 규칙 밴드 밖인 모드 수")
    l3 = (f"좌표 = focal 정규화(t=0 위치 원점, 진행방향 +x)  ·  시야 = 과거 마지막 {past_view_s:g}초 + 정답 + 예측"
          f"(최소 {MIN_SPAN_M:g} m, 그 앞 과거는 잘릴 수 있음)  ·  예측선은 원점에서 시작하게 그렸다(모델 출력은 t=0.1 s 부터)")
    l4 = ("▲ = 확률 1위 모드의 끝점 (칸 제목 괄호 = 그 확률)  ·  모델마다 시드 1개" if compare else
          f"나머지 모드: 확률 {P_FULL:g} 이상이면 가장 진하고 굵게  ·  시드 1개")
    return [l1, l2, l3, l4]


def past_for_view(past, past_view_s):
    n = int(round(past_view_s / C.DT)) + 1
    return past[-min(n, len(past)):]


# ---------------------------------------------------------------- 갤러리
def gallery_panel(ax, run, i, pw, v3=None, routes=True, past_view_s=PAST_VIEW_S):
    r = run.df.iloc[i]
    scene, inter = run.scene(i)
    al = run.alive[i]
    prob = run.prob[i]
    tr = run.traj[i].astype(np.float64)
    win, top = int(r["winner"]), int(r["top1"])
    past = run.pos[i, :C.OBS]
    gt = with_origin(run.y[i])
    xlim, ylim = view([past_for_view(past, past_view_s), gt, tr[al]])
    draw_map_dark(ax, scene, xlim, ylim)
    if routes and not bool(r["fallback"]):
        for k in range(int(r["n_distinct"])):
            if run.cache["route_mask"][i][k] > 0:
                rp = run.cache["routes"][i][k]
                ax.plot(rp[:, 0], rp[:, 1], color=C_ROUTE, lw=7.0, alpha=0.11, solid_capstyle="round", zorder=5)
    if v3 is not None:
        for m in range(v3.shape[0]):
            t = with_origin(v3[m])
            ax.plot(t[:, 0], t[:, 1], color=C_V3, lw=1.2, ls=(0, (2.0, 2.0)), alpha=0.6, zorder=6)
    C.draw_box(ax, 0, 0, 0, C_FOCAL, zorder=7, ec="white", lw=1.0)
    ax.plot(past[:, 0], past[:, 1], color=C_PAST, lw=2.6, zorder=10, path_effects=_pe(2.6))
    for m in np.argsort(prob):
        if not al[m] or m == win:
            continue
        a, lw = p_style(prob[m])
        t = with_origin(tr[m])
        ax.plot(t[:, 0], t[:, 1], color=C_OTHER, lw=lw, alpha=a, zorder=11, solid_capstyle="round")
        ax.scatter(*t[-1], s=12 + 10 * lw, color=C_OTHER, alpha=a, lw=0, zorder=11)
    ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=3.4, zorder=12, path_effects=_pe(3.4))
    ax.scatter(*gt[-1], s=46, color=C_GT, edgecolors=OUTLINE, linewidths=1.1, zorder=12.5)
    b = with_origin(tr[win])
    ax.plot(b[:, 0], b[:, 1], color=C_BEST, lw=2.7, ls=(0, (3.0, 1.9)), zorder=13, path_effects=_pe(2.7))
    mark_top(ax, tr[top, -1], f"1위{'=best' if top == win else ''} {prob[top]:.2f}", C_TOP, xlim)
    finish_axes(ax, xlim, ylim)
    panel_title(ax, [title_line1(r, inter),
                     f"minADE6 {float(r['minade']):.2f} / minFDE6 {float(r['minfde']):.2f} m  ·  "
                     f"1위{'=' if top == win else '≠'}best  ·  밴드 이탈 {off_text(run, i)} 모드"], pw)
    return inter


def gallery_handles(v3_on, routes_on):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=C_PAST, lw=2.6), Line2D([], [], color=C_GT, lw=3.4, marker="o", ms=6, mec=OUTLINE),
         Line2D([], [], color=C_BEST, lw=2.7, ls=(0, (3.0, 1.9))),
         Line2D([], [], color=C_OTHER, lw=4.0), Line2D([], [], color=C_OTHER, lw=1.1, alpha=0.3),
         Line2D([], [], color=C_TOP, marker="^", ms=11, ls="none", mec=OUTLINE),
         Patch(facecolor=C_FOCAL, edgecolor="white")]
    lab = ["과거 5초 (관측)", "실제 차량이 간 길 (정답 6초, ● 끝점)",
           "v4 예측 · best of 6 (정답 끝점에 가장 가까운 모드)",
           "v4 예측 · 나머지 5개 — 확률 높음: 진하고 굵게", "v4 예측 · 나머지 5개 — 확률 낮음: 흐리고 가늘게",
           "확률 1위 모드의 끝점 (숫자 = 확률)", "focal 차량 (t = 0 s)"]
    if routes_on:
        h.append(Line2D([], [], color=C_ROUTE, lw=7.0, alpha=0.25))
        lab.append("후보 경로 중심선 (지도에서 열거)")
    if v3_on:
        h.append(Line2D([], [], color=C_V3, lw=1.2, ls=(0, (2.0, 2.0)), alpha=0.8))
        lab.append("v3 규칙 모델 예측 6개 (비교용, lane_rules_s0)")
    return h, lab


def draw_gallery(run, idxs, rule, info, out, dpi, v3=None, routes=True, past_view_s=PAST_VIEW_S):
    v3_on = v3 is not None
    legend_rows = int(np.ceil((7 + int(routes) + int(v3_on)) / 3))
    header_in = 0.28 + 0.52 + 0.30 * 3 + 0.12 + 0.34 * legend_rows + 0.45
    foot = def_lines(past_view_s)
    if v3_on:
        foot.append(f"v3 = runs/lstm_{V3_TAG}.pth (raw5 입력 · 차선 규칙 · 학습 50,000개) — 저장된 예측"
                    f"(traj_lane_rules.npz)과 val 앞 {V3_N_CHECK}개 대조가 일치해 GPU 로 다시 추론했다")
    fig, axes, W, H, pw = grid_figure(header_in, footer_in=0.12 + 0.27 * len(foot) + 0.12, title_lines=2)
    inters = [gallery_panel(ax, run, i, pw, v3=(v3 or {}).get(i), routes=routes, past_view_s=past_view_s)
              for ax, i in zip(axes, idxs)]
    n = info["n"] if rule == "mean" else info["pool"]
    if rule == "mean":
        title = "v4 예측 궤적 — 평균 사례 (평균값 기준)"
        s1 = (f"모집단 val {n:,}  ·  평균 minADE6 {info['mean_ade']:.3f} m (표준편차 {info['sd_ade']:.3f})  ·  "
              f"평균 minFDE6 {info['mean_fde']:.3f} m (표준편차 {info['sd_fde']:.3f})  ·  "
              "거리 = |ADE−평균|/σ + |FDE−평균|/σ 가 가장 작은 9개 (칸 순서 = 거리 순)")
    else:
        title = "v4 예측 궤적 — 평균 사례 (중앙값 기준)"
        s1 = (f"모집단 val {n:,}{' (앞 ' + format(n, ',') + '개)' if n < len(run.df) else ''}  ·  "
              f"minFDE6 오름차순 {info['rank_from1'][0]:,}~{info['rank_from1'][1]:,}번째 (가운데 9개)  ·  "
              f"중앙값 minFDE6 {info['median_fde']:.3f} m · minADE6 {info['median_ade']:.3f} m  ·  "
              "visualize_map.py AVERAGE 와 같은 규칙")
    s2 = f"모델 {run.tag}  —  {run.desc()}"
    s3 = composition(run, idxs, inters)
    y = header(fig, W, H, title, [s1, s2, s3])
    h, lab = gallery_handles(v3_on, routes)
    legend(fig, W, H, y - 0.10, h, lab, ncol=3)
    footer(fig, H, foot)
    C.savefig(fig, out, dpi=dpi)
    return inters


# ---------------------------------------------------------------- 비교 그림
ARGS_IGNORE = ("tag", "save_every")       # 학습 결과에 영향이 없는 인자


def args_diff(ra, rb):
    keys = sorted(set(ra.args) | set(rb.args))
    return [k for k in keys if k not in ARGS_IGNORE and ra.args.get(k) != rb.args.get(k)]


def compare_panel(ax, ra, rb, i, pw, past_view_s=PAST_VIEW_S):
    r, q = ra.df.iloc[i], rb.df.iloc[i]
    scene, inter = ra.scene(i)
    past = ra.pos[i, :C.OBS]
    gt = with_origin(ra.y[i])
    A, B = ra.traj[i].astype(np.float64), rb.traj[i].astype(np.float64)
    wa, ta, wb, tb = int(r["winner"]), int(r["top1"]), int(q["winner"]), int(q["top1"])
    xlim, ylim = view([past_for_view(past, past_view_s), gt, A[[wa, ta]], B[[wb, tb]]])
    draw_map_dark(ax, scene, xlim, ylim)
    C.draw_box(ax, 0, 0, 0, C_FOCAL, zorder=7, ec="white", lw=1.0)
    ax.plot(past[:, 0], past[:, 1], color=C_PAST, lw=2.6, zorder=10, path_effects=_pe(2.6))
    # 정답은 굵게 밑에 깐다 — 예측이 겹쳐도 파란 테두리로 보인다
    ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=5.6, zorder=12, path_effects=_pe(5.6))
    ax.scatter(*gt[-1], s=70, color=C_GT, edgecolors=OUTLINE, linewidths=1.1, zorder=12.5)
    # 실선(첫 판)을 먼저, 점선(둘째 판)을 위에 — 겹치면 점선 틈으로 실선이 보인다
    for T, w, t, col, ls, z in ((A, wa, ta, C_A, "-", 14), (B, wb, tb, C_B, LS_B, 16)):
        if t != w:
            u = with_origin(T[t])
            ax.plot(u[:, 0], u[:, 1], color=col, lw=1.6, ls=ls, zorder=z, path_effects=_pe(1.6, 1.5))
        u = with_origin(T[w])
        ax.plot(u[:, 0], u[:, 1], color=col, lw=2.8, ls=ls, zorder=z + 0.5, path_effects=_pe(2.8, 1.5))
        mark_top(ax, T[t, -1], "", col, xlim, size=120)
    finish_axes(ax, xlim, ylim)

    def ln(run, rr, w, t):
        return (f"{run.short}  minADE6 {float(rr['minade']):.2f} / minFDE6 {float(rr['minfde']):.2f} m  ·  "
                f"1위{'=' if t == w else '≠'}best ({float(run.prob[i][t]):.2f})  ·  이탈 {off_text(run, i)}")
    panel_title(ax, [title_line1(r, inter), ln(ra, r, wa, ta), ln(rb, q, wb, tb)], pw,
                colors=[TXT, C_A, C_B], sizes=[13.0, 11.8, 11.8])
    return inter


def draw_compare(ra, rb, idxs, sta, stb, out, dpi, past_view_s=PAST_VIEW_S):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    header_in = 0.28 + 0.52 + 0.30 * 3 + 0.12 + 0.34 * 3 + 0.45
    foot = def_lines(past_view_s, compare=True)
    fig, axes, W, H, pw = grid_figure(header_in, footer_in=0.12 + 0.27 * len(foot) + 0.12, title_lines=3)
    inters = [compare_panel(ax, ra, rb, i, pw, past_view_s) for ax, i in zip(axes, idxs)]
    same = ra.desc().split(" · ")
    other = rb.desc().split(" · ")
    common = " · ".join(p for p in same if p in other)
    diff = args_diff(ra, rb)
    title = f"v4 예측 궤적 — {ra.short} 판 vs {rb.short} 판 (같은 시나리오 · {ra.short} 판 평균값 기준 9개)"
    s1 = (f"모집단 val {sta['n']:,}  ·  평균 minADE6 / minFDE6:  {ra.short} {sta['mean_ade']:.3f} / {sta['mean_fde']:.3f} m"
          f"   ·   {rb.short} {stb['mean_ade']:.3f} / {stb['mean_fde']:.3f} m  ·  "
          f"중앙값 minFDE6: {ra.short} {sta['median_fde']:.3f} · {rb.short} {stb['median_fde']:.3f} m")
    s2 = (f"{ra.short} = {ra.tag}  ·  {rb.short} = {rb.tag}  ·  공통: {common}  —  학습 인자 차이: "
          + (", ".join(f"{k} {ra.args.get(k)} / {rb.args.get(k)}" for k in diff) or "없음")
          + f" ({'·'.join(ARGS_IGNORE)} 제외)")
    s3 = (f"칸마다 두 판의 best of 6 (굵은 선)과 확률 1위 (가는 선 · ▲ 끝점) 만 그렸다  ·  1위 = best 면 굵은 선 끝에 ▲"
          f"   |   {composition(ra, idxs, inters)}")
    y = header(fig, W, H, title, [s1, s2, s3])
    h = [Line2D([], [], color=C_PAST, lw=2.6), Line2D([], [], color=C_GT, lw=5.6, marker="o", ms=7, mec=OUTLINE),
         Patch(facecolor=C_FOCAL, edgecolor="white"),
         Line2D([], [], color=C_A, lw=2.8), Line2D([], [], color=C_A, lw=1.6, marker="^", ms=10, mec=OUTLINE),
         Line2D([], [], color=C_B, lw=2.8, ls=LS_B),
         Line2D([], [], color=C_B, lw=1.6, ls=LS_B, marker="^", ms=10, mec=OUTLINE)]
    lab = ["과거 5초 (관측)", "실제 차량이 간 길 (정답 6초 · 굵게 밑에 깔았다, ● 끝점)", "focal 차량 (t = 0 s)",
           f"{ra.short} 판 · best of 6 (실선, 굵게)", f"{ra.short} 판 · 확률 1위 (실선, 가늘게 · ▲ 끝점)",
           f"{rb.short} 판 · best of 6 (점선, 굵게)", f"{rb.short} 판 · 확률 1위 (점선, 가늘게 · ▲ 끝점)"]
    # matplotlib 범례는 열 우선으로 [3, 2, 2] 개씩 채운다 -> 공통 | 첫 판 | 둘째 판
    legend(fig, W, H, y - 0.10, h, lab, ncol=3)
    footer(fig, H, foot)
    C.savefig(fig, out, dpi=dpi)
    return inters


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=C.DEFAULT_TAG)
    ap.add_argument("--rule", default="both", choices=["mean", "median", "both", "none"])
    ap.add_argument("--median-pool", dest="median_pool", type=int, default=0,
                    help="중앙값 기준의 모집단 = val 앞 N개 (0 = 전체). 예전 AVERAGE 그림은 300")
    ap.add_argument("--v3", default="auto", choices=["auto", "off"], help="v3 규칙 모델 예측을 흐린 점선으로 겹친다")
    ap.add_argument("--routes", type=int, default=1, help="후보 경로 중심선을 흐리게 깐다")
    ap.add_argument("--past-view", dest="past_view", type=float, default=PAST_VIEW_S,
                    help="시야 계산에 넣는 과거 길이 [s] (5 = 과거 전부)")
    ap.add_argument("--compare", default=None, help="다른 판 태그 — --tag 판 평균값 기준 9개에서 두 판 비교 그림")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dpi", type=int, default=140)
    a = ap.parse_args()
    if not 130 <= a.dpi <= 150:
        raise SystemExit("dpi 는 130~150 (산출물 규칙)")
    C.setup_mpl()
    dark_rc()
    t0 = time.time()
    run = Run(a.tag)
    gdir = run.dirs["base"] / "gallery"
    jpath = run.dirs["data"] / "gallery_picks.json"
    mean_idx, dist, st = pick_mean(run.df)
    med_idx, med = pick_median(run.df, pool=a.median_pool)
    med_full = med if a.median_pool <= 0 else pick_median(run.df)[1]
    # 모집단을 줄인 중앙값 그림은 기본 그림을 덮어쓰지 않게 이름을 따로 쓴다
    med_name = "avg_median" if a.median_pool <= 0 else f"avg_median_pool{med['pool']}"
    rec = {}
    if jpath.exists():
        try:
            rec = json.loads(jpath.read_text())
        except json.JSONDecodeError:
            rec = {}
        if rec.get("tag") != a.tag:
            rec = {}
    rec.update({
        "tag": a.tag, "model": run.meta["model"], "model_desc": run.desc(), "val_cache": run.meta["val_cache"],
        "dump_repro_ok": run.meta.get("repro", {}).get("ok"), "git_head": C.git_head(),
        "population": {"split": "val", "n": st["n"]},
        "stats": st,
        "rules": {
            "mean": "거리 = |minADE6 − 평균| / σ + |minFDE6 − 평균| / σ (전체 val, ddof=0) 가 가장 작은 9개, 거리 순. 폴백 포함",
            "median": "minFDE6 오름차순(안정 정렬) 정렬 위치 mid−4 … mid+4, mid = N // 2 (visualize_map.py AVERAGE 규칙)",
        },
        "median_rule": med_full,
        "defs": {"best": "살아있는 모드 중 끝점 오차 최소 (덤프 winner = 학습 WTA 승자)",
                 "top1": "확률 최대 모드 (덤프 top1)",
                 "off_mode": "60스텝 중 한 스텝이라도 밴드 밖(d > 좌 한계 또는 −d > 우 한계)인 살아있는 모드",
                 "inter_now": "t=0 위치가 교차로 차로 면(is_intersection, 자전거 차로 제외) 안",
                 "inter_fut": "정답 6초 경로의 한 점이라도 교차로 차로 면 안",
                 "n_distinct": "구별되는 후보 경로 수 (≤6), 폴백은 직진 1개",
                 "len": "원점부터 60스텝 궤적의 경로 길이 [m]",
                 "end_lon_lat": "끝점 오차를 정답 마지막 1초 변위 방향으로 분해 (종 + 앞섬, 횡 + 좌) [m]",
                 "prob_style": f"나머지 모드 진하기 0.30+0.70·q, 굵기 1.1+2.9·q, q = min(p/{P_FULL}, 1)"},
    })
    rec.setdefault("figures", {})
    rules = {"both": ["mean", "median"], "none": []}.get(a.rule, [a.rule])
    picks = {"mean": mean_idx, "median": med_idx}

    v3 = None
    if rules and a.v3 == "auto":
        need = sorted({i for rl in rules for i in picks[rl]})
        device = C.pick_device(a.device)
        info, v3 = v3_predict(run, need, device)
        rec["v3"] = info
        print(f"[v3] {'겹침' if v3 is not None else '안 겹침 — ' + info.get('skip', '')}"
              f"  (대조 {info.get('check', {}).get('pred_max_abs_diff_m', 'n/a')})", flush=True)
    elif rules:
        rec["v3"] = {"skip": "--v3 off"}

    for rl in rules:
        idx = picks[rl]
        name = "avg_mean" if rl == "mean" else med_name
        out = gdir / f"{name}.png"
        inters = draw_gallery(run, idx, rl, st if rl == "mean" else med, out, a.dpi, v3=v3, routes=bool(a.routes),
                              past_view_s=a.past_view)
        mc, cc = metric_check(run, idx), coord_check(run, idx)
        rows = [facts(run, i, f) for i, f in zip(idx, inters)]
        if rl == "mean":
            for rw_ in rows:
                rw_["dist"] = round(float(dist[rw_["idx"]]), 5)
        if v3 is not None:
            for rw_ in rows:
                rw_["v3"] = rec["v3"]["picks"].get(str(rw_["idx"]))
        rec["figures"][name] = {"png": str(out), "rule": rl, "rule_info": st if rl == "mean" else med,
                                "picks": rows, "metric_check": mc, "coord_check": cc,
                                "v3_overlay": v3 is not None, "routes": bool(a.routes), "dpi": a.dpi,
                                "past_view_s": a.past_view, "min_span_m": MIN_SPAN_M,
                                "made": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[gallery] {out}  지표 대조 {'통과' if mc['ok'] else '실패'} (최대 차 {mc['max_abs_diff_m']:.2e} m) · "
              f"좌표 {'통과' if cc['ok'] else '확인 필요'} (과거 끝 {cc['past_end_abs_max_m']:.1e} m, "
              f"예측 첫 점−정답 첫 점 최대 {cc['pred_first_vs_gt_first_m_max']:.2f} m)", flush=True)
        for rw_ in rows:
            print(f"  {rw_['sid'][:8]} {rw_['cls']:<5} ADE {rw_['minade']:.2f} FDE {rw_['minfde']:.2f} | {rw_['summary']}",
                  flush=True)

    if a.compare:
        rb = Run(a.compare)
        if not np.array_equal(rb.df["sid"].to_numpy(), run.df["sid"].to_numpy()):
            raise SystemExit(f"[gallery] {a.compare} 와 {a.tag} 의 시나리오 순서가 다르다")
        _, _, stb = pick_mean(rb.df)
        name = f"compare_{FILE_LAB.get(run.inp, run.tag)}_vs_{FILE_LAB.get(rb.inp, rb.tag)}_avg_mean.png"
        out = gdir / name
        inters = draw_compare(run, rb, mean_idx, st, stb, out, a.dpi, past_view_s=a.past_view)
        rows = []
        for i, f in zip(mean_idx, inters):
            fa, fb = facts(run, i, f), facts(rb, i, f)
            rows.append({"idx": int(i), "sid": fa["sid"], run.short: fa, rb.short: fb})
        rec["figures"]["compare_avg_mean"] = {
            "png": str(out), "other_tag": rb.tag, "other_model": rb.meta["model"], "other_stats": stb,
            "other_dump_repro_ok": rb.meta.get("repro", {}).get("ok"),
            "picks_from": f"{a.tag} 평균값 기준", "picks": rows, "past_view_s": a.past_view,
            "args_diff": {k: [run.args.get(k), rb.args.get(k)] for k in args_diff(run, rb)},
            "metric_check": {run.short: metric_check(run, mean_idx), rb.short: metric_check(rb, mean_idx)},
            "coord_check": {rb.short: coord_check(rb, mean_idx)},
            "made": time.strftime("%Y-%m-%d %H:%M:%S")}
        mcb = rec["figures"]["compare_avg_mean"]["metric_check"][rb.short]
        print(f"[compare] {out}  {rb.short} 지표 대조 {'통과' if mcb['ok'] else '실패'} "
              f"(최대 차 {mcb['max_abs_diff_m']:.2e} m)", flush=True)
        for rw_ in rows:
            fa, fb = rw_[run.short], rw_[rb.short]
            print(f"  {rw_['sid'][:8]} {fa['cls']:<5} {run.short}: {fa['summary']}\n"
                  f"  {'':8} {'':5} {rb.short}: {fb['summary']}", flush=True)

    jpath.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=float))
    print(f"[gallery] json {jpath}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
