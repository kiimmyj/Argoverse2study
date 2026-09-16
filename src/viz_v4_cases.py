"""
viz_v4_cases.py - 대표 시나리오 그림. '모델이 왜 이렇게 예측했나' 를 한 장에 본다.

고르는 법 (minADE6 백분위, 시드 0, 고른 ID 는 data/picks.json)
  좋음    하위 10% 에서 9개          평균   45~55 백분위에서 9개
  안좋음  90~99 백분위에서 9개        최악   상위 1% 에서 9개 (부록)

시나리오마다 한 장
  (a) 지도  주변 차로·후보 경로와 밴드·과거(회색)·정답(검정)·예측 6모드(확률로 색·두께, ★ 승자 ▲ 1위)·앞차
  (b) 시계열 (-5 ~ +6 s)  속도 · 가속도 · 진행방향 h · 잔차각 θ · 승자 경로 횡오프셋 d(+밴드) · 앞차 거리
  (c) 모드별 막대(확률·ADE·FDE·hinge·jitter)와 글상자(손실 항·상황·지도 복잡도)
그룹마다 지도만 모은 3×3 개요도 한 장.

  python src/viz_v4_cases.py                     # 주 모델 (먼저 viz_v4_dump.py)
  python src/viz_v4_cases.py --tag v4_l4nw_ah2_full_s0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

GROUPS = [("좋음", "good", 0.0, 10.0), ("평균", "avg", 45.0, 55.0),
          ("안좋음", "bad", 90.0, 99.0), ("최악", "worst", 99.0, 100.0)]
N_PICK = 9


def pick(df, seed=C.SEED):
    rng = np.random.default_rng(seed)
    p = df["minade"].to_numpy()
    out = {}
    for name, key, lo, hi in GROUPS:
        a, b = np.percentile(p, lo), np.percentile(p, hi)
        m = (p >= a) & (p <= b) if lo > 0 else (p <= b)
        pool = np.where(m)[0]
        ids = rng.choice(pool, N_PICK, replace=False)
        ids = ids[np.argsort(p[ids])]
        out[key] = {"name": name, "pct_range": [lo, hi], "minade_range": [float(a), float(b)],
                    "n_pool": int(len(pool)), "idx": [int(i) for i in ids],
                    "sid": [df["sid"].iloc[i] for i in ids],
                    "minade": [round(float(p[i]), 4) for i in ids]}
    return out


class Ctx:
    """그림에 필요한 배열을 한 번만 연다."""

    def __init__(self, tag):
        import pandas as pd
        from dataset_cached import CachedV4Dataset
        d = C.tag_dirs(tag)["data"]
        self.df = pd.read_parquet(d / "scenarios.parquet")
        self.pred = np.load(d / "pred.npz")
        self.raw = np.load(d / "raw.npz")
        self.meta = json.loads((d / "meta.json").read_text())
        self.P = {k: self.pred[k] for k in ("traj", "prob", "a", "theta", "v", "d", "h", "ade", "fde",
                                            "hinge", "jit", "alive")}
        self.R = {k: self.raw[k] for k in ("pos", "v_pos", "v_fld", "a_fld", "h", "lead_dist",
                                           "lead_track", "gt_d_w", "gt_band_w", "gt_k_w")}
        ds = CachedV4Dataset(C.VAL_CACHE)
        self.cache = {k: ds.raw(k) for k in ("routes", "route_tan", "route_band", "route_len",
                                             "route_mask", "origin", "theta")}
        self._scene = {}

    def scene(self, i):
        if i not in self._scene:
            sid = self.df["sid"].iloc[i]
            self._scene[i] = (C.build_scene(sid, self.cache["origin"][i], self.cache["theta"][i]),
                              others_at_obs(sid, self.cache["origin"][i], float(self.cache["theta"][i])))
        return self._scene[i]


def others_at_obs(sid, origin, theta):
    """t=49 에 관측된 다른 차량 (정규화 좌표, 진행방향)."""
    from av2.datasets.motion_forecasting import scenario_serialization as ss
    from dataset_map import _rotation_matrix
    scn = ss.load_argoverse_scenario_parquet(C.VAL_DIR / sid / f"scenario_{sid}.parquet")
    R = _rotation_matrix(-theta).astype(np.float64)
    out = []
    for tr in scn.tracks:
        if tr.track_id == scn.focal_track_id:
            continue
        for x in tr.object_states:
            if x.timestep == C.OBS - 1:
                p = (np.array(x.position, np.float64) - origin) @ R.T
                out.append((p[0], p[1], x.heading - theta, tr.object_type.value))
    return out


def _ang_near(h, ref):
    """h 를 unwrap 한 뒤 첫 값이 ref 근처가 되게 2π 배수만큼 옮긴다 [rad]."""
    h = np.unwrap(np.asarray(h, np.float64))
    return h - 2 * np.pi * np.round((h[0] - ref) / (2 * np.pi))


def draw_map(ax, cx, i, compact=False):
    df, P, R, ch = cx.df, cx.P, cx.R, cx.cache
    r = df.iloc[i]
    scene, others = cx.scene(i)
    al = P["alive"][i]
    nd = int(r["n_distinct"])
    win, top = int(r["winner"]), int(r["top1"])
    traj, prob = P["traj"][i], P["prob"][i]
    pos = R["pos"][i]
    lt = R["lead_track"][i]

    pts = [pos[20:], traj[al].reshape(-1, 2)]
    if np.isfinite(lt[C.OBS - 1]).all():
        pts.append(lt[C.OBS - 1:C.OBS][np.isfinite(lt[C.OBS - 1:C.OBS]).all(1)])
    xy = np.concatenate(pts)
    lo, hi = xy.min(0), xy.max(0)
    c = (lo + hi) / 2
    # 칸의 실제 가로세로비에 맞춰 시야를 편다 (직선 도로에서 정사각형 시야는 공간을 버린다)
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    ratio = (bb.height * fh) / max(bb.width * fw, 1e-6)
    hx, hy = max((hi - lo)[0] * 0.58, 20.0), max((hi - lo)[1] * 0.58, 20.0 * min(ratio, 1.0))
    if hy / hx < ratio:
        hy = hx * ratio
    else:
        hx = hy / ratio
    xlim, ylim = (c[0] - hx, c[0] + hx), (c[1] - hy, c[1] + hy)
    C.draw_scene_light(ax, scene, xlim, ylim, mark_lw=0.8 if compact else 1.0)

    # 후보 경로와 밴드 (구별 분기만)
    for k in range(nd):
        if not ch["route_mask"][i][k]:
            continue
        rp, rt, rb = ch["routes"][i][k], ch["route_tan"][i][k], ch["route_band"][i][k]
        poly = C.band_polygon(rp, rt, rb)
        ax.fill(poly[:, 0], poly[:, 1], color=C.C_BAND, alpha=0.045, lw=0, zorder=5)
        ax.plot(poly[:, 0], poly[:, 1], color=C.C_BAND, lw=0.5, alpha=0.5, zorder=5)
        on_win = (k == win % nd)
        ax.plot(rp[:, 0], rp[:, 1], color=C.INK2 if on_win else C.MUTED, lw=1.3 if on_win else 0.8,
                ls="-" if on_win else (0, (3, 2)), alpha=0.9, zorder=6)
        if not compact:
            inside = np.where((rp[:, 0] > xlim[0] + 2) & (rp[:, 0] < xlim[1] - 2)
                              & (rp[:, 1] > ylim[0] + 2) & (rp[:, 1] < ylim[1] - 2))[0]
            if len(inside):
                j = inside[int(len(inside) * (0.55 + 0.08 * k)) % len(inside)]
                ax.text(rp[j, 0], rp[j, 1], f"r{k}", fontsize=7, color=C.INK2, zorder=7, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    # 다른 차량 (t=0 s)
    for x, y, yaw, typ in others:
        if xlim[0] - 5 < x < xlim[1] + 5 and ylim[0] - 5 < y < ylim[1] + 5 and typ in ("vehicle", "bus", "motorcyclist"):
            C.draw_box(ax, x, y, yaw, "#b9b6ad", alpha=0.9, zorder=8,
                       length=2.2 if typ == "motorcyclist" else (11.0 if typ == "bus" else 4.6),
                       width=0.9 if typ == "motorcyclist" else (2.5 if typ == "bus" else 1.9))
    # 앞차 트랙
    if np.isfinite(lt).any():
        ax.plot(lt[:C.OBS, 0], lt[:C.OBS, 1], color=C.C_LEAD, lw=1.0, alpha=0.35, zorder=9)
        ax.plot(lt[C.OBS - 1:, 0], lt[C.OBS - 1:, 1], color=C.C_LEAD, lw=1.4, ls=(0, (2, 1.5)), zorder=9)
        if np.isfinite(lt[C.OBS - 1]).all():
            v = lt[C.OBS - 1] - lt[C.OBS - 3] if np.isfinite(lt[C.OBS - 3]).all() else np.array([1.0, 0.0])
            C.draw_box(ax, *lt[C.OBS - 1], np.arctan2(v[1], v[0]) if np.linalg.norm(v) > 0.05 else 0.0,
                       C.C_LEAD, zorder=10)

    # 과거·정답
    ax.plot(pos[:C.OBS, 0], pos[:C.OBS, 1], color=C.C_PAST, lw=2.2, zorder=11)
    ax.plot(pos[C.OBS - 1:, 0], pos[C.OBS - 1:, 1], color=C.C_GT, lw=2.2, zorder=14)
    ax.scatter(*pos[-1], s=22, color=C.C_GT, zorder=15, edgecolors=C.SURF, linewidths=1.2)

    # 예측 6모드 — 확률 낮은 것부터 그려 높은 것이 위에 오게
    for m in np.argsort(prob):
        if not al[m]:
            continue
        t = traj[m]
        ax.plot(t[:, 0], t[:, 1], color=C.prob_color(prob[m]), lw=0.8 + 3.0 * prob[m], zorder=12,
                solid_capstyle="round")
        if not compact:
            ax.text(t[-1, 0], t[-1, 1], f" m{m}", fontsize=7, color=C.INK2, zorder=16, va="bottom",
                    clip_on=True)
    tw = traj[win]
    ax.plot(tw[:, 0], tw[:, 1], color=C.C_WIN, lw=1.1, ls=(0, (3, 2)), zorder=13)
    ax.scatter(*tw[-1], marker="*", s=150 if not compact else 90, color=C.C_WIN, edgecolors=C.INK,
               linewidths=0.6, zorder=18)
    tt = traj[top]
    ax.scatter(*tt[-1], marker="^", s=70 if not compact else 45, color=C.C_TOP, edgecolors=C.SURF,
               linewidths=1.0, zorder=17)
    C.draw_box(ax, 0, 0, 0, "none", zorder=16, ec=C.INK, lw=1.4)
    C.draw_box(ax, 0, 0, 0, C.INK2, zorder=10.5, alpha=0.25, ec="none")
    C.scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(C.AXIS)


def map_legend(ax, loc="upper left", fontsize=7.5, ncol=1):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=C.C_PAST, lw=2.2), Line2D([], [], color=C.C_GT, lw=2.2),
         Line2D([], [], color=C.BLUE_RAMP[10], lw=3.0), Line2D([], [], color=C.BLUE_RAMP[4], lw=1.2),
         Line2D([], [], color=C.C_WIN, marker="*", ms=11, ls=(0, (3, 2)), mec=C.INK, mew=0.6),
         Line2D([], [], color=C.C_TOP, marker="^", ms=7, ls="none"),
         Line2D([], [], color=C.INK2, lw=1.3), Line2D([], [], color=C.MUTED, lw=0.8, ls=(0, (3, 2))),
         Patch(facecolor=C.C_BAND, alpha=0.25), Line2D([], [], color=C.C_LEAD, lw=1.4, ls=(0, (2, 1.5))),
         Patch(facecolor="#b9b6ad")]
    lab = ["과거 5초", "정답 6초", "예측 (확률 높음: 짙고 굵게)", "예측 (확률 낮음)", "★ 승자 = 끝점 오차 최소",
           "▲ 확률 1위", "승자가 탄 후보 경로", "다른 후보 경로", "밴드 (허용 횡오프셋)", "앞차 트랙 (■ = t 0 s)",
           "다른 차량 (t 0 s)"]
    return ax.legend(h, lab, loc=loc, fontsize=fontsize, frameon=True, facecolor="white",
                     edgecolor=C.GRID, framealpha=0.92, handlelength=2.2, ncol=ncol)


def _series(ax, cx, i):
    """시계열 6칸."""
    P, R = cx.P, cx.R
    r = cx.df.iloc[i]
    win, top = int(r["winner"]), int(r["top1"])
    nd = int(r["n_distinct"])
    same_route = (win % nd) == (top % nd)
    t, tp = C.T_AX, C.T_PRED
    both = win == top
    lw_lab = "승자 = 1위" if both else "승자 ★"

    def pred(a, key, conv=lambda x: x, top_ok=True):
        a.plot(tp, conv(P[key][i][win]), color=C.C_WIN, lw=1.6, label=lw_lab)
        if not both and top_ok:
            a.plot(tp, conv(P[key][i][top]), color=C.C_TOP, lw=1.6, label="1위 ▲")

    # 1 속도
    a = ax[0]
    a.axvspan(5.45, 6.0, color=C.GRID, alpha=0.8, lw=0, zorder=0)
    a.plot(t, R["v_pos"][i], color=C.C_GT, lw=1.6, label="정답 (위치 차분)")
    a.plot(t, R["v_fld"][i], color=C.MUTED, lw=1.0, ls=(0, (3, 2)), label="정답 (AV2 속도 필드·평활)")
    pred(a, "v")
    a.set_ylabel("속도 v\n[m/s]")
    a.text(5.72, 0.97, "라벨 끝\n인공 감속", transform=a.get_xaxis_transform(), fontsize=6.3,
           color=C.INK2, ha="center", va="top")
    # 2 가속도
    a = ax[1]
    a.axvspan(5.45, 6.0, color=C.GRID, alpha=0.8, lw=0, zorder=0)
    a.plot(t, R["a_fld"][i], color=C.C_GT, lw=1.6, label="정답 (속도 필드 평활)")
    pred(a, "a")
    lim = max(3.0, min(9.0, 1.15 * np.nanmax(np.abs(np.concatenate(
        [R["a_fld"][i], P["a"][i][win], P["a"][i][top]])))))
    a.set_ylim(-lim, lim)
    a.set_ylabel("가속도 a\n[m/s²]")
    # 3 진행방향
    a = ax[2]
    hg = np.asarray(R["h"][i], np.float64)
    hg = hg - 2 * np.pi * np.round(hg[C.OBS - 1] / (2 * np.pi))
    a.plot(t, np.degrees(hg), color=C.C_GT, lw=1.6, label="정답")
    a.plot(tp, np.degrees(_ang_near(P["h"][i][win], hg[C.OBS - 1])), color=C.C_WIN, lw=1.6)
    if not both:
        a.plot(tp, np.degrees(_ang_near(P["h"][i][top], hg[C.OBS - 1])), color=C.C_TOP, lw=1.6)
    a.set_ylabel("진행방향 h\n[°]")
    # 4 잔차각
    a = ax[3]
    thg = np.degrees(C.wrap(hg - R["gt_k_w"][i]))
    thg[R["v_fld"][i] < C.MOVE_V] = np.nan
    a.plot(t, thg, color=C.C_GT, lw=1.2, ls=(0, (3, 2)), label="정답 θ (승자 경로 기준, v≥1 m/s)")
    pred(a, "theta", np.degrees, top_ok=True)
    a.set_ylabel("잔차각 θ\n[°]")
    # 5 횡오프셋 (승자 경로 기준)
    a = ax[4]
    bd = R["gt_band_w"][i]
    a.plot(t, bd[:, 0], color=C.C_BAND, lw=1.0, ls=(0, (4, 2)), label="밴드 좌/우 한계")
    a.plot(t, -bd[:, 1], color=C.C_BAND, lw=1.0, ls=(0, (4, 2)))
    a.axhline(0, color=C.AXIS, lw=0.8)
    a.plot(t, R["gt_d_w"][i], color=C.C_GT, lw=1.6, label="정답 d")
    pred(a, "d", top_ok=same_route)
    ys = np.concatenate([R["gt_d_w"][i][C.OBS - 10:], P["d"][i][win], bd[C.OBS:, 0], -bd[C.OBS:, 1]])
    a.set_ylim(max(-12, np.nanmin(ys) - 0.8), min(12, np.nanmax(ys) + 0.8))
    a.set_ylabel("횡오프셋 d\n[m] (+좌)")
    a.text(0.99, 0.96, "승자 경로 기준" + ("" if same_route or both else " · 1위는 다른 경로라 d 생략"),
           transform=a.transAxes, ha="right", va="top", fontsize=7, color=C.INK2)
    # 6 앞차 거리
    a = ax[5]
    ld = R["lead_dist"][i]
    if np.isfinite(ld).any():
        a.plot(t, ld, color=C.C_LEAD, lw=1.6)
        a.set_ylim(0, min(C.LEAD_MAX_M, np.nanmax(ld) * 1.2 + 2))
        g49 = r["gap49"]
        txt = (f"t=0 s: {g49:.1f} m · 시간간격 {r['thw49']:.1f} s" if np.isfinite(r["thw49"])
               else (f"t=0 s: {g49:.1f} m · 정지 중" if np.isfinite(g49) else "t=0 s 앞차 없음"))
        a.text(0.99, 0.95, txt, transform=a.transAxes, ha="right", va="top", fontsize=7.5, color=C.INK2)
    else:
        a.text(0.5, 0.5, f"앞차 없음 (|횡거리| < {C.LEAD_LAT_M} m, {C.LEAD_MAX_M:.0f} m 안)",
               transform=a.transAxes, ha="center", va="center", fontsize=8, color=C.MUTED)
        a.set_ylim(0, 1)
    a.set_ylabel("앞차 거리\n[m]")
    a.set_xlabel("시간 [s]  (0 = 예측 시작)")
    for a in ax:
        a.axvline(0, color=C.INK2, lw=0.8, zorder=1)
        a.set_xlim(-5, 6)
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=C.C_GT, lw=1.6), Line2D([], [], color=C.MUTED, lw=1.0, ls=(0, (3, 2))),
          Line2D([], [], color=C.C_GT, lw=1.2, ls=(0, (3, 2))),
          Line2D([], [], color=C.C_WIN, lw=1.6), Line2D([], [], color=C.C_TOP, lw=1.6),
          Line2D([], [], color=C.C_BAND, lw=1.0, ls=(0, (4, 2))), Line2D([], [], color=C.C_LEAD, lw=1.6)]
    ls_ = ["정답 (v: 위치 차분 / a: 속도 필드 평활)", "정답 v (AV2 속도 필드·평활)",
           "정답 θ (승자 경로 기준, v≥1 m/s)", "승자 ★" if not both else "승자 = 1위",
           "확률 1위 ▲", "밴드 좌/우 한계", "앞차 거리 (경로 따라 중심 간)"]
    ax[0].legend(hs, ls_, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=3, fontsize=7.2,
                 borderaxespad=0.0, handlelength=2.0, columnspacing=1.2)


def _bars(ax, cx, i):
    P = cx.P
    r = cx.df.iloc[i]
    al = P["alive"][i]
    nd = int(r["n_distinct"])
    win, top = int(r["winner"]), int(r["top1"])
    x = np.arange(6)
    items = [("prob", "확률", "{:.2f}"), ("ade", "ADE [m]", "{:.2f}"), ("fde", "FDE [m]", "{:.2f}"),
             ("hinge", "hinge [m]\n(손실은 승자 제외)", "{:.3f}"), ("jit", "jitter", "{:.4f}")]
    for a, (k, lab, fmt) in zip(ax, items):
        vals = np.where(al, P[k][i], np.nan)
        cols = [C.C_WIN if m == win else (C.C_TOP if m == top else "#c3c2b7") for m in x]
        edge = [C.C_TOP if (m == win and m == top) else "none" for m in x]
        a.bar(x, np.nan_to_num(vals), width=0.62, color=cols, edgecolor=edge, linewidth=2.0, zorder=3)
        a.set_ylabel(lab)
        a.grid(axis="x", visible=False)
        top_v = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
        a.set_ylim(0, (top_v if top_v > 0 else 1.0) * 1.32)
        for m in (win, top):
            if al[m] and np.isfinite(vals[m]):
                a.text(m, vals[m] + 0.03 * a.get_ylim()[1], fmt.format(vals[m]), ha="center", va="bottom",
                       fontsize=6.8, color=C.INK)
        for m in x:
            if not al[m]:
                a.text(m, 0.02 * a.get_ylim()[1], "없음", ha="center", va="bottom", fontsize=6.5, color=C.MUTED)
        a.set_xticks(x)
        a.set_xticklabels([f"m{m}\nr{m % nd}" if al[m] else f"m{m}" for m in x], fontsize=7)
        a.tick_params(labelbottom=(a is ax[-1]))
    ax[0].set_title("모드별 (m = 슬롯, r = 후보 경로)", fontsize=9.5, loc="left", pad=22)
    from matplotlib.patches import Patch
    ax[0].legend([Patch(color=C.C_WIN), Patch(color=C.C_TOP),
                  Patch(facecolor=C.C_WIN, edgecolor=C.C_TOP, linewidth=2)],
                 ["승자 ★", "확률 1위 ▲", "승자 = 1위"], fontsize=6.8, loc="lower left",
                 bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.1)


def _textbox(ax, cx, i, group):
    r = cx.df.iloc[i]
    w = cx.meta["loss_weights"]
    ax.axis("off")
    th = f"{r['thw49']:.1f} s" if np.isfinite(r["thw49"]) else ("정지 중" if r["has_lead49"] else "—")
    gap = f"{r['gap49']:.1f} m" if np.isfinite(r["gap49"]) else "없음"
    mfg = f"{r['min_fut_gap']:.1f} m" if np.isfinite(r["min_fut_gap"]) else "없음"
    nd = int(r["n_distinct"])
    route_line = (f"정답 기준 경로 r{int(r['gt_route'])} ({r['gt_route_q']}) · 1위 경로 r{int(r['top1_route'])}"
                  f" · 승자 경로 r{int(r['win_route'])}"
                  + ("  → 분기 선택 오류" if r["route_err"] else ""))
    lines = [
        ("손실 항 (학습식을 이 시나리오에 적용)", True),
        (f"smoothL1 {r['l_sl1']:.3f}   CE {r['l_ce']:.3f}   hinge {r['l_hinge']:.3f} (×{w['offlane']:g}, {w['hinge']})"
         f"   jitter {r['l_jit']:.4f} (×{w['smooth']:g})   합계 {r['l_total']:.3f}", False),
        ("지표", True),
        (f"minADE6 {r['minade']:.2f} m (백분위 {r['pct_minade']:.1f})   minFDE6 {r['minfde']:.2f} m   "
         f"1위 ADE/FDE {r['top1_ade']:.2f} / {r['top1_fde']:.2f} m   miss {'예' if r['miss'] else '아니오'}", False),
        (f"이탈 {r['offlane']:.2f}   흔들림 θ {r['jit_theta']:.4f} · a {r['jit_a']:.4f}   7.3°초과 {r['exc_pct']:.2f}%   "
         f"끝점 퍼짐 {r['spread']:.1f} m   끝점 군집 {int(r['n_end_clusters'])}", False),
        ("상황 (정답 기반)", True),
        (f"{r['cls']}   Δh6 {r['dh6']:+.0f}°   Δd {r['dd6']:+.1f} m   v0 {r['v0']:.1f} m/s   "
         f"a 최대 {r['amax_f']:+.1f} · 최소 {r['amin_f']:+.1f} m/s²   6초 이동 {r['move6']:.0f} m", False),
        (f"앞차 {gap} · 시간간격 {th} · 미래 최소 {mfg}", False),
        ("지도 복잡도", True),
        (f"차선 {int(r['n_lanes'])} (30 m 안 {int(r['n_lanes30'])})   구별 분기 {nd}   도달 차로 {int(r['n_reachable'])}   "
         f"폴백 {'예' if r['fallback'] else '아니오'}   살아있는 모드 {int(r['n_alive'])}", False),
        (route_line, False),
        (f"정답 경로 확률 {r['gt_route_prob']:.2f} (순위 {int(r['gt_route_rank'])})   경로 커버리지 "
         f"{'있음' if r['coverage'] else '실패 (어느 밴드에도 안 들어감)'}", False),
    ]
    y = 0.98
    for txt, head in lines:
        ax.text(0.0, y, txt, transform=ax.transAxes, fontsize=8.6 if head else 8.0,
                fontweight="bold" if head else "normal", color=C.INK if head else C.INK2, va="top")
        y -= 0.085 if head else 0.080


def case_figure(cx, i, group, k, out):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    r = cx.df.iloc[i]
    fig = plt.figure(figsize=(21, 12))
    gs = GridSpec(12, 3, figure=fig, width_ratios=[1.3, 1.05, 0.62], wspace=0.16, hspace=0.9,
                  left=0.02, right=0.99, top=0.905, bottom=0.05)
    axm = fig.add_subplot(gs[0:7, 0])
    draw_map(axm, cx, i)
    axl = fig.add_subplot(gs[7, 0])
    axl.axis("off")
    map_legend(axl, loc="center", fontsize=7.3, ncol=3)
    axt = fig.add_subplot(gs[8:12, 0])
    _textbox(axt, cx, i, group)
    ts = [fig.add_subplot(gs[2 * j:2 * j + 2, 1]) for j in range(6)]
    _series(ts, cx, i)
    sub = gs[0:12, 2].subgridspec(5, 1, hspace=0.45)
    bs = [fig.add_subplot(sub[j, 0]) for j in range(5)]
    _bars(bs, cx, i)
    fig.suptitle(f"[{group} {k}/9]  {r['sid']}  ·  상황 {r['cls']}  ·  minADE6 {r['minade']:.2f} m "
                 f"(백분위 {r['pct_minade']:.1f})  ·  minFDE6 {r['minfde']:.2f} m  ·  승자{'=' if r['win_eq_top1'] else '≠'}1위",
                 fontsize=13, x=0.02, ha="left")
    return C.savefig(fig, out)


def overview(cx, ids, group, pr, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(16, 17.2))
    for a, i in zip(axes.flat, ids):
        r = cx.df.iloc[i]
        draw_map(a, cx, i, compact=True)
        a.set_title(f"{r['sid'][:8]} · {r['cls']} · 분기 {int(r['n_distinct'])}\n"
                    f"minADE6 {r['minade']:.2f} · minFDE6 {r['minfde']:.2f} m · 승자{'=' if r['win_eq_top1'] else '≠'}1위",
                    fontsize=10)
    leg = map_legend(axes.flat[0], fontsize=7)
    fig.suptitle(f"{group} 9개 — minADE6 백분위 {pr['pct_range'][0]:g}–{pr['pct_range'][1]:g} "
                 f"({pr['minade_range'][0]:.2f}–{pr['minade_range'][1]:.2f} m, 후보 {pr['n_pool']:,}개 중 시드 {C.SEED} 무작위)",
                 fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return C.savefig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=C.DEFAULT_TAG)
    ap.add_argument("--only", default=None, help="good,avg,bad,worst 중 일부만")
    a = ap.parse_args()
    C.setup_mpl()
    dirs = C.tag_dirs(a.tag)
    cx = Ctx(a.tag)
    picks = pick(cx.df)
    (dirs["data"] / "picks.json").write_text(json.dumps(
        {"tag": a.tag, "seed": C.SEED, "metric": "minADE6", "groups": picks}, indent=2, ensure_ascii=False))
    n = 0
    for key, pr in picks.items():
        if a.only and key not in a.only.split(","):
            continue
        for k, i in enumerate(pr["idx"], 1):
            case_figure(cx, i, pr["name"], k, dirs["cases"] / key / f"{k}_{cx.df['sid'].iloc[i][:8]}.png")
            n += 1
        overview(cx, pr["idx"], pr["name"], pr, dirs["cases"] / f"overview_{key}.png")
        n += 1
        print(f"[cases] {pr['name']} 9 + 개요", flush=True)
    print(f"[cases] 그림 {n}장 -> {dirs['cases']}", flush=True)


if __name__ == "__main__":
    main()
