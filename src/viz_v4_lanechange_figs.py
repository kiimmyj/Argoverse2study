"""
viz_v4_lanechange_figs.py - viz_v4_lanechange.py 의 그림·결정 트리 단계.

  python src/viz_v4_lanechange.py figs
  python src/viz_v4_lanechange.py tree

필수 요건(2026-09-17 사용자 지시) 반영: 시간축은 초·0 = 예측 시작, d 에 밴드선과 차선변경 띠,
묶음 그림(small multiples), 칸마다 표본 수·n<50 은 흐리게, 스텝마다 점, 대표 시나리오 궤적,
결정 트리(분석용·처리 로직용).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C
import viz_v4_lanechange as L

BASE = "v4_l4nw_ah2_full_sm1_cos30_s0"          # 기준판 (10 Hz · cos30 · 시드 0)
SMOOTH = "v4_l4nw_ah3s_full_sm1_cos30_s0"       # 평활판 σ0.15
HIT_M = 1.0                                     # [m] 횡 끝점오차가 이 안이면 '차선변경을 맞혔다'
GROUP_ORDER = ["A 표현 불가", "C 1위가 맞힘", "B 1위가 아님", "D 전부 못 맞힘"]
GROUP_COLOR = {"A 표현 불가": C.MUTED, "C 1위가 맞힘": C.C_GREEN,
               "B 1위가 아님": C.C_YELLOW, "D 전부 못 맞힘": C.C_RED}
INPUT_LABEL = {"ah2": "기준 10 Hz", "ah3s": "평활 σ0.15 10 Hz", "ah3": "평활 σ0.25 10 Hz",
               "ah2_2hz": "기준 2 Hz", "ah3_2hz": "평활 σ0.25 2 Hz"}
INPUT_COLOR = {"ah2": C.C_BLUE, "ah3s": C.C_AQUA, "ah3": C.C_GREEN,
               "ah2_2hz": C.C_ORANGE, "ah3_2hz": C.C_VIOLET}


# ---------------------------------------------------------------- 자료 묶음
class Box:
    def __init__(self):
        import pandas as pd
        from dataset_cached import CachedV4Dataset
        self.g = np.load(L.DATA / "gt.npz", allow_pickle=True)
        self.df = pd.read_parquet(L.BASE_DUMP / "scenarios.parquet")
        self.raw = np.load(L.BASE_DUMP / "raw.npz")
        self.groups = json.loads((L.DATA / "gt_groups.json").read_text())
        self.verify = json.loads((L.DATA / "infer_verify.json").read_text())
        self.N = len(self.df)
        nd = np.maximum(self.df["n_distinct"].to_numpy().astype(int), 1)
        self.nd = nd
        self.gr = self.df["gt_route"].to_numpy().astype(int)
        self.onR = (np.arange(6)[None, :] % nd[:, None]) == self.gr[:, None]
        self.D1 = self.g["def_D1_분류"]
        self.D2 = self.g["def_D2_횡이동"]
        self.D3 = self.g["def_D3_지도차로"]
        self.S1 = self.g["def_S_관측중시작"] & self.D2
        self.S2 = self.g["def_S_예측중새로"] & self.D2
        self.SM = self.g["def_S_중간"] & self.D2
        self.UT = self.g["def_UT_U턴"]
        self.tags = [os.path.basename(f)[5:-4] for f in sorted(glob.glob(str(L.DATA / "pred_*.npz")))]
        self._p = {}
        ds = CachedV4Dataset(str(C.VAL_CACHES["ah2"]))
        self.cache = {k: ds.raw(k) for k in ("routes", "route_tan", "route_band", "route_len",
                                             "route_mask", "origin", "theta", "route_sd0", "v0")}

    def p(self, tag):
        if tag not in self._p:
            self._p[tag] = np.load(L.DATA / f"pred_{tag}.npz")
        return self._p[tag]

    def sel(self, tag, mask):
        """mask 시나리오에 대한 모델 쪽 파생량."""
        p = self.p(tag)
        jj = {int(i): k for k, i in enumerate(p["sub"])}
        idx = np.nonzero(mask)[0]
        row = np.array([jj[int(i)] for i in idx])
        ar = np.arange(len(idx))
        pj = p["proj_d_sub"][row]
        al = p["alive"][idx]
        t1 = p["top1"][idx].astype(int)
        elat = np.abs(pj[:, :, -5:].mean(2) - self.g["d_e"][idx][:, None])
        need = np.abs(self.g["dd6_r"][idx])
        sgn = np.sign(self.g["dd6_r"][idx])
        dd_own = (p["dend"] - p["d0"])[idx]
        m_on = self.onR[idx] & al
        return dict(idx=idx, row=row, ar=ar, pj=pj, al=al, t1=t1, elat=elat, need=need, sgn=sgn,
                    e1=elat[ar, t1], ebest=np.where(al, elat, np.inf).min(1),
                    hit1=elat[ar, t1] <= HIT_M, hitany=np.where(al, elat, np.inf).min(1) <= HIT_M,
                    ach=np.where(m_on, dd_own * sgn[:, None], -np.inf).max(1) / need,
                    ach1=(dd_own[ar, t1] * sgn) / need, t1_on=self.onR[idx][ar, t1],
                    dmax1=p["dmax"][idx][ar, t1], band1=p["band_at_dmax"][idx][ar, t1],
                    over1=p["over_max"][idx][ar, t1], minade=p["minade"][idx],
                    minfde=p["minfde"][idx], miss=p["minfde"][idx] > C.MISS_M,
                    th=p["th_sub"][row], d=p["d_sub"][row], traj=p["traj_sub"][row],
                    band=p["band_sub"][row], v=p["v_sub"][row], prob=p["prob"][idx])

    def fail_groups(self, tag, mask):
        s = self.sel(tag, mask)
        A = self.g["min_maxd"][s["idx"]] > L.WIDE_W
        return s, {"A 표현 불가": A, "C 1위가 맞힘": ~A & s["hit1"],
                   "B 1위가 아님": ~A & s["hitany"] & ~s["hit1"],
                   "D 전부 못 맞힘": ~A & ~s["hitany"]}


def _n(ax, xs, ns, y=1.01):
    C.annotate_n(ax, xs, ns, y=y, fontsize=6.6)


def _bar(ax, xs, vals, ns, color, width=0.62):
    cols = [color if n >= C.MIN_N else C.GRID for n in ns]
    ax.bar(xs, vals, width=width, color=cols, edgecolor="none")


# ================================================================= 그림 1 — 규모·정의
def fig_scale(B, plt):
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    ax = axes[0]
    names = ["D1\n상황 분류", "D2\n|Δd6| ≥ 2.5 m", "D3\n지도 차로 변화", "D2 ∩ D3", "U턴\n|Δh6| ≥ 135°"]
    vals = [B.D1.sum(), B.D2.sum(), B.D3.sum(), (B.D2 & B.D3).sum(), B.UT.sum()]
    cols = [C.C_BLUE, C.C_ORANGE, C.C_AQUA, C.C_GREEN, C.MUTED]
    ax.bar(range(5), vals, color=cols, width=0.64)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}\n{100*v/B.N:.2f}%", ha="center", va="bottom", fontsize=7.6, color=C.INK2)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=6.8, linespacing=1.4)
    ax.set_ylabel("시나리오 수"); ax.set_ylim(0, max(vals) * 1.32)
    ax.set_title(f"정의별 규모 (val {B.N:,})", fontsize=10)

    ax = axes[1]
    dd = np.abs(B.g["dd6_r"])
    bins = np.arange(0, 6.01, 0.25)
    ax.hist(dd[B.D2], bins=bins, color=C.C_ORANGE, alpha=0.85, label=f"D2 (n={B.D2.sum():,})")
    ax.hist(dd[B.D1], bins=bins, color=C.C_BLUE, alpha=0.85, label=f"D1 (n={B.D1.sum():,})")
    ax.axvline(L.LC_D_M, color=C.INK2, lw=1.0, ls=(0, (3, 2)))
    ax.text(L.LC_D_M, ax.get_ylim()[1] * 0.96, " 임계 2.5 m", fontsize=7.2, color=C.INK2, va="top")
    ax.axvline(3.42, color=C.C_GREEN, lw=1.0, ls=(0, (1, 1.6)))
    ax.text(3.42, ax.get_ylim()[1] * 0.80, " 차로폭 3.42 m", fontsize=7.2, color=C.C_GREEN, va="top")
    ax.set_xlim(2.0, 6.0)
    ax.set_xlabel("정답 6초 횡이동 |Δd6| [m]"); ax.set_ylabel("시나리오 수")
    ax.legend(fontsize=7.6, loc="upper right"); ax.set_title("정답 횡이동의 크기", fontsize=10)

    ax = axes[2]
    sub = [("관측 구간에\n이미 시작", B.S1, C.C_AQUA), ("중간", B.SM, C.C_YELLOW),
           ("예측 구간에서\n새로 시작", B.S2, C.C_RED)]
    v = [m.sum() for _, m, _ in sub]
    ax.bar(range(3), v, color=[c for _, _, c in sub], width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x}\n{100*x/B.D2.sum():.0f}%", ha="center", va="bottom", fontsize=7.8, color=C.INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([s[0] for s in sub], fontsize=7.6)
    ax.set_ylim(0, max(v) * 1.35); ax.set_ylabel("시나리오 수")
    ax.set_title(f"D2 안의 시작 시점 (n={B.D2.sum()})", fontsize=10)

    ax = axes[3]
    for name, m, col in sub:
        t = B.g["t10"][m]
        t = t[np.isfinite(t)]
        ax.hist(t, bins=np.arange(-5, 6.01, 0.5), histtype="step", lw=1.8, color=col,
                label=f"{name.replace(chr(10), ' ')} (n={len(t)})")
    ax.axvline(0, color=C.INK, lw=1.2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.42)
    ax.text(0, ax.get_ylim()[1] * 0.60, " 0 s = 예측 시작", fontsize=7.2, color=C.INK, va="top")
    ax.set_xlabel("횡이동 10% 도달 시각 [s]"); ax.set_ylabel("시나리오 수")
    ax.legend(fontsize=6.8, loc="upper left"); ax.set_title("차선변경이 언제 시작되나", fontsize=10)
    fig.suptitle("차선변경 시나리오의 규모와 정의 — val 24,988 · 정답 기준 경로 대비 횡오프셋 d 로 판정",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, L.OUT / "f1_scale.png")


# ================================================================= 그림 2 — 지표
def fig_metrics(B, plt):
    s, grp = B.fail_groups(BASE, B.D2)
    p = B.p(BASE)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    sets = [("전체", np.ones(B.N, bool)), ("D1 분류", B.D1), ("D2 횡이동", B.D2),
            ("D2 관측중\n시작", B.S1), ("D2 예측중\n새로", B.S2)]
    ns = [int(m.sum()) for _, m in sets]
    for ax, key, lab, lim in ((axes[0], "minade", "minADE6 [m]", None),
                              (axes[1], "minfde", "minFDE6 [m]", None)):
        v = [p[key][m].mean() for _, m in sets]
        _bar(ax, range(len(sets)), v, ns, C.C_BLUE)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=7.8, color=C.INK2)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([f"{s_[0]}\nn={n:,}" for s_, n in zip(sets, ns)], fontsize=6.8, linespacing=1.4)
        ax.set_ylabel(lab); ax.set_ylim(0, max(v) * 1.25)
        ax.set_title(lab + " — 상황별", fontsize=10)
    ax = axes[2]
    v = [100 * (p["minfde"][m] > C.MISS_M).mean() for _, m in sets]
    _bar(ax, range(len(sets)), v, ns, C.C_RED)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.0f}%", ha="center", va="bottom", fontsize=7.8, color=C.INK2)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([f"{s_[0]}\nn={n:,}" for s_, n in zip(sets, ns)], fontsize=6.8, linespacing=1.4)
    ax.set_ylabel("miss = minFDE6 > 2 m [%]"); ax.set_ylim(0, 105)
    ax.set_title("miss — 상황별", fontsize=10)

    ax = axes[3]
    cov = [("정답을 ±1.75 m\n안에 담는 경로", B.g["n_cover175"] > 0),
           ("정답을 규칙 밴드\n안에 담는 경로", B.g["n_inband_f"] > 0),
           ("정답을 ±3.6 m\n안에 담는 경로", B.g["n_cover36"] > 0)]
    x = np.arange(3)
    for k, (nm, m) in enumerate(zip(("전체", "D2 차선변경"), (np.ones(B.N, bool), B.D2))):
        v = [100 * (c & m).sum() / m.sum() for _, c in cov]
        ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=[C.C_BLUE, C.C_ORANGE][k],
               label=f"{nm} (n={int(m.sum()):,})")
        for i, q in enumerate(v):
            ax.text(x[i] + (k - 0.5) * 0.36, q, f"{q:.0f}", ha="center", va="bottom", fontsize=7.2,
                    color=C.INK2)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cov], fontsize=7.2)
    ax.set_ylabel("그런 후보 경로가 있는 비율 [%]"); ax.set_ylim(0, 118); ax.legend(fontsize=7.6)
    ax.set_title("후보 경로 커버리지", fontsize=10)
    fig.suptitle(f"차선변경 시나리오의 지표와 후보 경로 — {BASE} (10 Hz · cos30 · 시드 0)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, L.OUT / "f2_metrics.png")


# ================================================================= 그림 3 — 실패 분해
def fig_failure(B, plt):
    s, grp = B.fail_groups(BASE, B.D2)
    sn, gn = B.fail_groups(BASE, ~B.D2 & np.isin(np.arange(B.N), B.p(BASE)["sub"]))
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    ax = axes[0]
    v = [100 * grp[k].mean() for k in GROUP_ORDER]
    ax.bar(range(4), v, color=[GROUP_COLOR[k] for k in GROUP_ORDER], width=0.62)
    for i, k in enumerate(GROUP_ORDER):
        ax.text(i, v[i], f"{int(grp[k].sum())}\n{v[i]:.0f}%", ha="center", va="bottom",
                fontsize=7.8, color=C.INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([k.replace(" ", "\n", 1) for k in GROUP_ORDER], fontsize=7.4)
    ax.set_ylim(0, max(v) * 1.35); ax.set_ylabel("D2 안의 비율 [%]")
    ax.set_title(f"실패 분해 (D2, n={len(s['idx'])})", fontsize=10)

    ax = axes[1]
    bins = np.arange(0, 7.01, 0.35)
    w = lambda x: np.full(len(x), 100.0 / len(x))
    ax.hist(np.clip(sn["e1"], 0, 7), bins=bins, color=C.GRID, weights=w(sn["e1"]),
            label=f"대조: 비-차선변경 (n={len(sn['idx']):,})")
    ax.hist(np.clip(s["e1"], 0, 7), bins=bins, histtype="step", lw=2.0, color=C.C_ORANGE,
            weights=w(s["e1"]), label=f"D2 차선변경 (n={len(s['idx'])})")
    ax.axvline(HIT_M, color=C.INK2, lw=1.0, ls=(0, (3, 2)))
    ax.text(HIT_M, ax.get_ylim()[1] * 0.97, f" 적중 기준 {HIT_M} m", fontsize=7.2, color=C.INK2, va="top")
    ax.set_xlabel("1위 모드의 횡 끝점오차 $e_{lat}$ [m]"); ax.set_ylabel("각 집합 안의 비율 [%]")
    ax.legend(fontsize=7.4)
    ax.set_title(f"1위 적중률  차선변경 {100*s['hit1'].mean():.0f}%  vs  대조 {100*sn['hit1'].mean():.0f}%", fontsize=9.6)

    ax = axes[2]
    need = s["need"]
    got = s["ach1"] * need
    ax.scatter(need, np.clip(got, -1, 6), s=9, color=C.C_BLUE, alpha=0.35, edgecolors="none")
    ax.plot([0, 6], [0, 6], color=C.INK2, lw=1.0, ls=(0, (3, 2)))
    ax.text(5.6, 5.7, "정답 = 모델", fontsize=7.2, color=C.INK2, ha="right")
    ax.axhline(0, color=C.AXIS, lw=0.8)
    ax.set_xlabel("정답 횡이동 |Δd6| [m]"); ax.set_ylabel("1위 모드 횡이동 (정답 방향) [m]")
    ax.set_xlim(2.3, 6); ax.set_ylim(-1, 6)
    ax.set_title(f"횡이동 부족 — 중앙 {np.median(need - got):.2f} m 모자람", fontsize=10)

    ax = axes[3]
    lab = ["1위 모드", "정답 경로 위\n모드 중 최선"]
    data = [s["ach1"], s["ach"]]
    parts = ax.violinplot([np.clip(d, -0.5, 1.5) for d in data], showmedians=True, widths=0.8)
    for b, c in zip(parts["bodies"], (C.C_BLUE, C.C_AQUA)):
        b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor("none")
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[k].set_color(C.INK2); parts[k].set_linewidth(1.0)
    ax.axhline(1.0, color=C.C_GREEN, lw=1.2, ls=(0, (3, 2)))
    ax.text(0.55, 1.02, "정답만큼", fontsize=7.2, color=C.C_GREEN, va="bottom")
    ax.set_xticks([1, 2]); ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_ylabel("달성률 = 모델 횡이동 / 정답 횡이동")
    ax.set_title(f"달성률 중앙 {np.median(s['ach1']):.2f} / {np.median(s['ach']):.2f}", fontsize=10)
    fig.suptitle("무엇을 못 하나 — 후보에 없어서가 아니라 '옆으로 안 움직여서' 다  "
                 f"({BASE})", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, L.OUT / "f3_failure.png")


# ================================================================= 그림 4 — 가설 (1) 평활
def fig_smoothing(B, plt):
    import pandas as pd
    rows = []
    for t in B.tags:
        s = B.sel(t, B.D2)
        p = B.p(t)
        rows.append(dict(tag=t, inp=B.verify[t]["input"], seed=int(t[-1]),
                         ade_all=p["minade"].mean(), ade_lc=s["minade"].mean(),
                         hit1=100 * s["hit1"].mean(), ach=np.median(s["ach1"]),
                         miss_lc=100 * s["miss"].mean()))
    R = pd.DataFrame(rows)
    order = ["ah2", "ah3s", "ah3", "ah2_2hz", "ah3_2hz"]
    order = [o for o in order if (R["inp"] == o).any()]
    # 시작 시점 부분집합도 시드별로 (평활이 '이미 시작한' 차선변경만 살릴 수도 있으니 따로 본다)
    sub_rows = []
    for t in B.tags:
        for nm, m in (("관측중 시작", B.S1), ("중간", B.SM), ("예측중 새로", B.S2)):
            q = B.sel(t, m)
            sub_rows.append(dict(tag=t, inp=B.verify[t]["input"], grp=nm,
                                 hit1=100 * q["hit1"].mean(), minade=q["minade"].mean()))
    SR = pd.DataFrame(sub_rows)
    SR.to_json(L.DATA / "tag_table_by_start.json", orient="records", force_ascii=False, indent=1)
    th0 = theta0_quality(B)
    fig, axes2 = plt.subplots(2, 4, figsize=(15.2, 7.6))
    axes = axes2[0]
    panels = [("ade_all", "전체 minADE6 [m]", False), ("ade_lc", "차선변경 minADE6 [m]", False),
              ("hit1", "차선변경 1위 적중률 [%]", True),
              ("ach", "1위 모드 횡이동 달성률 (중앙)", True)]
    for ax, (key, lab, up) in zip(axes, panels):
        for i, inp in enumerate(order):
            v = R.loc[R["inp"] == inp, key].to_numpy()
            col = INPUT_COLOR[inp]
            ax.plot([i, i], [v.min(), v.max()], color=col, lw=6, alpha=0.28, solid_capstyle="round")
            ax.scatter(np.full(len(v), i), v, s=42, color=col, edgecolors=C.SURF, linewidths=0.8, zorder=3)
            ax.text(i, v.max(), f"평균 {v.mean():.3g}", fontsize=6.8, color=C.INK2, va="bottom", ha="center")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([INPUT_LABEL[o].replace(" ", "\n", 1) + f"\n시드 {int((R['inp']==o).sum())}개"
                            for o in order], fontsize=6.6, linespacing=1.45)
        ax.set_ylabel(lab)
        rng = R.loc[R["inp"] == "ah2", key]
        ax.set_title(f"{lab}  ·  기준판 시드 범위 {rng.max()-rng.min():.3g}", fontsize=9.4)
    for ax, (gname, gmask) in zip(axes2[1], (("관측중 시작", B.S1), ("중간", B.SM), ("예측중 새로", B.S2))):
        for i, inp in enumerate(order):
            v = SR.loc[(SR["inp"] == inp) & (SR["grp"] == gname), "hit1"].to_numpy()
            col = INPUT_COLOR[inp]
            ax.plot([i, i], [v.min(), v.max()], color=col, lw=6, alpha=0.28, solid_capstyle="round")
            ax.scatter(np.full(len(v), i), v, s=42, color=col, edgecolors=C.SURF, linewidths=0.8, zorder=3)
            ax.text(i, v.max(), f"평균 {v.mean():.1f}", fontsize=6.8, color=C.INK2, va="bottom", ha="center")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([INPUT_LABEL[o].replace(" ", "\n", 1) for o in order], fontsize=6.6, linespacing=1.45)
        ax.set_ylabel("1위 적중률 [%]"); ax.set_ylim(0, 33)
        ax.set_title(f"{gname} (n={int(gmask.sum())})", fontsize=9.4)
    ax = axes2[1, 3]
    ax.axis("off")
    ax.text(0.02, 0.92, "읽는 법", fontsize=9.5, color=C.INK, va="top", weight="bold")
    ax.text(0.02, 0.78,
            "· 점 = 시드 하나, 막대 = 그 입력의 시드 범위\n"
            "· 두 입력의 막대가 겹치면 '가릴 수 없음'\n"
            "· 10 Hz 기준판 시드 범위: 전체 minADE6 0.105 m,\n"
            "  차선변경 minADE6 0.143 m, 적중률 5.6 pp\n"
            "· 겹치지 않는 유일한 칸은 2 Hz 적중률인데\n"
            "  평활 쪽이 나쁘다 (전체 13.7 → 11.2,\n"
            "  관측중 시작 24.8 → 20.0)", fontsize=8.0, color=C.INK2, va="top", linespacing=1.6)
    fig.suptitle("가설 (1) 전처리 평활 — 점 하나가 시드 하나, 막대는 시드 범위. "
                 "평활이 차선변경을 좋게 만든 칸은 없다 (2 Hz 적중률은 오히려 시드 범위 밖으로 나빠진다)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    f1 = C.savefig(fig, L.OUT / "f4_smoothing.png")

    # 입력 heading 품질 자체는 좋아졌는가
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    ax = axes[0]
    sets = list(th0["sets"])
    x = np.arange(len(sets))
    for k, inp in enumerate(("ah2", "ah3s", "ah3")):
        v = [th0["rms"][s_][inp] for s_ in sets]
        ax.bar(x + (k - 1) * 0.27, v, width=0.25, color=INPUT_COLOR[inp], label=INPUT_LABEL[inp])
        for i, q in enumerate(v):
            ax.text(x[i] + (k - 1) * 0.27, q, f"{q:.1f}", ha="center", va="bottom", fontsize=6.6, color=C.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s_}\nn={th0['n'][s_]:,}" for s_ in sets], fontsize=6.8, linespacing=1.4)
    ax.set_yscale("log"); ax.set_ylim(1.0, 60)
    ax.set_ylabel("θ0 오차 RMS [°] (로그축)"); ax.legend(fontsize=7.0, loc="upper right")
    ax.set_title("입력이 주는 시작 잔차각 θ0 의 오차", fontsize=10)

    ax = axes[1]
    for k, inp in enumerate(("정답", "ah2", "ah3s")):
        v = [th0["lat6"][s_][inp] for s_ in sets]
        ax.bar(x + (k - 1) * 0.27, v, width=0.25,
               color=C.INK if inp == "정답" else INPUT_COLOR[inp],
               label="정답 θ0" if inp == "정답" else INPUT_LABEL[inp])
    ax.plot(x, [th0["needed"][s_] for s_ in sets], "o-", color=C.C_RED, lw=1.6, ms=6,
            label="정답 횡이동 |Δd6|")
    ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=7.0)
    ax.set_ylabel("6·v0·sin θ0 [m]"); ax.legend(fontsize=7.0)
    ax.set_title("θ0 를 6초 유지하면 나오는 횡이동", fontsize=10)

    ax = axes[2]
    for k, inp in enumerate(("정답", "ah2", "ah3s")):
        v = [th0["corr"][s_][inp] for s_ in sets]
        ax.bar(x + (k - 1) * 0.27, v, width=0.25,
               color=C.INK if inp == "정답" else INPUT_COLOR[inp],
               label="정답 θ0 (천장)" if inp == "정답" else INPUT_LABEL[inp])
    ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=7.0)
    ax.set_ylabel("corr(θ0, 정답 Δd6)"); ax.set_ylim(0, 0.92); ax.legend(fontsize=7.0)
    ax.set_title("θ0 가 미래 횡이동을 얼마나 말해 주나", fontsize=10)
    fig.suptitle("평활은 입력 heading 을 실제로 좋게 만든다 — 그런데 '예측 구간에서 새로 시작'은 "
                 "천장(정답 θ0)부터가 낮다", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    f2 = C.savefig(fig, L.OUT / "f5_input_heading.png")
    R.to_json(L.DATA / "tag_table.json", orient="records", force_ascii=False, indent=1)
    (L.DATA / "theta0.json").write_text(json.dumps(th0, indent=1, ensure_ascii=False, default=float))
    return f1, f2


def theta0_quality(B):
    """입력이 주는 적분기 시작 잔차각 θ0 의 품질 (모델과 무관)."""
    from dataset_cached import CachedV4Dataset
    N = B.N
    ar = np.arange(N)
    gr = B.gr
    tan = np.asarray(B.cache["route_tan"])
    s0 = np.asarray(B.cache["route_sd0"])[ar, gr, 0]
    Lr = np.maximum(np.asarray(B.cache["route_len"])[ar, gr], 1e-3)
    M = tan.shape[2]
    ix = np.clip(s0 / Lr * (M - 1), 0, M - 1 - 1e-4)
    i0 = np.floor(ix).astype(int)
    f = (ix - i0)[:, None]
    t0 = tan[ar, gr, i0] * (1 - f) + tan[ar, gr, np.minimum(i0 + 1, M - 1)] * f
    k0 = np.arctan2(t0[:, 1], t0[:, 0])
    th_true = C.wrap(B.raw["h"][:, C.OBS - 1].astype(np.float64) - k0)
    v0 = np.asarray(B.cache["v0"])
    h0 = {k: np.asarray(CachedV4Dataset(str(v)).raw("h0")) for k, v in C.VAL_CACHES.items()}
    sets = {"전체 val": np.ones(N, bool), "D2 차선변경": B.D2,
            "관측중 시작": B.S1, "예측중 새로": B.S2}
    out = {"sets": list(sets), "n": {}, "rms": {}, "lat6": {}, "corr": {}, "needed": {}}
    for nm, m in sets.items():
        sg = np.sign(B.g["dd6_r"][m])
        out["n"][nm] = int(m.sum())
        out["rms"][nm] = {k: float(np.degrees(np.sqrt(np.mean((C.wrap(h0[k][m] - k0[m]) - th_true[m]) ** 2))))
                          for k in ("ah2", "ah3", "ah3s")}
        out["lat6"][nm] = {"정답": float(np.median(6.0 * v0[m] * np.sin(th_true[m]) * sg))}
        out["corr"][nm] = {"정답": float(np.corrcoef(th_true[m], B.g["dd6_r"][m])[0, 1])}
        for k in ("ah2", "ah3", "ah3s"):
            th = C.wrap(h0[k][m] - k0[m])
            out["lat6"][nm][k] = float(np.median(6.0 * v0[m] * np.sin(th) * sg))
            out["corr"][nm][k] = float(np.corrcoef(th, B.g["dd6_r"][m])[0, 1])
        out["needed"][nm] = float(np.median(np.abs(B.g["dd6_r"][m])))
    return out


# ================================================================= 그림 6 — 가설 (2) 밴드
def fig_band(B, plt):
    s, grp = B.fail_groups(BASE, B.D2)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    ax = axes[0]
    lab = ["±1.75 m\n(차로 반폭)", "실제 규칙 밴드\n(허용 쪽만 3.6)", "±3.6 m\n(넓힌 쪽)"]
    for k, (nm, m, col) in enumerate((("전체 val", np.ones(B.N, bool), C.C_BLUE),
                                      ("D2 차선변경", B.D2, C.C_ORANGE))):
        v = [100 * (B.g[q][m] > 0).mean() for q in ("over175", "over_rule", "over36")]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, width=0.34, color=col, label=f"{nm} (n={int(m.sum()):,})")
        for i, q in enumerate(v):
            ax.text(i + (k - 0.5) * 0.36, q, f"{q:.0f}", ha="center", va="bottom", fontsize=7.2, color=C.INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=7.2)
    ax.set_ylabel("정답이 밴드를 벗어나는 비율 [%]"); ax.set_ylim(0, 118); ax.legend(fontsize=7.4)
    ax.set_title("정답은 ±1.75 m 로는 못 담는다", fontsize=10)

    ax = axes[1]
    u = np.abs(s["dmax1"]) / np.maximum(s["band1"], 1e-6)
    for k in GROUP_ORDER:
        m = grp[k]
        if m.sum() < 5:
            continue
        ax.hist(np.clip(u[m], 0, 2.5), bins=np.arange(0, 2.51, 0.125), histtype="step", lw=1.9,
                density=True, color=GROUP_COLOR[k],
                label=f"{k} (n={int(m.sum())}, 중앙 {np.median(u[m]):.2f})")
    ax.axvline(1.0, color=C.INK, lw=1.3)
    ax.text(1.0, 0.02, " 밴드 상한", fontsize=7.2, color=C.INK, va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("1위 모드 max|d| / 밴드"); ax.set_ylabel("밀도")
    ax.legend(fontsize=6.4, loc="upper left")
    ax.set_title("밴드에 붙어 있나 — 맞힌 경우가 오히려 밴드에 가깝다", fontsize=9.6)

    ax = axes[2]
    wide = B.g["band_at_max"][s["idx"]] > 3.0
    narrow = B.g["band_at_max"][s["idx"]] < 2.0
    sets = [("밴드 3.6 m\n(차선변경 허용·교차로)", wide), ("밴드 1.75 m\n(규칙상 불가)", narrow)]
    ns = [int(m.sum()) for _, m in sets]
    v = [100 * s["hit1"][m].mean() for _, m in sets]
    v2 = [100 * s["hitany"][m].mean() for _, m in sets]
    ax.bar(np.arange(2) - 0.18, v, width=0.34, color=C.C_GREEN, label="1위 적중")
    ax.bar(np.arange(2) + 0.18, v2, width=0.34, color=C.C_AQUA, label="6모드 중 최선")
    for i in range(2):
        ax.text(i - 0.18, v[i], f"{v[i]:.0f}", ha="center", va="bottom", fontsize=7.4, color=C.INK2)
        ax.text(i + 0.18, v2[i], f"{v2[i]:.0f}", ha="center", va="bottom", fontsize=7.4, color=C.INK2)
    ax.set_xticks(range(2))
    ax.set_xticklabels([f"{a}\nn={n}" for (a, _), n in zip(sets, ns)], fontsize=7.0, linespacing=1.4)
    ax.set_ylabel("적중률 [%]"); ax.set_ylim(0, 52); ax.legend(fontsize=7.4)
    ax.set_title("자연 실험 — 밴드를 넓혀 줘도 안 맞힌다", fontsize=9.6)

    ax = axes[3]
    p = B.p(BASE)
    al = p["alive"]
    ax.hist(np.clip(p["hinge"][al], 0, 0.8), bins=np.arange(0, 0.81, 0.04), color=C.GRID,
            density=True, label=f"전체 살아있는 모드 (n={int(al.sum()):,})")
    ax.hist(np.clip(p["hinge"][s["idx"]][s["al"]], 0, 0.8), bins=np.arange(0, 0.81, 0.04),
            histtype="step", lw=2.0, color=C.C_ORANGE, density=True,
            label=f"D2 살아있는 모드 (n={int(s['al'].sum()):,})")
    ax.set_xlabel("모드별 이탈 hinge (평균 밴드 초과 [m])"); ax.set_ylabel("밀도")
    ax.legend(fontsize=7.0)
    ax.set_title(f"hinge 0 아닌 비율  전체 {100*(p['hinge'][al]>1e-6).mean():.0f}%  "
                 f"D2 {100*(p['hinge'][s['idx']][s['al']]>1e-6).mean():.0f}%", fontsize=9.2)
    fig.suptitle("가설 (2) 밴드 — 모델의 d 는 밴드로 잘리지 않는다(적분값). 밴드는 버려진 모드의 "
                 "hinge 로만 작용한다", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return C.savefig(fig, L.OUT / "f6_band.png")


# ================================================================= 그림 7 — θ·d 시계열
def fig_series(B, plt):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4))
    sets = [("D2 차선변경 전체", B.D2), ("관측 구간에 이미 시작", B.S1), ("예측 구간에서 새로 시작", B.S2)]
    t_gt = C.T_AX
    t_pr = C.T_PRED
    for col, (nm, m) in enumerate(sets):
        idx = np.nonzero(m)[0]
        sg = np.sign(B.g["dd6_r"][idx])
        # 정답 θ (정답 기준 경로) : dd/dt = v sinθ
        dg = B.g["d_g"][idx].astype(np.float64)
        v = B.raw["v_fld"][idx].astype(np.float64)
        thg = np.degrees(np.arcsin(np.clip(np.gradient(dg, C.DT, axis=1) / np.maximum(v, 0.5), -1, 1))) * sg[:, None]
        ax = axes[0, col]
        ax.plot(t_gt, np.median(thg, 0), color=C.C_GT, lw=2.0, label="정답", **C.step_kw(10, C.C_GT, ms=0))
        ax.fill_between(t_gt, np.percentile(thg, 25, axis=0), np.percentile(thg, 75, axis=0),
                        color=C.C_GT, alpha=0.10, lw=0)
        for tag, col_, lab in ((BASE, C.C_BLUE, "기준 (ah2)"), (SMOOTH, C.C_AQUA, "평활 (ah3s σ0.15)")):
            s = B.sel(tag, m)
            th1 = np.degrees(s["th"][s["ar"], s["t1"]]) * sg[:, None]
            md = np.median(th1, 0)
            ax.plot(t_pr, md, color=col_, lw=1.8, label=f"{lab} 1위 모드")
            ax.plot(t_pr[::5], md[::5], ls="none", **C.step_kw(10, col_))
        ax.axvline(0, color=C.AXIS, lw=1.0)
        ax.axhline(0, color=C.AXIS, lw=0.8)
        # 관측 구간 앞쪽 d 는 경로 시작점 밖이라 미분이 신뢰되지 않는다 -> −2 s 부터만 그린다
        ax.set_xlim(-2, 6); ax.set_ylim(-2, 12)
        ax.set_xlabel("시간 [s] (0 = 예측 시작)")
        ax.set_ylabel("잔차각 θ (정답 횡이동 방향 +) [°]")
        ax.set_title(f"{nm} (n={int(m.sum())})", fontsize=10)
        if col == 0:
            ax.legend(fontsize=7.2)
        ax = axes[1, col]
        d0 = B.g["d0_g"][idx]
        ax.plot(t_gt, np.median((dg - d0[:, None]) * sg[:, None], 0), color=C.C_GT, lw=2.0, label="정답")
        ax.fill_between(t_gt, np.percentile((dg - d0[:, None]) * sg[:, None], 25, axis=0),
                        np.percentile((dg - d0[:, None]) * sg[:, None], 75, axis=0),
                        color=C.C_GT, alpha=0.10, lw=0)
        for tag, col_, lab in ((BASE, C.C_BLUE, "기준 (ah2)"), (SMOOTH, C.C_AQUA, "평활 (ah3s σ0.15)")):
            s = B.sel(tag, m)
            dp = (s["pj"][s["ar"], s["t1"]] - d0[:, None]) * sg[:, None]
            md = np.median(dp, 0)
            ax.plot(t_pr, md, color=col_, lw=1.8, label=f"{lab} 1위 모드")
            ax.plot(t_pr[::5], md[::5], ls="none", **C.step_kw(10, col_))
        # 밴드선 (정답 기준 경로의 규칙 밴드 중앙값, d0 기준)
        bg = B.g["band_g"][idx].astype(np.float64)
        bl = np.median(np.where(sg[:, None] > 0, bg[:, :, 0], bg[:, :, 1]) - d0[:, None] * sg[:, None], 0)
        ax.plot(t_gt, bl, color=C.MUTED, lw=1.2, ls=(0, (4, 2)), label="규칙 밴드 상한(중앙)")
        ax.axhline(L.HALF_W, color=C.C_VIOLET, lw=1.0, ls=(0, (1, 1.6)))
        ax.text(-1.95, L.HALF_W, " ±1.75 m", fontsize=7.0, color=C.C_VIOLET, va="bottom")
        iv = (float(np.nanmedian(B.g["t10"][idx])), float(np.nanmedian(B.g["t90"][idx])))
        ax.axvspan(iv[0], iv[1], color=C.C_YELLOW, alpha=0.16, lw=0, zorder=0)
        ax.text(sum(iv) / 2, 0.03, "차선변경 구간(중앙)", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.8, color=C.INK2)
        ax.axvline(0, color=C.AXIS, lw=1.0)
        ax.set_xlim(-2, 6); ax.set_ylim(-1.2, 4.6)
        ax.set_xlabel("시간 [s] (0 = 예측 시작)"); ax.set_ylabel("횡오프셋 d − d(0) [m]")
        if col == 0:
            ax.legend(fontsize=7.0, loc="upper left")
    fig.suptitle("모델은 시작 잔차각을 제대로 받아 놓고 정답보다 3배 빨리 0 으로 되돌린다 "
                 "(관측중 시작: 1초 뒤 정답 7.1° vs 모델 2.0°) — 선은 중앙값, 띠는 사분위",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return C.savefig(fig, L.OUT / "f7_theta_d.png")


# ================================================================= 그림 8 — 대표 시나리오
def _others(sid, origin, theta):
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


def draw_case(ax, B, i, s=None, j=None, tag=BASE, ratio=0.30):
    p = B.p(tag)
    jj = {int(x): k for k, x in enumerate(p["sub"])}
    k = jj[int(i)]
    traj = p["traj_sub"][k]
    prob = p["prob"][i]
    al = p["alive"][i]
    top = int(p["top1"][i])
    win = int(p["winner"][i])
    pos = B.raw["pos"][i].astype(np.float64)
    sid = B.df["sid"].iloc[i]
    origin, theta = B.cache["origin"][i], float(B.cache["theta"][i])
    scene = C.build_scene(sid, origin, theta)
    xy = np.concatenate([pos[30:], traj[top], traj[win]])
    lo, hi = xy.min(0), xy.max(0)
    c = (lo + hi) / 2
    hx = max((hi - lo)[0] * 0.58, 24.0)
    hy = max((hi - lo)[1] * 0.58, 5.0)
    if hy / hx < ratio:
        hy = hx * ratio
    else:
        hx = hy / ratio
    xlim, ylim = (c[0] - hx, c[0] + hx), (c[1] - hy, c[1] + hy)
    C.draw_scene_light(ax, scene, xlim, ylim, mark_lw=0.8)
    nd = int(B.nd[i])
    for r in range(nd):
        rp, rt, rb = B.cache["routes"][i][r], B.cache["route_tan"][i][r], B.cache["route_band"][i][r]
        poly = C.band_polygon(rp, rt, rb)
        on_g = (r == B.gr[i])
        ax.fill(poly[:, 0], poly[:, 1], color=C.C_BAND, alpha=0.06 if on_g else 0.03, lw=0, zorder=5)
        ax.plot(rp[:, 0], rp[:, 1], color=C.INK2 if on_g else C.MUTED, lw=1.2 if on_g else 0.7,
                ls="-" if on_g else (0, (3, 2)), zorder=6)
    for x, y, yaw, typ in _others(sid, origin, theta):
        if xlim[0] - 4 < x < xlim[1] + 4 and ylim[0] - 4 < y < ylim[1] + 4 and typ in (
                "vehicle", "bus", "motorcyclist"):
            C.draw_box(ax, x, y, yaw, "#b9b6ad", alpha=0.9, zorder=8,
                       length=11.0 if typ == "bus" else 4.6, width=2.5 if typ == "bus" else 1.9)
    ax.plot(pos[:C.OBS, 0], pos[:C.OBS, 1], color=C.C_PAST, lw=2.2, zorder=11)
    C.past_dots(ax, pos[:C.OBS], C.input_hz(B.verify[tag]["input"]), C.C_PAST, zorder=11.2)
    ax.plot(pos[C.OBS - 1:, 0], pos[C.OBS - 1:, 1], color=C.C_GT, lw=2.4, zorder=14)
    ax.scatter(pos[C.OBS + 9:-1:10, 0], pos[C.OBS + 9:-1:10, 1], s=13, color=C.C_GT, zorder=14.5,
               edgecolors=C.SURF, linewidths=0.6)
    for m in np.argsort(prob):
        if not al[m]:
            continue
        t = traj[m]
        C.origin_join(ax, t[0], C.prob_color(prob[m]), lw=0.8 + 3.0 * prob[m])
        ax.plot(t[:, 0], t[:, 1], color=C.prob_color(prob[m]), lw=0.8 + 3.0 * prob[m], zorder=12)
    ax.scatter(*traj[top][-1], marker="^", s=68, color=C.C_TOP, edgecolors=C.SURF, linewidths=1.0, zorder=17)
    ax.scatter(*traj[win][-1], marker="*", s=120, color=C.C_WIN, edgecolors=C.INK, linewidths=0.6, zorder=18)
    C.draw_box(ax, 0, 0, 0, "none", zorder=16, ec=C.INK, lw=1.4)
    C.scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(C.AXIS)


def draw_case_series(ax, B, i, tag=BASE, t0=-2.0):
    p = B.p(tag)
    jj = {int(x): k for k, x in enumerate(p["sub"])}
    k = jj[int(i)]
    top = int(p["top1"][i])
    d_gt = B.g["d_g"][i].astype(np.float64)
    band = B.g["band_g"][i].astype(np.float64)
    keep = C.T_AX >= t0 - 1e-6
    ax.plot(C.T_AX[keep], d_gt[keep], color=C.C_GT, lw=2.0, label="정답 d")
    ax.scatter(C.T_AX[keep][::10], d_gt[keep][::10], s=11, color=C.C_GT, edgecolors=C.SURF,
               linewidths=0.5, zorder=5)
    ax.plot(C.T_AX[keep], band[keep, 0], color=C.MUTED, lw=1.1, ls=(0, (4, 2)), label="규칙 밴드")
    ax.plot(C.T_AX[keep], -band[keep, 1], color=C.MUTED, lw=1.1, ls=(0, (4, 2)))
    ax.axhline(L.HALF_W, color=C.C_VIOLET, lw=0.8, ls=(0, (1, 1.6)))
    ax.axhline(-L.HALF_W, color=C.C_VIOLET, lw=0.8, ls=(0, (1, 1.6)))
    dp = p["proj_d_sub"][k]
    al = p["alive"][i]
    for m in range(6):
        if al[m] and m != top:
            ax.plot(C.T_PRED, dp[m], color=C.prob_color(p["prob"][i][m]), lw=0.9, alpha=0.7)
    ax.plot(C.T_PRED, dp[top], color=C.C_TOP, lw=2.0, label="1위 모드 d")
    ax.scatter(C.T_PRED[4::10], dp[top][4::10], s=11, color=C.C_TOP, edgecolors=C.SURF, linewidths=0.5, zorder=5)
    iv = (float(B.g["t10"][i]), float(B.g["t90"][i]))
    if np.isfinite(iv).all():
        ax.axvspan(min(iv), max(iv), color=C.C_YELLOW, alpha=0.16, lw=0, zorder=0)
        ax.text(np.clip(sum(iv) / 2, t0 + 0.4, 5.6), 0.03, "차선변경", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.6, color=C.INK2, clip_on=True)
    ax.axvline(0, color=C.AXIS, lw=1.0)
    kk = C.T_AX >= -1.0
    lo = min(d_gt[kk].min(), dp[top].min(), -band[kk, 1].max()) - 0.6
    hi = max(d_gt[kk].max(), dp[top].max(), band[kk, 0].max()) + 0.6
    ax.set_xlim(t0, 6); ax.set_ylim(lo, hi)
    ax.set_xlabel("시간 [s]"); ax.set_ylabel("정답 기준 경로 횡오프셋 d [m]")
    ax.tick_params(labelsize=7.5)


def draw_case_theta(ax, B, i, tag=BASE, t0=-2.0):
    """잔차각 θ 시계열 — 정답(정답 기준 경로 d 의 기울기)과 1위 모드."""
    p = B.p(tag)
    jj = {int(x): k for k, x in enumerate(p["sub"])}
    k = jj[int(i)]
    top = int(p["top1"][i])
    d_gt = B.g["d_g"][i].astype(np.float64)
    v = B.raw["v_fld"][i].astype(np.float64)
    thg = np.degrees(np.arcsin(np.clip(np.gradient(d_gt, C.DT) / np.maximum(v, 0.5), -1, 1)))
    keep = C.T_AX >= t0 - 1e-6
    ax.plot(C.T_AX[keep], thg[keep], color=C.C_GT, lw=1.8, label="정답 θ")
    th = np.degrees(p["th_sub"][k][top])
    ax.plot(C.T_PRED, th, color=C.C_TOP, lw=1.8, label="1위 모드 θ")
    ax.plot(C.T_PRED[4::10], th[4::10], ls="none", **C.step_kw(10, C.C_TOP))
    ax.axvline(0, color=C.AXIS, lw=1.0); ax.axhline(0, color=C.AXIS, lw=0.8)
    iv = (float(B.g["t10"][i]), float(B.g["t90"][i]))
    if np.isfinite(iv).all():
        ax.axvspan(min(iv), max(iv), color=C.C_YELLOW, alpha=0.16, lw=0, zorder=0)
    kk = C.T_AX >= -1.0
    lo = min(thg[kk].min(), th.min()) - 1.0
    hi = max(thg[kk].max(), th.max()) + 1.0
    ax.set_xlim(t0, 6); ax.set_ylim(lo, hi)
    ax.set_xlabel("시간 [s]"); ax.set_ylabel("잔차각 θ [°]")
    ax.tick_params(labelsize=7.5)


def fig_cases(B, plt):
    s, grp = B.fail_groups(BASE, B.D2)
    rng = np.random.default_rng(C.SEED)
    picks = {}
    for k in ("C 1위가 맞힘", "D 전부 못 맞힘", "B 1위가 아님"):
        # 저속·정지 시나리오는 진행방향이 잡음이라 대표 사례로 쓰지 않는다
        clean = ((np.abs(B.g["d_g"][s["idx"]][:, C.OBS - 21] - B.g["d0_g"][s["idx"]]) < 5.0)
                 & (B.df["v0_f"].to_numpy()[s["idx"]] >= 4.0)
                 & (np.abs(B.df["dh6"].to_numpy()[s["idx"]]) < L.LC_HEAD_DEG))
        pool = np.nonzero(grp[k] & (s["need"] >= 3.0) & clean)[0]
        n = min(2, len(pool))
        picks[k] = [int(s["idx"][x]) for x in rng.choice(pool, n, replace=False)]
    order = [(k, i) for k in ("C 1위가 맞힘", "D 전부 못 맞힘", "B 1위가 아님") for i in picks[k]][:6]
    W, HR = 3.9, [0.60, 1.05, 0.85]
    fig, axes = plt.subplots(3, len(order), figsize=(W * len(order), 8.4),
                             gridspec_kw={"height_ratios": HR})
    map_ratio = (8.4 * HR[0] / sum(HR) * 0.80) / (W * 0.88)
    for c, (k, i) in enumerate(order):
        j = int(np.nonzero(s["idx"] == i)[0][0])
        axes[0, c].set_title(f"{k} · {B.df['sid'].iloc[i][:8]}\n정답 Δd6 {B.g['dd6_r'][i]:+.1f} m · "
                             f"1위 횡오차 {s['e1'][j]:.1f} m · minADE6 {s['minade'][j]:.2f} m",
                             fontsize=8.0)
        draw_case(axes[0, c], B, i, ratio=map_ratio)
        draw_case_series(axes[1, c], B, i)
        draw_case_theta(axes[2, c], B, i)
        if c == 0:
            axes[1, c].legend(fontsize=6.4, loc="upper left", framealpha=0.85, frameon=True,
                              facecolor="white", edgecolor=C.GRID)
            axes[2, c].legend(fontsize=6.4, loc="upper right", framealpha=0.85, frameon=True,
                              facecolor="white", edgecolor=C.GRID)
    fig.suptitle("대표 시나리오 — 회색 과거 · 검정 정답 · 파랑 6모드(짙을수록 확률 큼) ▲1위 ★승자 · "
                 "가운데는 정답 기준 경로 위 횡오프셋 d, 아래는 잔차각 θ", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    (L.DATA / "case_picks.json").write_text(json.dumps(
        {k: [B.df["sid"].iloc[i] for i in v] for k, v in picks.items()}, indent=1, ensure_ascii=False))
    return C.savefig(fig, L.OUT / "f8_cases.png")


# ================================================================= 처리 로직 결정 트리
def fig_logic_tree(B, plt):
    N = B.N
    nd = B.nd
    fb = B.df["fallback"].to_numpy()
    wide = B.g["wide_frac"]
    d2 = B.D2
    n_lc = int(d2.sum())
    cov175 = (B.g["n_cover175"] > 0)
    fmt = lambda n: f"{n:,} ({100*n/N:.1f}%)"
    root = {"text": f"val 시나리오\n{N:,}", "color": "#ffffff"}
    leaf = lambda t, col: {"text": t, "color": col}
    root["children"] = [
        ("지도가 경로를 줌", {
            "text": f"후보 경로 열거 (successor 만)\n{fmt(int((~fb).sum()))}",
            "color": C.BLUE_RAMP[1],
            "children": [
                ("차로 안 규칙 밴드", leaf(
                    f"±1.75 m\n(차선변경 불가 쪽)\n예측 스텝의 {100*(1-wide.mean()):.0f}%", C.BLUE_RAMP[3])),
                ("차선변경 허용 · 교차로", leaf(
                    f"±3.6 m\n예측 스텝의 {100*wide.mean():.0f}%", C.BLUE_RAMP[5])),
            ]}),
        ("경로 0개 (주차장 등)", leaf(f"폴백 직진 1개\n밴드 ±3.6\n{fmt(int(fb.sum()))}", C.GRID)),
    ]
    root2 = {"text": f"차선변경 정답 (D2)\n{n_lc}", "color": "#ffffff", "children": [
        ("목표 차로가 후보 경로로 열거됨\n(max|d| ≤ 1.75)", leaf(
            f"경로 선택으로 표현 가능\n{int((cov175 & d2).sum())} ({100*(cov175&d2).sum()/n_lc:.0f}%)", C.C_GREEN)),
        ("아니면 잔차 d 로만 표현", {
            "text": f"{int((~cov175 & d2).sum())} ({100*(~cov175&d2).sum()/n_lc:.0f}%)",
            "color": C.BLUE_RAMP[4],
            "children": [
                ("정답이 규칙 밴드 안", leaf(
                    f"{int((~cov175 & d2 & (B.g['over_rule'] <= 0)).sum())}"
                    f" ({100*(~cov175&d2&(B.g['over_rule']<=0)).sum()/n_lc:.0f}%)", C.BLUE_RAMP[6])),
                ("정답이 밴드 밖\n(hinge 대상)", leaf(
                    f"{int((~cov175 & d2 & (B.g['over_rule'] > 0)).sum())}"
                    f" ({100*(~cov175&d2&(B.g['over_rule']>0)).sum()/n_lc:.0f}%)", C.C_YELLOW)),
            ]}),
    ]}
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 4.6))
    C.draw_tree(axes[0], root, fontsize=8.0, edge_fs=7.4)
    axes[0].set_title("처리 로직 (a) 밴드는 어떻게 정해지나 — lane_frame.rule_band", fontsize=10)
    C.draw_tree(axes[1], root2, fontsize=8.0, edge_fs=7.4)
    axes[1].set_title("처리 로직 (b) 차선변경은 무엇으로 표현되나", fontsize=10)
    fig.suptitle("규칙이 갈리는 흐름과 실제 개수 (val 24,988)", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return C.savefig(fig, L.OUT / "f9_logic_tree.png")


# ================================================================= 분석용 결정 트리
def step_tree(a):
    plt = C.setup_mpl()
    from sklearn.tree import DecisionTreeClassifier, export_text
    B = Box()
    s, grp = B.fail_groups(BASE, B.D2)
    idx = s["idx"]
    feats = {
        "정답 횡이동 |Δd6| [m]": s["need"],
        "관측 2초 횡이동 [m]": np.abs(B.g["lat_obs"][idx]),
        "시작 잔차각 |θ0| [deg]": np.degrees(np.abs(B.g["th0_true"][idx])),
        "시작 속력 v0 [m/s]": B.df["v0_f"].to_numpy()[idx],
        "|Δh6| [deg]": np.abs(B.df["dh6"].to_numpy()[idx]),
        "최대 |d| 지점 밴드 [m]": B.g["band_at_max"][idx],
        "후보 분기 수": B.nd[idx].astype(float),
        "최선 경로 max|d| [m]": B.g["min_maxd"][idx],
        "앞차 시간간격 [s]": np.nan_to_num(B.df["thw49"].to_numpy()[idx], nan=99.0),
        "1위 확률": B.p(BASE)["prob"][idx][s["ar"], s["t1"]],
    }
    X = np.stack(list(feats.values()), 1)
    names = list(feats)
    y = s["hit1"].astype(int)
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = perm[:cut], perm[cut:]
    est = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, random_state=C.SEED)
    est.fit(X[tr], y[tr])
    acc_tr = est.score(X[tr], y[tr])
    acc_te = est.score(X[te], y[te])
    base_acc = max(1 - y[te].mean(), y[te].mean())
    from sklearn.metrics import roc_auc_score
    # 적중이 12% 뿐이라 한 번의 분할로는 AUC 가 흔들린다 — 20번 다시 나눠 평균을 적는다
    aucs = []
    for sd in range(20):
        r2 = np.random.default_rng(sd).permutation(len(y))
        a_, b_ = r2[:cut], r2[cut:]
        if y[b_].sum() == 0:
            continue
        e2 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, random_state=C.SEED).fit(X[a_], y[a_])
        aucs.append(float(roc_auc_score(y[b_], e2.predict_proba(X[b_])[:, 1])))
    auc_te, auc_sd = float(np.mean(aucs)), float(np.std(aucs))
    est_all = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, random_state=C.SEED).fit(X, y)
    node = C.sk_tree_nodes(
        est_all, names,
        value_fn=lambda t, j: float(t.value[j][0][1] / max(t.value[j][0].sum(), 1)),
        text_fn=lambda n, v: f"n={n}\n1위 적중 {100*v:.0f}%",
        thr_fmt=lambda nm, x: f"{x:.2f}", lo=0.0, hi=0.6)
    fig, ax = plt.subplots(figsize=(15.0, 5.2))
    C.draw_tree(ax, node, fontsize=7.8, edge_fs=7.2)
    ax.set_title(f"어떤 차선변경을 맞히나 — 얕은 결정 트리 (깊이 3, 잎 ≥ 40, D2 n={len(y)})\n"
                 f"적중이 드물어 정확도는 의미가 없다(검증 {acc_te:.3f}, 항상 '못 맞힘' 이라 해도 {base_acc:.3f}). "
                 f"순위 능력 AUC(검증 20분할) = {auc_te:.3f} ± {auc_sd:.3f} (우연 = 0.5). "
                 "서술용이며 인과가 아니다", fontsize=10.5)
    f = C.savefig(fig, L.OUT / "f10_tree.png")
    leaves = C.tree_leaves(node)
    # 잎 통계를 group-by 로 따로 대조
    rows = []
    lid = est_all.apply(X)
    for lf_ in leaves:
        m = lid == lf_["id"]
        rows.append({"규칙": " & ".join(lf_["rule"]), "n": int(m.sum()),
                     "1위 적중 [%]": round(100 * y[m].mean(), 1),
                     "트리 값 [%]": round(100 * lf_["value"], 1),
                     "minADE6": round(float(s["minade"][m].mean()), 3),
                     "대표 시나리오": B.df["sid"].iloc[idx[np.nonzero(m)[0][0]]]})
    chk = max(abs(r["1위 적중 [%]"] - r["트리 값 [%]"]) for r in rows)
    out = {"features": names, "acc_train": acc_tr, "acc_test": acc_te, "acc_baseline": base_acc,
           "auc_test_mean20": auc_te, "auc_test_sd20": auc_sd,
           "leaf_groupby_max_diff_pp": chk, "leaves": rows,
           "rules_text": export_text(est_all, feature_names=names, decimals=2)}
    (L.DATA / "tree.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[tree] 검증 정확도 {acc_te:.3f} (기준 {base_acc:.3f}) · AUC {auc_te:.3f}±{auc_sd:.3f} · "
          f"잎 대조 최대차 {chk:.2f} pp -> {f}")
    for r in rows:
        print(f"  n={r['n']:4d}  적중 {r['1위 적중 [%]']:5.1f}%  {r['규칙']}")
    return f


# ================================================================= 요약 수치
def step_figs(a):
    plt = C.setup_mpl()
    B = Box()
    made = [fig_scale(B, plt), fig_metrics(B, plt), fig_failure(B, plt)]
    made += list(fig_smoothing(B, plt))
    made += [fig_band(B, plt), fig_series(B, plt), fig_cases(B, plt), fig_logic_tree(B, plt)]
    summary(B)
    for f in made:
        print(f"[fig] {f}")


def summary(B):
    """리포트가 인용하는 숫자를 한 파일에 모은다."""
    s, grp = B.fail_groups(BASE, B.D2)
    sn, _ = B.fail_groups(BASE, ~B.D2 & np.isin(np.arange(B.N), B.p(BASE)["sub"]))
    p = B.p(BASE)
    u = np.abs(s["dmax1"]) / np.maximum(s["band1"], 1e-6)
    wide = B.g["band_at_max"][s["idx"]] > 3.0
    narrow = B.g["band_at_max"][s["idx"]] < 2.0
    a = p["minade"]
    fixed = a.copy(); fixed[B.D2 | B.D1] = a[~(B.D2 | B.D1)].mean()
    out = {
        "정의": B.groups["counts"],
        "규모": {"val": B.N, "D2_pct": round(100 * B.D2.mean(), 2),
                 "시작시점": {"관측중": int(B.S1.sum()), "중간": int(B.SM.sum()), "예측중새로": int(B.S2.sum())},
                 "U턴": int(B.UT.sum()), "U턴_pct": round(100 * B.UT.mean(), 3)},
        "지표": {k: {"n": int(m.sum()), "minADE6": round(float(p["minade"][m].mean()), 3),
                    "minFDE6": round(float(p["minfde"][m].mean()), 3),
                    "miss_pct": round(100 * float((p["minfde"][m] > C.MISS_M).mean()), 1)}
                for k, m in (("전체", np.ones(B.N, bool)), ("D1", B.D1), ("D2", B.D2),
                             ("D2_관측중시작", B.S1), ("D2_예측중새로", B.S2))},
        "커버리지": {"경로가_1.75안에_담음_pct": round(100 * float((B.g["n_cover175"][B.D2] > 0).mean()), 1),
                  "규칙밴드_안_pct": round(100 * float((B.g["n_inband_f"][B.D2] > 0).mean()), 1),
                  "3.6_안_pct": round(100 * float((B.g["n_cover36"][B.D2] > 0).mean()), 1),
                  "전체_1.75안_pct": round(100 * float((B.g["n_cover175"] > 0).mean()), 1),
                  "최선경로_max_d_중앙": round(float(np.median(B.g["min_maxd"][B.D2])), 2)},
        "실패분해": {k: {"n": int(v.sum()), "pct": round(100 * float(v.mean()), 1)} for k, v in grp.items()},
        "적중": {"1위_pct": round(100 * float(s["hit1"].mean()), 1),
                "최선_pct": round(100 * float(s["hitany"].mean()), 1),
                "대조_1위_pct": round(100 * float(sn["hit1"].mean()), 1),
                "대조_최선_pct": round(100 * float(sn["hitany"].mean()), 1),
                "달성률_1위_중앙": round(float(np.median(s["ach1"])), 2),
                "달성률_경로내최선_중앙": round(float(np.median(s["ach"])), 2),
                "부족량_중앙_m": round(float(np.median(s["need"] - s["ach1"] * s["need"])), 2)},
        "밴드": {"정답_1.75초과_pct": round(100 * float((B.g["over175"][B.D2] > 0).mean()), 1),
                "정답_규칙밴드초과_pct": round(100 * float((B.g["over_rule"][B.D2] > 0).mean()), 1),
                "정답_3.6초과_pct": round(100 * float((B.g["over36"][B.D2] > 0).mean()), 1),
                "3.6_적용_스텝비율_D2": round(float(B.g["wide_frac"][B.D2].mean()), 2),
                "3.6_적용_스텝비율_전체": round(float(B.g["wide_frac"].mean()), 2),
                "1위_maxd_over_band_중앙": round(float(np.median(u)), 2),
                "1위_밴드초과_pct": round(100 * float((s["over1"] > 1e-4).mean()), 1),
                "자연실험": {"밴드3.6": {"n": int(wide.sum()), "1위적중_pct": round(100 * float(s["hit1"][wide].mean()), 1)},
                          "밴드1.75": {"n": int(narrow.sum()), "1위적중_pct": round(100 * float(s["hit1"][narrow].mean()), 1)}},
                "hinge_0아닌_모드_pct": {"전체": round(100 * float((p["hinge"][p["alive"]] > 1e-6).mean()), 1),
                                    "D2": round(100 * float((p["hinge"][s["idx"]][s["al"]] > 1e-6).mean()), 1)}},
        "이득_상한": {"차선변경을_비LC평균으로_고치면_minADE6": [round(float(a.mean()), 4), round(float(fixed.mean()), 4)],
                  "시드_범위_10Hz_ah2": round(float(max(B.p(t)["minade"].mean() for t in B.tags if B.verify[t]["input"] == "ah2")
                                              - min(B.p(t)["minade"].mean() for t in B.tags if B.verify[t]["input"] == "ah2")), 4)},
        "모드다양성": {"같은경로_하위모드_Δd퍼짐_중앙_m": None},
    }
    dd_own = (p["dend"] - p["d0"])[s["idx"]]
    m_on = B.onR[s["idx"]] & s["al"]
    sp = [float(dd_own[i][m_on[i]].max() - dd_own[i][m_on[i]].min()) for i in range(len(s["idx"]))
          if m_on[i].sum() >= 2]
    ss = [float(p["smax"][s["idx"]][i][m_on[i]].max() - p["smax"][s["idx"]][i][m_on[i]].min())
          for i in range(len(s["idx"])) if m_on[i].sum() >= 2]
    out["모드다양성"] = {"n": len(sp), "Δd퍼짐_중앙_m": round(float(np.median(sp)), 2),
                    "s퍼짐_중앙_m": round(float(np.median(ss)), 2)}
    (L.DATA / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
