"""
viz_v4_stats.py - 상황·조건별 통계(3절), 손실로 본 '왜'(4절), 다양성(5절) 그래프와 숫자 요약.

한 분포로 뭉치지 않고 조건마다 따로 그린다. 모든 칸에 표본 수를 적고, n < MIN_N(50) 인 칸은 흐리게 그린다.
상자: 가운데 선 = 중앙값, 상자 = 25–75%, 수염 = 5–95%, 점 = 평균. 비율 막대의 가는 선 = 95% 신뢰구간(Wilson).
리포트에 쓰는 숫자는 전부 data/summary.json 에 남긴다.

  python src/viz_v4_stats.py                  # 주 모델 (먼저 viz_v4_dump.py)
  python src/viz_v4_stats.py --tag v4_l4nw_ah2_full_s0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

BOX_FACE, BOX_EDGE = C.BLUE_RAMP[1], C.C_BLUE
LOSS_COLS = [C.C_BLUE, C.C_ORANGE, C.C_AQUA, C.C_YELLOW]
GROUPS = [("좋음", 0, 10), ("평균", 45, 55), ("안좋음", 90, 99), ("최악", 99, 100)]
SUM = {}


# --------------------------------------------------------------------------- 공통 도구
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def box_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4],
            "mean": v.mean(), "fliers": [], "n": len(v)}


def boxes(ax, groups, labels, vert=True, show_n=True, log=False):
    """groups: 값 배열 리스트. n < MIN_N 이면 흐리게."""
    st = [box_stats(g) for g in groups]
    pos = np.arange(len(groups))
    for p_, s in zip(pos, st):
        if s is None:
            continue
        faint = C.faded(s["n"])
        ax.bxp([s], positions=[p_], orientation="vertical" if vert else "horizontal", widths=0.55,
               showmeans=True, showfliers=False,
               patch_artist=True, manage_ticks=False,
               boxprops=dict(facecolor=BOX_FACE, edgecolor=BOX_EDGE, lw=1.0, alpha=0.35 if faint else 1.0),
               medianprops=dict(color=C.INK, lw=1.6, alpha=0.4 if faint else 1.0),
               whiskerprops=dict(color=BOX_EDGE, lw=1.0, alpha=0.35 if faint else 1.0),
               capprops=dict(color=BOX_EDGE, lw=1.0, alpha=0.35 if faint else 1.0),
               meanprops=dict(marker="o", markerfacecolor=C.C_ORANGE, markeredgecolor=C.SURF,
                              markersize=5, alpha=0.4 if faint else 1.0))
    if vert:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        if show_n:
            C.annotate_n(ax, pos, [s["n"] if s else 0 for s in st])
        ax.set_xlim(-0.6, len(groups) - 0.4)
    else:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_ylim(len(groups) - 0.4, -0.6)
    if log:
        (ax.set_yscale if vert else ax.set_xscale)("log")
    return st


def rate_bars(ax, ks, ns, labels, vert=True, color=C.C_BLUE, pct=True, ref=None):
    ks, ns = np.asarray(ks, float), np.asarray(ns, float)
    r = np.where(ns > 0, ks / np.maximum(ns, 1), np.nan)
    ci = np.array([wilson(k, n) for k, n in zip(ks, ns)])
    sc = 100.0 if pct else 1.0
    pos = np.arange(len(ks))
    for p_, v, n, (lo, hi) in zip(pos, r, ns, ci):
        a = 0.35 if C.faded(n) else 1.0
        if vert:
            ax.bar(p_, v * sc, width=0.6, color=color, alpha=a, zorder=3)
            ax.plot([p_, p_], [lo * sc, hi * sc], color=C.INK, lw=1.0, alpha=a, zorder=4)
        else:
            ax.barh(p_, v * sc, height=0.6, color=color, alpha=a, zorder=3)
            ax.plot([lo * sc, hi * sc], [p_, p_], color=C.INK, lw=1.0, alpha=a, zorder=4)
    if ref is not None:
        (ax.axhline if vert else ax.axvline)(ref * sc, color=C.INK2, lw=1.0, ls=(0, (4, 2)), zorder=2)
    if vert:
        ax.set_xticks(pos); ax.set_xticklabels(labels)
        C.annotate_n(ax, pos, ns)
        ax.set_xlim(-0.6, len(ks) - 0.4)
    else:
        ax.set_yticks(pos); ax.set_yticklabels(labels)
        ax.set_ylim(len(ks) - 0.4, -0.6)
    return r


def mean_ci(ax, groups, labels, color=C.C_BLUE, vert=False, fmt="{:.3f}", ref=None):
    pos = np.arange(len(groups))
    out = []
    for p_, g in zip(pos, groups):
        g = np.asarray(g, float)
        g = g[np.isfinite(g)]
        n = len(g)
        if n == 0:
            out.append(np.nan)
            continue
        m, h = g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        a = 0.35 if C.faded(n) else 1.0
        if vert:
            ax.plot([p_, p_], [m - h, m + h], color=C.INK, lw=1.0, alpha=a)
            ax.scatter([p_], [m], s=36, color=color, alpha=a, zorder=3, edgecolors=C.SURF, linewidths=1.5)
        else:
            ax.plot([m - h, m + h], [p_, p_], color=C.INK, lw=1.0, alpha=a)
            ax.scatter([m], [p_], s=36, color=color, alpha=a, zorder=3, edgecolors=C.SURF, linewidths=1.5)
            ax.text(m + h, p_, "  " + fmt.format(m), va="center", fontsize=7.2, color=C.INK2)
        out.append(m)
    if ref is not None:
        (ax.axhline if vert else ax.axvline)(ref, color=C.INK2, lw=1.0, ls=(0, (4, 2)), zorder=1)
    if vert:
        ax.set_xticks(pos); ax.set_xticklabels(labels)
    else:
        ax.set_yticks(pos); ax.set_yticklabels(labels)
        ax.set_ylim(len(groups) - 0.4, -0.6)
    return out


def heat(ax, M, N, xl, yl, cmap_name="v4blue", fmt="{:.2f}", vmin=None, vmax=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, C.BLUE_RAMP)
    Mm = np.ma.masked_invalid(M)
    im = ax.imshow(Mm, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl)
    ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl)
    ax.grid(False)
    lo, hi = im.get_clim()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            n = N[i, j]
            if n == 0 or not np.isfinite(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=C.MUTED)
                continue
            dark = (M[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
            col = "white" if dark else C.INK
            if C.faded(n):
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=C.SURF, alpha=0.65, edgecolor="none"))
                col = C.MUTED
            ax.text(j, i - 0.12, fmt.format(M[i, j]), ha="center", va="center", fontsize=8, color=col)
            ax.text(j, i + 0.22, f"n={int(n):,}", ha="center", va="center", fontsize=6.3, color=col)
    return im


def grid_mean(df, rkey, rlabels, ckey, clabels, val="minade"):
    M = np.full((len(rlabels), len(clabels)), np.nan)
    N = np.zeros_like(M, dtype=int)
    for i, r in enumerate(rlabels):
        for j, c in enumerate(clabels):
            m = (df[rkey] == r) & (df[ckey] == c)
            N[i, j] = int(m.sum())
            if N[i, j]:
                M[i, j] = df.loc[m, val].mean()
    return M, N


def cls_labels(df):
    cnt = df["cls"].value_counts()
    return [f"{c}  (n={int(cnt.get(c, 0)):,})" for c in C.CLASSES]


def by_class(df, col):
    return [df.loc[df["cls"] == c, col].to_numpy() for c in C.CLASSES]


def note(fig, text, y=0.005):
    fig.text(0.01, y, text, fontsize=7.8, color=C.INK2, ha="left", va="bottom")


def prep(df):
    df = df.copy()
    df["v0_bin"] = C.bin_labels(df["v0"], "v0")
    df["absa"] = np.maximum(df["amax_f"].abs(), df["amin_f"].abs())
    df["amax_bin"] = C.bin_labels(df["absa"], "amax")
    df["absdh"] = df["dh6"].abs()
    df["dh_bin"] = C.bin_labels(df["absdh"], "dh")
    thw = C.bin_labels(df["thw49"], "thw")
    thw = np.where(df["lead_stopped49"], "정지 중", thw)
    thw = np.where(~df["has_lead49"], "앞차 없음", thw)
    df["thw_bin"] = thw
    df["lane_bin"] = np.select([df["n_lanes"] <= 15, df["n_lanes"] <= 19], ["≤15", "16–19"], "20")
    df["lane30_bin"] = np.select([df["n_lanes30"] <= 5, df["n_lanes30"] <= 10, df["n_lanes30"] <= 15],
                                 ["0–5", "6–10", "11–15"], "16–20")
    nr = df["n_reachable"]
    df["reach_bin"] = np.select([nr == 0, nr <= 5, nr <= 10, nr <= 20, nr <= 30],
                                ["0", "1–5", "6–10", "11–20", "21–30"], ">30")
    p = df["minade"].to_numpy()
    grp = np.full(len(df), "", dtype=object)
    for name, lo, hi in GROUPS:
        a, b = np.percentile(p, lo), np.percentile(p, hi)
        m = (p >= a) & (p <= b) if lo > 0 else (p <= b)
        grp[m & (grp == "")] = name
    df["grp"] = grp
    return df


# --------------------------------------------------------------------------- 3절
def sec3(df, out):
    plt = C.setup_mpl()
    labs = cls_labels(df)
    n_all = len(df)
    # S1 개수 + 오차 상자
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[0.8, 1, 1, 1]))
    cnt = [int((df["cls"] == c).sum()) for c in C.CLASSES]
    ax[0].barh(range(9), cnt, height=0.6, color=C.C_BLUE)
    for i, v in enumerate(cnt):
        ax[0].text(v, i, f"  {v:,} ({100 * v / n_all:.1f}%)", va="center", fontsize=7.8, color=C.INK2)
    ax[0].set_yticks(range(9)); ax[0].set_yticklabels(C.CLASSES); ax[0].set_ylim(8.6, -0.6)
    ax[0].set_xlim(0, max(cnt) * 1.45); ax[0].set_title("상황별 개수"); ax[0].grid(axis="y", visible=False)
    for a, col, t in zip(ax[1:], ["minade", "minfde", "top1_ade"], ["minADE6 [m]", "minFDE6 [m]", "확률 1위 모드 ADE [m]"]):
        boxes(a, by_class(df, col), labs if a is ax[1] else [""] * 9, vert=False)
        a.axvline(df[col].mean(), color=C.INK2, lw=1.0, ls=(0, (4, 2)))
        a.set_title(t); a.grid(axis="y", visible=False)
        if a is not ax[1]:
            a.set_yticklabels([])
    fig.suptitle("상황(정답 기반)별 오차 — 상자 25–75% · 수염 5–95% · 주황 점 평균 · 점선 전체 평균", x=0.01, ha="left")
    note(fig, f"val {n_all:,} 시나리오 · 분류는 상호배타, 우선순위 = 위에서 아래 (정의·임계값은 리포트 표).")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    C.savefig(fig, out / "s1_class_error.png")
    SUM["class"] = {c: {"n": int(n), "share": n / n_all,
                        "minade_mean": float(df.loc[df.cls == c, "minade"].mean()),
                        "minade_median": float(df.loc[df.cls == c, "minade"].median()),
                        "minfde_mean": float(df.loc[df.cls == c, "minfde"].mean()),
                        "top1_ade_mean": float(df.loc[df.cls == c, "top1_ade"].mean()),
                        "miss": float(df.loc[df.cls == c, "miss"].mean()),
                        "win_ne_top1": float((~df.loc[df.cls == c, "win_eq_top1"]).mean()),
                        "route_err": float(df.loc[df.cls == c, "route_err"].mean()),
                        "coverage_fail": float((~df.loc[df.cls == c, "coverage"]).mean()),
                        "offlane": float(df.loc[df.cls == c, "offlane"].mean()),
                        "exc_pct": float(df.loc[df.cls == c, "exc_pct"].mean()),
                        "spread_median": float(df.loc[df.cls == c, "spread"].median())}
                    for c, n in zip(C.CLASSES, cnt)}

    # S2 비율
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    for a, col, t, neg in zip(ax, ["miss", "win_eq_top1", "route_err"],
                              ["miss 비율 (minFDE6 > 2 m)", "승자 ≠ 확률 1위 비율", "분기 선택 오류 비율 (구별 분기 ≥ 2)"],
                              [False, True, False]):
        sub = df if col != "route_err" else df[df["n_distinct"] >= 2]
        ks = [int(((~sub.loc[sub.cls == c, col]) if neg else sub.loc[sub.cls == c, col]).sum()) for c in C.CLASSES]
        ns = [int((sub.cls == c).sum()) for c in C.CLASSES]
        ref = ((~sub[col]) if neg else sub[col]).mean()
        rate_bars(a, ks, ns, [f"{c} (n={n:,})" for c, n in zip(C.CLASSES, ns)] if a is ax[0] or col == "route_err" else [""] * 9,
                  vert=False, ref=ref)
        a.set_title(f"{t}\n전체 {ref * 100:.1f}% (점선)", fontsize=10.5); a.set_xlabel("%")
        a.grid(axis="y", visible=False)
        if a is ax[1]:
            a.set_yticklabels([])
    fig.suptitle("상황별 실패 비율 — 막대 = 비율, 가는 선 = 95% 신뢰구간, 점선 = 전체", x=0.01, ha="left")
    note(fig, "분기 선택 오류 = 확률 1위 모드의 후보 경로 ≠ 정답 기준 경로. 구별 분기가 1개면 정의상 0 이라 제외했다. "
              "정지 차량은 움직이지 않아 '정답 경로'가 사실상 정해지지 않으므로 그 행은 해석하지 않는다.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    C.savefig(fig, out / "s2_class_rates.png")

    # S3 실현가능성·퍼짐
    fig, ax = plt.subplots(1, 5, figsize=(19, 5.2))
    items = [("offlane", "이탈 (시나리오당 0~6)", "{:.2f}"), ("jit_theta", "흔들림 θ", "{:.4f}"),
             ("jit_a", "흔들림 a", "{:.4f}"), ("exc_pct", "7.3°초과 step [%]", "{:.2f}"),
             ("spread", "모드 끝점 퍼짐 [m]", "{:.1f}")]
    for a, (col, t, fmt) in zip(ax, items):
        mean_ci(a, by_class(df, col), labs if a is ax[0] else [""] * 9, fmt=fmt, ref=df[col].mean())
        a.set_title(t); a.grid(axis="y", visible=False)
        if a is not ax[0]:
            a.set_yticklabels([])
        xl = a.get_xlim(); a.set_xlim(xl[0], xl[1] + (xl[1] - xl[0]) * 0.25)
    fig.suptitle("상황별 실현가능성·다양성 지표 — 점 = 시나리오 평균, 선 = 95% 신뢰구간, 점선 = 전체 평균", x=0.01, ha="left")
    note(fig, "이탈 = 살아있는 모드별 '밴드 밖 step 비율'의 합. 흔들림 = Δ(dθ/8°)², Δ(a/8)² 의 스텝 평균. "
              "끝점 퍼짐은 살아있는 모드가 2개 이상인 시나리오만(폴백 제외).")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    C.savefig(fig, out / "s3_class_feasibility.png")

    # S4 지도 조건
    conds = [("lane_bin", ["≤15", "16–19", "20"], "차선 수 (lane_mask 합, 최대 20)"),
             ("lane30_bin", ["0–5", "6–10", "11–15", "16–20"], "30 m 안 차로 수"),
             ("n_distinct", [1, 2, 3, 4, 5, 6], "구별 분기 n_distinct"),
             ("reach_bin", ["0", "1–5", "6–10", "11–20", "21–30", ">30"], "도달 가능 차로 n_reachable"),
             ("fallback", [False, True], "폴백 (경로 없음)"),
             ("n_alive", [1, 6], "살아있는 모드 수")]
    fig, ax = plt.subplots(2, 6, figsize=(22, 8.4), gridspec_kw=dict(height_ratios=[1.4, 1]))
    SUM["map_cond"] = {}
    for j, (key, levels, t) in enumerate(conds):
        lab = [("예" if l else "아니오") if key == "fallback" else str(l) for l in levels]
        g = [df.loc[df[key] == l, "minade"].to_numpy() for l in levels]
        boxes(ax[0, j], g, lab)
        ax[0, j].set_title(t, pad=16)
        ax[0, j].axhline(df["minade"].mean(), color=C.INK2, lw=1.0, ls=(0, (4, 2)))
        ks = [int(df.loc[df[key] == l, "miss"].sum()) for l in levels]
        ns = [len(x) for x in g]
        rate_bars(ax[1, j], ks, ns, lab, ref=df["miss"].mean())
        ax[0, j].set_ylim(0, 6)
        ax[1, j].set_ylim(0, 100)
        SUM["map_cond"][key] = {str(l): {"n": n, "minade_mean": float(np.mean(x)) if n else None,
                                         "miss": k / n if n else None} for l, x, n, k in zip(levels, g, ns, ks)}
    ax[0, 0].set_ylabel("minADE6 [m] (5–95% 밖은 잘림)")
    ax[1, 0].set_ylabel("miss 비율 [%]")
    for a in ax[1]:
        for tx in list(a.texts):
            tx.set_visible(False)       # 아래 줄 n 은 위 줄과 같다
    fig.suptitle("지도 조건별 minADE6(위)와 miss 비율(아래) — 흐린 칸 n < 50, 점선 = 전체", x=0.01, ha="left")
    note(fig, "차선 수는 가까운 차선 20개 상한에 98% 가 걸려 정보가 거의 없다 — 그래서 '30 m 안 차로 수'를 함께 보였다. "
              "살아있는 모드 1 = 폴백(직진 가상 경로 1개, 944개)과 같은 집합이다.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    C.savefig(fig, out / "s4_map_conditions.png")

    # S5 동역학 조건
    thw_levels = ["<1 s", "1–2 s", "2–4 s", "≥4 s", "정지 중", "앞차 없음"]
    conds = [("v0_bin", C.BINS["v0"][1], "v0 [m/s]"),
             ("amax_bin", C.BINS["amax"][1], "미래 최대 |a| [m/s²]"),
             ("dh_bin", C.BINS["dh"][1], "6초 |Δh| [°]"),
             ("has_lead49", [True, False], "t=0 앞차"),
             ("thw_bin", thw_levels, "t=0 시간간격")]
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.4), gridspec_kw=dict(height_ratios=[1.4, 1]))
    SUM["dyn_cond"] = {}
    for j, (key, levels, t) in enumerate(conds):
        lab = [("있음" if l else "없음") if key == "has_lead49" else str(l) for l in levels]
        g = [df.loc[df[key] == l, "minade"].to_numpy() for l in levels]
        boxes(ax[0, j], g, lab)
        ax[0, j].set_title(t, pad=16)
        ax[0, j].axhline(df["minade"].mean(), color=C.INK2, lw=1.0, ls=(0, (4, 2)))
        ax[0, j].set_ylim(0, 8)
        ks = [int(df.loc[df[key] == l, "miss"].sum()) for l in levels]
        ns = [len(x) for x in g]
        rate_bars(ax[1, j], ks, ns, lab, ref=df["miss"].mean())
        ax[1, j].set_ylim(0, 100)
        for tx in list(ax[1, j].texts):
            tx.set_visible(False)
        if key == "thw_bin":
            for a in ax[:, j]:
                plt.setp(a.get_xticklabels(), rotation=25, ha="right")
        SUM["dyn_cond"][key] = {str(l): {"n": n, "minade_mean": float(np.mean(x)) if n else None,
                                         "miss": k / n if n else None} for l, x, n, k in zip(levels, g, ns, ks)}
    ax[0, 0].set_ylabel("minADE6 [m] (5–95% 밖은 잘림)")
    ax[1, 0].set_ylabel("miss 비율 [%]")
    fig.suptitle("동역학 조건별 minADE6(위)와 miss 비율(아래) — 흐린 칸 n < 50, 점선 = 전체", x=0.01, ha="left")
    note(fig, "v0 = 캐시의 AV2 속도 필드 속력. |a|·|Δh| 는 정답 미래 6초(속도 필드 평활·움직인 스텝 진행방향). "
              f"앞차 = focal 이 달린 경로 따라 {C.LEAD_MIN_M:g}<Δs≤{C.LEAD_MAX_M:g} m, |횡거리|<{C.LEAD_LAT_M} m, "
              f"방향차<{C.LEAD_HEAD_DEG:g}° 인 가장 가까운 차. 시간간격 = 거리/v0 (v0<{C.THW_MIN_V} m/s 는 '정지 중').")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    C.savefig(fig, out / "s5_dynamic_conditions.png")

    # S6 열지도 v0 × |Δh|
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
    yl, xl = C.BINS["v0"][1], C.BINS["dh"][1]
    M, N = grid_mean(df, "v0_bin", yl, "dh_bin", xl)
    im = heat(ax[0], M, N, xl, yl)
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="평균 minADE6 [m]")
    ax[0].set_xlabel("6초 |Δh| [°]"); ax[0].set_ylabel("v0 [m/s]"); ax[0].set_title("평균 minADE6")
    M2, _ = grid_mean(df.assign(miss_f=df["miss"].astype(float) * 100), "v0_bin", yl, "dh_bin", xl, "miss_f")
    im = heat(ax[1], M2, N, xl, yl, fmt="{:.0f}%")
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="miss 비율 [%]")
    ax[1].set_xlabel("6초 |Δh| [°]"); ax[1].set_title("miss 비율")
    fig.suptitle("v0 × 6초 방향변화 — 칸마다 값과 표본 수, 흐린 칸 n < 50", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    C.savefig(fig, out / "s6_heat_v0_dh.png")
    SUM["heat_v0_dh"] = {"rows": yl, "cols": xl, "minade": np.round(M, 3).tolist(), "n": N.tolist()}

    # S7 열지도 n_distinct × 상황
    fig, ax = plt.subplots(figsize=(13, 5.4))
    M, N = grid_mean(df, "n_distinct", [1, 2, 3, 4, 5, 6], "cls", C.CLASSES)
    im = heat(ax, M, N, C.CLASSES, ["1", "2", "3", "4", "5", "6"])
    fig.colorbar(im, ax=ax, shrink=0.85, label="평균 minADE6 [m]")
    ax.set_xlabel("상황"); ax.set_ylabel("구별 분기 n_distinct")
    ax.set_title("구별 분기 수 × 상황 — 평균 minADE6 과 표본 수 (흐린 칸 n < 50)", loc="left")
    fig.tight_layout()
    C.savefig(fig, out / "s7_heat_ndistinct_class.png")
    SUM["heat_nd_cls"] = {"minade": np.round(M, 3).tolist(), "n": N.tolist()}


# --------------------------------------------------------------------------- 4절
def sec4(df, P, R, meta, out):
    plt = C.setup_mpl()
    w = meta["loss_weights"]
    labs = cls_labels(df)
    # W1 손실 항 누적
    terms = [("l_sl1", "smoothL1", 1.0), ("l_ce", "CE", 1.0), ("l_hinge", f"hinge ×{w['offlane']:g}", w["offlane"]),
             ("l_jit", f"jitter ×{w['smooth']:g}", w["smooth"])]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    left = np.zeros(9)
    SUM["loss_by_class"] = {}
    for (col, lab, wt), colr in zip(terms, LOSS_COLS):
        v = np.array([df.loc[df.cls == c, col].mean() * wt for c in C.CLASSES])
        ax[0].barh(range(9), v, left=left, height=0.62, color=colr, label=lab, edgecolor=C.SURF, linewidth=1.5)
        left += v
        for c, x in zip(C.CLASSES, v):
            SUM["loss_by_class"].setdefault(c, {})[col] = float(x)
    for i, t in enumerate(left):
        jt = df.loc[df.cls == C.CLASSES[i], "l_jit"].mean() * w["smooth"]
        ax[0].text(t, i, f"  {t:.2f}  (jitter {jt:.4f})", va="center", fontsize=7.5, color=C.INK2)
    ax[0].set_yticks(range(9)); ax[0].set_yticklabels(labs); ax[0].set_ylim(8.6, -0.6)
    ax[0].set_xlim(0, left.max() * 1.35)
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, borderaxespad=0.2)
    ax[0].grid(axis="y", visible=False)
    ax[0].set_title("상황별 평균 손실 항 (학습식 그대로, 시나리오 평균)", pad=26)
    ax[0].set_xlabel("손실")
    # 비중
    share = np.array([[df.loc[df.cls == c, col].mean() * wt for col, _, wt in terms] for c in C.CLASSES])
    share = share / share.sum(1, keepdims=True) * 100
    lf = np.zeros(9)
    for k, colr in enumerate(LOSS_COLS):
        ax[1].barh(range(9), share[:, k], left=lf, height=0.62, color=colr, edgecolor=C.SURF, linewidth=1.5)
        for i in range(9):
            if share[i, k] > 7:
                ax[1].text(lf[i] + share[i, k] / 2, i, f"{share[i, k]:.0f}%", ha="center", va="center",
                           fontsize=7.5, color="white")
        lf += share[:, k]
    ax[1].set_yticks(range(9)); ax[1].set_yticklabels([""] * 9); ax[1].set_ylim(8.6, -0.6)
    ax[1].set_xlim(0, 100); ax[1].set_xlabel("비중 [%]"); ax[1].grid(axis="y", visible=False)
    ax[1].set_title("손실 합 대비 비중", pad=26)
    note(fig, "시나리오 손실 = train_v4.loss_fn + jitter 를 시나리오 하나에 적용 (첫 배치에서 학습 코드 값과 4.8e-7 차로 일치 확인). "
              "hinge 는 버려진 모드만 평균.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "w1_loss_by_class.png")

    # W2 CE vs minADE6
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    x, y = df["minade"].to_numpy(), df["l_ce"].to_numpy()
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("b", C.BLUE_RAMP)
    hb = ax[0].hexbin(np.clip(x, 0, 10), np.clip(y, 0, 4), gridsize=45, cmap=cm, bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax[0], label="시나리오 수 (log)")
    qs = np.percentile(x, np.arange(0, 101, 10))
    mids, meds = [], []
    for a_, b_ in zip(qs[:-1], qs[1:]):
        m = (x >= a_) & (x <= b_)
        mids.append(np.median(x[m])); meds.append(np.median(y[m]))
    ax[0].plot(np.clip(mids, 0, 10), meds, color=C.C_ORANGE, lw=2, marker="o", ms=5, label="minADE6 10분위별 CE 중앙")
    rho = spearmanr(x, y).statistic
    ax[0].set_xlabel("minADE6 [m] (10 m 에서 자름)"); ax[0].set_ylabel("CE (4 에서 자름)")
    ax[0].set_title(f"CE 와 minADE6 — Spearman ρ = {rho:.2f}", loc="left")
    ax[0].legend(loc="upper right")
    eq = df["win_eq_top1"].to_numpy()
    boxes(ax[1], [y[eq], y[~eq]], [f"승자 = 1위", f"승자 ≠ 1위"])
    ax[1].set_ylabel("CE"); ax[1].set_title("CE 분포 — 승자가 1위일 때와 아닐 때", loc="left")
    fig.tight_layout()
    C.savefig(fig, out / "w2_ce_vs_minade.png")
    SUM["ce"] = {"spearman_minade_ce": float(rho), "ce_median_eq": float(np.median(y[eq])),
                 "ce_median_ne": float(np.median(y[~eq])), "ce_decile_median": [float(v) for v in meds]}

    # W3 승자≠1위 의 비용
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, col, t in ((ax[0], "minade", "minADE6"), (ax[1], "top1_ade", "확률 1위 모드 ADE")):
        for msk, lab, colr in ((eq, f"승자 = 1위 (n={eq.sum():,})", C.C_BLUE), (~eq, f"승자 ≠ 1위 (n={(~eq).sum():,})", C.C_ORANGE)):
            v = np.sort(df.loc[msk, col].to_numpy())
            a.plot(v, np.arange(1, len(v) + 1) / len(v), color=colr, lw=2, label=f"{lab} · 중앙 {np.median(v):.2f} m")
        a.set_xscale("log"); a.set_xlabel(f"{t} [m] (log)"); a.set_ylabel("누적 비율")
        a.set_title(f"{t} 누적분포", loc="left"); a.legend(loc="upper left")
    dec = np.digitize(x, np.percentile(x, np.arange(10, 100, 10)))
    ks = [int((~eq[dec == d]).sum()) for d in range(10)]
    ns = [int((dec == d).sum()) for d in range(10)]
    rate_bars(ax[2], ks, ns, [f"{d * 10}–{d * 10 + 10}" for d in range(10)], ref=(~eq).mean())
    for tx in list(ax[2].texts):
        tx.set_visible(False)
    ax[2].set_xlabel("minADE6 백분위 구간"); ax[2].set_ylabel("승자 ≠ 1위 [%]")
    ax[2].set_title("minADE6 구간별 '승자 ≠ 1위' 비율", loc="left")
    plt.setp(ax[2].get_xticklabels(), rotation=30, ha="right")
    note(fig, "승자 = 끝점 오차가 가장 작은 모드(학습 WTA 와 같은 정의). 1위 ADE 는 모델 확률만 믿고 한 궤적을 고를 때의 오차다.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "w3_top1_mismatch.png")
    SUM["top1"] = {"win_eq_top1": float(eq.mean()),
                   "minade_median_eq": float(np.median(x[eq])), "minade_median_ne": float(np.median(x[~eq])),
                   "top1ade_median_eq": float(df.loc[eq, "top1_ade"].median()),
                   "top1ade_median_ne": float(df.loc[~eq, "top1_ade"].median()),
                   "top1ade_mean": float(df["top1_ade"].mean()), "top1fde_mean": float(df["top1_fde"].mean()),
                   "ne_by_decile": [k / n for k, n in zip(ks, ns)]}

    # W4 안좋음 그룹의 오차 출처
    gnames = [g[0] for g in GROUPS]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    SUM["groups"] = {}
    rows = [("route_err", "분기 선택 오류\n(1위 경로 ≠ 정답 경로)"), ("win_route_err", "승자 경로 ≠ 정답 경로"),
            ("cov_fail", "경로 커버리지 실패"), ("miss", "miss")]
    d2 = df.assign(cov_fail=~df["coverage"])
    width = 0.2
    for k, (col, lab) in enumerate(rows):
        for j, g in enumerate(gnames):
            sub = d2[(d2.grp == g) & ((d2.n_distinct >= 2) if col in ("route_err", "win_route_err") else True)]
            kk, nn = int(sub[col].sum()), len(sub)
            lo, hi = wilson(kk, nn)
            ax[0].bar(k + (j - 1.5) * width, 100 * kk / nn, width=width * 0.92, color=C.SERIES[j],
                      label=g if k == 0 else None, zorder=3)
            ax[0].plot([k + (j - 1.5) * width] * 2, [100 * lo, 100 * hi], color=C.INK, lw=0.9, zorder=4)
            SUM["groups"].setdefault(g, {})[col] = kk / nn
    ax[0].set_xticks(range(len(rows))); ax[0].set_xticklabels([r[1] for r in rows], fontsize=8)
    ax[0].set_ylabel("%"); ax[0].legend(title="minADE6 그룹", loc="upper left")
    ax[0].set_title("그룹별 실패 유형 비율 (분기 오류 2종은 구별 분기 ≥ 2 만)", loc="left")
    bad = d2[d2.grp == "안좋음"]
    ds, dd = bad["ds_end"].to_numpy(), bad["dd_end"].to_numpy()
    hb = ax[1].hexbin(np.clip(ds, -40, 40), np.clip(dd, -15, 15), gridsize=40, cmap=cm, mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax[1], label="시나리오 수 (log)")
    ax[1].axhline(0, color=C.INK2, lw=0.8); ax[1].axvline(0, color=C.INK2, lw=0.8)
    ax[1].set_xlabel("Δs = 예측 − 정답 [m] (+ 앞질러 감)"); ax[1].set_ylabel("Δd = 예측 − 정답 [m] (+ 좌)")
    fs = (np.abs(ds) > np.abs(dd)).mean()
    ax[1].set_title(f"안좋음(백분위 90–99) 승자 끝점 오차 분해 — 종방향 우세 {fs * 100:.0f}%", loc="left")
    for j, g in enumerate(gnames):
        sub = d2[d2.grp == g]
        a_s, a_d = sub["ds_end"].abs().median(), sub["dd_end"].abs().median()
        ax[2].bar(j - 0.18, a_s, width=0.34, color=C.C_BLUE, label="|Δs| 중앙 (종방향)" if j == 0 else None, zorder=3)
        ax[2].bar(j + 0.18, a_d, width=0.34, color=C.C_ORANGE, label="|Δd| 중앙 (횡방향)" if j == 0 else None, zorder=3)
        ax[2].text(j - 0.18, a_s, f"{a_s:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax[2].text(j + 0.18, a_d, f"{a_d:.1f}", ha="center", va="bottom", fontsize=7.5)
        SUM["groups"][g].update({"n": len(sub), "abs_ds_median": float(a_s), "abs_dd_median": float(a_d),
                                 "lon_dominant": float((sub["ds_end"].abs() > sub["dd_end"].abs()).mean()),
                                 "ds_median": float(sub["ds_end"].median())})
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels([f"{g}\n(n={int((d2.grp == g).sum()):,})" for g in gnames])
    ax[2].set_ylabel("[m]"); ax[2].legend(loc="upper left")
    ax[2].set_title("그룹별 승자 끝점 오차의 종/횡 성분", loc="left")
    note(fig, "Δs, Δd 는 승자 모드가 탄 후보 경로의 Frenet 좌표(예측은 적분기 상태, 정답은 lane_frame.to_frame 투영). "
              "그룹 = minADE6 백분위 풀 전체(그림으로 고른 9개가 아니라).")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "w4_error_sources.png")

    # W5 라벨 끝 인공 감속
    N = len(df)
    ar = np.arange(N)
    pos = R["pos"].astype(np.float64)
    step = np.linalg.norm(np.diff(pos, axis=1), axis=2) / C.DT          # (N,109) step t -> t+1
    fut = step[:, C.OBS - 1:]                                             # k=0..59 : 49+k -> 50+k
    vref = fut[:, 40:50].mean(1)
    mv = vref > 5.0
    win, top = df["winner"].to_numpy(), df["top1"].to_numpy()
    vw = P["v"][ar, win].astype(np.float64); vt = P["v"][ar, top].astype(np.float64)
    rw, rt = vw[:, 40:50].mean(1), vt[:, 40:50].mean(1)
    ok = mv & (rw > 1) & (rt > 1)
    k = np.arange(30, 60)
    tt = (k + 1) * C.DT
    g_rat = np.median(fut[ok][:, k] / vref[ok, None], axis=0)
    vf = R["v_fld"][:, C.OBS - 1:].astype(np.float64)          # 속도 필드(평활) — t=49+k
    vfr = vf[:, 40:50].mean(1)
    okf = ok & (vfr > 1)
    f_rat = np.median(vf[okf][:, k] / vfr[okf, None], axis=0)
    w_rat = np.median(vw[ok][:, k] / rw[ok, None], axis=0)
    t_rat = np.median(vt[ok][:, k] / rt[ok, None], axis=0)
    past = step[:, :C.OBS - 1]
    p_rat = np.median(past[ok][:, :15] / past[ok][:, 15:25].mean(1, keepdims=True), axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.0))
    ax[0].plot(tt, g_rat, color=C.C_GT, lw=2, marker="o", ms=3, label="정답 (위치 차분)")
    ax[0].plot(tt, f_rat, color=C.MUTED, lw=1.4, ls=(0, (3, 2)), label="정답 (AV2 속도 필드·평활)")
    ax[0].plot(tt, w_rat, color=C.C_WIN, lw=2, label="승자 모드 v")
    ax[0].plot(tt, t_rat, color=C.C_TOP, lw=2, label="확률 1위 모드 v")
    ax[0].axhline(1, color=C.INK2, lw=0.8, ls=(0, (4, 2)))
    ax[0].set_xlabel("예측 시간 [s]"); ax[0].set_ylabel("속력 / (4.1–5.0 s 평균)")
    ax[0].set_title("예측 끝 1.5초의 속력 비 (중앙값)", loc="left"); ax[0].legend(loc="lower left")
    ax[1].plot(np.arange(1, 16) * C.DT - 5.0, p_rat, color=C.C_GT, lw=2, marker="o", ms=3, label="정답 과거 창 시작")
    ax[1].axhline(1, color=C.INK2, lw=0.8, ls=(0, (4, 2)))
    ax[1].set_xlabel("시간 [s]"); ax[1].set_ylabel("속력 / (−3.5~−2.6 s 평균)")
    ax[1].set_title("과거 창 시작에도 같은 모양 — 11초 창 양 끝의 인공물", loc="left")
    ax[1].legend(loc="lower right")
    dfc = lambda s, r: ((r[:, None] - s[:, 50:60]) * C.DT).sum(1)
    dg, dw_, dt_ = dfc(fut[ok], vref[ok]), dfc(vw[ok], rw[ok]), dfc(vt[ok], rt[ok])
    vb = [5, 10, 15, 20, 40]
    v_ok = vref[ok]
    by_v = {}
    for j, (lab, dd_, colr) in enumerate((("정답", dg, C.C_GT), ("승자", dw_, C.C_WIN), ("1위", dt_, C.C_TOP))):
        ys = [np.median(dd_[(v_ok >= a_) & (v_ok < b_)]) for a_, b_ in zip(vb[:-1], vb[1:])]
        by_v[lab] = [round(float(y_), 3) for y_ in ys]
        ax[2].bar(np.arange(4) + (j - 1) * 0.26, ys, width=0.24, color=colr, label=lab, zorder=3)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels([f"{a_}–{b_}" for a_, b_ in zip(vb[:-1], vb[1:])])
    ax[2].set_xlabel("4.1–5.0 s 정답 속력 [m/s]"); ax[2].set_ylabel("마지막 1초 부족 거리 [m] (중앙)")
    ax[2].set_title("등속 대비 마지막 1초에 덜 간 거리", loc="left"); ax[2].legend(loc="upper left")
    fig.suptitle(f"라벨 끝 인공 감속 — 모델이 그대로 배웠다 (이동 시나리오 n={int(ok.sum()):,}: 4.1–5.0 s 정답 속력 > 5 m/s)",
                 x=0.01, ha="left")
    note(fig, f"AV2 속도 필드(평활)의 같은 비는 {f_rat[-10:].min():.3f}~{f_rat[-10:].max():.3f} 로 감속이 없다. "
              f"위치 궤적만 끝 스텝 속력이 {g_rat[-1]:.2f} 배로 떨어진다.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    C.savefig(fig, out / "w5_label_end_artifact.png")
    at = P["a"][ar, top].astype(np.float64)
    SUM["end_artifact"] = {"n": int(ok.sum()), "gt_ratio_k50_59": g_rat[-10:].round(3).tolist(),
                           "win_ratio_k50_59": w_rat[-10:].round(3).tolist(),
                           "top_ratio_k50_59": t_rat[-10:].round(3).tolist(),
                           "gt_deficit_median": float(np.median(dg)), "win_deficit_median": float(np.median(dw_)),
                           "top_deficit_median": float(np.median(dt_)),
                           "top_a_last6_mean": at[ok][:, 54:].mean(0).round(2).tolist(),
                           "top_a_k40_50_mean": float(at[ok][:, 40:50].mean()),
                           "past_start_ratio": p_rat[:5].round(3).tolist(),
                           "field_ratio_k50_59": f_rat[-10:].round(3).tolist(),
                           "deficit_by_speed": {"bins": [f"{a_}-{b_}" for a_, b_ in zip(vb[:-1], vb[1:])],
                                                "n": [int(((v_ok >= a_) & (v_ok < b_)).sum())
                                                      for a_, b_ in zip(vb[:-1], vb[1:])], **by_v}}

    # W6 시작 잔차각 θ0
    th0 = np.degrees(np.abs(P["theta"][ar, top, 0].astype(np.float64)))
    thg = np.degrees(np.abs(C.wrap(R["h"][:, C.OBS - 1] - R["gt_k_w"][:, C.OBS - 1])))
    bins = [0, 5, 15, 30, 60, 91]
    blab = ["<5", "5–15", "15–30", "30–60", "60–90"]
    b = np.digitize(th0, bins[1:-1])
    v0 = df["v0"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    boxes(ax[0], [df["minade"].to_numpy()[b == j] for j in range(5)], blab)
    ax[0].set_ylim(0, 10); ax[0].axhline(df["minade"].mean(), color=C.INK2, lw=1.0, ls=(0, (4, 2)))
    ax[0].set_xlabel("확률 1위 모드의 시작 잔차각 |θ0| [°]"); ax[0].set_ylabel("minADE6 [m]")
    ax[0].set_title("시작 잔차각이 클수록 오차가 크다", loc="left", pad=16)
    ks = [int((v0[b == j] < 2).sum()) for j in range(5)]
    ns = [int((b == j).sum()) for j in range(5)]
    rate_bars(ax[1], ks, ns, blab)
    ax[1].set_xlabel("|θ0| [°]"); ax[1].set_ylabel("v0 < 2 m/s 비율 [%]")
    ax[1].set_title("큰 |θ0| 는 대부분 저속 출발", loc="left", pad=16)
    hb = ax[2].hexbin(thg, th0, gridsize=40, cmap=cm, bins="log", mincnt=1, extent=(0, 180, 0, 90))
    fig.colorbar(hb, ax=ax[2], label="시나리오 수 (log)")
    r_ = np.corrcoef(th0, thg)[0, 1]
    ax[2].set_xlabel("정답 |θ| at t=0 (승자 경로 기준) [°]"); ax[2].set_ylabel("예측 |θ0| [°]")
    ax[2].set_title(f"예측 θ0 와 정답 θ(t=0) — 상관 {r_:.2f}", loc="left")
    note(fig, "θ0 = wrap(h0 − k(s0)) (θ0 guard: |값|>90° 면 −k(s0)). h0 는 관측 마지막 위치차분 방향이라 저속에서 잡음이 크다. "
              "첫 스텝 θ 로 쟀다(|dθ| ≤ 8°).")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "w6_start_theta.png")
    SUM["theta0"] = {"bins": blab, "n": ns, "minade_mean": [float(df["minade"].to_numpy()[b == j].mean()) for j in range(5)],
                     "v0_lt2_share": [k_ / n_ for k_, n_ in zip(ks, ns)], "corr_with_gt": float(r_),
                     "share_ge15": float((th0 >= 15).mean())}


# --------------------------------------------------------------------------- 5절
def sec5(df, out):
    plt = C.setup_mpl()
    labs = cls_labels(df)
    multi = df[df["n_alive"] >= 2]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    boxes(ax[0], [multi.loc[multi.cls == c, "spread"].to_numpy() for c in C.CLASSES],
          [f"{c} (n={int((multi.cls == c).sum()):,})" for c in C.CLASSES], vert=False)
    ax[0].set_xlabel("모드 끝점 퍼짐 [m]"); ax[0].set_title("상황별 끝점 퍼짐", loc="left")
    ax[0].grid(axis="y", visible=False)
    boxes(ax[1], [multi.loc[multi.n_distinct == k, "spread"].to_numpy() for k in range(1, 7)],
          [str(k) for k in range(1, 7)])
    ax[1].set_xlabel("구별 분기 n_distinct"); ax[1].set_ylabel("끝점 퍼짐 [m]")
    ax[1].set_title("분기 수별 끝점 퍼짐", loc="left", pad=16)
    boxes(ax[2], [multi.loc[multi.v0_bin == l, "spread"].to_numpy() for l in C.BINS["v0"][1]], C.BINS["v0"][1])
    ax[2].set_xlabel("v0 [m/s]"); ax[2].set_ylabel("끝점 퍼짐 [m]")
    ax[2].set_title("속력별 끝점 퍼짐", loc="left", pad=16)
    note(fig, "끝점 퍼짐 = 살아있는 모드 끝점의 쌍별 평균 거리. 살아있는 모드가 1개인 폴백(944개)은 뺐다.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "d1_endpoint_spread.png")
    SUM["spread"] = {"median_all": float(multi["spread"].median()),
                     "by_nd_median": {k: float(multi.loc[multi.n_distinct == k, "spread"].median()) for k in range(1, 7)},
                     "by_v0_median": {l: float(multi.loc[multi.v0_bin == l, "spread"].median()) for l in C.BINS["v0"][1]}}

    # D2 분기 덮기
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    nds = list(range(1, 7))
    M = np.zeros((6, 6)); Nn = np.zeros((6, 6), int)
    for i, k in enumerate(nds):
        sub = multi[multi.n_distinct == k]
        for j in range(6):
            Nn[i, j] = int((sub.n_end_clusters == j + 1).sum())
        M[i] = 100 * Nn[i] / max(Nn[i].sum(), 1)
    heat(ax[0], M, Nn, [str(j) for j in range(1, 7)], [f"{k} (n={int(Nn[i].sum()):,})" for i, k in enumerate(nds)],
         fmt="{:.0f}%")
    ax[0].set_xlabel("끝점 군집 수 (2.5 m 안은 같은 군집)"); ax[0].set_ylabel("구별 분기 n_distinct")
    ax[0].set_title("모드 6개가 실제로 몇 곳에 끝나나 (행 비율)", loc="left")
    boxes(ax[1], [multi.loc[multi.n_distinct == k, "eff_branches"].to_numpy() for k in nds], [str(k) for k in nds])
    ax[1].plot(range(6), nds, color=C.INK2, lw=1, ls=(0, (4, 2)), label="상한 (분기 수)")
    ax[1].set_xlabel("구별 분기 n_distinct"); ax[1].set_ylabel("유효 분기 수 = exp(경로 확률 엔트로피)")
    ax[1].set_title("확률이 몇 개 분기에 실제로 퍼지나", loc="left", pad=16); ax[1].legend(loc="upper left")
    ranks = []
    for k in nds[1:]:
        sub = multi[multi.n_distinct == k]
        r1 = (sub.gt_route_rank == 1).mean() * 100
        r2 = (sub.gt_route_rank == 2).mean() * 100
        ranks.append((k, len(sub), r1, r2, 100 - r1 - r2))
    xs = np.arange(len(ranks))
    bot = np.zeros(len(ranks))
    for jj, (lab, colr) in enumerate((("1위", C.C_BLUE), ("2위", C.C_ORANGE), ("3위 이하", C.C_AQUA))):
        vals = np.array([r[2 + jj] for r in ranks])
        ax[2].bar(xs, vals, bottom=bot, color=colr, width=0.6, label=lab, edgecolor=C.SURF, linewidth=1.5)
        for x_, b_, v_ in zip(xs, bot, vals):
            if v_ > 6:
                ax[2].text(x_, b_ + v_ / 2, f"{v_:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        bot += vals
    ax[2].set_xticks(xs); ax[2].set_xticklabels([f"{r[0]}\n(n={r[1]:,})" for r in ranks])
    ax[2].set_xlabel("구별 분기 n_distinct"); ax[2].set_ylabel("%"); ax[2].set_ylim(0, 100)
    ax[2].set_title("정답 기준 경로의 확률 순위", loc="left", pad=24)
    ax[2].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
    note(fig, "경로 확률 = 같은 후보 경로에 배정된 슬롯들의 확률 합(슬롯 i 의 경로 = i mod n_distinct). "
              "정답 기준 경로 = (밴드 안·나란함 > 나란함 > 밴드 안 > 전체) 중 정답 평균 |d| 최소.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "d2_branch_coverage.png")
    SUM["branch"] = {"clusters_row_pct": np.round(M, 1).tolist(),
                     "eff_median_by_nd": {k: float(multi.loc[multi.n_distinct == k, "eff_branches"].median()) for k in nds},
                     "p05_mean_by_nd": {k: float(multi.loc[multi.n_distinct == k, "n_branch_p05"].mean()) for k in nds},
                     "gt_rank": [{"nd": r[0], "n": r[1], "r1": r[2], "r2": r[3], "r3p": r[4]} for r in ranks],
                     "gt_rank1_all_nd2p": float((multi.loc[multi.n_distinct >= 2, "gt_route_rank"] == 1).mean())}

    # D3 경로 커버리지 실패
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    fail = ~df["coverage"]
    ks = [int(fail[df.cls == c].sum()) for c in C.CLASSES]
    ns = [int((df.cls == c).sum()) for c in C.CLASSES]
    rate_bars(ax[0], ks, ns, labs, vert=False, ref=fail.mean())
    ax[0].set_xlabel("%"); ax[0].grid(axis="y", visible=False)
    ax[0].set_title(f"경로 커버리지 실패 — 전체 {fail.mean() * 100:.1f}%", loc="left")
    g = [df.loc[~fail, "minade"].to_numpy(), df.loc[fail, "minade"].to_numpy()]
    boxes(ax[1], g, ["커버리지 있음", "실패"])
    ax[1].set_ylim(0, 10); ax[1].set_ylabel("minADE6 [m]")
    ax[1].set_title("커버리지 실패 때의 minADE6", loc="left", pad=16)
    lev = [("폴백", df.fallback)] + [(str(k), (~df.fallback) & (df.n_distinct == k)) for k in range(1, 7)]
    ks = [int((fail & m).sum()) for _, m in lev]
    ns = [int(m.sum()) for _, m in lev]
    rate_bars(ax[2], ks, ns, [l for l, _ in lev], ref=fail.mean())
    ax[2].set_xlabel("구별 분기 n_distinct"); ax[2].set_ylabel("커버리지 실패 [%]")
    ax[2].set_title("분기 수별 커버리지 실패", loc="left", pad=16)
    note(fig, "커버리지 실패 = 정답 미래 60스텝이 어느 후보 경로의 규칙 밴드(좌/우 한계) 안에도 끝까지 들지 않음 "
              "(route_recall.py 의 '규칙밴드' 정의, 64점 리샘플 경로 기준).")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    C.savefig(fig, out / "d3_route_coverage.png")
    SUM["coverage"] = {"fail": float(fail.mean()), "minade_mean_ok": float(g[0].mean()),
                       "minade_mean_fail": float(g[1].mean()), "minade_median_ok": float(np.median(g[0])),
                       "minade_median_fail": float(np.median(g[1])),
                       "fail_by_nd": {l: k / n for (l, _), k, n in zip(lev, ks, ns)},
                       "share_of_bad_pool": float(fail[df.grp == "안좋음"].mean()),
                       "share_of_worst_pool": float(fail[df.grp == "최악"].mean())}


def misc(df):
    o = df[df.cls == "기타"]
    dv = o["vend_f"] - o["v0_f"]
    SUM["other_breakdown"] = {"n": len(o), "v0_le2": float((o.v0_f <= C.CONST_V0).mean()),
                              "gentle_acc": float(((o.v0_f > C.CONST_V0) & (dv >= C.CONST_DV)).mean()),
                              "gentle_dec": float(((o.v0_f > C.CONST_V0) & (dv <= -C.CONST_DV)).mean()),
                              "curve_15_30": float(((o.dh6.abs() >= C.LC_MAX_DEG) & (o.dh6.abs() <= C.TURN_DEG)).mean())}
    acc = df[df.cls == "급가속"]
    SUM["acc_from_stop"] = float((acc.v0_f < 1.0).mean())
    SUM["overall"] = {"n": len(df), "minade": float(df.minade.mean()), "minfde": float(df.minfde.mean()),
                      "miss": float(df.miss.mean()), "win_eq_top1": float(df.win_eq_top1.mean()),
                      "route_err_nd2p": float(df.loc[df.n_distinct >= 2, "route_err"].mean()),
                      "coverage": float(df.coverage.mean()), "has_lead49": float(df.has_lead49.mean()),
                      "thw_median": float(np.nanmedian(df.thw49)), "gt_route_q": df.gt_route_q.value_counts().to_dict(),
                      "lc_ok": float(df.lc_ok.mean()),
                      "loss_mean": {k: float(df[k].mean()) for k in ("l_sl1", "l_ce", "l_hinge", "l_jit", "l_total")},
                      "spread_median": float(df.loc[df.n_alive >= 2, "spread"].median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=C.DEFAULT_TAG)
    a = ap.parse_args()
    import pandas as pd
    dirs = C.tag_dirs(a.tag)
    df = prep(pd.read_parquet(dirs["data"] / "scenarios.parquet"))
    meta = json.loads((dirs["data"] / "meta.json").read_text())
    P = np.load(dirs["data"] / "pred.npz")
    R = np.load(dirs["data"] / "raw.npz")
    Pd = {k: P[k] for k in ("v", "a", "theta")}
    Rd = {k: R[k] for k in ("pos", "h", "gt_k_w", "v_fld")}
    sec3(df, dirs["stats"])
    print("[stats] 3절", flush=True)
    sec4(df, Pd, Rd, meta, dirs["why"])
    print("[stats] 4절", flush=True)
    sec5(df, dirs["div"])
    print("[stats] 5절", flush=True)
    misc(df)
    SUM["tag"] = a.tag
    SUM["repro"] = meta["repro"]
    (dirs["data"] / "summary.json").write_text(json.dumps(SUM, indent=2, ensure_ascii=False, default=float))
    print(f"[stats] -> {dirs['data'] / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
