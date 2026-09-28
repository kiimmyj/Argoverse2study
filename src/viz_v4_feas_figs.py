"""
viz_v4_feas_figs.py - viz_v4_feas.py / viz_v4_labelfit.py 가 만든 숫자로 그림을 그린다.

**모집단 규칙 (2026-09-29 독립 검토 반영):** 기본은 **top-1 모드(실제 예측)** 다.
살아있는 모드 6슬롯은 "6모드 평균 = 상한"으로 따로 표시한다 — 모집단을 섞으면 결론이 뒤집힌다.

필수 요건: 단위 통일(초·초당) · 판별 가능한 색·표식 · 스텝 점 · 묶음 그래프(조건별 small multiples) ·
칸마다 표본 수(n<50 흐리게) · 결정 트리(분석용·처리 로직용) · 대표 시나리오 · 결과 특성 표.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C
import viz_v4_feas as FE

OUT = FE.OUT
DATA = FE.DATA
MAIN_TAG = "v4_l4nw_ah2_full_sm1_cos30_s0"
POP = "top1"                 # 리포트·그림의 기본 모집단
POP_LAB = {"top1": "top-1 모드 (실제 예측)", "alive": "살아있는 모드 6슬롯 (상한)"}
# 세 정의를 쓰는 칸은 판 색과 겹치면 안 된다 — 서수 파랑 램프를 쓴다 (①→③ 자기참조에서 좌표로)
DEF_RAMP = ["#b7d3f6", "#3987e5", "#0d366b"]


def load():
    res = json.loads((DATA / "feas.json").read_text())
    lf = json.loads((DATA / "labelfit.json").read_text()) if (DATA / "labelfit.json").exists() else None
    dz = np.load(DATA / "dists.npz")
    return res, lf, dz


def series(res):
    out = []
    for tag, name, sm, sched, hz, col, mk in FE.TAGS:
        if tag in res["tags"]:
            out.append((tag, name, col, mk, sm, sched, hz))
    return out


def finish(fig, top=0.86):
    fig.tight_layout()
    fig.subplots_adjust(top=top)


def vlabel(ax, x, y, txt, top, logy=False, fontsize=7.2, rot=0, color=None):
    """막대 위 값 글. 좁은 묶음 막대는 rot=90 으로 세워 겹침을 막는다."""
    yy = y * 1.10 if (logy and y > 0) else y + 0.025 * top
    ax.text(x, yy, txt, ha="center", va="bottom", fontsize=fontsize, rotation=rot,
            color=color or C.INK2, zorder=12)


def barpanel(ax, labels, vals, cols, ns=None, gt=None, title="", ylab="", fmt="{:.2f}",
             logy=False, gt_lab="정답 라벨", gt_side="left", rot=0):
    """막대 한 칸. gt 를 주면 검은 점선으로 정답 기준선을 긋고 글은 막대 값과 겹치지 않는 쪽에 둔다."""
    x = np.arange(len(labels))
    vals = [float(v) for v in vals]
    for i, (v, c) in enumerate(zip(vals, cols)):
        al = 0.35 if (ns is not None and C.faded(ns[i])) else 1.0
        ax.bar(x[i], v, width=0.68, color=c, alpha=al, edgecolor=C.SURF, lw=0.6, zorder=3)
    pos = [v for v in vals if np.isfinite(v) and v > 0]
    if gt is not None and np.isfinite(gt) and gt > 0:
        pos += [gt]
    top = max(pos) if pos else 1.0
    lo = min(pos) if pos else 0.1
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(lo * 0.25, top * 6.0)
    else:
        ax.set_ylim(0, top * 1.38)
    if gt is not None and np.isfinite(gt):
        ax.axhline(gt, color=C.INK, ls=(0, (4, 2)), lw=1.4, zorder=4)
        # 정답선 글은 막대 위 글과 안 겹치게 축 **아래쪽 빈 곳**에 붙인다
        ax.annotate(f"{gt_lab} {fmt.format(gt)}", xy=(0.012, gt), xycoords=("axes fraction", "data"),
                    xytext=(0, -9), textcoords="offset points", fontsize=7.0, color=C.INK,
                    va="top", ha="left", zorder=6,
                    bbox=dict(facecolor=C.SURF, edgecolor="none", pad=0.6, alpha=0.92))
    for i, v in enumerate(vals):
        if np.isfinite(v):
            vlabel(ax, x[i], v, fmt.format(v), top, logy, rot=rot)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6, rotation=22, ha="right")
    ax.set_title(title, fontsize=9.6)
    ax.set_ylabel(ylab)
    ax.grid(axis="x", visible=False)


def set_row_ylim(ax0, vals, logy):
    pos = [float(v) for v in vals if np.isfinite(v) and v > 0]
    if not pos:
        return
    if logy:
        ax0.set_ylim(min(pos) * 0.25, max(pos) * 6.0)
    else:
        ax0.set_ylim(0, max(pos) * 1.38)


# ============================================================= 측정 1
def fig_m1_violations(res, plt):
    """두 모집단을 나란히. top-1 이 실제 예측이고, 6슬롯 평균은 상한이다."""
    S = series(res)
    names = [s[1] for s in S]
    metrics = [("over_label_pct", f"|Δψ| > {FE.LABEL_DTHETA_DEG}°/step\n(라벨 상한 · 기존 지표와 같은 임계)"),
               ("R_under3_pct", f"회전반경 R = v/|ψ̇| < {FE.R_MIN:g} m\n(DKM)"),
               ("R3pt_under3_pct", f"연속 3점 외접원 반경 < {FE.R_MIN:g} m\n(같은 양의 다른 표현)"),
               ("yawrate_over_pct", f"|요레이트| > {FE.YAWRATE_MAX} rad/s\n(nuPlan ego_is_comfortable)"),
               ("yawacc_over_pct", f"|요가속| > {FE.YAWACC_MAX} rad/s²\n(nuPlan)")]
    fig, axes = plt.subplots(1, 5, figsize=(18.4, 5.1))
    x = np.arange(len(S)); w = 0.38
    for ax, (k, ti) in zip(axes, metrics):
        v1 = [res["tags"][s[0]]["m1"]["top1"][k] for s in S]
        v2 = [res["tags"][s[0]]["m1"]["alive"][k] for s in S]
        for i, s in enumerate(S):
            ax.bar(x[i] - w / 2, v1[i], width=w, color=s[2], edgecolor=C.SURF, lw=0.5, zorder=3)
            ax.bar(x[i] + w / 2, v2[i], width=w, color=s[2], alpha=0.42, hatch="///",
                   edgecolor=C.SURF, lw=0.5, zorder=3)
            vlabel(ax, x[i] - w / 2, v1[i], f"{v1[i]:.2f}", max(v1 + v2), True, 6.6, rot=90)
            vlabel(ax, x[i] + w / 2, v2[i], f"{v2[i]:.2f}", max(v1 + v2), True, 6.6, rot=90,
                   color=C.MUTED)
        g = res["gt"]["m1"][k]
        ax.axhline(g, color=C.INK, ls=(0, (4, 2)), lw=1.5, zorder=4)
        ax.annotate(f"정답 라벨 {g:.2f}", xy=(0.012, g), xycoords=("axes fraction", "data"),
                    xytext=(0, -9), textcoords="offset points", fontsize=7.2, color=C.INK,
                    va="top", ha="left", bbox=dict(facecolor=C.SURF, edgecolor="none", pad=0.6))
        ax.set_yscale("log")
        allv = [v for v in v1 + v2 + [g] if v > 0]
        ax.set_ylim(min(allv) * 0.3, max(allv) * 30.0)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.6, rotation=22, ha="right")
        ax.set_ylabel("위반 step 비율 [%]  (로그)")
        ax.set_title(ti, fontsize=9.4)
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=C.MUTED, label=POP_LAB["top1"]),
                            Patch(facecolor=C.MUTED, alpha=0.42, hatch="///", label=POP_LAB["alive"])],
                   fontsize=7.0, loc="upper left")
    n1 = res["tags"][S[0][0]]["m1"]["top1"]["n_step_pair"]
    n2 = res["tags"][S[0][0]]["m1"]["alive"]["n_step_pair"]
    fig.suptitle("측정 1 — 예측 **좌표**에서 복원한 운동학 기준 위반율 (val 24,988)\n"
                 f"heading = 연속 두 점 차분 · |Δp|/0.1 s ≥ {FE.MOVE_V:g} m/s 인 스텝만 유효 · "
                 f"이웃 두 스텝이 모두 유효할 때만 Δ를 센다 (유효 쌍 top-1 {n1:,} · 6슬롯 {n2:,}) · 세로축 로그\n"
                 "**R<3 m 은 top-1 으로 재면 정답보다 낮다 — 모집단을 섞으면 결론이 뒤집힌다**",
                 fontsize=11.2)
    finish(fig, 0.76)
    return C.savefig(fig, OUT / "m1_1_violation_bars.png")


def fig_m1_self_vs_xy(res, plt):
    S = series(res)
    names = [s[1] for s in S]
    fig, axes = plt.subplots(1, 4, figsize=(19.6, 5.2))
    x = np.arange(len(S)); w = 0.27
    M = lambda t, k: res["tags"][t]["m1"][POP][k]
    self_v = [M(s[0], "self_dtheta_over_pct_sameden") for s in S]
    aux_v = [M(s[0], "aux_dh_over_pct_sameden") for s in S]
    xy_v = [M(s[0], "over_label_pct") for s in S]
    fmt = lambda v: (f"{v:.1e}" if v < 0.01 else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}"))

    ax = axes[0]
    for j, (v, lab) in enumerate([(self_v, "① 자기참조 |Δθ|"),
                                  (aux_v, "② 내부 |Δh| = |Δk + Δθ|"),
                                  (xy_v, "③ 복원 |Δψ| (좌표)")]):
        ax.bar(x + (j - 1) * w, np.maximum(v, 1e-4), width=w, color=DEF_RAMP[j], label=lab,
               edgecolor=C.SURF, lw=0.6)
        for xi, vi in zip(x + (j - 1) * w, v):
            ax.text(xi, max(vi, 1e-4) * 1.14, fmt(vi), ha="center", va="bottom", fontsize=6.6,
                    rotation=90, color=C.INK2)
    g = res["gt"]["m1"]["over_label_pct"]
    ax.axhline(g, color=C.INK, ls=(0, (4, 2)), lw=1.4)
    ax.annotate(f"정답 라벨 {g:.2f}", xy=(0.012, g), xycoords=("axes fraction", "data"),
                xytext=(0, -9), textcoords="offset points", fontsize=7.2, color=C.INK, va="top",
                ha="left", bbox=dict(facecolor=C.SURF, edgecolor="none", pad=0.6))
    ax.set_yscale("log"); ax.set_ylim(2e-4, 400)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel(f"|Δ| > {FE.LABEL_DTHETA_DEG}°/step 인 step 비율 [%]  (로그)")
    ax.set_title("같은 분모(유효 이웃 쌍)에서 세 정의를 나란히", fontsize=9.6, pad=24)
    ax.legend(fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ratio = [xy_v[i] / max(self_v[i], 1e-9) for i in range(len(S))]
    for i, s in enumerate(S):
        ax.bar(i, ratio[i], width=0.66, color=s[2], edgecolor=C.SURF, lw=0.6)
        ax.text(i, ratio[i] * 1.12, f"{ratio[i]:,.1f}×", ha="center", va="bottom", fontsize=8,
                color=C.INK2)
    ax.axhline(1.0, color=C.INK, lw=1.2, ls=(0, (4, 2)))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("복원 |Δψ| 위반율 ÷ 자기참조 |Δθ| 위반율  (로그)")
    ax.set_title("자기참조 지표가 얼마나 낙관적인가\n(1 보다 크면 좌표로 재면 더 나쁘다)", fontsize=9.6)
    ax.grid(axis="x", visible=False)

    # ③ 크기: |Δψ| 를 함께 그리고 '가산 분해가 아님'을 못박는다
    ax = axes[2]
    w4 = 0.2
    keys = [("mean_abs_dpsi_deg", "|Δψ| (좌변)", C.C_RED),
            ("mean_abs_dtheta_deg", "|Δθ|", DEF_RAMP[0]),
            ("mean_abs_dk_deg", "|Δk| 경로 곡률", DEF_RAMP[1]),
            ("mean_abs_geom_deg", "|기하| = |Δψ − Δh|", DEF_RAMP[2])]
    for j, (k, lab, col) in enumerate(keys):
        v = [M(s[0], k) for s in S]
        ax.bar(x + (j - 1.5) * w4, v, width=w4, color=col, label=lab, edgecolor=C.SURF, lw=0.5)
        for xi, vi in zip(x + (j - 1.5) * w4, v):
            ax.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.0, rotation=90,
                    color=C.INK2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("유효 쌍 평균 [°/step]")
    d0 = res["tags"][MAIN_TAG]["m1"][POP]
    ax.set_title("각 항의 **크기** (가산 분해가 아니다)\n"
                 f"부호 항등식 Δψ = Δθ + Δk + 기하 는 성립(잔차 {d0['identity_max_resid_deg']:.1e}°)이지만\n"
                 f"절대값 합 {d0['sum_of_parts_deg']:.2f} ≠ |Δψ| {d0['mean_abs_dpsi_deg']:.2f}",
                 fontsize=8.8)
    ax.legend(fontsize=6.8); ax.grid(axis="x", visible=False)

    # ④ 반사실: 한 항을 지우면 위반율이 어디까지 내려가나 → 주항이 무엇인가
    ax = axes[3]
    base = [M(s[0], "over_label_pct") for s in S]
    cf = [("cf_drop_geom_over_pct", "기하 제거", DEF_RAMP[2]),
          ("cf_drop_dk_over_pct", "Δk 제거", DEF_RAMP[1]),
          ("cf_drop_dtheta_over_pct", "Δθ 제거", DEF_RAMP[0])]
    ax.bar(x - 1.5 * w4, base, width=w4, color=C.C_RED, label="원래", edgecolor=C.SURF, lw=0.5)
    for j, (k, lab, col) in enumerate(cf):
        v = [M(s[0], k) for s in S]
        ax.bar(x + (j - 0.5) * w4, v, width=w4, color=col, label=lab, edgecolor=C.SURF, lw=0.5)
    for i, s in enumerate(S):
        vals = [base[i]] + [M(s[0], k) for k, _, _ in cf]
        for j, vi in enumerate(vals):
            ax.text(x[i] + (j - 1.5) * w4, vi * 1.10, f"{vi:.2f}", ha="center", va="bottom",
                    fontsize=6.0, rotation=90, color=C.INK2)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel(f"|Δψ| > {FE.LABEL_DTHETA_DEG}°/step [%]  (로그)")
    ax.set_title("반사실 — 한 항을 0 으로 두면 위반율이 얼마가 되나\n"
                 "**가장 많이 내려가는 항이 주항이다**", fontsize=9.0, pad=22)
    ax.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              columnspacing=0.9, handlelength=1.2, handletextpad=0.4)
    ax.grid(axis="x", visible=False)

    fig.suptitle(f"측정 1 판별 — 벌점이 직접 누르는 변수(Δθ) 대신 좌표에서 재면 결과가 뒤집히는가 "
                 f"(val 24,988 · {POP_LAB[POP]})\n"
                 "모델의 내부 heading h = k(s) + θ 는 **모델이 실제로 내놓는 궤적의 heading 이 아니다** — "
                 "7.3° 지표와 흔들림 벌점이 둘 다 그 값에 걸려 있다", fontsize=11.0)
    finish(fig, 0.80)
    return C.savefig(fig, OUT / "m1_2_self_vs_xy.png")


def fig_m1_by_cond(res, plt, key="m1_by_cls_top1", fname="m1_3_by_class.png", order=None,
                   title="상황별", gt_key=None):
    S = series(res)
    order = order or [c for c in C.CLASSES if c in res["tags"][S[0][0]][key]]
    metrics = [("over_label_pct", f"|Δψ| > {FE.LABEL_DTHETA_DEG}°/step [%]", True),
               ("R_under3_pct", f"R = v/|ψ̇| < {FE.R_MIN:g} m [%]", True),
               ("yawrate_over_pct", f"|요레이트| > {FE.YAWRATE_MAX} rad/s [%]", True),
               ("invalid_pair_pct", f"무효 쌍 (속력 < {FE.MOVE_V:g} m/s) [%]", False)]
    fig, axes = plt.subplots(len(metrics), len(order),
                             figsize=(2.15 * len(order) + 1.4, 2.7 * len(metrics) + 1.4),
                             sharey="row")
    axes = np.atleast_2d(axes)
    for r, (met, ylab, lg) in enumerate(metrics):
        row_vals = []
        for c, cn in enumerate(order):
            ax = axes[r, c]
            vals, cols, ns = [], [], []
            for tag, name, col, mkr, sm, sched, hz in S:
                st = res["tags"][tag][key].get(cn)
                vals.append(st[met] if st else np.nan)
                ns.append(st["n_step_pair"] if st else 0)
                cols.append(col)
            st0 = res["tags"][S[0][0]][key].get(cn, {})
            gtst = res["gt"].get(gt_key, {}).get(cn) if gt_key else None
            ttl = ""
            if r == 0:
                ttl = f"{cn}\n시나리오 n = {st0.get('n_scen', 0):,}\n유효 쌍 ≈ {ns[0]:,}"
            barpanel(ax, [s[1] for s in S], vals, cols, ns=ns,
                     gt=(gtst[met] if gtst else None), title=ttl,
                     ylab=(ylab if c == 0 else ""), fmt="{:.2f}", logy=lg, gt_lab="정답")
            row_vals += [v for v in vals] + ([gtst[met]] if gtst else [])
            if r < len(metrics) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([s[1] for s in S], fontsize=6.6, rotation=50, ha="right")
        set_row_ylim(axes[r, 0], row_vals, lg)
    gtxt = "검은 점선 = 같은 식으로 잰 정답 라벨 · " if gt_key else "(이 구간 나눔은 정답 기준선을 정의하지 않는다) · "
    fig.suptitle(f"측정 1 — {title} 묶음 (val 24,988 · {POP_LAB[POP]} · {gtxt}"
                 f"위 세 줄 세로축 로그 · n < {C.MIN_N} 이면 흐리게)", fontsize=11.5)
    finish(fig, 0.88)
    return C.savefig(fig, OUT / fname)


def fig_m1_anatomy(res, plt):
    """위반이 어디에 있나 — 스텝 속력, 위반 크기, 변위, 그리고 MOVE_V 임계 민감도."""
    S = series(res)
    fig, axes = plt.subplots(1, 4, figsize=(18.6, 5.2))
    x = np.arange(len(FE.SPD_LAB))
    gb = res["gt"]["m1"]["by_speed"]
    ax = axes[0]
    for tag, name, col, mkr, sm, sched, hz in S:
        bs = res["tags"][tag]["m1"][POP]["by_speed"]
        ax.plot(x, [bs.get(l, {}).get("over_label_pct", np.nan) for l in FE.SPD_LAB], color=col,
                marker=mkr, ms=7, mec=C.SURF, mew=0.6, lw=1.7, label=name)
    ax.plot(x, [gb.get(l, {}).get("over_label_pct", np.nan) for l in FE.SPD_LAB], color=C.INK,
            marker="*", ms=10, mec=C.SURF, mew=0.5, lw=2.1, ls=(0, (4, 2)), label="정답 라벨")
    bs0 = res["tags"][MAIN_TAG]["m1"][POP]["by_speed"]
    spd_tick = [f"{l}\nn={bs0.get(l, {}).get('n', 0):,}\n(정답 {gb.get(l, {}).get('n', 0):,})"
                for l in FE.SPD_LAB]
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(spd_tick, fontsize=7.0)
    ax.set_xlabel("스텝 속력 [m/s]"); ax.set_ylabel(f"|Δψ| > {FE.LABEL_DTHETA_DEG}°/step [%] (로그)")
    ax.set_title("위반율은 **저속 스텝**에 몰려 있나", fontsize=9.8)
    ax.legend(fontsize=7.0, loc="upper right")

    ax = axes[1]
    w = 0.15
    for i, (tag, name, col, mkr, sm, sched, hz) in enumerate(S):
        bs = res["tags"][tag]["m1"][POP]["by_speed"]
        ax.bar(x + (i - len(S) / 2) * w, [bs.get(l, {}).get("viol_share_pct", 0) for l in FE.SPD_LAB],
               width=w, color=col, label=name, edgecolor=C.SURF, lw=0.4)
    ax.bar(x + (len(S) / 2) * w, [gb.get(l, {}).get("viol_share_pct", 0) for l in FE.SPD_LAB],
           width=w, color=C.INK, label="정답 라벨", edgecolor=C.SURF, lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(spd_tick, fontsize=7.0)
    ax.set_xlabel("스텝 속력 [m/s]  (n = 유효 쌍 수)"); ax.set_ylabel("전체 위반 중 이 구간의 몫 [%]")
    ax.set_title("위반이 어느 속력 구간에 모여 있나", fontsize=9.8)
    ax.legend(fontsize=6.8); ax.grid(axis="x", visible=False)

    ax = axes[2]
    xm = np.arange(len(FE.MAG_LAB))
    gm = res["gt"]["m1"]["by_magnitude"]
    for i, (tag, name, col, mkr, sm, sched, hz) in enumerate(S):
        bm = res["tags"][tag]["m1"][POP]["by_magnitude"]
        ax.bar(xm + (i - len(S) / 2) * w, [bm[l]["share_of_viol_pct"] for l in FE.MAG_LAB],
               width=w, color=col, label=name, edgecolor=C.SURF, lw=0.4)
    ax.bar(xm + (len(S) / 2) * w, [gm[l]["share_of_viol_pct"] for l in FE.MAG_LAB], width=w,
           color=C.INK, label="정답 라벨", edgecolor=C.SURF, lw=0.4)
    bm0 = res["tags"][MAIN_TAG]["m1"][POP]["by_magnitude"]
    mag_tick = [f"{l}\nn={bm0[l]['n']:,}" for l in FE.MAG_LAB]
    ax.set_yscale("log"); ax.set_xticks(xm); ax.set_xticklabels(mag_tick, fontsize=7.2)
    ax.set_xlabel("|Δψ| 크기  (n = 주 모델의 해당 step 수)")
    ax.set_ylabel("전체 위반 중 몫 [%] (로그)")
    ax.set_title("위반의 크기 분포 — 대부분 경계 바로 위인가", fontsize=9.8)
    ax.grid(axis="x", visible=False)

    # MOVE_V 임계 민감도 — 배수의 크기는 임계가 정한다
    ax = axes[3]
    mvs = [float(k) for k in FE.MV_GRID]
    for tag, name, col, mkr, sm, sched, hz in S:
        mv = res["tags"][tag]["m1"][POP]["mv_sensitivity"]
        ax.plot(mvs, [mv[f"{m:g}"]["over_label_pct"] for m in mvs], color=col, marker=mkr, ms=7,
                mec=C.SURF, mew=0.6, lw=1.7, label=name)
    gmv = res["gt"]["m1"]["mv_sensitivity"]
    ax.plot(mvs, [gmv[f"{m:g}"]["over_label_pct"] for m in mvs], color=C.INK, marker="*", ms=10,
            mec=C.SURF, mew=0.5, lw=2.1, ls=(0, (4, 2)), label="정답 라벨")
    ax.axvline(FE.MOVE_V, color=C.C_RED, ls=":", lw=1.4)
    ax.text(FE.MOVE_V * 1.04, 0.30, f"우리가 쓴\nMOVE_V = {FE.MOVE_V:g}",
            transform=ax.get_xaxis_transform(), fontsize=7.2, color=C.C_RED, va="center",
            bbox=dict(facecolor=C.SURF, edgecolor="none", pad=0.6, alpha=0.9))
    main = res["tags"][MAIN_TAG]["m1"][POP]["mv_sensitivity"]
    rr = [main[f"{m:g}"]["over_label_pct"] / gmv[f"{m:g}"]["over_label_pct"] for m in mvs]
    ax.text(0.02, 0.04, "주 모델 ÷ 정답 배수\n" + "\n".join(f"mv {m:g} → {r:.2f}×"
                                                        for m, r in zip(mvs, rr)),
            transform=ax.transAxes, fontsize=7.0, color=C.INK2, ha="left", va="bottom",
            bbox=dict(facecolor=C.SURF, edgecolor=C.AXIS, lw=0.6, pad=2.5, alpha=0.95))
    ax.set_yscale("log"); ax.set_xlabel("MOVE_V 임계 [m/s]")
    ax.set_ylabel(f"|Δψ| > {FE.LABEL_DTHETA_DEG}°/step [%] (로그)")
    ax.set_title("임계 민감도 — **정답만** 크게 흔들린다\n"
                 "필터를 안 걸면(mv 0) 정답이 더 나쁘다", fontsize=9.6)
    ax.legend(fontsize=7.0, loc="center right")
    fig.suptitle(f"측정 1 해부 — 복원 |Δψ| 위반이 **어디서** 나오나 (val 24,988 · {POP_LAB[POP]} · "
                 "유효 이웃 쌍만)", fontsize=11.5)
    finish(fig, 0.855)
    return C.savefig(fig, OUT / "m1_6_violation_anatomy.png")


def fig_m1_fold(res, dz, plt):
    """측정 1 에서 나온 구조 결함 — 프레네 접힘 (κ·d > 1). 빈도와 정확도 기여를 함께."""
    S = series(res)
    names = [s[1] for s in S]
    cols = [s[2] for s in S]
    fig, axes = plt.subplots(1, 4, figsize=(18.4, 5.1))
    ax = axes[0]
    x = np.arange(len(S)); w = 0.38
    v1 = [res["tags"][s[0]]["m1"][POP]["fold_dot_pct"] for s in S]
    v2 = [res["tags"][s[0]]["m1"][POP]["fold_geo_pct"] for s in S]
    for j, (v, lab) in enumerate([(v1, "주 검출기 (스텝벡터 부호)"),
                                  (v2, "보조 (기하식 1 − κ·d)")]):
        ax.bar(x + (j - 0.5) * w, v, width=w, color=DEF_RAMP[2 * j], label=lab,
               edgecolor=C.SURF, lw=0.6)
        for xi, vi in zip(x + (j - 0.5) * w, v):
            ax.text(xi, vi * 1.08, f"{vi:.3f}", ha="center", va="bottom", fontsize=6.8, color=C.INK2)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.6, rotation=22, ha="right")
    ax.set_ylabel("접힌 step 비율 [%]  (로그)")
    jac = [res["tags"][s[0]]["m1"][POP]["fold_jaccard"] for s in S]
    ax.set_xlabel(f"두 검출기 Jaccard {min(jac):.2f}–{max(jac):.2f}\n"
                  "(유병률 0.01% 대라 '일치율 99.9%' 는 뜻이 없다)", fontsize=7.6)
    ax.set_title("프레네 접힘 — 경로 좌표로는 전진인데\n좌표로는 뒤로 가는 step", fontsize=9.2, pad=18)
    ax.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              columnspacing=0.9, handlelength=1.2, handletextpad=0.4)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for tag, name, col, mkr, sm, sched, hz in S:
        k = f"{tag}__kd"
        if k not in dz:
            continue
        xs = np.sort(dz[k])
        q = 1 - np.arange(len(xs)) / len(xs)
        st = max(1, len(xs) // 3000)
        ax.plot(xs[::st], q[::st], color=col, lw=1.7, label=name)
    ax.axvline(1.0, color=C.C_RED, ls=(0, (4, 2)), lw=1.6)
    ax.text(1.12, 0.5, "κ·d = 1\n(접힘 경계)", fontsize=7.6, color=C.C_RED, va="center", ha="left")
    ax.set_xlim(-2, 4); ax.set_yscale("log"); ax.set_ylim(2e-6, 1.5)
    ax.set_xlabel("κ·d  (경로 곡률 × 횡오프셋, 무차원)")
    ax.set_ylabel("P(κ·d > x)")
    ax.set_title("호길이 배율 (1 − κ·d) 이 음수가 되는 꼬리\n(살아있는 모드)", fontsize=9.6)
    ax.legend(fontsize=7.2, loc="lower left")

    # 정확도 기여 — 상관이지 인과가 아니다
    ax = axes[2]
    fi = res["tags"][MAIN_TAG]["fold_impact"]
    labs = ["접힘 있는 모드\nADE", "접힘 없는 모드\nADE", "top-1 접힘 시나리오\nminADE6",
            "그 밖 시나리오\nminADE6"]
    vals = [fi["mode_ade_folded"], fi["mode_ade_clean"], fi["minade6_top1_folded"],
            fi["minade6_top1_clean"]]
    ns = [fi["n_mode_folded"], None, fi["top1_fold_scen_n"], None]
    for i, (v, c) in enumerate(zip(vals, [C.C_RED, C.MUTED, C.C_RED, C.MUTED])):
        ax.bar(i, v, width=0.66, color=c, edgecolor=C.SURF, lw=0.6)
        ax.text(i, v, f"{v:.2f} m" + (f"\nn={ns[i]:,}" if ns[i] else ""), ha="center", va="bottom",
                fontsize=7.4, color=C.INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_ylim(0, max(vals) * 1.42)
    ax.set_ylabel("오차 [m]")
    ax.set_title(f"정확도 기여 상한 ≈ {fi['minade6_upper_bound_gain']:.4f} m "
                 f"({100 * fi['minade6_upper_bound_gain'] / fi['minade6_all']:.2f}% of "
                 f"{fi['minade6_all']:.3f})\n접힘은 |d| 가 크게 벗어난 모드에서만 생긴다 — "
                 "**상관이지 인과가 아니다**", fontsize=8.8)
    ax.grid(axis="x", visible=False)

    # 기하 설명 — 실측값으로
    ax = axes[3]
    R, d_fold = 4.47, 5.10
    th = np.linspace(-0.9, 0.9, 200)
    cen = np.array([0.0, R])
    arc = lambda dd: cen + (R - dd) * np.stack([np.sin(th), -np.cos(th)], 1)
    path = arc(0.0)
    ax.plot(path[:, 0], path[:, 1], color=C.INK, lw=2.0, label=f"경로 중심선 (R = {R:g} m)")
    for dd, cc in ((2.0, C.C_BLUE), (d_fold, C.C_RED)):
        off = arc(dd)
        ax.plot(off[:, 0], off[:, 1], color=cc, lw=2.0,
                label=f"d = {dd:g} m → κ·d = {dd / R:.2f}" + ("  ← 접힘" if dd > R else ""))
        ax.scatter(off[::28, 0], off[::28, 1], s=18, color=cc, edgecolors=C.SURF, lw=0.5, zorder=5)
        j0, j1 = (120, 150) if dd <= R else (150, 120)
        ax.annotate("", xy=off[j1], xytext=off[j0],
                    arrowprops=dict(arrowstyle="-|>", color=cc, lw=1.8))
    ax.annotate("", xy=path[150], xytext=path[120],
                arrowprops=dict(arrowstyle="-|>", color=C.INK, lw=1.6))
    ax.text(0.5, 0.80, "s 가 늘어도 빨간 점은 **반대로** 간다", transform=ax.transAxes, fontsize=8.2,
            color=C.C_RED, ha="center", va="top")
    ax.set_aspect("equal"); ax.legend(fontsize=7.0, loc="upper left")
    ax.set_title("왜 생기나 — 궤적점 = P(s) + N(s)·d\n"
                 "호길이 배율 (1 − κ·d) < 0 이면 뒤집힌다\n"
                 f"(실측 `fe57c764` step 47: κ = {1 / R:.3f} 1/m · d = {d_fold} m · κ·d = 1.143)",
                 fontsize=8.8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.suptitle("측정 1 부산물 — **프레네 접힘**: model_v4.rollout 에는 |d| 가 국소 곡률반경 1/κ 를 넘는 "
                 "경우를 막는 장치가 없다.\n좌표에서 재기 전에는 보이지 않던 결함이다 (액션 (a, dθ) 는 "
                 f"정상 범위 안에 있다) · val 24,988 · {POP_LAB[POP]}", fontsize=11.0)
    finish(fig, 0.78)
    return C.savefig(fig, OUT / "m1_5_frenet_fold.png")


# ============================================================= 측정 2
PIPE_LAB = {"J1": "J1 액션 a(t) 차분", "J2": "J2 위치 3차차분 (원시)", "J2v": "J2v 위치 벡터 3차차분",
            "J3": "J3 위치 → 평활 속력 → a → 차분", "J4": "J4 속도채널 → 평활 → a → 차분"}


def fig_m2_dist(res, dz, plt):
    S = series(res)
    pipes = ["J1", "J2", "J3", "J4"]
    fig, axes = plt.subplots(1, 4, figsize=(17.8, 5.0), sharey=True)
    for ax, p in zip(axes, pipes):
        for tag, name, col, mkr, sm, sched, hz in S:
            k = f"{tag}__J_{p}"
            if k not in dz:
                continue
            x = np.sort(np.abs(dz[k])[np.isfinite(dz[k])])
            q = 1.0 - np.arange(len(x)) / len(x)
            st = max(1, len(x) // 3000)
            ax.plot(x[::st], q[::st], color=col, lw=1.7, label=name)
        gk = f"gt__J_{p}"
        if gk in dz:
            x = np.sort(np.abs(dz[gk])[np.isfinite(dz[gk])])
            q = 1.0 - np.arange(len(x)) / len(x)
            st = max(1, len(x) // 3000)
            ax.plot(x[::st], q[::st], color=C.INK, lw=2.4, ls=(0, (4, 2)),
                    label="정답 라벨" + (" (Frenet 역변환 액션 · 앞 4,000)" if p == "J1" else ""))
        for thr, lab, cc in ((FE.JERK_LK, "0.9 LK-SDE", C.C_GREEN),
                             (FE.JERK_NUPLAN, "4.13 nuPlan", C.C_YELLOW)):
            ax.axvline(thr, color=cc, lw=1.3, ls=":")
            ax.text(thr * 1.06, 3e-5, lab, rotation=90, fontsize=6.8, color=cc, ha="left", va="bottom")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-2, 5e2); ax.set_ylim(2e-5, 1.5)
        ax.set_xlabel("|저크| [m/s³]")
        ax.set_title(PIPE_LAB[p], fontsize=9.6)
        ax.legend(fontsize=6.6, loc="lower left")       # 칸마다 범례 (J1 은 모집단이 다르다)
    axes[0].set_ylabel("P(|저크| > x)  (생존함수)")
    fig.suptitle("측정 2 — 저크 분포 (살아있는 모드). 같은 식을 모델과 정답에 똑같이 건다. "
                 "J1·J2 의 정답 곡선은 **라벨 위치 잡음**이 그대로 들어온 값이라 참값이 아니다 "
                 "— 판별은 J3·J4 로 한다", fontsize=11.2)
    finish(fig, 0.845)
    return C.savefig(fig, OUT / "m2_1_jerk_dist.png")


def fig_m2_summary(res, plt):
    S = series(res)
    names = [s[1] for s in S]
    cols = [s[2] for s in S]
    Q = lambda t, p, k: res["tags"][t]["m2"][POP][p][k]
    panels = [("J3", "mean_abs", "평균 |저크| [m/s³]", "J3 위치 → 평활", False),
              ("J4", "mean_abs", "평균 |저크| [m/s³]", "J4 속도채널 → 평활", False),
              ("J3", "p90", "p90 |저크| [m/s³]", "J3 p90", False),
              ("J3", "over_0.9_pct", "0.9 m/s³ 초과 [%]", "J3 · LK-SDE 임계", False),
              ("J3", "over_4.13_pct", "4.13 m/s³ 초과 [%]", "J3 · nuPlan 임계", False),
              ("J1", "mean_abs", "평균 |저크| [m/s³]", "J1 액션 a 차분 (로그)", True)]
    fig, axes = plt.subplots(2, 4, figsize=(17.4, 8.8))
    axes = axes.ravel()
    for ax, (p, k, ylab, ti, lg) in zip(axes, panels):
        vals = [Q(s[0], p, k) for s in S]
        gt = res["gt"]["m2"].get(p, {}).get(k)
        if p == "J1":
            gt = res["gt"]["m2"].get("J1_gt_action", {}).get(k)
        barpanel(ax, names, vals, cols, gt=gt, title=ti, ylab=ylab, fmt="{:.2f}", logy=lg)
    ax = axes[6]
    x = np.arange(len(S)); w = 0.27
    for j, (kk, lab) in enumerate([("J3", "J3 ↔ 정답 J3"), ("J4", "J4 ↔ 정답 J4"),
                                   ("J3_vs_gtJ4", "J3 ↔ 정답 J4")]):
        v = [res["tags"][s[0]]["m2"][POP]["wass_vs_gt"][kk] for s in S]
        ax.bar(x + (j - 1) * w, v, width=w, color=DEF_RAMP[j], label=lab, edgecolor=C.SURF, lw=0.6)
        for xi, vi in zip(x + (j - 1) * w, v):
            ax.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.4, color=C.INK2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.2, rotation=30, ha="right")
    ax.set_ylabel("Wasserstein 거리 [m/s³]")
    ax.set_title("예측 저크 분포 ↔ 정답 저크 분포\n(DKM·LK-SDE 형식 · 표본 3M 상한)", fontsize=9.6)
    ax.legend(fontsize=7); ax.grid(axis="x", visible=False)

    ax = axes[7]
    g3 = res["gt"]["m2"]["J3"]["mean_abs"]; g4 = res["gt"]["m2"]["J4"]["mean_abs"]
    r4 = [Q(s[0], "J4", "mean_abs") / g4 for s in S]
    r3 = [Q(s[0], "J3", "mean_abs") / g3 for s in S]
    for j, (v, lab) in enumerate([(r4, "J4 (속도채널) 기준"), (r3, "J3 (위치평활) 기준")]):
        ax.bar(x + (j - 0.5) * 0.36, v, width=0.36, color=DEF_RAMP[2 * j], label=lab,
               edgecolor=C.SURF, lw=0.6)
        for xi, vi in zip(x + (j - 0.5) * 0.36, v):
            ax.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.8, color=C.INK2)
    ax.axhspan(0, 0.8, color=C.C_GREEN, alpha=0.12, lw=0, zorder=0)
    ax.axhline(1.0, color=C.INK, ls=(0, (4, 2)), lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.2, rotation=30, ha="right")
    ax.set_ylabel("모델 평균 |저크| ÷ 정답 평균 |저크|")
    ax.set_ylim(0, max(r4 + r3) * 1.35)
    # 판정 문턱이 정답 추정 구간 안에 있다 — 교차점을 적는다
    mean4 = np.mean([Q(s[0], "J4", "mean_abs") for s in S if res["tags"][s[0]]["smooth"] == 1.0])
    ax.set_title("판별 — 같은 식으로 잰 정답 대비 비율\n"
                 f"정답 참값이 J4 {g4:.2f} ~ J3 {g3:.2f} m/s³ 사이라 **판정이 확정되지 않는다**\n"
                 f"(벌점1 판 평균 {mean4:.2f} m/s³ → 문턱 0.8 교차 {mean4 / 0.8:.3f} · "
                 f"1.25 교차 {mean4 / 1.25:.3f} m/s³)", fontsize=8.6)
    ax.legend(fontsize=7); ax.grid(axis="x", visible=False)
    fig.suptitle(f"측정 2 — 저크 요약 (val 24,988 · {POP_LAB[POP]} · 검은 점선 = 같은 식으로 잰 정답 라벨 · "
                 "문턱 0.8 / 1.25 는 **자체 기준**, 문헌 출처 없음)", fontsize=11.2)
    finish(fig, 0.90)
    return C.savefig(fig, OUT / "m2_2_jerk_summary.png")


def fig_m2_by_class(res, plt):
    S = series(res)
    key = "m2_by_cls_top1"
    order = [c for c in C.CLASSES if c in res["tags"][S[0][0]][key]]
    fig, axes = plt.subplots(2, len(order), figsize=(2.15 * len(order) + 1.4, 7.8), sharey="row")
    axes = np.atleast_2d(axes)
    for r, p in enumerate(["J3", "J4"]):
        row_vals = []
        for c, cn in enumerate(order):
            ax = axes[r, c]
            vals = [res["tags"][s[0]][key][cn][p]["mean_abs"] for s in S]
            ns = [res["tags"][s[0]][key][cn][p]["n"] for s in S]
            nsc = res["tags"][S[0][0]][key][cn].get("n_scen", 0)
            gt = res["gt"]["m2_by_cls"].get(cn, {}).get(p, {}).get("mean_abs")
            barpanel(ax, [s[1] for s in S], vals, [s[2] for s in S], ns=ns, gt=gt,
                     title=(f"{cn}\n시나리오 n = {nsc:,}" if r == 0 else ""),
                     ylab=(f"{PIPE_LAB[p]}\n평균 |저크| [m/s³]" if c == 0 else ""), fmt="{:.2f}",
                     gt_lab="정답")
            row_vals += list(vals) + ([gt] if gt else [])
            if r == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([s[1] for s in S], fontsize=6.6, rotation=50, ha="right")
        set_row_ylim(axes[r, 0], row_vals, False)
    fig.suptitle(f"측정 2 — 상황별 묶음 ({POP_LAB[POP]}). 위: J3(위치→평활) · 아래: J4(속도채널→평활). "
                 "검은 점선 = 같은 식으로 잰 정답 라벨(전체 24,988)", fontsize=11.2)
    finish(fig, 0.86)
    return C.savefig(fig, OUT / "m2_3_jerk_by_class.png")


# ============================================================= 측정 3
def fig_m3_spectrum(res, plt):
    S = series(res)
    g = res["gt"].get("m3")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    for ax, key, ylab, ti in ((axes[0], "spec_a", "진폭 [m/s²]", "종방향 가속도 a(t)"),
                              (axes[1], "spec_dtheta", "진폭 [°/s]", "잔차 요레이트 dθ/dt")):
        for tag, name, col, mkr, sm, sched, hz in S:
            m3 = res["tags"][tag]["m3"][POP]
            f = np.array(m3["freq"]); M = np.array(m3[key])
            ax.plot(f[1:], M[1:], color=col, lw=1.7, label=f"{name} (입력 {hz} Hz)",
                    marker=mkr, ms=3.6, mec=C.SURF, mew=0.4)
        if g:
            f = np.array(g["freq"])
            ax.plot(f[1:], np.array(g[key])[1:], color=C.INK, lw=2.3, ls=(0, (4, 2)),
                    marker="*", ms=6, mec=C.SURF, mew=0.4, label=f"정답 액션열 (앞 {g['n']:,})")
            if "moving" in g:
                ax.plot(f[1:], np.array(g["moving"][key])[1:], color=C.MUTED, lw=1.7,
                        ls=(0, (1.4, 1.4)), label=f"정답 · 6초 내내 움직인 {g['n_moving']:,}")
        for lo, hi in FE.BANDS[1:]:
            ax.axvline(lo, color=C.AXIS, lw=0.9, ls=":")
        ax.set_yscale("log"); ax.set_xlabel("주파수 [Hz]"); ax.set_ylabel(ylab)
        ax.set_title(ti, fontsize=10); ax.set_xlim(0, 5.05)
    axes[0].legend(fontsize=7.4, loc="lower left")
    axes[1].legend(fontsize=7.4, loc="lower left")
    fig.suptitle(f"측정 3 — 60스텝(6 s · 출력 10 Hz) 액션열의 평균 진폭 스펙트럼 ({POP_LAB[POP]}).\n"
                 "Hann 창 + 1차 추세 제거 · 편측 진폭(coherent gain 보정) · 스텝마다 표식 · "
                 "점선 세로줄 = 대역 경계 0.5 / 1 / 2 Hz", fontsize=11.2)
    finish(fig, 0.82)
    return C.savefig(fig, OUT / "m3_1_spectrum.png")


def fig_m3_bands(res, plt):
    S = series(res)
    g = res["gt"].get("m3")
    gm = g.get("moving", g) if g else None
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
    x = np.arange(len(FE.BANDS)); w = 0.13
    for r, (key, unit, nm) in enumerate((("band_a", "m/s²", "a(t)"),
                                         ("band_dtheta", "°/s", "dθ/dt"))):
        ax = axes[r, 0]
        for i, (tag, name, col, mkr, sm, sched, hz) in enumerate(S):
            ax.bar(x + (i - len(S) / 2 - 0.5) * w, res["tags"][tag]["m3"][POP][key], width=w,
                   color=col, label=name, edgecolor=C.SURF, lw=0.4)
        if g:
            ax.bar(x + (len(S) / 2 - 0.5) * w, g[key], width=w, color=C.INK, label="정답 (전체)",
                   edgecolor=C.SURF, lw=0.4)
            ax.bar(x + (len(S) / 2 + 0.5) * w, gm[key], width=w, color=C.MUTED,
                   label="정답 (내내 움직임)", edgecolor=C.SURF, lw=0.4)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(FE.BAND_LAB, fontsize=8)
        ax.set_ylabel(f"대역 진폭 합 [{unit}]")
        ax.set_title(f"{nm} 대역별 진폭  (대역 = 반개구간 (lo, hi])", fontsize=9.4)
        if r == 0:
            ax.legend(fontsize=6.8, ncol=2)
        ax.grid(axis="x", visible=False)

        ax = axes[r, 1]
        if gm:
            for i, (tag, name, col, mkr, sm, sched, hz) in enumerate(S):
                v = np.array(res["tags"][tag]["m3"][POP][key]) / np.maximum(np.array(gm[key]), 1e-12)
                ax.plot(x, v, color=col, marker=mkr, ms=7, mec=C.SURF, mew=0.6, lw=1.7, label=name)
            ax.axhline(1.0, color=C.INK, ls=(0, (4, 2)), lw=1.4)
            ax.text(3.45, 1.06, "정답 = 1", fontsize=7.6, color=C.INK, va="bottom", ha="right")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(FE.BAND_LAB, fontsize=8)
        ax.set_ylabel("모델 ÷ 정답 (내내 움직인 시나리오)")
        ax.set_title(f"{nm} 대역 진폭 — 정답 대비 비율", fontsize=9.6)
        ax.grid(axis="x", visible=False)

        ax = axes[r, 2]
        kk = "Sm_a" if r == 0 else "Sm_dtheta"
        barpanel(ax, [s[1] for s in S], [res["tags"][s[0]]["m3"][POP][kk] for s in S],
                 [s[2] for s in S], gt=(gm[kk] if gm else None), fmt="{:.4f}", logy=True,
                 title=f"CAPS 평활도 Sm ({nm})\nSm = (2/(n·f_s))·Σ M_i·f_i — 작을수록 매끄럽다",
                 ylab=f"Sm [{unit}]  (로그)", gt_lab="정답(움직임)")
    fig.suptitle(f"측정 3 — 대역별 진폭과 CAPS 평활도 Sm (val 24,988 · {POP_LAB[POP]} / "
                 "정답 액션열은 앞 4,000 시나리오)", fontsize=11.2)
    finish(fig, 0.90)
    return C.savefig(fig, OUT / "m3_2_bands_sm.png")


def fig_m3_delta30(res, plt):
    """30에폭 코사인이 15에폭 상수 대비 어디를 키웠나 — **bin 단위**로 본다.
    대역 집계는 경계 규약에 민감해서 주장을 bin 으로 내린다."""
    T15, T30 = "v4_l4nw_ah2_full_sm1_s0", MAIN_TAG
    f = np.array(res["tags"][T30]["m3"][POP]["freq"])
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    for ax, key, nm, unit in ((axes[0], "spec_a", "a(t)", "m/s²"),
                              (axes[1], "spec_dtheta", "dθ/dt", "°/s")):
        for pop, ls, lw, mk in ((POP, "-", 2.0, "o"), ("alive", (0, (3, 2)), 1.5, "s")):
            r = (np.array(res["tags"][T30]["m3"][pop][key])
                 / np.array(res["tags"][T15]["m3"][pop][key]))
            ax.plot(f[1:], r[1:], color=(C.C_VIOLET if pop == POP else C.MUTED), lw=lw, ls=ls,
                    marker=mk, ms=4.6, mec=C.SURF, mew=0.5, label=POP_LAB[pop])
            up = [(f[i], r[i]) for i in range(1, len(f)) if r[i] > 1]
            bx, by, step = ((16, -38, -24) if pop == POP else (12, 26, 24))
            for q, (fx, fy) in enumerate(up):
                off = (bx, by + step * q)
                ax.annotate(f"{fx:.2f} Hz  {fy:.3f}", xy=(fx, fy), xytext=off,
                            textcoords="offset points", fontsize=7.0,
                            color=(C.C_RED if pop == POP else C.MUTED),
                            arrowprops=dict(arrowstyle="-", lw=0.7,
                                            color=(C.C_RED if pop == POP else C.MUTED)))
                ax.scatter([fx], [fy], s=60, facecolors="none",
                           edgecolors=(C.C_RED if pop == POP else C.MUTED), lw=1.4, zorder=6)
        ax.axhline(1.0, color=C.INK, ls=(0, (4, 2)), lw=1.4)
        for lo, hi in FE.BANDS[1:]:
            ax.axvline(lo, color=C.AXIS, lw=0.9, ls=":")
        ax.set_xlim(0, 5.05); ax.set_ylim(0, 1.35)
        ax.set_xlabel("주파수 [Hz]"); ax.set_ylabel("30에폭 코사인 ÷ 15에폭 상수")
        ax.set_title(f"{nm} bin 단위 진폭비 [{unit}]\n빨간 동그라미 = 1 을 넘은 bin", fontsize=9.6)
        ax.legend(fontsize=7.4, loc="lower left")

    ax = axes[2]
    ws = res.get("window_sensitivity")
    if ws:
        cases = list(ws["cases"])
        xb = np.arange(len(FE.BAND_LAB)); w = 0.2
        for i, cse in enumerate(cases):
            ax.bar(xb + (i - len(cases) / 2 + 0.5) * w, ws["cases"][cse]["a"], width=w,
                   color=C.BLUE_RAMP[2 + 3 * i], label=cse, edgecolor=C.SURF, lw=0.4)
        ax.axhline(1.0, color=C.INK, ls=(0, (4, 2)), lw=1.4)
        ax.set_xticks(xb); ax.set_xticklabels(FE.BAND_LAB, fontsize=8.4)
        ax.set_ylabel("a(t) 대역 진폭비 (30 ÷ 15)")
        ax.set_ylim(0, 1.35)
        ax.set_title("**대역 집계는 창함수·추세제거에 민감하다**\n"
                     "(살아있는 모드 · 1 을 넘는 대역 수가 창에 따라 0~2개)", fontsize=9.2)
        ax.legend(fontsize=7.0); ax.grid(axis="x", visible=False)
    fig.suptitle("측정 3 판별 — 흔들림 벌점은 0.1 s 이웃 차이만 벌한다. "
                 "뒤반(학습률 감소 + 추가 학습) 뒤 **어디가 커졌나**\n"
                 "1 을 넘는 곳은 **0.5–0.67 Hz(주기 1.5–2 s) 근처 bin 뿐**이고 나머지는 모두 줄었다 · "
                 "두 조건(에폭 수·학습률 스케줄)이 함께 바뀐 비교라 대조판이 없다 — 묶어서 부른다",
                 fontsize=10.8)
    finish(fig, 0.80)
    return C.savefig(fig, OUT / "m3_3_band_delta_30ep.png")


# ============================================================= 측정 4
def fig_m4(lf, res, plt):
    if lf is None:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.4))
    fits = lf["fits"]
    order = [("sl1", "목적 = smooth_l1\n(학습 거리 손실)"), ("sl1_jit1", "목적 = smooth_l1\n+ 1.0·흔들림")]
    cols = [C.C_BLUE, C.C_VIOLET]
    for ax, kk, yl, ti in ((axes[0, 0], "ADE", "ADE [m]", "정답 재현 오차 ADE — 우리 액션 공간의 하한"),
                           (axes[0, 1], "FDE", "FDE [m]", "정답 재현 오차 FDE")):
        barpanel(ax, [o[1] for o in order], [fits[o[0]][kk] for o in order], cols,
                 title=ti, ylab=yl, fmt="{:.3f}")
        ax.set_xticklabels([o[1] for o in order], fontsize=8, rotation=0, ha="center")
        a_ = fits.get("ade")
        if a_ is not None:
            ax.text(0.5, 0.94, f"(목적 = ADE 판은 4,000 스텝에서 **수렴하지 않아** 뺐다 — "
                               f"ADE {a_['ADE']:.3f} > smooth_l1 판 {fits['sl1']['ADE']:.3f})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=C.MUTED)

    ax = axes[0, 2]
    pc = lf["penalty_cost"]
    obs = (res["tags"]["v4_l4nw_ah2_full_sm1_s0"]["val24988"]["minADE6"]
           - res["tags"]["v4_l4nw_ah2_full_s0"]["val24988"]["minADE6"])
    rc0 = lf.get("route_comparison") or {}
    bars = [("적합 하한\n정답 경로 고정\n(단일 궤적 ADE · n=4,000)", pc["delta_ADE"], C.C_VIOLET)]
    if rc0:
        bars.append(("적합 하한\n모델 top-1 경로\n(단일 궤적 ADE · n=4,000)",
                     rc0["top1_route_delta_ADE"], C.C_ORANGE))
    bars.append(("관찰된 대가\n1.480 → 1.586\n(6모드 minADE6 · n=24,988)", obs, C.C_RED))
    for i, (lab, v, col) in enumerate(bars):
        ax.bar(i, v, width=0.62, color=col, edgecolor=C.SURF, lw=0.6)
        ax.text(i, v, f"+{v:.3f} m", ha="center", va="bottom", fontsize=9.5, color=C.INK2)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.4)
    ax.set_ylim(0, max(b[1] for b in bars) * 1.35)
    ax.set_ylabel("Δ 오차 [m]")
    ax.set_title(f"흔들림 벌점의 최소 손해 vs 실제 손해 — 약 {100 * pc['delta_ADE'] / obs:.0f}%\n"
                 "**같은 양이 아니다**: 왼쪽 둘은 정답 궤적 하나를 되맞춘 ADE,\n"
                 "오른쪽은 6모드 중 최소인 minADE6. 비율은 크기 감각으로만 읽는다", fontsize=8.4)
    ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    st = [fits[o[0]]["saturation"] for o in order]
    x = np.arange(len(order)); w = 0.26
    for j, (kk, lab) in enumerate([("a_step_pct", "|a| 가 상한 8 m/s² 에 붙은 step"),
                                   ("dtheta_step_pct", "|dθ| 가 상한 8°/step 에 붙은 step"),
                                   ("any_scen_pct", "둘 중 하나가 붙은 시나리오")]):
        v = [s[kk] for s in st]
        ax.bar(x + (j - 1) * w, v, width=w, color=DEF_RAMP[j], label=lab, edgecolor=C.SURF, lw=0.5)
        for xi, vi in zip(x + (j - 1) * w, v):
            ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.8, color=C.INK2)
    ax.set_xticks(x); ax.set_xticklabels([o[1] for o in order], fontsize=8)
    ax.set_ylabel("비율 [%]"); ax.set_ylim(0, 100)
    ax.set_title(f"제약이 실제로 걸린 비율 (|값| ≥ {100 * lf['limits']['sat_frac']:.0f}% 상한)",
                 fontsize=9.6)
    ax.legend(fontsize=7); ax.grid(axis="x", visible=False)

    ax = axes[1, 1]
    rc = lf.get("route_comparison")
    if rc:
        labs = ["정답 경로\n(가정: 경로를 안다)", "모델 top-1 경로\n(실제로 고른 경로)"]
        v1 = [rc["gt_route_sl1_ADE"], rc["top1_route_sl1_ADE"]]
        for i, v in enumerate(v1):
            ax.bar(i, v, width=0.6, color=[C.C_BLUE, C.C_ORANGE][i], edgecolor=C.SURF, lw=0.6)
            ax.text(i, v, f"{v:.3f} m", ha="center", va="bottom", fontsize=9, color=C.INK2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylim(0, max(v1) * 1.45); ax.set_ylabel("재현 ADE [m]")
        ax.set_title("'정답 경로를 안다'는 가정이 얼마나 낙관적인가\n"
                     f"top-1 이 정답 경로인 비율은 {rc['top1_eq_gt_route_pct_all']:.1f}% 뿐인데도\n"
                     f"재현 하한은 {v1[0]:.3f} → {v1[1]:.3f} m 밖에 안 오른다 — **결론이 더 강해진다**",
                     fontsize=8.6)
        ax.grid(axis="x", visible=False)

    ax = axes[1, 2]
    bcl = lf["by_class"]
    cl = [c for c in C.CLASSES if c in bcl]
    xx = np.arange(len(cl))
    for j, (o, lab) in enumerate(order):
        ax.plot(xx, [bcl[c][f"{o}_ADE"] for c in cl], color=cols[j], marker="sD"[j], ms=6,
                mec=C.SURF, mew=0.6, lw=1.6, label=lab.replace("\n", " "))
    ymax = max(bcl[c]["sl1_jit1_ADE"] for c in cl)
    for i, c in enumerate(cl):
        ax.text(i, ymax * 1.04, f"n={bcl[c]['n']:,}", ha="center", va="bottom", fontsize=6.4,
                color=C.MUTED if bcl[c]["n"] < C.MIN_N else C.INK2, rotation=90)
    ax.set_ylim(0, ymax * 1.5)
    ax.set_xticks(xx); ax.set_xticklabels(cl, fontsize=7.6, rotation=30, ha="right")
    ax.set_ylabel("재현 ADE [m]"); ax.set_title("상황별 재현 오차", fontsize=9.6)
    ax.legend(fontsize=6.8); ax.grid(axis="x", visible=False)
    fig.suptitle(f"측정 4 — 정답 궤적을 우리 액션 공간으로 되맞춘 재현 하한 "
                 f"(val 앞 {lf['n']:,} · Adam {lf['iters']} 스텝 · 모델 가중치 사용 안 함)",
                 fontsize=11.4)
    finish(fig, 0.905)
    return C.savefig(fig, OUT / "m4_1_labelfit.png")


def fig_m4_oob(lf, plt):
    if lf is None:
        return None
    o = lf["gt_action_out_of_bounds"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    ax = axes[0]
    keys = [("a_over_8_step_pct", "|a| > 8 m/s²"), ("dtheta_over_8deg_step_pct", "|dθ| > 8°/step"),
            ("theta_over_90deg_step_pct", "|θ| > 90°"), ("any_step_pct", "셋 중 하나")]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w / 2, [o[k] for k, _ in keys], width=w, color=C.C_RED, label="모든 step",
           edgecolor=C.SURF, lw=0.5)
    ax.bar(x + w / 2, [o["mv_" + k] for k, _ in keys], width=w, color=C.C_BLUE,
           label=f"움직인 step (AV2 속도 ≥ {FE.MOVE_V:g} m/s · 전체의 {o['moving_step_frac_pct']:.0f}%)",
           edgecolor=C.SURF, lw=0.5)
    for i, (k, _) in enumerate(keys):
        ax.text(i - w / 2, o[k], f"{o[k]:.1f}", ha="center", va="bottom", fontsize=7.4, color=C.INK2)
        ax.text(i + w / 2, o["mv_" + k], f"{o['mv_' + k]:.1f}", ha="center", va="bottom",
                fontsize=7.4, color=C.INK2)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8.4)
    ax.set_ylabel("step 비율 [%]")
    ax.set_title("정답 액션열이 우리 상한 밖인 비율\n(Frenet 정확 역변환 · 제약 없음)", fontsize=9.8)
    ax.legend(fontsize=7.2); ax.grid(axis="x", visible=False)

    ax = axes[1]
    z = np.load(DATA / "gt_actions.npz")
    a = np.abs(np.asarray(z["a"], np.float64)).ravel()
    mv = np.asarray(z["moving"]).ravel()
    for arr, lab, col in ((a, "모든 step", C.C_RED), (a[mv], "움직인 step", C.C_BLUE)):
        xs = np.sort(arr[np.isfinite(arr)])
        q = 1 - np.arange(len(xs)) / len(xs)
        st = max(1, len(xs) // 3000)
        ax.plot(xs[::st], q[::st], color=col, lw=1.8, label=lab)
    ax.axvline(8.0, color=C.INK, ls=(0, (4, 2)), lw=1.4)
    ax.text(8.6, 3e-5, "우리 상한 8 m/s²", rotation=90, fontsize=7.4, color=C.INK, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, 5e2); ax.set_ylim(2e-5, 1.5)
    ax.set_xlabel("|a| [m/s²]"); ax.set_ylabel("P(|a| > x)")
    ax.set_title("정답 라벨을 10 Hz 로 두 번 미분한 가속도\n"
                 f"p99 = {o['a_p99']:.1f} m/s² — 사람이 못 내는 값 = 라벨 잡음", fontsize=9.8)
    ax.legend(fontsize=7.4)
    fig.suptitle("측정 4 보조 — 라벨이 우리 액션 공간 밖인 정도, 그리고 그것이 라벨 잡음인 정도 "
                 f"(val 앞 {lf['n']:,})", fontsize=11.3)
    finish(fig, 0.845)
    return C.savefig(fig, OUT / "m4_2_label_oob.png")


# ============================================================= 결정 트리 (요건 11)
def fig_trees(res, lf, plt):
    """분석용 얕은 트리 + 처리 로직 트리. 서술용이지 인과가 아니다.

    목표를 '모델 J4 저크 − 정답 J4 저크' 에서 **모델의 절대 저크**로 바꿨다 —
    이전 목표는 가르는 변수(정답의 최대·최소 a)가 목표의 분모에 들어가 준순환적이었다.
    """
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(C.tag_dirs(MAIN_TAG)["data"] / "scenarios.parquet")
    sm = np.load(DATA / f"scen_{MAIN_TAG}.npz")
    sg = np.load(DATA / "scen_gt.npz")
    N = len(df)
    y = np.asarray(sm["jerk_mean_J4"][:N], np.float64)      # 모델 top-1 의 절대 저크 [m/s³]
    gt_j = np.asarray(sg["jerk_mean_J4"][:N], np.float64)
    feats = {
        "v0 [m/s]": df["v0_f"].to_numpy(),
        "|Δh6| [°]": np.abs(df["dh6"].to_numpy()),
        "최대 a [m/s²]": df["amax_f"].to_numpy(),
        "최소 a [m/s²]": df["amin_f"].to_numpy(),
        "6초 이동거리 [m]": df["move6"].to_numpy(),
        "분기 수": df["n_distinct"].to_numpy().astype(float),
        "30m 차로 수": df["n_lanes30"].to_numpy().astype(float),
        "앞차 시간간격 [s]": np.nan_to_num(df["thw49"].to_numpy(), nan=99.0),
        "정답경로 평균|d| [m]": df["gt_route_meand"].to_numpy(),
        "폴백 경로": df["fallback"].to_numpy().astype(float),
    }
    X = np.stack(list(feats.values()), 1)
    names = list(feats)
    idx = np.arange(N)
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, idx, test_size=0.3, random_state=C.SEED)
    est = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=C.SEED)
    est.fit(Xtr, ytr)
    r2_tr, r2_te = est.score(Xtr, ytr), est.score(Xte, yte)

    fig, axes = plt.subplots(2, 1, figsize=(17.4, 12.6),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax = axes[0]
    root = C.sk_tree_nodes(
        est, names,
        value_fn=lambda t, j: float(t.value[j].ravel()[0]),
        text_fn=lambda n, v: f"n={n:,}\n평균 |저크| {v:.2f}",
        thr_fmt=lambda nm, x: f"{x:.3g}")
    C.draw_tree(ax, root, fontsize=8.0, edge_fs=7.6)
    ax.set_title("분석용 결정 트리 — 「top-1 예측의 **절대** 저크(J4, m/s³)가 큰 시나리오는 어디인가」  "
                 f"({res['tags'][MAIN_TAG]['name']} · 깊이 3 · 잎당 ≥ 200)\n"
                 f"val 24,988 을 70:30 으로 나눠 학습 {len(ytr):,} / 검증 {len(yte):,} · "
                 f"**노드의 n 은 학습 분할 수다** · 전체 평균 {y.mean():.2f} m/s³ "
                 f"(정답 {gt_j.mean():.2f}) · R² 학습 {r2_tr:.3f} / 검증 {r2_te:.3f} · "
                 "서술용이지 인과가 아니다", fontsize=10.2)

    ax = axes[1]
    m1 = res["tags"][MAIN_TAG]["m1"][POP]
    n_traj, n_pair = m1["n_traj"], m1["n_step_pair"]
    n_step = n_traj * (C.FUT - 1)
    n_bad = n_step * m1["invalid_step_pct"] / 100.0
    n_pair_all = n_traj * (C.FUT - 2)
    g1 = res["gt"]["m1"]
    logic = {
        "text": f"예측 궤적 60점 → 이웃 차분 59 step\n{n_traj:,} 시나리오 × 59 = {n_step:,} step\n"
                f"{POP_LAB[POP]}",
        "color": "#ffffff", "children": [
            (f"|Δp|/0.1 s < {FE.MOVE_V:g} m/s\nstep 의 {m1['invalid_step_pct']:.1f}%",
             {"text": f"방향 무효\n{int(round(n_bad)):,} step\n(정지·저속 — 위치차분 방향이 잡음)",
              "color": C.BLUE_RAMP[1]}),
            (f"≥ {FE.MOVE_V:g} m/s\nstep 의 {100 - m1['invalid_step_pct']:.1f}%",
             {"text": f"ψ = atan2(Δy, Δx)\n{int(round(n_step - n_bad)):,} step", "color": C.BLUE_RAMP[3],
              "children": [
                  (f"이웃 쌍 {n_pair_all:,} 중 하나라도 무효\n**쌍 기준** {m1['invalid_pair_pct']:.1f}%",
                   {"text": f"Δψ 세지 않는다\n{n_pair_all - n_pair:,} 쌍\n(직전값으로 메우지 않는다)",
                    "color": C.BLUE_RAMP[1]}),
                  (f"둘 다 유효\n쌍의 {100 - m1['invalid_pair_pct']:.1f}%",
                   {"text": f"Δψ = wrap(ψ[k+1] − ψ[k])\n{n_pair:,} 쌍", "color": C.BLUE_RAMP[5],
                    "tc": "white", "children": [
                        (f"|Δψ| ≤ {FE.LABEL_DTHETA_DEG}°",
                         {"text": f"정상\n{100 - m1['over_label_pct']:.2f}%", "color": C.BLUE_RAMP[2]}),
                        (f"|Δψ| > {FE.LABEL_DTHETA_DEG}°",
                         {"text": f"위반\n{m1['over_label_pct']:.2f}%\n(정답 라벨 {g1['over_label_pct']:.2f}%)",
                          "color": C.BLUE_RAMP[9], "tc": "white"})]})]})]}
    C.draw_tree(ax, logic, fontsize=8.4, edge_fs=7.6, y_gap=1.0)
    ax.set_title("처리 로직 결정 트리 — 측정 1 의 heading 복원 규칙과 실제 개수 "
                 f"({res['tags'][MAIN_TAG]['name']} · {POP_LAB[POP]} · "
                 "무효 비율은 step 기준과 쌍 기준이 다르다)", fontsize=10.2)
    fig.tight_layout()
    p = C.savefig(fig, OUT / "t1_trees.png")

    # 잎 표 + 대표 시나리오 + group-by 대조
    leaf_all = est.apply(X)
    leaves = C.tree_leaves(root)
    lines = [f"# 분석용 트리: top-1 예측의 절대 저크 J4 [m/s³]  [{MAIN_TAG}]",
             f"# 전체 n={N:,}  평균={y.mean():.4f} (정답 {gt_j.mean():.4f})  "
             f"학습 {len(ytr):,} / 검증 {len(yte):,}  R² 학습={r2_tr:.4f} 검증={r2_te:.4f}",
             "# 아래 잎의 n 은 **학습 분할** 수다 (전체의 70%)", "",
             "규칙 | n(학습분할) | 평균 |저크| | 대표 시나리오", "---|---|---|---"]
    chk = []
    for nd in leaves:
        m = leaf_all == nd["id"]
        sub = np.where(m)[0]
        rep = sub[np.argmin(np.abs(y[sub] - nd["value"]))] if len(sub) else -1
        sid = df["sid"].iloc[rep][:8] if rep >= 0 else "-"
        lines.append(" & ".join(nd["rule"]) + f" | {nd['n']:,} | {nd['value']:.2f} | `{sid}`")
        chk.append({"leaf": int(nd["id"]), "n_tree": int(nd["n"]), "n_groupby": int(m.sum()),
                    "value_tree": float(nd["value"]),
                    "value_groupby": float(y[m].mean()) if m.any() else None,
                    "rep_sid": df["sid"].iloc[rep] if rep >= 0 else None, "rule": nd["rule"]})
    lines += ["", "## sklearn export_text", export_text(est, feature_names=names, decimals=3)]
    (DATA / "tree_rules.txt").write_text("\n".join(lines))
    (DATA / "tree_check.json").write_text(json.dumps(
        {"target": "top-1 예측의 평균 |저크| J4 [m/s³]", "r2_train": r2_tr, "r2_test": r2_te,
         "overall_mean": float(y.mean()), "gt_mean": float(gt_j.mean()), "leaves": chk},
        indent=2, ensure_ascii=False))
    return p


# ============================================================= 대표 시나리오 (요건 8)
def fig_cases(res, plt):
    """위반 사례 두 갈래를 나란히 —
    (A) 전형: |Δψ| 7.3–15° (위반의 92%) · (B) 꼬리: 90–180° (0.2%, 프레네 접힘)."""
    import pandas as pd
    from dataset_cached import CachedV4Dataset
    d = C.tag_dirs(MAIN_TAG)["data"]
    pz = np.load(d / "pred.npz")
    rz = np.load(d / "raw.npz")
    df = pd.read_parquet(d / "scenarios.parquet")
    ds = CachedV4Dataset(C.VAL_CACHE)
    N = len(df)
    ar = np.arange(N)
    top1 = pz["top1"].astype(int)
    P = np.asarray(pz["traj"][ar, top1], np.float64)
    TH = np.asarray(pz["theta"][ar, top1], np.float64)
    H = np.asarray(pz["h"][ar, top1], np.float64)
    V = np.asarray(pz["v"][ar, top1], np.float64)
    k = FE.xy_kin(P)
    o2 = k["ok2"]
    dpsi = np.degrees(np.abs(k["dpsi"]))
    dth = np.degrees(np.abs(np.diff(TH, axis=-1)))[:, 1:]
    mx_psi = np.where(o2, dpsi, 0).max(1)
    mx_th = np.where(o2, dth, 0).max(1)
    minade = df["minade"].to_numpy()
    # (A) 전형: 7.3–15° 사이에서만 위반, 자기참조는 통과
    okA = ((mx_psi > FE.LABEL_DTHETA_DEG) & (mx_psi < 15.0) & (mx_th < FE.LABEL_DTHETA_DEG)
           & (minade < 2.0) & (df["v0_f"].to_numpy() > 5.0))
    scoreA = -np.abs(mx_psi - 10.0)
    idxA = np.argsort(-np.where(okA, scoreA, -np.inf))[:2]
    # (B) 꼬리: 90° 초과 (프레네 접힘 포함)
    okB = (mx_psi > 90.0) & (mx_th < FE.LABEL_DTHETA_DEG) & (minade < 3.0)
    idxB = np.argsort(-np.where(okB, mx_psi - mx_th, -np.inf))[:2]
    idx = list(idxA) + list(idxB)
    kinds = ["전형 (7.3–15°, 위반의 92%)", "전형 (7.3–15°, 위반의 92%)",
             "꼬리 (>90°, 위반의 0.2%)", "꼬리 (>90°, 위반의 0.2%)"]

    fig, axes = plt.subplots(2, len(idx), figsize=(5.0 * len(idx), 9.8),
                             gridspec_kw={"height_ratios": [1.5, 1.0]})
    axes = np.atleast_2d(axes)
    picks = []
    for j, i in enumerate(idx):
        i = int(i)
        sid = df["sid"].iloc[i]
        origin, theta = np.asarray(ds.raw("origin")[i]), float(ds.raw("theta")[i])
        past = np.asarray(rz["pos"][i], np.float64)[:C.OBS]
        y = np.asarray(ds.raw("y")[i], np.float64)
        p = P[i]
        ax = axes[0, j]
        xl = yl = None
        try:
            scene = C.build_scene(sid, origin, theta)
            allp = np.concatenate([past, y, p])
            xl = (allp[:, 0].min() - 12, allp[:, 0].max() + 12)
            yl = (allp[:, 1].min() - 12, allp[:, 1].max() + 12)
            sp = max(xl[1] - xl[0], yl[1] - yl[0]) / 2
            cx, cy = np.mean(xl), np.mean(yl)
            xl, yl = (cx - sp, cx + sp), (cy - sp, cy + sp)
            C.draw_scene_light(ax, scene, xl, yl)
        except Exception as e:
            print("[case] 지도 실패", sid, e, flush=True)
        ax.plot(past[:, 0], past[:, 1], color=C.C_PAST, lw=2.0, zorder=8)
        C.past_dots(ax, np.asarray(rz["pos"][i], np.float64), 10, C.C_PAST)
        ax.plot(y[:, 0], y[:, 1], color=C.C_GT, lw=2.2, zorder=9, label="정답")
        ax.scatter(y[9::10, 0], y[9::10, 1], s=14, color=C.C_GT, edgecolors=C.SURF, lw=0.5, zorder=10)
        C.origin_join(ax, p[0], C.C_TOP)
        ax.plot(p[:, 0], p[:, 1], color=C.C_TOP, lw=2.2, zorder=9, label="예측 (확률 1위)")
        ax.scatter(p[9::10, 0], p[9::10, 1], s=14, color=C.C_TOP, edgecolors=C.SURF, lw=0.5, zorder=10)
        bad = np.where(o2[i] & (dpsi[i] > FE.LABEL_DTHETA_DEG))[0]
        if len(bad):
            ax.scatter(p[bad + 1, 0], p[bad + 1, 1], s=46, facecolors="none", edgecolors=C.C_RED,
                       lw=1.5, zorder=12, label=f"|Δψ| > {FE.LABEL_DTHETA_DEG}° step")
        if xl:
            ax.set_xlim(*xl); ax.set_ylim(*yl)
            C.scale_bar(ax, xl, yl)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{kinds[j]}\n{sid[:8]} · {df['cls'].iloc[i]} · minADE6 {minade[i]:.2f} m\n"
                     f"max|Δθ| {mx_th[i]:.1f}°/step (통과) vs max|Δψ| {mx_psi[i]:.1f}°/step (위반)",
                     fontsize=9.0)
        if j == 0:
            ax.legend(fontsize=7.4, loc="upper left")

        ax = axes[1, j]
        t = C.T_PRED[1:-1]
        ax.plot(t, dth[i], color=DEF_RAMP[0], lw=1.7, label="① |Δθ| 자기참조",
                **C.step_kw(10, color=DEF_RAMP[0], ms=3.0))
        dh = np.degrees(np.abs(C.wrap(np.diff(H[i]))))[1:]
        ax.plot(t, dh, color=DEF_RAMP[1], lw=1.7, label="② |Δh| = |Δk + Δθ|",
                **C.step_kw(10, color=DEF_RAMP[1], ms=3.0))
        ax.plot(t, np.where(o2[i], dpsi[i], np.nan), color=DEF_RAMP[2], lw=1.9,
                label="③ |Δψ| 좌표 복원", **C.step_kw(10, color=DEF_RAMP[2], ms=3.0))
        ax.axhline(FE.LABEL_DTHETA_DEG, color=C.C_RED, ls=(0, (4, 2)), lw=1.3)
        ax.text(0.05, FE.LABEL_DTHETA_DEG, f" 라벨 상한 {FE.LABEL_DTHETA_DEG}°/step", fontsize=7,
                color=C.C_RED, va="bottom")
        ax2 = ax.twinx()
        ax2.plot(C.T_PRED, V[i], color=C.MUTED, lw=1.2, ls=(0, (1.5, 1.5)))
        ax2.set_ylabel("속력 [m/s] (회색 점선)", color=C.MUTED, fontsize=8.0)
        ax2.tick_params(labelsize=7.5, colors=C.MUTED); ax2.grid(False)
        ax.set_xlabel("시간 [s]  (0 = 예측 시작)"); ax.set_ylabel("|Δ| [°/step]")
        ax.set_xlim(0, 6.05)
        if j == 0:
            ax.legend(fontsize=7.2, loc="upper left")
        picks.append({"i": i, "sid": sid, "kind": kinds[j], "cls": df["cls"].iloc[i],
                      "max_dtheta_deg": float(mx_th[i]), "max_dpsi_deg": float(mx_psi[i]),
                      "minade": float(minade[i]), "n_bad_step": int(len(bad))})
    fig.suptitle("대표 시나리오 — 자기참조 |Δθ| 는 라벨 상한을 지키는데 좌표에서 복원한 |Δψ| 는 넘는 사례\n"
                 f"왼쪽 둘 = **전형**(위반의 92%가 여기) · 오른쪽 둘 = 꼬리(0.2%) · "
                 f"{res['tags'][MAIN_TAG]['name']} · 확률 1위 모드 · 1초 간격 점 · 빨간 동그라미 = 위반 step",
                 fontsize=11.0)
    finish(fig, 0.885)
    (DATA / "case_picks.json").write_text(json.dumps(picks, indent=2, ensure_ascii=False))
    return C.savefig(fig, OUT / "r1_cases.png")


# ============================================================= 결과 특성 표
def write_table(res, lf):
    S = series(res)
    rows = []
    for tag, name, col, mk, sm, sched, hz in S:
        t = res["tags"][tag]
        m1, m1a = t["m1"]["top1"], t["m1"]["alive"]
        m2, m3 = t["m2"]["top1"], t["m3"]["top1"]
        rows.append({
            "판": name, "벌점": sm, "스케줄": sched, "입력Hz": hz,
            "minADE6": t["val24988"]["minADE6"],
            "자기참조 dθ>7.3° (top1)": m1["self_dtheta_over_pct_sameden"],
            "복원 Δψ>7.3° (top1)": m1["over_label_pct"],
            "복원 Δψ>7.3° (6슬롯)": m1a["over_label_pct"],
            "R<3m (top1)": m1["R_under3_pct"], "R<3m (6슬롯)": m1a["R_under3_pct"],
            "요레이트>0.95 (top1)": m1["yawrate_over_pct"],
            "요가속>1.93 (top1)": m1["yawacc_over_pct"],
            "프레네 접힘 step (top1)": m1["fold_dot_pct"],
            "J3 평균|저크|": m2["J3"]["mean_abs"], "J4 평균|저크|": m2["J4"]["mean_abs"],
            "J4/정답J4": m2["J4"]["mean_abs"] / res["gt"]["m2"]["J4"]["mean_abs"],
            "J3/정답J3": m2["J3"]["mean_abs"] / res["gt"]["m2"]["J3"]["mean_abs"],
            "W(J3↔정답J3)": m2["wass_vs_gt"]["J3"], "W(J4↔정답J4)": m2["wass_vs_gt"]["J4"],
            "Sm_a": m3["Sm_a"], "Sm_dθ": m3["Sm_dtheta"],
            "a 0.5–1Hz": m3["band_a"][1], "a 2–5Hz": m3["band_a"][3],
        })
    g1, g2 = res["gt"]["m1"], res["gt"]["m2"]
    g = {"판": "정답 라벨", "벌점": "-", "스케줄": "-", "입력Hz": "-", "minADE6": None,
         "자기참조 dθ>7.3° (top1)": None,
         "복원 Δψ>7.3° (top1)": g1["over_label_pct"], "복원 Δψ>7.3° (6슬롯)": g1["over_label_pct"],
         "R<3m (top1)": g1["R_under3_pct"], "R<3m (6슬롯)": g1["R_under3_pct"],
         "요레이트>0.95 (top1)": g1["yawrate_over_pct"], "요가속>1.93 (top1)": g1["yawacc_over_pct"],
         "프레네 접힘 step (top1)": None,
         "J3 평균|저크|": g2["J3"]["mean_abs"], "J4 평균|저크|": g2["J4"]["mean_abs"],
         "J4/정답J4": 1.0, "J3/정답J3": 1.0, "W(J3↔정답J3)": 0.0, "W(J4↔정답J4)": 0.0}
    if "m3" in res["gt"]:
        gm = res["gt"]["m3"].get("moving", res["gt"]["m3"])
        g.update({"Sm_a": gm["Sm_a"], "Sm_dθ": gm["Sm_dtheta"],
                  "a 0.5–1Hz": gm["band_a"][1], "a 2–5Hz": gm["band_a"][3]})
    rows.append(g)
    (DATA / "table.json").write_text(json.dumps(
        {"population": POP, "note": "top1 = 확률 1위 모드(실제 예측), 6슬롯 = 살아있는 모드 평균(상한)",
         "rows": rows}, indent=2, ensure_ascii=False))
    return rows


def main():
    plt = C.setup_mpl()
    OUT.mkdir(parents=True, exist_ok=True)
    res, lf, dz = load()
    made = [
        fig_m1_violations(res, plt),
        fig_m1_self_vs_xy(res, plt),
        fig_m1_fold(res, dz, plt),
        fig_m1_anatomy(res, plt),
        fig_m1_by_cond(res, plt, gt_key="m1_by_cls"),
        fig_m1_by_cond(res, plt, key="m1_by_v0_top1", fname="m1_4_by_v0.png",
                       order=C.BINS["v0"][1], title="시작 속력 v0 [m/s] 구간별", gt_key=None),
        fig_m2_dist(res, dz, plt),
        fig_m2_summary(res, plt),
        fig_m2_by_class(res, plt),
        fig_m3_spectrum(res, plt),
        fig_m3_bands(res, plt),
        fig_m3_delta30(res, plt),
        fig_m4(lf, res, plt),
        fig_m4_oob(lf, plt),
        fig_trees(res, lf, plt),
        fig_cases(res, plt),
    ]
    write_table(res, lf)
    for p in made:
        if p:
            print("[fig]", p, flush=True)
    return made


if __name__ == "__main__":
    main()
