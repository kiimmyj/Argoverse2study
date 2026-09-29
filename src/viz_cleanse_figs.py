#!/usr/bin/env python
"""viz_cleanse_figs.py - 데이터 정제 전수 조사 그림.

  python src/viz_cleanse_figs.py                 # 전부
  python src/viz_cleanse_figs.py --only c7 c8    # 일부만

산출물 viz/v4/cleanse/*.png
  c1 항목별 비율 (train vs val)            c5 heading 품질 (Δh · 미정의 · 속도 · 램프)
  c2 맵 이탈 상세                          c6 겹침 히트맵
  c3 밴드 이탈 분포·조건별                  c7 대표 시나리오 궤적 (지도 위)
  c4 heading 뒤집힘 (이동거리 구간별)        c8 결정 트리 (커버리지 실패)

정의·임계값은 cleanse_census.py 와 viz_cleanse_report.py 에만 있다. 여기서는 새로 만들지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import viz_v4_common as C                                             # noqa: E402
import viz_cleanse_report as R                                        # noqa: E402
from cleanse_census import PATH_LBL, CAUSE, TYPE_NAME, T_ALL, OBS, FUT, LAT_FAR_M  # noqa: E402

REPO = SRC.parent
OUT = REPO / "viz/v4/cleanse"
DATA = OUT / "data"
DATA_ROOT = Path("/data/argoverse2/motion_forecasting")
SPLIT_COLOR = {"train": C.C_BLUE, "val": C.C_ORANGE}


def load_all():
    D = {}
    for sp in ("train", "val"):
        d = R.load(sp)
        if d is not None:
            D[sp] = d
    if not D:
        raise SystemExit("census_*.npz 가 없다")
    return D


def bar_pairs(ax, labels, vals, ns, splits, horiz=True, fmt="{:.2f}%", fs=7.2):
    """분할(train/val) 두 개를 나란히 그리는 막대. 칸마다 n 을 적는다."""
    y = np.arange(len(labels))
    w = 0.8 / max(len(splits), 1)
    for i, sp in enumerate(splits):
        off = (i - (len(splits) - 1) / 2) * w
        v = np.asarray(vals[sp], float)
        if horiz:
            ax.barh(y + off, v, height=w * 0.92, color=SPLIT_COLOR[sp], label=sp, zorder=3)
            for j, (vv, nn) in enumerate(zip(v, ns[sp])):
                ax.text(vv + 0.01 * max(v.max(), 1e-9), y[j] + off,
                        f"{fmt.format(vv)}  (n={int(nn):,})", va="center", ha="left",
                        fontsize=fs, color=C.INK2)
        else:
            ax.bar(y + off, v, width=w * 0.92, color=SPLIT_COLOR[sp], label=sp, zorder=3)
            for j, (vv, nn) in enumerate(zip(v, ns[sp])):
                ax.text(y[j] + off, vv, f"{fmt.format(vv)}\nn={int(nn):,}", va="bottom",
                        ha="center", fontsize=fs, color=C.INK2)
    if horiz:
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlim(0, max(max(np.max(vals[sp]) for sp in splits) * 1.42, 1e-6))
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(y); ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------- c1 항목별 비율
def fig_c1(D, plt):
    A = [k for k in R.FLAGS if k[0] in "AF"]
    B = [k for k in R.FLAGS if k[0] == "B"]
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 10.2),
                             gridspec_kw={"height_ratios": [len(A), len(B)]})
    for ax, keys, title in ((axes[0], A, "A. 맵 이탈 계열 + focal 종류"),
                            (axes[1], B, "B. heading 오기입 계열")):
        vals = {sp: [100.0 * R.FLAGS[k][1](d).mean() for k in keys] for sp, d in D.items()}
        ns = {sp: [R.FLAGS[k][1](d).sum() for k in keys] for sp, d in D.items()}
        bar_pairs(ax, keys, vals, ns, list(D))
        ax.set_xlabel("시나리오 비율 [%]")
        ax.set_title(title + "   (모집단: " +
                     " · ".join(f"{sp} {len(d['sid']):,}" for sp, d in D.items()) + " 시나리오)")
        ax.legend(loc="lower right")
    fig.suptitle("데이터 정제 후보 — 항목별 시나리오 비율 (전수)", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return C.savefig(fig, OUT / "c1_overview.png")


# ---------------------------------------------------------------- c2 맵 이탈 상세
def fig_c2(D, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))

    # (a) focal 종류별
    ax = axes[0, 0]
    codes = sorted({int(x) for d in D.values() for x in np.unique(d["f_type"])})
    labs = [TYPE_NAME.get(c, str(c)) for c in codes]
    x = np.arange(len(codes))
    for i, (sp, d) in enumerate(D.items()):
        for j, (key, col, mk) in enumerate((("route_none", C.C_BLUE, "o"),
                                            ("cov_fail", C.C_ORANGE, "s"))):
            v, n = [], []
            for c_ in codes:
                m = d["f_type"] == c_
                q = d["route_none"][m].astype(bool) if key == "route_none" else (d["coverage"][m] == 0)
                v.append(100.0 * q.mean() if m.sum() else np.nan)
                n.append(int(m.sum()))
            ax.plot(x, v, marker=mk, ms=7 if sp == "val" else 5.5, ls="--" if sp == "val" else "-",
                    color=col, alpha=1.0 if sp == "train" else 0.65,
                    label=f"{'경로없음' if key == 'route_none' else '커버리지 실패'} · {sp}")
    ax.set_xticks(x)
    d0 = D["train"] if "train" in D else D["val"]
    ax.set_xticklabels([f"{l}\nn={int((d0['f_type'] == c_).sum()):,}" for l, c_ in zip(labs, codes)],
                       fontsize=7.6)
    ax.set_ylabel("해당 종류 안에서의 비율 [%]")
    ax.set_title(f"(a) focal 종류가 이탈의 1차 원인  (n 은 {list(D)[0]} 기준)")
    ax.set_ylim(-3, 68)
    ax.legend(fontsize=7.4, ncol=2, loc="upper right")

    # (b) 스텝 단위 차로밖
    ax = axes[0, 1]
    keys = ["중심선 5 m 밖 (관측 50스텝)", "중심선 5 m 밖 (정답 60스텝)",
            "같은방향 차로 5 m 밖 (관측)", "같은방향 차로 5 m 밖 (정답)",
            "주행가능영역 밖 (관측)", "주행가능영역 밖 (정답)"]
    S = R.step_rates(D)
    vals = {sp: [S[sp][k] for k in keys] for sp in D}
    ns = {sp: [len(D[sp]["sid"]) * (OBS if "관측" in k else FUT) for k in keys] for sp in D}
    bar_pairs(ax, [k.replace(" (", "\n(") for k in keys], vals, ns, list(D), fs=6.8)
    ax.set_xlabel("스텝 비율 [%]   (n = 스텝 수)")
    ax.set_title(f"(b) focal 이 차로 밖인 스텝 — 기준 {LAT_FAR_M:.0f} m")
    ax.legend(loc="lower right")

    # (c) 밴드 폭별 커버리지 실패
    ax = axes[1, 0]
    B = R.band_table(D)
    keys = ["±1.75 고정", "규칙 밴드(현행)", "±3.6 고정",
            "규칙 밴드(차량계열 focal 만)", "±3.6 (차량계열 focal 만)"]
    vals = {sp: [B[sp][k] for k in keys] for sp in D}
    ns = {sp: [B[sp]["차량계열 focal n" if "차량계열" in k else "전체 n"] for k in keys] for sp in D}
    bar_pairs(ax, keys, vals, ns, list(D), fs=7.0)
    ax.set_xlabel("커버리지 실패 [%]")
    ax.set_title("(c) 밴드를 넓히면 실패가 줄지만 보장이 사라진다")
    ax.legend(loc="lower right")

    # (d) 실패 원인
    ax = axes[1, 1]
    CT = R.cause_table(D)
    names = list(CT[list(D)[0]]["원인"])
    y = np.arange(len(D))
    left = np.zeros(len(D))
    cols = [C.C_BLUE, C.C_ORANGE, C.C_AQUA, C.C_YELLOW, C.C_MAGENTA, C.C_GREEN]
    for i, nm in enumerate(names):
        v = np.array([CT[sp]["원인"][nm]["pct"] for sp in D])
        ax.barh(y, v, left=left, color=cols[i % len(cols)], label=nm, height=0.55, zorder=3)
        for j, (vv, ll) in enumerate(zip(v, left)):
            if vv >= 4:
                ax.text(ll + vv / 2, y[j], f"{vv:.0f}%", ha="center", va="center",
                        fontsize=7.4, color="white" if i in (0, 1, 5) else C.INK)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{sp}\n실패 {CT[sp]['실패 건수']:,}건\n({CT[sp]['실패%']:.2f}%)" for sp in D])
    ax.set_xlabel("커버리지 실패 안에서의 원인 구성 [%]")
    ax.set_title("(d) 실패의 대부분은 '밴드만 넘음' — 경로는 맞는데 폭이 모자라다")
    ax.legend(fontsize=7.4, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", visible=False)
    fig.suptitle("A. 맵 이탈 — 전수 집계", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return C.savefig(fig, OUT / "c2_map.png")


# ---------------------------------------------------------------- c3 밴드 이탈 분포
def fig_c3(D, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))

    ax = axes[0, 0]
    med = {}
    for sp, d in D.items():
        f = d["coverage"] == 0
        v = 100.0 * d["frac_out"][f]
        med[sp] = (np.median(v), (v > 90).mean() * 100)
        ax.hist(v, bins=np.arange(0, 102, 4), histtype="step", lw=1.9,
                color=SPLIT_COLOR[sp], density=True,
                label=f"{sp} (n={int(f.sum()):,}) 중앙 {med[sp][0]:.0f}%, 90%↑ 가 {med[sp][1]:.0f}%")
    ax.set_xlabel("정답 미래 60스텝 중 밴드 밖 스텝 비율 [%]")
    ax.set_ylabel("밀도")
    ax.set_title("(a) 살짝 넘는 것과 통째로 벗어난 것이 섞여 있다")
    ax.legend(fontsize=7.4)

    ax = axes[0, 1]
    for sp, d in D.items():
        for f, ls, lab in ((d["coverage"] == 1, "-", "커버리지 성공"), (d["coverage"] == 0, "--", "실패")):
            v = np.sort(np.clip(d["gt_maxd"][f], 0, 20))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, ls=ls, color=SPLIT_COLOR[sp],
                    lw=1.7, label=f"{sp} · {lab} (n={int(f.sum()):,})")
    for x_, lab in ((1.75, "±1.75"), (3.6, "±3.6")):
        ax.axvline(x_, color=C.MUTED, lw=1.0, ls=":")
        ax.text(x_, 3, lab, fontsize=7.4, color=C.MUTED, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("정답 기준 경로에서의 최대 |d| [m]")
    ax.set_ylabel("누적 비율 [%]")
    ax.set_xlim(0, 12)
    ax.set_title("(b) 얼마나 넘는가 — 밴드를 얼마나 넓혀야 하는지")
    ax.legend(fontsize=7.2)

    # 조건별 실패율 (요건 5: 하나로 뭉치지 말 것)
    for ax, key, edges, xlab, title in (
            (axes[1, 0], "move6", [0, 1, 5, 10, 20, 40, 60, 1e9], "6초 이동거리 [m]",
             "(c) 이동거리 구간별 커버리지 실패율"),
            (axes[1, 1], "v0", [0, 1, 3, 6, 10, 15, 1e9], "예측 시작 속력 v0 [m/s]",
             "(d) 시작 속력 구간별 커버리지 실패율")):
        labs = [f"{edges[i]:g}–{edges[i + 1]:g}" if edges[i + 1] < 1e8 else f"{edges[i]:g}+"
                for i in range(len(edges) - 1)]
        vals, ns = {}, {}
        for sp, d in D.items():
            road = np.isin(d["f_type"], list(R.ROAD_TYPES))       # 보행자를 섞으면 구간 효과가 종류 효과에 덮인다
            b = np.digitize(d[key], edges) - 1
            vals[sp] = [100.0 * (d["coverage"][road & (b == i)] == 0).mean()
                        if (road & (b == i)).any() else 0.0 for i in range(len(labs))]
            ns[sp] = [int((road & (b == i)).sum()) for i in range(len(labs))]
        bar_pairs(ax, labs, vals, ns, list(D), horiz=False, fs=5.9)
        ax.set_xlabel(xlab); ax.set_ylabel("커버리지 실패 [%]")
        ax.set_title(title + " — 차량 계열 focal 만")
        ax.set_ylim(0, max(max(vals[sp]) for sp in D) * 1.35)
        ax.legend(fontsize=7.6)
    fig.suptitle("A3. 정답이 후보 경로 밴드를 얼마나·언제 벗어나나", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return C.savefig(fig, OUT / "c3_band.png")


# ---------------------------------------------------------------- c4 뒤집힘
def fig_c4(D, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
    T = R.track_tables(D)

    for ax, grp, title in ((axes[0, 0], "차량 계열", "(a) 차량 계열 트랙"),
                           (axes[0, 1], "그 외", "(b) 보행자·기타 트랙")):
        vals, ns = {}, {}
        for sp in D:
            t = T[sp]["이동거리별 뒤집힘"][grp]
            vals[sp] = [t[k]["뒤집힘%"] for k in PATH_LBL]
            ns[sp] = [t[k]["판정가능"] for k in PATH_LBL]
        bar_pairs(ax, PATH_LBL, vals, ns, list(D), horiz=False, fs=5.5)
        ax.set_xlabel("트랙 총 이동거리")
        ax.set_ylabel("AV2 heading 이 이동방향과 180° [%]")
        ax.set_title(title + "  (n = 판정 가능한 트랙)")
        ax.set_ylim(0, 52)
        ax.legend(fontsize=7.6)

    # (c) 판정 불가 비율 — '뒤집힘'이 아니라 '비교 기준이 없음'
    ax = axes[1, 0]
    vals, ns = {}, {}
    for sp in D:
        t = T[sp]["이동거리별 뒤집힘"]["차량 계열"]
        vals[sp] = [t[k]["판정불가%"] for k in PATH_LBL]
        ns[sp] = [t[k]["트랙"] for k in PATH_LBL]
    bar_pairs(ax, PATH_LBL, vals, ns, list(D), horiz=False, fs=5.5)
    ax.set_xlabel("트랙 총 이동거리"); ax.set_ylabel("판정 불가 [%]")
    ax.set_title("(c) 안 움직인 트랙은 '이동방향' 자체가 없다 (차량 계열)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.6)

    # (d) 우리 전처리의 뒤집기와 그 결과
    ax = axes[1, 1]
    labs = ["align_ref 가 뒤집음\n(판정가능 트랙 대비)", "게이트판이 뒤집음\n(전체 트랙 대비)",
            "뒤집힌 트랙이\n차로 반대", "안 뒤집힌 트랙이\n차로 반대"]
    vals, ns = {}, {}
    for sp in D:
        a = T[sp]["전처리 뒤집기(align_ref)"]
        c_ = T[sp]["뒤집기가 차로 방향과 맞나"]
        vals[sp] = [a["뒤집음%(판정가능 대비)"], a["게이트판 뒤집음%(전체 대비)"],
                    c_["뒤집힌 트랙이 차로 반대%"], c_["안 뒤집힌 트랙이 차로 반대%"]]
        ns[sp] = [a["판정가능"], a["입력후보 주변트랙"], c_["그중 뒤집힌 트랙"],
                  c_["검사한 트랙"] - c_["그중 뒤집힌 트랙"]]
    bar_pairs(ax, labs, vals, ns, list(D), horiz=False, fs=6.2)
    ax.set_ylabel("트랙 비율 [%]")
    ax.set_title("(d) align_ref 의 뒤집기는 대부분 차로 반대로 간다 = 틀렸다")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7.6)
    fig.suptitle("B1·B2. heading 180° 뒤집힘 — 주변 차량 트랙 전수", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return C.savefig(fig, OUT / "c4_flip.png")


# ---------------------------------------------------------------- c5 heading 품질
def ramp_curve(split="val", n=3000):
    """창 가장자리 램프를 스텝별로 다시 잰다 (npz 에는 양 끝 비율만 있다)."""
    import pyarrow.parquet as pq
    dirs = [p for p in sorted((DATA_ROOT / split).iterdir()) if p.is_dir()][:n]
    acc, cnt = np.zeros(T_ALL - 1), 0
    for d in dirs:
        t = pq.read_table(d / f"scenario_{d.name}.parquet")
        col = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
        m = col["track_id"].astype(str) == str(col["focal_track_id"][0])
        k = col["timestep"][m].astype(int)
        if len(k) != T_ALL:
            continue
        P = np.stack([col["position_x"][m], col["position_y"][m]], 1)[np.argsort(k)]
        V = np.linalg.norm(np.stack([col["velocity_x"][m], col["velocity_y"][m]], 1), axis=1)[np.argsort(k)]
        if np.median(V[10:100]) < 3.0:
            continue
        vp = np.linalg.norm(np.diff(P, axis=0), axis=1) / 0.1
        ref = np.median(vp[10:100])
        if ref < 1e-6:
            continue
        acc += vp / ref
        cnt += 1
    return acc / max(cnt, 1), cnt


def fig_c5(D, plt, ramp_n=3000):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
    S = R.step_rates(D)

    ax = axes[0, 0]
    keys = ["|Δh| > 7.3°/step (우리 h)", "|Δh| > 7.3°/step (AV2 필드)",
            "|Δh| > 90°/step (우리 h)", "|Δh| > 90°/step (AV2 필드)"]
    vals = {sp: [S[sp][k] for k in keys] for sp in D}
    ns = {sp: [len(D[sp]["sid"]) * (T_ALL - 1)] * len(keys) for sp in D}
    bar_pairs(ax, [k.replace(" (", "\n(") for k in keys], vals, ns, list(D), fs=6.8)
    ax.set_xlabel("스텝 비율 [%]  (n = focal 스텝 수)")
    ax.set_title("(a) 물리 한계를 넘는 방향 변화 — 우리 h 가 AV2 필드보다 훨씬 거칠다")
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    labs = ["focal 방향 미정의", "주변 트랙 방향 미정의", "속도 불일치 > 1 m/s", "속도 불일치 > 2 m/s"]
    vals, ns = {}, {}
    for sp, d in D.items():
        T_ = R.track_tables({sp: d})[sp]
        vals[sp] = [100.0 * d["f_undef"].mean(), T_["전처리 뒤집기(align_ref)"]["방향 미정의 트랙%"],
                    S[sp]["속도 불일치 > 1 m/s"], S[sp]["속도 불일치 > 2 m/s"]]
        ns[sp] = [len(d["sid"]), int(d["a_tracks"].sum()), int(d["f_n_dv"].sum()),
                  int(d["f_n_dv"].sum())]
    bar_pairs(ax, labs, vals, ns, list(D), fs=7.0)
    ax.set_xlabel("비율 [%]")
    ax.set_title("(b) 방향을 세울 수 없는 대상 · 속도 필드와의 불일치")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    cur, cnt = ramp_curve("val", ramp_n)
    t = (np.arange(T_ALL - 1) - (OBS - 1)) * 0.1
    ax.plot(t, cur, color=C.C_BLUE, lw=1.6, **C.step_kw(10, C.C_BLUE))
    ax.axhline(1.0, color=C.MUTED, lw=1.0, ls=":")
    ax.axvline(0.0, color=C.MUTED, lw=1.0, ls="--")
    for x_, lab in ((t[0], f"첫 스텝 {cur[0]:.2f}"), (t[-1], f"끝 스텝 {cur[-1]:.2f}")):
        ax.annotate(lab, (x_, cur[0] if x_ < 0 else cur[-1]), textcoords="offset points",
                    xytext=(16 if x_ < 0 else -16, 14), fontsize=7.8, color=C.INK2,
                    ha="left" if x_ < 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=C.AXIS, lw=0.9))
    ax.set_xlabel("시간 [s]  (0 = 예측 시작)")
    ax.set_ylabel("위치차분 속력 / 창 중앙 중앙값")
    ax.set_title(f"(c) 창 가장자리 램프 — 버릴 수 없는 결함 (val {cnt:,} 시나리오)")

    ax = axes[1, 1]
    for sp, d in D.items():
        m = d["f_ramp_ok"].astype(bool)
        for arr, ls, lab in ((d["f_ramp_head"][m], "-", "첫 스텝"), (d["f_ramp_tail"][m], "--", "끝 스텝")):
            v = np.sort(np.clip(arr, 0, 2))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, ls=ls, lw=1.7,
                    color=SPLIT_COLOR[sp], label=f"{sp} · {lab} (n={int(m.sum()):,})")
    ax.axvline(1.0, color=C.MUTED, lw=1.0, ls=":")
    ax.set_xlabel("스텝 속력 / 1~2초(9~10초) 속력 중앙값")
    ax.set_ylabel("누적 비율 [%]")
    ax.set_title("(d) 램프는 거의 모든 시나리오에 있다")
    ax.legend(fontsize=7.2)
    fig.suptitle("B3·B4·B5. 방향·속도 신호의 품질", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return C.savefig(fig, OUT / "c5_heading.png")


# ---------------------------------------------------------------- c6 겹침
def fig_c6(D, plt):
    K = R.KEY_FLAGS
    fig, axes = plt.subplots(1, len(D), figsize=(7.4 * len(D), 7.2), squeeze=False)
    for ax, (sp, d) in zip(axes[0], D.items()):
        M = {k: R.FLAGS[k][1](d) for k in K}
        n = len(d["sid"])
        Z = np.zeros((len(K), len(K)))
        for i, a in enumerate(K):
            for j, b in enumerate(K):
                Z[i, j] = 100.0 * (M[a] & M[b]).sum() / max(M[a].sum(), 1)
        im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(K)):
            for j in range(len(K)):
                c_ = int((M[K[i]] & M[K[j]]).sum())
                ax.text(j, i, f"{Z[i, j]:.0f}%\n{c_:,}", ha="center", va="center", fontsize=6.6,
                        color="white" if Z[i, j] > 55 else C.INK)
        ax.set_xticks(range(len(K)))
        ax.set_xticklabels([k.replace(" ", "\n", 1) for k in K], fontsize=6.8, rotation=35, ha="right")
        ax.set_yticks(range(len(K)))
        ax.set_yticklabels([f"{k}\n(n={int(M[k].sum()):,})" for k in K], fontsize=6.8)
        anyf = np.zeros(n, bool)
        for k in K:
            anyf |= M[k]
        ax.set_title(f"[{sp}] 행 플래그 중 열 플래그도 걸린 비율\n"
                     f"전체 {n:,} · 하나라도 걸림 {int(anyf.sum()):,} ({100 * anyf.mean():.1f}%)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045, label="행 대비 교집합 [%]")
    fig.suptitle("항목이 같은 시나리오에 몰려 있는가 (겹침)", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return C.savefig(fig, OUT / "c6_overlap.png")


# ---------------------------------------------------------------- c7 대표 시나리오
def pick_cases(D):
    """대표 시나리오 6개 — 항목마다 '전형적인' 한 건. (split, sid, 제목, 설명)"""
    out = []
    d = D["val"]
    sid = d["sid"]

    def first(mask, key=None, desc=True, k=0):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        if key is not None:
            idx = idx[np.argsort(-d[key][idx] if desc else d[key][idx])]
        return int(idx[min(k, len(idx) - 1)])

    walk = np.isin(d["f_type"], list(R.WALK_TYPES))
    veh = np.isin(d["f_type"], list(R.ROAD_TYPES))
    cand = [
        (first(d["route_none"].astype(bool) & veh & (d["move6"] > 20), "move6"),
         "A1 경로 없음", "지도가 후보 경로를 못 준다 — 직진 폴백 1개로 때우는 장면"),
        (first(walk & (d["coverage"] == 0), "move6"),
         "F focal 이 보행자·자전거", "차로 위 (s, d) 출력 공간이 성립하지 않는 대상"),
        (first((d["coverage"] == 0) & (d["cause"] == 6) & veh & (d["move6"] > 20), "gt_maxd"),
         "A3 밴드만 초과", "경로는 맞는데 정답이 규칙 밴드 밖으로 나간다"),
        (first((d["coverage"] == 0) & (d["cause"] == 2) & veh, "move6"),
         "A3 나란한 후보 없음", "후보 경로가 정답 진행방향과 나란하지 않다"),
        (first(d["f_flip_av2"].astype(bool) & (d["move6"] > 10), "move6"),
         "B1 focal 라벨 뒤집힘", "AV2 heading 이 이동방향과 180° 어긋난 focal"),
        (first((d["far_obs"] + d["far_fut"] > 60) & veh, "move6"),
         "A2 차로 밖", "주차 구획·미매핑 노면 — 중심선에서 5 m 밖"),
    ]
    for i, (j, t, desc) in enumerate(cand):
        if j is not None:
            out.append(("val", str(sid[j]), t, desc, j))
    return out


def fig_c7(D, plt):
    import hdmap_render as hr
    import lane_frame as lf
    from lane_graph import LaneGraph, REACH_MARGIN_M
    from cleanse_census import load_tracks, _at, DEDUP_M, N_MODES, PRED_SEC
    from heading_decomp import build_heading

    cases = pick_cases(D)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.4))
    for ax, (sp, sid, title, desc, j) in zip(axes.ravel(), cases):
        sdir = DATA_ROOT / sp / sid
        tracks, _ = load_tracks(sdir)
        ftr = next(t for t in tracks if t["focal"])
        pos, hav2, vel = ftr["pos"], ftr["hav2"], ftr["vel"]
        origin, theta = pos[OBS - 1].astype(np.float32), float(hav2[OBS - 1])
        scene = hr.build_scene(sdir / f"log_map_archive_{sid}.json", origin, theta)
        Rm = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        pn = (pos - origin) @ Rm.T

        raw = json.loads((sdir / f"log_map_archive_{sid}.json").read_text())
        g = LaneGraph.from_json_dict(raw)
        speed = float(np.linalg.norm(vel[OBS - 1]))
        d0 = np.array([np.cos(theta), np.sin(theta)])
        starts = g.candidate_lanes(pos[OBS - 1], d0, path=pos[:OBS])
        reach = g.reachable(starts, max(20.0, speed * PRED_SEC) + REACH_MARGIN_M) if starts else set()
        rts = lf.build_routes(g, starts, reach, v0=speed) if starts else []
        if rts:
            order = np.argsort([lf.to_frame(pos[:OBS], rr)[2] for rr in rts])
            rts = [rts[i] for i in order]
            hz = max(10.0, speed * PRED_SEC)
            keep = []
            for rr in rts:
                q = np.array([_at(rr, hz * f) for f in (0.5, 1.0)])
                if all(np.linalg.norm(q - np.array([_at(o, hz * f) for f in (0.5, 1.0)]),
                                      axis=1).max() > DEDUP_M for o in keep):
                    keep.append(rr)
                if len(keep) >= N_MODES:
                    break
            rts = keep

        span = max(35.0, 1.15 * np.abs(pn).max())
        xlim = (pn[:, 0].mean() - span, pn[:, 0].mean() + span)
        ylim = (pn[:, 1].mean() - span * 0.72, pn[:, 1].mean() + span * 0.72)
        C.draw_scene_light(ax, scene, xlim, ylim, mark_lw=0.8)
        for k, rr in enumerate(rts):
            q = lf.resample_route(rr, 64)
            ll, lr = lf.rule_band(g, rr)
            idx = np.clip((q["s"] / max(q["len"], 1e-6) * (len(rr["s"]) - 1)).astype(int),
                          0, len(rr["s"]) - 1)
            P = (q["pts"] - origin) @ Rm.T
            Tn = q["tan"] @ Rm.T
            band = np.stack([ll[idx], lr[idx]], axis=1)
            poly = C.band_polygon(P, Tn, band)
            ax.fill(poly[:, 0], poly[:, 1], color=C.C_AQUA, alpha=0.13, lw=0, zorder=5)
            ax.plot(P[:, 0], P[:, 1], color=C.C_AQUA, lw=1.0, alpha=0.85, zorder=6)
        ax.plot(pn[:OBS, 0], pn[:OBS, 1], color=C.C_PAST, lw=2.2, zorder=8, label="과거 5초")
        ax.plot(pn[OBS - 1:, 0], pn[OBS - 1:, 1], color=C.C_GT, lw=2.2, zorder=9, label="정답 6초")
        ax.plot(pn[OBS - 1::10, 0], pn[OBS - 1::10, 1], ls="none", zorder=10,
                **C.step_kw(2, C.C_GT))
        C.draw_box(ax, pn[OBS - 1, 0], pn[OBS - 1, 1], 0.0, C.C_ORANGE, zorder=11)
        # 정규화 프레임이라 AV2 heading 은 **항상 +x** 다(주황 상자). 이동방향 화살표와 어긋나면 그게 뒤집힘이다.
        hn, _ = build_heading(pos[:OBS], h_ref=hav2[:OBS])
        a_ = float(hn[-1] - theta)
        L = 0.11 * (xlim[1] - xlim[0])
        ax.annotate("", xy=(pn[OBS - 1, 0] + L * np.cos(a_), pn[OBS - 1, 1] + L * np.sin(a_)),
                    xytext=(pn[OBS - 1, 0], pn[OBS - 1, 1]), zorder=12,
                    arrowprops=dict(arrowstyle="-|>", color=C.C_VIOLET, lw=1.8,
                                    shrinkA=0, shrinkB=0))
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        C.scale_bar(ax, xlim, ylim)
        d = D[sp]
        ax.set_title(f"{title}\n{sid[:8]} · {TYPE_NAME.get(int(d['f_type'][j]), '?')} · {desc}\n"
                     f"경로 {len(rts)}개 · 6초 이동 {d['move6'][j]:.0f} m · "
                     f"최대|d| {d['gt_maxd'][j]:.1f} m · 차로밖 {int(d['far_obs'][j] + d['far_fut'][j])}/110 스텝",
                     fontsize=8.6)
    axes[0, 0].legend(fontsize=7.6, loc="upper left")
    fig.suptitle("대표 시나리오 — 회색 과거 5초 · 검정 정답 6초(1초 점) · 청록 후보 경로와 규칙 밴드\n"
                 "주황 상자 = AV2 heading (정규화 때문에 항상 오른쪽) · 보라 화살표 = 위치차분 진행방향",
                 y=0.997, fontsize=10.6)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return C.savefig(fig, OUT / "c7_cases.png")


# ---------------------------------------------------------------- c8 결정 트리
def fig_c8(D, plt):
    from sklearn.tree import DecisionTreeClassifier
    sp = "train" if "train" in D else "val"
    d = D[sp]
    feats = {
        "focal 보행자·자전거": np.isin(d["f_type"], list(R.WALK_TYPES)).astype(float),
        "후보 경로 수": d["n_routes"].astype(float),
        "6초 이동 [m]": d["move6"].astype(float),
        "시작 속력 [m/s]": d["v0"].astype(float),
        "|6초 방향변화| [°]": np.abs(d["dh6"]).astype(float),
        "차로밖 스텝": (d["far_obs"] + d["far_fut"]).astype(float),
    }
    X = np.stack(list(feats.values()), axis=1)
    y = (d["coverage"] == 0).astype(int)
    est = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(200, len(y) // 200),
                                 random_state=0).fit(X, y)
    names = list(feats)
    root = C.sk_tree_nodes(
        est, names,
        value_fn=lambda t, j: float(t.value[j][0][1] / max(t.value[j][0].sum(), 1e-9)) * 100,
        text_fn=lambda n, v: f"n={n:,}\n실패 {v:.1f}%",
        thr_fmt=lambda nm, x: f"{x:.0f}" if "보행" not in nm else "0.5",
        lo=0.0, hi=100.0)
    leaves = C.tree_leaves(root)
    fig = plt.figure(figsize=(15.0, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    C.draw_tree(ax, root, fontsize=8.0, edge_fs=7.4, y_gap=1.0)
    ax.set_title(f"[{sp}] 커버리지 실패를 설명하는 얕은 트리 (깊이 3, sklearn) — "
                 f"서술용이며 인과가 아니다.  전체 실패율 {100 * y.mean():.2f}% (n={len(y):,})",
                 fontsize=10.5)
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    rows = [[" · ".join(l["rule"]) or "(뿌리)", f"{l['n']:,}", f"{l['n'] / len(y) * 100:.1f}%",
             f"{l['value']:.1f}%"] for l in sorted(leaves, key=lambda z: -z["value"])]
    tb = ax2.table(cellText=rows, colLabels=["잎 규칙", "시나리오", "전체 대비", "커버리지 실패율"],
                   cellLoc="left", colLoc="left", loc="upper center",
                   colWidths=[0.58, 0.12, 0.12, 0.16])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.0)
    tb.scale(1, 1.55)
    for (i, _), cell in tb.get_celld().items():
        cell.set_edgecolor(C.GRID)
        cell.set_facecolor(C.SURF if i else "#eef2f7")
        cell.set_text_props(color=C.INK if i else C.INK2)
    return C.savefig(fig, OUT / "c8_tree.png")


# ---------------------------------------------------------------- c9 처리 결정 도식 (요건 11 — 처리 로직용)
def fig_c9(D, plt):
    sp = "train" if "train" in D else "val"
    d = D[sp]
    n = len(d["sid"])
    walk = np.isin(d["f_type"], list(R.WALK_TYPES))
    rn = d["route_none"].astype(bool)
    cf = d["coverage"] == 0
    flip = d["f_flip_av2"].astype(bool)

    def box(m, txt, color, tc=None):
        return {"text": f"{txt}\n{int(m.sum()):,}건 ({100 * m.mean():.2f}%)",
                "color": color, "tc": tc or C.INK}

    g_walk = walk
    g_rn = ~walk & rn
    g_cf = ~walk & ~rn & cf
    g_ok = ~walk & ~rn & ~cf
    drop_c, fix_c, keep_c = "#f7d9cd", "#fdeecb", "#d8ecdf"
    root = {"text": f"{sp} 전체\n{n:,} 시나리오", "color": "#e8eef6",
            "children": [
                ("focal 이 보행자·자전거", {**box(g_walk, "차로 출력공간이 성립 안 함\n→ 학습에서 분리", drop_c),
                                     "children": [
                                         ("", box(g_walk & rn, "그중 경로 0개", drop_c)),
                                         ("", box(g_walk & ~rn, "그중 경로 있음\n(보도 위를 걷는다)", drop_c))]}),
                ("차량 계열", {"text": f"차량·버스·오토바이\n{int((~walk).sum()):,}건 ({100 * (~walk).mean():.2f}%)",
                           "color": "#e8eef6",
                           "children": [
                               ("경로 0개", box(g_rn, "91%는 거리 측정 버그(정점 vs 폴리라인)\n→ 먼저 고치고 남는 것만 버린다", fix_c)),
                               ("경로 있음 · 정답이 밴드 밖", box(g_cf, "밴드·후보를 보정\n(버리면 회전이 통째로 빠진다)", fix_c)),
                               ("경로 있음 · 정답이 밴드 안", box(g_ok, "그대로 학습", keep_c))]})]}
    fig = plt.figure(figsize=(15.4, 9.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    C.draw_tree(ax, root, fontsize=8.6, edge_fs=7.8, y_gap=1.0)
    ax.set_title(f"[{sp}] 정제 처리 도식 — 갈래별 건수 (상호배타)", fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    DC = R.drop_cost(D)
    keys = list(DC[sp])
    vals = {s_: [DC[s_][k]["pct"] for k in keys] for s_ in D}
    ns = {s_: [DC[s_][k]["n"] for k in keys] for s_ in D}
    bar_pairs(ax2, keys, vals, ns, list(D), fs=7.0)
    ax2.set_xlabel("버리면 빠지는 시나리오 비율 [%]")
    ax2.set_title("기준별 '버리면 얼마나 잃나'  (n = 빠지는 시나리오 수)")
    ax2.legend(loc="lower right")
    fig.suptitle("무엇을 버리고 무엇을 고칠 것인가", y=0.995, fontsize=12.5)
    fig.subplots_adjust(left=0.235, right=0.97, top=0.94, bottom=0.07)
    return C.savefig(fig, OUT / "c9_decision.png")


# ---------------------------------------------------------------- c10 버리면 무엇을 잃나 (정확도)
def fig_c10(D, plt):
    sc = R.score_join(D)
    if sc is None:
        print("  (점수 parquet 이 없어 c10 을 건너뛴다)")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    base = sc["전체 minADE6"]

    ax = axes[0]
    keys = [k for k in sc["집단"] if k != "전체"]
    v = [sc["집단"][k]["minADE6"] for k in keys]
    n = [sc["집단"][k]["n"] for k in keys]
    y = np.arange(len(keys))
    col = [C.C_RED if x > base else C.C_GREEN for x in v]
    ax.barh(y, v, color=col, height=0.62, zorder=3)
    ax.axvline(base, color=C.INK2, lw=1.2, ls="--", label=f"val 전체 {base:.3f} m")
    ax.legend(loc="lower right", fontsize=8.0)
    for j, (vv, nn) in enumerate(zip(v, n)):
        ax.text(vv + 0.08, y[j], f"{vv:.3f}  (n={nn:,})", va="center", fontsize=7.4, color=C.INK2)
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=8.2); ax.invert_yaxis()
    ax.set_xlim(0, max(v) * 1.35)
    ax.set_xlabel("minADE6 [m]")
    ax.set_title(f"(a) 집단별 정확도  ({sc['태그'][:26]}…)")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ch = [sc["빼면"][k]["변화"] for k in keys]
    ax.barh(y, ch, color=[C.C_GREEN if x < 0 else C.C_RED for x in ch], height=0.62, zorder=3)
    ax.axvline(0, color=C.AXIS, lw=1.0)
    for j, (cc, k) in enumerate(zip(ch, keys)):
        ax.text(cc + (0.002 if cc >= 0 else -0.002), y[j],
                f"{cc:+.4f} → {sc['빼면'][k]['남는 minADE6']:.4f}", va="center",
                ha="left" if cc >= 0 else "right", fontsize=7.4, color=C.INK2)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlim(min(ch) * 1.9 - 0.01, max(ch) * 1.9 + 0.01)
    ax.set_xlabel("그 집단을 빼면 전체 minADE6 가 얼마나 변하나 [m]")
    ax.set_title("(b) 초록 = 빼면 평균이 좋아진다")
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    ck = list(sc["커버리지 실패 원인별"])
    cv = [sc["커버리지 실패 원인별"][k]["minADE6"] for k in ck]
    cn = [sc["커버리지 실패 원인별"][k]["n"] for k in ck]
    y2 = np.arange(len(ck))
    ax.barh(y2, cv, color=C.C_ORANGE, height=0.6, zorder=3)
    ax.axvline(base, color=C.INK2, lw=1.2, ls="--")
    for j, (vv, nn) in enumerate(zip(cv, cn)):
        ax.text(vv + 0.08, y2[j], f"{vv:.2f}  (n={nn:,})", va="center", fontsize=7.4, color=C.INK2)
    ax.set_yticks(y2); ax.set_yticklabels(ck, fontsize=8.2); ax.invert_yaxis()
    ax.set_xlim(0, max(cv) * 1.35)
    ax.set_xlabel("minADE6 [m]")
    ax.set_title("(c) 커버리지 실패 원인별 정확도")
    ax.grid(axis="y", visible=False)
    fig.suptitle("버리면 무엇을 잃나 — 이미 학습된 판의 val 24,988 시나리오별 점수와 붙여 본다",
                 y=0.98, fontsize=12.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return C.savefig(fig, OUT / "c10_score.png")


FIGS = {"c1": fig_c1, "c2": fig_c2, "c3": fig_c3, "c4": fig_c4, "c5": fig_c5,
        "c6": fig_c6, "c7": fig_c7, "c8": fig_c8, "c9": fig_c9, "c10": fig_c10}


DOCS_FIG = REPO / "docs/figures/v4/cleanse"     # .gitignore 가 viz/**.png 는 무시하고 docs/figures 만 남긴다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--ramp-n", type=int, default=3000)
    ap.add_argument("--no-copy", action="store_true", help="docs/figures 로 복사하지 않는다")
    a = ap.parse_args()
    plt = C.setup_mpl()
    D = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, fn in FIGS.items():
        if a.only and k not in a.only:
            continue
        p = fn(D, plt, a.ramp_n) if k == "c5" else fn(D, plt)
        if p is not None and not a.no_copy:
            import shutil
            DOCS_FIG.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, DOCS_FIG / Path(p).name)
        print("wrote", p)


if __name__ == "__main__":
    main()
