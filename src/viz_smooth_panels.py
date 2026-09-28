"""
viz_smooth_panels.py - 평활 전 vs 후 대표 시나리오 패널. smooth_study.py --task cases 결과(cases.npz)를 읽는다.

한 장에 시나리오 하나:
  왼쪽  지도 위 원본 궤적과 평활 궤적 (점 = 0.1 s 스텝, 2 Hz 표본은 큰 빈 원)
  오른쪽 시계열  heading vs 시간(가장 중요) · 속력 · 가속도 · 요레이트
  램프 구간(인덱스 0~4)과 정지 구간(AV2 속력 < 1 m/s)을 띠로 표시

  python src/viz_smooth_panels.py                 # 상황별 3개씩
  python src/viz_smooth_panels.py --methods none,sg5_2,g025,g040,kf_ca_v
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import viz_v4_common as V
import smooth_study as S

OUT, DATA = S.OUT / "panels", S.DATA
SHOW = ["none", "sg5_2", "g025", "g040", "kf_ca_v"]
LBL = {"none": "평활 없음", "sg5_2": "SG(5,2) idx4~ [현재]", "g025": "가우시안 σ=0.25 s",
       "g040": "가우시안 σ=0.40 s", "bw05": "Butterworth 0.5 Hz", "kf_ca_v": "칼만 CA + 속도필드",
       "kin": "운동학 제약 적합", "ang_g025": "가우스 σ=0.25 + 각도 직접"}
COL = {"none": V.MUTED, "sg5_2": V.C_BLUE, "g025": V.C_ORANGE, "g040": V.C_RED,
       "bw05": V.C_AQUA, "kf_ca_v": V.C_GREEN, "kin": V.C_VIOLET, "ang_g025": V.C_MAGENTA}


def unwrap_deg(h):
    return np.degrees(np.unwrap(h))


def draw_one(plt, z, meta, i, methods, out):
    sid = meta["sid"][i]
    cls = meta["cls"][i]
    pos = z["pos"][i]
    vel = z["vel"][i]
    head = z["head"][i]
    origin, yaw = pos[-1], head[-1]
    R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    to_n = lambda p: (p - origin) @ R.T
    t = (np.arange(S.OBS) - (S.OBS - 1)) * S.DT               # -4.9 … 0 s
    spd_f = np.linalg.norm(vel, axis=1)
    stop_m = spd_f < 1.0

    fig = plt.figure(figsize=(15.0, 7.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1.35], hspace=0.28, wspace=0.16)
    axm = fig.add_subplot(gs[:, 0])
    axs = [fig.add_subplot(gs[r, 1]) for r in range(4)]

    # ---------------- 지도 + 궤적
    P = to_n(pos)
    pad = max(8.0, 0.18 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))
    xlim = (P[:, 0].min() - pad, P[:, 0].max() + pad)
    ylim = (P[:, 1].min() - pad, P[:, 1].max() + pad)
    sp = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    cx, cy = np.mean(xlim), np.mean(ylim)
    xlim, ylim = (cx - sp / 2, cx + sp / 2), (cy - sp / 2, cy + sp / 2)
    try:
        V.draw_scene_light(axm, V.build_scene(sid, origin, float(yaw)), xlim, ylim)
    except Exception as e:
        axm.text(0.5, 0.5, f"지도 없음: {e}", transform=axm.transAxes, ha="center", fontsize=8)
    for k in methods:
        Q = to_n(z[f"ps_{k}"][i])
        axm.plot(Q[:, 0], Q[:, 1], color=COL[k], lw=1.7 if k != "none" else 2.4,
                 alpha=1.0 if k != "none" else 0.75, zorder=10, label=LBL.get(k, k))
        axm.scatter(Q[:, 0], Q[:, 1], s=7, color=COL[k], edgecolors=V.SURF, linewidths=0.3, zorder=11)
        Q2 = to_n(z[f"p2_{k}"][i])
        axm.scatter(Q2[:, 0], Q2[:, 1], s=40, facecolors="none", edgecolors=COL[k], linewidths=1.3, zorder=12)
    V.draw_box(axm, 0, 0, 0.0, V.INK, alpha=0.25, zorder=13)
    axm.set_xlim(*xlim)
    axm.set_ylim(*ylim)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    axm.grid(False)
    V.scale_bar(axm, xlim, ylim)
    axm.legend(loc="upper left", fontsize=7.8, framealpha=0.85, frameon=True)
    axm.set_title("관측 4.9 s 궤적 (작은 점 = 0.1 s 스텝, 빈 원 = 2 Hz 표본, 회색 상자 = 예측 시작 위치)",
                  loc="left", fontsize=9.0)

    # ---------------- 시계열
    def bands(ax):
        ax.axvspan(t[0], t[S.RAMP - 1], color=V.C_YELLOW, alpha=0.18, lw=0, zorder=0)
        d = np.diff(stop_m.astype(int))
        s0 = list(np.flatnonzero(d == 1) + 1)
        s1 = list(np.flatnonzero(d == -1) + 1)
        if stop_m[0]:
            s0 = [0] + s0
        if stop_m[-1]:
            s1 = s1 + [len(t) - 1]
        for a, b in zip(s0, s1):
            ax.axvspan(t[a], t[b], color=V.C_MAGENTA, alpha=0.12, lw=0, zorder=0)

    ax = axs[0]
    c_ref = np.arctan2(vel[:, 1], vel[:, 0]) - yaw
    ref = unwrap_deg(c_ref)
    ax.plot(t, ref, color=V.INK, lw=1.4, ls="--", zorder=4, label="AV2 속도필드 course (독립 기준)")
    for k in methods:
        # 저장된 h 는 city 프레임이다 — 기준(속도필드 course)과 같은 정규화 프레임으로 옮긴다
        y = unwrap_deg(S.wrap(z[f"h_{k}"][i] - yaw))
        y = y - np.round((np.median(y - ref)) / 360.0) * 360.0
        ax.plot(t, y, color=COL[k], lw=1.6, zorder=5, **V.step_kw(10, COL[k], ms=2.8, mew=0.3))
        y2 = unwrap_deg(S.wrap(z[f"h2_{k}"][i] - yaw))
        y2 = y2 - np.round((np.median(y2 - ref[S.IDX_2HZ])) / 360.0) * 360.0
        ax.plot(t[S.IDX_2HZ], y2, color=COL[k], lw=0, **V.step_kw(2, "none", mec=COL[k], mew=1.3))
    ax.set_ylabel("진행방향 h [°]")
    ax.set_title("heading vs 시간 — 가장 중요한 그래프 (정규화 프레임, 큰 빈 원 = 2 Hz)", loc="left", fontsize=9.4)
    ax.legend(fontsize=7.2, ncol=2, loc="best")

    ax = axs[1]
    ax.plot(t, spd_f, color=V.INK, lw=1.4, ls="--", label="AV2 속도필드")
    for k in methods:
        v = np.linalg.norm(np.diff(z[f"ps_{k}"][i], axis=0), axis=1) / S.DT
        ax.plot(t[:-1], v, color=COL[k], lw=1.4, **V.step_kw(10, COL[k], ms=2.6, mew=0.3))
    ax.set_ylabel("속력 [m/s]")
    ax.legend(fontsize=7.2, loc="best")

    ax = axs[2]
    for k in methods:
        v = np.linalg.norm(np.diff(z[f"ps_{k}"][i], axis=0), axis=1) / S.DT
        a = np.diff(v) / S.DT
        ax.plot(t[:-2], a, color=COL[k], lw=1.4, **V.step_kw(10, COL[k], ms=2.6, mew=0.3))
    ax.axhline(0, color=V.AXIS, lw=0.8)
    for s in (8, -8):
        ax.axhline(s, color=V.C_RED, lw=0.8, ls=":")
    ax.set_ylabel("가속도 [m/s²]")

    ax = axs[3]
    wref = np.degrees(S.wrap(np.diff(c_ref))) / S.DT
    ax.plot(t[:-1], wref, color=V.INK, lw=1.4, ls="--")
    for k in methods:
        w = np.diff(unwrap_deg(S.wrap(z[f"h_{k}"][i] - yaw))) / S.DT
        ax.plot(t[:-1], w, color=COL[k], lw=1.4, **V.step_kw(10, COL[k], ms=2.6, mew=0.3))
    ax.axhline(0, color=V.AXIS, lw=0.8)
    for s in (54.4, -54.4):
        ax.axhline(s, color=V.C_RED, lw=0.8, ls=":")
    ax.set_ylabel("요레이트 [°/s]")
    ax.set_xlabel("시각 [s]  (0 = 예측 시작)")
    for ax in axs:
        bands(ax)
        ax.set_xlim(t[0], t[-1])
    for ax in axs[:-1]:
        ax.set_xticklabels([])

    dh = float(z["dcourse"][i])
    sm = float(z["spmed"][i])
    dtx = "정의 안 됨(1 m/s 넘는 스텝 없음)" if not np.isfinite(dh) else f"{dh:+.1f}°"
    fig.suptitle(f"[{cls}]  {sid}   ·   관측 구간 속도필드 Δcourse {dtx} · 중앙 속력 {sm:.1f} m/s   ·   "
                 "노랑 띠 = 창 시작 램프(인덱스 0~4), 분홍 띠 = AV2 속력 < 1 m/s (이 구간은 속도필드 course 도 뜻이 없다)",
                 fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return V.savefig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(SHOW))
    a = ap.parse_args()
    methods = [m for m in a.methods.split(",") if m]
    plt = V.setup_mpl()
    z = np.load(DATA / "cases.npz")
    meta = json.loads((DATA / "cases.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for cls, idxs in meta["picks"].items():
        for j, i in enumerate(idxs):
            p = OUT / f"panel_{cls}_{j+1}.png"
            made.append(draw_one(plt, z, meta, i, methods, p))
            print("  ", p.name, flush=True)
    print(f"패널 {len(made)}장 -> {OUT}")


if __name__ == "__main__":
    main()
