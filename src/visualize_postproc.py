"""
visualize_postproc.py - 규칙 후처리(①안) 전후를 한 그림에 보여준다.

각 칸에:
  - 실제 차량 궤적(정답, 파랑 굵게) + 과거 5초(흰색)
  - 살아남은 모드: 빨강 점선, 굵기·진하기가 후처리 후 확률에 비례
  - 억제된 모드: 회색 점선 + X 표시 (규칙 위반으로 확률 0이 된 것)
  - 1순위 모드: 초록 (후처리로 1순위가 바뀌었으면 제목에 표시)
칸 제목에 '왜 이렇게 예측했나'를 읽을 수 있는 맥락을 적는다:
  현재 속력 / 교차로 여부 / 도달 가능 차로 수 / 억제된 모드 수 / 정답의 회전각

  python src/visualize_postproc.py --n 9 --out postproc_avg.png
"""
import argparse, sys
from pathlib import Path
sys.path.append("src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 라벨용 CJK 폰트 (없으면 기본 폰트로 떨어지고 한글만 깨진다)
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.patheffects as pe

import hdmap_render as hr

MAP = Path("/data/argoverse2/motion_forecasting/val")
PAST, GT, KEEP, SUPP, TOP = "#c9ccd2", "#4da3ff", "#ff5c33", "#5a5f68", "#5ce08a"


def _outline(lw):
    return [pe.Stroke(linewidth=lw + 1.8, foreground="#101115"), pe.Normal()]


def draw(ax, rec, aspect=1.0, min_span=60.0):
    xy = np.concatenate([rec["hist"], rec["gt"]] + list(rec["pred"]), axis=0)
    lo, hi = xy.min(0), xy.max(0)
    cx, cy = (lo + hi) / 2
    xr, yr = (hi - lo) * 1.2 + 14.0
    hw = max(xr / 2, min_span / 2, (yr / 2) * aspect)
    xlim, ylim = (cx - hw, cx + hw), (cy - hw / aspect, cy + hw / aspect)
    hr.draw_scene(ax, rec["scene"], xlim, ylim, mark_lw=1.4)

    # 억제된 모드 먼저 (뒤로)
    for m in np.where(rec["viol"])[0]:
        t = rec["pred"][m]
        ax.plot(t[:, 0], t[:, 1], color=SUPP, lw=1.3, ls=(0, (2, 3)), zorder=4)
        ax.scatter(t[-1, 0], t[-1, 1], marker="x", s=42, color="#8b3a3a", lw=1.8, zorder=6)
    # 살아남은 모드
    for m in np.where(~rec["viol"])[0]:
        if m == rec["top_new"]:
            continue
        t = rec["pred"][m]
        ax.plot(t[:, 0], t[:, 1], color=KEEP, lw=1.5, ls="--",
                alpha=0.35 + 0.6 * float(rec["p_new"][m]), zorder=5)
    h, g = rec["hist"], rec["gt"]
    ax.plot(h[:, 0], h[:, 1], color=PAST, lw=2.4, zorder=7, path_effects=_outline(2.4))
    ax.plot(g[:, 0], g[:, 1], color=GT, lw=3.0, zorder=8, path_effects=_outline(3.0))
    t = rec["pred"][rec["top_new"]]
    ax.plot(t[:, 0], t[:, 1], color=TOP, lw=2.4, ls="--", zorder=9, path_effects=_outline(2.4))
    hr.draw_vehicle(ax, 0, 0, 0, color="#ff5c33")
    hr.draw_scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3a3d44")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/traj_lane_rules.npz")
    ap.add_argument("--pp", default="runs/pp_lane_rules.npz")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--out", default="postproc_avg.png")
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    p = np.load(args.pp, allow_pickle=True)
    N = len(p["prob_new"])
    ade = np.linalg.norm(d["pred"][:N] - d["gt"][:N, None], axis=3).mean(2)
    fde = np.linalg.norm(d["pred"][:N, :, -1] - d["gt"][:N, -1][:, None], axis=2)
    minade = ade.min(1)

    # 평균 근처 + 억제가 실제로 일어난 장면 (후처리 효과가 보이는 칸을 고른다)
    nsupp = p["viol"].sum(1)
    ok = np.where((nsupp > 0) & (nsupp < 6))[0]
    order = ok[np.argsort(np.abs(minade[ok] - np.median(minade)))]
    picks = sorted(order[:args.n])
    print(f"후보 {len(ok)}개 중 평균 근처 {len(picks)}개 선택")

    fig, axes = plt.subplots(3, 3, figsize=(17, 17.6), facecolor=hr.BG)
    for ax, i in zip(axes.flat, picks):
        sid = str(d["scenario_id"][i])
        rec = {"hist": d["hist"][i], "gt": d["gt"][i], "pred": d["pred"][i],
               "viol": p["viol"][i], "p_new": p["prob_new"][i],
               "top_new": int(p["prob_new"][i].argmax()),
               "scene": hr.build_scene(MAP/sid/f"log_map_archive_{sid}.json",
                                       d["origin"][i], float(d["theta"][i]))}
        draw(ax, rec)
        t_old = int(d["probs"][i].argmax())
        changed = "  1순위 교체" if t_old != rec["top_new"] else ""
        turn = float(p["gt_turn_deg"][i])
        kind = "우회전" if turn < -25 else ("좌회전" if turn > 25 else "직진")
        ax.set_title(
            f"{sid[:8]}  {p['speed_kph'][i]:.0f}km/h  {kind}({turn:+.0f}°)"
            f"{'  교차로' if p['at_inter'][i] > 0.5 else ''}\n"
            f"억제 {int(p['viol'][i].sum())}/6  도달가능차로 {int(p['n_reach'][i])}개  "
            f"minADE {minade[i]:.2f}m  top1 {ade[i, rec['top_new']]:.2f}m{changed}",
            fontsize=10.5, color=hr.TEXT)
    lines = [plt.Line2D([], [], color=PAST, lw=2.4), plt.Line2D([], [], color=GT, lw=3),
             plt.Line2D([], [], color=TOP, lw=2.4, ls="--"),
             plt.Line2D([], [], color=KEEP, lw=1.5, ls="--"),
             plt.Line2D([], [], color=SUPP, lw=1.3, ls=(0, (2, 3)))]
    labs = ["과거 5초", "실제 차량 궤적(정답)", "후처리 후 1순위",
            "살아남은 모드", "억제된 모드(규칙 위반)"]
    leg = axes.flat[0].legend(lines, labs, fontsize=9.5, loc="upper left",
                              facecolor="#22242a", edgecolor="#3a3d44", framealpha=0.92)
    for t in leg.get_texts():
        t.set_color(hr.TEXT)
    fig.suptitle("규칙 후처리(①안) — 회색 X 는 규칙 위반으로 확률을 0으로 깎은 모드",
                 fontsize=15, color=hr.TEXT, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=120, facecolor=hr.BG, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
