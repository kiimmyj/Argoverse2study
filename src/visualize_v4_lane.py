"""
visualize_v4_lane.py - ④안(차로 기준 좌표계)의 출력 공간을 그림으로 보인다.

각 칸에:
  실제 차량 궤적(파랑) / 지금 모델의 자유좌표 예측(흐린 빨강) /
  ④가 표현할 수 있는 형태로 투영한 것(주황 굵게) / 선택된 경로 중심선(연두 얇게)

투영본은 정의상 도로 밖·역주행이 나올 수 없다. 그것이 ④의 보장이다.

  python src/visualize_v4_lane.py --n 9 --out v4_lane_avg.png
"""
import argparse, sys
from pathlib import Path
sys.path.append("src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager

try:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

import hdmap_render as hr

MAP = Path("/data/argoverse2/motion_forecasting/val")
PAST, GT, FREE, V4, BEST = "#c9ccd2", "#31b0ff", "#6e4038", "#ffa53d", "#5ce08a"


def _o(lw):
    return [pe.Stroke(linewidth=lw + 1.8, foreground="#101115"), pe.Normal()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/traj_lane_rules.npz")
    ap.add_argument("--sim", default="runs/v4sim.npz")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--pp", default="runs/pp_lane_rules.npz",
                    help="장면 맥락(회전각/교차로) — 좌·우회전 케이스를 섞기 위해")
    ap.add_argument("--mix", default="3,3,3", help="좌회전,우회전,직진 개수")
    ap.add_argument("--max-kinked", dest="max_kinked", type=int, default=1,
                    help="꺾임이 있는 모드 수 상한 (0 이면 완전히 매끄러운 장면만)")
    ap.add_argument("--max-ade", dest="max_ade", type=float, default=3.0,
                    help="투영 후 minADE 상한 [m]. 투영이 실패한 장면을 뺀다")
    ap.add_argument("--min-disp", dest="min_disp", type=float, default=10.0,
                    help="정답 궤적의 최소 이동거리 [m]. 정지 차량이 회전으로 잘못 뽑히는 것을 막는다")
    ap.add_argument("--out", default="v4_lane_avg.png")
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    v = np.load(args.sim, allow_pickle=True)
    N = len(v["ok"])
    gt = d["gt"][:N]
    ade_free = np.linalg.norm(d["pred"][:N] - gt[:, None], axis=3).mean(2)
    ade_v4 = np.linalg.norm(v["proj"] - gt[:, None], axis=3).mean(2)
    minade = ade_free.min(1)

    # 선택: 투영이 제대로 된 장면만 쓰고, 좌회전 / 우회전 / 직진을 섞는다.
    # (투영이 실패한 장면은 ④의 성질이 아니라 투영의 한계라 그림에 넣을 가치가 없다)
    ctx = np.load(args.pp, allow_pickle=True)
    turn, inter = ctx["gt_turn_deg"][:N], ctx["at_inter"][:N] > 0.5
    disp = np.linalg.norm(gt[:, -1] - gt[:, 0], axis=1)     # 실제 이동거리
    # 투영 궤적에 급격한 방향반전(꺾임)이 있는 장면은 뺀다
    dv = np.diff(v["proj"], axis=2)
    with np.errstate(invalid="ignore"):
        ang = np.abs(np.degrees(np.diff(np.arctan2(dv[..., 1], dv[..., 0]), axis=2)))
    ang = np.minimum(ang, 360 - ang)
    moving = np.linalg.norm(dv, axis=3)[..., :-1] > 0.05
    kinked = np.any(np.where(moving, ang, 0) > 100, axis=2)      # 모드별 꺾임 여부
    fde_all = np.linalg.norm(v["proj"][:, :, -1] - gt[:, None, -1], axis=2)
    # 투영 실패 모드는 큰 값으로 눌러 argmin 에서 빠지게 한다 (전부 실패면 0번, 어차피 걸러짐)
    bidx = np.argmin(np.where(v["ok"] & np.isfinite(fde_all), fde_all, np.inf), axis=1)
    best_smooth = ~kinked[np.arange(N), bidx]                    # 초록선(best)은 반드시 매끄럽게
    smooth_ok = (kinked.sum(axis=1) <= args.max_kinked) & best_smooth
    with np.errstate(invalid="ignore"):
        best_v4 = np.nanmin(np.where(v["ok"], ade_v4, np.nan), axis=1)
        good = (v["ok"].all(1) & np.isfinite(best_v4)
                & (best_v4 <= args.max_ade)                    # 투영이 정확했던 장면만
                & (v["off_after"].sum(1) == 0)                 # 투영 후 위반이 0 인 장면만
                & (v["wrong_after"].sum(1) == 0)               # (그림의 주장과 어긋나지 않게)
                & (disp >= args.min_disp)
                & smooth_ok)
    # 이동거리 조건이 없으면 '거의 정지한 차'가 회전으로 뽑힌다 —
    # 회전각을 궤적 양끝 방향차로 재는데, 안 움직이면 그 방향이 노이즈이기 때문.
    nl, nr, ns = (int(x) for x in args.mix.split(","))
    picks = []
    for lab, mask, k in [("좌회전", good & (turn > 30) & inter, nl),
                         ("우회전", good & (turn < -30) & inter, nr),
                         ("직진", good & (np.abs(turn) < 10), ns)]:
        idx = np.where(mask)[0]
        if lab == "직진":                      # 직진은 평균에 가까운 것
            idx = idx[np.argsort(np.abs(minade[idx] - np.median(minade)))]
        else:                                  # 회전은 많이 움직인 것부터 (그림이 잘 보인다)
            idx = idx[np.argsort(-disp[idx])]
        picks += list(idx[:k])
        print(f"  {lab} 후보 {int(mask.sum())}개 중 {min(k, len(idx))}개"
              f"  (이동 {np.round(disp[idx[:k]], 0)} m)")
    picks = picks[:args.n]

    fig, axes = plt.subplots(3, 3, figsize=(17, 17.6), facecolor=hr.BG)
    for ax, i in zip(axes.flat, picks):
        sid = str(d["scenario_id"][i])
        scene = hr.build_scene(MAP / sid / f"log_map_archive_{sid}.json",
                               d["origin"][i], float(d["theta"][i]))
        free, v4 = d["pred"][i], v["proj"][i]
        xy = np.concatenate([d["hist"][i], gt[i], free.reshape(-1, 2),
                             v4.reshape(-1, 2)], axis=0)
        xy = xy[~np.isnan(xy).any(1)]
        lo, hi = xy.min(0), xy.max(0)
        cx, cy = (lo + hi) / 2
        hw = max(*(hi - lo) * 0.62, 30.0)
        xlim, ylim = (cx - hw, cx + hw), (cy - hw, cy + hw)
        hr.draw_scene(ax, scene, xlim, ylim, mark_lw=1.4)

        for m in range(6):                                   # 현재 v3 (참고용, 흐리게)
            t = free[m]
            ax.plot(t[:, 0], t[:, 1], color=FREE, lw=1.4, ls="--", alpha=0.45, zorder=3)
        # v4 예측 6개 중 정답에 가장 가까운 것을 best 로 표시
        fde_v4 = np.linalg.norm(v4[:, -1] - gt[i, -1], axis=1)
        bi = int(np.nanargmin(np.where(v["ok"][i], fde_v4, np.nan)))
        for m in range(6):
            t = v4[m]
            if np.isnan(t).any() or m == bi:
                continue
            ax.plot(t[:, 0], t[:, 1], color=V4, lw=2.3, alpha=0.9, zorder=5)
        h, g = d["hist"][i], gt[i]
        ax.plot(h[:, 0], h[:, 1], color=PAST, lw=2.4, zorder=6, path_effects=_o(2.4))
        # 실제 궤적은 항상 맨 위에. 짧아서 안 보이는 경우가 있어 끝점에 점을 찍는다.
        ax.plot(g[:, 0], g[:, 1], color=GT, lw=2.8, zorder=11,
                path_effects=[pe.Stroke(linewidth=5.0, foreground="#0b1520"), pe.Normal()],
                solid_capstyle="round")
        # best 는 정답 위에 점선으로 얹는다. 정확히 맞을수록 파랑 위에 초록 점선이 보인다.
        tb = v4[bi]
        if not np.isnan(tb).any():
            ax.plot(tb[:, 0], tb[:, 1], color=BEST, lw=2.4, ls=(0, (4.5, 3.5)), zorder=13,
                    path_effects=_o(2.4))
        hr.draw_vehicle(ax, 0, 0, 0, color="#ff5c33")
        hr.draw_scale_bar(ax, xlim, ylim)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#3a3d44")
        tg = float(turn[i])
        kind = "좌회전" if tg > 30 else ("우회전" if tg < -30 else "직진")
        ax.set_title(
            f"{sid[:8]}  {kind}({tg:+.0f}°)"
            f"{'·교차로' if inter[i] else ''}  경로후보 {int(v['n_route'][i])}개\n"
            f"도로이탈 {int(v['off_before'][i].sum())}/6 → {int(v['off_after'][i].sum())}/6   "
            f"역주행 {int(v['wrong_before'][i].sum())}/6 → {int(v['wrong_after'][i].sum())}/6   "
            f"minADE {minade[i]:.2f} → {np.nanmin(ade_v4[i]):.2f}m",
            fontsize=10.5, color=hr.TEXT)

    lines = [plt.Line2D([], [], color=PAST, lw=2.3),
             plt.Line2D([], [], color=GT, lw=2.8),
             plt.Line2D([], [], color=BEST, lw=2.4, ls=(0, (4, 3))),
             plt.Line2D([], [], color=V4, lw=2.3),
             plt.Line2D([], [], color=FREE, lw=1.4, ls="--", alpha=0.6)]
    labs = ["과거 5초 (관측 구간)", "실제 차량이 간 길 (정답)",
            "v4 예측 · best of 6", "v4 예측 · 나머지 5개",
            "현재 v3 예측 (비교용)"]
    leg = axes.flat[0].legend(lines, labs, fontsize=9.5, loc="upper left",
                              facecolor="#22242a", edgecolor="#3a3d44", framealpha=0.92)
    for t in leg.get_texts():
        t.set_color(hr.TEXT)
    fig.suptitle("v4 예측 결과 — 차로 기준 좌표계 (최종 형태)",
                 fontsize=18, color="#e8ecf2", y=0.995, va="top")
    fig.text(0.5, 0.9755, "갈 수 있는 차로 위에서만 예측하므로 도로 이탈과 역주행이 나올 수 없다"
             "   ·   칸 제목의 0/6 은 6개 예측 중 위반 개수",
             ha="center", va="top", fontsize=12, color="#9aa3ad")
    fig.text(0.5, 0.006, "※ 현재 그림은 v3 예측을 v4 출력 공간으로 투영해 만든 예시다. "
             "제약의 효과를 보이기 위한 것이며 v4 학습 후 정확도와는 다르다.",
             ha="center", fontsize=9.5, color="#6b727b")
    fig.tight_layout(rect=[0.006, 0.018, 0.994, 0.9635])
    fig.savefig(args.out, dpi=120, facecolor=hr.BG)   # bbox_inches="tight" 는 제목 여백을 깎아 겹친다
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
