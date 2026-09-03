"""
visualize_lane_rules.py - lane_graph 의 이동 규칙을 지도 위에 그려서 눈으로 확인한다.

  왼쪽  : 규칙 없이 본 지도 (dataset_map.py 가 모델에 주는 것과 같은 상태)
  오른쪽: 규칙 적용 — 대향차로/도달 가능 여부/차선변경 가능 방향이 구분된다

  python src/visualize_lane_rules.py --n 2
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append("src")
from lane_graph import LaneGraph, Move, REACH_MARGIN_M
from lane_graph_check import focal_track, OBS_LEN

BG = "#14151a"; GREY = "#4a4d55"
OPP = "#e0554a"        # 대향차로 (가면 역주행)
REACH = "#4ec98a"      # 규칙상 도달 가능
SAMEDIR = "#6f7787"    # 같은 방향이지만 도달 불가
PAST = "#f2f2f2"; FUT = "#59a5f5"


def draw(ax, g, reach, h0, title, rules):
    for lid, ln in g.lanes.items():
        c = ln.centerline
        if not rules:
            col, lw, a = GREY, 1.2, 0.9
        elif float(ln.direction @ h0) <= 0:
            col, lw, a = OPP, 1.6, 0.95
        elif lid in reach:
            col, lw, a = REACH, 2.4, 1.0
        else:
            col, lw, a = SAMEDIR, 1.2, 0.7
        ax.plot(c[:, 0], c[:, 1], color=col, lw=lw, alpha=a, zorder=2, solid_capstyle="round")
        if rules and float(ln.direction @ h0) > 0:      # 진행 방향 화살표
            m = len(c) // 2
            d = ln.direction * 3.0
            ax.arrow(c[m, 0], c[m, 1], d[0], d[1], head_width=1.4, head_length=1.6,
                     fc=col, ec=col, alpha=0.8, zorder=3, length_includes_head=True)
    if rules:                                            # 차선변경 가능한 연결
        for lid, es in g.edges.items():
            for nxt, mv in es:
                if mv not in (Move.LEFT, Move.RIGHT):
                    continue
                a = g.lanes[lid].centerline.mean(axis=0)
                b = g.lanes[nxt].centerline.mean(axis=0)
                ax.annotate("", xy=b, xytext=a, zorder=4,
                            arrowprops=dict(arrowstyle="->", color="#e8c65a",
                                            lw=1.0, alpha=0.75, shrinkA=2, shrinkB=2))
    ax.set_title(title, color="#d8dae0", fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="lane_rules.png")
    a = ap.parse_args()

    root = Path("/data/argoverse2/motion_forecasting/val")
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]

    picked = []
    for d in dirs:
        raw = json.loads((d / f"log_map_archive_{d.name}.json").read_text())
        g = LaneGraph.from_json_dict(raw)
        pos, head = focal_track(d)
        if len(pos) < OBS_LEN + 2:
            continue
        dist = float(np.linalg.norm(np.diff(pos[OBS_LEN - 1:], axis=0), axis=1).sum())
        h0 = np.array([np.cos(head[OBS_LEN - 1]), np.sin(head[OBS_LEN - 1])])
        n_opp = sum(1 for l in g.lanes.values() if float(l.direction @ h0) <= 0)
        # 대향차로가 많고 실제로 움직이는 장면을 고른다
        if dist > 25 and n_opp >= 8:
            picked.append((d, g, pos, h0, dist))
        if len(picked) >= a.n:
            break

    fig, axes = plt.subplots(len(picked), 2, figsize=(13, 6 * len(picked)),
                             facecolor=BG, squeeze=False)
    for r, (d, g, pos, h0, dist) in enumerate(picked):
        S = g.candidate_lanes(pos[OBS_LEN - 1], h0)
        R = g.reachable(S, dist + REACH_MARGIN_M)
        n_opp = sum(1 for l in g.lanes.values() if float(l.direction @ h0) <= 0)
        for c, rules in ((0, False), (1, True)):
            ax = axes[r][c]
            ax.set_facecolor(BG)
            # 그림 안 글자는 영문으로 둔다 (matplotlib 기본 폰트에 한글이 없다)
            draw(ax, g, R, h0,
                 f"NO RULES  -  all {len(g.lanes)} lanes look identical" if not rules
                 else f"WITH RULES  -  {n_opp} oncoming excluded, {len(R)} reachable",
                 rules)
            ax.plot(pos[:OBS_LEN, 0], pos[:OBS_LEN, 1], color=PAST, lw=2.6, zorder=6)
            ax.plot(pos[OBS_LEN:, 0], pos[OBS_LEN:, 1], color=FUT, lw=2.6, ls="--", zorder=6)
            ax.scatter(*pos[OBS_LEN - 1], s=70, color=FUT, ec="white", lw=1.2, zorder=7)
            cx, cy = pos[OBS_LEN - 1]
            ax.set_xlim(cx - 60, cx + 60); ax.set_ylim(cy - 45, cy + 45)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#2a2c33")

    handles = [Line2D([], [], color=REACH, lw=2.5, label="reachable by rules"),
               Line2D([], [], color=SAMEDIR, lw=1.5, label="same direction, not reachable"),
               Line2D([], [], color=OPP, lw=2, label="oncoming (wrong-way if entered)"),
               Line2D([], [], color="#e8c65a", lw=1.2, label="lane change allowed"),
               Line2D([], [], color=PAST, lw=2.5, label="focal past"),
               Line2D([], [], color=FUT, lw=2.5, ls="--", label="focal future (GT)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, facecolor=BG,
               edgecolor="#2a2c33", labelcolor="#c9ccd2", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(a.out, dpi=110, facecolor=BG, bbox_inches="tight")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
