"""
visualize_lane_rules_grid.py - 규칙의 최종 성능을 시나리오 단위로 점수화해 격자로 본다.

점수 = 적중(0/1) × 후보 축소율
   적중     : focal 이 실제로 도착한 차로가 규칙상 도달 가능한가
   후보 축소율: 1 - (규칙 적용 후보 수 / 규칙 없을 때 후보 수)

놓치면(적중 0) 아무리 많이 줄여도 0점이다. 규칙이 맞으면서 동시에 좁혀야 점수가 오른다.

  python src/visualize_lane_rules_grid.py --scan 300
"""
import argparse, json, sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append("src")
from lane_graph import LaneGraph, Move, REACH_MARGIN_M, LANE_CHANGE_COST_M
from lane_graph_check import focal_track, OBS_LEN, MIN_MOVE_M, _naive_reachable

BG="#14151a"; OPP="#e0554a"; REACH="#4ec98a"; SAMEDIR="#6f7787"
PAST="#f2f2f2"; FUT="#59a5f5"; MISS="#ffb454"


def score_scenario(d):
    raw=json.loads((d/f"log_map_archive_{d.name}.json").read_text())
    g=LaneGraph.from_json_dict(raw)
    pos,head=focal_track(d)
    if len(pos)<OBS_LEN+2: return None
    dist=float(np.linalg.norm(np.diff(pos[OBS_LEN-1:],axis=0),axis=1).sum())
    if dist<MIN_MOVE_M: return None
    h0=np.array([np.cos(head[OBS_LEN-1]),np.sin(head[OBS_LEN-1])])
    h1=np.array([np.cos(head[-1]),np.sin(head[-1])])
    S=g.candidate_lanes(pos[OBS_LEN-1],h0)
    E=g.candidate_lanes(pos[-1],h1,top_k=8)
    if not S or not E: return None
    budget=dist+REACH_MARGIN_M
    R=g.reachable(S,budget); N=_naive_reachable(g,S,budget)
    hit=any(e in R for e in E)
    cut=1.0-len(R)/max(len(N),1)
    return dict(dir=d, g=g, pos=pos, h0=h0, R=R, N=N, E=E, hit=hit,
                cut=cut, score=(1.0 if hit else 0.0)*cut, dist=dist)


def panel(ax, r):
    g,R,h0,pos = r["g"], r["R"], r["h0"], r["pos"]
    for lid,ln in g.lanes.items():
        c=ln.centerline
        if float(ln.direction@h0)<=0:  col,lw,a = OPP,1.3,0.9
        elif lid in R:                 col,lw,a = REACH,2.2,1.0
        else:                          col,lw,a = SAMEDIR,1.0,0.6
        ax.plot(c[:,0],c[:,1],color=col,lw=lw,alpha=a,zorder=2,solid_capstyle="round")
    if not r["hit"]:                       # 놓친 도착 차로를 따로 표시
        for e in r["E"]:
            c=g.lanes[e].centerline
            ax.plot(c[:,0],c[:,1],color=MISS,lw=2.6,zorder=5,solid_capstyle="round")
    ax.plot(pos[:OBS_LEN,0],pos[:OBS_LEN,1],color=PAST,lw=2.0,zorder=6)
    ax.plot(pos[OBS_LEN:,0],pos[OBS_LEN:,1],color=FUT,lw=2.0,ls="--",zorder=6)
    ax.scatter(*pos[OBS_LEN-1],s=40,color=FUT,ec="white",lw=1.0,zorder=7)
    cx,cy=pos[OBS_LEN-1]
    ax.set_xlim(cx-55,cx+55); ax.set_ylim(cy-42,cy+42)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#2a2c33")
    tag = "HIT" if r["hit"] else "MISS"
    ax.set_title(f"{r['dir'].name[:8]}   score {r['score']:.2f}   {tag}   "
                 f"{len(r['N'])}->{len(R)} lanes",
                 color="#d8dae0" if r["hit"] else MISS, fontsize=8.5)


def grid(rows, path, heading):
    fig,axes=plt.subplots(3,3,figsize=(15,12),facecolor=BG)
    for ax,r in zip(axes.ravel(), rows):
        ax.set_facecolor(BG); panel(ax,r)
    for ax in axes.ravel()[len(rows):]:
        ax.set_facecolor(BG); ax.axis("off")
    fig.suptitle(heading, color="#e6e8ec", fontsize=13, y=0.985)
    h=[Line2D([],[],color=REACH,lw=2.5,label="reachable by rules"),
       Line2D([],[],color=SAMEDIR,lw=1.5,label="same direction, not reachable"),
       Line2D([],[],color=OPP,lw=2,label="oncoming (excluded)"),
       Line2D([],[],color=MISS,lw=2.5,label="actual arrival lane the rules missed"),
       Line2D([],[],color=PAST,lw=2,label="focal past"),
       Line2D([],[],color=FUT,lw=2,ls="--",label="focal future (GT)")]
    fig.legend(handles=h,loc="lower center",ncol=3,facecolor=BG,edgecolor="#2a2c33",
               labelcolor="#c9ccd2",fontsize=9)
    fig.tight_layout(rect=[0,0.055,1,0.975])
    fig.savefig(path,dpi=105,facecolor=BG,bbox_inches="tight")
    print(f"saved -> {path}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scan",type=int,default=300)
    a=ap.parse_args()
    root=Path("/data/argoverse2/motion_forecasting/val")
    dirs=[p for p in sorted(root.iterdir()) if p.is_dir()][:a.scan]
    rs=[x for x in (score_scenario(d) for d in dirs) if x]
    s=np.array([r["score"] for r in rs])
    hit=np.array([r["hit"] for r in rs])
    # 0점에는 두 종류가 섞인다: 놓친 것과, 맞혔지만 후보가 하나도 안 줄어든 것.
    n_miss=int((~hit).sum()); n_nocut=int(((s==0)&hit).sum())
    print(f"평가 {len(rs)}개  점수 평균 {s.mean():.3f}  중앙값 {np.median(s):.3f}")
    print(f"  적중 {100*hit.mean():.1f}%  (놓침 {n_miss}개)")
    print(f"  0점 {int((s==0).sum())}개 = 놓침 {n_miss} + 맞혔지만 축소 0 {n_nocut}")

    med=np.median(s)
    typical=[rs[i] for i in np.argsort(np.abs(s-med))[:9]]
    # 최악 9개는 '놓친 것' 우선, 그다음 점수 낮은 순
    order=np.lexsort((s, hit))
    worst=[rs[i] for i in order[:9]]
    grid(typical, "lane_rules_typical.png",
         f"Typical cases (score near median {med:.2f})  -  score = hit x candidate reduction")
    grid(worst, "lane_rules_worst.png",
         "Worst 9 cases  -  where the rules lose the actual arrival lane")


if __name__=="__main__":
    main()
