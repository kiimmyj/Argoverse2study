"""
visualize_matching.py - 차로 매칭 3단계를 실제 시나리오로 그린다.

  STEP 1  시작 차로 후보 : 현재 위치 반경 5m 안 + 진행방향이 맞는 차로를 가까운 순 4개
  STEP 2  도달 범위 확장 : 후보들에서 '주행거리 + 60m' 예산만큼 진행/차선변경으로 뻗음
  STEP 3  도착 차로 판정 : 도착점 반경 5m 안 차로 중 하나라도 도달 범위에 있으면 적중

  python src/visualize_matching.py
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.append("src")
from lane_graph import LaneGraph, CAND_RADIUS_M, CAND_TOP_K, REACH_MARGIN_M
from lane_graph_check import focal_track, OBS_LEN

BG="#14151a"; GREY="#3f434c"; OPP="#e0554a"; CAND="#f0c martin"
CAND="#f0c14b"; REACH="#4ec98a"; PAST="#f2f2f2"; FUT="#59a5f5"; RING="#8f96a3"


def base(ax, g, faint=True):
    for ln in g.lanes.values():
        ax.plot(ln.centerline[:,0], ln.centerline[:,1], color=GREY,
                lw=1.0, alpha=0.55 if faint else 0.9, zorder=1)


def frame(ax, cx, cy, half_w, half_h, title):
    ax.set_xlim(cx-half_w, cx+half_w); ax.set_ylim(cy-half_h, cy+half_h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color("#2a2c33")
    ax.set_title(title, color="#e6e8ec", fontsize=10.5, pad=8)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="lane_matching.png")
    a=ap.parse_args()
    root=Path("/data/argoverse2/motion_forecasting/val")

    for d in [p for p in sorted(root.iterdir()) if p.is_dir()]:
        raw=json.loads((d/f"log_map_archive_{d.name}.json").read_text())
        g=LaneGraph.from_json_dict(raw)
        pos,head=focal_track(d)
        if len(pos)<OBS_LEN+2: continue
        dist=float(np.linalg.norm(np.diff(pos[OBS_LEN-1:],axis=0),axis=1).sum())
        h0=np.array([np.cos(head[OBS_LEN-1]),np.sin(head[OBS_LEN-1])])
        h1=np.array([np.cos(head[-1]),np.sin(head[-1])])
        S=g.candidate_lanes(pos[OBS_LEN-1],h0)
        E=g.candidate_lanes(pos[-1],h1,top_k=8)
        near_opp=sum(1 for ln in g.lanes.values()
                     if float(ln.direction@h0)<=0
                     and float(np.min(np.linalg.norm(ln.centerline-pos[OBS_LEN-1],axis=1)))<=CAND_RADIUS_M)
        if 30<dist<70 and len(S)>=3 and len(E)>=3 and 20<len(g.lanes)<70 and near_opp>=1: break

    p0,p1=pos[OBS_LEN-1],pos[-1]
    R=g.reachable(S,dist+REACH_MARGIN_M)
    hit=any(e in R for e in E)

    fig,axes=plt.subplots(1,3,figsize=(17,6.2),facecolor=BG)

    # ---- STEP 1
    ax=axes[0]; base(ax,g)
    near=[(float(np.min(np.linalg.norm(ln.centerline-p0,axis=1))),lid)
          for lid,ln in g.lanes.items()]
    for dd,lid in near:
        if dd>CAND_RADIUS_M: continue
        ln=g.lanes[lid]
        if float(ln.direction@h0)<=0:      # 반경 안이지만 방향이 반대 -> 후보 제외
            ax.plot(ln.centerline[:,0],ln.centerline[:,1],color=OPP,lw=2.4,zorder=3)
    for i,lid in enumerate(S):
        c=g.lanes[lid].centerline
        ax.plot(c[:,0],c[:,1],color=CAND,lw=2.8,zorder=4)
        m=np.argmin(np.linalg.norm(c-p0,axis=1))
        ax.annotate(f"{i+1}", xy=c[m], xytext=(6,6), textcoords="offset points",
                    color="#1a1a1a", fontsize=9, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.22", fc=CAND, ec="none"))
    ax.add_patch(Circle(p0, CAND_RADIUS_M, fill=False, ec=RING, lw=1.4, ls="--", zorder=5))
    ax.plot(pos[:OBS_LEN,0],pos[:OBS_LEN,1],color=PAST,lw=2.4,zorder=6)
    ax.scatter(*p0,s=70,color=FUT,ec="white",lw=1.2,zorder=7)
    frame(ax,*p0,21,14,f"STEP 1  start candidates\nradius {CAND_RADIUS_M:.0f} m, "
                       f"heading-matched  ->  {len(S)} candidates (max {CAND_TOP_K})")

    # ---- STEP 2
    ax=axes[1]; base(ax,g)
    for lid,ln in g.lanes.items():
        if float(ln.direction@h0)<=0:
            ax.plot(ln.centerline[:,0],ln.centerline[:,1],color=OPP,lw=1.3,alpha=0.85,zorder=2)
    for lid in R:
        c=g.lanes[lid].centerline
        ax.plot(c[:,0],c[:,1],color=REACH,lw=2.4,zorder=3)
    for lid in S:
        c=g.lanes[lid].centerline
        ax.plot(c[:,0],c[:,1],color=CAND,lw=3.0,zorder=4)
    ax.plot(pos[:OBS_LEN,0],pos[:OBS_LEN,1],color=PAST,lw=2.2,zorder=6)
    ax.scatter(*p0,s=60,color=FUT,ec="white",lw=1.2,zorder=7)
    mx,my=(p0+p1)/2
    frame(ax,mx,my,75,50,f"STEP 2  expand within budget\n"
                         f"{dist:.0f} m travelled + {REACH_MARGIN_M:.0f} m margin "
                         f"= {dist+REACH_MARGIN_M:.0f} m  ->  {len(R)} lanes reachable")

    # ---- STEP 3
    ax=axes[2]; base(ax,g)
    for lid in R:
        c=g.lanes[lid].centerline
        ax.plot(c[:,0],c[:,1],color=REACH,lw=2.4,alpha=0.9,zorder=3)
    for lid in E:
        c=g.lanes[lid].centerline
        ax.plot(c[:,0],c[:,1],color=FUT if lid in R else OPP,lw=3.0,zorder=4)
    ax.add_patch(Circle(p1, CAND_RADIUS_M, fill=False, ec=RING, lw=1.4, ls="--", zorder=5))
    ax.plot(pos[OBS_LEN:,0],pos[OBS_LEN:,1],color=FUT,lw=2.4,ls="--",zorder=6)
    ax.scatter(*p1,s=70,color=FUT,ec="white",lw=1.2,zorder=7)
    n_in=sum(1 for e in E if e in R)
    frame(ax,*p1,21,14,f"STEP 3  arrival check\n{n_in} of {len(E)} arrival candidates "
                       f"are reachable  ->  {'HIT' if hit else 'MISS'}")

    h=[Line2D([],[],color=CAND,lw=3,label="start lane candidates"),
       Line2D([],[],color=REACH,lw=2.5,label="reachable within budget"),
       Line2D([],[],color=OPP,lw=2.5,label="oncoming - excluded from candidates"),
       Line2D([],[],color=RING,lw=1.4,ls="--",label=f"{CAND_RADIUS_M:.0f} m matching radius"),
       Line2D([],[],color=PAST,lw=2.4,label="focal past"),
       Line2D([],[],color=FUT,lw=2.4,ls="--",label="focal future (GT)")]
    fig.legend(handles=h,loc="lower center",ncol=3,facecolor=BG,edgecolor="#2a2c33",
               labelcolor="#c9ccd2",fontsize=9)
    fig.suptitle(f"Lane matching, step by step   -   scenario {d.name[:8]}",
                 color="#e6e8ec",fontsize=13,y=0.98)
    fig.tight_layout(rect=[0,0.10,1,0.95])
    fig.savefig(a.out,dpi=110,facecolor=BG,bbox_inches="tight")
    print(f"saved -> {a.out}   (start {len(S)}, reachable {len(R)}, arrival {len(E)}, "
          f"{'HIT' if hit else 'MISS'})")


if __name__=="__main__":
    main()
