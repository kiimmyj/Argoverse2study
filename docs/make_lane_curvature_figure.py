"""원시 중심선 곡률이 왜 잡음인지, 스플라인 적합이 무엇을 하는지 한 장으로 보인다.

  python docs/make_lane_curvature_figure.py            # lane_curvature.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lane_frame import STEP, build_routes            # noqa: E402
from lane_graph import LaneGraph, REACH_MARGIN_M     # noqa: E402
from lane_spline import fit_route, route_vertices    # noqa: E402

BG = "#14151a"; PANEL = "#1c1e25"; TXT = "#e6e8ec"; DIM = "#9aa0aa"
RAW = "#e0554a"; FIT = "#4ec98a"; VTX = "#e8c65a"; GRID = "#2c303a"
ROOT = Path("/data/argoverse2/motion_forecasting")


def fd_kappa(pts):
    """densify 된 점열의 유한차분 곡률 — 스플라인을 안 썼을 때 나오는 것."""
    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    den[den < 1e-12] = 1e-12
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / den


def pick():
    """정점 꺾임이 크고 충분히 긴 경로 하나. 잡음의 정체가 잘 보이는 표본을 고른다."""
    best = None
    for d in sorted((ROOT / "val").iterdir())[:60]:
        g = LaneGraph.from_json_dict(
            json.loads((d / f"log_map_archive_{d.name}.json").read_text()))
        for lid in list(g.lanes)[:400]:
            routes = build_routes(g, [lid], None, max_hops=4, min_len_m=70.0, max_routes=4)
            for r in routes:
                V = route_vertices(g, r)
                a = np.arctan2(*np.diff(V, axis=0)[:, ::-1].T)
                kk = np.degrees(np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))
                score = kk.max() if len(kk) else 0.0
                if 70.0 < r["s"][-1] < 140.0 and (best is None or score > best[0]):
                    best = (score, g, r, V)
        if best and best[0] > 25.0:
            break
    return best[1:]


g, route, V = pick()
fit = fit_route(g, route)
k_raw = fd_kappa(route["pts"])

fig = plt.figure(figsize=(15.5, 10.4), facecolor=BG)
gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 1.0, 1.0], width_ratios=[1.0, 1.0],
                      hspace=0.40, wspace=0.16, left=0.06, right=0.985, top=0.895, bottom=0.065)

# ---------------------------------------------------------------- ① 기하 (전체 / 확대)
ax = fig.add_subplot(gs[0, 0], facecolor=PANEL)
ax.plot(route["pts"][:, 0], route["pts"][:, 1], color=RAW, lw=3.4, alpha=0.55,
        label="densify 한 중심선 (정점 사이를 직선으로)", zorder=2)
ax.plot(fit["pts"][:, 0], fit["pts"][:, 1], color=FIT, lw=1.6,
        label="적합한 3차 평활 스플라인", zorder=3)
ax.scatter(V[:, 0], V[:, 1], s=22, color=VTX, zorder=4, edgecolor=BG, linewidth=0.5,
           label=f"지도가 준 정점 {len(V)}개 (차로당 10점)")
ax.set_aspect("equal")
ax.set_title("① 경로 전체 — 이 배율에서는 둘이 겹쳐 보인다", color=TXT, fontsize=11.5,
             pad=7, loc="left")
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=8.5, loc="lower left")

# 확대할 곳: 정점 간격이 가장 넓으면서 굽은 구간 = 새그가 가장 큰 곳
sv = fit["s_of_vertex"]
gapmid = (sv[:-1] + sv[1:]) / 2
sag = np.abs(np.interp(gapmid, fit["s"], fit["kappa"])) * (sv[1:] - sv[:-1]) ** 2 / 8
j = int(sag.argmax())
c = fit["pts"][np.argmin(np.abs(fit["s"] - gapmid[j]))]
half = max(6.0, float(sv[j + 1] - sv[j]))
axz = fig.add_subplot(gs[0, 1], facecolor=PANEL)
axz.plot(route["pts"][:, 0], route["pts"][:, 1], color=RAW, lw=3.4, alpha=0.75, zorder=2)
axz.plot(fit["pts"][:, 0], fit["pts"][:, 1], color=FIT, lw=2.2, zorder=3)
axz.scatter(V[:, 0], V[:, 1], s=52, color=VTX, zorder=4, edgecolor=BG, linewidth=0.8)
axz.set_xlim(c[0] - half, c[0] + half); axz.set_ylim(c[1] - half, c[1] + half)
axz.set_aspect("equal")
axz.set_title(f"② 새그가 가장 큰 정점 구간(간격 {sv[j+1]-sv[j]:.1f} m)을 확대 — 직선보간이 잘라먹는 만큼",
              color=TXT, fontsize=11.5, pad=7, loc="left")
axz.annotate(f"새그 kappa·h²/8 = {sag[j]:.2f} m", xy=(c[0], c[1]),
             xytext=(0.04, 0.10), textcoords="axes fraction", color=VTX, fontsize=10,
             arrowprops=dict(arrowstyle="->", color=VTX, lw=1.2))

# ---------------------------------------------------------------- ③ 곡률
ax2 = fig.add_subplot(gs[1, :], facecolor=PANEL)
ax2.plot(route["s"], k_raw, color=RAW, lw=1.1, label="적합 전 — 유한차분")
ax2.plot(fit["s"], fit["kappa"], color=FIT, lw=2.2, label="적합 후 — kappa_lane(s)")
ax2.axhline(0, color=GRID, lw=1)
for y in (0.2, -0.2):
    ax2.axhline(y, color=DIM, lw=0.8, ls=":")
ax2.text(0.004, 0.5, "점선 = 반경 5 m", transform=ax2.transAxes, color=DIM, fontsize=8.5,
         va="center")
ax2.set_ylabel("kappa  [1/m]", color=TXT, fontsize=10)
ax2.set_title("③ 곡률 — 선분 위에서는 0, 정점에서 델타함수. 적합 후에는 회전이 실제로 "
              "일어나는 구간에 퍼진다", color=TXT, fontsize=11.5, pad=7, loc="left")
ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=9, loc="lower left")

# ---------------------------------------------------------------- ④ 곡률 변화율
ax3 = fig.add_subplot(gs[2, :], facecolor=PANEL)
ax3.plot(route["s"][:-1], np.abs(np.diff(k_raw)) / STEP, color=RAW, lw=1.1, label="적합 전")
ax3.plot(fit["s"][:-1], np.abs(np.diff(fit["kappa"])) / STEP, color=FIT, lw=2.2, label="적합 후")
ax3.set_yscale("symlog", linthresh=1e-3)
ax3.set_xlabel("호길이 s  [m]", color=TXT, fontsize=10)
ax3.set_ylabel("|dkappa/ds|  [1/m²]", color=TXT, fontsize=10)
ax3.set_title("④ 곡률 변화율 — 조향 속도에 대응한다. 세로축 로그", color=TXT,
              fontsize=11.5, pad=7, loc="left")
ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=9, loc="upper left")

for a in (ax, axz, ax2, ax3):
    a.tick_params(colors=DIM, labelsize=9)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.grid(color=GRID, lw=0.6, alpha=0.6)

fig.suptitle("v4 L0 — 경로별 곡률 테이블 kappa_lane(s)", color=TXT, fontsize=16,
             fontweight="bold", x=0.06, ha="left", y=0.962)
fig.text(0.985, 0.962, f"차로 {len(route['lanes'])}개 · 길이 {fit['len']:.1f} m · "
         f"정점 {len(V)}개", color=DIM, fontsize=10, ha="right", va="center")
out = Path(__file__).resolve().parent.parent / "lane_curvature.png"
fig.savefig(out, facecolor=BG, dpi=130)
print(f"저장: {out}")
print(f"  |kappa|     적합 전 max {np.abs(k_raw).max():8.3f}  ->  적합 후 max {np.abs(fit['kappa']).max():.4f} 1/m")
print(f"  |dkappa/ds| 적합 전 max {np.abs(np.diff(k_raw)).max()/STEP:8.3f}  ->  적합 후 max "
      f"{np.abs(np.diff(fit['kappa'])).max()/STEP:.4f} 1/m^2")
