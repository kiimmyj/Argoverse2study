"""2026-10-29 미팅용 그림 다섯 장 — 핵심만 뽑아 한 장씩 (docs/meeting_1029_core.md 가 쓴다).

왜 따로 만드나 — 조사 문서(meeting_1029_ah.md)는 근거를 다 적어 길다. 미팅은 결론을 전달하는
자리라, 같은 수치를 그림 한 장에 하나의 주장으로 접어 둔다. 숫자는 전부 조사 문서·실측
기록에서 그대로 가져오고, 여기서 새로 계산하는 것은 m3(θ = h − k 실례)뿐이다.

  m1_v4_why.png      v4 를 왜 접는가 — 개별 효과는 작고 합치면 더해지지 않는다
  m2_invariance.png  조사 핵심 — invariance 방식이 θ 신규성을 가른다(역상관)
  m3_theta.png       θ = h − k 가 실제 장면에서 무엇인가 (모델 없이 데이터만)
  m4_rate.png        5 Hz 질문 — 길이는 답이 있고 레이트는 비어 있다
  m5_plan.png        SIMPL 파이프라인 검증과 실험 단계·비용

  python viz_meeting_1029.py            # 다섯 장 전부
  python viz_meeting_1029.py --only m3  # 한 장만
"""
import argparse
import json
from pathlib import Path

import numpy as np

import viz_v4_common as C

OUT = C.ROOT / "viz" / "meeting_1029"
CACHE = C.VAL_CACHES["ah2"]


# ───────────────────────────────────────────────── m1 — v4 를 왜 접는가

# 조사 문서 1.2 의 표를 그대로 옮긴다 (paired Δ minADE6, 음수가 개선)
EFFECTS = [
    ("data cleansing",            -0.043, "3 seed 부호 일치"),
    ("smoothing (ah3)",           -0.053, "검출 불가"),
    ("route history",             -0.040, "3판 중 1판 발산"),
    ("neighborhood 30 m",         -0.028, "검출 불가"),
    ("coordinate penalty",         0.000, "대가 없음"),
    ("lane heading align 3.0",    +0.017, "차선변경에서만 −0.069"),
    ("downsampling 2 Hz",         +0.072, "3 seed 모두 나쁨"),
]
BASE_SEEDS = [1.3586, 1.3562, 1.3625]                               # 좌표 벌점만 · 30 epoch
BEST_SEEDS = [1.3930, 1.4227, 1.4133, 1.4065, 1.3834]               # 근거 넷 전부 · 100 epoch


def m1(plt):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    names = [e[0] for e in EFFECTS][::-1]
    vals = np.array([e[1] for e in EFFECTS])[::-1]
    notes = [e[2] for e in EFFECTS][::-1]
    cols = [C.C_GREEN if v < -0.001 else (C.C_RED if v > 0.001 else C.MUTED) for v in vals]
    ax.barh(range(len(vals)), vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(range(len(vals)), names)
    ax.axvline(0, color=C.INK2, lw=1.0, zorder=4)
    for i, (v, nt) in enumerate(zip(vals, notes)):
        side = 0.004 if v >= 0 else -0.004
        ax.text(v + side, i, f"{v:+.3f}  ·  {nt}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.2, color=C.INK2)
    ax.set_xlim(-0.115, 0.135)
    ax.set_xlabel("paired Δ minADE6  [m]   ← 좋아짐      나빠짐 →")
    ax.set_title("① 개별 효과는 0.005~0.08 규모이고 부호가 갈린다", loc="left")
    ax.grid(axis="y", visible=False)

    rng = np.random.default_rng(0)
    for k, (seeds, lab, col) in enumerate([(BASE_SEEDS, "기준\n좌표 벌점만 · 30 epoch", C.C_BLUE),
                                           (BEST_SEEDS, "근거 넷 전부\ncleansing+평활+lane-yaw+주변차량 · 100 epoch",
                                            C.C_RED)]):
        xs = k + rng.uniform(-0.055, 0.055, len(seeds))
        bx.scatter(xs, seeds, s=46, color=col, zorder=5, lw=0)
        bx.plot([k - 0.22, k + 0.22], [np.mean(seeds)] * 2, color=col, lw=2.4, zorder=4)
        bx.text(k + 0.27, np.mean(seeds), f"{np.mean(seeds):.4f}", ha="left", va="center",
                fontsize=10, color=col, fontweight="bold")
        bx.text(k, 1.295, lab, ha="center", va="top", fontsize=8.4, color=C.INK2)
    bx.annotate("", xy=(1, np.mean(BEST_SEEDS) - 0.012), xytext=(0, np.mean(BASE_SEEDS) - 0.012),
                arrowprops=dict(arrowstyle="-|>", color=C.INK2, lw=1.2,
                                connectionstyle="arc3,rad=0.3"))
    bx.text(0.5, 1.337, "+0.045", ha="center", fontsize=10.5, color=C.INK, fontweight="bold")
    bx.axhline(1.079, color=C.C_ORANGE, lw=1.3, ls="--", zorder=2)
    bx.text(-0.45, 1.082, "후보 경로의 오라클 1.079 — 구조적 상한", ha="left", va="bottom",
            fontsize=8.2, color=C.C_ORANGE)
    bx.axhline(1.0, color=C.C_AQUA, lw=1.3, ls=":", zorder=2)
    bx.text(-0.45, 1.003, "목표 minADE6 ≤ 1.0", ha="left", va="bottom", fontsize=8.2, color=C.C_AQUA)
    bx.set_xlim(-0.5, 1.6)
    bx.set_ylim(0.96, 1.46)
    bx.set_xticks([])
    bx.set_ylabel("val minADE6  [m]")
    bx.set_title("② 합치면 더해지지 않는다 — seed 전부 기준보다 나쁨", loc="left")
    bx.grid(axis="x", visible=False)

    fig.suptitle("v4 를 잠정 보류하는 이유 — 어떤 요인이 작용하는지 분리되지 않는다",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, OUT / "m1_v4_why.png")


# ───────────────────────────────────────── m2 — invariance 가 θ 신규성을 가른다

#  (이름, 교체 난이도 1최易~5재구현, θ 신규성 1완전중복~5완전신규, 그룹, 단일GPU, 역할)
MODELS = [
    ("QCNet",        2.0, 1.0, "A", False, "θ 대조군 (참고만)"),
    ("SIMPL",        1.0, 2.0, "A", True,  "(a,h) 가설 주 무대"),
    ("HPTR",         2.7, 1.6, "A", False, "제외 (CC BY-NC · 5일+)"),
    ("Forecast-MAE", 2.2, 4.3, "B", True,  "θ 가설 처치군"),
    ("DeMo",         1.0, 3.1, "B", True,  "정확도 최고 0.61"),
    ("ProphNet",     4.9, 4.0, "B", False, "제외 (코드 없음)"),
    ("GANet",        5.0, 4.9, "B", False, "제외 (빈 저장소)"),
]


def m2(plt):
    from matplotlib.lines import Line2D
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                 gridspec_kw={"width_ratios": [1.05, 1]})

    # 왼쪽 — 두 그룹이 invariance 를 얻는 방식 (개념도)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.grid(False)
    box = dict(boxstyle="round,pad=0.45", lw=1.1)
    ax.text(5, 9.4, "불변성(invariance)을 어디서 얻는가", ha="center", fontsize=11,
            color=C.INK, fontweight="bold")

    ax.text(2.5, 7.9, "Group A  구조적 불변\nQCNet · SIMPL · HPTR", ha="center", va="center",
            fontsize=9.4, color=C.C_BLUE,
            bbox=dict(facecolor="#eaf1fb", edgecolor=C.C_BLUE, **box))
    ax.text(2.5, 5.3, "기하가 쌍별 상대 descriptor 로만\n들어간다 — 절대량은 MLP 에\n한 번도 닿지 않는다",
            ha="center", va="center", fontsize=8.6, color=C.INK2)
    ax.text(2.5, 2.9, "θ = h − k 가 이미\nattention 입력에 있다\n→ 대체로 중복", ha="center", va="center",
            fontsize=9.2, color=C.INK,
            bbox=dict(facecolor="#f2f1ec", edgecolor=C.AXIS, **box))
    ax.annotate("", xy=(2.5, 3.9), xytext=(2.5, 6.6),
                arrowprops=dict(arrowstyle="-|>", color=C.C_BLUE, lw=1.4))

    ax.text(7.5, 7.9, "Group B  정규화에 의한 불변\nForecast-MAE · DeMo · ProphNet · GANet",
            ha="center", va="center", fontsize=9.4, color=C.C_ORANGE,
            bbox=dict(facecolor="#fdeee7", edgecolor=C.C_ORANGE, **box))
    ax.text(7.5, 5.3, "장면 전체를 focal agent 기준으로\n한 번 회전시킨다 — 상대 변위는\nencode 하지만 상대 각도는 아니다",
            ha="center", va="center", fontsize=8.6, color=C.INK2)
    ax.text(7.5, 2.9, "θ = h − k 가\n진짜 새 정보다\n→ 효과가 있다면 여기", ha="center", va="center",
            fontsize=9.2, color=C.INK,
            bbox=dict(facecolor="#f2f1ec", edgecolor=C.AXIS, **box))
    ax.annotate("", xy=(7.5, 3.9), xytext=(7.5, 6.6),
                arrowprops=dict(arrowstyle="-|>", color=C.C_ORANGE, lw=1.4))
    ax.text(5, 1.0, "한 모델에서만 재면 “θ 가 안 듣는다” 와 “그 구조엔 이미 있다” 를 구별할 수 없다",
            ha="center", fontsize=9.0, color=C.INK, style="italic")

    # 오른쪽 — 역상관 산점도 (라벨 방향은 겹치지 않게 모델마다 손으로 정한다)
    LAB = {"QCNet": (14, 6, "left", "center"), "SIMPL": (16, -2, "left", "center"),
           "HPTR": (16, 2, "left", "center"), "Forecast-MAE": (0, -16, "center", "top"),
           "DeMo": (16, 2, "left", "center"), "ProphNet": (-14, 0, "right", "center"),
           "GANet": (-14, 0, "right", "center")}
    for name, dx, dy, grp, gpu, role in MODELS:
        col = C.C_BLUE if grp == "A" else C.C_ORANGE
        bx.scatter(dx, dy, s=270 if gpu else 160, color=col, alpha=0.9 if gpu else 0.32,
                   lw=1.6 if gpu else 0.0, edgecolor=C.INK if gpu else "none", zorder=5)
        ox, oy, ha, va = LAB[name]
        bx.annotate(f"{name}\n{role}", (dx, dy), xytext=(ox, oy), textcoords="offset points",
                    ha=ha, va=va, fontsize=8.2, color=C.INK if gpu else C.MUTED, zorder=6)
    bx.plot([0.75, 5.3], [1.25, 5.1], color=C.MUTED, lw=1.2, ls="--", zorder=2)
    bx.text(3.55, 3.15, "교체가 쉬울수록\nθ 가 덜 보탠다", fontsize=9.4, color=C.INK2,
            rotation=29, ha="center", va="center", style="italic", zorder=3)
    bx.set_xlim(0.4, 6.1); bx.set_ylim(0.55, 5.75)
    bx.set_xticks([1, 2, 3, 4, 5], ["최易\n(2줄)", "쉬움", "중간", "어려움", "재구현"])
    bx.set_yticks([1, 2, 3, 4, 5], ["완전\n중복", "현재 step\n중복", "부분\n중복", "거의\n신규", "완전\n신규"])
    bx.set_xlabel("(a, h) 입력 교체 난이도")
    bx.set_ylabel("θ = h − k 의 신규성", labelpad=12)
    bx.set_title("두 축이 역상관 — 그래서 역할을 나눠야 한다", loc="left")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=C.C_BLUE, label="Group A 구조적 불변"),
                       Line2D([], [], marker="o", ls="", ms=9, color=C.C_ORANGE, label="Group B 정규화"),
                       Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=C.INK,
                              color="none", label="단일 24 GB GPU 로 학습 가능")],
              loc="lower right", fontsize=8.2)

    fig.suptitle("조사의 핵심 — invariance 방식이 (a, h)·θ 의 값어치를 전부 가른다",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, OUT / "m2_invariance.png")


# ─────────────────────────────── m3 — θ = h − k 가 실제 장면에서 무엇인가

def _centerline_cloud(amap, origin, theta, step=1.0):
    """지도의 모든 차로 중심선을 정규화 좌표로 촘촘히 깔고 점마다 접선각을 붙인다."""
    from dataset_map import _rotation_matrix
    R = _rotation_matrix(-float(theta)).astype(np.float64)
    pts, tans = [], []
    for seg in amap.get_scenario_lane_segments():
        cl = amap.get_lane_segment_centerline(seg.id)[:, :2].astype(np.float64)
        cl = (cl - origin) @ R.T
        d = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] < 1e-6:
            continue
        q = np.arange(0.0, s[-1], step)
        p = np.stack([np.interp(q, s, cl[:, 0]), np.interp(q, s, cl[:, 1])], axis=1)
        t = np.gradient(p, axis=0)
        pts.append(p)
        tans.append(np.arctan2(t[:, 1], t[:, 0]))
    return np.concatenate(pts), np.concatenate(tans)


def _nearest_lane_angle(pos, head, pts, tans, gate_deg=90.0):
    """각 궤적점에서 진행방향이 맞는 가장 가까운 중심선 점의 접선각 k 와 그 점."""
    k = np.empty(len(pos)); idx = np.empty(len(pos), np.int64)
    gate = np.deg2rad(gate_deg)
    for i, (p, h) in enumerate(zip(pos, head)):
        ok = np.abs(C.wrap(tans - h)) < gate                  # 대향 차로를 고르지 않게 막는다
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            cand = np.arange(len(pts))
        j = cand[np.argmin(np.sum((pts[cand] - p) ** 2, axis=1))]
        k[i], idx[i] = tans[j], j
    return k, idx


def _load_focal(sid, origin, theta):
    from av2.datasets.motion_forecasting import scenario_serialization as ss
    from dataset_map import _rotation_matrix
    scn = ss.load_argoverse_scenario_parquet(C.VAL_DIR / sid / f"scenario_{sid}.parquet")
    tr = next(t for t in scn.tracks if t.track_id == scn.focal_track_id)
    st = sorted(tr.object_states, key=lambda s: s.timestep)
    R = _rotation_matrix(-float(theta)).astype(np.float64)
    pos = (np.array([s.position for s in st], np.float64) - origin) @ R.T
    head = np.array([s.heading for s in st], np.float64) - float(theta)
    vel = np.array([s.velocity for s in st], np.float64) @ R.T
    return np.array([s.timestep for s in st]), pos, head, vel


def _pick_turning(n_scan=4000, lo=55.0, hi=100.0):
    """우회전처럼 '차로를 따라 크게 도는' 판을 고른다 — 미래 변위각의 순변화로 판정."""
    y = np.load(CACHE / "y.npy", mmap_mode="r")
    sids = json.loads((CACHE / "scenario_id.json").read_text())
    best = []
    for i in range(min(n_scan, len(sids))):
        p = np.asarray(y[i], np.float64)
        d = np.diff(p, axis=0)
        sp = np.linalg.norm(d, axis=1)
        if sp.min() < 0.25:                                   # 멈추는 판은 각이 불안정하다
            continue
        a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        net = np.rad2deg(a[-1] - a[0])
        if lo < abs(net) < hi and np.linalg.norm(p[-1]) > 40:
            best.append((abs(net), i, sids[i]))
    best.sort()
    return best[len(best) // 2]                               # 중앙값 근처 — 전형적인 판


def m3(plt, sid=None):
    from av2.map.map_api import ArgoverseStaticMap
    import matplotlib.gridspec as gs

    sids = json.loads((CACHE / "scenario_id.json").read_text())
    if sid is None:
        _, i, sid = _pick_turning()
    else:
        i = sids.index(sid)
    origin = np.asarray(np.load(CACHE / "origin.npy", mmap_mode="r")[i], np.float64)
    theta = float(np.load(CACHE / "theta.npy", mmap_mode="r")[i])
    ts, pos, head, vel = _load_focal(sid, origin, theta)
    amap = ArgoverseStaticMap.from_json(C.VAL_DIR / sid / f"log_map_archive_{sid}.json")
    pts, tans = _centerline_cloud(amap, origin, theta)
    k, kidx = _nearest_lane_angle(pos, head, pts, tans)
    th = C.wrap(head - k)                                     # θ = h − k
    spd = np.linalg.norm(vel, axis=1)
    acc = np.gradient(spd, 0.1)                               # a = d|v|/dt  (AV2 는 가속도를 주지 않는다)
    t = (ts - C.OBS + 1) * 0.1                                # 관측 끝 = 0 s

    fig = plt.figure(figsize=(13.2, 5.4))
    g = gs.GridSpec(3, 2, width_ratios=[1.3, 1], height_ratios=[1, 1, 1], figure=fig,
                    hspace=0.32, wspace=0.14)
    ax = fig.add_subplot(g[:, 0])
    scene = C.build_scene(sid, origin, theta)
    pad = 12
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    half = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])) / 2 + pad
    ar = 1.22                                                 # 축 상자가 세로로 길다 — 그 비율에 맞춘다
    xlim, ylim = (cx - half / ar, cx + half / ar), (cy - half, cy + half)
    C.draw_scene_light(ax, scene, xlim, ylim)

    obs = ts < C.OBS
    ax.plot(pts[np.unique(kidx)][:, 0], pts[np.unique(kidx)][:, 1], ".", ms=2.0,
            color=C.C_AQUA, alpha=0.5, zorder=5, label="k 를 읽은 차로 중심선")
    ax.plot(pos[obs, 0], pos[obs, 1], "-", color=C.C_PAST, lw=2.6, zorder=6, label="과거 5 s")
    ax.plot(pos[~obs, 0], pos[~obs, 1], "-", color=C.C_GT, lw=2.6, zorder=7, label="정답 6 s")
    ax.plot(pos[::10, 0], pos[::10, 1], "o", ms=3.2, mfc="white", mec=C.INK, mew=0.9, zorder=8)
    for j in range(0, len(pos), 15):                          # θ 를 눈으로 읽는 곳
        L = 5.5
        ax.annotate("", xy=(pos[j, 0] + L * np.cos(k[j]), pos[j, 1] + L * np.sin(k[j])),
                    xytext=(pos[j, 0], pos[j, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=C.C_AQUA, lw=1.5), zorder=9)
        ax.annotate("", xy=(pos[j, 0] + L * np.cos(head[j]), pos[j, 1] + L * np.sin(head[j])),
                    xytext=(pos[j, 0], pos[j, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=C.C_ORANGE, lw=1.5), zorder=9)
    C.draw_box(ax, pos[C.OBS - 1, 0], pos[C.OBS - 1, 1], head[C.OBS - 1], C.C_BLUE, zorder=10)
    C.scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"{sid[:8]} · 회전 — 주황 h(차량 진행방향) · 청록 k(최근접 차로 접선)", loc="left",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8.4, facecolor="white", framealpha=0.85, frameon=True)

    def panel(row, y, color, label, unit, extra=None):
        p = fig.add_subplot(g[row, 1])
        p.axvspan(t[0], 0, color=C.GRID, alpha=0.55, lw=0, zorder=1)
        p.plot(t, y, color=color, lw=1.9, zorder=4)
        if extra is not None:
            p.plot(t, extra[0], color=extra[1], lw=1.5, ls="--", zorder=3)
        p.axvline(0, color=C.INK2, lw=0.9, ls=":", zorder=5)
        p.set_ylabel(f"{label}  [{unit}]", fontsize=8.8)
        p.tick_params(labelsize=8)
        if row < 2:
            p.set_xticklabels([])
        else:
            p.set_xlabel("시간  [s]   (0 = 관측 끝)", fontsize=8.8)
        return p

    thd = np.rad2deg(th)
    h_net = np.rad2deg(np.unwrap(head)[-1] - np.unwrap(head)[0])
    med, mx = np.median(np.abs(thd)), np.abs(thd).max()
    jump = np.abs(np.diff(np.rad2deg(np.unwrap(k)))).max()    # k 가 한 step 에 튀는 최대 폭

    p1 = panel(0, np.rad2deg(np.unwrap(head)), C.C_ORANGE, "h, k", "°",
               extra=(np.rad2deg(np.unwrap(k)), C.C_AQUA))
    p1.set_title(f"차량은 {abs(h_net):.0f}° 를 돈다 — k(청록)가 그 몫을 거의 그대로 갖고 있다",
                 loc="left", fontsize=9.4)
    p2 = panel(1, thd, C.C_VIOLET, "θ = h − k", "°")
    p2.axhline(0, color=C.MUTED, lw=0.9)
    jt = t[1:][np.argmax(np.abs(np.diff(np.rad2deg(np.unwrap(k)))))]
    p2.annotate(f"k 가 {jump:.0f}°/step 튄 자리", xy=(jt, thd[np.argmin(thd)]),
                xytext=(jt + 1.6, thd.min() * 0.95), fontsize=8.0, color=C.C_RED, va="center",
                arrowprops=dict(arrowstyle="-|>", color=C.C_RED, lw=1.0))
    p2.text(t[2], thd.max(), f"|θ| 중앙 {med:.1f}° · 최대 {mx:.0f}°", fontsize=8.2,
            color=C.INK2, va="top")
    panel(2, acc, C.C_GREEN, "a = d|v|/dt", "m/s²")

    fig.suptitle(f"θ = h − k 는 “차로가 꺾이는 몫”을 빼고 남는 각 — {abs(h_net):.0f}° 회전이 중앙 {med:.1f}° 로 접힌다"
                 "  ·  단 최근접 차로 규칙은 교차로에서 튄다",
                 x=0.012, ha="left", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = C.savefig(fig, OUT / "m3_theta.png")
    (OUT / "m3_theta.json").write_text(json.dumps(
        {"sid": sid, "idx": int(i), "theta_abs_mean_deg": float(np.rad2deg(np.abs(th)).mean()),
         "theta_abs_max_deg": float(np.rad2deg(np.abs(th)).max()),
         "theta_abs_median_deg": float(med), "k_max_jump_deg_per_step": float(jump),
         "head_net_deg": float(h_net),
         "k_net_deg": float(np.rad2deg(np.unwrap(k)[-1] - np.unwrap(k)[0]))}, indent=1))
    return path


# ───────────────────────────────────── m4 — 5 Hz 질문: 길이는 답이 있고 레이트는 비어 있다

LEN_ABL = {                                                   # arXiv:2603.10597, AV2 val, minADE6
    "QCNet-Ori": [0.900, 0.777, 0.752, 0.725, 0.726],
    "QCNet-IT":  [0.741, 0.734, 0.730, 0.726, 0.726],
    "DeMo-Ori":  [0.861, 0.700, 0.671, 0.662, 0.658],
    "DeMo-IT":   [0.675, 0.661, 0.660, 0.659, 0.658],
}


def m4(plt):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.6),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    sec = np.array([1, 2, 3, 4, 5])
    for name, ys in LEN_ABL.items():
        col = C.C_BLUE if name.startswith("QCNet") else C.C_GREEN
        ls = "-" if name.endswith("IT") else (0, (5, 2.5))
        ax.plot(sec, ys, color=col, linestyle=ls, marker="o", ms=4.5, lw=1.9, label=name)
    ax.annotate("4 s ≈ 5 s\n(0.725 vs 0.726)", xy=(4, 0.725), xytext=(3.0, 0.80),
                fontsize=9, color=C.INK,
                arrowprops=dict(arrowstyle="-|>", color=C.INK2, lw=1.1))
    ax.set_xticks(sec)
    ax.set_xlabel("관측 history 길이  [s]   (AV2 는 5 s 제공)")
    ax.set_ylabel("AV2 val minADE6  [m]")
    ax.set_title("길이 ablation 은 답이 뚜렷하다 — 뒤쪽 1 s 는 값이 없다", loc="left")
    ax.legend(ncol=2, loc="upper right")
    ax.text(1.0, 0.645, "-Ori / -IT 의 정의는 원문에서 확인하지 못했다 — 표에서 추정", fontsize=7.8,
            color=C.MUTED, va="bottom")

    bx.axis("off"); bx.grid(False)
    bx.set_xlim(0, 10); bx.set_ylim(0, 10)
    bx.text(0.2, 9.5, "그런데 레이트를 고정 길이에서 바꾼 통제 ablation 은", fontsize=10.2,
            color=C.INK, va="top")
    bx.text(0.2, 8.6, "공개 문헌에 없다  (독립 검색 4회 · 같은 결론)", fontsize=10.2,
            color=C.C_RED, va="top", fontweight="bold")
    rows = [("반대 증거", C.C_RED, [
        "SHARP (AV2 val): 5 s context 0.64 → 1 s 0.76  (+18.8 %)",
        "POP: “성능은 관측 길이와 직접 상관” — 특히 긴 지평",
        "FlexiLength (CVPR'24): 학습과 다른 길이면 상당한 저하",
        "우리 실측: 10 Hz → 2 Hz 는 3 seed 모두 나쁨  (+0.072)"]),
            ("우호 증거", C.C_GREEN, [
        "QCNet 공식 AV2 명령이 `--num_t2m_steps 30` — 앞 2 s 를 마스킹",
        "SIMPL 이 50 step 중 앞 2 개를 버림 (`# ! tmp solution`)",
        "E2E driving 연구(arXiv:2605.10388): 작은 모델은 중간 주파수 최고",
        "  └ 기전은 과적합이 아니라 capacity 라고 말한다"])]
    y = 7.5
    for title, col, items in rows:
        bx.text(0.2, y, title, fontsize=9.6, color=col, fontweight="bold", va="top")
        y -= 0.75
        for s in items:
            bx.text(0.6, y, "· " + s, fontsize=8.7, color=C.INK2, va="top")
            y -= 0.66
        y -= 0.3
    bx.text(0.2, y - 0.1, "→ 이번 실험은 각 SOTA 모델의 기본 레이트를 따르고,\n    레이트는 별도 ablation 축으로 둔다",
            fontsize=10, color=C.INK, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f1ec", edgecolor=C.AXIS, lw=1.0))

    fig.suptitle("“5 Hz 다운샘플링이 과적합을 막아 주는가” — 문헌이 답을 갖고 있지 않다",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, OUT / "m4_rate.png")


# ───────────────────────────────────────── m5 — 파이프라인 검증과 실험 단계

STAGES = [("0", "배포 ckpt 평가", 0.05, "파이프라인 검증", C.C_GREEN, "끝"),
          ("1", "R0 원본 입력", 45, "우리 루프가 0.78 근처를 내는가", C.C_BLUE, ""),
          ("2", "R1 (a, h, v0) 입력", 45, "R1 − R0 이 핵심 수치", C.C_ORANGE, ""),
          ("4", "R2 출력까지 (a, Δθ)", 45, "출력 교체의 대가", C.C_VIOLET, ""),
          ("5", "R3 Forecast-MAE + θ", 45, "θ 가설 (처치군)", C.C_MAGENTA, "")]


def m5(plt):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1.25]})

    vals = [("논문 보고\n(체크포인트 분실)", 0.720, C.MUTED),
            ("저자 재학습 ckpt\nREADME 보고", 0.777, C.C_BLUE),
            ("우리 환경에서\n같은 ckpt 재평가", 0.777, C.C_GREEN)]
    ax.bar(range(3), [v for _, v, _ in vals], color=[c for _, _, c in vals], width=0.58, zorder=3)
    for j, (lab, v, c) in enumerate(vals):
        ax.text(j, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, color=c, fontweight="bold")
    ax.set_xticks(range(3), [lab for lab, _, _ in vals], fontsize=8.6)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("AV2 val minADE6  [m]")
    ax.annotate("", xy=(2, 0.845), xytext=(1, 0.845),
                arrowprops=dict(arrowstyle="<|-|>", color=C.C_GREEN, lw=1.3))
    ax.text(1.5, 0.862, "완전 일치 — 전처리·평가 경로 검증됨", ha="center", fontsize=9,
            color=C.C_GREEN, fontweight="bold")
    ax.set_title("① 24,988 판 전수 평가 175 s · 소수 셋째 자리까지 일치", loc="left")
    ax.grid(axis="x", visible=False)

    bx.grid(False)
    y0 = 0.0
    for num, name, hours, why, col, done in STAGES:
        bx.barh(y0, hours, height=0.52, color=col, alpha=0.92, zorder=3)
        bx.text(-1.2, y0, f"{num}", ha="right", va="center", fontsize=10.5, color=col,
                fontweight="bold")
        bx.text(0.8, y0 + 0.37, name, ha="left", va="bottom", fontsize=9.4, color=C.INK)
        tag = f"{hours:g} h" if hours > 1 else "3 분"
        bx.text(hours + 1.0, y0, f"{tag}  ·  {why}" + ("   — 끝" if done else ""),
                ha="left", va="center", fontsize=8.8, color=C.INK2)
        y0 -= 1.0
    bx.text(0.8, y0 + 0.45, "3 · 차이가 작으면 그때 seed 를 붙인다 (조건부)", fontsize=8.8,
            color=C.MUTED, style="italic")
    bx.set_xlim(-2, 86)
    bx.set_ylim(y0 - 0.2, 0.9)
    bx.set_yticks([])
    bx.set_xlabel("단일 RTX PRO 4000 Blackwell 24 GB 기준 소요  [시간]")
    bx.set_title("② 단계적으로 간다 — 2조건 × 3 seed 를 미리 돌리면 270 h", loc="left")
    bx.text(84, 0.62, "1 epoch 54.6 분 · GPU 사용률 중앙 88 % · 로더 대기 0.2 %\n"
                          "backward 가 65 % — 연산 바운드라 45 h 는 바닥에 가깝다",
            ha="right", va="bottom", fontsize=8.4, color=C.INK2,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f1ec", edgecolor=C.AXIS, lw=1.0))
    bx.set_ylim(y0 - 0.2, 2.1)

    fig.suptitle("1차 무대는 SIMPL — 파이프라인은 이미 검증됐다",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return C.savefig(fig, OUT / "m5_plan.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="m1|m2|m3|m4|m5")
    ap.add_argument("--sid", default=None, help="m3 에 쓸 시나리오를 직접 지정")
    a = ap.parse_args()
    plt = C.setup_mpl()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [a.only] if a.only else ["m1", "m2", "m3", "m4", "m5"]
    for name in todo:
        fn = globals()[name]
        p = fn(plt, a.sid) if name == "m3" else fn(plt)
        print(f"{name} → {p}")


if __name__ == "__main__":
    main()
