"""
viz_v4_penalty_example.py - 흔들림 벌점 세 판이 같은 시나리오에서 어떻게 다르게 움직이는지 본다.

왜: 24판 표(벌점 없음 8.47% / 좌표 1.0 2.28% / 액션 1.0 0.76%, minADE6 +0.000 vs +0.081)는
    "무엇이 얼마나" 는 말해 주지만 "어떻게" 는 안 보인다. 같은 시나리오를 세 판으로 나란히 그리고,
    같은 코드로 구간별 집계를 함께 낸다.

세 판 (모두 ah2 입력 · 30에폭 코사인 · 시드 0 · 같은 val 캐시라 시나리오 순서가 같다)
    벌점 없음   v4_l4nw_ah2_full_sm0_cos30_s0
    액션 벌점   v4_l4nw_ah2_full_sm1_cos30_s0     모델이 낸 Δθ 의 스텝 변화에 hinge  (옛 기본)
    좌표 벌점   v4_l4nw_ah2_full_smxy1_cos30_s0   예측 좌표에서 복원한 ψ 의 스텝 변화에 hinge (현재 기본)

그림 (viz/v4/penalty/)
    penalty_mech.png   직진 예시 — 벌점이 흔들림을 어떻게 없애나 (|Δψ| · |Δθ| 시계열)
    penalty_turn.png   회전 예시 — 세 판이 회전을 어떻게 다르게 그리나 (누적 진행방향 · |Δψ|)
    penalty_bins.png   집계 — 정답 회전량 구간별 |Δψ| 중앙값·p99·위반율·minADE6
    penalty_box.png    같은 구간을 상자로 — 분포 모양(중앙·사분위·꼬리)

측정 정의는 src/score_xy_viol.py 와 같다(움직인 스텝만, 임계 7.3°/step). 시나리오는 규칙으로 고르고
규칙을 그림 안에 적는다(체리피킹 방지). 고른 결과·집계값은 penalty_example_picks.json 에 남긴다.

    python src/viz_v4_penalty_example.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C
import viz_v4_cases as CS

TAGS = [("벌점 없음", "v4_l4nw_ah2_full_sm0_cos30_s0", "#e34948", "o"),
        ("액션 벌점 1.0 (옛 기본)", "v4_l4nw_ah2_full_sm1_cos30_s0", C.C_BLUE, "s"),
        ("좌표 벌점 1.0 (현재 기본)", "v4_l4nw_ah2_full_smxy1_cos30_s0", C.C_GREEN, "^")]
OUT = C.VIZ_ROOT / "penalty"
MOVE_V, DT, LIM = C.MOVE_V, C.DT, C.LABEL_DTHETA_DEG
T_D = np.arange(2, C.FUT) * DT          # |Δψ| 는 스텝 3개를 써서 나오니 t = 0.2 s 부터
BINS = [(0.0, 5.0, "직진 <5°"), (5.0, 30.0, "완만 5–30°"), (30.0, 1e9, "회전 ≥30°")]


def dpsi_deg(traj):
    """(...,T,2) -> (|Δψ| [°/step], 양쪽 스텝이 움직였나). score_xy_viol.dpsi_deg 와 같은 정의."""
    d = np.diff(traj, axis=-2)
    sp = np.linalg.norm(d, axis=-1) / DT
    psi = np.arctan2(d[..., 1], d[..., 0])
    dp = (np.diff(psi, axis=-1) + np.pi) % (2 * np.pi) - np.pi
    ok = (sp[..., 1:] >= MOVE_V) & (sp[..., :-1] >= MOVE_V)
    return np.abs(np.degrees(dp)), ok


def psi_cum_deg(traj):
    """(T,2) -> 누적 진행방향 변화 [°] (움직인 스텝만 더한다). |Δψ| 와 같은 시간축."""
    d = np.diff(traj, axis=0)
    sp = np.linalg.norm(d, axis=-1) / DT
    psi = np.arctan2(d[:, 1], d[:, 0])
    dp = (np.diff(psi) + np.pi) % (2 * np.pi) - np.pi
    ok = (sp[1:] >= MOVE_V) & (sp[:-1] >= MOVE_V)
    return np.degrees(np.cumsum(dp * ok))


def turn_deg(traj):
    """총 회전량 [°] — score_xy_viol.gt_turn_deg 와 같은 정의(움직인 스텝의 부호합 절댓값)."""
    d = np.diff(traj, axis=-2)
    sp = np.linalg.norm(d, axis=-1) / DT
    psi = np.arctan2(d[..., 1], d[..., 0])
    dp = (np.diff(psi, axis=-1) + np.pi) % (2 * np.pi) - np.pi
    ok = (sp[..., 1:] >= MOVE_V) & (sp[..., :-1] >= MOVE_V)
    return np.abs(np.degrees((dp * ok).sum(-1)))


def med(x, m):
    return float(np.median(x[m])) if m.any() else np.nan


class Pair:
    """판 하나 — 그림용 Ctx + 승자 모드 파생량."""

    def __init__(self, name, tag, color, marker):
        self.name, self.tag, self.color, self.marker = name, tag, color, marker
        self.cx = CS.Ctx(tag)
        df = self.cx.df
        i = np.arange(len(df))
        w = df["winner"].to_numpy().astype(int)
        self.win = self.cx.P["traj"][i, w]                            # (N,60,2) 승자 = 끝점 오차 최소 모드
        self.dth = np.degrees(np.abs(self.cx.pred["dtheta"][i, w]))   # 모델 액션 |Δθ| [°/step]
        self.minade = df["minade"].to_numpy()
        self.win_route = df["win_route"].to_numpy()
        self.dp, self.ok = dpsi_deg(self.win)                         # (N,58)
        self.exc = ((self.dp > LIM) & self.ok).sum(1)
        self.steps = self.ok.sum(1)
        self.turn = turn_deg(self.win)
        with np.errstate(invalid="ignore"):
            self.dp_med = np.array([med(self.dp[k], self.ok[k]) for k in range(len(df))])

    def viol_pct(self, m=None):
        m = np.ones(len(self.exc), bool) if m is None else m
        return 100.0 * self.exc[m].sum() / max(self.steps[m].sum(), 1)

    def dp_pool(self, m):
        return self.dp[m][self.ok[m]]


def ts_panel(ax, xs, series, ylab, lim=None, lim_lab=None, ncol=2):
    """시계열 한 칸 — 선 + 스텝마다 점(필수 요건 7)."""
    for y, color, marker, lab, lw in series:
        ax.plot(xs, y, color=color, lw=lw, marker=marker, ms=3.0, mec=C.SURF, mew=0.3, label=lab, zorder=3)
    if lim is not None:
        ax.axhline(lim, color=C.C_ORANGE, lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.text(xs[-1], lim, f"{lim_lab} ", color=C.C_ORANGE, fontsize=7.6, va="bottom", ha="right")
    ax.set_xlabel("예측 시각 [s]")
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 6.05)
    ax.legend(loc="upper left", fontsize=8.0, ncol=ncol)


def note(ax, lines, fs=9.0):
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=fs,
            color=C.INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=C.GRID, lw=0.8))


def maps_row(fig, axes, P, i, extra):
    for ax, p in zip(axes, P):
        CS.draw_map(ax, p.cx, i)
        ax.set_title(f"{p.name}\nminADE6 {p.minade[i]:.2f} m · 승자 모드 위반 {p.exc[i]}/{p.steps[i]} step"
                     + extra(p), fontsize=10.2)
    h, lab = CS.legend_items(P[0].cx.hz)
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.015, 0.948), ncol=6, fontsize=8.0, frameon=True,
               facecolor="white", edgecolor=C.GRID, handlelength=2.2, columnspacing=1.3)


def series_dpsi(P, G, i):
    s = [(np.where(p.ok[i], p.dp[i], np.nan), p.color, p.marker, p.name, 1.5) for p in P]
    g, gok = dpsi_deg(G["pos"][i])
    s.append((np.where(gok, g, np.nan), C.C_GT, "o", "정답", 2.0))
    return s


def fig_mech(P, G, out):
    """직진 예시 — 벌점이 흔들림을 없애는 모습."""
    import matplotlib.pyplot as plt
    A, B, D = P                                     # 벌점 없음 · 액션 · 좌표
    df = A.cx.df
    same = np.all([p.win_route == df["gt_route"] for p in P], axis=0)
    near = np.all([np.abs(p.minade - D.minade) < 0.3 for p in (A, B)], axis=0)
    ok = (G["turn"] < 5.0) & (df["v0_f"].to_numpy() >= 6.0) & same & near & (A.steps > 50) & (D.steps > 50)
    i = int(np.argmax(np.where(ok, A.exc - D.exc, -1)))
    r = df.iloc[i]

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 11.4))
    maps_row(fig, axes[0], P, i, lambda p: "")
    ts_panel(axes[1][0], T_D, series_dpsi(P, G, i),
             "|Δψ| — 예측 좌표에서 복원한 진행방향의 스텝 변화 [°/step]", LIM, f"라벨 임계 {LIM}°/step")
    ts_panel(axes[1][1], T_D, [(p.dth[i][2:], p.color, p.marker, p.name, 1.5) for p in P],
             "|Δθ| — 모델이 낸 액션의 스텝 변화 [°/step]", LIM, f"벌점 임계 {LIM}°/step")
    note(axes[1][2], [
        "읽는 방법",
        "위 세 지도는 같은 시나리오를 세 판으로 그린 것이다. 회색 = 과거 5초, 검정 = 정답 6초,",
        "파란 선 6개 = 예측 6모드(짙고 굵을수록 확률 높음), ★ = 승자(끝점 오차가 가장 작은 모드).",
        "",
        f"왼쪽 아래 |Δψ| = '예측 좌표가 한 스텝(0.1초)에 몇 도 꺾였나'. {LIM}° 를 넘으면 위반으로 센다",
        "(AV2 정답 라벨 전수조사 p99.99 가 7.3°). 가운데 |Δθ| 는 '모델이 낸 액션' 쪽 각이다.",
        "",
        "이 시나리오에서 —",
        f"   벌점 없음  |Δψ| 가 임계를 {A.exc[i]}/{A.steps[i]} 스텝 넘는다. 좌표가 톱니처럼 꺾인다.",
        f"   액션 벌점  |Δθ| 가 바닥에 붙는다 ({B.exc[i]}/{B.steps[i]}). 벌점이 직접 누르는 변수다.",
        f"   좌표 벌점  임계 위만 잘린다 ({D.exc[i]}/{D.steps[i]}). 임계 아래 흔들림은 남는다 — hinge 라서",
        "              '넘는 분량'만 벌을 주기 때문이다.",
        "",
        "지도에서는 세 판이 거의 겹쳐 보인다. 차이는 아래 시계열에서만 드러난다 —",
        "그래서 벌점을 걸어도 minADE6 가 거의 안 변하는 것이다(좌표 1.0 은 +0.000).",
        "",
        "직진에서는 경로 곡률 k(s) ≈ 0 이라 ψ = k(s) + θ 에서 Δψ ≈ Δθ 다 — 두 벌점이 사실상",
        "같은 변수를 누른다. 차이는 회전에서 난다(다음 그림).",
        "",
        "전체 수치 (val 24,988 · 3시드 평균 · 좌표 기준 top-1 위반율)",
        "   벌점 없음 8.47% · minADE6 1.359     좌표 1.0 2.28% · 1.359 (+0.000, 공짜)",
        "   액션 1.0  0.76% · 1.440 (+0.081)     정답 라벨 0.35%",
        "",
        f"고른 규칙: 정답 회전량 < 5° · 시작 속력 ≥ 6 m/s · 세 판 모두 정답 경로를 탐 ·",
        f"   minADE6 차 0.3 m 미만인 {int(ok.sum()):,}개 중 (벌점 없음 − 좌표 벌점) 위반 step 최대",
        f"   → {r['sid'][:8]} · 상황 {r['cls']} · 시작 속력 {r['v0_f']:.1f} m/s",
    ], fs=8.6)
    fig.suptitle("벌점이 하는 일 — 같은 직진 시나리오, 세 판 (ah2 입력 · 30에폭 코사인 · 시드 0 · val 전체)",
                 fontsize=14, x=0.015, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    C.savefig(fig, out)
    return {"idx": i, "sid": r["sid"], "cls": r["cls"], "n_pool": int(ok.sum()),
            "exc": {p.name: f"{p.exc[i]}/{p.steps[i]}" for p in P},
            "minade": {p.name: round(float(p.minade[i]), 3) for p in P}}


def fig_turn(P, G, out):
    """회전 예시 — 액션 벌점 판이 회전 구간에서 정답만큼 꺾지 않는 모습."""
    import matplotlib.pyplot as plt
    A, B, D = P
    df = A.cx.df
    same = np.all([p.win_route == df["gt_route"] for p in P], axis=0)
    grp = (G["turn"] >= C.TURN_DEG) & (G["turn"] < 180.0) & same   # 180° 이상은 제자리 선회·잡음이라 뺀다
    short = B.dp_med - G["dp_med"]            # 액션 판 |Δψ| 중앙값 − 정답 (음수 = 정답보다 덜 꺾는다)
    # 경로 자체를 놓친 판(minADE6 큼)이나 회전을 통째로 안 돈 판은 '꺾는 방식' 비교에 쓸 수 없다
    tracked = np.all([(p.minade < 3.0) & (p.turn > 0.5 * np.maximum(G["turn"], 1e-6)) for p in P], axis=0)
    grp = grp & np.isfinite(short) & np.isfinite(G["dp_med"])
    typ = float(np.nanmedian(short[grp]))
    cand = np.where(grp & tracked & (short < 0))[0]
    i = int(cand[np.argmin(np.abs(short[cand] - typ))])
    r = df.iloc[i]

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 11.4))
    maps_row(fig, axes[0], P, i, lambda p: f" · 승자 회전량 {p.turn[i]:.0f}°")
    ser = [(psi_cum_deg(p.win[i]), p.color, p.marker, p.name, 1.5) for p in P]
    ser.append((psi_cum_deg(G["pos"][i]), C.C_GT, "o", "정답", 2.0))
    ts_panel(axes[1][0], T_D, ser, "누적 진행방향 변화 [°] — 0 s 부터 얼마나 돌았나")
    ts_panel(axes[1][1], T_D, series_dpsi(P, G, i), "|Δψ| [°/step]", LIM, f"라벨 임계 {LIM}°/step")
    note(axes[1][2], [
        "읽는 방법",
        "왼쪽 아래: 선이 위(아래)로 갈수록 왼쪽(오른쪽)으로 많이 돈 것이다. 검정(정답)보다 평평하면 덜 돈 것이다.",
        "가운데: 한 스텝에 몇 도씩 꺾고 있나. 회전 중에는 정답도 1°/step 안팎으로 **꾸준히** 꺾는다.",
        "",
        f"회전군 — 정답 총 회전량 ≥ {C.TURN_DEG:.0f}° · 세 판 모두 정답 경로를 탄 {int(grp.sum()):,}개",
        f"   정답 평균 회전량 {G['turn'][grp].mean():.1f}°",
        *[f"   {p.name:22s} {p.turn[grp].mean():5.1f}°  (|Δψ| 중앙값 {np.median(p.dp_pool(grp)):.2f}°/step)"
          for p in P],
        f"   정답 |Δψ| 중앙값 {np.median(G['dp'][grp][G['ok'][grp]]):.2f}°/step",
        "",
        "즉 세 판 모두 정답보다 덜 돈다. 다만 **덜 도는 방식**이 다르다 —",
        "   액션 벌점 판은 중앙값이 정답의 1/4 수준이라 회전 구간 내내 거의 안 꺾다가 몇 스텝에서 몰아 꺾는다.",
        "   좌표 벌점 판은 중앙값이 정답보다 커서 회전 중에도 잔잔한 흔들림이 남는다.",
        "",
        "이 차이가 정확도로 나타난다 (같은 시나리오 짝 차이 minADE6 · 3시드 평균 · 정답 회전량 구간별)",
        "   액션 − 벌점 없음: 직진 +0.055 · 완만 +0.093 · 회전 +0.164 m  (대가가 회전에서 가장 크다)",
        "      단 회전 구간은 시드마다 +0.086 / +0.102 / +0.304 로 흔들린다 — 방향은 일치, 크기는 폭이 넓다.",
        "   좌표 − 벌점 없음: 직진 −0.006 · 완만 −0.003 · 회전 +0.030 m  (거의 공짜)",
        "",
        f"고른 규칙: 위 회전군에서 액션 판의 |Δψ| 중앙값이 정답보다 작은 시나리오 중 그 차이가",
        f"   **군 중앙값({typ:+.2f}°)에 가장 가까운** 것 (극단값을 피하려는 것)",
        f"   → {r['sid'][:8]} · 상황 {r['cls']} · 정답 회전량 {G['turn'][i]:.0f}°",
    ], fs=8.6)
    fig.suptitle("회전에서 무엇이 다른가 — 같은 회전 시나리오, 세 판 (ah2 입력 · 30에폭 코사인 · 시드 0 · val 전체)",
                 fontsize=14, x=0.015, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    C.savefig(fig, out)
    return {"idx": i, "sid": r["sid"], "cls": r["cls"], "n_pool": int(grp.sum()),
            "gt_turn_deg": round(float(G["turn"][i]), 1),
            "turn": {p.name: round(float(p.turn[i]), 1) for p in P},
            "group_turn_mean": {"정답": round(float(G["turn"][grp].mean()), 1),
                                **{p.name: round(float(p.turn[grp].mean()), 1) for p in P}}}


def fig_bins(P, G, out):
    """집계 — 정답 회전량 구간별로 세 판을 나란히."""
    import matplotlib.pyplot as plt
    A, B, D = P
    ms = [(G["turn"] >= lo) & (G["turn"] < hi) for lo, hi, _ in BINS]
    labs = [lab for _, _, lab in BINS]
    x = np.arange(len(BINS))
    w = 0.2
    rows = {
        "|Δψ| 중앙값 [°/step]\n(정답에 가까울수록 좋다)":
            [("정답", C.C_GT, [np.median(G["dp"][m][G["ok"][m]]) for m in ms])]
            + [(p.name, p.color, [np.median(p.dp_pool(m)) for m in ms]) for p in P],
        "|Δψ| p99 [°/step]\n(꼬리 — 튀는 정도)":
            [("정답", C.C_GT, [np.percentile(G["dp"][m][G["ok"][m]], 99) for m in ms])]
            + [(p.name, p.color, [np.percentile(p.dp_pool(m), 99) for m in ms]) for p in P],
        "좌표 기준 위반율 [%]\n(|Δψ| > 7.3°/step 인 스텝 비율)":
            [(p.name, p.color, [p.viol_pct(m) for m in ms]) for p in P],
        "minADE6 [m]\n(작을수록 정확)":
            [(p.name, p.color, [p.minade[m].mean() for m in ms]) for p in P],
    }
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.2))
    cells = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]      # 오른쪽 칸 둘은 범례·읽는 방법
    for ax, (ylab, ser) in zip(cells, rows.items()):
        n = len(ser)
        for j, (lab, color, v) in enumerate(ser):
            ax.bar(x + (j - (n - 1) / 2) * w, v, w * 0.92, color=color, label=lab,
                   edgecolor=C.SURF, linewidth=0.6)
            for xi, vi in zip(x + (j - (n - 1) / 2) * w, v):
                ax.text(xi, vi, f"{vi:.2f}" if max(v) < 20 else f"{vi:.1f}", ha="center", va="bottom",
                        fontsize=7.0, color=C.INK2)
        ax.set_xticks(x); ax.set_xticklabels(labs)
        ax.set_ylabel(ylab)
        ax.set_xlabel("정답 총 회전량 구간")
        ax.legend(fontsize=7.8, ncol=2)
        if "위반율" in ylab:
            ax.axhline(0.35, color=C.C_GT, lw=1.0, ls=(0, (4, 2)))
            ax.text(-0.45, max(0.35, 0.03 * ax.get_ylim()[1]) + 0.45, "정답 라벨 0.35%", fontsize=7.4,
                    color=C.INK2, va="bottom", ha="left")
    C.annotate_n(axes[0][0], x, [int(m.sum()) for m in ms])
    note(axes[1][2], [
        "읽는 방법",
        "가로축은 **정답이 6초 동안 실제로 돈 각도**로 나눈 세 구간이다(위 칸 괄호 = 시나리오 수).",
        "막대 네 개는 정답·세 판이며, 위 두 칸은 정답에 가까울수록 좋다.",
        "",
        "읽히는 것",
        "1. 벌점 없음 판은 모든 구간에서 |Δψ| 중앙값이 정답의 3~25배다. 톱니처럼 흔들린다.",
        "2. 액션 벌점 판은 **정답보다도 매끄럽다**(회전 구간 중앙값 0.24° vs 정답 1.10°).",
        "   회전 중에도 거의 안 꺾다가 몇 스텝에서 몰아 꺾어 p99 는 오히려 크다.",
        "3. 좌표 벌점 판은 중앙값이 정답보다 크고(2.0~2.5°) 위반율은 중간이다.",
        "   hinge 가 임계 위만 자르기 때문에 '꼬리는 눌리고 중앙은 남는' 모양이 된다.",
        "4. minADE6 는 세 판 모두 회전 구간에서 2 m 대로 커진다. 회전이 가장 어려운 구간이고,",
        "   액션 벌점의 대가도 여기에서 가장 크다 — 이 그림(시드 0)에서 +0.086 m, 3시드 평균으로는 +0.164 m.",
        "",
        "정의: |Δψ| 는 예측 좌표에서 복원한 진행방향의 스텝 변화, 움직인 스텝(≥ 1 m/s)만 센다.",
        "   측정 코드는 src/score_xy_viol.py 와 같은 정의이고, 승자 모드 기준이다.",
        "   (24판 표의 2.28%·8.47% 는 top-1 모드·3시드 평균이라 숫자가 조금 다르다.)",
    ], fs=8.8)
    axes[0][2].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    axes[0][2].legend(h, l, loc="center left", fontsize=10.5, frameon=True, facecolor="white",
                      edgecolor=C.GRID, handlelength=2.4, title="막대 색", title_fontsize=10.5)
    fig.suptitle("구간별 집계 — 정답 회전량으로 나눈 세 판 (ah2 입력 · 30에폭 코사인 · 시드 0 · val 24,988)",
                 fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)
    return {lab: {"n": int(m.sum()),
                  "정답_|dpsi|_p50": round(float(np.median(G["dp"][m][G["ok"][m]])), 3),
                  **{p.name: {"minADE6": round(float(p.minade[m].mean()), 3),
                              "viol_pct": round(p.viol_pct(m), 2),
                              "dpsi_p50": round(float(np.median(p.dp_pool(m))), 2),
                              "dpsi_p99": round(float(np.percentile(p.dp_pool(m), 99)), 2),
                              "turn_mean": round(float(p.turn[m].mean()), 1)} for p in P}}
            for (lo, hi, lab), m in zip(BINS, ms)}


def fig_box(P, G, out):
    """구간별 분포 — 막대(중앙값·p99 두 점)로는 안 보이는 분포 모양을 상자로 본다."""
    import matplotlib.pyplot as plt
    ms = [(G["turn"] >= lo) & (G["turn"] < hi) for lo, hi, _ in BINS]
    labs = [lab for _, _, lab in BINS]
    ser = [("정답", C.C_GT)] + [(p_.name, p_.color) for p_ in P]
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 7.2))

    def draw(ax, groups, ylab, title, ref=None, ref_lab=None):
        pos, cols = [], []
        data = []
        for b in range(len(BINS)):
            for j, (name, color) in enumerate(ser):
                if groups[b][j] is None:
                    continue
                data.append(groups[b][j])
                pos.append(b * (len(ser) + 1.2) + j)
                cols.append(color)
        bp = ax.boxplot(data, positions=pos, whis=(5, 95), showfliers=False, showmeans=True,
                        patch_artist=True, widths=0.78,
                        meanprops=dict(marker="D", ms=4.0, mfc=C.SURF, mec=C.INK, mew=0.9),
                        medianprops=dict(color=C.INK, lw=1.6),
                        whiskerprops=dict(color=C.AXIS, lw=1.0), capprops=dict(color=C.AXIS, lw=1.0))
        for b_, c in zip(bp["boxes"], cols):
            b_.set(facecolor=c, alpha=0.55, edgecolor=C.AXIS, lw=0.9)
        ax.set_xticks([b * (len(ser) + 1.2) + (len(ser) - 1) / 2 for b in range(len(BINS))])
        ax.set_xticklabels(labs, fontsize=9)
        ax.set_xlabel("정답 총 회전량 구간")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.8)
        if ref is not None:
            ax.axhline(ref, color=C.C_ORANGE, lw=1.2, ls=(0, (4, 2)))
            ax.text(ax.get_xlim()[1], ref, f"{ref_lab} ", color=C.C_ORANGE, fontsize=7.8, va="bottom",
                    ha="right")

    gpsi = [[G["dp"][m][G["ok"][m]]] + [p_.dp_pool(m) for p_ in P] for m in ms]
    draw(axes[0], gpsi, "|Δψ| [°/step]", "step마다의 꺾임 분포", LIM, f"라벨 임계 {LIM}°/step")
    axes[0].set_ylim(-0.3, 12)
    gade = [[None] + [p_.minade[m] for p_ in P] for m in ms]
    draw(axes[1], gade, "minADE6 [m]", "시나리오마다의 거리오차 분포")
    axes[1].set_ylim(0, 6)
    from matplotlib.patches import Patch
    axes[0].legend([Patch(facecolor=c, alpha=0.55, edgecolor=C.AXIS) for _, c in ser],
                   [n for n, _ in ser], fontsize=8.4, loc="upper left", ncol=2)
    q = lambda v, k: np.percentile(v, k)
    m_turn = ms[2]
    note(axes[2], [
        "읽는 방법",
        "상자 = 가운데 50%(Q1~Q3) · 상자 안 선 = 중앙값 · ◆ = 평균 · 수염 = 5~95 백분위입니다.",
        "바깥값은 점으로 찍지 않았습니다(왼쪽 칸은 step 수가 수백만 개라 점을 찍으면 상자가 묻힙니다).",
        "앞 그림의 막대는 중앙값·p99 **두 점**이었습니다. 이 그림은 같은 값의 **분포 모양**을 봅니다.",
        "",
        "왼쪽 칸에서 읽히는 것 (회전 구간 기준)",
        f"   정답       중앙 {q(gpsi[2][0], 50):.2f} · 상자 {q(gpsi[2][0], 25):.2f}~{q(gpsi[2][0], 75):.2f} "
        f"· 수염 끝 {q(gpsi[2][0], 95):.2f}",
        *[f"   {n:10s} 중앙 {q(gpsi[2][j + 1], 50):.2f} · 상자 {q(gpsi[2][j + 1], 25):.2f}~"
          f"{q(gpsi[2][j + 1], 75):.2f} · 수염 끝 {q(gpsi[2][j + 1], 95):.2f}"
          for j, (n, _) in enumerate(ser[1:])],
        "",
        "   벌점 없음은 상자가 통째로 정답 위에 있습니다 — 모든 step 이 과하게 꺾입니다.",
        "   액션 벌점은 상자가 바닥에 눌려 **정답보다도 아래**입니다. 회전 중에도 안 꺾는다는 뜻입니다.",
        "   좌표 벌점은 상자가 정답 위에 있지만 벌점 없음보다 낮습니다 — 꼬리를 자른 모양입니다.",
        "",
        "오른쪽 칸 — 세 판의 상자가 거의 겹칩니다. 회전 구간에서도 중앙값 차이는 "
        f"{q(P[1].minade[m_turn], 50) - q(P[0].minade[m_turn], 50):+.2f} m(액션−없음) 수준이고,",
        "   앞 그림의 평균 차이는 **긴 꼬리**(상위 5%가 5 m 를 넘는 구간)에서 나옵니다.",
        "   그래서 평균 하나로 비교할 때 seed 마다 크게 흔들렸던 것입니다(+0.086 / +0.102 / +0.304).",
    ], fs=8.4)
    fig.suptitle("구간별 분포 — 상자로 본 세 판 (ah2 · 30에폭 코사인 · 시드 0 · val 24,988)",
                 fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    C.savefig(fig, out)
    return {lab: {n: {"p50": round(float(q(gpsi[b][j], 50)), 3), "p95": round(float(q(gpsi[b][j], 95)), 3)}
                  for j, (n, _) in enumerate(ser)} for b, lab in enumerate(labs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    C.setup_mpl()
    out = Path(a.out)
    P = [Pair(*t) for t in TAGS]
    for p in P:
        print(f"[판] {p.name:26s} {p.tag}  승자 모드 좌표 위반 {p.viol_pct():.2f}%  minADE6 {p.minade.mean():.4f}",
              flush=True)
    gpos = np.load(C.tag_dirs(P[0].tag)["data"] / "raw.npz")["pos"][:, C.OBS:]   # 정답 60스텝 (t = 0.1~6.0 s)
    gdp, gok = dpsi_deg(gpos)
    G = {"pos": gpos, "turn": turn_deg(gpos), "dp": gdp, "ok": gok,
         "dp_med": np.array([med(gdp[k], gok[k]) for k in range(len(gpos))])}
    print(f"[정답] 회전군(≥{C.TURN_DEG:.0f}°) {int((G['turn'] >= C.TURN_DEG).sum()):,}개 · "
          f"|Δψ| 중앙값 {np.median(gdp[gok]):.2f}°/step", flush=True)
    picks = {"tags": {p.name: p.tag for p in P},
             "winner_viol_pct": {p.name: round(p.viol_pct(), 3) for p in P},
             "mech": fig_mech(P, G, out / "penalty_mech.png"),
             "turn": fig_turn(P, G, out / "penalty_turn.png"),
             "bins": fig_bins(P, G, out / "penalty_bins.png"),
             "box": fig_box(P, G, out / "penalty_box.png")}
    (out / "penalty_example_picks.json").write_text(json.dumps(picks, indent=2, ensure_ascii=False))
    print(f"[done] {out} — penalty_mech.png · penalty_turn.png · penalty_bins.png · penalty_box.png",
          flush=True)


if __name__ == "__main__":
    main()
