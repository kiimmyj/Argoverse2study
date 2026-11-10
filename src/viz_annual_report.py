#!/usr/bin/env python3
"""연차보고서 '최신 연구 환경 변화에 대한 조사' 그림 2장.

  conda run -n av2 python src/viz_annual_report.py

수치는 전부 아래 표에 출처와 함께 적었다(~/paper-kb/notes/*, EvalAI API 2026-11-10 조회).
손으로 문서에 옮기지 말고 이 표를 고친 뒤 다시 그린다.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "annual_report")

# dataviz 기본 팔레트 slot 1·2 (밝은 면 검증값), 글자·축은 텍스트 토큰
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# ── 그림 1: 데이터셋 규모 ──────────────────────────────────────────────
# (이름, 총 시간 h, 수집 방식, 라벨, 출처)
DATASETS = [
    ("NGSIM",        1.5,  "부감 촬영", "1.5 h · 고속도로",           "krajewski2018highd 표 I"),
    ("nuScenes",     5.5,  "차량 센서", "5.5 h · 2→6 s @2 Hz",        "caesar2020nuscenes 표 1, 예측 과제 devkit"),
    ("highD",        16.5, "부감 촬영", "16.5 h · 고속도로",          "krajewski2018highd 표 I"),
    ("Argoverse 1",  320,  "차량 센서", "320 h · 2→3 s @10 Hz",       "chang2019argoverse1 3.1절"),
    ("WOMD",         574,  "차량 센서", "574 h · 1→8 s @10 Hz",       "ettinger2021womd 표 1"),
    ("Argoverse 2",  763,  "차량 센서", "763 h · 5→6 s @10 Hz",       "wilson2023argoverse2 표 2"),
    ("Lyft Level 5", 1118, "차량 센서", "1,118 h · 단일 노선",        "houston2020one 표 2"),
]

# ── 그림 2: AV2 test 단일 에이전트 minADE6 (K=6, 5 s→6 s), 논문 보고치 ──
# (모델, 학회·연도, x 위치, minADE6, 앙상블?, 출처)
AV2 = [
    ("QCNet",         "CVPR'23",    2023.0, 0.65,  False, "zhou2023qcnet 표 1"),
    ("QCNet",         "CVPR'23",    2023.0, 0.62,  True,  "zhou2023qcnet 표 1"),
    ("ProphNet",      "CVPR'23",    2023.2, 0.68,  False, "wang2023prophnet 표 3"),
    ("Forecast-MAE",  "ICCV'23",    2023.4, 0.709, False, "cheng2023forecastmae 표 3"),
    ("Forecast-MAE",  "ICCV'23",    2023.4, 0.690, True,  "cheng2023forecastmae 표 3"),
    ("SIMPL",         "RA-L'24",    2024.0, 0.72,  False, "zhang2024simpl 표 II"),
    ("DeMo",          "NeurIPS'24", 2024.2, 0.61,  False, "zhang2024demo 표 1"),
    ("DeMo",          "NeurIPS'24", 2024.2, 0.60,  True,  "zhang2024demo 표 1"),
    ("Tamba",         "CVPR'25",    2025.0, 0.64,  False, "huang2025trajectory 표 1 (앙상블 여부 미기재)"),
    ("FutureNet-LoF", "ICRA'25",    2025.2, 0.58,  True,  "wang2025futurenet 표 1"),
]
LEADER = ("2026-11 리더보드 1위 IMR", 0.5714)   # EvalAI 4098, brier-minFDE6 정렬 1위의 minADE6


def setup():
    path = [f for f in font_manager.findSystemFonts() if "NotoSansCJK" in f and "Regular" in f]
    if path:
        font_manager.fontManager.addfont(path[0])
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path[0]).get_name()
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.size": 10, "axes.unicode_minus": False})


def fig_datasets():
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=200)
    ys = range(len(DATASETS))
    for y, (name, h, kind, label, _) in zip(ys, DATASETS):
        car = kind == "차량 센서"
        ax.hlines(y, 1, h, color=GRID, lw=1, zorder=1)
        ax.scatter(h, y, s=70, marker="o" if car else "D", color=BLUE if car else ORANGE,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(h * 1.25, y, label, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(ys), [d[0] for d in DATASETS], color=INK, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(1, 30000)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_xlabel("공개된 총 주행 시간 [h, 로그 축]")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    h1 = ax.scatter([], [], marker="o", color=BLUE, label="차량 센서 기반 (도심, HD 맵 제공)")
    h2 = ax.scatter([], [], marker="D", color=ORANGE, label="부감 촬영 기반 (고속도로)")
    ax.legend(handles=[h1, h2], loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_av2():
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=200)
    # 라벨 자리(dx, dy, ha, va) — 점·선과 겹치지 않게 점마다 정함
    place = {"ProphNet": (0.04, -0.004, "left", "top"), "Forecast-MAE": (-0.04, 0.004, "right", "bottom"),
             "FutureNet-LoF": (-0.04, 0.004, "right", "bottom")}
    for name, venue, x, v, ens, _ in AV2:
        ax.scatter(x, v, s=70, marker="D" if ens else "o", color=ORANGE if ens else BLUE,
                   edgecolor="white", linewidth=1.5, zorder=3)
        labelled = (not ens) or name == "FutureNet-LoF"
        if labelled:
            dx, dy, ha, va = place.get(name, (0.04, 0.004, "left", "bottom"))
            ax.text(x + dx, v + dy, f"{name} ({venue})", fontsize=8, color=INK2, ha=ha, va=va)
    for name in {a[0] for a in AV2 if a[4]}:   # 같은 모델의 단일↔앙상블을 잇는 선
        pts = [a for a in AV2 if a[0] == name]
        if len(pts) == 2:
            ax.plot([pts[0][2]] * 2, [pts[0][3], pts[1][3]], color=GRID, lw=1, zorder=1)
    ax.axhline(LEADER[1], color=MUTED, lw=1, ls="--", zorder=0)
    ax.text(2025.75, LEADER[1] - 0.004, f"{LEADER[0]} {LEADER[1]:.3f}", fontsize=8, color=MUTED,
            ha="right", va="top")
    ax.set_xlim(2022.5, 2025.8)
    ax.set_xticks([2023, 2024, 2025], ["2023", "2024", "2025"])
    ax.set_ylim(0.55, 0.75)
    ax.set_ylabel("AV2 test minADE6 [m]  (낮을수록 좋음)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    h1 = ax.scatter([], [], marker="o", color=BLUE, label="단일 모델")
    h2 = ax.scatter([], [], marker="D", color=ORANGE, label="앙상블")
    ax.legend(handles=[h1, h2], loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def main():
    setup()
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig1_datasets.png", fig_datasets), ("fig2_av2_sota.png", fig_av2)]:
        fig = fn()
        fig.savefig(os.path.join(OUT, name), facecolor="white")
        print("저장:", os.path.normpath(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
