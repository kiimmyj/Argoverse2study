"""heading 전수 조사 결과 표를 그림으로 만든다 (메시지·발표용)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager

try:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

BG, CARD, HEAD = "#16171b", "#1f2126", "#2a2d34"
TXT, DIM, HL, OK = "#e8ecf2", "#98a0aa", "#ffb454", "#5ce08a"

T1 = ("표 1.  튀는 값 — 스텝간 변화 |Δh|     전체 249,880 시나리오 / 25,737,880 step",
      ["구간", "값", "의미"],
      [["중앙 (p50)", "0.050°", "절반이 이 이하 — 거의 미동 없음"],
       ["p99", "2.75°", ""],
       ["p99.99", "7.3°", "정상값 상한 (요레이트 73°/s, 물리적으로 가능)"],
       ["> 30° 인 step", "11 건", "2,573만 중 0.00004%  ·  시나리오 7개 (0.0028%)"],
       ["> 90° 인 step", "2 건", "시나리오 1개"],
       ["최대", "174.1°", "train ff2286c7 — 1,740°/s, 차량 한계의 20배 = 라벨 오류"]],
      [2, 3, 5], [0.26, 0.14, 0.60])

T2 = ("표 2.  박스 틀어짐 — 트랙별 계통 편향     val 3,000 시나리오 / 주행 트랙 25,279개",
      ["측정", "값", "의미"],
      [["편향 평균", "−0.04°", "한쪽으로 돌아간 경향 없음"],
       ["편향 중앙", "−0.01°", ""],
       ["트랙 간 산포 (std)", "4.58°", "개별 트랙은 ±5° 흔들림 (실제 슬립각 포함)"],
       ["|편향| > 5°", "0.63%", "트랙 158개 / 25,279"],
       ["|편향| > 10°", "0.36%", "트랙 91개"],
       ["트랙 안 변화", "0.18°", "전반부 vs 후반부 — 틀어져도 일정하게 유지"]],
      [0, 2, 5], [0.30, 0.14, 0.56])

T3 = ("표 3.  못 잡는 문제 — 결측·끊김     val 5,000 시나리오",
      ["대상", "결과", "의미"],
      [["focal step 수", "전부 110", "110 미만인 시나리오 0개"],
       ["focal 중간 끊김", "0 건", "gap 있는 시나리오 0개"],
       ["focal heading NaN", "0 개", "결측값 없음"],
       ["다른 차량 (참고)", "82.4% 끊김", "시나리오당 55대 중 45대 — focal 만 완전 보장"]],
      [0, 1, 2], [0.28, 0.16, 0.56])

RH, HH, GAP, TITLE = 0.032, 0.036, 0.030, 0.035


def draw(fig, y, spec):
    title, cols, rows, hi, w = spec
    fig.text(0.055, y, title, fontsize=15, color=TXT, va="top", weight="bold")
    y -= TITLE
    x0, x1 = 0.055, 0.945
    tot = x1 - x0
    h = HH + RH * len(rows)
    fig.patches.append(FancyBboxPatch((x0, y - h), tot, h, boxstyle="round,pad=0.006",
                                      transform=fig.transFigure, facecolor=CARD,
                                      edgecolor="#33373f", lw=1.2, zorder=1))
    fig.patches.append(Rectangle((x0, y - HH), tot, HH, transform=fig.transFigure,
                                 facecolor=HEAD, edgecolor="none", zorder=2))
    xs = [x0 + 0.018]
    for f in w[:-1]:
        xs.append(xs[-1] + f * tot)
    for c, xx in zip(cols, xs):
        fig.text(xx, y - HH / 2, c, fontsize=12.5, color=DIM, va="center", zorder=3)
    for i, r in enumerate(rows):
        ry = y - HH - RH * i
        if i % 2:
            fig.patches.append(Rectangle((x0, ry - RH), tot, RH, transform=fig.transFigure,
                                         facecolor="#232630", edgecolor="none", zorder=2))
        bold = i in hi
        for j, (cell, xx) in enumerate(zip(r, xs)):
            col = TXT
            if bold and j <= 1:
                col = HL if "°" in cell or "," in cell or cell.isdigit() else TXT
            if cell in ("0",):
                col = OK
            fig.text(xx, ry - RH / 2, cell, fontsize=13 if bold else 12.5, color=col,
                     va="center", zorder=3,
                     weight="bold" if bold and j <= 1 else "normal")
    return y - h - GAP


fig = plt.figure(figsize=(11.6, 13.8), dpi=140, facecolor=BG)
fig.text(0.055, 0.976, "AV2 heading 을 θ = h − k 에 쓸 수 있는가", fontsize=20,
         color=TXT, va="top", weight="bold")
fig.text(0.055, 0.941, "튀는 값 · 박스 틀어짐 · 결측 세 가지를 실측",
         fontsize=12.5, color=DIM, va="top")
y = 0.885
for t in (T1, T2, T3):
    y = draw(fig, y, t)
fig.text(0.055, 0.055, "θ = h − k 의 실측 분포는 ±10° 안에 91.7%.  heading 노이즈(p99.99 7.3°)와 "
         "박스 편향(std 4.58°)이 그보다 작고, 편향은 트랙 안에서 일정하다.",
         fontsize=11.5, color=DIM, va="center")
fig.text(0.055, 0.026, "→ 결론: 쓸 수 있다.  가드는 '스텝당 |Δh| > 30° 면 직전 값 유지' 하나면 충분 "
         "(정상값 상한의 4배, 이상치 11건 전부 포착, 영향 시나리오 0.0028%)",
         fontsize=12, color=OK, va="center")
fig.savefig("/home/user/Argoverse2study/heading_quality_tables.png", facecolor=BG)
print("saved -> heading_quality_tables.png")
