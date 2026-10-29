"""아이디어 노트용 그림 세 장 — docs/idea_note_ah.md 가 쓴다.

왜 따로 만드나 — 미팅 그림(viz_meeting_1029.py)은 "결론을 전달"하는 용도라 수치가 앞에 선다.
아이디어 노트는 처음 읽는 사람이 **생각의 갈래**를 따라올 수 있어야 하므로, 같은 기록을
① 계보도 ② 뒤집힌 주장 대조 ③ 지금 돌고 있는 실험의 상태 로 다시 접는다.
새로 계산하는 것은 없다. n3 만 r0.log 를 그때그때 읽는다.

  n1_idea_tree.png    아이디어 계보도 — 질문 하나에서 갈라진 근거·반론·실험
  n2_overturned.png   그럴듯했지만 측정으로 뒤집힌 주장 다섯
  n3_r0_progress.png  R0(원본 SIMPL) 진행 — LR warmup 통과 여부와 목표선

  python viz_idea_note.py            # 세 장 전부
  python viz_idea_note.py --only n3  # 한 장만
"""
import argparse
import re
from pathlib import Path

import numpy as np

import viz_v4_common as C

OUT = C.ROOT / "viz" / "idea_note"
R0_LOG = Path("/home/user/av2_simpl/r0.log")

VERDICT = {                      # 판정 → (색, 글자)
    "live":  (C.C_GREEN,  "살아남음"),
    "flip":  (C.C_RED,    "뒤집힘"),
    "open":  (C.C_ORANGE, "미결"),
    "drop":  (C.MUTED,    "기각"),
    "run":   (C.C_BLUE,   "진행 중"),
}


def card(ax, x, y, w, h, title, body, kind=None, fs=8.6, tfs=9.4, align="left"):
    """글상자 하나. kind 가 있으면 왼쪽에 판정 띠를 붙인다."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    col = VERDICT[kind][0] if kind else C.AXIS
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8",
                                fc="white", ec=col, lw=1.1, zorder=3))
    if kind:
        ax.add_patch(Rectangle((x, y), 0.55, h, fc=col, ec="none", zorder=4))
    tx = x + 1.6 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    ax.text(tx, y + h - 1.5, title, ha=ha, va="top", fontsize=tfs, color=C.INK,
            fontweight="bold", zorder=5)
    if body:
        ax.text(tx, y + h - 4.3, body, ha=ha, va="top", fontsize=fs, color=C.INK2,
                linespacing=1.55, zorder=5)
    return x + w / 2, y


def arrow(ax, p, q, col=None, lw=1.3, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=col or C.AXIS, lw=lw, zorder=2,
                                 shrinkA=1, shrinkB=2))


def blank(plt, w, h, ylo=0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(ylo, 100)
    ax.axis("off")
    ax.set_facecolor(C.SURF)
    return fig, ax


# ───────────────────────────────────────── n1 — 아이디어 계보도

BELIEVE = [
    ("행동 공간이 좁다", "사람은 (x, y) 가 아니라 가·감속과\n조향으로 운전한다. 같은 궤적을\n더 적은 자유도로 쓴다.", "open"),
    ("표현이 불변이다", "(a, h) 는 좌표계 회전·평행이동에\n영향받지 않는다.", "open"),
    ("지도와 묶인다", "θ = h − k 로 두면 '차로를 얼마나\n벗어났나' 가 그대로 상태가 된다.", "open"),
]
DOUBT = [
    ("이미 불변인 모델엔 중복", "QCNet·SIMPL·HPTR 은 구조가 이미\n상대좌표라 θ 가 새 정보가 아니다.\n정규화로 푸는 쪽(Forecast-MAE·DeMo)\n에만 새롭다 — 그런데 두 축이 역상관.", "live"),
    ("v4 가 그걸로 안 됐다", "근거 확인된 요소 넷을 다 쌓고\n100에폭 5시드 → 1.4038.\n기준(좌표 벌점만) 1.3591 보다\n+0.045 나쁘고 5시드 전부 나쁘다.", "live"),
    ("후보 경로 기하가 막았다", "후보 집합의 오라클 경로조차\n정답에서 평균 1.079 m.\n목표 minADE6 ≤ 1.0 과 같은 자릿수.", "live"),
]
LADDER = [
    ("R0  원본 SIMPL", "입력 교체 말고 바뀐 것(GPU 8→1,\n유효 배치 64→16, 미검증 LR)을\n먼저 따로 떼어 본다. 50에폭.", "run"),
    ("R1  (a, h) 2채널", "지도 분기는 손대지 않는다.\n**R1 − R0 이 가설의 답이다.**", "open"),
    ("R2  출력도 (a, Δθ)", "Frenet 적분기. 최근접 차로 k 가\n교차로에서 튀는 것이 미해결.", "open"),
    ("R3  Forecast-MAE + θ", "B그룹 대조군. θ 가 진짜 새 정보인\n쪽에서도 같은 결론인지.", "open"),
]


def n1(plt):
    fig, ax = blank(plt, 14.2, 7.9, ylo=7.0)

    card(ax, 22, 90, 56, 8.5, "(a, h) 로 기술하면 (x, y) 보다 예측이 쉬워지는가?",
         "— 이 노트 전체가 이 질문 하나를 가른다", tfs=12.2, fs=9.2, align="center")

    cols = [(1.5, "믿는 이유", BELIEVE, 31),
            (34.5, "의심하는 이유", DOUBT, 31),
            (67.5, "가르는 방법", LADDER, 31)]
    for x0, head, items, w in cols:
        ax.text(x0 + w / 2, 85.0, head, ha="center", va="center", fontsize=10.6,
                color=C.INK, fontweight="bold")
        arrow(ax, (50, 90), (x0 + w / 2, 86.6))
        y = 80.0
        for i, (t, b, k) in enumerate(items):
            h = 13.0 if len(items) == 3 else 15.0
            h = 4.2 + 2.35 * (b.count("\n") + 1) + 3.0
            y -= h + 2.4
            cx, _ = card(ax, x0, y, w, h, t, b.replace("**", ""), kind=k)
            if i and len(items) == 4:
                arrow(ax, (x0 + 2.0, y + h + 2.4), (x0 + 2.0, y + h + 0.4), col=C.C_BLUE)
    ax.text(26.0, 10.2, "왼쪽 띠 색 = 판정", ha="right", va="center",
            fontsize=8.8, color=C.MUTED)
    for i, k in enumerate(("live", "run", "open")):
        x = 29.0 + i * 16.0
        ax.add_patch(plt.Rectangle((x, 9.75), 1.2, 0.95, fc=VERDICT[k][0], ec="none"))
        ax.text(x + 2.2, 10.2, VERDICT[k][1] + (" — 측정이 뒷받침" if k == "live" else
                " — 지금 돌고 있음" if k == "run" else " — 아직 안 가름"),
                ha="left", va="center", fontsize=8.8, color=C.INK2)
    fig.suptitle("아이디어 계보도 — 질문 하나에서 갈라진 근거·반론·실험",
                 fontsize=13.4, y=0.975, color=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    C.savefig(fig, OUT / "n1_idea_tree.png")


# ───────────────────────────────────────── n2 — 뒤집힌 주장

FLIPS = [
    ("“a 는 가속도다”",
     "속도 증분이고 **첫 값이 v0** 다",
     "누적합 마지막값 vs 캐시 v0 상관 0.986 · 중앙오차 0.163 m/s (val 3,000판)"),
    ("“(a, h) 만으론 v0 를 몰라\n3채널이 필요하다”",
     "**2채널로 충분하다**",
     "(a, h) 만 적분해 과거 5초 복원 중앙 0.56 m · p90 1.28 m (val 400판)"),
    ("“h 를 위치차분으로 쓰는 건\n슬립각 때문이다”",
     "슬립각은 0.20° 로 **없다**.\n이유는 a 와의 **무손실 복원 짝맞춤**",
     "속력만 AV2 velocity 로 바꾸면 주행 구간 1.2 m 틀어짐 (|Δp|/dt 와 0.274 m/s 어긋남)"),
    ("“뒤집힌 yaw 를\n트랙 단위로 고쳐야 한다”",
     "트랙 단위 뒤집힘 **0.00%**.\n낱개 step 잡음이라 **저속 게이트**가 일한다",
     "이동거리 층화: 0~2 m 45.1% · 2~5 m 16.3% · 5~20 m 3.6% · 20 m+ 0.00%"),
    ("“저속 보조는 15.5% 의 예외다”",
     "SIMPL 의 전 agent 에서는 **65.3% · 다수 경로**",
     "1 m/s 미만 step 몫: focal 22.6% / 전 트랙 65.3% / 비차량 75.0% (step 198,050개)"),
]


def n2(plt):
    fig, ax = blank(plt, 14.2, 7.0, ylo=13.5)
    y = 92.0
    for before, after, how in FLIPS:
        nb, na = before.count("\n") + 1, after.count("\n") + 1
        h = 3.0 + 2.9 * max(nb, na)
        card(ax, 1.5, y - h, 36, h, "", "", kind="flip")
        ax.text(3.8, y - h / 2, before, ha="left", va="center", fontsize=9.3,
                color=C.C_RED, linespacing=1.5, zorder=5)
        arrow(ax, (38.6, y - h / 2), (43.4, y - h / 2), col=C.INK2, lw=1.6)
        card(ax, 44.0, y - h, 54.5, h, "", "", kind="live")
        ax.text(46.3, y - h / 2, after.replace("**", ""), ha="left", va="center",
                fontsize=9.3, color=C.INK, linespacing=1.5, zorder=5)
        y -= h + 1.1
        ax.text(3.8, y + 0.2, "측정:  " + how, ha="left", va="top", fontsize=8.1,
                color=C.MUTED, zorder=5)
        y -= 4.6
    ax.text(50, 15.8, "왼쪽은 코드만 보면 그렇게 읽히는 말이고, 오른쪽은 측정이 말한 것이다. "
                      "다섯 개 모두 2026-10-28~29 에 뒤집혔다.",
            ha="center", va="center", fontsize=8.8, color=C.MUTED)
    fig.suptitle("그럴듯했지만 측정으로 뒤집힌 주장 다섯",
                 fontsize=13.4, y=0.972, color=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    C.savefig(fig, OUT / "n2_overturned.png")


# ───────────────────────────────────────── n3 — R0 진행

def read_r0():
    """r0.log → 에폭별 (train loss, lr, train minADE6, val loss, val minADE6)."""
    if not R0_LOG.exists():
        return []
    rows, cur = [], None
    for ln in R0_LOG.read_text(errors="replace").splitlines():
        m = re.match(r"^Epoch (\d+)", ln)
        if m:
            cur = {"ep": int(m.group(1))}
            rows.append(cur)
            seen_val = False
            continue
        if cur is None:
            continue
        m = re.search(r"\[Training\] Avg\. loss: ([\d.]+).*lr: ([\d.e-]+)", ln)
        if m:
            cur["tr"], cur["lr"] = float(m.group(1)), float(m.group(2))
            continue
        m = re.search(r"\[Validation\] Avg\. loss: ([\d.]+)", ln)
        if m:
            cur["va"] = float(m.group(1))
            seen_val = True
            continue
        m = re.search(r"minade_k: ([\d.]+)", ln)
        if m:
            cur["va_ade" if cur.get("va") is not None and "tr_ade" in cur else "tr_ade"] = float(m.group(1))
    return [r for r in rows if "tr" in r]


def n3(plt):
    rows = read_r0()
    ep = np.array([r["ep"] for r in rows])
    tr = np.array([r["tr"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    vi = [i for i, r in enumerate(rows) if "va" in r]
    vep = ep[vi]
    va = np.array([rows[i]["va"] for i in vi])
    vade = np.array([rows[i].get("va_ade", np.nan) for i in vi])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 4.9))
    xhi = ep[-1] + max(2.1, 0.42 * ep[-1])

    ax.plot(ep, tr, color=C.C_BLUE, marker="o", ms=4.2)
    ax.plot(vep, va, color=C.C_ORANGE, marker="s", ms=4.6)
    ax.text(ep[-1] + 0.15, tr[-1], " train 손실", va="center", fontsize=9, color=C.C_BLUE)
    ax.text(vep[-1] + 0.15, va[-1], " val 손실", va="center", fontsize=9, color=C.C_ORANGE)
    ax.set_xlabel("에폭"); ax.set_ylabel("손실")
    ax.set_ylim(0.28, 0.48); ax.set_xlim(-0.35, xhi)
    cx = ax.twinx()
    cx.plot(ep, lr * 1e4, color=C.C_GREEN, marker="^", ms=4.0, lw=1.3, ls="--")
    cx.set_ylabel("학습률 [×1e-4]", color=C.C_GREEN)
    cx.set_ylim(0, 11.5); cx.grid(False)
    cx.tick_params(axis="y", colors=C.C_GREEN)
    mid = len(ep) // 2
    cx.text(ep[mid], lr[mid] * 1e4 + 0.5, "학습률", ha="center", va="bottom",
            fontsize=9, color=C.C_GREEN)
    cx.axhline(10, color=C.C_GREEN, lw=0.8, ls=":", alpha=0.7)
    cx.text(ep[-1] + 0.2, 10.0, " warmup 정점 1e-3", va="center", fontsize=8.4,
            color=C.C_GREEN)
    ax.set_title("손실과 학습률 — warmup 정점을 통과했는가  (에폭 0 train 1.52 는 축 밖)")

    bx.plot(vep, vade, color=C.C_TOP, marker="o", ms=5.0, zorder=4)
    for x, y in zip(vep, vade):
        bx.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.4, color=C.C_TOP)
    bx.text(vep[-1] + 0.12, vade[-1], " R0 val minADE6", va="center", fontsize=9,
            color=C.C_TOP)
    for val, lab, col in ((1.0, "사용자 목표 ≤ 1.0", C.C_ORANGE),
                          (0.777, "배포 체크포인트 0.777 (우리가 재평가해 일치시킴)", C.C_GREEN),
                          (0.72, "논문 보고치 0.72 (체크포인트 분실)", C.MUTED)):
        bx.axhline(val, color=col, lw=1.1, ls="--")
        bx.text(vep[0] + 0.1, val + 0.008, lab, va="bottom", ha="left",
                fontsize=8.2, color=col)
    bx.set_xlabel("에폭"); bx.set_ylabel("val minADE6 [m]")
    bx.set_ylim(0.6, 1.25)
    bx.set_xlim(vep[0] - 0.4, vep[-1] + max(2.0, 0.42 * vep[-1]))
    bx.set_title("정확도 — SIMPL 의 minADE 는 최소 FDE 모드의 ADE 라 보수적")

    fig.suptitle(f"R0 (원본 SIMPL · 50에폭 예정) 진행 — 에폭 {ep[-1]} 까지",
                 fontsize=13.0, y=0.985, color=C.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.savefig(fig, OUT / "n3_r0_progress.png")
    print("  n3 rows:", [(r["ep"], r["tr"], r.get("va"), r.get("va_ade")) for r in rows])


FIGS = {"n1": n1, "n2": n2, "n3": n3}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FIGS))
    a = ap.parse_args()
    plt = C.setup_mpl()
    for k, f in FIGS.items():
        if a.only and k != a.only:
            continue
        f(plt)
        print("그림", k, "완료")
