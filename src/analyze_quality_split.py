"""입력 품질 분리 측정의 분석 — (a, h) 의 열세가 '주변 차 보조비율'을 따라가는가.

가르려는 것
  (a) **표현이 나쁘다** — 보조비율과 무관하게 열세가 **평평**해야 한다.
  (b) **주변 차 입력을 뭉갰다** — 보조비율이 낮은 장면에서 열세가 **줄어야** 한다.

왜 s43 도 같이 보나 — 버킷별 격차에도 잡음이 있다. **시드만 바꾼 s43 − R0 가 그 눈금**이다.
s43 쪽이 평평하고 0 근처인데 R1 쪽만 기울어야 신호다. 자가 없으면 아무 말도 못 한다.

⚠️ **이 측정의 자체 교란** — 보조비율이 낮은 장면은 차가 빨리 움직이는 장면이고, 원래 예측이
쉬울 수 있다. 그래서 버킷마다 **R0 자신의 오차**(=그 버킷이 원래 얼마나 어려운가)와
**focal 속력**을 같이 찍는다. R0 는 (a, h) 를 안 쓰므로 난이도의 정직한 대리값이다.

  python src/analyze_quality_split.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIMPL = Path("/home/user/av2_simpl")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import viz_v4_common as C

OUT = ROOT / "viz" / "qsplit"
TAGS = ("R0", "R1", "s43")
# 학습 로그의 에폭 49 검증값. 재평가가 이것과 **차이까지** 맞는지 보는 자체 검산.
LOGGED = {"R0": 0.795, "R1": 0.830, "s43": 0.801}
KEY = "fill_sur_valid"      # 주변 차, 결측 패딩을 뺀 유효 step 에서의 보조비율
NB = 5


def load():
    d = {t: dict(np.load(SIMPL / "tools" / f"qsplit_{t}.npz", allow_pickle=True)) for t in TAGS}
    ids = [d[t]["seq_id"] for t in TAGS]
    assert all((ids[0] == x).all() for x in ids[1:]), "장면 순서가 세 판에서 다르다"
    return d


def mean_ci(x):
    """짝지은 장면 차이의 평균과 2SE. 장면은 서로 독립 표본이라 이 CI 는 쓸 수 있다.

    ⚠️ 단, 이것은 **이 런 쌍 안에서의** 불확실성이다. '다른 시드로 다시 학습하면
    어떻게 되나'는 안 재 준다 — 그건 s43 열이 하는 일이다.
    """
    x = np.asarray(x, float)
    return float(x.mean()), float(2 * x.std(ddof=1) / np.sqrt(len(x)))


def table(d):
    f = d["R0"][KEY].astype(float)
    ok = np.isfinite(f)                      # 주변 차가 없는 장면은 보조비율이 정의되지 않는다
    q = np.quantile(f[ok], np.linspace(0, 1, NB + 1))
    q[0], q[-1] = -np.inf, np.inf
    rows = []
    for i in range(NB):
        m = ok & (f >= q[i]) & (f < q[i + 1]) if i < NB - 1 else ok & (f >= q[i])
        r = {"i": i, "n": int(m.sum()),
             "fill_lo": float(np.nanmin(f[m])), "fill_hi": float(np.nanmax(f[m])),
             "fill": float(f[m].mean()),
             "r0_ade": float(d["R0"]["ade6"][m].mean()),
             "v": float(d["R0"]["focal_v"][m].mean()),
             "n_agent": float(d["R0"]["n_agent"][m].mean())}
        for t in ("R1", "s43"):
            g, h = mean_ci(d[t]["ade6"][m] - d["R0"]["ade6"][m])
            r[f"{t}_gap"], r[f"{t}_hw"] = g, h
        rows.append(r)
    return rows, int((~ok).sum())


def by(d, key, nb=NB, label=""):
    """아무 축으로나 5등분해서 격차를 본다. 보조비율 말고 **속력**도 봐야 한다 —
    보조비율이 낮은 버킷이 하필 제일 빠른 버킷이라 둘이 엉켜 있다."""
    f = d["R0"][key].astype(float)
    ok = np.isfinite(f)
    q = np.quantile(f[ok], np.linspace(0, 1, nb + 1)); q[0], q[-1] = -np.inf, np.inf
    print(f"\n── {label or key} 로 5등분 ──")
    print(f"{'버킷':<4}{'범위':>16}{'장면':>8}{'R0 오차':>9}{'보조비율':>9}"
          f"{'R1−R0':>16}{'s43−R0':>16}")
    out = []
    for i in range(nb):
        m = ok & (f >= q[i]) & (f < q[i + 1]) if i < nb - 1 else ok & (f >= q[i])
        g1, h1 = mean_ci(d["R1"]["ade6"][m] - d["R0"]["ade6"][m])
        g2, h2 = mean_ci(d["s43"]["ade6"][m] - d["R0"]["ade6"][m])
        print(f"{i+1:<4}{f[m].min():>7.1f}~{f[m].max():<8.1f}{int(m.sum()):>8,}"
              f"{d['R0']['ade6'][m].mean():>9.3f}{d['R0'][KEY][m].mean()*100:>8.0f}%"
              f"{g1:>+10.4f}±{h1:.4f}{g2:>+10.4f}±{h2:.4f}")
        out.append({"lo": float(f[m].min()), "hi": float(f[m].max()), "n": int(m.sum()),
                    "R1_gap": g1, "R1_hw": h1, "s43_gap": g2, "s43_hw": h2,
                    "r0_ade": float(d["R0"]["ade6"][m].mean())})
    return out


def stratified(d, nd=4, nf=3):
    """**이동거리를 통제하고** 보조비율로 쪼갠다 — 이 측정의 결론은 여기서 난다.

    왜 필요한가 — 보조비율이 낮은 장면은 **빠르고 멀리 가는 장면**이다(≤2% 구간의 평균 속력이
    11.5 m/s, 전체는 6.1). 그리고 (a, h) 의 열세는 **멀리 가는 장면에 몰려 있다**. 그래서
    "깨끗한 장면일수록 격차가 크다" 는 관찰은 **보조비율의 효과가 아니라 이동거리의 효과**일 수
    있다. 둘을 가르려면 이동거리를 고정한 안에서 보조비율로 나눠야 한다.
    """
    disp, f = d["R0"]["fut_disp"].astype(float), d["R0"][KEY].astype(float)
    qd = np.quantile(disp, np.linspace(0, 1, nd + 1))
    out = []
    print(f"\n── 이동거리를 통제하고 보조비율로 쪼갠 것 ({nd}×{nf}) ──")
    for s in range(nd):
        ms = (disp >= qd[s]) & (disp <= qd[s + 1] if s == nd - 1 else disp < qd[s + 1])
        qf = np.quantile(f[ms], np.linspace(0, 1, nf + 1))
        print(f"\n  이동 {qd[s]:.0f}~{qd[s+1]:.0f} m (n={int(ms.sum()):,}) · "
              f"R0 오차 {d['R0']['ade6'][ms].mean():.3f}")
        row = []
        for b_ in range(nf):
            mb = ms & (f >= qf[b_]) & (f <= qf[b_ + 1] if b_ == nf - 1 else f < qf[b_ + 1])
            g1, h1 = mean_ci(d["R1"]["ade6"][mb] - d["R0"]["ade6"][mb])
            g2, h2 = mean_ci(d["s43"]["ade6"][mb] - d["R0"]["ade6"][mb])
            print(f"    보조 {qf[b_]*100:>3.0f}~{qf[b_+1]*100:<3.0f}% n={int(mb.sum()):>5,} "
                  f"R1−R0 {g1:+.4f}±{h1:.4f} · s43−R0 {g2:+.4f}±{h2:.4f}")
            row.append({"fill_lo": float(qf[b_]), "fill_hi": float(qf[b_ + 1]),
                        "n": int(mb.sum()), "R1_gap": g1, "R1_hw": h1,
                        "s43_gap": g2, "s43_hw": h2})
        out.append({"disp_lo": float(qd[s]), "disp_hi": float(qd[s + 1]),
                    "n": int(ms.sum()), "r0_ade": float(d["R0"]["ade6"][ms].mean()),
                    "cells": row})
    return out


def clean_tail(d, ths=(0.02, 0.05, 0.10, 0.20, 0.30)):
    """보조비율 **누적 문턱** — 5분위는 가장 낮은 칸도 17% 라 '깨끗한 쪽'을 못 본다.
    (이 컷은 다른 세션이 제안했고, 그쪽 수치와 대조해 같은 값을 얻었다.)

    ⚠️ 이 컷만 보면 "깨끗할수록 격차가 크다" 가 되는데, **깨끗한 꼬리가 곧 빠른 꼬리**라
    이동거리와 엉켜 있다. 판정은 `stratified()` 로 한다.
    """
    f = d["R0"][KEY].astype(float)
    out = []
    print("\n── 보조비율 누적 문턱 (⚠️ 이동거리와 엉켜 있다) ──")
    print(f"{'문턱':>8}{'장면':>8}{'속력':>8}{'이동':>8}{'R0 오차':>9}{'R1−R0':>16}{'s43−R0':>16}")
    for th in ths:
        m = f <= th
        g1, h1 = mean_ci(d["R1"]["ade6"][m] - d["R0"]["ade6"][m])
        g2, h2 = mean_ci(d["s43"]["ade6"][m] - d["R0"]["ade6"][m])
        print(f"{th*100:>7.0f}%{int(m.sum()):>8,}{d['R0']['focal_v'][m].mean():>7.1f}m"
              f"{d['R0']['fut_disp'][m].mean():>7.0f}m{d['R0']['ade6'][m].mean():>9.3f}"
              f"{g1:>+10.4f}±{h1:.4f}{g2:>+10.4f}±{h2:.4f}")
        out.append({"th": th, "n": int(m.sum()), "R1_gap": g1, "R1_hw": h1,
                    "s43_gap": g2, "s43_hw": h2,
                    "v": float(d["R0"]["focal_v"][m].mean()),
                    "disp": float(d["R0"]["fut_disp"][m].mean()),
                    "r0_ade": float(d["R0"]["ade6"][m].mean())})
    return out


def fig(rows, path, disp=None):
    plt = C.setup_mpl()
    n = 3 if disp else 2
    axs = plt.subplots(1, n, figsize=(6.4 * n, 5.0))[1]
    fig, (axL, axR) = axs[0].figure, axs[:2]
    x = [r["fill"] * 100 for r in rows]

    for t, lab, col in (("R1", "R1 − R0  ((a, h) 로 바꾼 것)", C.C_ORANGE),
                        ("s43", "R0-s43 − R0  (시드만 바꾼 것 · 잡음 눈금)", C.C_VIOLET)):
        y = np.array([r[f"{t}_gap"] for r in rows])
        e = np.array([r[f"{t}_hw"] for r in rows])
        axL.plot(x, y, marker="o", ms=7, lw=2.4, color=col, label=lab)
        axL.fill_between(x, y - e, y + e, color=col, alpha=0.18, lw=0)
    axL.axhline(0, color=C.INK, lw=1.0)
    axL.set_xlabel("주변 차 보조비율 [%] — 오른쪽일수록 (a, h) 입력이 더 뭉개진 장면")
    axL.set_ylabel("minADE6 격차 [m]  (+ 가 나쁨)")
    axL.set_title("① 열세가 보조비율을 따라가는가", fontsize=12)
    axL.legend(fontsize=9, loc="best")

    axR.plot(x, [r["r0_ade"] for r in rows], marker="s", ms=7, lw=2.4, color=C.C_BLUE,
             label="R0 자신의 오차 — 그 버킷이 원래 얼마나 어려운가")
    axR.set_xlabel("주변 차 보조비율 [%]"); axR.set_ylabel("R0 minADE6 [m]")
    axR.set_title("② ⚠️ 교란 — 버킷마다 난이도가 다르다", fontsize=12)
    ax2 = axR.twinx()
    ax2.plot(x, [r["v"] for r in rows], marker="^", ms=6, lw=2.0, ls=(0, (4, 3)),
             color=C.C_GREEN, label="focal 평균 속력")
    ax2.set_ylabel("focal 속력 [m/s]", color=C.C_GREEN); ax2.grid(False)
    h1, l1 = axR.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")

    if disp:
        axD = axs[2]
        xd = [(r["lo"] + r["hi"]) / 2 for r in disp]
        for key, lab, col in (("R1", "R1 − R0", C.C_ORANGE), ("s43", "R0-s43 − R0", C.C_VIOLET)):
            y = np.array([r[f"{key}_gap"] for r in disp])
            e = np.array([r[f"{key}_hw"] for r in disp])
            axD.errorbar(xd, y, yerr=e, marker="o", ms=7, lw=2.4, color=col, capsize=4, label=lab)
        axD.axhline(0, color=C.INK, lw=1.0)
        axD.set_xscale("log")
        axD.set_xlabel("focal 이 미래 6초에 실제로 간 거리 [m] · 로그축")
        axD.set_ylabel("minADE6 격차 [m]")
        axD.set_title("③ 열세는 **움직이는 장면**에 몰려 있다".replace("**", ""), fontsize=12)
        axD.legend(fontsize=9, loc="best")
        axD.annotate("거의 안 움직인 장면에선\n열세가 0 과 구별되지 않는다",
                     xy=(xd[0], disp[0]["R1_gap"]), xytext=(18, 46), textcoords="offset points",
                     fontsize=9.5, color=C.INK2,
                     bbox=dict(boxstyle="round,pad=0.35", fc=C.SURF, ec=C.GRID, lw=0.7),
                     arrowprops=dict(arrowstyle="->", color=C.INK2, lw=0.9))

    fig.suptitle("입력 품질 분리 측정 — val 24,988 장면을 다시 채점해 쪼갠 것", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return C.savefig(fig, path)


def main():
    d = load()
    OUT.mkdir(parents=True, exist_ok=True)

    print("자체 검산 — 재평가 평균이 학습 로그와 맞는가")
    # ⚠️ 절대값은 **안 맞는 게 정상**이다. train.py 의 검증 DataLoader 가 drop_last=True 라
    #    학습 중 검증은 24,988판이 아니라 **24,976판**만 본다. 버려지는 마지막 12판이 우연히
    #    쉬워서(R0 오차 0.40, 전체 0.79) 로그값이 +0.0006 높게 찍힌다.
    #    (다른 세션이 찾아낸 것을 여기서 재현해 확인했다.)
    for t in TAGS:
        m, m12 = float(d[t]["ade6"].mean()), float(d[t]["ade6"][:24976].mean())
        print(f"  {t:4s} 전수 {m:.4f} · drop_last 흉내 [:24976] {m12:.4f} · 로그 {LOGGED[t]} "
              f"{'✅' if abs(m12 - LOGGED[t]) < 5e-4 else '❌'}")
    g1 = float(d["R1"]["ade6"].mean() - d["R0"]["ade6"].mean())
    g2 = float(d["s43"]["ade6"].mean() - d["R0"]["ade6"].mean())
    print(f"  → R1−R0 {g1:+.4f} (로그 +0.035) · s43−R0 {g2:+.4f} (로그 +0.006)  "
          f"**차이는 맞는다 — 공통 오프셋은 상쇄된다**")

    rows, n_na = table(d)
    print(f"\n보조비율({KEY})이 정의 안 되는 장면 {n_na}개는 뺐다 (주변 차가 없다).\n")
    print(f"{'버킷':<4}{'보조비율':>18}{'장면':>8}{'R0 오차':>9}{'속력':>7}{'주변차':>7}"
          f"{'R1−R0':>16}{'s43−R0':>16}")
    for r in rows:
        print(f"{r['i']+1:<4}{r['fill_lo']*100:>7.0f}~{r['fill_hi']*100:<4.0f}"
              f"({r['fill']*100:>4.0f}%){r['n']:>8,}{r['r0_ade']:>9.3f}{r['v']:>7.1f}"
              f"{r['n_agent']:>7.0f}"
              f"{r['R1_gap']:>+10.4f}±{r['R1_hw']:.4f}{r['s43_gap']:>+10.4f}±{r['s43_hw']:.4f}")

    lo, hi = rows[0], rows[-1]
    print(f"\n가장 깨끗한 버킷 → 가장 더러운 버킷")
    for t in ("R1", "s43"):
        a, b = lo[f"{t}_gap"], hi[f"{t}_gap"]
        ha, hb = lo[f"{t}_hw"], hi[f"{t}_hw"]
        sep = abs(b - a) > (ha + hb)
        print(f"  {t:4s} {a:+.4f} → {b:+.4f}  (변화 {b-a:+.4f}) · "
              f"오차막대 겹침 {'아니오 — 구별된다' if sep else '예 — 구별 못 한다'}")
    print(f"  난이도(R0 오차) {lo['r0_ade']:.3f} → {hi['r0_ade']:.3f} · "
          f"속력 {lo['v']:.1f} → {hi['v']:.1f} m/s")

    tail = clean_tail(d)
    strat = stratified(d)
    spd = by(d, "focal_v", label="focal 평균 속력 [m/s]")
    fut = by(d, "fut_disp", label="미래 6초 이동거리 [m]")

    (OUT / "numbers.json").write_text(json.dumps(
        {"rows": rows, "by_speed": spd, "by_disp": fut, "clean_tail": tail, "stratified": strat, "means": {t: float(d[t]["ade6"].mean()) for t in TAGS},
         "key": KEY, "n_na": n_na}, ensure_ascii=False, indent=1))
    print("\n→", fig(rows, OUT / "q1_gap_vs_fill.png", disp=fut))
    print("→", OUT / "numbers.json")


if __name__ == "__main__":
    main()
