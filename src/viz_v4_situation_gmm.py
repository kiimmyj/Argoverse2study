"""
viz_v4_situation_gmm.py - 상황을 군집으로 나눠 모델을 설명한다 (GMM + 결정 트리).

왜 이게 필요한가
    전체 minADE6 1.359 m 한 줄로는 "어떤 상황에서 무엇을 못하는지" 가 안 보인다. 규칙으로 만든
    상황 라벨(정지·좌회전·…·기타 9종)은 경계가 사람이 정한 임계라 절반 가까이가 '기타'(43%)로 빠진다.
    그래서 **모델 출력을 전혀 쓰지 않은 상황 특성 6개**로 가우시안 혼합(GMM)을 적합해 상황을 군집으로
    나누고, 군집마다 모델 지표를 따로 재서 '모델이 어디에서 어떻게 행동하나' 를 설명한다.

무엇을 쓰나 (상황 특성 — 정답 미래의 운동학. 모델 예측은 한 개도 안 들어간다)
    v0        예측 시작 속력 [m/s]           dv     6초 속도 변화 (끝 − 시작) [m/s]
    amin/amax 미래 6초 최소·최대 가속도 [m/s²]
    |Δh|      6초 진행방향 총 변화 [°]       |Δd|   기준 경로 횡변위 [m]
    꼬리 1%는 양쪽에서 자르고(p0.5·p99.5) 표준화한다. 좌·우 대칭은 크기만 써서 군집을 반으로 쪼개지 않는다.

K 는 어떻게 고르나
    70/30 으로 나눠 학습하고 보류표본 평균 로그가능도를 본다. 증가분이 0.2 nats 이상인 마지막 K 를 쓴다
    (BIC 는 K=11 까지 계속 내려가지만 군집이 쪼개지기만 하고 해석이 안 된다 — 두 곡선을 그림에 함께 남긴다).

그림 (viz/v4/<tag>/situation/)
    g1_select.png    K 선택 근거 · 군집 크기 · 소속 확신도
    g2_space.png     특성 공간에서 군집이 어디에 있나 + 군집 중심 표
    g3_metrics.png   군집별 정확도·위반율·스텝별 오차·전체 오차 기여도
    g4_behavior.png  군집별 모델 행동 (끝점 종·횡 편향, 모드 다양성, 확률, 밴드 이탈)
    g5_tree.png      군집을 재현하는 얕은 결정 트리 (서술용 규칙이지 인과가 아니다)
    g6_cases.png     군집마다 대표 시나리오 한 개 (중앙값에 가장 가까운 판)
    g7_box.png       군집별 분포 박스 플롯 — 평균 막대가 가리는 폭
    summary.json / summary.md

    python src/viz_v4_situation_gmm.py --tag v4_l4nw_ah2_full_smxy1_cos30_s0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C
import viz_v4_cases as CS

DEFAULT = "v4_l4nw_ah2_full_smxy1_cos30_s0"
FEATS = [("v0_f", "v0", "예측 시작 속력 [m/s]"),
         ("dv", "Δv", "6초 속도 변화 [m/s]"),
         ("amin_f", "a_min", "최소 가속도 [m/s²]"),
         ("amax_f", "a_max", "최대 가속도 [m/s²]"),
         ("adh6", "|Δh|", "6초 진행방향 변화 [°]"),
         ("add6", "|Δd|", "경로 횡변위 [m]")]
K_RANGE = range(2, 13)
TREE_DEPTH = 4          # 결정 트리 깊이 — 3 은 잎 8개로 정확도 59%, 4 는 12개로 65%
MIN_GAIN = 0.2          # 보류표본 로그가능도 증가분 [nats] — 이 값 이상인 마지막 K 를 쓴다
CLIP_Q = (0.5, 99.5)
SEED = C.SEED


def load(tag):
    import pandas as pd
    d = C.tag_dirs(tag)["data"]
    df = pd.read_parquet(d / "scenarios.parquet")
    df["dv"] = df["vend_f"] - df["v0_f"]
    df["adh6"] = df["dh6"].abs()
    df["add6"] = df["dd6"].abs()
    # 좌표 기준 위반율 — 같은 판을 다른 코드(src/score_xy_viol.py)로 잰 시나리오별 값
    per = np.load(C.RUNS / "v4_xy_viol_per.npz")
    if f"{tag}|ade" in per.files:
        gap = float(np.abs(per[f"{tag}|ade"] - df["minade"].to_numpy()).max())
        if gap > 1e-4:
            raise SystemExit(f"교차검증 실패 — 덤프와 score_xy_viol 의 minADE6 가 {gap:.2e} 다르다")
        df["viol"] = 100.0 * per[f"{tag}|exc"] / np.maximum(per[f"{tag}|steps"], 1)
        print(f"[교차검증] minADE6 덤프 = score_xy_viol (최대 차 {gap:.1e}) · 위반율은 top-1 모드 기준", flush=True)
    else:
        df["viol"] = np.nan
        print("[교차검증] 이 태그는 runs/v4_xy_viol_per.npz 에 없다 — 위반율 칸을 비운다", flush=True)
    return df


def features(df):
    X = df[[k for k, _, _ in FEATS]].to_numpy(np.float64)
    lo, hi = np.percentile(X, CLIP_Q[0], axis=0), np.percentile(X, CLIP_Q[1], axis=0)
    Xc = np.clip(X, lo, hi)
    mu, sd = Xc.mean(0), Xc.std(0)
    return Xc, (Xc - mu) / sd, {"clip_lo": lo.tolist(), "clip_hi": hi.tolist(),
                                "mean": mu.tolist(), "std": sd.tolist()}


def select_k(Z):
    """K 후보마다 (BIC, 보류표본 평균 로그가능도) 를 내고 규칙으로 K 를 고른다."""
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Z))
    tr, te = idx[:int(0.7 * len(Z))], idx[int(0.7 * len(Z)):]
    rows = []
    for k in K_RANGE:
        g = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=3).fit(Z[tr])
        rows.append({"k": k, "bic": float(g.bic(Z[tr])), "holdout": float(g.score(Z[te]))})
        print(f"[K 탐색] K={k:2d}  BIC {rows[-1]['bic']:10.0f}  보류 로그가능도 {rows[-1]['holdout']:7.4f}", flush=True)
    for i in range(1, len(rows)):
        rows[i]["gain"] = rows[i]["holdout"] - rows[i - 1]["holdout"]
    best = max([r["k"] for r in rows[1:] if r["gain"] >= MIN_GAIN], default=rows[0]["k"])
    return best, rows


def name_cluster(c):
    """군집 중심(원래 단위) → 이름. 규칙은 공개한다 — 사람이 붙인 이름이지 모델이 준 것이 아니다."""
    v0, dv, amin, amax, adh = c["v0"], c["Δv"], c["a_min"], c["a_max"], c["|Δh|"]
    if v0 < 1.5 and dv > 3.0:
        return "정지 → 출발"
    if v0 < 1.5:
        return "저속 배회·정체"
    if dv < -1.5 and adh < C.TURN_DEG:
        return "감속 → 정지"
    if adh >= C.TURN_DEG:
        return "급가감속 섞인 회전" if (amax >= 3.0 or amin <= -2.0) else "회전"
    if adh >= 5.0:
        return "완만한 곡선"
    if amax >= 1.5 or amin <= -1.5:
        return "직진 가감속"
    return "정속 순항"


def fit(Z, k):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=5).fit(Z)
    post = g.predict_proba(Z)
    return g, post.argmax(1), post.max(1)


def stability(Z, k, lab):
    """시드를 바꿔 다시 적합했을 때 같은 묶음이 나오나 (조정 랜드 지수)."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    out = {}
    for s in (SEED + 1, SEED + 2):
        g = GaussianMixture(k, covariance_type="full", random_state=s, n_init=5).fit(Z)
        out[f"seed{s}"] = round(float(adjusted_rand_score(lab, g.predict(Z))), 3)
    return out


def summarize(df, X, lab, conf, k, win_err):
    """군집마다 상황 중심 + 모델 지표. 순서는 크기 내림차순."""
    rows = []
    N = len(df)
    tot = df["minade"].mean()
    for c in np.argsort(-np.bincount(lab, minlength=k)):
        m = lab == c
        d = df[m]
        cen = {s: float(X[m, j].mean()) for j, (_, s, _) in enumerate(FEATS)}
        rows.append({
            "cluster": int(c), "n": int(m.sum()), "pct": round(100.0 * m.mean(), 2),
            "name": name_cluster(cen), "conf": round(float(conf[m].mean()), 3),
            "center": {s: round(v, 2) for s, v in cen.items()},
            "minade": round(float(d["minade"].mean()), 3),
            "top1_ade": round(float(d["top1_ade"].mean()), 3),
            "miss_pct": round(100.0 * float(d["miss"].mean()), 1),
            "viol_pct": round(float(d["viol"].mean()), 2),
            "offlane": round(float(d["offlane"].mean()), 3),
            "ds_end": round(float(d["ds_end"].median()), 2),
            "dd_end": round(float(d["dd_end"].median()), 2),
            "n_end_clusters": round(float(d["n_end_clusters"].mean()), 2),
            "top1_prob": round(float(d["top1_prob"].mean()), 3),
            "route_err_pct": round(100.0 * float(d["route_err"].mean()), 1),
            "share_of_error": round(100.0 * float(d["minade"].sum() / (tot * N)), 1),
            "err_curve": win_err[m].mean(0).round(3).tolist(),
            "top_cls": {k2: int(v) for k2, v in d["cls"].value_counts().head(3).items()},
        })
    return rows


def box_panel(ax, data, labels, colors, ylab, title, ref=None, ref_lab=None, zero=False):
    """박스 플롯 한 칸 — 상자 = 사분위(Q1~Q3), 가운데 선 = 중앙값, ◆ = 평균, 수염 = 5~95 백분위.
    바깥값(5% 미만·95% 초과)은 점으로 찍지 않는다 — 2만 5천 개를 찍으면 상자가 안 보인다."""
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, showmeans=True, patch_artist=True,
                    widths=0.62, meanprops=dict(marker="D", ms=4.2, mfc=C.SURF, mec=C.INK, mew=0.9),
                    medianprops=dict(color=C.INK, lw=1.6),
                    whiskerprops=dict(color=C.AXIS, lw=1.0), capprops=dict(color=C.AXIS, lw=1.0))
    for b, c in zip(bp["boxes"], colors):
        b.set(facecolor=c, alpha=0.55, edgecolor=C.AXIS, lw=0.9)
    if ref is not None:
        ax.axhline(ref, color=C.INK2, lw=1.1, ls=(0, (4, 2)))
        if ref_lab:
            ax.text(0.55, ref, f" {ref_lab}", fontsize=7.6, color=C.INK2, va="bottom", ha="left")
    if zero:
        ax.axhline(0, color=C.AXIS, lw=1.0)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5)
    return bp


def g1_select(rows_k, kbest, lab, conf, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14.6, 9.2))
    ks = [r["k"] for r in rows_k]
    ax[0][0].plot(ks, [r["bic"] for r in rows_k], color=C.C_BLUE, marker="o", ms=5, mec=C.SURF)
    ax[0][0].set_xlabel("군집 수 K"); ax[0][0].set_ylabel("BIC (학습 70%) — 낮을수록 좋다")
    ax[0][0].set_title("BIC 는 K=11 까지 계속 내려간다", fontsize=10.5)
    ax[0][1].plot(ks, [r["holdout"] for r in rows_k], color=C.C_ORANGE, marker="o", ms=5, mec=C.SURF)
    for r in rows_k[1:]:
        ax[0][1].annotate(f"{r['gain']:+.2f}", (r["k"], r["holdout"]), textcoords="offset points",
                          xytext=(0, -14), ha="center", fontsize=7.2, color=C.INK2)
    ax[0][1].axvline(kbest, color=C.C_GREEN, lw=1.4, ls=(0, (4, 2)))
    ax[0][1].text(kbest, ax[0][1].get_ylim()[0], f" K={kbest} 선택", color=C.C_GREEN, fontsize=9, va="bottom")
    ax[0][1].set_xlabel("군집 수 K"); ax[0][1].set_ylabel("보류표본 평균 로그가능도 — 높을수록 좋다")
    ax[0][1].set_title(f"증가분(숫자)이 {MIN_GAIN} nats 이상인 마지막 K", fontsize=10.5)
    cnt = np.bincount(lab, minlength=kbest)
    o = np.argsort(-cnt)
    ax[1][0].bar(range(kbest), 100.0 * cnt[o] / len(lab), color=C.C_BLUE, edgecolor=C.SURF)
    for j, c in enumerate(o):
        ax[1][0].text(j, 100.0 * cnt[c] / len(lab), f"{cnt[c]:,}", ha="center", va="bottom", fontsize=7.6,
                      color=C.INK2)
    ax[1][0].set_xticks(range(kbest)); ax[1][0].set_xticklabels([f"C{c}" for c in o])
    ax[1][0].set_xlabel("군집"); ax[1][0].set_ylabel("비중 [%]")
    ax[1][0].set_title("군집 크기", fontsize=10.5)
    ax[1][1].hist(conf, bins=40, color=C.C_AQUA, edgecolor=C.SURF)
    ax[1][1].axvline(0.6, color=C.C_ORANGE, lw=1.2, ls=(0, (4, 2)))
    ax[1][1].text(0.6, ax[1][1].get_ylim()[1] * 0.95, f" 0.6 미만 {100.0 * (conf < 0.6).mean():.1f}%",
                  color=C.C_ORANGE, fontsize=8.5, va="top")
    ax[1][1].set_xlabel("소속 확신도 (가장 큰 사후확률)"); ax[1][1].set_ylabel("시나리오 수")
    ax[1][1].set_title("경계에 있는 시나리오 — 규칙 분류로는 못 적는 값", fontsize=10.5)
    fig.suptitle("① 군집을 몇 개로 나눌 것인가 — 선택 근거", fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)


def g2_space(X, lab, rows, out):
    import matplotlib.pyplot as plt
    pairs = [(0, 4, "v0 [m/s]", "|Δh| [°]"), (0, 1, "v0 [m/s]", "Δv [m/s]"),
             (2, 3, "a_min [m/s²]", "a_max [m/s²]"), (4, 5, "|Δh| [°]", "|Δd| [m]")]
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(X), size=min(7000, len(X)), replace=False)
    cols = {r["cluster"]: C.SERIES[j % len(C.SERIES)] for j, r in enumerate(rows)}
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.6))
    for ax, (a, b, xl, yl) in zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]], pairs):
        for r in rows:
            m = lab[sub] == r["cluster"]
            ax.scatter(X[sub][m, a], X[sub][m, b], s=4, color=cols[r["cluster"]], alpha=0.45, lw=0)
        for r in rows:
            cen = list(r["center"].values())
            ax.scatter(cen[a], cen[b], s=90, color=cols[r["cluster"]], edgecolors=C.INK, linewidths=1.0,
                       zorder=5)
            ax.annotate(f"C{r['cluster']}", (cen[a], cen[b]), textcoords="offset points", xytext=(6, 5),
                        fontsize=8.6, color=C.INK, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes[0][2].axis("off")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cols[r["cluster"]], marker="o", ls="none", ms=8) for r in rows]
    axes[0][2].legend(h, [f"C{r['cluster']} {r['name']} · {r['pct']:.1f}%" for r in rows], loc="center left",
                      fontsize=9.6, frameon=True, facecolor="white", edgecolor=C.GRID,
                      title="군집 (크기 순)", title_fontsize=10.5)
    lines = ["읽는 방법", "점 하나가 시나리오 하나다(7,000개만 그렸다). 색이 군집, 테두리 큰 점이 군집 중심이다.",
             "축은 모두 **정답 쪽 상황 값**이다 — 모델 예측은 군집을 만드는 데 한 개도 안 들어갔다.", "",
             "군집 중심 (원래 단위)"]
    lines.append("   " + "군집".ljust(22) + "".join(s.rjust(9) for _, s, _ in FEATS))
    for r in rows:
        lines.append("   " + f"C{r['cluster']} {r['name']}".ljust(22)
                     + "".join(f"{v:9.2f}" for v in r["center"].values()))
    lines += ["", "자주 섞여 있는 규칙 라벨(상위 3)"]
    for r in rows:
        lines.append(f"   C{r['cluster']} {r['name']}: "
                     + ", ".join(f"{k2} {v:,}" for k2, v in r["top_cls"].items()))
    axes[1][2].axis("off")
    axes[1][2].text(0.0, 1.0, "\n".join(lines), transform=axes[1][2].transAxes, va="top", ha="left",
                    fontsize=7.6, family="Noto Sans Mono CJK KR", color=C.INK, linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C.GRID, lw=0.8))
    fig.suptitle("② 군집이 특성 공간 어디에 있나", fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)


def g3_metrics(rows, tot, out):
    import matplotlib.pyplot as plt
    labs = [f"C{r['cluster']}\n{r['pct']:.1f}%" for r in rows]      # 이름은 오른쪽 표·범례에 둔다 (칸이 좁다)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.2))
    ax = axes[0][0]
    ax.bar(x - 0.2, [r["minade"] for r in rows], 0.38, color=C.C_BLUE, label="minADE6 (6모드 중 최선)")
    ax.bar(x + 0.2, [r["top1_ade"] for r in rows], 0.38, color=C.C_ORANGE, label="top-1 ADE (확률 1위 모드)")
    ax.axhline(tot["minade"], color=C.C_BLUE, lw=1.1, ls=(0, (4, 2)))
    ax.axhline(tot["top1_ade"], color=C.C_ORANGE, lw=1.1, ls=(0, (4, 2)))
    ax.set_ylabel("평균 거리오차 [m]"); ax.legend(fontsize=8.2)
    ax.set_title(f"정확도 — 점선은 전체 평균 ({tot['minade']:.2f} / {tot['top1_ade']:.2f} m)", fontsize=10.5)
    ax = axes[0][1]
    ax.bar(x - 0.2, [r["miss_pct"] for r in rows], 0.38, color=C.C_VIOLET, label="miss율 (minFDE6 > 2 m)")
    ax.bar(x + 0.2, [r["viol_pct"] for r in rows], 0.38, color=C.C_GREEN, label="좌표 기준 위반율 [%]")
    ax.axhline(tot["miss_pct"], color=C.C_VIOLET, lw=1.1, ls=(0, (4, 2)))
    ax.axhline(tot["viol_pct"], color=C.C_GREEN, lw=1.1, ls=(0, (4, 2)))
    ax.set_ylabel("[%]"); ax.legend(fontsize=8.2)
    ax.set_title("놓침과 실현가능성", fontsize=10.5)
    ax = axes[1][0]
    t = np.arange(1, 61) * C.DT
    for j, r in enumerate(rows):
        ax.plot(t, r["err_curve"], color=C.SERIES[j % len(C.SERIES)], lw=1.6,
                marker="o", ms=2.6, mec=C.SURF, mew=0.3, label=f"C{r['cluster']} {r['name']}")
    ax.set_xlabel("예측 시각 [s]"); ax.set_ylabel("승자 모드의 거리오차 [m]")
    ax.legend(fontsize=7.4, ncol=2); ax.set_title("오차가 언제 벌어지나 (스텝마다 점)", fontsize=10.5)
    ax = axes[1][1]
    ax.bar(x, [r["share_of_error"] for r in rows], 0.6, color=C.C_AQUA, label="전체 오차 기여 [%]")
    ax.plot(x, [r["pct"] for r in rows], color=C.C_RED, marker="o", ms=5, mec=C.SURF, lw=1.4,
            label="시나리오 비중 [%]")
    ax.set_ylabel("[%]"); ax.legend(fontsize=8.2)
    ax.set_title("전체 minADE6 를 누가 만드는가", fontsize=10.5)
    for a in (axes[0][0], axes[0][1], axes[1][1]):
        a.set_xticks(x); a.set_xticklabels(labs, fontsize=8.6)
    axes[0][2].axis("off"); axes[1][2].axis("off")
    big = max(rows, key=lambda r: r["share_of_error"])
    worst = max(rows, key=lambda r: r["minade"])
    best = min(rows, key=lambda r: r["minade"])
    axes[0][2].text(0.0, 1.0, "\n".join([
        "읽는 방법",
        "가로축은 ①②에서 만든 군집이다(괄호 = 비중). 막대 네 칸은 모두 같은 판·같은 val 24,988 이다.",
        "점선은 전체 평균이다. 막대가 점선보다 높으면 그 상황이 평균보다 어렵다는 뜻이다.",
        "",
        "minADE6 = 6개 모드 중 정답에 가장 가까운 모드의 평균 거리오차.",
        "top-1 ADE = 모델이 확률 1위로 고른 모드의 오차. 둘의 간격이 크면 '정답을 모드 안에 갖고는",
        "   있는데 고르지 못한다' 는 뜻이다.",
        "miss율 = 끝점 오차가 2 m 를 넘은 비율. 좌표 위반율 = |Δψ| > 7.3°/step 인 스텝 비율.",
        "",
        "이 판에서 읽히는 것",
        f"   가장 정확: C{best['cluster']} {best['name']} {best['minade']:.2f} m",
        f"   가장 어려움: C{worst['cluster']} {worst['name']} {worst['minade']:.2f} m "
        f"(miss {worst['miss_pct']:.0f}%)",
        f"   오차 기여 1위: C{big['cluster']} {big['name']} — 시나리오의 {big['pct']:.1f}% 인데",
        f"      전체 오차의 {big['share_of_error']:.1f}% 를 만든다",
    ]), transform=axes[0][2].transAxes, va="top", ha="left", fontsize=8.8, color=C.INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=C.GRID, lw=0.8))
    hdr = ["군집", "n", "minADE6", "top-1", "miss%", "위반%", "기여%"]
    tbl = ["   " + hdr[0].ljust(22) + "".join(h.rjust(9) for h in hdr[1:])]
    for r in rows:
        tbl.append("   " + f"C{r['cluster']} {r['name']}".ljust(22)
                   + f"{r['n']:9,}{r['minade']:9.2f}{r['top1_ade']:9.2f}{r['miss_pct']:9.1f}"
                     f"{r['viol_pct']:9.2f}{r['share_of_error']:9.1f}")
    tbl.append("   " + "전체".ljust(22) + f"{tot['n']:9,}{tot['minade']:9.2f}{tot['top1_ade']:9.2f}"
                                          f"{tot['miss_pct']:9.1f}{tot['viol_pct']:9.2f}{100.0:9.1f}")
    axes[1][2].text(0.0, 1.0, "\n".join(["군집별 숫자"] + tbl), transform=axes[1][2].transAxes, va="top",
                    ha="left", fontsize=7.8, family="Noto Sans Mono CJK KR", color=C.INK, linespacing=1.6,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C.GRID, lw=0.8))
    fig.suptitle("③ 군집별 정확도 — 전체 한 줄로는 안 보이던 것", fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)


def g4_behavior(rows, tot, out):
    import matplotlib.pyplot as plt
    labs = [f"C{r['cluster']}\n{r['pct']:.1f}%" for r in rows]      # 이름은 오른쪽 표·범례에 둔다 (칸이 좁다)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.2))
    ax = axes[0][0]
    ax.bar(x - 0.2, [r["ds_end"] for r in rows], 0.38, color=C.C_BLUE, label="종방향 ds (앞뒤)")
    ax.bar(x + 0.2, [r["dd_end"] for r in rows], 0.38, color=C.C_MAGENTA, label="횡방향 dd (좌우)")
    ax.axhline(0, color=C.AXIS, lw=1.0)
    ax.set_ylabel("승자 모드 끝점 − 정답 끝점 [m] (중앙값)"); ax.legend(fontsize=8.2)
    ax.set_title("어느 쪽으로 틀리나 — 음수 = 정답보다 덜 갔다", fontsize=10.5)
    ax = axes[0][1]
    ax.bar(x - 0.2, [r["n_end_clusters"] for r in rows], 0.38, color=C.C_AQUA, label="서로 다른 끝점 묶음 수")
    ax.axhline(tot["n_end_clusters"], color=C.C_AQUA, lw=1.1, ls=(0, (4, 2)))
    a2 = ax.twinx()
    a2.plot(x, [r["top1_prob"] for r in rows], color=C.C_ORANGE, marker="o", ms=5, mec=C.SURF, lw=1.4,
            label="1위 모드 확률")
    a2.set_ylabel("1위 모드 확률", color=C.C_ORANGE); a2.grid(False)
    ax.set_ylabel("끝점 묶음 수 (최대 6)"); ax.legend(fontsize=8.2, loc="upper left")
    a2.legend(fontsize=8.2, loc="upper right")
    ax.set_title("모드를 몇 갈래로 벌리나 · 얼마나 확신하나", fontsize=10.5)
    ax = axes[1][0]
    ax.bar(x - 0.2, [r["offlane"] for r in rows], 0.38, color=C.C_GREEN, label="밴드 이탈 [step]")
    ax.bar(x + 0.2, [r["route_err_pct"] / 10.0 for r in rows], 0.38, color=C.C_YELLOW,
           label="1위 모드가 정답 경로를 못 고른 비율 [%÷10]")
    ax.axhline(tot["offlane"], color=C.C_GREEN, lw=1.1, ls=(0, (4, 2)))
    ax.set_ylabel("밴드 이탈 [step] / 경로 오선택 [%÷10]"); ax.legend(fontsize=8.2)
    ax.set_title("규칙 쪽 — 밴드와 경로 선택", fontsize=10.5)
    for a in (axes[0][0], axes[0][1], axes[1][0]):
        a.set_xticks(x); a.set_xticklabels(labs, fontsize=8.6)
    stop = min(rows, key=lambda r: r["n_end_clusters"])     # 모드를 가장 적게 벌리는 군집
    turn = max(rows, key=lambda r: r["center"]["|Δh|"])
    axes[0][2].axis("off"); axes[1][1].axis("off"); axes[1][2].axis("off")
    axes[0][2].text(0.0, 1.0, "\n".join([
        "읽는 방법",
        "왼쪽 위: 승자 모드의 6초 뒤 끝점이 정답 끝점에서 어느 쪽으로 어긋났나(중앙값).",
        "   ds 가 음수면 정답보다 **덜 나아간** 것이고, dd 가 음수면 정답보다 오른쪽이다.",
        "가운데 위: 6개 모드의 끝점을 2.5 m 안이면 한 묶음으로 세어 '진짜 몇 갈래를 내놓나' 를 본다.",
        "   주황 선은 1위 모드에 준 확률이다. 묶음이 적은데 확률이 높으면 '한 갈래를 확신' 하는 상태다.",
        "왼쪽 아래: 밴드(허용 횡오프셋)를 벗어난 스텝 수와, 1위 모드가 정답이 간 후보 경로를 못 고른 비율.",
        "",
        "이 판에서 읽히는 것",
        f"   C{turn['cluster']} {turn['name']}: 끝점이 종방향으로 {turn['ds_end']:+.2f} m — 회전에서 덜 나아간다.",
        f"   C{stop['cluster']} {stop['name']}: 끝점 묶음 {stop['n_end_clusters']:.1f} 개 · 1위 확률 "
        f"{stop['top1_prob']:.2f} — 거의 한 갈래만 내놓는다.",
        "   모드 다양성과 확률은 상황에 따라 크게 다르다. 전체 평균 하나로는 '모드가 죽었다/살았다' 를",
        "   말할 수 없다.",
    ]), transform=axes[0][2].transAxes, va="top", ha="left", fontsize=8.8, color=C.INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=C.GRID, lw=0.8))
    fig.suptitle("④ 군집별 모델 행동 — 어느 쪽으로, 몇 갈래로, 얼마나 확신하며 틀리나",
                 fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)


def g5_tree(X, lab, rows, out):
    """군집을 재현하는 얕은 결정 트리 — 서술용 규칙(필수 요건 11)."""
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    names = [s for _, s, _ in FEATS]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    tr, te = idx[:int(0.7 * len(X))], idx[int(0.7 * len(X)):]
    est = DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=200, random_state=SEED).fit(X[tr], lab[tr])
    acc = float((est.predict(X[te]) == lab[te]).mean())
    nm = {r["cluster"]: r["name"] for r in rows}

    def val(t, j):
        v = t.value[j][0]
        return float(v.max() / max(v.sum(), 1e-9))

    def txt(n, v):
        return f"n {n:,}\n순도 {100 * v:.0f}%"

    root = C.sk_tree_nodes(est, names, val, txt, thr_fmt=lambda name, x: f"{x:.1f}")
    for nd in C.tree_leaves(root):
        j = nd["id"]
        v = est.tree_.value[j][0]
        c = est.classes_[int(v.argmax())]
        nd["text"] = f"C{c} {nm[int(c)]}\nn {nd['n']:,} · 순도 {100 * nd['value']:.0f}%"
    fig, axes = plt.subplots(2, 1, figsize=(19.0, 11.4), height_ratios=[2.2, 1.0])
    C.draw_tree(axes[0], root, fontsize=7.4, edge_fs=7.2)
    axes[0].set_title(f"상황 특성 → 군집 (sklearn DecisionTreeClassifier · 깊이 {TREE_DEPTH} · 잎 ≥ 200 · "
                      f"학습 70% · 검증 정확도 {100 * acc:.1f}%)", fontsize=11)
    axes[1].axis("off")
    axes[1].text(0.0, 1.0, "\n".join([
        "읽는 방법",
        "위 트리는 GMM 이 만든 군집을 **사람이 읽을 수 있는 임계로 다시 적은 것**이다. 위에서 아래로 내려가며",
        "조건에 맞는 쪽으로 가면 군집이 나온다. 잎 상자의 '순도' 는 그 잎에 모인 시나리오 중 적힌 군집의 비율이다.",
        "",
        f"검증 정확도 {100 * acc:.1f}% 는 '깊이 {TREE_DEPTH} 규칙으로 군집의 {100 * acc:.0f}% 를 재현한다' 는 뜻이다.",
        "잎에 한 번도 안 나오는 군집이 있으면 그 군집은 이 임계들만으로는 떼어지지 않는 것이다",
        "(예: 직진 가감속 — 속도·회전만 보면 정속 순항과 겹치고 가속도 축에서만 갈린다).",
        "100% 가 아닌 이유는 GMM 이 경계를 확률로 두기 때문이다 — 트리는 서술용 요약이지 군집의 정의가 아니다.",
        "",
        "이것은 **상관 관계를 요약한 서술**이지 인과가 아니다. '속력이 낮아서 오차가 작다' 가 아니라",
        "'속력이 낮은 묶음에서 오차가 작게 관측된다' 로 읽어야 한다.",
        "",
        "쓰는 법: 새 시나리오를 군집에 넣을 때 GMM 모델 파일 없이도 이 규칙으로 대략 분류할 수 있다.",
        "   평가군을 만들거나(예: 회전 전용 평가), 실패 사례를 묶어 볼 때 쓴다.",
    ]), transform=axes[1].transAxes, va="top", ha="left", fontsize=9.2, color=C.INK, linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=C.GRID, lw=0.8))
    fig.suptitle("⑤ 군집을 규칙으로 다시 적으면", fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)
    return {"acc": round(acc, 3), "rules": [{"cluster": int(est.classes_[int(est.tree_.value[nd["id"]][0].argmax())]),
                                             "n": nd["n"], "purity": round(nd["value"], 3),
                                             "rule": nd["rule"]} for nd in C.tree_leaves(root)]}


def g6_cases(cx, df, lab, rows, out):
    """군집마다 대표 시나리오 한 개 — minADE6 가 그 군집 중앙값에 가장 가까운 판."""
    import matplotlib.pyplot as plt
    n = len(rows)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 5.0 * nrow))
    # 지도 시야는 칸의 실제 가로세로비로 정해진다 — 자리를 먼저 확정하고 그린다(나중에 바꾸면 여백이 남는다)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.865, bottom=0.015, wspace=0.07, hspace=0.14)
    picks = []
    for ax, r in zip(axes.flat, rows):
        m = np.where(lab == r["cluster"])[0]
        v = df["minade"].to_numpy()[m]
        i = int(m[np.argmin(np.abs(v - np.median(v)))])
        picks.append({"cluster": r["cluster"], "idx": i, "sid": df["sid"].iloc[i],
                      "minade": round(float(df["minade"].iloc[i]), 3), "cls": df["cls"].iloc[i]})
        CS.draw_map(ax, cx, i, compact=True)
        ax.set_title(f"C{r['cluster']} {r['name']} ({r['pct']:.1f}%)\n{df['sid'].iloc[i][:8]} · 규칙 라벨 "
                     f"{df['cls'].iloc[i]} · minADE6 {df['minade'].iloc[i]:.2f} m (군집 중앙값 "
                     f"{np.median(v):.2f})", fontsize=9.2)
    for ax in axes.flat[n:]:
        ax.axis("off")
    h, l = CS.legend_items(cx.hz)
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.012, 0.962), ncol=6, fontsize=8.0, frameon=True,
               facecolor="white", edgecolor=C.GRID, handlelength=2.2, columnspacing=1.3)
    fig.suptitle("⑥ 군집마다 대표 시나리오 — minADE6 가 군집 중앙값에 가장 가까운 판", fontsize=14,
                 x=0.012, ha="left", y=0.985)
    C.savefig(fig, out)
    return picks


def g7_box(df, lab, rows, tot, out):
    """군집별 분포 — 막대(평균)가 가리는 폭을 본다."""
    import matplotlib.pyplot as plt
    cols = [C.SERIES[j % len(C.SERIES)] for j in range(len(rows))]
    labels = [f"C{r['cluster']}\n{r['pct']:.1f}%" for r in rows]
    sel = [lab == r["cluster"] for r in rows]
    panels = [("minade", "minADE6 [m]", "정확도 — 6모드 중 최선", tot["minade"], "전체 평균", False),
              ("top1_ade", "top-1 ADE [m]", "확률 1위 모드의 오차", tot["top1_ade"], "전체 평균", False),
              ("ds_end", "종방향 ds [m]", "끝점이 앞뒤로 얼마나 어긋나나 (− = 덜 나아감)", None, None, True),
              ("dd_end", "횡방향 dd [m]", "끝점이 좌우로 얼마나 어긋나나 (− = 오른쪽)", None, None, True)]
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.2))
    cells = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    stat = {}
    for ax, (col, ylab, title, ref, rlab, zero) in zip(cells, panels):
        data = [df[col].to_numpy()[m] for m in sel]
        box_panel(ax, data, labels, cols, ylab, title, ref, rlab, zero)
        if col in ("minade", "top1_ade"):
            ax.set_ylim(0, np.percentile(np.concatenate(data), 97))
        else:
            lo, hi = np.percentile(np.concatenate(data), [2, 98])
            ax.set_ylim(lo, hi)
        stat[col] = [{"cluster": int(r["cluster"]),
                      "p25": round(float(np.percentile(d, 25)), 3),
                      "p50": round(float(np.percentile(d, 50)), 3),
                      "p75": round(float(np.percentile(d, 75)), 3),
                      "p95": round(float(np.percentile(d, 95)), 3),
                      "mean": round(float(d.mean()), 3)} for r, d in zip(rows, data)]
    axes[0][2].axis("off"); axes[1][2].axis("off")
    md = {r["cluster"]: s_ for r, s_ in zip(rows, stat["minade"])}
    worst = max(rows, key=lambda r: md[r["cluster"]]["p95"])
    skew = max(rows, key=lambda r: md[r["cluster"]]["mean"] - md[r["cluster"]]["p50"])
    axes[0][2].text(0.0, 1.0, "\n".join([
        "읽는 방법",
        "상자 = 가운데 50%(Q1~Q3), 상자 안 선 = 중앙값, ◆ = 평균, 수염 = 5~95 백분위입니다.",
        "바깥값은 점으로 찍지 않았습니다 — 2만 5천 개를 찍으면 상자가 보이지 않습니다.",
        "③·④의 막대는 **평균 하나**였습니다. 이 그림은 같은 수치의 **폭**을 봅니다.",
        "",
        "왜 보는가 — 평균은 꼬리에 끌려갑니다.",
        f"   {'C%d %s' % (skew['cluster'], skew['name'])}: 중앙값 {md[skew['cluster']]['p50']:.2f} m 인데 "
        f"평균 {md[skew['cluster']]['mean']:.2f} m 입니다.",
        "   즉 '평균적으로 그 정도 틀린다' 가 아니라 '대부분은 더 잘 맞고 일부가 크게 틀린다' 입니다.",
        f"   {'C%d %s' % (worst['cluster'], worst['name'])}: 상위 5%가 {md[worst['cluster']]['p95']:.1f} m 를 넘습니다.",
        "",
        "끝점 편차(아래 두 칸)는 상자가 0 선을 어느 쪽으로 넘는지를 봅니다.",
        "   상자 전체가 0 아래면 '가끔' 이 아니라 **체계적으로** 덜 나아가는 것입니다.",
    ]), transform=axes[0][2].transAxes, va="top", ha="left", fontsize=8.8, color=C.INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=C.GRID, lw=0.8))
    tb = ["   " + "군집".ljust(22) + "".join(h.rjust(8) for h in ("p25", "중앙값", "p75", "p95", "평균"))]
    for r, d in zip(rows, stat["minade"]):
        tb.append("   " + f"C{r['cluster']} {r['name']}".ljust(22)
                  + f"{d['p25']:8.2f}{d['p50']:8.2f}{d['p75']:8.2f}{d['p95']:8.2f}{d['mean']:8.2f}")
    allm = df["minade"].to_numpy()
    tb.append("   " + "전체".ljust(22) + "".join(f"{v:8.2f}" for v in (
        np.percentile(allm, 25), np.percentile(allm, 50), np.percentile(allm, 75),
        np.percentile(allm, 95), allm.mean())))
    axes[1][2].text(0.0, 1.0, "\n".join(["minADE6 분포 [m]"] + tb), transform=axes[1][2].transAxes,
                    va="top", ha="left", fontsize=7.8, family="Noto Sans Mono CJK KR", color=C.INK,
                    linespacing=1.6, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C.GRID, lw=0.8))
    fig.suptitle("⑦ 군집별 분포 — 평균 막대가 가리는 폭", fontsize=14, x=0.015, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    C.savefig(fig, out)
    return stat


def write_md(tag, kbest, rows, tot, stab, tree, picks, path):
    L = [f"# 상황 군집 요약 — {tag}", "",
         f"- val 24,988 · GMM K={kbest} (full covariance · 시드 {SEED}) · 특성 6개는 정답 쪽 상황 값만 쓴다",
         f"- 군집 안정성(조정 랜드 지수, 시드 바꿔 재적합): {stab}",
         f"- 깊이 3 결정 트리로 군집을 재현한 검증 정확도: {100 * tree['acc']:.1f}%", "",
         "| 군집 | 비중 | v0 [m/s] | Δv | \\|Δh\\| [°] | minADE6 | top-1 ADE | miss% | 위반% | 기여% | 해석 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["center"]
        why = []
        if r["minade"] > tot["minade"] * 1.2:
            why.append("평균보다 어렵다")
        if r["minade"] < tot["minade"] * 0.8:
            why.append("평균보다 쉽다")
        if r["top1_ade"] > 2.0 * r["minade"]:
            why.append("정답을 모드에 갖고도 못 고른다")
        if r["n_end_clusters"] < 2.5:
            why.append("모드가 한 갈래로 모인다")
        if r["ds_end"] < -0.2:
            why.append("종방향으로 덜 나아간다")
        L.append(f"| C{r['cluster']} {r['name']} | {r['pct']:.1f}% | {c['v0']:.1f} | {c['Δv']:+.1f} | "
                 f"{c['|Δh|']:.0f} | {r['minade']:.2f} | {r['top1_ade']:.2f} | {r['miss_pct']:.0f} | "
                 f"{r['viol_pct']:.2f} | {r['share_of_error']:.1f} | " + " · ".join(why or ["—"]) + " |")
    L += ["", f"| 전체 | 100% | | | | {tot['minade']:.2f} | {tot['top1_ade']:.2f} | {tot['miss_pct']:.0f} | "
              f"{tot['viol_pct']:.2f} | 100 | |", "",
          "## 대표 시나리오", ""]
    for p in picks:
        L.append(f"- C{p['cluster']} — `{p['sid']}` (규칙 라벨 {p['cls']}, minADE6 {p['minade']:.2f} m)")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT)
    ap.add_argument("--k", type=int, default=None, help="K 를 직접 지정 (기본은 보류표본 규칙)")
    a = ap.parse_args()
    C.setup_mpl()
    out = C.tag_dirs(a.tag)["base"] / "situation"
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.tag)
    X, Z, scal = features(df)
    kbest, rows_k = select_k(Z)
    if a.k:
        kbest = a.k
    print(f"[K] {kbest} 선택", flush=True)
    g, lab, conf = fit(Z, kbest)
    stab = stability(Z, kbest, lab)
    print(f"[안정성] 조정 랜드 지수 {stab}", flush=True)

    d = C.tag_dirs(a.tag)["data"]
    pred = np.load(d / "pred.npz")
    y = np.load(d / "raw.npz")["pos"][:, C.OBS:]
    i = np.arange(len(df))
    win = pred["traj"][i, df["winner"].to_numpy().astype(int)]
    win_err = np.linalg.norm(win - y, axis=-1)                      # (N,60) 스텝별 거리오차
    rows = summarize(df, X, lab, conf, kbest, win_err)
    tot = {"n": len(df), "minade": float(df["minade"].mean()), "top1_ade": float(df["top1_ade"].mean()),
           "miss_pct": 100.0 * float(df["miss"].mean()), "viol_pct": float(df["viol"].mean()),
           "offlane": float(df["offlane"].mean()), "n_end_clusters": float(df["n_end_clusters"].mean())}

    g1_select(rows_k, kbest, lab, conf, out / "g1_select.png")
    g2_space(X, lab, rows, out / "g2_space.png")
    g3_metrics(rows, tot, out / "g3_metrics.png")
    g4_behavior(rows, tot, out / "g4_behavior.png")
    tree = g5_tree(X, lab, rows, out / "g5_tree.png")
    box = g7_box(df, lab, rows, tot, out / "g7_box.png")
    picks = g6_cases(CS.Ctx(a.tag), df, lab, rows, out / "g6_cases.png")
    np.save(out / "labels.npy", lab)
    (out / "summary.json").write_text(json.dumps(
        {"tag": a.tag, "n": len(df), "k": kbest, "k_search": rows_k, "min_gain_nats": MIN_GAIN,
         "features": [{"col": c, "short": s, "desc": t} for c, s, t in FEATS], "scaling": scal,
         "stability_ari": stab, "tree": tree, "overall": tot, "clusters": rows, "cases": picks,
         "box": box},
        indent=2, ensure_ascii=False, default=float))
    write_md(a.tag, kbest, rows, tot, stab, tree, picks, out / "summary.md")
    print(f"[done] {out} — 그림 6장 · summary.json · summary.md", flush=True)


if __name__ == "__main__":
    main()
