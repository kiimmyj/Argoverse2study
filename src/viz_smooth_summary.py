"""
viz_smooth_summary.py - 후보 평활기 비교 요약 그림. smooth_study.py --task eval 결과(eval.json)를 읽는다.

  fig1  지표 ①~⑦ small multiples (방법 축, 단위 통일, n 표기)
  fig2  ①–② 상충 (Pareto) — 10 Hz · 2 Hz, 독립 기준 대비 요레이트 RMSE 를 색으로
  fig3  2 Hz 다운샘플 뒤 (지표 ⑧) 와 합성 검증(등속 직선 · 등회전)

  python src/viz_smooth_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import viz_v4_common as V
import smooth_study as S

OUT, DATA = S.OUT, S.DATA

GROUP_COLOR = {"기준": V.MUTED, "SG": V.C_BLUE, "가우시안": V.C_ORANGE, "저역통과": V.C_AQUA,
               "스플라인": V.C_YELLOW, "칼만": V.C_GREEN, "각도": V.C_VIOLET, "제약적합": V.C_RED}
SHORT = {"none": "없음", "sg5_2": "SG5,2*", "sg5_2_ns": "SG5,2 전체", "sg7_2": "SG7,2", "sg9_2": "SG9,2",
         "sg11_2": "SG11,2", "sg9_3": "SG9,3", "sg11_3": "SG11,3",
         "g015": "가우스.15", "g025": "가우스.25", "g040": "가우스.40", "g025q": "가우스.25q",
         "bw05": "BW 0.5Hz", "bw10": "BW 1Hz", "bw20": "BW 2Hz",
         "spl05": "스플0.5s", "spl10": "스플1.0s",
         "kf_cv": "KF CV", "kf_ca": "KF CA", "kf_ctra": "KF CTRA",
         "kf_cv_v": "KF CV+v", "kf_ca_v": "KF CA+v", "kf_ctra_v": "KF CTRA+v",
         "kf_ca_r5": "KF CA σ5", "kf_ca_v2": "KF CA+v σ.2", "kf_ctra_r5": "KF CTRA σ5",
         "ang_none": "각도만", "ang_sg9": "SG9+각도", "ang_g025": "가우스.25+각도",
         "ang_kfcav": "KF CA+v+각도", "kin": "운동학제약"}


def main():
    plt = V.setup_mpl()
    d = json.loads((DATA / "eval.json").read_text())
    M, m0 = d["methods"], d["meta"]
    keys = list(M)
    base = M["none"]
    x = np.arange(len(keys))
    cols = [GROUP_COLOR.get(M[k]["group"], V.MUTED) for k in keys]
    lab = [SHORT.get(k, k) for k in keys]

    # ---------------------------------------------------------------- fig1 small multiples
    specs = [
        ("① 직진 |Δh| p90  [°/s]", lambda r: r["m1"]["dh_p90"], f"직진 n={m0['n_straight']:,}", True),
        ("① 직진 추세제거 잔차 σ  [°]", lambda r: r["m1"]["resid_std_p50"], f"직진 n={m0['n_straight']:,}", True),
        ("① 지그재그(Δh 부호반전) [%]", lambda r: r["m1"]["flip_pct"], f"직진 n={m0['n_straight']:,}", True),
        ("② 최대 요레이트 비 (÷속도필드)", lambda r: r["m2"]["peak_ratio_p50"], f"회전 n={m0['n_turn']:,}", False),
        ("② 순 회전량 비 (÷속도필드)", lambda r: r["m2"]["net_ratio_p50"], f"회전 n={m0['n_turn']:,}", False),
        ("② 요레이트 RMSE (÷속도필드) [°/s]", lambda r: r["m2"]["w_rmse_degs"], f"회전 n={m0['n_turn']:,}", True),
        ("③ |h − course| 중앙 [°]", lambda r: r["m3"]["all"]["p50"], "1 m/s 이상 스텝", True),
        ("④ 위치 RMSE [cm]", lambda r: r["m4"]["rmse_m"] * 100, "원본 위치 대비", True),
        ("⑤ 정지 차량 가짜 이동 스텝 [%]", lambda r: r["m5"]["fake_step_pct"], f"정지 n={m0['n_stop_eval']:,}", True),
        ("⑤ 정지 차량 180° 뒤집힘 판정 [%]", lambda r: r["m5"]["flip_pct"], f"정지 n={m0['n_stop_eval']:,}", True),
        ("⑥ 첫 스텝 속력 비 (1 이 정상)", lambda r: r["m6"]["first_ratio_p50"], f"n={base['m6']['n']:,}", False),
        ("⑦ |저크| < 4 m/s³ 비율 [%]", lambda r: r["m7"]["j_under4_pct"], "전체 focal", False),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.2), sharex=True)
    for ax, (ttl, fn, pop, lower_better) in zip(axes.ravel(), specs):
        y = np.array([fn(M[k]) for k in keys])
        ax.bar(x, y, color=cols, width=0.74)
        # 한 방법(kf_ctra_v)이 발산해 축을 깨므로, 나머지의 최대값 기준으로 자르고 넘친 값은 글로 적는다
        ok = np.array([k != "kf_ctra_v" for k in keys])
        hi = y[ok].max() * 1.18
        if y.max() > hi:
            ax.set_ylim(min(0, y.min()), hi)
            for i in np.flatnonzero(y > hi):
                ax.annotate(f"{y[i]:.0f}↑", (x[i], hi), ha="center", va="top", fontsize=7.0,
                            color=V.C_RED, rotation=90)
        b = fn(base)
        ax.axhline(b, color=V.INK, lw=1.0, ls="--")
        ax.text(len(keys) - 0.4, b, " 기준선", va="center", ha="left", fontsize=7.2, color=V.INK)
        ax.set_title(f"{ttl}   ({pop}, {'작을수록' if lower_better else '클수록'} 좋다)",
                     loc="left", fontsize=9.6)
        cur = fn(M["sg5_2"])
        ax.plot([keys.index("sg5_2")], [cur], marker="v", ms=8, color=V.INK, zorder=5, ls="none")
    for ax in axes[-1]:
        ax.set_xticks(x, lab, rotation=90, fontsize=7.6)
    fig.suptitle("후보 평활기 비교 — 지표 ①~⑦  ·  val "
                 f"{m0['n_scen']:,} 시나리오 · 관측 50스텝 중 인덱스 {m0['eval_range'][0]}~{m0['eval_range'][1]-1} 만 사용 (창 시작 램프 제외)\n"
                 "* = 지금 쓰는 방식 SG(5,2) (▼ 표시).  점선 = 평활 없음.  색 = 방법 계열",
                 fontsize=11.5, x=0.01, ha="left")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, label=g) for g, c in GROUP_COLOR.items()],
               loc="upper right", ncol=8, fontsize=8.6, frameon=False, bbox_to_anchor=(0.995, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    V.savefig(fig, OUT / "sum_1_metrics.png")

    # ---------------------------------------------------------------- fig2 Pareto
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for ax, (xf, yf, xl, yl, ttl) in zip(axes, [
            (lambda r: r["m1"]["dh_p90"], lambda r: r["m2"]["peak_ratio_p50"],
             "① 직진 |Δh| p90 [°/s]  (작을수록)", "② 최대 요레이트 비  (클수록)", "10 Hz — 사용자 기준 ①과 ②"),
            (lambda r: r["m8_1"]["dh_p90"], lambda r: r["m2"]["net_ratio_p50"],
             "①(2 Hz) 직진 |Δh| p90 [°/s]", "② 순 회전량 비  (클수록)", "2 Hz 다운샘플 뒤 vs 순 회전량"),
            (lambda r: r["m1"]["w_rmse_degs"], lambda r: r["m2"]["w_rmse_degs"],
             "직진 요레이트 RMSE [°/s]", "회전 요레이트 RMSE [°/s]", "독립 기준(속도필드)과의 차이 — 둘 다 작을수록")]):
        kk = keys if ax is not axes[2] else [k for k in keys if k != "kf_ctra_v"]
        for k in kk:
            r = M[k]
            c = GROUP_COLOR.get(r["group"], V.MUTED)
            big = k in ("none", "sg5_2")
            ax.plot(xf(r), yf(r), marker="o", ms=10 if big else 7, color=c, ls="none",
                    mec=V.INK if big else V.SURF, mew=1.4 if big else 0.6, zorder=6 if big else 4)
        V.place_labels(ax, [(xf(M[k]), yf(M[k])) for k in kk], [SHORT.get(k, k) for k in kk], fontsize=7.0)
        if ax is axes[2]:
            ax.text(0.99, 0.03, "KF CTRA+v (직진 94.9 · 회전 189.1) 는 EKF 가 발산해 축 밖", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.4, color=V.C_RED)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    fig.suptitle("① 잡음 억제와 ② 회전 보존은 직접 상충한다  ·  굵은 테두리 = 평활 없음 / 지금 방식\n"
                 f"직진 n={m0['n_straight']:,} · 회전 n={m0['n_turn']:,} · val {m0['n_scen']:,} 시나리오",
                 fontsize=11.0, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    V.savefig(fig, OUT / "sum_2_pareto.png")

    # ---------------------------------------------------------------- fig3 2 Hz + 합성 검증
    ver = json.loads((DATA / "verify.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    ax = axes[0, 0]
    w = 0.4
    for j, (nm, fn, col) in enumerate((("10 Hz", lambda r: r["m1"]["dh_p90"], V.C_BLUE),
                                       ("2 Hz (평활→다운샘플)", lambda r: r["m8_1"]["dh_p90"], V.C_ORANGE))):
        ax.bar(x + (j - 0.5) * w, [fn(M[k]) for k in keys], w, color=col, label=nm)
    ax.set_xticks(x, lab, rotation=90, fontsize=7.4)
    ax.set_ylabel("직진 |Δh| p90 [°/s]")
    ax.set_title(f"⑧ 다운샘플이 평활보다 크게 줄인다 (직진 n={m0['n_straight']:,})", loc="left")
    ax.legend()

    ax = axes[0, 1]
    for j, (nm, fn, col) in enumerate((("10 Hz", lambda r: r["m3"]["all"]["p50"], V.C_BLUE),
                                       ("2 Hz", lambda r: r["m8_3"]["all"]["p50"], V.C_ORANGE))):
        ax.bar(x + (j - 0.5) * w, [fn(M[k]) for k in keys], w, color=col, label=nm)
    ax.set_xticks(x, lab, rotation=90, fontsize=7.4)
    ax.set_ylabel("|h − 속도필드 course| 중앙 [°]")
    ax.set_title("③ 독립 기준과의 차이 — 속도 필드를 관측으로 넣은 판만 내려간다", loc="left")
    ax.legend()

    for ax, key, ttl, ylab, ref in ((axes[1, 0], "등속 직선(12 m/s) 왜곡", "합성 ① 등속 직선(12 m/s, 잡음 0) 을 그대로 두는가",
                                     "첫 스텝 속력 비", "first_speed_ratio"),
                                    (axes[1, 1], "등회전(R=30 m, ω=0.4 rad/s) 보존",
                                     "합성 ② 등회전(잡음 0) 의 요레이트를 얼마나 남기는가", "요레이트 비", "yaw_rate_mid")):
        vk = ver[key]
        ks = [k for k in keys if k in vk]
        xs = np.arange(len(ks))
        if ref == "first_speed_ratio":
            y1 = [vk[k]["first_speed_ratio"] for k in ks]
            ax.bar(xs, y1, 0.74, color=[GROUP_COLOR.get(M[k]["group"], V.MUTED) for k in ks], label="첫 스텝 속력 비")
            ax2 = ax.twinx()
            ax2.plot(xs, [vk[k]["max_dev_cm"] for k in ks], ls="none", marker="D", ms=5, color=V.INK)
            ax2.set_ylabel("최대 위치 변화 [cm] (◆)")
            ax.set_ylim(0.9, 1.1)
        else:
            ax.bar(xs - 0.19, [vk[k]["yaw_rate_mid"] for k in ks], 0.38,
                   color=[GROUP_COLOR.get(M[k]["group"], V.MUTED) for k in ks], label="창 안쪽(스텝 10~35)")
            ax.bar(xs + 0.19, [vk[k]["yaw_rate_min"] for k in ks], 0.38, color=V.MUTED, alpha=0.55,
                   label="창 안 최솟값 (경계 포함)")
            ax.axhline(1.0, color=V.INK, lw=1.0, ls="--")
            ax.set_ylim(-0.1, 1.15)
            ax.legend(fontsize=8)
        ax.set_xticks(xs, [SHORT.get(k, k) for k in ks], rotation=90, fontsize=7.4)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left")
    fig.suptitle("⑧ 2 Hz 다운샘플 뒤  ·  잡음 없는 합성 신호에서의 성질 (실데이터 ②가 '신호 감쇠'인지 '잡음 제거'인지 가른다)",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    V.savefig(fig, OUT / "sum_3_2hz_synth.png")
    print("그림 3장 ->", OUT)


if __name__ == "__main__":
    main()
