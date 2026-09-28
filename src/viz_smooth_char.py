"""
viz_smooth_char.py - 1단계(데이터 특성) 그림.  smooth_study.py --task char 의 결과를 읽어 그린다.

  fig1  위치 잡음: 정지 차량 잔차의 스텝별 크기 · 스텝 변위 분포 · 잔차 자기상관
  fig2  스펙트럼: 위치 · 속력 · 진행방향, 속력 구간별 + 정지 차량(= 순수 잡음)
  fig3  AV2 속도 필드와의 관계 + 창 가장자리 램프

  python src/viz_smooth_char.py
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

OUT = S.OUT
DATA = S.DATA


def main():
    plt = V.setup_mpl()
    ch = json.loads((DATA / "char.json").read_text())
    z = np.load(DATA / "char.npz")
    n = ch["noise"]
    ns, na = ch["n_scen"], ch["n_agent"]

    # ---------------------------------------------------------------- fig1 잡음
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
    ax = axes[0]
    for name, col in (("주변차량", V.C_BLUE), ("focal", V.C_ORANGE)):
        if name not in n:
            continue
        y = np.array(n[name]["per_step_rms"]) * 100
        ax.plot(np.arange(len(y)) * S.DT, y, color=col, label=f"{name} (n={n[name]['n']:,})",
                **V.step_kw(10, col))
    ax.axvspan(0, (S.RAMP - 1) * S.DT, color=V.C_YELLOW, alpha=0.18, lw=0, zorder=0)
    ax.text(0.2, 0.02, "램프", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7.4, color=V.INK2)
    ax.set_xlabel("관측 창 안 시각 [s]  (0 = −4.9 s)")
    ax.set_ylabel("1차 추세 제거 잔차 rms [cm]")
    ax.set_title("정지 차량 위치의 흔들림 — 스텝별", loc="left")
    ax.legend(loc="upper center")

    ax = axes[1]
    st = n["색"]["주변차량"]
    txt = (f"정지 차량 스텝 변위 (n={n['주변차량']['n']:,} 트랙 × 49 스텝)\n"
           f"  중앙 {n['주변차량']['step_disp_p50']*100:.2f} cm · p99 {n['주변차량']['step_disp_p99']*100:.1f} cm\n"
           f"  1 cm 미만 {st['step_under_1cm_pct']:.1f} %\n"
           f"  ≥ {S.V_MIN:.0f} m/s (= 10 cm/스텝) {st['step_ge_vmin_pct']:.2f} %\n\n"
           f"잔차 표준편차 (1차 추세 제거)\n"
           f"  종 {n['주변차량']['std_lin_lonlat'][0]*100:.1f} cm · 횡 {n['주변차량']['std_lin_lonlat'][1]*100:.1f} cm\n\n"
           f"백색 성분 σ (Δ²x 에서, Var(Δ²x)=6σ²)\n"
           f"  rms {st['sig_white_rms_cm']:.2f} cm · MAD {st['sig_white_mad_cm']:.3f} cm\n\n"
           f"등속 직진(≥8 m/s, |Δcourse|<2°) 직선 적합 잔차 (n={n['등속직진']['lin']['n']:,})\n"
           f"  1차: 종 {n['등속직진']['lin']['std_lonlat'][0]*100:.1f} · 횡 {n['등속직진']['lin']['std_lonlat'][1]*100:.1f} cm\n"
           f"  2차: 종 {n['등속직진']['quad']['std_lonlat'][0]*100:.1f} · 횡 {n['등속직진']['quad']['std_lonlat'][1]*100:.1f} cm")
    ax.text(0.0, 1.0, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.6,
            color=V.INK, linespacing=1.6, family="Noto Sans CJK KR")
    ax.axis("off")
    ax.set_title("잡음 크기 — 두 가지 추정이 횡방향에서 일치한다 (7 cm)", loc="left")

    ax = axes[2]
    for name, col in (("주변차량", V.C_BLUE), ("focal", V.C_ORANGE)):
        if name not in n["색"]:
            continue
        a = n["색"][name]["resid_autocorr_lag1_10"]
        ax.plot(np.arange(1, len(a) + 1) * S.DT, a, color=col, label=name, **V.step_kw(10, col))
    ax.axhline(0, color=V.AXIS, lw=0.8)
    ax.set_xlabel("지연 [s]")
    ax.set_ylabel("잔차 자기상관")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("잡음은 백색이 아니다 — 1 s 지연에도 0.3", loc="left")
    ax.legend()
    fig.suptitle(f"1단계 ① 위치 잡음  ·  val {ns:,} 시나리오, 정지 = 관측 50스텝 AV2 속력 최대 < {S.STOP_VMAX} m/s",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    V.savefig(fig, OUT / "char_1_noise.png")

    # ---------------------------------------------------------------- fig2 스펙트럼
    lbl = ch["spectrum"]["bins"]
    nn = ch["spectrum"]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    cols = V.SERIES
    f = z["spec_freq"]
    for ax, pre, ttl, unit in ((axes[0], "pos_x", "위치 x (1차 추세 제거)", "진폭 [m]"),
                               (axes[1], "spd", "속력 (위치차분)", "진폭 [m/s]"),
                               (axes[2], "hed", "진행방향 (unwrap, 위치차분)", "진폭 [°]")):
        for i, L in enumerate(lbl):
            k = f"spec_{pre}_{i}"
            if k not in z.files:
                continue
            ax.loglog(f[1:], z[k][1:], color=cols[i], lw=1.6, label=f"{L} m/s (n={nn[L]:,})")
        if pre == "pos_x":
            ax.loglog(z["spec_freq_stop"][1:], z["spec_pos_stop"][1:], color=V.INK, lw=2.0, ls="--",
                      label=f"정지 차량 = 순수 잡음 (n={ch['spectrum']['n_stop']:,})")
        if pre == "spd":
            ax.loglog(f[1:], z["spec_spdfield_2"][1:], color=V.INK, lw=2.0, ls="--",
                      label="5–10 m/s, AV2 속도 필드")
        ax.axvline(1.0, color=V.C_RED, lw=1.0, ls=":")
        ax.text(1.05, 0.03, "1 Hz", transform=ax.get_xaxis_transform(), fontsize=7.6, color=V.C_RED)
        ax.set_xlabel("주파수 [Hz]")
        ax.set_ylabel(unit)
        ax.set_title(ttl, loc="left")
        ax.legend(fontsize=7.4, loc="lower left")
    fig.suptitle(f"1단계 ② 진폭 스펙트럼  ·  focal {ns:,}, 창 = {ch['spectrum']['window']} (Hann, Δf = {ch['spectrum']['df_hz']:.2g} Hz)\n"
                 "위치: 1 Hz 위에서 이동 차량과 정지 차량(잡음)의 크기가 같아진다  ·  속력: 위치차분은 1 Hz 위에 잡음 바닥이 있고 속도 필드는 계속 떨어진다",
                 fontsize=10.6, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    V.savefig(fig, OUT / "char_2_spectrum.png")

    # ---------------------------------------------------------------- fig3 속도 필드 · 창 가장자리
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
    ax = axes[0]
    co = ch["course"]
    bins = [b for b in co["focal"]]
    x = np.arange(len(bins))
    w = 0.38
    for j, (who, col) in enumerate((("focal", V.C_BLUE), ("주변차량", V.C_ORANGE))):
        p50 = [co[who][b]["p50"] for b in bins]
        p90 = [co[who][b]["p90"] for b in bins]
        ax.bar(x + (j - 0.5) * w, p50, w, color=col, label=f"{who} 중앙")
        ax.plot(x + (j - 0.5) * w, p90, ls="none", marker="_", ms=14, mew=2.0, color=V.INK,
                label="p90" if j == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(x, [b.replace(" ", "\n") for b in bins])
    ax.set_xlabel("구간 속력 [m/s]")
    ax.set_ylabel("|위치차분 방향 − 속도필드 course| [°]")
    ax.set_title("속도 필드는 1 m/s 이상에서만 기준이 된다", loc="left")
    ax.legend(fontsize=7.8)
    ax.set_ylim(0.2, 700)
    for i, b in enumerate(bins):
        ax.text(i, 1.0, f"n={co['focal'][b]['n']//1000:,}k/{co['주변차량'][b]['n']//1000:,}k",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=V.MUTED)

    ax = axes[1]
    e = ch["edge"]
    r = np.array(e["ratio_step_p50"])
    t = np.arange(len(r)) * S.DT
    ax.fill_between(t, e["ratio_step_p25"], e["ratio_step_p75"], color=V.C_BLUE, alpha=0.18, lw=0)
    ax.plot(t, r, color=V.C_BLUE, lw=1.8)
    ax.axhline(1.0, color=V.AXIS, lw=1.0)
    for a, b in ((0, (S.RAMP - 1) * S.DT), (t[-1] - (S.RAMP - 1) * S.DT, t[-1])):
        ax.axvspan(a, b, color=V.C_YELLOW, alpha=0.2, lw=0, zorder=0)
    ax.axvline((S.OBS - 1) * S.DT, color=V.C_RED, lw=1.2, ls="--")
    ax.text((S.OBS - 1) * S.DT + 0.1, 0.97, "예측 시작 (t=49)", transform=ax.get_xaxis_transform(),
            fontsize=7.4, color=V.C_RED, va="top")
    ax.set_xlabel("11초 창 안 시각 [s]")
    ax.set_ylabel("위치차분 속력 ÷ 속도필드 속력")
    ax.set_title(f"창 양 끝 0.5초는 위치가 느리다 (첫 스텝 {e['first_ratio_p50']:.3f}배)", loc="left")

    ax = axes[2]
    a = np.array(e["absa_step_p50"])
    ax.plot(np.arange(len(a)) * S.DT, a, color=V.C_ORANGE, lw=1.8)
    for lo, hi in ((0, (S.RAMP - 1) * S.DT), ((len(a) - S.RAMP) * S.DT, (len(a) - 1) * S.DT)):
        ax.axvspan(lo, hi, color=V.C_YELLOW, alpha=0.2, lw=0, zorder=0)
    ax.axhline(8.0, color=V.C_RED, lw=1.0, ls=":")
    ax.set_ylim(0, 11.5)
    ax.text(4.0, 8.3, "사람 운전 상한 8 m/s²", fontsize=7.4, color=V.C_RED)
    ax.set_xlabel("11초 창 안 시각 [s]")
    ax.set_ylabel("|가속도| 중앙 [m/s²]  (위치차분)")
    ax.set_title("램프 구간의 가속도는 물리적으로 불가능하다", loc="left")
    fig.suptitle(f"1단계 ③ AV2 속도 필드와의 관계 · 창 가장자리  ·  램프는 기준속력 > 3 m/s 인 focal {e['n']:,}대",
                 fontsize=11.0, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    V.savefig(fig, OUT / "char_3_field_edge.png")
    print("그림 3장 ->", OUT)


if __name__ == "__main__":
    main()
