"""전처리 평활 전 vs 후 — 상황별 대표 시나리오 패널과 구간별 요약.

전(현행 대조판) : ah_features_v3(sigma_s=0, blend=False, drop_ramp=False)
                  = 평활 없음 + V_MIN 1 m/s 하드 스위치 + 옛 align_ref + 램프 포함
후(채택안)      : ah_features_v3(sigma_s=0.25, blend=True, drop_ramp=True)
                  = 가우시안 σ=0.25 s + 속력가중 혼합(교차 3 m/s) + 이동 구간 뒤집힘 판정 + 램프 제외

  python src/viz_smooth_ba.py --limit 400 --hz 2
출력: viz/v4/smoothing_ba/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heading_decomp as hd
import smooth_study as S
import viz_v4_common as V

OUT = Path(__file__).resolve().parents[1] / "viz" / "v4" / "smoothing_ba"   # main() 에서 hz 별로 바꾼다
DT = 0.1
OBS = 50
JIT_HI = 10.0                      # 참고선 [°/s] — 평활 연구의 jitter p90 눈금
BINS = [(0.0, 5.0, "직진 <5°"), (5.0, 30.0, "완만 5–30°"), (30.0, 1e9, "회전 ≥30°")]


def pair(pos, head, step):
    """(전, 후) = (시간[s], a 채널, h[deg], |Δh|[°/step], 속력[m/s])"""
    yaw0 = float(head[OBS - 1])
    out = []
    for sigma, blend, drop in ((0.0, False, False), (hd.GAUSS_SIGMA_S, True, True)):
        feat, _ = hd.ah_features_v3(pos[:OBS], head[:OBS], yaw0, step=step,
                                    sigma_s=sigma, blend=blend, drop_ramp=drop)
        n = len(feat)
        t = np.arange(n) * DT * step                      # 마지막 관측이 오른쪽 끝이 되도록 뒤에서 맞춘다
        t = t - t[-1]
        h = np.degrees(feat[:, 1])
        dh = np.abs(np.degrees(np.diff(np.unwrap(feat[:, 1])))) / (DT * step)   # [°/s]
        lo = hd.RAMP_SKIP if drop else 0                  # ah_features_v3 와 같은 색인·평활
        idx = np.arange(OBS - 1, lo - 1, -step)[::-1]
        assert len(idx) == n, f"색인 {len(idx)} != 특징 {n}"
        sm = pos[:OBS].copy()
        if sigma > 0:                                      # 램프(0~skip)는 평활 창에 넣지 않는다
            sm[hd.RAMP_SKIP:] = hd.gauss_pad_smooth(pos[hd.RAMP_SKIP:OBS], sigma_steps=sigma / DT)
        p = sm[idx]
        sp = np.r_[np.nan, np.linalg.norm(np.diff(p, axis=0), axis=1) / (DT * step)]
        out.append(dict(t=t, a=feat[:, 0], h=h, dh=dh, sp=sp, p=p))
    return out


def turn_amount(head):
    g = hd.guard(np.asarray(head[:OBS], np.float64))
    return float(np.abs(np.degrees(hd.wrap(np.diff(np.unwrap(g))))).sum())


def panel(name, pos, head, step, tag, sid):
    import matplotlib.pyplot as plt
    be, af = pair(pos, head, step)
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 9.6), height_ratios=[2.0, 1.0, 1.0, 1.0],
                             layout="constrained")
    fig.suptitle(f"평활 전 vs 후 — {name}  ·  {sid[:8]}  ·  입력 {'10' if step == 1 else '2'} Hz",
                 fontsize=11.5, x=0.01, ha="left")

    ax = axes[0]                                           # 궤적 (관측 구간)
    for d, lbl, c in ((be, "전", V.C_RED), (af, "후", V.C_BLUE)):
        ax.plot(d["p"][:, 0], d["p"][:, 1], "-o", ms=3.2, lw=1.1, color=c, label=lbl, alpha=0.95)
    ax.plot(pos[:OBS, 0], pos[:OBS, 1], ".", ms=2.2, color=V.MUTED, label="원본 10 Hz", zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="best")

    for ax, key, ylab, extra in ((axes[1], "h", "진행방향 h [°]", None),
                                 (axes[2], "dh", "방향 변화율 |Δh|/Δt [°/s]", JIT_HI),
                                 (axes[3], "sp", "속력 [m/s]", None)):
        for d, lbl, c in ((be, "전", V.C_RED), (af, "후", V.C_BLUE)):
            y = d[key]
            x = d["t"][1:] if key == "dh" else d["t"]
            ax.plot(x, y, "-o", ms=3.0, lw=1.1, color=c, label=lbl)
        if extra:
            ax.axhline(extra, color=V.INK2, lw=0.9, ls="--")
            ax.text(0.005, extra, f" 참고 {extra:.0f}°/s", transform=ax.get_yaxis_transform(),
                    fontsize=7.4, color=V.INK2, va="bottom")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[3].set_xlabel("관측 끝 기준 시간 [s]")          # 맨 아래에만 축 이름
    axes[1].legend(loc="best")
    V.savefig(fig, OUT / f"panel_{tag}.png")


def summary(sids, poss, heads, step):
    import matplotlib.pyplot as plt
    rows = []
    for pos, head in zip(poss, heads):
        be, af = pair(pos, head, step)
        jk = lambda d: np.percentile(np.abs(np.diff(d["sp"][1:], 2)) / (DT * step) ** 2, 90) \
            if np.isfinite(d["sp"][1:]).all() and len(d["sp"]) > 4 else np.nan
        rows.append((turn_amount(head), np.percentile(be["dh"], 90), np.percentile(af["dh"], 90),
                     np.nanmax(be["sp"]), jk(be), jk(af)))
    r = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    xs = np.arange(len(BINS))
    stat = []
    for ax, (i_be, i_af), ylab in ((axes[0], (1, 2), "방향 변화율 p90 [°/s]"),
                                   (axes[1], (4, 5), "jerk p90 [m/s³]")):
        b, a = [], []
        for lo, hi, _ in BINS:
            m = (r[:, 0] >= lo) & (r[:, 0] < hi)
            b.append(np.nanmedian(r[m, i_be]) if m.any() else np.nan)
            a.append(np.nanmedian(r[m, i_af]) if m.any() else np.nan)
        ax.bar(xs - 0.19, b, 0.36, color=V.C_RED, label="전")
        ax.bar(xs + 0.19, a, 0.36, color=V.C_BLUE, label="후")
        ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n({int(((r[:,0]>=lo)&(r[:,0]<hi)).sum())}건)"
                                               for lo, hi, l in BINS], fontsize=8)
        ax.set_ylabel(ylab); ax.legend(); ax.grid(axis="y", alpha=0.25)
        stat.append((b, a))
    fig.suptitle(f"구간별 중앙값 — 정답 회전량으로 나눈 {len(r)}개 시나리오 (입력 {'10' if step == 1 else '2'} Hz)",
                 fontsize=11, x=0.01, ha="left")
    V.savefig(fig, OUT / "summary_bins.png")
    for lo, hi, lbl in BINS:
        print(f"  구간 {lbl:12} {int(((r[:,0]>=lo)&(r[:,0]<hi)).sum()):4d}건")
    for (lo, hi, lbl), p90b, p90a, vb, va in zip(BINS, stat[0][0], stat[0][1], stat[1][0], stat[1][1]):
        print(f"{lbl:12} |Δh|/Δt p90 {p90b:6.2f} → {p90a:6.2f} °/s | jerk p90 {vb:6.2f} → {va:6.2f} m/s³")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--hz", type=int, default=2, choices=[2, 10])
    ap.add_argument("--cases", action="store_true", help="차선변경·U턴 보존만 본다")
    a = ap.parse_args()
    step = 5 if a.hz == 2 else 1
    global OUT
    OUT = OUT / f"hz{a.hz}"
    OUT.mkdir(parents=True, exist_ok=True)
    V.setup_mpl()

    dirs = S.scen_dirs("val", limit=a.limit, seed=0)
    from multiprocessing import Pool
    with Pool(16) as pool:
        res = [r for r in pool.imap(S.one_scenario, dirs, chunksize=8) if r is not None]
    sids = [r[0] for r in res]
    poss = [r[1] for r in res]
    heads = [r[3] for r in res]
    print(f"시나리오 {len(sids)}개")

    if a.cases:
        cases(sids, poss, heads, step)
        print("출력:", OUT)
        return

    turn = np.array([turn_amount(h) for h in heads])
    vmax = np.array([np.linalg.norm(np.diff(p[:OBS], axis=0), axis=1).max() / DT for p in poss])
    fast = vmax >= 5.0                                   # 주차 기동이 대표로 뽑히지 않게
    def pick(score, mask):
        sc = np.where(mask, score, 1e9)
        return int(np.argmin(sc))
    picks = [("직진", pick(turn, fast)),
             ("완만한 회전", pick(np.abs(turn - 15), fast)),
             ("회전", pick(np.abs(turn - 90), fast)),
             ("저속·정지", int(np.argmin(vmax)))]
    for name, i in picks:
        tag = {"직진": "straight", "완만한 회전": "mild", "회전": "turn", "저속·정지": "slow"}[name]
        panel(name, poss[i], heads[i], step, tag, sids[i])
        print(f"{name:10} {sids[i][:8]} 회전량 {turn[i]:6.1f}° 최대속력 {vmax[i]:5.1f} m/s")
    summary(sids, poss, heads, step)
    print("출력:", OUT)



# ------------------------------------------------------------------ 차선변경·U턴 보존
def obs_signal(pos, head, step, sigma, drop):
    """관측 구간에서 (횡이동 [m], 총 방향변화 [°]) — 지도 없이 자기 진행방향 기준."""
    lo = hd.RAMP_SKIP if drop else 0
    sm = pos[:OBS].copy()
    if sigma > 0:
        sm[hd.RAMP_SKIP:] = hd.gauss_pad_smooth(pos[hd.RAMP_SKIP:OBS], sigma_steps=sigma / DT)
    idx = np.arange(OBS - 1, lo - 1, -step)[::-1]
    p = sm[idx]
    d = np.diff(p, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    lat = float(np.sum(np.linalg.norm(d, axis=1) * np.sin(ang - ang[0])))   # 시작 방향 기준 횡이동
    return lat, float(np.degrees(ang[-1] - ang[0]))


def future_class(pos, head):
    """정답 미래(50~109)로 상황 분류. 지도 없이 자기 진행방향 기준."""
    p = pos[OBS - 1:]
    d = np.diff(p, axis=0)
    sp = np.linalg.norm(d, axis=1)
    if sp.sum() < 5.0:
        return "정지·저속"
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    dh = np.degrees(ang[-1] - ang[0])
    lat = float(np.sum(sp * np.sin(ang - ang[0])))
    if abs(dh) >= 150:
        return "U턴"
    if abs(dh) < 15 and abs(lat) >= 2.5:
        return "차선변경"
    if abs(dh) >= 30:
        return "회전"
    return "직진"


def cases(sids, poss, heads, step):
    """상황별로 평활 뒤 신호가 얼마나 남는지 + 대표 패널"""
    cls = np.array([future_class(p, h) for p, h in zip(poss, heads)])
    rows = {}
    for name in ("직진", "차선변경", "회전", "U턴"):
        m = np.flatnonzero(cls == name)
        if len(m) == 0:
            continue
        rb = np.array([obs_signal(poss[i], heads[i], step, 0.0, False) for i in m])
        ra = np.array([obs_signal(poss[i], heads[i], step, hd.GAUSS_SIGMA_S, True) for i in m])
        keep_lat = np.median(np.abs(ra[:, 0]) / np.maximum(np.abs(rb[:, 0]), 1e-6))
        keep_dh = np.median(np.abs(ra[:, 1]) / np.maximum(np.abs(rb[:, 1]), 1e-6))
        rows[name] = (len(m), np.median(np.abs(rb[:, 0])), np.median(np.abs(ra[:, 0])), keep_lat,
                      np.median(np.abs(rb[:, 1])), np.median(np.abs(ra[:, 1])), keep_dh)
        print(f"{name:7} {len(m):4d}건 | 관측 횡이동 {rows[name][1]:5.2f} → {rows[name][2]:5.2f} m "
              f"(보존 {keep_lat*100:5.1f}%) | 방향변화 {rows[name][4]:6.2f} → {rows[name][5]:6.2f}° "
              f"(보존 {keep_dh*100:5.1f}%)")
        if name in ("차선변경", "U턴"):
            sig = np.array([obs_signal(poss[j], heads[j], step, 0.0, False) for j in m])
            if name == "차선변경":      # 관측 구간이 곡선 주행이 아닌 것 중 횡이동이 큰 것
                ok = np.abs(sig[:, 1]) < 15.0
                cand = np.flatnonzero(ok) if ok.any() else np.arange(len(m))
            else:                      # U턴은 관측에서 이미 돌고 있는 것
                cand = np.arange(len(m))
            i = int(m[cand[np.argmax(np.abs(sig[cand, 0]))]])
            panel(name, poss[i], heads[i], step, {"차선변경": "lanechange", "U턴": "uturn"}[name], sids[i])
            print(f"   대표: {sids[i][:8]}")
    return rows

if __name__ == "__main__":
    main()
