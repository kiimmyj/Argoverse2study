"""적분기 상한 — 정답 데이터에서 **어디까지가 물리이고 어디부터가 잡음인지** 가른다.

처음엔 "정답 꼬리의 p99.9 를 상한으로 쓰자" 였는데 **그 방법이 틀렸다.** 재 보니
차량 종가속 p99 가 16.8 m/s², 최대 213 m/s² 였고, 움직이면 안 되는 **정적 물체**가
횡가속 63 m/s² 를 냈다. 정적 물체가 내는 값은 정의상 **전부 추적 잡음**이다.

왜 잡음이 꼬리를 먹나 — AV2 위치는 인지·추적 출력이라 몇 cm 씩 떨린다. dt=0.1 s 에서
2 cm 떨림이 0.2 m/s 속도 오차, 즉 2 m/s² 가속도가 된다. 그리고 **저속일수록 심하다**:
0.5 m/s 로 가는 차는 한 step 에 5 cm 움직이는데, 그게 떨림과 같은 크기라 위치차분
진행방향이 아무 데나 가리킨다(그래서 |Δh| 가 179°/step 까지 나온다).

그래서 이 스크립트가 하는 일이 바뀌었다
---------------------------------------
① **속도로 층화**해서 잡음이 어디서 물리에 자리를 내주는지 본다.
② **정적 물체**를 잡음 기준선으로 같이 찍는다 — 차량 값이 그보다 크지 않으면 못 믿는다.
③ 물리에서 온 후보 상한이 **진짜 거동을 얼마나 자르는지**(binding 비율) 센다.
   상한은 정답 분포의 꼬리가 아니라 **타이어 마찰 한계**에서 와야 한다.

히스토그램으로 모은다 — 전체 train 을 다 읽어도 메모리가 안 는다.

    python src/measure_kinematic_limits.py --n 20000
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import viz_v4_common as C

FEAT = Path("/data/argoverse2/simpl_features/train")
OUT = C.ROOT / "viz" / "kinematic_limits"
DT, V_MIN = 0.1, 0.5
TYPES = ["차량", "보행자", "오토바이", "자전거", "버스", "알수없음", "정적"]
# (이름, 하한, 상한) — 속도 [m/s]. 잡음은 저속에서 크고 물리는 고속에서 크다.
SBINS = [("≥0.5", 0.5, 1e9), ("0.5~2", 0.5, 2), ("2~5", 2, 5),
         ("5~10", 5, 10), ("10~20", 10, 20), ("≥20", 20, 1e9)]
# (이름, 상한값, 칸수) — 히스토그램 격자
GRID = [("a_lon", 120.0, 2400), ("a_lat", 120.0, 2400), ("dh_deg", 180.0, 1800)]
QS = [50, 90, 99, 99.9]
# 후보 상한 — 타이어 마찰에서 온다. μ≈0.8 승용차 건조 노면 → μg ≈ 7.8 m/s².
# 횡가속 6.87 = bahari2022vehicle 식 6 의 μ=0.7 · 종가속 8.0 = cui2019dkm 각주 클리핑 상수
CAND = {"a_lon": [4.0, 6.0, 8.0, 10.0], "a_lat": [4.0, 6.0, 6.87, 8.0, 10.0]}


def _hist(x, top, nb):
    return np.histogram(np.clip(x, 0, top * (1 - 1e-9)), bins=nb, range=(0, top))[0]


def chunk(paths):
    """파일 묶음 하나 → [타입, 속도구간, 양] 히스토그램 합 + 최대값."""
    import torch
    sys.path.insert(0, "/home/user/av2_simpl")
    from simpl import ah_feat
    H = np.zeros((7, len(SBINS), len(GRID), max(g[2] for g in GRID)), np.int64)
    M = np.zeros((7, len(SBINS), len(GRID)))
    nfail = 0
    for p in paths:
        try:
            T = pickle.load(open(p, "rb")).iloc[0]["TRAJS"]
            pos = torch.from_numpy(np.asarray(T["trajs_pos"], np.float32))
            yaw = torch.from_numpy(np.asarray(T["trajs_ang"], np.float32))
            ok = np.asarray(T["has_flags"]) > 0
            tid = np.asarray(T["trajs_type"])[:, 0, :].argmax(1)
            v, h, _ = ah_feat.speed_and_heading(pos, yaw, dt=DT, v_min=V_MIN)
            v, h = v.numpy(), h.numpy()
            dv = np.abs(np.diff(v, axis=1)) / DT
            dh = (np.diff(h, axis=1) + np.pi) % (2 * np.pi) - np.pi
            vals = [dv, np.abs(v[:, :-1] * dh / DT), np.degrees(np.abs(dh))]
            pair = ok[:, 1:] & ok[:, :-1]
            v0 = v[:, :-1]
            for k in range(7):
                m = (tid == k)[:, None] & pair
                if not m.any():
                    continue
                for bi, (_, lo, hi) in enumerate(SBINS):
                    sel = m & (v0 >= lo) & (v0 < hi)
                    if not sel.any():
                        continue
                    for gi, (_, top, nb) in enumerate(GRID):
                        x = vals[gi][sel]
                        H[k, bi, gi, :nb] += _hist(x, top, nb)
                        M[k, bi, gi] = max(M[k, bi, gi], float(x.max()))
        except Exception:
            nfail += 1
    return H, M, nfail


def quant(h, top, nb, qs=QS):
    """히스토그램에서 분위수. 칸 가운데값으로 읽는다(해상도 = top/nb)."""
    n = h[:nb].sum()
    if n == 0:
        return {f"p{q}": float("nan") for q in qs} | {"n": 0}
    c = np.cumsum(h[:nb])
    edges = (np.arange(nb) + 0.5) * (top / nb)
    out = {"n": int(n)}
    for q in qs:
        out[f"p{q}"] = float(edges[np.searchsorted(c, n * q / 100.0)])
    return out


def fig(plt, S):
    """두 패널 — ① 방향은 저속에서 잡음이다 ② 고속에서는 물리가 보인다."""
    labs = [b[0] for b in SBINS[1:]]                    # 전체(≥0.5) 는 빼고 구간만
    xi = np.arange(len(labs))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, key, ylab, title, ref in (
            (axL, "dh_deg", "|Δh| p99  [°/step]",
             "① 방향은 5 m/s 아래에서 잡음이다 — 정적 물체와 같은 값이 나온다", None),
            (axR, "a_lat", "|횡가속| p99  [m/s²]",
             "② 위에서는 물리가 보인다 — 타이어 마찰 한계와 같은 자리", 7.8)):
        # ⚠️ 두 선이 겹치는 자리(저속)가 있어 라벨을 같은 쪽에 달면 포갠다 —
        #    차량은 아래, 정적은 위로 고정한다(차량이 늘 아래에 있다).
        for k, col, nm, dy in ((0, C.C_TOP, "차량", -14), (6, C.MUTED, "정적 물체 (= 잡음 기준선)", 8)):
            y = [S[TYPES[k]][b][key]["p99"] for b in labs]
            ax.plot(xi, y, "o-", color=col, label=nm, ms=6, zorder=3)
            for x, v in zip(xi, y):
                ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=8.2, color=col)
        if ref:
            ax.axhline(ref, color=C.C_RED, lw=1.5, ls="--", zorder=2)
            ax.text(0.02, ref, " μg ≈ 7.8  (타이어 마찰 한계)", color=C.C_RED, fontsize=8.6,
                    va="bottom", ha="left", transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_xticks(xi); ax.set_xticklabels(labs)
        ax.set_xlabel("직전 step 속도 [m/s]"); ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.2, loc="left")
        ax.legend(fontsize=8.6, framealpha=0.93)
        ax.margins(y=0.30)
    fig.tight_layout()
    return C.savefig(fig, OUT / "k1_noise_vs_physics.png")


DOC = C.ROOT / "docs" / "action_spec.md"
BANDS = ["0.5~2", "2~5", "5~10", "10~20", "≥20"]


def md_block(S, bind, n_scene):
    """규격 문서의 표 둘. 손으로 적으면 상한을 바꿀 때 한쪽이 낡는다."""
    V, St = S["차량"], S["정적"]
    L = [f"AV2 **train** {n_scene:,}판 (`src/measure_kinematic_limits.py`). "
         f"⚠️ val 로 상한을 정하면 평가에 쓸 데이터로 설정값을 맞추는 셈이다.", "",
         "| 직전 step 속도 | 차량 \\|Δh\\| p99 | **정적 물체** \\|Δh\\| p99 | 차량 \\|횡가속\\| p99 |",
         "| --- | --- | --- | --- |"]
    for b in BANDS:
        hi = V[b]["dh_deg"]["p99"] >= 90          # 잡음과 구별 안 되는 구간
        L.append(f"| {b} m/s | {'' if hi else '**'}{V[b]['dh_deg']['p99']:.1f}°{'' if hi else '**'} "
                 f"| **{St[b]['dh_deg']['p99']:.1f}°** | {V[b]['a_lat']['p99']:.1f} |")
    L += ["", "**움직이면 안 되는 정적 물체가 모든 구간에서 \\|Δh\\| p99 ≈ 179°** — 정의상 전부 잡음이다. "
              "차량도 **5 m/s 아래에서는 구별이 안 된다.**", "",
          "| 차량 step 중 상한에 걸리는 비율 | " + " | ".join(f"{c:g}" for c in CAND["a_lat"]) + " |",
          "| --- | " + " | ".join("---" for _ in CAND["a_lat"]) + " |"]
    for b in ("5~10", "10~20", "≥20"):
        L.append(f"| 횡가속 · {b} m/s | " +
                 " | ".join(("**" if c == 6.87 else "") + f"{bind['a_lat'][b][str(c)]:.2f}%" +
                            ("**" if c == 6.87 else "") for c in CAND["a_lat"]) + " |")
    return "\n".join(L)


def fill_doc(S, bind, n_scene):
    if not DOC.exists():
        return
    txt = DOC.read_text()
    a, b = "<!-- KINLIM:START -->", "<!-- KINLIM:END -->"
    if a not in txt or b not in txt:
        print("  (표식 KINLIM 없음 — 건너뜀)")
        return
    head, rest = txt.split(a, 1)
    _, tail = rest.split(b, 1)
    DOC.write_text(f"{head}{a}\n\n{md_block(S, bind, n_scene)}\n\n{b}{tail}")
    print(f"  문서 갱신 → {DOC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(FEAT.glob("*.pkl"))
    rng = np.random.default_rng(a.seed)
    if a.n and a.n < len(files):
        files = [files[i] for i in rng.choice(len(files), a.n, replace=False)]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"train {len(files):,}장면 · 워커 {a.workers}")

    size = max(1, len(files) // (a.workers * 4))
    parts = [files[i:i + size] for i in range(0, len(files), size)]
    H = np.zeros((7, len(SBINS), len(GRID), max(g[2] for g in GRID)), np.int64)
    M = np.zeros((7, len(SBINS), len(GRID)))
    nfail = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for j, (h, m, f) in enumerate(ex.map(chunk, parts)):
            H += h; M = np.maximum(M, m); nfail += f
            if (j + 1) % 10 == 0:
                print(f"  묶음 {j+1}/{len(parts)}", flush=True)
    print(f"끝 — 실패 {nfail:,}\n")

    S = {}
    for k in range(7):
        S[TYPES[k]] = {}
        for bi, (bn, _, _) in enumerate(SBINS):
            e = {}
            for gi, (gn, top, nb) in enumerate(GRID):
                e[gn] = quant(H[k, bi, gi], top, nb) | {"max": float(M[k, bi, gi])}
            S[TYPES[k]][bn] = e

    def row(k, bi):
        e = S[TYPES[k]][SBINS[bi][0]]
        lo, la, dh = e["a_lon"], e["a_lat"], e["dh_deg"]
        return (f"{SBINS[bi][0]:>7s}{lo['n']:>13,}{lo['p50']:>8.2f}{lo['p90']:>8.2f}"
                f"{lo['p99']:>8.2f}{la['p50']:>9.2f}{la['p90']:>8.2f}{la['p99']:>8.2f}"
                f"{dh['p50']:>9.2f}{dh['p99']:>8.2f}")

    hdr = (f"{'속도':>7s}{'n(step)':>13s}{'종가속 p50':>8s}{'p90':>8s}{'p99':>8s}"
           f"{'횡가속 p50':>9s}{'p90':>8s}{'p99':>8s}{'|Δh| p50':>9s}{'p99':>8s}")
    for k in (0, 6):
        print(f"── {TYPES[k]} " + "─" * 70)
        print(hdr)
        for bi in range(len(SBINS)):
            print(row(k, bi))
        print()
    print("종·횡가속 [m/s²] · |Δh| [°/step] · 결측 제외 · 속도는 직전 step 기준\n")

    # 후보 상한이 진짜 거동을 얼마나 자르나 — **속도 구간별**로.
    # ⚠️ 전체(≥0.5)로 내면 잡음이 지배하는 저속 구간이 비율을 부풀린다.
    bind = {}
    for gi, (gn, top, nb) in enumerate(GRID[:2]):
        edges = (np.arange(nb) + 0.5) * (top / nb)
        bind[gn] = {}
        for bi, (bn, _, _) in enumerate(SBINS):
            h = H[0, bi, gi][:nb]; n = h.sum()
            bind[gn][bn] = {str(c): (float(h[edges > c].sum()) / n * 100 if n else float("nan"))
                            for c in CAND[gn]}
    for gn, lab in (("a_lon", "종가속"), ("a_lat", "횡가속")):
        print(f"{lab} 상한이 차량 step 을 자르는 비율 [%] — 속도 구간별 (train)")
        print(f"{'속도':>8s}" + "".join(f"{c:>10.1f}" for c in CAND[gn]))
        for bn, _, _ in SBINS:
            print(f"{bn:>8s}" + "".join(f"{bind[gn][bn][str(c)]:>9.2f}%" for c in CAND[gn]))
        print()

    (OUT / "numbers.json").write_text(json.dumps(
        {"n_scene": len(files), "source": "train", "v_min": V_MIN, "dt": DT,
         "note": "ah_feat.speed_and_heading 정의. 결측 제외. 히스토그램 분위수.",
         "by_type": S, "binding_pct_vehicle": bind}, ensure_ascii=False, indent=1))
    fill_doc(S, bind, len(files))
    print(f"그림 → {fig(C.setup_mpl(), S)}")
    print(f"숫자 → {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
