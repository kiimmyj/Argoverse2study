"""미팅용 그림 — R2 가 왜 실패했나: 출력 헤드의 방향 채널이 0 으로 붕괴했다.

두 패널. 왼쪽이 증상(오차), 오른쪽이 원인(붕괴). 패널 제목은 주장으로 쓴다.

숫자 출처 (손으로 적지 않는다)
    viz/dh_collapse/init.json   학습 전 — ~/av2_simpl/tools/diag_action_saturation.py
    viz/dh_collapse/ep10.json   10 에폭 체크포인트 — 같은 스크립트, 같은 640 장면
    ~/av2_simpl/r0.log · r1.log · r2_dh_collapse_killed.log   에폭별 val 지표

⚠️ 교차검증 (시각화 요건 #9-2). 정답 `ΣΔh` 를 **torch 도 AH.invert 도 안 쓰는 경로**로
   다시 센다 — SIMPL feature pickle 에서 대상 차 미래 좌표를 직접 읽어 numpy 로 atan2·wrap·합.
   두 경로가 어긋나면 그림을 내지 않는다.

    python src/viz_meeting_1126.py
    python src/viz_meeting_1126.py --no-verify     # 교차검증 생략 (pickle 읽기가 느릴 때)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import viz_v4_common as C                                           # noqa: E402

ROOT = SRC.parent
OUT = ROOT / "viz" / "dh_collapse"
SIMPL = Path("/home/user/av2_simpl")
FEAT_VAL = Path("/data/argoverse2/simpl_features/val")
OBS = 50

RUNS = [("R0 — 원본", "r0.log", C.INK),
        ("R1 — 입력만 (a, h)", "r1.log", C.C_BLUE),
        ("R2 — 출력도 (a, Δh)", "r2_dh_collapse_killed.log", C.C_RED)]
QLAB = ["중앙", "p90", "p99", "최대"]


# ------------------------------------------------------------------ 로그 읽기
def val_curve(path):
    """[(에폭, minADE6)] — `[Validation]` **다음 줄**만 val 이다.

    ⚠️ `[Training]` 다음 줄도 같은 모양이라, 그걸 섞으면 train 지표를 val 로 읽는다
    (2026-11-26 에 실제로 그렇게 오독했다).
    """
    pat = re.compile(r"minade_k: ([\d.]+)")
    lines = Path(path).read_text().splitlines()
    out, ep = [], None
    for i, s in enumerate(lines):
        if s.startswith("Epoch "):
            ep = int(s.split()[1])
        elif s.startswith("[Validation]") and i + 1 < len(lines):
            m = pat.search(lines[i + 1])
            if m:
                out.append((ep, float(m.group(1))))
    return out


# ------------------------------------------------------------------ 교차검증
def gt_sum_dh_numpy(n_scene, obs=OBS):
    """정답 `|ΣΔh|` 분위를 **pickle → numpy** 로 다시 센다. torch 도 AH 도 안 쓴다.

    ⚠️ 대는 상대는 진단기의 `sum_dh_raw`(게이트 전)다. `sum_dh`(저속 동결·clip 후)와
       대면 **다른 양을 비교하는 것**이라 당연히 어긋난다 — 2026-11-26 에 그렇게 짰다가
       검증기가 잡았다(진단기 최대 181° vs numpy 1009°).
    """
    import pandas as pd
    files = sorted(FEAT_VAL.glob("*.pkl"))[:n_scene]
    if not files:
        return None
    tot = []
    for f in files:
        # pickle 은 한 행짜리 DataFrame 이고 TRAJS 칸 안에 dict 가 들어 있다
        t = pd.read_pickle(f)["TRAJS"].values[0]
        pos = np.asarray(t["trajs_pos"][0], float)                  # [110,2] 대상 차
        has = np.asarray(t["has_flags"][0]).astype(bool)
        if not has[obs:].all():
            continue
        d = np.diff(pos[obs - 1:], axis=0)                          # 관측 마지막을 닻으로
        h = np.arctan2(d[:, 1], d[:, 0])
        dh = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
        tot.append(abs(np.degrees(dh).sum()))
    return np.percentile(tot, [50, 90, 99, 100]).tolist() if tot else None


# ------------------------------------------------------------------ 그림
def fig_collapse(plt, S):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))

    # ① 증상 — 오차 곡선. 검증 에폭마다 점(요건 #7), 끝점 라벨 말고 범례(겹침 방지)
    for lab, log, col in RUNS:
        c = val_curve(SIMPL / log)
        if not c:
            continue
        x, y = zip(*c)
        n = f"{lab}  (10에폭에서 중단)" if "R2" in lab else lab
        ax1.plot(x, y, "-o", color=col, ms=3.4, label=n,
                 lw=2.2 if "R2" in lab else 1.6,
                 zorder=3 if "R2" in lab else 2)
    ax1.set_xlim(0, 10.4)
    ax1.set_xlabel(f"에폭 (R2 가 돈 공통 구간 1~{S['cmp_ep']})")
    ax1.set_ylabel("val minADE6 [m]  ↓ 낮을수록 좋다")
    ax1.set_title("① R2 는 R0 의 2배에서 평평하다", loc="left", fontweight="bold")
    ax1.legend(loc="center left")
    # ⚠️ 화살표는 **두 런이 다 있는** 에폭에 둔다. R2 가 없는 자리에 그으면 없는 비교가 된다
    ax1.annotate("", xy=(S["cmp_ep"], S["r2_ep"]), xytext=(S["cmp_ep"], S["r0_ep"]),
                 arrowprops=dict(arrowstyle="<->", color=C.MUTED, lw=1.0))
    ax1.text(S["cmp_ep"] + 0.25, (S["r2_ep"] + S["r0_ep"]) / 2, f"×{S['ratio']:.1f}",
             ha="left", va="center", fontsize=9.5, color=C.INK2, fontweight="bold")

    # ② 원인 — ΣΔh 분위. 로그축(세 자리 차이라 선형으로는 모델 선이 바닥에 붙는다)
    x = np.arange(len(QLAB))
    for lab, key, col, mk in (("정답", "gt_sum", C.INK, "s"),
                              ("모델 · 학습 전", "init_sum", C.MUTED, "^"),
                              ("모델 · 10에폭", "ep10_sum", C.C_RED, "o")):
        ax2.plot(x, S[key], "-" + mk, color=col, ms=6, label=lab,
                 lw=2.2 if "10에폭" in lab else 1.6,
                 zorder=3 if "10에폭" in lab else 2)
    ax2.set_yscale("log")
    ax2.set_xticks(x, QLAB)
    ax2.set_xlabel("640 장면 분위")
    ax2.set_ylabel("모드별 총 방향 변화 |ΣΔh| [°]")
    ax2.set_title("② 원인 — 방향 채널이 0 으로 붕괴 (mode collapse)",
                  loc="left", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0.09, 420)
    ax2.axhspan(0.09, 1.2, color=C.C_RED, alpha=0.07, zorder=0)
    # 글상자는 빈 자리(붉은 띠의 왼쪽 아래)에 둔다 — 마지막 점(최대)에 닿지 않게
    ax2.text(-0.12, 0.135,
             f"학습이 {S['shrink']:.0f}배 줄였다 — 학습 전보다도 작다\n"
             f"(중앙 {S['init_sum'][0]:.2f}° → {S['ep10_sum'][0]:.2f}°)",
             ha="left", va="center", fontsize=8.6, color=C.C_RED,
             bbox=dict(fc=C.SURF, ec=C.C_RED, lw=0.7, pad=3))

    fig.tight_layout()
    return C.savefig(fig, OUT / "m1_dh_collapse.png")


# ------------------------------------------------------------------ 숫자
def collect():
    d0 = json.loads((OUT / "init.json").read_text())
    d1 = json.loads((OUT / "ep10.json").read_text())
    assert d0["gt"]["sum_dh"] == d1["gt"]["sum_dh"], \
        "두 판의 정답 행이 다르다 — 같은 장면을 안 쓴 것이다"

    c0 = {e: v for e, v in val_curve(SIMPL / "r0.log")}
    c2 = {e: v for e, v in val_curve(SIMPL / RUNS[2][1])}
    ep = max(e for e in c2 if e in c0)                 # 공통 마지막 검증 에폭

    S = {"n_scene": d1["focal"]["n_scene"],
         "gt_sum": d1["gt"]["sum_dh"], "gt_sum_raw": d1["gt"]["sum_dh_raw"],
         "init_sum": d0["focal"]["sum_dh"], "ep10_sum": d1["focal"]["sum_dh"],
         "init_spread": d0["focal"]["mode_spread"], "ep10_spread": d1["focal"]["mode_spread"],
         "init_ratio": d0["focal"]["dh_ratio"], "ep10_ratio": d1["focal"]["dh_ratio"],
         "sat_dh": d1["focal"]["sat_dh"], "sat_a": d1["focal"]["sat_a"],
         "gt_sat_dh": d1["gt"]["sat_dh"], "gt_sat_a": d1["gt"]["sat_a"],
         "cmp_ep": ep, "r0_ep": c0[ep], "r2_ep": c2[ep]}
    S["ratio"] = S["r2_ep"] / S["r0_ep"]
    S["shrink"] = S["init_sum"][0] / S["ep10_sum"][0]
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    S = collect()
    print(f"공통 검증 에폭 {S['cmp_ep']}  ·  R0 {S['r0_ep']:.3f}  R2 {S['r2_ep']:.3f}"
          f"  →  ×{S['ratio']:.2f}")
    print(f"|ΣΔh| 중앙   정답 {S['gt_sum'][0]:.2f}°  학습전 {S['init_sum'][0]:.2f}°"
          f"  10에폭 {S['ep10_sum'][0]:.2f}°   (학습이 {S['shrink']:.1f}배 줄였다)")
    print(f"선회율 포화  모델 {S['sat_dh'] * 100:.2f}%  정답 {S['gt_sat_dh'] * 100:.2f}%"
          f"   → clip 은 걸리지 않았다")

    if not a.no_verify:
        v = gt_sum_dh_numpy(S["n_scene"])
        if v is None:
            print("⚠️ 교차검증 건너뜀 — feature pickle 을 못 찾았다")
        else:
            print("\n교차검증 — 게이트 **전** 생짜 합끼리 (pickle → numpy, torch 없음)")
            print("            " + "  ".join(f"{q:>7s}" for q in QLAB))
            print("  진단기    " + "  ".join(f"{x:7.2f}" for x in S["gt_sum_raw"]))
            print("  numpy     " + "  ".join(f"{x:7.2f}" for x in v))
            bad = [q for q, x, y in zip(QLAB, S["gt_sum_raw"], v)
                   if not (0.5 <= (x + 1e-9) / (y + 1e-9) <= 2.0)]
            if bad:
                raise SystemExit(f"❌ 두 경로가 2배 넘게 어긋난다: {bad} — 그림을 내지 않는다")
            print("  ✅ 네 분위 모두 2배 안쪽")
            S["verify_numpy"] = v

    plt = C.setup_mpl()
    p = fig_collapse(plt, S)
    (OUT / "numbers.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
    print(f"\n그림 → {p}\n숫자 → {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
