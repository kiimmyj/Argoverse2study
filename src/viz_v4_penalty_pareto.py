"""
viz_v4_penalty_pareto.py - penalty 설정 비교: 정확도 vs 좌표 기준 실현가능성 (2026-10-06).

입력은 `runs/v4_xy_viol.json` (src/score_xy_viol.py 산출). 세 칸으로 나눈다.
  (a) Pareto — minADE6 [m] vs top-1 궤적의 |Δψ| > 7.3°/step 비율 [%]. seed 3판을 점으로 모두 찍고
      평균을 큰 표식으로 둔다. 정답 라벨의 비율을 세로선으로 같이 둔다.
  (b) 정답 회전량 구간별 |Δψ| 중앙값 [°] — 정답은 회전에서 커진다(0.12 -> 1.10°). 그 구조를
      재현하는지 본다. 한 분포로 뭉치지 않는다(요건 5), 축·단위·색은 세 칸 모두 같다(요건 6).
  (c) 같은 구간별 ADE 를 'no penalty' 기준 차이로 — penalty 의 대가가 어디에 몰리는지.

  python src/viz_v4_penalty_pareto.py
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "v4_xy_viol.json"
OUT = C.VIZ_ROOT / "penalty"
BINS = ["직진 <5°", "완만 5-30°", "회전 >=30°"]

# (표시이름, 태그틀, 색, 표식) — 색은 '무엇을 누르나'로 묶는다: 회색 없음 / 주황 action / 파랑 coordinate
CONDS = [   # (이름, 태그틀, 색, 표식, (a) 칸 글자 자리)
    ("no penalty",         "v4_l4nw_ah2_full_sm0_cos30_s%d",   C.MUTED,    "o", (-9, -14, "right")),
    ("coord penalty 1.0",  "v4_l4nw_ah2_full_smxy1_cos30_s%d", C.C_BLUE,   "s", (0, -17, "center")),
    ("coord penalty 3.0",  "v4_l4nw_ah2_full_smxy3_cos30_s%d", "#184f95",  "D", (11, 3, "left")),
    ("+ lane-yaw 1.0",     "v4_l4nw_ah2_full_smxy1_ly1_cos30_s%d", C.C_GREEN, "P", (0, -17, "center")),
    ("+ lane-yaw 3.0",     "v4_l4nw_ah2_full_smxy1_ly3_cos30_s%d", C.C_AQUA, "X", (-10, 7, "right")),
    ("action penalty 1.0", "v4_l4nw_ah2_full_sm1_cos30_s%d",   C.C_ORANGE, "^", (-10, -3, "right")),
    ("action+coord",       "v4_l4nw_ah2_full_smboth_cos30_s%d", C.C_RED,   "v", (11, 1, "left")),
]
BASE = "v4_l4nw_ah2_full_sm0_cos30_s%d"        # (c) 의 기준


def seeds(R, pat):
    return [R[pat % s] for s in (0, 1, 2) if pat % s in R]


def main():
    plt = C.setup_mpl()
    R = json.loads(SRC.read_text())
    gt = R[CONDS[0][1] % 0]["gt"]
    gt_by = R[CONDS[0][1] % 0]["by_turn"]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))

    # ---------------- (a) Pareto
    ax = axes[0]
    ax.axvline(gt["over_pct"], color=C.INK, lw=1.0, ls="--", zorder=1)
    ax.text(gt["over_pct"], 0.99, f" 정답 라벨 {gt['over_pct']:.2f}%", color=C.INK, fontsize=7.8,
            transform=ax.get_xaxis_transform(), va="top", ha="left")
    for name, pat, col, mk, (dx, dy, ha) in CONDS:
        rs = seeds(R, pat)
        if not rs:
            continue
        x = [r["xy_top1"]["over_pct"] for r in rs]
        y = [r["minADE6"] for r in rs]
        ax.scatter(x, y, s=16, color=col, alpha=0.45, marker=mk, lw=0, zorder=2)
        ax.scatter([st.mean(x)], [st.mean(y)], s=104, color=col, marker=mk,
                   edgecolor=C.SURF, lw=1.1, zorder=3)
        ax.annotate(name, (st.mean(x), st.mean(y)), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.2, color=col, ha=ha, va="center", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("top-1 궤적의 |Δψ| > 7.3°/step 비율 [%]  (낮을수록 실현가능)")
    ax.set_ylabel("minADE6 [m]  (낮을수록 정확)")
    ax.set_title("(a) 정확도 vs 실현가능성 — 좌표 기준", loc="left")
    ax.set_xticks([0.35, 1, 2, 5, 10])
    ax.set_xticklabels(["0.35", "1", "2", "5", "10"])

    # ---------------- (b) 구간별 |Δψ| 중앙값
    ax = axes[1]
    xs = range(len(BINS))
    ax.plot(xs, [gt_by[b]["gt"]["p50_deg"] for b in BINS], color=C.INK, lw=2.2, marker="*",
            ms=11, label="정답 라벨", zorder=4)
    for name, pat, col, mk, _ in CONDS:
        rs = seeds(R, pat)
        if not rs:
            continue
        ys = [st.mean(r["by_turn"][b]["xy_top1"]["p50_deg"] for r in rs) for b in BINS]
        ax.plot(xs, ys, color=col, marker=mk, ms=6, lw=1.8, label=name)
    ax.set_yscale("log")
    ax.set_xticks(list(xs)); ax.set_xticklabels(BINS)
    ax.set_xlabel("정답의 총 진행방향 변화량으로 나눈 구간")
    ax.set_ylabel("top-1 |Δψ| 중앙값 [°/step]")
    ax.set_title("(b) 회전 구조를 재현하나 — 정답은 9배 커진다", loc="left")
    ax.legend(loc="upper left", ncol=2, fontsize=7.0)

    # ---------------- (c) 구간별 ADE 차이
    ax = axes[2]
    others = [c for c in CONDS if c[1] != BASE and (c[1] % 0) in R]
    w = 0.78 / max(len(others), 1)
    for j, (name, pat, col, mk, _) in enumerate(others):
        ys = [st.mean(R[pat % s]["by_turn"][b]["minADE6"] - R[BASE % s]["by_turn"][b]["minADE6"]
                      for s in (0, 1, 2)) for b in BINS]
        off = (j - (len(others) - 1) / 2.0) * w
        ax.bar([x + off for x in xs], ys, width=w, color=col, label=name, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x + off, y + 0.004, f"{y:+.3f}", ha="center", va="bottom",
                    fontsize=6.0, color=col, rotation=90)
    ax.axhline(0, color=C.AXIS, lw=1.0)
    ax.set_xticks(list(xs)); ax.set_xticklabels(BINS)
    ax.set_ylim(-0.03, 0.25)
    ax.set_xlabel("정답의 총 진행방향 변화량으로 나눈 구간")
    ax.set_ylabel("minADE6 차이 [m]  (no penalty 대비, 같은 seed)")
    ax.set_title("(c) penalty 의 대가는 회전에 몰린다", loc="left")
    ax.legend(loc="upper left", fontsize=7.6)

    fig.suptitle("penalty 설정 비교 — val 24,988 · 조건당 seed 3판 · 30에폭 cosine · best 체크포인트",
                 x=0.012, ha="left", fontsize=10.5, color=C.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = C.savefig(fig, OUT / "penalty_pareto.png")
    print("저장", p)

    # 숫자 표 (문서에 그대로 옮긴다)
    rows = []
    for name, pat, _, _, _ in CONDS:
        rs = seeds(R, pat)
        if not rs:
            continue
        rows.append({
            "name": name, "seeds": len(rs),
            "minADE6": st.mean(r["minADE6"] for r in rs),
            "minADE6_range": max(r["minADE6"] for r in rs) - min(r["minADE6"] for r in rs),
            "over_top1_pct": st.mean(r["xy_top1"]["over_pct"] for r in rs),
            "over_alive_pct": st.mean(r["xy_alive"]["over_pct"] for r in rs),
            "theta_over_pct": st.mean(r["theta_over_pct"] for r in rs),
            "p50_by_turn": {b: st.mean(r["by_turn"][b]["xy_top1"]["p50_deg"] for r in rs) for b in BINS},
            "over_by_turn": {b: st.mean(r["by_turn"][b]["xy_top1"]["over_pct"] for r in rs) for b in BINS},
            "ade_by_turn": {b: st.mean(r["by_turn"][b]["minADE6"] for r in rs) for b in BINS},
        })
    tbl = {"gt": {"over_top1_pct": gt["over_pct"], "p50_deg": gt["p50_deg"], "p99_deg": gt["p99_deg"],
                  "p50_by_turn": {b: gt_by[b]["gt"]["p50_deg"] for b in BINS},
                  "over_by_turn": {b: gt_by[b]["gt"]["over_pct"] for b in BINS},
                  "n_by_turn": {b: gt_by[b]["n"] for b in BINS}},
           "conditions": rows}
    q = OUT / "penalty_table.json"
    q.write_text(json.dumps(tbl, indent=1, ensure_ascii=False))
    print("저장", q)


if __name__ == "__main__":
    main()
