"""
visualize_rule_effect.py - 차로 규칙 유무를 같은 장면에서 나란히 비교한다.

왼쪽: 규칙 없음(대조군)  /  오른쪽: 규칙 포함
두 모델이 가장 다르게 예측한 장면을 골라서, 규칙이 무엇을 지웠는지 눈으로 본다.

  python src/visualize_rule_effect.py --n 4
  python src/visualize_rule_effect.py --pick offroad   # 대조군만 도로를 벗어난 장면
"""
import argparse, json, sys
from pathlib import Path
sys.path.append("src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

import hdmap_render as hr
from dataset_map import _rotation_matrix
from visualize_map import draw_case, style_legend

MAP = Path("/data/argoverse2/motion_forecasting/val")
RUNS = Path("runs")


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return {k: d[k] for k in d.files}


def offroad_mask(pred, origin, theta, sid):
    """(6,60,2) 정규화 → 각 모드가 도로를 벗어나는지 (6,)"""
    f = MAP / str(sid) / f"log_map_archive_{sid}.json"
    das = json.load(open(f)).get("drivable_areas", {})
    polys = [MplPath(np.array([[p["x"], p["y"]] for p in v["area_boundary"]])) for v in das.values()]
    if not polys:
        return np.zeros(6, bool)
    R = _rotation_matrix(float(theta))
    out = np.zeros(6, bool)
    for m in range(6):
        pts = pred[m] @ R.T + origin
        inside = np.zeros(len(pts), bool)
        for P in polys:
            inside |= P.contains_points(pts)
        out[m] = (~inside).any()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--pick", default="offroad", choices=["offroad", "fde"])
    ap.add_argument("--scan", type=int, default=400, help="후보를 훑을 시나리오 수")
    ap.add_argument("--out", default="rule_effect.png")
    args = ap.parse_args()

    A = load(RUNS / "traj_lane_norule.npz")     # 대조군
    B = load(RUNS / "traj_lane_rules.npz")      # 규칙 포함
    assert (A["scenario_id"] == B["scenario_id"]).all()

    # 후보 고르기
    cands = []
    for i in range(min(args.scan, len(A["gt"]))):
        sid = str(A["scenario_id"][i])
        if args.pick == "offroad":
            oa = offroad_mask(A["pred"][i], A["origin"][i], A["theta"][i], sid)
            ob = offroad_mask(B["pred"][i], B["origin"][i], B["theta"][i], sid)
            score = int(oa.sum()) - int(ob.sum())      # 대조군만 이탈할수록 큼
            if score > 0:
                cands.append((score, float(A["min_fde"][i]), i, int(oa.sum()), int(ob.sum())))
        else:
            score = float(A["min_fde"][i]) - float(B["min_fde"][i])
            if score > 0:
                cands.append((score, 0.0, i, 0, 0))
    cands.sort(key=lambda c: (-c[0], -c[1]))
    picks = cands[:args.n]
    if not picks:
        print("조건에 맞는 장면이 없다"); return
    print(f"후보 {len(cands)}개 중 {len(picks)}개 선택")

    fig, axes = plt.subplots(len(picks), 2, figsize=(15, 7.2 * len(picks)), facecolor=hr.BG)
    if len(picks) == 1:
        axes = axes[None, :]
    for r, (score, _, i, na, nb) in enumerate(picks):
        sid = str(A["scenario_id"][i])
        scene = hr.build_scene(MAP / sid / f"log_map_archive_{sid}.json",
                               A["origin"][i], float(A["theta"][i]))
        for c, (D, lab, nviol) in enumerate([(A, "no rules", na), (B, "with rules", nb)]):
            rec = {"hist": D["hist"][i], "gt": D["gt"][i],
                   "trajs": [D["pred"][i][m] for m in range(6)],
                   "probs": D["probs"][i], "best": int(D["best"][i]),
                   "min_ade": float(D["min_ade"][i]), "min_fde": float(D["min_fde"][i]),
                   "scene": scene}
            ax = axes[r, c]
            draw_case(ax, rec, lab, aspect=7.5 / 7.2, mark_lw=1.5, lw_scale=1.15)
            ax.set_title(f"{lab}  |  off-road modes {nviol}/6  |  "
                         f"minADE {rec['min_ade']:.2f}m  minFDE {rec['min_fde']:.2f}m",
                         fontsize=12, color=hr.TEXT)
        axes[r, 0].text(0.01, 0.99, f"{sid[:8]}", transform=axes[r, 0].transAxes,
                        color="#7a7f88", fontsize=9, va="top")
    style_legend(axes[0, 0], fontsize=10)
    fig.suptitle("Lane-rule effect: same scene, left = no rules, right = with rules",
                 fontsize=15, color=hr.TEXT, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=120, facecolor=hr.BG, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
