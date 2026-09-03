"""
postprocess_rules.py - 규칙 위반 모드의 확률을 깎는 후처리 (①안).

학습된 모델은 그대로 두고, 예측 6개 중 규칙을 어긴 것의 확률을 0으로 만들고 재정규화한다.
모델 구조·재학습이 필요 없어 지금 있는 가중치로 즉시 적용된다.

위반 판정 3종 (각각 따로 저장하므로 조합을 바꿔가며 볼 수 있다):
  offroad  : 궤적의 한 점이라도 drivable_areas 밖
  wrongway : 끝점의 진행방향이 가장 가까운 차로 방향과 135도 넘게 어긋남
  unreach  : 끝점의 최근접 차로가 규칙상 도달 가능 집합에 없음

주의: minADE/minFDE 는 6개 중 최소라 후처리로 바뀌지 않는다. 바뀌는 것은
'모델이 1순위로 내놓는 답'과 확률가중 지표다. 그것이 실제로 쓰이는 값이다.

  python src/postprocess_rules.py --in runs/traj_lane_rules.npz --out runs/pp_lane_rules.npz
"""
import argparse, json, sys
from pathlib import Path
sys.path.append("src")

import numpy as np
from matplotlib.path import Path as MplPath

from dataset_map import _rotation_matrix
from lane_graph import LaneGraph, REACH_MARGIN_M

MAP = Path("/data/argoverse2/motion_forecasting/val")
DT = 0.1


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/traj_lane_rules.npz")
    ap.add_argument("--out", default="runs/pp_lane_rules.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    sid, org, th = d["scenario_id"], d["origin"], d["theta"]
    hist, gt, pred, probs = d["hist"], d["gt"], d["pred"], d["probs"]
    N = len(sid) if args.limit == 0 else min(args.limit, len(sid))

    off = np.zeros((N, 6), bool); wrong = np.zeros((N, 6), bool); unre = np.zeros((N, 6), bool)
    gt_off = np.zeros(N, bool)
    ctx = {k: np.zeros(N, np.float32) for k in
           ["speed_kph", "n_reach", "gt_turn_deg", "gt_dist", "at_inter", "n_lane"]}

    for i in range(N):
        s = str(sid[i])
        raw = json.load(open(MAP / s / f"log_map_archive_{s}.json"))
        polys = [MplPath(np.array([[p["x"], p["y"]] for p in v["area_boundary"]]))
                 for v in raw.get("drivable_areas", {}).values()]
        g = LaneGraph.from_json_dict(raw, centerline="json")
        R = _rotation_matrix(float(th[i]))
        to_city = lambda a: a @ R.T + org[i]

        # --- 장면 맥락 (왜 이렇게 예측했나를 설명하는 값들) ---
        v_now = np.linalg.norm(hist[i][-1] - hist[i][-2]) / DT
        ctx["speed_kph"][i] = v_now * 3.6
        ctx["gt_dist"][i] = np.linalg.norm(gt[i][-1] - gt[i][0])
        d0, d1 = unit(gt[i][2] - gt[i][0]), unit(gt[i][-1] - gt[i][-3])
        ctx["gt_turn_deg"][i] = np.degrees(np.arctan2(np.cross(d0, d1), np.dot(d0, d1)))
        ctx["n_lane"][i] = len(g.lanes)

        h0 = np.array([np.cos(float(th[i])), np.sin(float(th[i]))])
        starts = g.candidate_lanes(org[i].astype(np.float64), h0, path=to_city(hist[i]))
        budget = max(20.0, v_now * 6.0) + REACH_MARGIN_M
        reach = g.reachable(starts, budget) if starts else set()
        ctx["n_reach"][i] = len(reach)
        cur = g.nearest_lane(org[i].astype(np.float64), h0)
        ctx["at_inter"][i] = float(getattr(g.lanes[cur], "is_intersection", False)) if cur in g.lanes else 0.0

        def outside(pts):
            if not polys:
                return np.zeros(len(pts), bool)
            ins = np.zeros(len(pts), bool)
            for P in polys:
                ins |= P.contains_points(pts)
            return ~ins

        gt_off[i] = outside(to_city(gt[i])).any()

        for m in range(6):
            p = to_city(pred[i][m])
            off[i, m] = outside(p).any()
            end_dir = unit(p[-1] - p[-4])
            # 방향 필터를 걸면 candidate_lanes 가 대향차로를 아예 빼버려서
            # 역주행이 구조적으로 검출되지 않는다. 기하적으로만 고른다.
            near = g.candidate_lanes(p[-1], None, radius=5.0, top_k=8)
            if not near:
                unre[i, m] = True
                wrong[i, m] = False          # 차로가 없으면 방향 판정 불가
                continue
            # 도달성: 반경 5m 안 차로 중 하나라도 도달 가능하면 통과 (문서에서 검증된 프로토콜)
            unre[i, m] = not any(l in reach for l in near)
            # 역주행: 가장 가까운 차로(방향 무관)의 진행방향과 비교
            ld = unit(np.asarray(g.lanes[near[0]].direction)[:2])
            wrong[i, m] = float(np.dot(ld, end_dir)) < -0.7071
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N}", flush=True)

    viol = off | wrong | unre
    newp = probs[:N] * (~viol)
    allbad = newp.sum(axis=1) <= 1e-9
    newp[allbad] = probs[:N][allbad]                 # 6개 다 위반이면 원래 확률 유지
    newp = newp / newp.sum(axis=1, keepdims=True)

    np.savez_compressed(args.out, scenario_id=sid[:N], offroad=off, wrongway=wrong,
                        unreach=unre, viol=viol, prob_new=newp.astype(np.float32),
                        gt_offroad=gt_off, all_violated=allbad,
                        **{k: v for k, v in ctx.items()})
    print(f"\n=== 후처리 결과 ({N} 시나리오) ===")
    print(f"  위반 모드 비율   도로이탈 {off.mean()*100:.2f}%  역주행 {wrong.mean()*100:.2f}%  "
          f"도달불가 {unre.mean()*100:.2f}%  합집합 {viol.mean()*100:.2f}%")
    print(f"  6개 전부 위반    {allbad.mean()*100:.2f}%  (이때는 후처리 미적용)")
    print(f"  정답(gt) 도로이탈 {gt_off.mean()*100:.2f}%  ← 측정 오탐 상한")
    old1, new1 = probs[:N].argmax(1), newp.argmax(1)
    print(f"  1순위 모드가 바뀐 시나리오 {np.mean(old1 != new1)*100:.1f}%")
    print(f"  저장 -> {args.out}")


if __name__ == "__main__":
    main()
