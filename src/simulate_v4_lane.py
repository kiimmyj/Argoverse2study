"""
simulate_v4_lane.py - ④안(차로 기준 좌표계)의 출력 공간을 시뮬레이션한다.

학습된 ④ 모델은 아직 없다. 대신 지금 모델의 예측을 ④가 표현할 수 있는 형태로
투영해서, 구조적 제약이 무엇을 보장하는지 보인다.

  예측 (x,y) 자유좌표  →  [규칙이 허용하는 경로 선택 + (s,d) 로 표현 + d 클립]  →  (x,y)

투영 후에는 정의상 위반이 나올 수 없다. 그것을 실제로 재서 확인한다.

  python src/simulate_v4_lane.py --limit 300 --out runs/v4sim.npz
"""
import argparse, json, sys
from pathlib import Path
sys.path.append("src")

import numpy as np
from matplotlib.path import Path as MplPath

from dataset_map import _rotation_matrix
from lane_graph import LaneGraph, REACH_MARGIN_M
import lane_frame as lf

MAP = Path("/data/argoverse2/motion_forecasting/val")


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/traj_lane_rules.npz")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--half-w", dest="half_w", type=float, default=lf.LANE_HALF_W,
                    help="횡오프셋 상한 [m]. 차로 반폭")
    ap.add_argument("--out", default="runs/v4sim.npz")
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    N = min(args.limit, len(d["scenario_id"]))
    proj = np.full((N, 6, 60, 2), np.nan, np.float32)
    ok = np.zeros((N, 6), bool)
    nroute = np.zeros(N, np.int32)
    off_b = np.zeros((N, 6), bool); off_a = np.zeros((N, 6), bool)
    wr_b = np.zeros((N, 6), bool);  wr_a = np.zeros((N, 6), bool)

    for i in range(N):
        sid = str(d["scenario_id"][i])
        raw = json.load(open(MAP / sid / f"log_map_archive_{sid}.json"))
        g = LaneGraph.from_json_dict(raw, centerline="json")
        polys = [MplPath(np.array([[p["x"], p["y"]] for p in v["area_boundary"]]))
                 for v in raw.get("drivable_areas", {}).values()]
        R = _rotation_matrix(float(d["theta"][i]))
        to_city = lambda a: a @ R.T + d["origin"][i]
        to_norm = lambda a: (a - d["origin"][i]) @ R

        hist = to_city(d["hist"][i])
        v_now = np.linalg.norm(d["hist"][i][-1] - d["hist"][i][-2]) / 0.1
        h0 = np.array([np.cos(float(d["theta"][i])), np.sin(float(d["theta"][i]))])
        starts = g.candidate_lanes(d["origin"][i].astype(np.float64), h0, path=hist)
        reach = g.reachable(starts, max(20.0, v_now * 6.0) + REACH_MARGIN_M) if starts else set()
        routes = lf.build_routes(g, starts, reach)
        nroute[i] = len(routes)

        def viol(p):
            """도로이탈 / 역주행 판정.
            역주행은 반경 5m 안 차로 중 '하나라도' 방향이 맞으면 아닌 것으로 본다.
            최근접 1개로 판정하면 차로 가장자리에서 옆 대향차로로 스냅되어 오탐이 크다
            (정답 궤적 기준 오탐 5.70% → 0.30%)."""
            o = True
            if polys:
                ins = np.zeros(len(p), bool)
                for P in polys:
                    ins |= P.contains_points(p)
                o = (~ins).any()
            ed = unit(p[-1] - p[-4])
            near = g.candidate_lanes(p[-1], None, radius=5.0, top_k=8)
            w = False
            if near:
                cos = [float(np.dot(unit(np.asarray(g.lanes[l].direction)[:2]), ed)) for l in near]
                w = max(cos) < -0.7071
            return o, w

        for m in range(6):
            p_city = to_city(d["pred"][i][m])
            off_b[i, m], wr_b[i, m] = viol(p_city)
            r = lf.project(p_city, routes, half_w=args.half_w)
            if r is None:
                continue
            ok[i, m] = True
            proj[i, m] = to_norm(r["traj"]).astype(np.float32)
            off_a[i, m], wr_a[i, m] = viol(r["traj"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N}", flush=True)

    np.savez_compressed(args.out, scenario_id=d["scenario_id"][:N], proj=proj, ok=ok,
                        n_route=nroute, off_before=off_b, off_after=off_a,
                        wrong_before=wr_b, wrong_after=wr_a)

    gt, pred = d["gt"][:N], d["pred"][:N]
    ade_b = np.linalg.norm(pred - gt[:, None], axis=3).mean(2)
    ade_a = np.linalg.norm(proj - gt[:, None], axis=3).mean(2)
    m = ok
    print(f"\n=== ④ 출력 공간 시뮬레이션 ({N} 시나리오) ===")
    print(f"  경로를 못 만든 시나리오 {int((~ok).all(1).sum())}개 "
          f"(차로 후보 0개) — ④가 예측 자체를 못 하는 경우")
    print(f"  시나리오당 경로 후보 평균 {nroute.mean():.1f}개\n")
    print(f"  {'지표':<26}{'투영 전':>10}{'투영 후':>10}")
    print(f"  {'도로이탈 (모드)':<26}{off_b[m].mean()*100:>9.2f}%{off_a[m].mean()*100:>9.2f}%")
    print(f"  {'역주행 (모드)':<26}{wr_b[m].mean()*100:>9.2f}%{wr_a[m].mean()*100:>9.2f}%")
    print(f"  {'minADE6':<26}{ade_b.min(1).mean():>10.3f}{np.nanmin(np.where(m, ade_a, np.nan), 1).mean():>10.3f}")
    print(f"  {'모드 평균 ADE':<26}{ade_b[m].mean():>10.3f}{ade_a[m].mean():>10.3f}")
    print(f"\n  저장 -> {args.out}")


if __name__ == "__main__":
    main()
