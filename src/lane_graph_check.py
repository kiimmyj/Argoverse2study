"""
lane_graph_check.py - lane_graph.py 의 이동 규칙을 실제 궤적으로 검증한다.

규칙이 옳다면 **focal 이 실제로 지나간 미래 차로는 규칙상 도달 가능해야 한다.**
너무 느슨하면 도달 집합이 지도 전체가 되어 쓸모가 없고, 너무 빡빡하면 실제로 간 곳을
못 간다고 하게 된다. 규칙을 켠 경우와 끈 경우를 나란히 재서 둘 다 잡는다.

  python src/lane_graph_check.py --limit 200
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.append("src")
from av2.datasets.motion_forecasting import scenario_serialization
from lane_graph import LaneGraph, Move, REACH_MARGIN_M, LANE_CHANGE_COST_M

OBS_LEN = 50
MIN_MOVE_M = 5.0          # 사실상 정지한 차는 차로 판정이 무의미해 제외


def focal_track(sdir):
    s = scenario_serialization.load_argoverse_scenario_parquet(
        sdir / f"scenario_{sdir.name}.parquet")
    f = next(t for t in s.tracks if t.track_id == s.focal_track_id)
    st = sorted(f.object_states, key=lambda x: x.timestep)
    return (np.array([x.position for x in st], dtype=np.float64),
            np.array([x.heading for x in st], dtype=np.float64))


def _naive_reachable(g, starts, budget):
    """규칙을 끈 비교군: 방향도 표시도 안 보고 successors + 좌우 인접을 전부 허용."""
    best = {s: budget for s in starts}
    q = deque(starts)
    while q:
        c = q.popleft()
        L = g.lanes[c]
        for nb in L.successors + [L.left_neighbor, L.right_neighbor]:
            if nb is None or nb not in g.lanes:
                continue
            cost = L.length if nb in L.successors else LANE_CHANGE_COST_M
            r = best[c] - cost
            if r >= 0 and r > best.get(nb, -1.0) + 1e-6:
                best[nb] = r
                q.append(nb)
    return set(best)


def run(limit):
    root = Path("/data/argoverse2/motion_forecasting/val")
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()][:limit]

    rows, skip, rej, edges, n_lane = [], 0, {}, {m.value: 0 for m in Move}, 0
    for d in dirs:
        raw = json.loads((d / f"log_map_archive_{d.name}.json").read_text())
        g = LaneGraph.from_json_dict(raw)
        n_lane += len(g.lanes)
        for v in g.edges.values():
            for _, mv in v:
                edges[mv.value] += 1
        for v in g.rejected.values():
            for _, mv, why in v:
                rej[f"{mv.value}:{why}"] = rej.get(f"{mv.value}:{why}", 0) + 1

        pos, head = focal_track(d)
        if len(pos) < OBS_LEN + 2:
            skip += 1
            continue
        dist = float(np.linalg.norm(np.diff(pos[OBS_LEN - 1:], axis=0), axis=1).sum())
        if dist < MIN_MOVE_M:
            skip += 1
            continue
        h0 = np.array([np.cos(head[OBS_LEN - 1]), np.sin(head[OBS_LEN - 1])])
        h1 = np.array([np.cos(head[-1]), np.sin(head[-1])])

        S = g.candidate_lanes(pos[OBS_LEN - 1], h0)
        E = g.candidate_lanes(pos[-1], h1, top_k=8)
        if not S or not E:
            skip += 1
            continue

        budget = dist + REACH_MARGIN_M
        R = g.reachable(S, budget)
        N = _naive_reachable(g, S, budget)
        opp = lambda ss: sum(1 for l in ss if float(g.lanes[l].direction @ h0) <= 0)
        rows.append((any(e in R for e in E), any(e in N for e in E),
                     len(R), len(N), opp(R), opp(N), len(g.lanes)))

    a = np.array(rows, dtype=float)
    print(f"시나리오 {len(dirs)}개 (판정 {len(rows)}, 제외 {skip}) / 차로 {n_lane:,}개\n")
    print(f"{'':14}{'적중률':>10}{'후보 차로':>12}{'지도 대비':>10}{'역주행 차로':>13}")
    for i, lab in ((0, "규칙 적용"), (1, "규칙 없음")):
        j = 0 if i == 0 else 1
        print(f"{lab:14}{100*a[:,i].mean():9.1f}%{a[:,2+j].mean():11.1f}개"
              f"{100*(a[:,2+j]/a[:,6]).mean():9.1f}%{a[:,4+j].mean():12.2f}개")
    print(f"\n→ 적중률 {100*a[:,0].mean():.1f}% 를 지키면서 후보를 "
          f"{100*(1-a[:,2].mean()/a[:,3].mean()):.0f}% 줄이고, "
          f"역주행 차로를 {100*(1-a[:,4].mean()/a[:,5].mean()):.0f}% 걷어냄")

    print(f"\n[엣지] {edges}")
    print(f"[규칙에 걸려 잘린 인접 연결]")
    for k, v in sorted(rej.items(), key=lambda x: -x[1])[:6]:
        print(f"   {k:30} {v:6,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    run(ap.parse_args().limit)
