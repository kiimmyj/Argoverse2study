"""
explore_map.py - HD map(log_map_archive_*.json) 구조 탐색.

시나리오 하나의 지도를 열어:
  - 어떤 요소(차선, 횡단보도 등)가 있는지
  - 차선 하나가 어떤 정보를 담는지 (중심선 좌표, 경계, 연결 등)
  - focal 궤적과 지도를 겹쳐 시각화 (나중 HD map plotting의 기반)

사용:
  python src/explore_map.py                 # train 첫 시나리오
  python src/explore_map.py --sid <id>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

DATA_ROOT = Path("/data/argoverse2/motion_forecasting")
FOCAL_COLOR, AV_COLOR, OTHER_COLOR = "#ECA25B", "#007672", "#9DC3E6"
LANE_COLOR = "#BBBBBB"


def pick_dir(split, sid):
    split_dir = DATA_ROOT / split
    if sid:
        return split_dir / sid
    return next(p for p in sorted(split_dir.iterdir()) if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--sid", default=None)
    args = ap.parse_args()

    sdir = pick_dir(args.split, args.sid)
    sid = sdir.name
    map_path = sdir / f"log_map_archive_{sid}.json"
    scn_path = sdir / f"scenario_{sid}.parquet"
    if not map_path.exists():
        raise SystemExit(f"지도 파일 없음: {map_path}")

    # 1) raw json 최상위 구조 --------------------------------------
    with open(map_path) as f:
        raw = json.load(f)
    print("=" * 60)
    print("① MAP JSON 최상위 키")
    for k in raw.keys():
        v = raw[k]
        n = len(v) if isinstance(v, (dict, list)) else "-"
        print(f"   {k:30s}  (원소 {n}개)")

    # 2) av2-api 로 지도 로드 (구조화된 접근) ----------------------
    amap = ArgoverseStaticMap.from_json(map_path)
    lane_ids = list(amap.vector_lane_segments.keys())
    print(f"\n② 차선(lane segment): {len(lane_ids)}개")

    # 차선 하나 뜯어보기
    lane = amap.vector_lane_segments[lane_ids[0]]
    print(f"\n③ 차선 하나 (id={lane_ids[0]}) 속성")
    for attr in ["id", "is_intersection", "lane_type",
                 "right_lane_boundary", "left_lane_boundary",
                 "predecessors", "successors"]:
        if hasattr(lane, attr):
            val = getattr(lane, attr)
            # 경계선은 점 개수만 표시
            if "boundary" in attr and hasattr(val, "xyz"):
                print(f"   {attr:22s}: {val.xyz.shape[0]}개 점")
            else:
                print(f"   {attr:22s}: {val}")

    # 중심선 좌표 (예측 모델이 쓸 핵심!)
    centerline = amap.get_lane_segment_centerline(lane_ids[0])  # (N, 3)
    print(f"\n④ 중심선(centerline) 좌표: shape {centerline.shape}")
    print(f"   처음 3점:\n{np.round(centerline[:3], 1)}")

    # 3) focal 궤적 + 지도 겹쳐 그리기 ----------------------------
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scn_path)
    focal = next(t for t in scenario.tracks if t.track_id == scenario.focal_track_id)
    fxy = np.array([s.position for s in focal.object_states])

    fig, ax = plt.subplots(figsize=(12, 12))
    # 모든 차선 중심선 (회색)
    for lid in lane_ids:
        c = amap.get_lane_segment_centerline(lid)
        ax.plot(c[:, 0], c[:, 1], color=LANE_COLOR, lw=0.8, zorder=1)
    # focal 궤적 (주황)
    ax.plot(fxy[:, 0], fxy[:, 1], color=FOCAL_COLOR, lw=3, zorder=5, label="focal")
    ax.scatter(fxy[0, 0], fxy[0, 1], color=FOCAL_COLOR, s=40, zorder=6)
    ax.scatter(fxy[-1, 0], fxy[-1, 1], color=FOCAL_COLOR, marker="X", s=60, zorder=6)

    # focal 주변으로 확대
    cx, cy = fxy[:, 0].mean(), fxy[:, 1].mean()
    ax.set_xlim(cx - 60, cx + 60)
    ax.set_ylim(cy - 60, cy + 60)
    ax.set_aspect("equal")
    ax.set_title(f"{sid[:8]} @ {scenario.city_name}  (focal on HD map)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(f"map_{sid[:8]}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n지도+궤적 시각화 저장 → {out.resolve()}")


if __name__ == "__main__":
    main()
