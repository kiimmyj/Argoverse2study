"""
dataset_lane.py - 차로 이동 규칙(lane_graph)을 피처로 얹은 Dataset.

dataset_map.py 는 지도를 '가까운 차선 20개의 중심선 (20,10,2)' 로만 준다. 좌표뿐이라
모델은 어느 차선이 대향차로인지, 넘을 수 있는 실선인지, 애초에 갈 수 있는 곳인지 모른다.

여기서는 같은 20개 차선에 규칙 피처 (20,10) 을 나란히 붙인다.

  [0:2] 진행방향 단위벡터 (focal 기준 회전)   ← 역주행 판단의 핵심
  [2]   focal 과 같은 방향인가
  [3]   교차로 내부인가
  [4:7] 차로 종류 (VEHICLE / BUS / BIKE)
  [7]   왼쪽으로 차선변경 가능한가
  [8]   오른쪽으로 차선변경 가능한가
  [9]   규칙상 도달 가능한 차로인가        ← 예측 공간을 절반으로 줄여주는 신호

궤적(x, y)은 dataset_map.py 와 비트 단위로 같고, 차선(lanes)도 밀리미터 이하(중앙값 3e-5 m)로
일치한다 — float32 연산 순서 차이일 뿐이다. 규칙 피처를 넣은 효과만 따로 볼 수 있다.
그러면서 ArgoverseStaticMap 객체를 만들지 않고, 20개를 고른 뒤 리샘플링해서 조금 더 빠르다.

  python src/dataset_lane.py        # 동작 확인 + dataset_map 대비 속도
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting import scenario_serialization

from dataset_map import OBS_LEN, PRED_LEN, N_LANES, N_PTS, _resample, _rotation_matrix
from lane_graph import LaneGraph, REACH_MARGIN_M

import json

PRED_SEC = PRED_LEN / 10.0        # 예측 구간 6초


class Av2LaneRuleDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 limit: Optional[int] = None, with_rules: bool = True,
                 centerline: str = "api", select: str = "centroid"):
        """centerline / select 기본값은 dataset_map.py 와 숫자까지 일치하도록 맞춰져 있다.
        (select="nearest" 는 차선까지의 최단거리로 고르는 개선안이지만 기존 실험과 달라진다)"""
        self.root = Path(data_root) / split
        self.with_rules = with_rules
        self.centerline, self.select = centerline, select
        self.dirs = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / f"scenario_{d.name}.parquet").exists():
                self.dirs.append(d)
        if limit is not None:
            self.dirs = self.dirs[:limit]

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        sdir = self.dirs[idx]
        sid = sdir.name

        # --- 궤적 (dataset_map.py 와 동일) ---
        s = scenario_serialization.load_argoverse_scenario_parquet(
            sdir / f"scenario_{sid}.parquet")
        focal = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        st = sorted(focal.object_states, key=lambda x: x.timestep)
        pos = np.array([x.position for x in st], dtype=np.float32)
        vel = np.array([x.velocity for x in st], dtype=np.float32)
        head = np.array([x.heading for x in st], dtype=np.float32)

        origin = pos[OBS_LEN - 1].copy()
        theta = head[OBS_LEN - 1].copy()
        R = _rotation_matrix(-theta)
        pos_n = (pos - origin) @ R.T
        x = np.concatenate([pos_n, vel @ R.T, (head - theta).reshape(-1, 1)], axis=1)[:OBS_LEN]
        y = pos_n[OBS_LEN:]

        # --- 지도: 원본 JSON에서 직접 그래프를 만든다 ---
        # av2 API 를 거치면 중심선을 10점으로 다시 만들어 주는데, 원본(중앙값 12점)을
        # 그대로 쓰는 편이 정보 손실도 적고 빠르다.
        raw = json.loads((sdir / f"log_map_archive_{sid}.json").read_text())
        g = LaneGraph.from_json_dict(raw, centerline=self.centerline)

        ids = list(g.lanes)
        norm = {i: _resample((g.lanes[i].centerline.astype(np.float32) - origin) @ R.T, N_PTS)
                for i in ids}
        if self.select == "centroid":       # dataset_map.py 와 동일한 기준
            d = np.array([np.linalg.norm(norm[i].mean(axis=0)) for i in ids])
        else:                               # 차선까지의 최단거리 (더 합리적이지만 기준선이 달라짐)
            d = np.array([np.min(np.linalg.norm(norm[i], axis=1)) for i in ids])
        sel = [ids[o] for o in np.argsort(d)[:N_LANES]]

        lane_arr = np.zeros((N_LANES, N_PTS, 2), dtype=np.float32)
        lane_mask = np.zeros(N_LANES, dtype=np.float32)
        for i, lid in enumerate(sel):
            lane_arr[i] = norm[lid]
            lane_mask[i] = 1.0

        out = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "lanes": torch.from_numpy(lane_arr),
            "lane_mask": torch.from_numpy(lane_mask),
            "origin": torch.from_numpy(origin),
            "theta": torch.tensor(theta, dtype=torch.float32),
            "scenario_id": sid,
        }
        if not self.with_rules:
            return out

        # --- 규칙 피처 ---
        h0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        # 예측 시점에는 미래 주행거리를 모르니 현재 속도로 추정한다 (6초 × 현재 속력).
        speed = float(np.linalg.norm(vel[OBS_LEN - 1]))
        budget = max(20.0, speed * PRED_SEC) + REACH_MARGIN_M
        starts = g.candidate_lanes(origin.astype(np.float64), h0,
                                   path=pos_n[:OBS_LEN].astype(np.float64) @ R + origin)
        reach = g.reachable(starts, budget) if starts else set()

        feat = np.zeros((N_LANES, LaneGraph.N_FEAT + 1), dtype=np.float32)
        for i, lid in enumerate(sel):
            feat[i] = g.lane_features(lid, h0, reach)
        out["lane_feat"] = torch.from_numpy(feat)
        out["n_reachable"] = torch.tensor(len(reach), dtype=torch.float32)
        return out


if __name__ == "__main__":
    import time
    from dataset_map import Av2MapDataset
    root = "/data/argoverse2/motion_forecasting"
    ds = Av2LaneRuleDataset(root, "val", limit=30)
    b = ds[0]
    print(f"x {tuple(b['x'].shape)}  y {tuple(b['y'].shape)}  "
          f"lanes {tuple(b['lanes'].shape)}  lane_feat {tuple(b['lane_feat'].shape)}")
    f = b["lane_feat"].numpy()
    m = b["lane_mask"].numpy().astype(bool)
    print(f"\n실제 차선 {int(m.sum())}개 중")
    print(f"  focal 과 같은 방향   {int(f[m,2].sum()):3d}개")
    print(f"  교차로 내부          {int(f[m,3].sum()):3d}개")
    print(f"  좌 차선변경 가능      {int(f[m,7].sum()):3d}개")
    print(f"  우 차선변경 가능      {int(f[m,8].sum()):3d}개")
    print(f"  규칙상 도달 가능      {int(f[m,9].sum()):3d}개  (지도 전체 도달 {int(b['n_reachable'])}개)")

    old = Av2MapDataset(root, "val", limit=30)
    for name, d in (("dataset_map (기존)", old), ("dataset_lane (규칙 포함)", ds)):
        d[0]
        t = time.perf_counter()
        for i in range(20):
            d[i]
        print(f"{name:26} {(time.perf_counter()-t)/20*1000:6.1f} ms/샘플")
