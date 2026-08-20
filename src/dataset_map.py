"""
Av2MapDataset - 궤적 + HD map(차선) 을 함께 반환하는 Dataset.

기존 Av2FocalDataset 을 확장:
  - 궤적: 입력 (50,5) / 정답 (60,2)  (기존과 동일, t=49 정규화)
  - 지도: focal 근처 차선 20개, 각 10점 → (20,10,2)  (궤적과 동일 정규화)
  - lane_mask: 실제 차선이 20개 안 될 때 padding 표시 (20,)
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

OBS_LEN, PRED_LEN = 50, 60
N_LANES, N_PTS = 20, 10          # 차선 20개, 각 10점


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _resample(polyline: np.ndarray, n: int) -> np.ndarray:
    """(M,2) 폴리라인을 길이 따라 균등한 n점으로 리샘플링 → (n,2)."""
    if len(polyline) == 1:
        return np.repeat(polyline, n, axis=0)
    # 누적 거리
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(seg)])
    if dist[-1] == 0:
        return np.repeat(polyline[:1], n, axis=0)
    target = np.linspace(0, dist[-1], n)
    x = np.interp(target, dist, polyline[:, 0])
    y = np.interp(target, dist, polyline[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


class Av2MapDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 limit: Optional[int] = None):
        self.root = Path(data_root) / split
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
        s = scenario_serialization.load_argoverse_scenario_parquet(
            sdir / f"scenario_{sid}.parquet")

        # --- 궤적 (기존과 동일) ---
        focal = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        states = sorted(focal.object_states, key=lambda st: st.timestep)
        pos = np.array([st.position for st in states], dtype=np.float32)
        vel = np.array([st.velocity for st in states], dtype=np.float32)
        head = np.array([st.heading for st in states], dtype=np.float32)

        origin = pos[OBS_LEN - 1].copy()
        theta = head[OBS_LEN - 1].copy()
        R = _rotation_matrix(-theta)

        pos_n = (pos - origin) @ R.T
        vel_n = vel @ R.T
        head_n = (head - theta).reshape(-1, 1)
        feat = np.concatenate([pos_n, vel_n, head_n], axis=1)
        x = feat[:OBS_LEN]
        y = pos_n[OBS_LEN:]

        # --- 지도 (차선) ---
        amap = ArgoverseStaticMap.from_json(sdir / f"log_map_archive_{sid}.json")
        lane_ids = list(amap.vector_lane_segments.keys())

        lanes, dists = [], []
        for lid in lane_ids:
            c = amap.get_lane_segment_centerline(lid)[:, :2]  # (M,2) x,y만
            c_n = (c - origin) @ R.T                          # 궤적과 동일 정규화
            c_r = _resample(c_n, N_PTS)                       # (10,2)
            lanes.append(c_r)
            dists.append(np.linalg.norm(c_r.mean(axis=0)))    # 원점(=focal)까지 거리

        # focal 근처 20개 선택
        order = np.argsort(dists)[:N_LANES]
        lane_arr = np.zeros((N_LANES, N_PTS, 2), dtype=np.float32)
        lane_mask = np.zeros(N_LANES, dtype=np.float32)       # 1=실제, 0=padding
        for i, o in enumerate(order):
            lane_arr[i] = lanes[o]
            lane_mask[i] = 1.0

        return {
            "x": torch.from_numpy(x),                    # (50,5)
            "y": torch.from_numpy(y),                    # (60,2)
            "lanes": torch.from_numpy(lane_arr),         # (20,10,2)
            "lane_mask": torch.from_numpy(lane_mask),    # (20,)
            "origin": torch.from_numpy(origin),
            "theta": torch.tensor(theta, dtype=torch.float32),
            "scenario_id": sid,
        }


if __name__ == "__main__":
    ds = Av2MapDataset("/data/argoverse2/motion_forecasting", "train", limit=5)
    print(f"dataset size : {len(ds)}")
    b = ds[0]
    print(f"x     : {tuple(b['x'].shape)}         (50, 5) 기대")
    print(f"y     : {tuple(b['y'].shape)}         (60, 2) 기대")
    print(f"lanes : {tuple(b['lanes'].shape)}   (20, 10, 2) 기대")
    print(f"lane_mask : {tuple(b['lane_mask'].shape)}  합={int(b['lane_mask'].sum())} (실제 차선 수)")
    print(f"x[49] pos : {b['x'][OBS_LEN-1, :2].numpy()}  (거의 0,0)")
    print(f"lanes[0] 첫 점 : {b['lanes'][0,0].numpy()}  (정규화 좌표)")
