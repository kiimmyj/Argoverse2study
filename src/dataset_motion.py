"""
dataset_motion.py - 궤적 입력 표현만 바꿔 끼우는 Dataset 래퍼.

Av2MapDataset 을 그대로 감싸고 x 채널만 motion_repr 로 갈아끼운다.
지도(lanes/lane_mask), 정답 y, 시나리오 순서가 모두 동일하므로
표현끼리 사과 대 사과 비교가 된다.

  raw5   (50,5)  x, y, vx, vy, heading        ← 현재 v3 (기준선)
  vah3   (50,3)  v[km/h], a[km/h/s], h[rad]   ← ① 안
  ah0_2  (50,2)  a누적(첫값=v0), h[rad]        ← ② 안
  vah4   (50,4)  v, a, sin h, cos h           ← ① + wrap 제거
  vh2    (50,2)  v, h                         ← 참고
"""
from torch.utils.data import Dataset
import numpy as np
import torch

from dataset_map import Av2MapDataset
import motion_repr as mr

# (ReprConfig, traj_to_motion 인자)
REPRS = {
    "raw5":  None,
    "vah3":  (mr.ReprConfig(mode="vah",  heading="rad"),    dict(smooth=5, stop_ms=1.0)),
    "ah0_2": (mr.ReprConfig(mode="ah0",  heading="rad"),    dict(smooth=1, stop_ms=1.0)),
    "vah4":  (mr.ReprConfig(mode="vah",  heading="sincos"), dict(smooth=5, stop_ms=1.0)),
    "vh2":   (mr.ReprConfig(mode="vh",   heading="rad"),    dict(smooth=1, stop_ms=1.0)),
}


def in_dim(name: str) -> int:
    if name == "raw5":
        return 5
    return REPRS[name][0].channels()


class Av2MotionDataset(Dataset):
    def __init__(self, data_root, split="train", limit=None, repr_name="vah3"):
        if repr_name not in REPRS:
            raise ValueError(f"모르는 표현: {repr_name} (가능: {list(REPRS)})")
        self.base = Av2MapDataset(data_root, split, limit)
        self.repr_name = repr_name
        self.dirs = self.base.dirs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        spec = REPRS[self.repr_name]
        if spec is None:
            return b
        cfg, kw = spec
        pos = b["x"][:, :2].numpy().astype(np.float64)     # 정규화 좌표 과거 50 step
        m = mr.traj_to_motion(pos, **kw)
        b["x"] = torch.from_numpy(mr.encode(m, cfg))
        return b


if __name__ == "__main__":
    root = "/data/argoverse2/motion_forecasting"
    for name in REPRS:
        ds = Av2MotionDataset(root, "val", limit=3, repr_name=name)
        b = ds[0]
        x = b["x"]
        print(f"{name:6s} x={tuple(x.shape)} in_dim={in_dim(name):d}  "
              f"평균 {x.mean():+.3f} std {x.std():.3f} |max| {x.abs().max():.2f}  "
              f"lanes={tuple(b['lanes'].shape)} y={tuple(b['y'].shape)}")
