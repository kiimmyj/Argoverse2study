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
  aw0_2  (50,2)  페달(v0 포함), 핸들(h0 포함)      ← 핸들링 2채널
  vad3   (50,3)  v, a, 조향각                   ← 핸들링 3채널
  vaw3   (50,3)  v, a, 요레이트
  vahd5  (50,5)  v, a, sin h, cos h, 조향각      ← 상태+조작 혼합
  vahw5  (50,5)  v, a, sin h, cos h, 요레이트     ← 핸들링 추가형(ω)
  vahdf5 (50,5)  v, a, sin h, cos h, 조향각       ← 핸들링 추가형(δ)
  vawf3  (50,3)  v, a, 요레이트                   ← 핸들링 대체형(ω)
  vadf3  (50,3)  v, a, 조향각                     ← 핸들링 대체형(δ)
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
    # --- 핸들링(조작) 관점: h(방향=상태) 대신 조향/요레이트(조작) ---
    "aw0_2": (mr.ReprConfig(mode="aw0"),                    dict(smooth=1, stop_ms=1.0)),
    "vad3":  (mr.ReprConfig(mode="vad"),                    dict(smooth=5, stop_ms=1.0)),
    "vaw3":  (mr.ReprConfig(mode="vaw"),                    dict(smooth=5, stop_ms=1.0)),
    "vahd5": (mr.ReprConfig(mode="vahd"),                   dict(smooth=5, stop_ms=1.0)),
    # 실측 권장안: 상태(v, a, h) + AV2 heading 필드 기반 깨끗한 요레이트
    "vahw5": (mr.ReprConfig(mode="vahw"),                   dict(smooth=5, stop_ms=1.0)),
    "vahdf5":(mr.ReprConfig(mode="vahdf"),                  dict(smooth=5, stop_ms=1.0)),
    "vawf3": (mr.ReprConfig(mode="vawf"),                   dict(smooth=5, stop_ms=1.0)),
    "vadf3": (mr.ReprConfig(mode="vadf"),                   dict(smooth=5, stop_ms=1.0)),
}
# heading 필드가 필요한 모드 (조작 신호를 거기서 만든다)
NEEDS_FIELD = {"vahw", "vahdf", "vawf", "vadf"}


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
        if cfg.mode in NEEDS_FIELD:
            # heading 필드(x 의 5번째 채널)로 깨끗한 요레이트를 만든다.
            # 위치와 정합하지 않지만 입력 채널로는 정합성이 필요 없다.
            kw = dict(kw, heading_field=b["x"][:, 4].numpy().astype(np.float64))
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
