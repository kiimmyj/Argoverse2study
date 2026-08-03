"""
Av2FocalDataset - AV2 Motion Forecasting focal-agent 궤적 Dataset.

각 시나리오 → (입력 50xD, 정답 60x2) 텐서.
  - 입력 피처 D=5 : (x, y, vx, vy, heading)
  - t=49(마지막 관측) 기준 정규화: 원점 이동 + heading 회전
  - 역변환용 정보(origin, rot)도 함께 반환 (채점 시 city 좌표 복원용)
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting import scenario_serialization

OBS_LEN, PRED_LEN = 50, 60          # 관측 5초 / 예측 6초 @10Hz


def _rotation_matrix(theta: float) -> np.ndarray:
    """+theta 만큼 회전시키는 2x2 행렬 (좌표계를 heading이 +x가 되게 정렬)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


class Av2FocalDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 limit: Optional[int] = None):
        """
        data_root : /data/argoverse2/motion_forecasting
        split     : train / val / test
        limit     : 앞에서 N개만 사용 (빠른 실험용). None이면 전체.
        """
        self.split_dir = Path(data_root) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"경로 없음: {self.split_dir}")

        # 시나리오 parquet 경로 목록 미리 수집
        self.paths = []
        for sid_dir in sorted(self.split_dir.iterdir()):
            if not sid_dir.is_dir():
                continue
            p = sid_dir / f"scenario_{sid_dir.name}.parquet"
            if p.exists():
                self.paths.append(p)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise RuntimeError(f"parquet 없음: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        s = scenario_serialization.load_argoverse_scenario_parquet(self.paths[idx])

        # focal track 찾기
        focal = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        states = sorted(focal.object_states, key=lambda st: st.timestep)

        # 시점별 원시값 추출 (focal은 110 step 모두 존재)
        pos = np.array([st.position for st in states], dtype=np.float32)   # (110, 2)
        vel = np.array([st.velocity for st in states], dtype=np.float32)   # (110, 2)
        head = np.array([st.heading for st in states], dtype=np.float32)   # (110,)

        # 기준점: t=49 (마지막 관측)
        origin = pos[OBS_LEN - 1].copy()          # (2,)  마지막 관측 위치
        theta = head[OBS_LEN - 1].copy()          # 스칼라  마지막 관측 방향
        # 좌표계를 -theta 회전 → focal heading이 +x 를 향하게
        R = _rotation_matrix(-theta)              # (2, 2)

        # 정규화
        pos_n = (pos - origin) @ R.T              # 이동 + 회전   (110, 2)
        vel_n = vel @ R.T                         # 회전만        (110, 2)
        head_n = (head - theta).reshape(-1, 1)    # 상대 각도     (110, 1)

        feat = np.concatenate([pos_n, vel_n, head_n], axis=1)   # (110, 5)

        # 입력(과거 50) / 정답(미래 60의 위치만)
        x = feat[:OBS_LEN]                        # (50, 5)
        y = pos_n[OBS_LEN:]                       # (60, 2)

        return {
            "x": torch.from_numpy(x),                       # (50, 5) 입력: 과거5초
            "y": torch.from_numpy(y),                       # (60, 2) 정답: 미래6초
            # origin·theta: 정규화에 쓴 값 → 예측을 city 좌표로 역변환할 때 사용
            "origin": torch.from_numpy(origin),             # (2,)  t=49 city 좌표(원점)
            "theta": torch.tensor(theta, dtype=torch.float32),  # t=49 heading(회전각)
            "scenario_id": s.scenario_id,
        }


if __name__ == "__main__":
    # 간단 동작 확인
    ds = Av2FocalDataset("/data/argoverse2/motion_forecasting", "train", limit=5)
    print(f"dataset size : {len(ds)}")
    sample = ds[0]
    print(f"x (input)  : {tuple(sample['x'].shape)}   (50, 5) 기대")
    print(f"y (target) : {tuple(sample['y'].shape)}   (60, 2) 기대")
    print(f"origin     : {sample['origin'].numpy()}   (t=49 city 좌표)")
    print(f"theta      : {float(sample['theta']):.3f} rad")
    print(f"scenario   : {sample['scenario_id']}")
    # 정규화 확인: 입력 마지막 시점(t=49)의 위치는 원점(0,0) 근처여야 함
    print(f"x[49] pos  : {sample['x'][OBS_LEN-1, :2].numpy()}   (거의 0,0 이어야 정상)")
