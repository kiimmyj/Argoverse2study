"""
dataset_agents.py - 주변 차량 과거 궤적을 v4 입력에 붙인다 (2026-09-30).

사용자 지시: "neighborhood vehicle = 어떻게 적용하지? **일단은 단순하게 가자**
(예: 반경 30 m 또는 scene 안의 모든 객체 전부 다)".

그래서 새 전처리를 만들지 않고 이미 구워 둔 캐시를 그대로 붙인다
(`src/prep_agents_heading.py`, t=0 에 관측되는 가까운 32대 · 과거 50스텝 · focal 정규화 프레임).

무엇을 주나 (스텝마다 5채널, 기본 2 Hz 로 10스텝)
    dx, dy   focal 원점 기준 상대 위치 [m] / 50
    sin h, cos h   그 차량의 진행방향 (focal 프레임)
    obs      그 스텝이 관측됐나 (0/1)

왜 이 형태인가
- 위치만 주면 정지·저속 차량의 방향을 모델이 다시 추정해야 한다. h 는 이미 우리가
  전처리로 만든 값이라(위치차분 + 저속 보조) 그대로 얹는 편이 싸다.
- 관측 마스크를 채널로 주는 이유: 주변 차량은 중간이 끊긴다(스텝의 21.9% 가 미관측).
  0 으로 채운 값과 "실제로 0 인 값"을 모델이 구별할 수 있어야 한다.

정렬
- 주변 차량 캐시와 v4 메모리맵 캐시는 **같은 정렬된 디렉터리 순서**로 굽는다. 그래서 i 번째가
  같은 시나리오다. 붙일 때 `scenario_id.json` 을 대조해 확인한다(어긋나면 예외).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_ROOT = Path("/data/argoverse2/cache/v4")
POS_SCALE = 50.0          # 상대 위치 정규화 [m] — 반경 30 m 안이면 |값| <= 0.6
DEFAULT_RADIUS_M = 30.0   # 사용자가 고른 "단순한" 기준
DEFAULT_STEP = 5          # 과거 50스텝 -> 10스텝 (2 Hz). 입력 길이를 줄이고 잡음을 줄인다
MAX_AGENTS = 32           # 캐시 K


def find_agents_cache(split, n, root=CACHE_ROOT, with_pos=True):
    """`agents_hist_<split>_<key>_n<N>[_pos]` 중 n 이 맞는 것을 찾는다. 여러 개면 최신."""
    pat = f"agents_hist_{split}_*_n{n}" + ("_pos" if with_pos else "")
    cands = [p for p in sorted(root.glob(pat)) if p.is_dir()]
    if with_pos:
        cands = [p for p in cands if p.name.endswith("_pos")]
    else:
        cands = [p for p in cands if not p.name.endswith("_pos")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


class AgentsStore:
    """주변 차량 캐시를 memmap 으로 열어 두고 인덱스로 꺼낸다."""

    def __init__(self, cache_dir, radius_m=DEFAULT_RADIUS_M, step=DEFAULT_STEP):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if not self.meta.get("with_pos"):
            raise ValueError(f"위치가 없는 캐시다 — --with-pos 로 다시 구워라: {self.dir}")
        self.sids = json.loads((self.dir / "scenario_id.json").read_text())
        self.radius_m, self.step = float(radius_m), int(step)
        self._mm = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r")
                    for k in ("pos", "h", "h_src", "dist0", "atype", "is_av")}
        self.T = self.meta["T"]
        self.idx = np.arange(self.T - 1, -1, -self.step)[::-1].copy()   # 마지막 관측 포함

    def __len__(self):
        return len(self.sids)

    def get(self, i):
        """(agents (K,T',5) float32, mask (K,) float32) — mask 0 은 '그 슬롯은 비었다'."""
        pos = np.asarray(self._mm["pos"][i])[:, self.idx]          # (K,T',2) focal 프레임
        h = np.asarray(self._mm["h"][i])[:, self.idx]              # (K,T')
        src = np.asarray(self._mm["h_src"][i])[:, self.idx]        # 0 = 미관측
        dist0 = np.asarray(self._mm["dist0"][i])                   # (K,)
        obs = (src > 0).astype(np.float32)
        feat = np.stack([pos[..., 0] / POS_SCALE, pos[..., 1] / POS_SCALE,
                         np.sin(h) * obs, np.cos(h) * obs, obs], axis=-1).astype(np.float32)
        # 반경 밖·빈 슬롯은 마스크로 끈다. dist0 는 t=0 거리다(빈 슬롯은 0 으로 채워져 있으니
        # 관측 스텝이 하나도 없는 슬롯을 빈 것으로 본다).
        alive = obs.sum(axis=1) > 0
        keep = alive & (dist0 <= self.radius_m)
        return feat * keep[:, None, None], keep.astype(np.float32)


class WithAgents(Dataset):
    """기존 Dataset 에 주변 차량 두 필드(agents, agents_mask)를 얹는 얇은 래퍼."""

    def __init__(self, base, store, check_sids=True):
        self.base, self.store = base, store
        if check_sids:
            bs = getattr(base, "sids", None)
            if bs is None:
                bs = [p.name for p in getattr(base, "dirs", [])]
            n = min(len(bs), len(store.sids), 64)
            if n and list(bs[:n]) != list(store.sids[:n]):
                raise ValueError("주변 차량 캐시와 시나리오 순서가 다르다 — 같은 limit 으로 다시 구워라")
        if len(base) > len(store):
            raise ValueError(f"주변 차량 캐시가 짧다: {len(store)} < {len(base)}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        out = self.base[i]
        a, m = self.store.get(i)
        out["agents"] = torch.from_numpy(a)
        out["agents_mask"] = torch.from_numpy(m)
        return out


def attach_agents(base, split, n, radius_m=DEFAULT_RADIUS_M, step=DEFAULT_STEP, root=CACHE_ROOT):
    """split·n 에 맞는 주변 차량 캐시를 찾아 래핑한다. 없으면 예외."""
    d = find_agents_cache(split, n, root=root, with_pos=True)
    if d is None:
        raise SystemExit(f"주변 차량 캐시 없음 (split={split}, n={n}) — "
                         f"python src/prep_agents_heading.py --split {split} --with-pos 를 먼저 돌려라")
    return WithAgents(base, AgentsStore(d, radius_m=radius_m, step=step)), d


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_cached import CachedV4Dataset
    d = find_agents_cache("val", 24988)
    print("캐시:", d)
    st = AgentsStore(d)
    a, m = st.get(0)
    print("agents", a.shape, "mask", m.shape, "살아있는 슬롯", int(m.sum()),
          "|값| 최대 %.3f" % np.abs(a).max())
