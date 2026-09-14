"""prepare_v4.py 가 구운 전처리 캐시를 읽는다.

Av2LaneRuleDataset 과 **같은 dict 를 돌려준다** — 그래서 train_v4.py 의 to_dev /
loss_fn / evaluate 를 그대로 쓸 수 있고, DataLoader 도 그대로 붙는다.

원본은 시나리오마다 parquet + log_map_archive JSON 을 다시 읽고 LaneGraph 를 다시
만든다(실측 ~96 ms/샘플). 여기서는 필드별 .npy memmap 을 슬라이스만 한다. 파싱도
압축해제도 없다. 결과가 결정적(deterministic)이므로 두 경로는 숫자까지 같아야 한다 —
`verify_against_source()` 가 그걸 표본으로 확인한다.

캐시 디렉터리 구조:
    <field>.npy        필드별 (N, *shape) memmap
    scenario_id.json   길이 N 문자열 리스트 (memmap 에 못 넣는 것)
    meta.json          설정 · 소스 해시 · git 상태 · 필드 스펙
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def source_files(src_dir, root="dataset_lane"):
    """root 모듈이 (전이적으로) import 하는 src/ 안의 .py 목록. 캐시 키에 넣는다.

    손으로 목록을 적으면 import 가 늘 때마다 빠진다 — 실제로 heading_decomp.py(build_heading)와
    dataset_map.py(_resample·_rotation_matrix)가 빠져 있어서, 그 파일을 고쳐도 낡은 캐시가
    재사용될 수 있었다. import 문을 ast 로 따라가 저장소 안 파일만 모은다
    (함수 안 import 까지 넣으므로 필요 이상으로 넓을 수는 있어도 좁지는 않다).
    """
    import ast
    src_dir = Path(src_dir)
    seen, todo = set(), [root]
    while todo:
        name = todo.pop()
        p = src_dir / f"{name}.py"
        if name in seen or not p.exists():
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                todo += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                todo.append(node.module.split(".")[0])
    return tuple(sorted(f"{n}.py" for n in seen))


class CachedV4Dataset(Dataset):
    def __init__(self, cache_dir, limit=None):
        self.dir = Path(cache_dir)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"캐시가 아니다 (meta.json 없음): {self.dir}")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.sids = json.loads((self.dir / "scenario_id.json").read_text())

        self._mm = {}
        for name in self.meta["fields"]:
            self._mm[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")

        n = self.meta["n"]
        self.n = n if limit is None else min(limit, n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # memmap 슬라이스는 뷰다. torch 로 넘기기 전에 복사해야 워커에서 안전하다.
        # np.ascontiguousarray 를 쓰면 안 된다 — ndim>=1 을 강제해서 0차원 스칼라
        # (theta, v0, n_distinct, n_reachable, route_fallback) 가 () 대신 (1,) 이
        # 되고, collate 후 v0 가 (B,1) 이 되어 l0 의 rollout 이 깨진다.
        out = {k: torch.from_numpy(np.array(v[i])) for k, v in self._mm.items()}
        out["scenario_id"] = self.sids[i]
        return out

    def raw(self, field):
        """검증용 대량 접근 — memmap 을 그대로 준다 (복사 없음)."""
        return self._mm[field]

    # ── 무결성 ──────────────────────────────────────────────────────────

    def is_stale(self, repo_src):
        """캐시를 구운 뒤 전처리 소스가 바뀌었는지 본다.

        prepare_v4.py 가 캐시 키에 소스 해시를 넣으므로 소스가 바뀌면 보통 새 키의
        새 캐시가 생긴다. 이 함수는 캐시 디렉터리를 손으로 지정해 쓰는 경우를 위한
        방어막이다. (해시가 다르면 True)
        """
        import hashlib
        repo_src = Path(repo_src)
        if set(source_files(repo_src)) != set(self.meta["source_hashes"]):
            return True                       # import 구성 자체가 바뀌었다
        for fn, want in self.meta["source_hashes"].items():
            p = repo_src / fn
            if not p.exists():
                return True
            got = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            if got != want:
                return True
        return False

    def verify_against_source(self, data_root, split=None, n=8, atol=0.0):
        """표본 몇 개를 원본 전처리와 대조한다. 캐시가 진짜 같은 값인지 확인용.

        원본 __getitem__ 은 비싸므로(~96 ms) 기본 8개만 본다. 불일치하면 어느 필드가
        얼마나 어긋났는지 리스트로 돌려준다. 빈 리스트면 통과다.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataset_lane import Av2LaneRuleDataset

        split = split or self.meta["split"]
        src = Av2LaneRuleDataset(data_root, split, self.meta["limit"],
                                 **self.meta["dataset_kwargs"])
        bad = []
        idx = np.linspace(0, len(self) - 1, min(n, len(self))).astype(int)
        for i in idx:
            a, b = self[int(i)], src[int(i)]
            if a["scenario_id"] != b["scenario_id"]:
                bad.append(f"[{i}] scenario_id {a['scenario_id']} != {b['scenario_id']}")
                continue
            for k in self.meta["fields"]:
                x = a[k].numpy()
                y = np.asarray(b[k])
                if x.shape != y.shape:
                    bad.append(f"[{i}] {k} shape {x.shape} != {y.shape}")
                elif not np.allclose(x, y, atol=atol, rtol=0, equal_nan=True):
                    d = np.abs(x - y)
                    bad.append(f"[{i}] {k} 최대차 {d.max():.3e}")
        return bad


def open_cache(cache_root, split, dataset_kwargs, repo_src, limit):
    """prepare_v4.py 와 같은 규칙으로 캐시 디렉터리를 찾는다. 없으면 None."""
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps({"split": split, **dataset_kwargs}, sort_keys=True).encode())
    for fn in source_files(repo_src):
        h.update(hashlib.sha256((Path(repo_src) / fn).read_bytes()).hexdigest().encode())
    d = Path(cache_root) / f"{split}_{h.hexdigest()[:16]}_n{limit}"
    return d if (d / "meta.json").exists() else None
