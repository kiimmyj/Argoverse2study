"""
prep_agents_heading.py - 주변 차량의 과거 heading 을 전체 데이터셋에서 전처리해 캐시로 굽는다 (모델 입력용).

v4 의 Dataset(`dataset_lane.py`)은 focal 트랙 하나만 읽는다. 주변 차량을 입력으로 쓰는 모델(상호작용)을
만들 때를 위해, 같은 규칙으로 주변 차량의 진행방향 h 를 미리 만들어 둔다.

무엇을 만드나 (시나리오마다)
----------------------------
- 대상: t = 0 (스텝 49) 에 관측되는 focal 이외 트랙 중 종류가 AGENT_TYPES 인 것.
  focal 위치에서 가까운 순으로 K = 32 대. 자율주행 차량(track_id "AV")도 주변 차량으로 넣는다.
- 과거 5초 50스텝의 h — `heading_decomp.build_heading` (focal 과 같은 규칙):
  - 위치 차분으로 만들고, 1 m/s 미만만 AV2 heading 으로 채운다(180° 뒤집힘 교정·튐 가드 포함).
  - 중간에 끊긴 트랙은 관측된 이웃 스텝끼리 차분한다.
- focal 정규화 프레임이다: h_n = wrap(h_city − θ_focal). θ_focal 은 t = 0 의 focal AV2 heading 이다(dataset_lane 과 같다).
- **미래(스텝 50~109)는 한 줄도 읽지 않는다.** 모델 입력으로 써도 누수가 없다.
- 순서는 정렬된 시나리오 디렉터리 순서다 — prepare_v4 캐시와 같은 인덱스로 붙여 쓴다(scenario_id.json 으로 확인).

필드 (memmap .npy, N = 시나리오 수, K = 32, T = 50)
  h        (N,K,T) float32   진행방향 [rad], focal 프레임. 없는 칸은 0
  h_src    (N,K,T) uint8     0 = 미관측/없음, 1 = 위치 차분, 2 = AV2 보조, 3 = 이웃 값 유지 (build_heading 의 src)
  pos0     (N,K,2) float32   t = 0 위치, focal 프레임 [m]
  dist0    (N,K)   float32   t = 0 에 focal 과의 거리 [m]
  atype    (N,K)   uint8     0 = 빈 칸, 1 = vehicle, 2 = bus, 3 = motorcyclist, 4 = cyclist
  is_av    (N,K)   uint8     자율주행 차량 트랙이면 1
  n_cand   (N,)    int16     자르기 전 후보 수 (K 를 넘으면 가까운 K 대만 남는다)
  (--with-pos 를 주면 pos (N,K,T,2) float32 — 과거 50스텝 위치, 없는 칸 0)

  python src/prep_agents_heading.py --split val --workers 40
  python src/prep_agents_heading.py --split train --workers 40
  python src/prep_agents_heading.py --split val --limit 64 --check      # 인과성·focal 재현 확인만
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2.datasets.motion_forecasting import scenario_serialization
from dataset_map import _rotation_matrix
from heading_decomp import build_heading, wrap

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/argoverse2/motion_forecasting")
CACHE_ROOT = Path("/data/argoverse2/cache/v4")
OBS_LEN = 50
K = 32
AGENT_TYPES = {"vehicle": 1, "bus": 2, "motorcyclist": 3, "cyclist": 4}


def agents_one(sdir, with_pos=False, loader=None):
    """시나리오 디렉터리 하나 → 주변 차량 과거 heading. 미래 스텝은 쓰지 않는다."""
    load = loader or scenario_serialization.load_argoverse_scenario_parquet
    s = load(sdir / f"scenario_{sdir.name}.parquet")
    focal = next(t for t in s.tracks if t.track_id == s.focal_track_id)
    fst = {o.timestep: o for o in focal.object_states}
    origin = np.asarray(fst[OBS_LEN - 1].position, dtype=np.float32)
    theta = np.float32(fst[OBS_LEN - 1].heading)
    R = _rotation_matrix(-theta)

    cands = []
    for t in s.tracks:
        if t.track_id == s.focal_track_id:
            continue
        code = AGENT_TYPES.get(t.object_type.value)
        if code is None:
            continue
        past = {o.timestep: o for o in t.object_states if o.timestep < OBS_LEN}
        if (OBS_LEN - 1) not in past:                   # t = 0 에 안 보이면 입력 후보가 아니다
            continue
        p0 = np.asarray(past[OBS_LEN - 1].position, dtype=np.float32)
        cands.append((float(np.linalg.norm(p0 - origin)), code, t.track_id == "AV", past))
    cands.sort(key=lambda c: c[0])
    n_cand = len(cands)

    h = np.zeros((K, OBS_LEN), np.float32)
    src = np.zeros((K, OBS_LEN), np.uint8)
    pos0 = np.zeros((K, 2), np.float32)
    dist0 = np.zeros(K, np.float32)
    atype = np.zeros(K, np.uint8)
    is_av = np.zeros(K, np.uint8)
    pos = np.zeros((K, OBS_LEN, 2), np.float32) if with_pos else None
    for i, (d, code, av, past) in enumerate(cands[:K]):
        P = np.full((OBS_LEN, 2), np.nan)
        H = np.full(OBS_LEN, np.nan)
        obs = np.zeros(OBS_LEN, bool)
        for ts, o in past.items():
            P[ts] = o.position
            H[ts] = o.heading
            obs[ts] = True
        hc, sc = build_heading(P, obs=obs, h_ref=H)
        ok = obs & np.isfinite(hc)
        h[i, ok] = wrap(hc[ok] - float(theta)).astype(np.float32)
        src[i, ok] = sc[ok]
        pn = (P.astype(np.float32) - origin) @ R.T
        pos0[i] = pn[OBS_LEN - 1]
        dist0[i] = d
        atype[i] = code
        is_av[i] = av
        if with_pos:
            pos[i, obs] = pn[obs]
    out = dict(h=h, h_src=src, pos0=pos0, dist0=dist0, atype=atype, is_av=is_av,
               n_cand=np.int16(min(n_cand, 32767)))
    if with_pos:
        out["pos"] = pos
    return sdir.name, out


def _work(args):
    sdir, with_pos = args
    return agents_one(sdir, with_pos)


def source_hash():
    h = hashlib.sha256()
    hashes = {}
    for f in ("prep_agents_heading.py", "heading_decomp.py", "dataset_map.py", "motion_repr.py",
              "lane_frame.py", "lane_graph.py"):
        b = (REPO / "src" / f).read_bytes()
        hashes[f] = hashlib.sha256(b).hexdigest()[:12]
        h.update(b)
    return h.hexdigest()[:16], hashes


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def check(split, n):
    """① 인과성: 미래 스텝을 흔들어도 결과가 비트 단위로 같다. ② focal 에 같은 규칙을 걸면 ah2 캐시 h 와 같다."""
    import dataclasses
    from heading_decomp import ah_features
    rng = np.random.default_rng(0)
    dirs = sorted(d for d in (DATA_ROOT / split).iterdir() if d.is_dir())[:n]
    orig = scenario_serialization.load_argoverse_scenario_parquet

    def shaken(path):
        s = orig(path)
        for t in s.tracks:
            new = []
            for o in t.object_states:
                if o.timestep >= OBS_LEN:
                    o = dataclasses.replace(o, position=(o.position[0] + rng.normal(0, 5), o.position[1] + rng.normal(0, 5)),
                                            heading=o.heading + rng.normal(0, 1.0))
                new.append(o)
            t.object_states[:] = new
        return s

    leak = 0
    for d in dirs:
        _, a = agents_one(d, with_pos=True)
        _, b = agents_one(d, with_pos=True, loader=shaken)
        leak += sum(not np.array_equal(a[k], b[k]) for k in a)
    print(f"[check] 인과성 {len(dirs)}개: 미래를 흔든 뒤 바뀐 필드 {leak}개 ({'통과' if leak == 0 else '실패'})")

    # focal 재현 — build_heading 경로가 dataset 의 ah2 h 와 같은가 (focal 은 전 스텝 관측)
    worst = 0.0
    for d in dirs:
        s = orig(d / f"scenario_{d.name}.parquet")
        f = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        st = sorted(f.object_states, key=lambda o: o.timestep)[:OBS_LEN]
        pos = np.array([o.position for o in st], np.float32)
        head = np.array([o.heading for o in st], np.float32)
        theta = float(head[-1])
        ref = ah_features(pos, head, theta)[0][:, 1]
        hc, _ = build_heading(pos.astype(np.float64), obs=np.ones(OBS_LEN, bool), h_ref=head.astype(np.float64))
        mine = wrap(hc - theta).astype(np.float32)
        worst = max(worst, float(np.abs(wrap(mine.astype(np.float64) - ref)).max()))
    print(f"[check] focal 에 같은 규칙: ah2 입력 h 와 최대 차 {worst:.2e} rad ({'통과' if worst < 1e-5 else '확인 필요'})")
    return leak == 0 and worst < 1e-5


def build(split, limit, workers, with_pos, root, rebuild):
    dirs = sorted(d for d in (DATA_ROOT / split).iterdir()
                  if d.is_dir() and (d / f"scenario_{d.name}.parquet").exists())
    if limit:
        dirs = dirs[:limit]
    n = len(dirs)
    key, hashes = source_hash()
    name = f"agents_hist_{split}_{key}_n{n}" + ("_pos" if with_pos else "")
    out = root / name
    if out.exists() and not rebuild:
        print(f"[{split}] 이미 있다: {out}")
        return out
    tmp = out.with_suffix(".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    specs = {"h": ((n, K, OBS_LEN), np.float32), "h_src": ((n, K, OBS_LEN), np.uint8),
             "pos0": ((n, K, 2), np.float32), "dist0": ((n, K), np.float32),
             "atype": ((n, K), np.uint8), "is_av": ((n, K), np.uint8), "n_cand": ((n,), np.int16)}
    if with_pos:
        specs["pos"] = ((n, K, OBS_LEN, 2), np.float32)
    mm = {k: np.lib.format.open_memmap(tmp / f"{k}.npy", mode="w+", dtype=dt, shape=sh)
          for k, (sh, dt) in specs.items()}
    tot = sum(a.nbytes for a in mm.values())
    print(f"[{split}] {n:,} 시나리오 → {tmp} ({tot / 2**30:.2f} GiB, workers {workers})", flush=True)

    sids = [None] * n
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        for i, (sid, r) in enumerate(pool.imap(_work, [(d, with_pos) for d in dirs], chunksize=64)):
            for k, v in r.items():
                mm[k][i] = v
            sids[i] = sid
            if (i + 1) % 20000 == 0 or i + 1 == n:
                el = time.perf_counter() - t0
                print(f"[{split}] {i + 1:,}/{n:,}  {el:.0f}s  ETA {el / (i + 1) * (n - i - 1):.0f}s", flush=True)
    for a in mm.values():
        a.flush()
    assert sids == [d.name for d in dirs]
    (tmp / "scenario_id.json").write_text(json.dumps(sids))
    src = np.asarray(mm["h_src"])
    kept = np.asarray(mm["atype"]) > 0
    stats = {
        "agents_per_scenario_mean": float(kept.sum(1).mean()),
        "n_cand_mean": float(np.asarray(mm["n_cand"]).mean()),
        "truncated_scenarios_pct": float((np.asarray(mm["n_cand"]) > K).mean() * 100),
        "step_src_pct_of_kept_steps": {c: float((src[kept] == v).mean() * 100)
                                       for c, v in (("none", 0), ("posdiff", 1), ("av2", 2), ("hold", 3))},
        "type_pct": {t: float((np.asarray(mm["atype"])[kept] == c).mean() * 100) for t, c in AGENT_TYPES.items()},
    }
    (tmp / "meta.json").write_text(json.dumps({
        "split": split, "n": n, "limit": limit, "K": K, "T": OBS_LEN, "with_pos": with_pos,
        "agent_types": AGENT_TYPES, "select": "t=0(스텝 49)에 관측, focal 위치에서 가까운 순 K 대, AV 포함",
        "heading": "heading_decomp.build_heading(pos[:50], obs, h_ref=AV2 heading[:50]) → wrap(h − θ_focal)",
        "h_src": {"0": "미관측/없음", "1": "위치 차분", "2": "AV2 보조(저속)", "3": "이웃 값 유지"},
        "future_steps_read": False, "order": "정렬된 시나리오 디렉터리 (prepare_v4 캐시와 같은 인덱스)",
        "source_key": key, "source_hashes": hashes, "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--", "src"), "sec": time.perf_counter() - t0,
        "stats": stats,
    }, indent=2, ensure_ascii=False))
    tmp.rename(out)
    print(f"[{split}] 완료 {time.perf_counter() - t0:.0f}s → {out}")
    print(json.dumps(stats, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전체")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--with-pos", action="store_true", help="과거 50스텝 위치도 함께 굽는다")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--check", action="store_true", help="인과성·focal 재현만 확인하고 끝낸다")
    args = ap.parse_args()
    if args.check:
        ok = check(args.split, args.limit or 64)
        raise SystemExit(0 if ok else 1)
    build(args.split, args.limit, args.workers, args.with_pos, Path(args.cache_root), args.rebuild)


if __name__ == "__main__":
    main()
