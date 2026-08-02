"""
AV2 Motion Forecasting - open one scenario, print its structure, and plot trajectories.

Usage:
  python explore_scenario.py                     # first scenario in train
  python explore_scenario.py --split val
  python explore_scenario.py --sid <scenario_id>
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from av2.datasets.motion_forecasting import scenario_serialization

DATA_ROOT = Path("/data/argoverse2/motion_forecasting")
OBS_LEN = 50  # first 50 steps = 5s observed

FOCAL_COLOR, AV_COLOR, OTHER_COLOR = "#ECA25B", "#007672", "#9DC3E6"


def pick_scenario(split: str, sid: str | None) -> Path:
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        raise SystemExit(f"path not found: {split_dir}")
    if sid:
        return split_dir / sid / f"scenario_{sid}.parquet"
    sid_dir = next(p for p in sorted(split_dir.iterdir()) if p.is_dir())
    return sid_dir / f"scenario_{sid_dir.name}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--sid", default=None)
    args = ap.parse_args()

    path = pick_scenario(args.split, args.sid)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    s = scenario_serialization.load_argoverse_scenario_parquet(path)

    # 1) scenario level
    print("=" * 58)
    print("[1] SCENARIO")
    print(f"   scenario_id : {s.scenario_id}")
    print(f"   city_name   : {s.city_name}")
    print(f"   timesteps   : {len(s.timestamps_ns)}   (11s @10Hz = 110)")
    print(f"   #tracks     : {len(s.tracks)}")
    print(f"   focal_id    : {s.focal_track_id}")

    # 2) track summary
    cat = Counter(t.category.name for t in s.tracks)
    typ = Counter(t.object_type.value for t in s.tracks)
    print("\n[2] TRACKS summary")
    print(f"   category : {dict(cat)}")
    print(f"   type     : {dict(typ)}")

    # 3) focal track
    focal = next(t for t in s.tracks if t.track_id == s.focal_track_id)
    states = sorted(focal.object_states, key=lambda st: st.timestep)
    obs = [st for st in states if st.observed]
    fut = [st for st in states if not st.observed]
    print("\n[3] FOCAL TRACK")
    print(f"   type/category : {focal.object_type.value} / {focal.category.name}")
    print(f"   states        : {len(states)}  =  observed {len(obs)} + future {len(fut)}")
    s0, sN = states[0], states[-1]
    print(f"   first t={s0.timestep:>3} obs={s0.observed}  pos={tuple(round(v,1) for v in s0.position)}")
    if obs:
        last_obs = obs[-1]  # t=49, normalization reference
        print(f"   t={last_obs.timestep:>3} (last observed)  pos={tuple(round(v,1) for v in last_obs.position)}  heading={last_obs.heading:.2f} rad")
    print(f"   last  t={sN.timestep:>3} obs={sN.observed}  pos={tuple(round(v,1) for v in sN.position)}")

    # 4) visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    for t in s.tracks:
        xy = np.array([st.position for st in t.object_states])
        if xy.shape[0] < 2:
            continue
        if t.track_id == s.focal_track_id:
            color, z, lw = FOCAL_COLOR, 5, 2.5
        elif t.track_id == "AV":
            color, z, lw = AV_COLOR, 4, 2.0
        else:
            color, z, lw = OTHER_COLOR, 2, 1.0
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, zorder=z)
        ax.scatter(xy[0, 0], xy[0, 1], color=color, s=18, zorder=z)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="X", s=45, zorder=z)

    ax.set_aspect("equal")
    ax.set_title(f"{s.scenario_id[:8]}  @ {s.city_name}   (focal=orange, AV=teal, others=light)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)

    out = Path(f"scenario_{s.scenario_id[:8]}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved figure -> {out.resolve()}")


if __name__ == "__main__":
    main()
