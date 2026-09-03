"""
check_heading_quality.py - AV2 focal heading 이 v4 의 h = k + θ 분해에 쓸 만한지 전수 검사.

v4 는 방향을 '차로각 k + 차체 잔차각 θ' 로 분해한다. 이때 heading 라벨이 튀면
θ 가 오염되므로, 스텝간 변화 |Δh| 를 전 데이터에서 세어 튀는 값의 규모를 확인한다.

AV2 논문은 motion forecasting 트랙이 자동 멀티센서 퓨전 결과이며
"imperfect tracking" 이라고 명시한다 (arXiv 2301.00493). 실제로 얼마나 그런지 재는 것.

  python src/check_heading_quality.py          # train+val+test 전체 (약 70분, 56코어)
"""
import sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization

ROOT = Path("/data/argoverse2/motion_forecasting")
BINS = np.concatenate([np.arange(0, 10, 0.05), np.arange(10, 190, 1.0), [1e9]])

def one(d):
    try:
        s = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
        f = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        st = sorted(f.object_states, key=lambda x: x.timestep)
        H = np.array([x.heading for x in st], float)
        if len(H) < 2:
            return None
        dh = np.degrees(np.abs((np.diff(H) + np.pi) % (2 * np.pi) - np.pi))
        hist, _ = np.histogram(dh, bins=BINS)
        return (hist, float(dh.max()), len(dh),
                int((dh > 30).sum()), int((dh > 90).sum()), int((dh > 150).sum()), d.name)
    except Exception:
        return None

def run(split, workers=56):
    dirs = [p for p in sorted((ROOT / split).iterdir()) if p.is_dir()]
    with Pool(workers) as pool:
        res = [r for r in pool.imap_unordered(one, dirs, chunksize=200) if r]
    hist = np.sum([r[0] for r in res], axis=0)
    mx = np.array([r[1] for r in res])
    steps = sum(r[2] for r in res)
    n30, n90, n150 = (sum(r[i] for r in res) for i in (3, 4, 5))
    worst = sorted(res, key=lambda r: -r[1])[:5]
    return dict(split=split, n=len(res), steps=steps, hist=hist, mx=mx,
                n30=n30, n90=n90, n150=n150, worst=[(r[6], r[1]) for r in worst])

def pct(hist, q):
    c = np.cumsum(hist) / hist.sum()
    i = np.searchsorted(c, q / 100.0)
    return BINS[min(i, len(BINS) - 2)]

if __name__ == "__main__":
    out = []
    for sp in ["val", "test", "train"]:
        r = run(sp); out.append(r)
        print(f"[{sp}] focal {r['n']:,}개 / {r['steps']:,} step  "
              f">30° {r['n30']:,}  >90° {r['n90']:,}  >150° {r['n150']:,}", flush=True)
    H = np.sum([r["hist"] for r in out], axis=0)
    MX = np.concatenate([r["mx"] for r in out])
    S = sum(r["steps"] for r in out); N = sum(r["n"] for r in out)
    n30, n90, n150 = (sum(r[k] for r in out) for k in ("n30", "n90", "n150"))
    print("\n" + "=" * 68)
    print(f"=== 전체 {N:,} 시나리오 / {S:,} step ===")
    print("=" * 68)
    print(f"  |Δh| 중앙 {pct(H,50):.3f}°   p95 {pct(H,95):.2f}°   p99 {pct(H,99):.2f}°   "
          f"p99.9 {pct(H,99.9):.1f}°   p99.99 {pct(H,99.99):.1f}°")
    print(f"  최대 {MX.max():.1f}°")
    print(f"\n  |Δh| > 30°  : {n30:,} step ({n30/S*100:.5f}%)")
    print(f"  |Δh| > 90°  : {n90:,} step ({n90/S*100:.6f}%)")
    print(f"  |Δh| > 150° : {n150:,} step ({n150/S*100:.6f}%)")
    print(f"\n  시나리오 단위 최대 |Δh| 의 분포: 중앙 {np.median(MX):.2f}°  "
          f"p99 {np.percentile(MX,99):.1f}°  p99.9 {np.percentile(MX,99.9):.1f}°")
    print(f"  최대 |Δh| > 30° 인 시나리오 {int((MX>30).sum()):,} / {N:,} ({(MX>30).mean()*100:.4f}%)")
    print(f"  최대 |Δh| > 90° 인 시나리오 {int((MX>90).sum()):,} ({(MX>90).mean()*100:.4f}%)")
    for r in out:
        print(f"\n  [{r['split']}] 최악 5개: " + ", ".join(f"{s[:8]} {a:.0f}°" for s, a in r["worst"]))
