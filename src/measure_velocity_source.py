"""속도를 **어디서 가져올 것인가** — AV2 가 준 속도 vs 위치차분으로 다시 구한 속도.

왜 재나
-------
입력 규격의 `a` 는 **위치차분 속력**에서 만든다(`|Δp|/dt`). 그런데 AV2 는 속도 필드를
따로 준다(`velocity_x, velocity_y`). 왜 안 쓰는지가 규격에 "복원이 정의상 0 이 된다" 한
줄로만 적혀 있었다. **그 선택의 대가가 얼마인지는 안 적혀 있었다.**

⚠️ **AV2 에 가속도 필드는 없다.** 열은 position · heading · velocity 뿐이다.
   그래서 비교 대상은 "AV2 가속도" 가 아니라 "AV2 속도" 다.

이것이 왜 중요하냐면 — **원본 SIMPL 입력에는 AV2 속도벡터 2칸이 들어 있었다**
(`av2_preprocess.py:191`). R1 은 움직임 6칸을 3칸으로 줄이면서 그 2칸을 **버렸다.**
즉 "R1 이 왜 졌나" 의 후보가 하나 더 있다 — 표현이 나빠서도, 반쪽만 바꿔서도 아니고,
**추적기가 걸러 준 깨끗한 신호를 하나 버려서**.

    python src/measure_velocity_source.py --n 300
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import viz_v4_common as C

OUT = C.ROOT / "viz" / "velocity_source"
DOC = C.ROOT / "docs" / "action_spec.md"
DT = 0.1
QS = [50, 90, 99]


def one(f):
    """장면 하나 → (속도 차이, 위치차분 변화율, AV2 변화율, 복원오차 둘)."""
    d = pd.read_parquet(f, columns=["track_id", "timestep", "position_x", "position_y",
                                    "velocity_x", "velocity_y", "object_type"])
    gap, a_pd, a_av2, r_pd, r_av2 = [], [], [], [], []
    for _, g in d[d.object_type == "vehicle"].groupby("track_id"):
        g = g.sort_values("timestep")
        # 연속 트랙만 — 끊긴 구간을 차분하면 점프가 '가속도' 로 들어온다
        if len(g) < 60 or np.diff(g.timestep.values).max() != 1:
            continue
        p = g[["position_x", "position_y"]].to_numpy(float)
        v_av2 = np.linalg.norm(g[["velocity_x", "velocity_y"]].to_numpy(float), axis=1)
        dp = np.diff(p, axis=0)
        v_pd = np.linalg.norm(dp, axis=1) / DT
        h = np.arctan2(dp[:, 1], dp[:, 0])
        u = np.stack([np.cos(h), np.sin(h)], 1)
        gap.append(np.abs(v_av2[:-1] - v_pd))
        a_pd.append(np.abs(np.diff(v_pd)) / DT)
        a_av2.append(np.abs(np.diff(v_av2)) / DT)
        for v, box in ((v_pd, r_pd), (v_av2[:-1], r_av2)):
            box.append(float(np.linalg.norm(
                p[0] + np.cumsum((v * DT)[:, None] * u, 0)[-1] - p[-1])))
    return gap, a_pd, a_av2, r_pd, r_av2


def md_block(S):
    q = lambda d, k: " | ".join(f"{d[f'p{x}']:.{k}f}" for x in QS)
    return "\n".join([
        f"AV2 val {S['n_scene']}장면 · 차량 트랙 {S['n_track']:,}개 "
        f"(`src/measure_velocity_source.py`):", "",
        "| | 중앙 | p90 | p99 |", "| --- | --- | --- | --- |",
        f"| AV2 속도 − 위치차분 속도 [m/s] | {q(S['gap'], 3)} |",
        f"| 속도 **변화율** — 위치차분에서 [m/s²] | {q(S['a_pd'], 2)} |",
        f"| 속도 **변화율** — AV2 속도에서 [m/s²] | {q(S['a_av2'], 2)} |", "",
        "| 끝점 복원 오차 | 중앙 | p99 |", "| --- | --- | --- |",
        f"| **위치차분 속도로 적분** | **{S['rec_pd']['p50']:.4f} m** | **{S['rec_pd']['p99']:.4f} m** |",
        f"| AV2 속도로 적분 | {S['rec_av2']['p50']:.4f} m | {S['rec_av2']['p99']:.4f} m |",
    ])


def fill(S):
    if not DOC.exists():
        return
    t = DOC.read_text()
    a, b = "<!-- VELSRC:START -->", "<!-- VELSRC:END -->"
    if a not in t or b not in t:
        print("  (표식 VELSRC 없음 — 건너뜀)")
        return
    head, rest = t.split(a, 1)
    _, tail = rest.split(b, 1)
    DOC.write_text(f"{head}{a}\n\n{md_block(S)}\n\n{b}{tail}")
    print(f"  문서 갱신 → {DOC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    files = sorted(glob.glob("/data/argoverse2/motion_forecasting/val/*/*.parquet"))[:a.n]
    acc = [[] for _ in range(5)]
    for f in files:
        for box, got in zip(acc, one(f)):
            box += got
    gap, a_pd, a_av2 = (np.concatenate(x) for x in acc[:3])
    r_pd, r_av2 = (np.asarray(x) for x in acc[3:])

    def qd(x):
        return {f"p{q}": float(np.percentile(x, q)) for q in QS} | {"n": int(np.size(x))}
    S = {"n_scene": len(files), "n_track": len(r_pd), "gap": qd(gap),
         "a_pd": qd(a_pd), "a_av2": qd(a_av2), "rec_pd": qd(r_pd), "rec_av2": qd(r_av2),
         "note": "AV2 에 가속도 필드는 없다. velocity 필드와 위치차분 속도의 비교다."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
    print(md_block(S).replace("| ", "  ").replace("|", ""))
    fill(S)
    print(f"\n숫자 → {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
