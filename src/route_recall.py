"""
route_recall.py - L3 선행 확인. 후보 경로 상위 K개가 정답 미래를 표현할 수 있는가.

왜 이걸 먼저 재나
-----------------
L3 는 '후보 경로 K개를 예측 모드로 삼는' 설계다. 후보 집합에 정답 경로가 없으면
그 차량은 **학습으로 넘을 수 없는 천장**에 걸린다 — 모델을 아무리 잘 만들어도
표현할 수 없는 미래이기 때문이다. 문서(docs/v4_map_levels.md)가 정한 판정선은
recall@5 >= 0.90 이고, 미달이면 L3·L0 을 함께 포기하고 L2+L4 로 확정하라고 되어 있다.

'표현 가능' 의 정의
-------------------
경로가 (1) 미래 주행거리를 **덮고**, (2) 정답 궤적을 투영했을 때 **max|d| <= w** 인 것.
w 는 출력 공간이 허용하는 횡오프셋 상한이다. 지금 project() 는 half_w = 1.75 (차로 반폭)
로 자르므로 그게 '출하 형상'이고, 회랑 밴드를 넓히면 3.6 까지 갈 수 있다. 둘 다 잰다.

순위를 무엇으로 매기나
----------------------
recall@K 는 '상위 K개' 이므로 순위가 결과를 바꾼다. 두 가지를 비교한다.
  length : 현행 build_routes 의 정렬(호길이 내림차순) — 미래를 안 보지만 임의적이다
  past   : **관측된 과거 5초와의 적합도** — 모델이 실제로 쓸 수 있는 정보만 쓴다
past 가 더 좋다면 L3 의 모드 순서를 이걸로 매기면 되고, 그건 공짜다.

  python src/route_recall.py --limit 3000
"""
import argparse
import hashlib
import json
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane_frame as lf
from heading_decomp import build_heading, focal, graph_of, OBS_LEN
from lane_graph import Move, REACH_MARGIN_M

ROOT = Path("/data/argoverse2/motion_forecasting")
KS = (1, 3, 5, 6, 10, 20, 40)
WS = (1.75, 2.5, 3.6)
WIDE_M = 3.6          # 규칙상 갈 수 있는 쪽으로 넓힐 때의 상한 [m]


def one(d, min_move_m=0.0):
    try:
        pos, vel, hav2 = focal(d)
        if len(pos) < OBS_LEN + 2:
            return None
        # 관측 구간만 넘긴다 — 전방차분이라 전체를 넣으면 h[OBS_LEN-1] 이 첫 미래 위치를 쓰고,
        # 그 방향이 아래 candidate_lanes 의 시작 차로 선택에 들어가 recall 을 미래 쪽으로 기울인다.
        h, _ = build_heading(pos[:OBS_LEN], h_ref=hav2[:OBS_LEN])
        g = graph_of(d)
        p0 = pos[OBS_LEN - 1]
        h0 = np.array([np.cos(h[OBS_LEN - 1]), np.sin(h[OBS_LEN - 1])])
        starts = g.candidate_lanes(p0, h0, path=pos[:OBS_LEN])
        if not starts:
            return None
        v0 = float(np.linalg.norm(pos[OBS_LEN - 1] - pos[OBS_LEN - 2])) / 0.1
        reach = g.reachable(starts, max(20.0, v0 * 6.0) + REACH_MARGIN_M)
        routes = lf.build_routes(g, starts, reach, v0=v0)
        if not routes:
            return None
        past, fut = pos[:OBS_LEN], pos[OBS_LEN:]
        move = float(np.linalg.norm(np.diff(fut, axis=0), axis=1).sum())

        # 각 경로에 대해: 과거 적합도(순위용), 미래 max|d|(표현력), 덮는가
        score, mx, okr = [], [], []
        for r in routes:
            _, dp, mdp = lf.to_frame(past, r)
            sf, df, _ = lf.to_frame(fut, r)
            score.append(mdp)
            mx.append(float(np.abs(df).max()))
            ll, lr = lf.rule_band(g, r)                 # 규칙 기반 비대칭 밴드
            j = np.clip(np.searchsorted(r["s"], sf), 0, len(r["s"]) - 1)
            okr.append(bool(np.all((df <= ll[j]) & (-df <= lr[j]))))
        mx = np.array(mx); okr = np.array(okr)
        ord_len = np.arange(len(routes))                 # build_routes 가 이미 길이순
        ord_past = np.argsort(np.array(score))
        out = {}
        for name, o in (("length", ord_len), ("past", ord_past)):
            for w in WS + ("rule",):
                ok = okr[o] if w == "rule" else (mx[o] <= w)
                for K in KS:
                    out[(name, w, K)] = bool(ok[:K].any())
        return out, move, len(routes)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-move", dest="min_move", type=float, default=10.0,
                    help="이 거리 미만만 움직인 시나리오는 따로 집계 (정지 차량은 경로가 무의미)")
    ap.add_argument("--out", default=None,
                    help="표를 json 으로 남긴다 (예: runs/v4_route_recall.json) — 문서 수치의 출처")
    a = ap.parse_args()
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one, dirs, chunksize=8) if r]
    move = np.array([r[1] for r in res])
    nrt = np.array([r[2] for r in res])
    sel_all = np.ones(len(res), bool)
    sel_mov = move >= a.min_move

    print(f"\n=== [{a.split}] L3 후보 경로 recall  (시나리오 {len(res):,}) ===")
    print(f"시나리오당 경로 {nrt.mean():.1f}개   6초 이동거리 중앙 {np.median(move):.1f} m   "
          f"{a.min_move:.0f} m 이상 움직인 시나리오 {sel_mov.mean()*100:.1f}%")
    print("\n'표현 가능' = 정답 미래를 그 경로에 투영했을 때 max|d| <= w")
    for tag, sel in (("전체", sel_all), (f"{a.min_move:.0f} m 이상 이동", sel_mov)):
        print(f"\n--- {tag} (n={int(sel.sum()):,}) ---")
        for name in ("length", "past"):
            lab = "현행 정렬(길이)" if name == "length" else "과거 적합도 정렬"
            print(f"  [{lab}]  " + "  ".join(
                f"{('w='+str(w)) if w != 'rule' else '규칙밴드':>8}" for w in WS + ("rule",)))
            for K in KS:
                row = f"  recall@{K:<3}   "
                for w in WS + ("rule",):
                    v = np.array([r[0][(name, w, K)] for r in res])[sel].mean() * 100
                    mark = "*" if (K == 5 and v >= 90.0) else " "
                    row += f"  {v:6.1f}%{mark}"
                print(row)
    print("\n* = 문서 판정선 recall@5 >= 0.90 충족")

    if a.out:
        # 두 모집단(전체 / 일정 거리 이상 이동)을 모두 남긴다 — 게이트를 어느 쪽으로 읽느냐에 따라 미달 폭이 달라진다.
        subsets = {"all": sel_all, f"move>={a.min_move:g}m": sel_mov}
        table = {tag: {"n": int(sel.sum()),
                       "recall": {name: {str(w): {str(K): round(float(
                           np.array([r[0][(name, w, K)] for r in res])[sel].mean()), 4)
                           for K in KS} for w in WS + ("rule",)} for name in ("length", "past")}}
                 for tag, sel in subsets.items()}
        out = {"tag": Path(a.out).stem, "date": time.strftime("%Y-%m-%d"),
               "code_sha12": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
               "args": vars(a), "scenarios": len(res), "table": table,
               "keys": "table[모집단].recall[정렬(length=현행 길이순, past=과거 적합도)][밴드 w 또는 rule][K]"}
        Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
