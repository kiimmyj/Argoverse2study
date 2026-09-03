"""
lane_frame.py - 차로 기준 좌표계(④안)의 출력 공간을 구현한다.

④안은 좌표를 직접 뱉지 않는다. 대신:
  ① 갈 차로(경로)를 도달 가능한 것들 중에서 고르고
  ② 그 경로 중심선을 따라 (진행거리 s, 횡오프셋 d) 로 예측하고
  ③ s, d 를 중심선에 얹어 좌표로 환원한다

이렇게 하면 구조적으로 보장된다:
  - 고른 경로가 규칙상 도달 가능하므로 → **역주행 불가능**
  - d 를 차로 반폭(±1.75m)으로 제한하므로 → **도로 이탈 불가능**
  - s 를 단조증가로 두므로 → **경로 역행 불가능**

확률적 억제(②안 손실 벌점)와 달리 학습이 어떻든 위반이 나올 수 없다.

여기서는 학습된 ④ 모델 대신, 기존 예측을 이 좌표계로 **투영**해서
출력 공간이 무엇을 허용/금지하는지 보인다.
"""
from typing import List, Tuple, Optional

import numpy as np

LANE_HALF_W = 1.75      # 차로 반폭 [m] — d 의 상한
STEP = 0.25             # 경로 중심선 리샘플 간격 [m]


def densify(poly: np.ndarray, step: float = STEP) -> Tuple[np.ndarray, np.ndarray]:
    """폴리라인을 등간격으로 촘촘히. (점 (M,2), 누적거리 (M,)) 를 돌려준다."""
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return poly[:1], s[:1]
    t = np.arange(0.0, s[-1], step)
    t = np.append(t, s[-1])
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1), t


def stitch(polys: List[np.ndarray], max_gap: float = 3.0) -> np.ndarray:
    """차로 중심선들을 이어 하나의 경로로. 이음매의 중복점·역행을 정리한다.

    그냥 concatenate 하면 차로 경계에서 점이 겹치거나 역방향으로 튀어
    호길이(s)가 비단조가 되고, 투영 결과가 지그재그로 꺾인다.
    """
    out = [np.asarray(polys[0], float)]
    for q in polys[1:]:
        q = np.asarray(q, float)
        prev = out[-1][-1]
        gap = float(np.linalg.norm(q[0] - prev))
        if gap < 1e-3:
            q = q[1:]                                  # 중복점
        elif gap > max_gap:
            n = int(gap / STEP)                        # 벌어진 이음매는 직선으로 메움
            bridge = prev + np.linspace(0, 1, n)[:, None] * (q[0] - prev)
            out.append(bridge[1:-1])
        if len(q):
            out.append(q)
    P = np.concatenate(out)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])
    return P[keep]


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """가장자리를 보존하는 이동평균. w<=1 이면 그대로."""
    if w <= 1 or len(x) < 3:
        return x
    w = min(w, len(x) if len(x) % 2 else len(x) - 1)
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(w) / w, mode="valid")[:len(x)]


def tangents(pts: np.ndarray) -> np.ndarray:
    g = np.gradient(pts, axis=0)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return g / n


def build_routes(graph, starts, reach, max_hops: int = 4,
                 min_len_m: float = 30.0, max_routes: int = 40) -> List[dict]:
    """시작 차로에서 규칙이 허용하는 이동만 따라가며 '경로'(차로 시퀀스)를 만든다.

    graph.movable(lane) 이 규칙(successors / 허용된 차선변경 / U턴)만 돌려주므로
    여기서 나온 경로는 전부 규칙을 만족한다.
    """
    routes, seen = [], set()
    stack = [(s,) for s in starts]
    while stack and len(routes) < max_routes * 4:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        pts = stitch([np.asarray(graph.lanes[l].centerline, float)[:, :2] for l in path])
        d, s = densify(pts)
        if s[-1] >= min_len_m or len(path) >= max_hops:
            routes.append({"lanes": path, "pts": d, "s": s, "tan": tangents(d)})
        if len(path) < max_hops:
            for nxt, _mv in graph.movable(path[-1]):
                if reach and nxt not in reach:
                    continue
                if nxt in path:
                    continue
                stack.append(path + (nxt,))
    routes.sort(key=lambda r: -r["s"][-1])
    return routes[:max_routes]


def to_frame(traj: np.ndarray, route: dict, fwd_window_m: float = 25.0
             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """궤적 (T,2) → (s (T,), d (T,), 평균 |d|). d 는 좌(+)/우(−) 부호.

    각 점의 대응 위치를 **직전 점보다 앞에서만** 찾는다. 전역 최근접으로 찾으면
    경로가 자기 근처로 되돌아오는 구간(교차로·U턴)에서 대응점이 멀리 튀고,
    그 결과 s 가 앞뒤로 요동쳐 궤적이 꺾인다.
    """
    P, S, T = route["pts"], route["s"], route["tan"]
    win = max(4, int(fwd_window_m / STEP))
    j_prev, js = 0, np.empty(len(traj), int)
    for i, q in enumerate(traj):
        lo = j_prev if i else 0
        hi = min(len(P), (j_prev + win) if i else len(P))
        k = int(np.argmin(((P[lo:hi] - q) ** 2).sum(axis=1))) + lo
        js[i] = j_prev = k
    v = traj - P[js]
    t = T[js]
    d = v[:, 0] * (-t[:, 1]) + v[:, 1] * t[:, 0]
    s = S[js] + (v[:, 0] * t[:, 0] + v[:, 1] * t[:, 1])
    return s, d, float(np.abs(d).mean())


def from_frame(s: np.ndarray, d: np.ndarray, route: dict) -> np.ndarray:
    """(s, d) → 궤적 (T,2). 중심선 위 s 지점에 법선 방향으로 d 만큼."""
    P, S, T = route["pts"], route["s"], route["tan"]
    s = np.clip(s, S[0], S[-1])
    x = np.interp(s, S, P[:, 0]); y = np.interp(s, S, P[:, 1])
    tx = np.interp(s, S, T[:, 0]); ty = np.interp(s, S, T[:, 1])
    n = np.sqrt(tx ** 2 + ty ** 2); n[n < 1e-9] = 1.0
    tx, ty = tx / n, ty / n
    return np.stack([x - ty * d, y + tx * d], axis=1)


def project(traj: np.ndarray, routes: List[dict], half_w: float = LANE_HALF_W,
            monotonic: bool = True, smooth_w: int = 7,
            max_d_rate: float = 0.12) -> Optional[dict]:
    """궤적을 ④안이 표현할 수 있는 형태로 투영한다.

    경로들 중 가장 잘 맞는 것을 고르고, d 를 ±half_w 로 자르고,
    s 를 단조증가로 만든 뒤 좌표로 되돌린다.
    """
    if not routes:
        return None
    best = None
    for r in routes:
        s, d, md = to_frame(traj, r)
        if best is None or md < best[0]:
            best = (md, s, d, r)
    md, s, d, r = best
    # 학습된 v4 는 (s, d) 프로파일을 직접 내놓으므로 매끄럽다.
    # 여기서는 기존 예측을 투영한 것이라 잔떨림이 남아 같은 수준으로 다듬는다.
    d_c = np.clip(smooth(d, smooth_w), -half_w, half_w)
    # s 가 단조 고정으로 멈춘 구간에서 d 만 좌우로 흔들리면 궤적이 180° 꺾인다.
    # 실제 차량은 0.1초에 옆으로 확 못 움직이므로 횡방향 변화율을 제한한다.
    step = np.diff(d_c)
    d_c = d_c[:1].tolist()
    for ds in step:
        d_c.append(d_c[-1] + float(np.clip(ds, -max_d_rate, max_d_rate)))
    d_c = np.array(d_c)
    s_c = smooth(s, smooth_w)
    if monotonic:
        s_c = np.maximum.accumulate(s_c)
    out = from_frame(s_c, d_c, r)
    return {"traj": out, "route": r, "s": s_c, "d": d_c,
            "mean_abs_d_before": md, "clipped": float(np.mean(np.abs(d) > half_w)),
            "n_lanes": len(r["lanes"])}
