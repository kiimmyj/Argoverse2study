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

from lane_graph import Move          # 순환 import 없음: lane_graph 는 lane_frame 을 안 쓴다


LANE_HALF_W = 1.75      # 차로 반폭 [m] — d 의 상한
STEP = 0.25             # 경로 중심선 리샘플 간격 [m]
# successor 이음매 허용 오차 [m]. 실측 gap 은 27,915쌍에서 max 0.0000 m 라
# 이걸 넘으면 '메워야 할 틈'이 아니라 지도가 이상한 것이다.
SEAM_GAP_M = 0.5
# 경로가 '접혔다'고 볼 접선각 변화 [deg]. **접힘 검출기이지 급회전 필터가 아니다.**
# 30° 로 두면 안 된다 — 지도에 20.4 m 에 133° 를 도는 정상적인 교차로 차로가 실재하고
# (27,509개 차로 중 json 3개 max 106°, api 8개 max 51°), 그런 차로를 조용히 지운다.
# succ-only 에서 90° 초과는 실측 0건이라 이 게이트는 안전망으로만 남는다.
MAX_KINK_DEG = 90.0
# 경로가 자기 자신에게 되돌아온 것으로 볼 거리 [m] = 차로 폭.
# 이보다 가까워지면 같은 (s, d) 가 두 곳을 가리켜 from_frame 이 다가(多價)가 된다.
SELF_CLEAR_M = 2 * LANE_HALF_W
SELF_CLEAR_ARC_M = 10.0                 # 이만큼 호길이가 떨어진 쌍만 본다 (인접점 제외)
# 규칙상 차선변경이 허용된 쪽으로 넓힐 때의 횡오프셋 상한 [m].
# 왜 필요한가: succ-only 경로에서는 차선변경이 잔차 d 로 표현되므로 d 가 차로폭만큼 움직인다.
# ±1.75(차로 반폭)로 자르면 표현력 천장이 recall@40 = 79.7% 에서 막힌다(실측).
# 왜 평평하게 ±3.6 으로 안 넓히나: 그러면 밴드의 27.6% 가 대향차로, 21% 가 도로 밖이 되어
# '역주행·도로이탈 불가능'이라는 출력 공간의 보장이 통째로 사라진다.
WIDE_HALF_W = 3.6
HORIZON_S = 6.0                         # 예측 지평 [s]
ROUTE_MARGIN_M = 20.0                   # 지평 위에 얹는 경로 길이 여유 [m]


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


def stitch(polys: List[np.ndarray], max_gap: float = 3.0,
           bridge: bool = True) -> np.ndarray:
    """차로 중심선들을 이어 하나의 경로로. 이음매의 중복점·역행을 정리한다.

    그냥 concatenate 하면 차로 경계에서 점이 겹치거나 역방향으로 튀어
    호길이(s)가 비단조가 되고, 투영 결과가 지그재그로 꺾인다.

    bridge=False 면 벌어진 이음매를 **메우지 않고 ValueError 를 낸다** — 기준 경로에는
    이쪽을 써야 한다. 직선으로 메우는 분기는 successor 에서 한 번도 발동한 적이 없고
    (실측 2,651건 전부 LEFT/RIGHT/UTURN), 차선변경에서만 21 m 짜리 '뒤로 가는 다리'를
    만들었다. 이웃 차로는 현재 차로 옆에서 처음부터 시작하기 때문이다.
    그 다리 위에서 접선이 180° 뒤집혀 k 가 오염되고 theta = h - k 가 통째로 뒤집힌다.
    """
    out = [np.asarray(polys[0], float)]
    for q in polys[1:]:
        q = np.asarray(q, float)
        prev = out[-1][-1]
        gap = float(np.linalg.norm(q[0] - prev))
        if gap < 1e-3:
            q = q[1:]                                  # 중복점
        elif gap > max_gap:
            if not bridge:
                raise ValueError(f"이음매가 {gap:.2f} m 벌어졌다 (허용 {max_gap:.2f} m)")
            n = int(gap / STEP)                        # 벌어진 이음매는 직선으로 메움
            br = prev + np.linspace(0, 1, n)[:, None] * (q[0] - prev)
            out.append(br[1:-1])
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


def route_kink_deg(tan: np.ndarray, trim: int = 2) -> float:
    """경로 접선각의 스텝간 변화 최대값 [deg]. 경로가 접혔는지 재는 유일한 계기다.

    양 끝 trim 스텝은 뺀다. densify + np.gradient 는 끝점에서 편차분이 되어 실제보다 큰
    꺾임을 만든다 (최대 꺾임이 맨 끝 스텝인 경우가 13.4%, 양끝 2스텝을 빼면 >30° 비율이
    8.11% -> 1.51%).

    tangents() 의 n<1e-9 가드는 안전망이 아니다 — 접힘점에서는 중앙차분이 상쇄되어
    |grad| 가 0.25 -> 0.018 로 붕괴하는데도 그냥 단위벡터로 정규화되어 통과한다.
    """
    a = np.arctan2(np.asarray(tan)[:, 1], np.asarray(tan)[:, 0])
    if len(a) < 3:
        return 0.0
    da = np.degrees(np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))
    if trim > 0 and len(da) > 2 * trim + 1:
        da = da[trim:-trim]
    return float(da.max())


def self_return_idx(pts: np.ndarray, s: np.ndarray, clear: float = SELF_CLEAR_M,
                    arc: float = SELF_CLEAR_ARC_M, sub_m: float = 2.0) -> Optional[int]:
    """경로가 자기 자신에게 되돌아오는 첫 지점의 인덱스. 없으면 None.

    successor 만 따라가도 블록을 한 바퀴 돌아 출발점으로 돌아오는 사슬이 나온다.
    `nxt in path` 는 차로 중복만 막지 **공간적 되돌아옴은 못 막는다** — 실측으로
    시작점과 68.9 m 지점의 거리가 0.000 m 인 완전한 고리 경로가 나왔다.
    그런 경로는 같은 (s, d) 가 두 곳을 가리켜 from_frame 이 다가가 되고,
    to_frame 의 첫 대응점이 어느 끝을 잡을지 정해지지 않는다.

    max_hops 를 4 -> 6 으로 올리면서 생긴 문제다 (4홉 0건 -> 6홉 발생).
    """
    k = max(1, int(sub_m / STEP))
    P, S = pts[::k], s[::k]
    if len(P) < 3:
        return None
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    hit = np.triu((np.abs(S[:, None] - S[None, :]) >= arc) & (d2 < clear ** 2), 1)
    if not hit.any():
        return None
    return int(np.nonzero(hit)[1].min()) * k           # 되돌아옴이 성립하는 가장 이른 지점


def rule_band(graph, route, half_w: float = LANE_HALF_W,
              wide: float = WIDE_HALF_W) -> Tuple[np.ndarray, np.ndarray]:
    """경로 위 각 점의 좌/우 횡오프셋 상한 (lim_left, lim_right).

    **규칙상 갈 수 있는 쪽으로만** 넓힌다. lane_graph 가 이미 표시 종류 + 방향 검사로
    좌/우 차선변경 가능 여부를 계산해 두므로 그대로 쓴다.

    교차로 안은 예외로 양쪽을 넓힌다 — 교차로 차로는 좌우 이웃이 없어 규칙만 보면
    반폭으로 묶이는데, 실제 차량은 회전호를 그보다 크게 벗어나 코너를 자른다.
    이 완화 하나로 recall@5 가 83.1% -> 87.4% 로 오른다(실측).

    실측 recall@5 (val 3,000, 과거 적합도 정렬):
        ±1.75 고정      77.9%   (천장 79.7%)  보장 최대
        규칙 밴드        87.4%   (천장 89.9%)  보장 유지          <- 채택
        ±3.6 고정       94.5%   (천장 96.6%)  보장 상실
    """
    seg, acc = [], 0.0
    for lid in route["lanes"]:
        c = np.asarray(graph.lanes[lid].centerline, float)[:, :2]
        L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        mv = {m for _, m in graph.edges.get(lid, [])}
        inter = graph.lanes[lid].is_intersection
        seg.append((acc, acc + L,
                    wide if (inter or Move.LEFT in mv) else half_w,
                    wide if (inter or Move.RIGHT in mv) else half_w))
        acc += L
    S = route["s"]
    ll = np.full(len(S), half_w); lr = np.full(len(S), half_w)
    for a_, b_, wl, wr in seg:
        m = (S >= a_) & (S <= b_)
        ll[m], lr[m] = wl, wr
    return ll, lr


def resample_route(route, n: int) -> dict:
    """경로를 호길이 **등간격** n 점으로 다시 뽑는다. 모델 입력·L0 조회용.

    등간격이라 s -> 인덱스가 s / route_len * (n-1) 로 끝난다 (탐색 불필요).
    접선은 각도로 보간하면 ±pi 에서 튀므로 **단위벡터로 보간한 뒤 정규화**한다.
    """
    S = route["s"]
    t = np.linspace(0.0, float(S[-1]), n)
    P = np.stack([np.interp(t, S, route["pts"][:, 0]),
                  np.interp(t, S, route["pts"][:, 1])], axis=1)
    T = np.stack([np.interp(t, S, route["tan"][:, 0]),
                  np.interp(t, S, route["tan"][:, 1])], axis=1)
    nn = np.linalg.norm(T, axis=1, keepdims=True); nn[nn < 1e-9] = 1.0
    return {"pts": P, "tan": T / nn, "s": t, "len": float(S[-1])}


def build_routes(graph, starts, reach, max_hops: Optional[int] = None,
                 min_len_m: Optional[float] = None, max_routes: int = 40,
                 v0: Optional[float] = None, legacy: bool = False,
                 max_kink_deg: Optional[float] = None,
                 stats: Optional[dict] = None) -> List[dict]:
    """시작 차로에서 **successor 만** 따라가며 '경로'(차로 시퀀스)를 만든다.

    왜 successor 만 쓰나
    --------------------
    graph.movable() 은 차선변경(LEFT/RIGHT)도 돌려주지만 그것을 **기하로** 따라가면 안 된다.
    이웃 차로는 현재 차로 옆에서 '처음부터' 시작하므로(종방향 겹침/A 중앙값 0.992)
    이어붙이면 이음매가 중앙값 21 m 뒤로 접힌다 — 실측으로 열거된 경로의 54% 가 접혀 있었고
    (차선변경이 든 경로의 99%, succ 만 쓴 경로는 0%), 그 경로를 기준으로 잡으면
    theta = h - k 가 그 구간 통째로 180° 뒤집힌다.

    더 중요한 이유는 **지도가 전이 지점을 주지 않는다**는 것이다. 어디서 차로를 바꿀지는
    두 차로가 겹치는 구간 전체에 걸친 연속 자유변수라, 이음매 위치를 고르는 순간 지도에
    없는 값을 기하에 박게 되고 그 전이 곡률(0.026~0.071 1/m, 실제 완만한 커브의 3~7배)이
    kappa_lane('도로의 휘어짐')으로 둔갑한다. v4 의 h = k + theta 는 k 가 도로만 담아야 성립한다.

    차선변경은 버리는 게 아니라 **다른 축으로 옮긴다**:
      · 위상까지 바뀌는 차선변경 — candidate_lanes 반경 5.0 m 가 차로 폭 3.42 m 보다 커서
        이웃 차로가 이미 시작 후보에 들어온다. 목표 차로 사슬이 별도의 succ-only 경로로
        열거되므로 여기서 따로 만들 필요가 없다.
      · 같은 차로 사슬 안에서의 차선변경 — 잔차 (d, theta) 가 표현한다.
    실측: recall@6(경로가 GT 를 덮고 max|d|<=3.6) 87.1% -> 93.1%, GT 차선변경 부분집합은
    97.2% -> 100.0%. 반면 recall@40 은 94.3% vs 93.6% 로 동률 — 즉 차선변경 경로는
    표현 가능한 집합에 아무것도 보태지 않으면서 상위 K 자리만 먹었다.

    UTURN 을 왜 빼나
    ----------------
    lane_graph.reachable() 이 이미 같은 이유로 뺀다 — UTURN 은 SUCC 여러 개를 건너뛴
    '표식'이라 이음매가 중앙값 32 m 벌어져 있다. 같은 목적지는 SUCC 사슬로 도달한다
    (복원 성공 18/18, 복원해서 넣어도 경로 집합이 안 변함).

    min_len_m 을 왜 v0 로 정하나
    ----------------------------
    from_frame 은 s 를 경로 끝으로 clip 하므로, 경로가 주행거리보다 짧으면 궤적이 끝에서
    얼어붙어 학습으로 못 넘는 ADE 하한이 생긴다. 필요 s 는 중앙 36.5 m 인데 차로 길이
    중앙이 19.8 m 라 4홉으로는 못 채운다(실측: 시나리오의 34.5% 가 6초를 못 덮었다).
    6홉 + min_len = 6*v0 + 20 m 로 두면 '주행거리보다 짧은 경로'가 12.5% -> 1.6% 가 된다.

    legacy=True 는 이 커밋 이전 동작(차선변경·UTURN 포함, 다리 메움, 4홉, 30 m)을 그대로
    돌려준다. A/B 비교용이며 라벨 생성에 쓰면 안 된다. 명시한 인자는 덮어쓰지 않는다.
    """
    if max_hops is None:
        max_hops = 4 if legacy else 6
    if max_kink_deg is None:
        max_kink_deg = float("inf") if legacy else MAX_KINK_DEG
    if min_len_m is None:
        min_len_m = (30.0 if (legacy or v0 is None)
                     else max(30.0, float(v0) * HORIZON_S + ROUTE_MARGIN_M))

    n_seam = n_kink = n_loop = 0
    dropped = []
    if max_hops is None:                # 시그니처만 Optional 로 바뀌고 본문이 안 따라와 있었다.
        max_hops = 4                    # 기존 호출부(인자 없음)의 동작을 그대로 유지한다.
    routes, seen = [], set()
    stack = [(s,) for s in starts]
    while stack and len(routes) < max_routes * 4:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)

        # 확장 후보를 **먼저** 만든다. 막다른 길이면 길이가 모자라도 채택해야 하기 때문이다.
        # (이걸 안 하면 지도 경계나 reach 예산으로 6홉 전에 끊긴 사슬이 통째로 버려져
        #  '경로 0개' 시나리오가 새로 생긴다 — 실측 0.155%, train 20만 기준 약 390개)
        ext = []
        if len(path) < max_hops:
            for nxt, mv in graph.movable(path[-1]):
                if not legacy and mv is not Move.SUCC:
                    continue
                if reach and nxt not in reach:
                    continue
                if nxt in path:
                    continue
                ext.append(path + (nxt,))

        polys = [np.asarray(graph.lanes[l].centerline, float)[:, :2] for l in path]
        try:
            pts = (stitch(polys) if legacy
                   else stitch(polys, max_gap=SEAM_GAP_M, bridge=False))
        except ValueError:
            # 벌어진 이음매는 확장해도 그대로 남는다. 여기서만 하위 트리를 가지친다.
            n_seam += 1
            continue

        if len(pts) >= 2:
            d, sarc = densify(pts)
            cut = None if legacy else self_return_idx(d, sarc)
            if cut is not None and cut >= 2:
                n_loop += 1
                d, sarc = d[:cut], sarc[:cut]           # 되돌아오기 직전까지만 쓴다
            t = tangents(d)
            # 꺾임 게이트는 채택만 막고 **확장은 막지 않는다**. 급회전 차로가 경로 중간에
            # 있을 뿐이면 그 뒤 경로는 멀쩡하다. 3 m 미만은 trim 이 안 먹어 게이트에서 뺀다.
            kink = route_kink_deg(t) if sarc[-1] >= 3.0 else 0.0
            if kink > max_kink_deg:
                n_kink += 1
                if len(dropped) < 20:
                    dropped.append((path, round(kink, 2)))
            elif (sarc[-1] >= min_len_m or len(path) >= max_hops
                  or (not legacy and not ext)):     # legacy 는 '수정 전 그대로'여야 한다
                routes.append({"lanes": path, "pts": d, "s": sarc, "tan": t})

        stack.extend(ext)
    routes.sort(key=lambda r: -r["s"][-1])
    routes = routes[:max_routes]
    if stats is not None:                    # 게이트가 몇 번 걸렸는지 밖에서 볼 수 있게
        for k_, v_ in (("n_seam_drop", n_seam), ("n_kink_drop", n_kink),
                       ("n_loop_cut", n_loop), ("n_routes", len(routes))):
            stats[k_] = stats.get(k_, 0) + v_
        stats.setdefault("kink_examples", []).extend(dropped)
    return routes


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
