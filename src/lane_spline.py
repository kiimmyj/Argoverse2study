"""
lane_spline.py - succ-only 경로 위에 매끄러운 곡선을 적합해 kappa_lane(s) 테이블을 만든다.

왜 필요한가
-----------
v4 L0 은 곡률을 kappa = kappa_lane(s) + delta 로 분해한다. 그런데 지금 지도에서 곡률을
그냥 뽑으면 **잡음이 나온다**. 원인은 지도 데이터의 형태에 있다.

  AV2 차로 중심선은 **차로당 정확히 10점**이다 (val 120 시나리오 8,030개 차로 전부).
  차로 길이 중앙이 19.97 m 라 정점 간격이 중앙 2.22 m / p95 6.51 m / max 20.36 m 다.
  lane_frame.densify() 는 그 사이를 **직선으로** 채운다.

직선 위에서는 곡률이 0 이고 정점에서만 방향이 꺾이므로, 유한차분 곡률은 '선분 위 0 +
정점 델타함수'가 된다. 실측 (val 120 시나리오, 경로 1,147개, 40.3만 점):

    |kappa|      p50 0.0000  p95 0.0382  p99 0.2994  max 34.128 1/m   <- 반경 2.9 cm
    |dkappa/ds|  p50 0.0000  p95 0.2033  p99 0.8940  max 133.8  1/m^2

10점은 매끄러운 도로를 성기게 샘플한 것이지 도로가 실제로 꺾인 자리가 아니다. 그래서
정점 사이를 직선이 아니라 **곡선으로** 메운다.

무엇을 골랐고 왜인가
--------------------
`--task compare` 로 잰 결과다 (val 120, 경로 1,147개).

  ① 3차 스플라인 (5차 아님)
     C2 라 **곡률이 연속**이고 그게 L0 의 요구사항이다. 5차(C4, dkappa/ds 까지 연속)도
     재봤지만 같은 tol=0.02 에서 |dkappa/ds| max 가 0.736 -> 1.618 로 나빠지고 끝에서
     1 m 구간의 |kappa| 평균이 0.0108 -> 0.0190 으로 부푼다. 차로당 10점뿐인 데이터에
     차수를 올리면 매끄러워지는 게 아니라 흔들린다.

  ② 잔차제약(FITPACK) 이지 벌점평활(smoothing spline) 이 아니다  <- 여기가 핵심
     벌점평활은 **경계에서 f''=0 을 강제한다**. 부과하는 게 아니라 벌점 자체에서
     나온다(Reinsch: 평활 스플라인은 곧 natural spline). 결과가 이렇다:

        경로 끝에서의 거리        0~1m     1~2m    2~5m   5~10m  10~20m
        벌점평활 lam=1        0.00134  0.00492 0.01005 0.01398 0.01478   <- 끝에서 11배 붕괴
        FITPACK  tol=0.02    0.01079  0.00964 0.01090 0.01409 0.01479   <- 평탄

     참값을 아는 합성 곡선으로 확인하면 명확하다.

        곡률 복원          내부(중앙80%)      s=0 끝점
        R=25 m 사분원   벌점 +2.4% / FIT +0.7%   벌점 -100.0% / FIT +12.2%
        R=150 m 커브    벌점 +2.7% / FIT +0.1%   벌점 -100.0% / FIT  +2.5%

     벌점평활은 양 끝에서 **정확히 0** 을 낸다. FITPACK 은 급커브에서 +12%, 완만한
     커브에서 +2.5% 과대에 그치고 내부 정확도도 더 낫다.

     이게 왜 치명적이냐면 **오염 구간이 실제로 쓰는 s 구간에 걸리기 때문**이다.
     실측(val 400): 예측 시작점이 경로 앞끝 5 m 안인 경우 18.2%, 6초 뒤 지점이 뒷끝
     5 m 안인 경우 13.4% — 합쳐서 **시나리오의 31.1%** 가 오염 구간을 쓴다.

     원호로 연장했다 잘라내는 보정도 해봤지만 절반만 회복하고(0.0067 vs 내부 0.0134)
     비용이 2배가 된다. FITPACK 은 그 문제가 애초에 없다.

  ③ tol = 0.02 m
     FITPACK 의 s 는 '잔차 제곱합의 상한'이라 점당 허용 편차 tol 로 환산해 s = n*tol^2
     로 준다. tol 은 **지도 정점에서 얼마나 벗어나도 되는가** [m] 다.

        tol    정점편차 평균/p95/max      |dkappa/ds| p95/p99/max
        0.00   0.000  0.001  0.007       0.0278  0.1117  13.290   <- 보간. 곡률이 튄다
        0.01   0.006  0.020  0.055       0.0160  0.0584   2.375
        0.02   0.012  0.037  0.116       0.0139  0.0462   0.736   <- 채택
        0.05   0.027  0.086  0.283       0.0115  0.0378   0.517
        0.20   0.087  0.304  0.758       0.0092  0.0295   0.804

     0.01 -> 0.02 에서 |dkappa/ds| max 가 2.375 -> 0.736 로 무너진다. 그 위로는 꼬리가
     거의 안 줄고 정점을 버리는 대가만 남는다. tol=0.02 의 정점편차 max 0.104 m 는
     차로 폭 3.42 m 의 3.0% 다.

  ④ 정점 가중은 균등
     간격이 0.24~20 m 로 널뛰니 호길이 가중이 나을 것 같지만, 재보면 차이가 없다
     (|kappa| max 0.713 -> 0.722, 정점편차는 오히려 나빠짐). 남는 큰 곡률은 과대표집이
     아니라 지도가 실제로 주장하는 급회전이다.

차선변경 blend 를 왜 안 넣나
----------------------------
경로가 이미 succ-only 다 (lane_frame.build_routes). 차선변경을 기하에 넣으면 지도에 없는
전이 지점을 골라야 하고, 그 전이 곡률(0.026~0.071 1/m)이 '도로의 휘어짐'으로 둔갑한다.
차선변경은 잔차 (d, theta) 가 표현한다. **이 파일도 같은 원칙을 지킨다 — 경로 폴리라인
하나를 매끄럽게 만들 뿐, 경로에 없는 기하를 만들지 않는다.**

사용:
    from lane_spline import fit_route, kappa_at
    fit = fit_route(graph, route)          # build_routes 가 돌려준 route 하나
    fit["kappa"]                           # (N,) 호길이 등간격 kappa_lane [1/m]
    kappa_at(fit, s, d)                    # 임의 s 에서 조회 (횡오프셋 위상 보정 포함)

측정 재현:
    python src/lane_spline.py --task compare   --limit 120   # 곡선 종류 비교
    python src/lane_spline.py --task kappa     --limit 300   # 적합 전후 분포
    python src/lane_spline.py --task rebase    --limit 300   # d/theta 라벨 재기준화 크기
    python src/lane_spline.py --task cost      --limit 200   # 시나리오당 비용, 20만 추정
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.interpolate import splev, splprep

from lane_frame import STEP

# 스플라인 차수. 3차면 C2 -> 곡률 연속. 5차는 |dkappa/ds| max 가 2배로 나빠진다.
SPLINE_K = 3
# 정점당 허용 편차 [m]. FITPACK 의 s = n * tol^2 로 환산해 넘긴다.
# 0.01 -> 0.02 에서 |dkappa/ds| max 가 2.375 -> 0.736 로 무너지는 무릎이다.
FIT_TOL_M = 0.02
# 호길이 적분용 조밀 샘플 간격 [m]. 현으로 호를 근사하는 상대오차가 kappa^2 h^2 / 24 라
# 최악 kappa=0.7 에서도 h=0.1 이면 2e-5 로 무시할 수 있다.
ARC_STEP = 0.1
# 적합에 필요한 최소 정점 수 / 최소 길이 [m]. 못 넘으면 유한차분 fallback 으로 간다.
MIN_VERTS = SPLINE_K + 1
MIN_LEN_M = 2.0


# --------------------------------------------------------------------- 경로 -> 정점열
def dedup(polys: Sequence[np.ndarray]) -> np.ndarray:
    """중심선들을 이어붙이고 중복점을 없앤다.

    successor 이음매는 실측 gap 이 27,915쌍에서 max 0.0000 m 라 앞 차로의 끝점과 뒤
    차로의 시작점이 정확히 겹친다. 안 지우면 매개변수 u 에 길이 0 구간이 생겨 splprep 이
    특이해진다.
    """
    out = [np.asarray(polys[0], float)]
    for q in polys[1:]:
        q = np.asarray(q, float)
        out.append(q[1:] if np.linalg.norm(q[0] - out[-1][-1]) < 1e-3 else q)
    P = np.concatenate(out)
    return P[np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])]


def route_vertices(graph, route) -> np.ndarray:
    """경로의 **원본 정점열**. 적합은 반드시 여기에 한다.

    densify 된 route["pts"] 에 적합하면 안 된다 — 그 점들은 정점 사이를 직선보간해 만든
    것이라 '여기는 직선이다'라는 가짜 증거를 간격에 비례한 **개수만큼** 집어넣는다.
    간격 0.24 m 구간과 20 m 구간의 가중치가 80배 벌어지는 것도 같은 문제다.
    """
    return dedup([np.asarray(graph.lanes[l].centerline, float)[:, :2] for l in route["lanes"]])


# --------------------------------------------------------------------- 적합
def _resample_by_arclength(tck, u0: float, u1: float, step: float) -> Optional[tuple]:
    """스플라인을 **호길이 등간격**으로 뽑는다. (pts, tan, s, kappa, u_of_s, S_of_u)

    두 번에 나눠 한다. ① 조밀한 u 격자에서 호길이 S(u) 를 적분하고 ② 원하는 s 격자에
    해당하는 u 를 역보간해 **그 u 에서 스플라인을 다시 평가**한다. ②에서 결과값을
    보간하지 않고 다시 평가하는 이유는, 곡률처럼 비선형인 양을 선형보간하면 봉우리가
    깎이기 때문이다.
    """
    n = max(64, int((u1 - u0) / ARC_STEP) + 1)
    ud = np.linspace(u0, u1, n)
    x, y = splev(ud, tck)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    L = float(S[-1])
    if not np.isfinite(L) or L < 2 * step:
        return None
    s = np.append(np.arange(0.0, L, step), L)
    ui = np.interp(s, S, ud)
    x, y = splev(ui, tck)
    dx, dy = splev(ui, tck, der=1)
    ddx, ddy = splev(ui, tck, der=2)
    sp = np.hypot(dx, dy)
    sp[sp < 1e-12] = 1e-12
    # 곡률은 매개변수 불변량이라 u 가 호길이가 아니어도 이 식이 그대로 맞다.
    kappa = (dx * ddy - dy * ddx) / sp ** 3
    return (np.stack([x, y], 1), np.stack([dx / sp, dy / sp], 1), s, kappa, ud, S)


def fit_polyline(P: np.ndarray, tol: float = FIT_TOL_M, k: int = SPLINE_K,
                 step: float = STEP) -> Optional[dict]:
    """정점열 (M,2) 에 평활 스플라인을 적합하고 호길이 등간격으로 뽑는다.

    반환 dict 는 build_routes 의 route 와 같은 키(pts/tan/s)를 갖는 **상위집합**이라
    lane_frame 의 to_frame / from_frame / project 에 그대로 넣을 수 있다.
    """
    P = np.asarray(P, float)
    if len(P) < max(MIN_VERTS, k + 1):
        return None
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if u[-1] < MIN_LEN_M:
        return None
    try:
        tck, _ = splprep([P[:, 0], P[:, 1]], u=u, k=k, s=len(P) * tol ** 2)
    except Exception:
        return None
    out = _resample_by_arclength(tck, u[0], u[-1], step)
    if out is None:
        return None
    pts, tan, s, kappa, ud, S = out
    if not np.isfinite(kappa).all():
        return None
    return {"pts": pts, "tan": tan, "s": s, "kappa": kappa, "len": float(s[-1]),
            "s_of_vertex": np.interp(u, ud, S), "n_ctrl": len(tck[0]) - k - 1,
            "fallback": False}


def fit_route(graph, route, tol: float = FIT_TOL_M, k: int = SPLINE_K,
              step: float = STEP) -> dict:
    """build_routes 가 만든 경로 하나에 스플라인을 적합한다.

    적합이 안 되면 None 이 아니라 **원본 경로에 유한차분 곡률을 얹어** 돌려준다.
    경로를 통째로 잃으면 L3 후보 집합이 줄어 recall 천장이 내려가는데, 그건 곡률이
    거친 것보다 나쁘다. 판별은 fit["fallback"] 로 한다.
    """
    V = route_vertices(graph, route)
    fit = fit_polyline(V, tol=tol, k=k, step=step)
    if fit is None:
        return _fallback(route)
    fit["lanes"] = route["lanes"]
    # 차로 경계의 호길이. rule_band 는 **원본** 차로 길이를 누적해 구간을 잡는데 적합
    # 곡선은 길이가 미세하게 달라 뒤로 갈수록 어긋난다. 정점 인덱스로 직접 잡아 넘겨둔다.
    nv, bnd, sv = 0, [], fit["s_of_vertex"]
    for lid in route["lanes"]:
        nv += len(np.asarray(graph.lanes[lid].centerline)) - (1 if bnd else 0)
        bnd.append(float(sv[min(nv - 1, len(sv) - 1)]))
    fit["lane_s"] = np.array(bnd)
    return fit


def _fallback(route) -> dict:
    """적합이 안 되는 경로(정점 4개 미만, 길이 2 m 미만)의 최소 동작.

    곡률은 접선각을 호길이로 미분해 만든다 — 원시 유한차분과 같은 잡음이 남지만,
    이런 경로는 어차피 2 m 미만이라 6초 지평에서 실질적으로 안 쓰인다.
    """
    pts, tan, s = route["pts"], route["tan"], route["s"]
    kappa = np.zeros(len(s))
    if len(s) >= 3:
        kappa = np.gradient(np.unwrap(np.arctan2(tan[:, 1], tan[:, 0])), s)
    return {"pts": pts, "tan": tan, "s": s, "kappa": kappa, "len": float(s[-1]),
            "lanes": route["lanes"], "fallback": True, "n_ctrl": 0,
            "lane_s": np.array([float(s[-1])]), "s_of_vertex": np.array([0.0, float(s[-1])])}


def fit_routes(graph, routes: List[dict], **kw) -> List[dict]:
    """시나리오의 경로 전부. 같은 차로 사슬은 build_routes 가 이미 중복 제거해 준다."""
    return [fit_route(graph, r, **kw) for r in routes]


# --------------------------------------------------------------------- 조회
def kappa_at(fit: dict, s, d=None) -> np.ndarray:
    """주행거리 s [m] 에서 kappa_lane 을 읽는다. d 를 주면 호길이 위상을 보정한다.

    왜 보정이 필요한가
    ------------------
    적분기가 주는 s 는 **차가 실제 달린 거리**이고 테이블은 **중심선 호길이**로 색인된다.
    차가 중심선에서 d 만큼 벗어난 채 곡률 kappa 인 커브를 돌면 평행곡선의 호길이 미분이

        ds_off = (1 - kappa*d) * ds_ref

    라 두 거리가 어긋난다. (d 는 lane_frame.to_frame 과 같은 좌(+) 부호. 좌회전
    kappa>0 에서 안쪽 d>0 이 더 짧다.) 따라서 달린 거리에 대응하는 중심선 거리는

        s_ref = s_off / (1 - kappa*d)

    **docs/action_design_explained.md 의 `s_ref = s * (1 - kappa*d)` 는 방향이 반대다.**
    그 식은 중심선 거리에서 실주행 거리를 얻는 식이라, 조회에 쓰려면 곱이 아니라 나눗셈이다.
    반경 25 m 원의 안쪽 2 m 로 사분원을 돌면 36.128 m 를 달리는데 대응 중심선 거리는
    39.270 m 다. 36.128/(1-0.04*2)=39.270 이고 36.128*(1-0.04*2)=33.238 이다.

    안 고치면 kappa=0.02, d=1.5 m, s=60 m 에서 s_ref 가 1.8 m 어긋나고, 실측 기준
    heading 민감도(s 가 1 m 어긋나면 p99 16 deg)로 회전 구간에서 방향이 틀어진다.

    남는 2차 항 — 차가 실제로 그리는 곡률은 kappa/(1 - kappa*d) 라 중심선 곡률과 다르다
    (kappa=0.05, d=1.5 에서 8%). 그건 잔차 delta 가 흡수한다. 여기서 같이 보정하면
    delta 의 정의가 d 에 의존하게 되어 학습 대상이 흔들린다.
    """
    S, K = fit["s"], fit["kappa"]
    s = np.asarray(s, float)
    if d is not None:
        k0 = np.interp(np.clip(s, S[0], S[-1]), S, K)   # 보정에 쓸 곡률은 무보정 조회로 족하다
        den = 1.0 - k0 * np.asarray(d, float)
        # d 가 회전중심을 넘어가면 평행곡선이 뒤집힌다. 실주행에서 |kappa*d| < 0.2 지만
        # (kappa p99 0.154, |d| <= 3.6 -> 0.55 이므로 교차로에서 닿을 수 있다) 방어한다.
        den = np.where(np.abs(den) < 0.2, np.where(den < 0, -0.2, 0.2), den)
        s = s / den
    return np.interp(np.clip(s, S[0], S[-1]), S, K)


def dkappa_ds(fit: dict) -> np.ndarray:
    """곡률 변화율 [1/m^2]. 조향 속도에 대응하므로 실현가능성 점검에 쓴다."""
    return np.gradient(fit["kappa"], fit["s"])


# ===================================================================== 측정 (CLI)
import argparse          # noqa: E402  — 아래는 측정용. 라이브러리로 import 할 때는 안 쓰인다.
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from multiprocessing import Pool   # noqa: E402
from pathlib import Path           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/data/argoverse2/motion_forecasting")


def _routes_of(d: Path):
    """시나리오 하나에서 (graph, routes, 미래궤적, 시작 lane) 을 만든다. 측정 공통 전처리."""
    from av2.datasets.motion_forecasting import scenario_serialization
    from lane_frame import build_routes
    from lane_graph import LaneGraph, REACH_MARGIN_M
    from heading_decomp import OBS_LEN, DT

    g = LaneGraph.from_json_dict(json.loads((d / f"log_map_archive_{d.name}.json").read_text()))
    sc = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
    f = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
    st = sorted(f.object_states, key=lambda x: x.timestep)
    pos = np.array([x.position for x in st], float)
    if len(pos) < OBS_LEN + 1:
        return None
    v = pos[OBS_LEN - 1] - pos[OBS_LEN - 2]
    if np.linalg.norm(v) < 1e-6:
        return None
    sp = float(np.linalg.norm(v)) / DT
    starts = g.candidate_lanes(pos[OBS_LEN - 1], v / np.linalg.norm(v), path=pos[:OBS_LEN])
    if not starts:
        return None
    reach = g.reachable(starts, max(20.0, sp * 6.0) + REACH_MARGIN_M)
    routes = build_routes(g, starts, reach, v0=sp)
    return (g, routes, pos) if routes else None


def _fd_kappa(pts: np.ndarray) -> np.ndarray:
    """densify 된 점열의 유한차분 곡률 — 스플라인을 안 썼을 때 나오는 것."""
    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    den[den < 1e-12] = 1e-12
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / den


def _task_kappa(d: Path):
    """적합 전후의 |kappa| · |dkappa/ds| 분포."""
    r = _routes_of(d)
    if r is None:
        return None
    g, routes, _ = r
    ka, kb, da, db, nfb = [], [], [], [], 0
    for rt in routes:
        kb.append(np.abs(_fd_kappa(rt["pts"])))
        db.append(np.abs(np.diff(_fd_kappa(rt["pts"]))) / STEP)
        f = fit_route(g, rt)
        nfb += int(f["fallback"])
        ka.append(np.abs(f["kappa"]))
        da.append(np.abs(np.diff(f["kappa"])) / STEP)
    return (np.concatenate(kb), np.concatenate(ka),
            np.concatenate(db), np.concatenate(da), len(routes), nfb)


def _task_rebase(d: Path):
    """기준을 폴리라인 -> 적합곡선으로 바꿀 때 d / theta 라벨이 얼마나 움직이나.

    theta = h - k 이고 h 는 궤적에서만 나오므로 **theta 의 변화는 기준 접선각 k 의 변화**와
    부호만 반대다 (dtheta = -dk). 그래서 h 를 만들 필요가 없다.
    """
    from lane_frame import to_frame
    from heading_decomp import OBS_LEN, build_heading, wrap
    r = _routes_of(d)
    if r is None:
        return None
    g, routes, pos = r
    fut = pos[OBS_LEN - 1:]
    best, bm = None, np.inf
    for rt in routes:                              # 오라클 — 정답에 가장 잘 맞는 경로
        _, _, md = to_frame(fut, rt)
        if md < bm:
            best, bm = rt, md
    f = fit_route(g, best)
    if f["fallback"]:
        return None
    s0, d0, _ = to_frame(fut, best)
    s1, d1, _ = to_frame(fut, f)

    def ang(s, rt):
        return np.arctan2(np.interp(s, rt["s"], rt["tan"][:, 1]),
                          np.interp(s, rt["s"], rt["tan"][:, 0]))
    a0, a1 = ang(s0, best), ang(s1, f)
    dk = np.degrees(np.abs(wrap(a1 - a0)))
    # theta 자체와 그 **스텝간 떨림**. 재기준화가 잡음 제거인지 단순 이동인지는 이걸로 갈린다.
    h, src = build_heading(pos[OBS_LEN - 1:])
    # src==1 = 위치차분으로 정한 h. 저속 보조(src 2/3)는 h 자체가 신뢰구간 밖이라
    # theta 를 재는 데 못 쓴다 (docs/v4_heading_representation.md 의 기준속도 1.0 m/s).
    ok = np.isfinite(h) & (src == 1)
    th0, th1 = wrap(h - a0), wrap(h - a1)
    j0 = np.degrees(np.abs(wrap(np.diff(th0))))[ok[:-1] & ok[1:]]
    j1 = np.degrees(np.abs(wrap(np.diff(th1))))[ok[:-1] & ok[1:]]
    return (np.abs(d1 - d0), dk, np.abs(s1 - s0), np.abs(d0), np.abs(d1),
            np.degrees(np.abs(th0[ok])), np.degrees(np.abs(th1[ok])), j0, j1)


def _task_cost(d: Path):
    """시나리오당 비용 — 지도 읽기 / 경로 열거 / 스플라인 적합을 나눠 잰다."""
    from av2.datasets.motion_forecasting import scenario_serialization
    from lane_frame import build_routes
    from lane_graph import LaneGraph, REACH_MARGIN_M
    from heading_decomp import OBS_LEN, DT
    t0 = time.perf_counter()
    g = LaneGraph.from_json_dict(json.loads((d / f"log_map_archive_{d.name}.json").read_text()))
    sc = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
    t1 = time.perf_counter()
    f = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
    st = sorted(f.object_states, key=lambda x: x.timestep)
    pos = np.array([x.position for x in st], float)
    if len(pos) < OBS_LEN + 1:
        return None
    v = pos[OBS_LEN - 1] - pos[OBS_LEN - 2]
    if np.linalg.norm(v) < 1e-6:
        return None
    sp = float(np.linalg.norm(v)) / DT
    starts = g.candidate_lanes(pos[OBS_LEN - 1], v / np.linalg.norm(v), path=pos[:OBS_LEN])
    if not starts:
        return None
    reach = g.reachable(starts, max(20.0, sp * 6.0) + REACH_MARGIN_M)
    routes = build_routes(g, starts, reach, v0=sp)
    t2 = time.perf_counter()
    if not routes:
        return None
    fits = fit_routes(g, routes)
    t3 = time.perf_counter()
    npt = sum(len(x["kappa"]) for x in fits)
    return ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, len(routes), npt)


def _q(x, name, unit="", pcts=(50, 95, 99, 99.9)):
    x = np.asarray(x)
    cols = "  ".join(f"p{p:<4g} {np.percentile(x, p):9.4f}" for p in pcts)
    print(f"  {name:22s} n={len(x):>9,}  평균 {x.mean():8.4f}  {cols}  max {x.max():10.3f} {unit}")


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["kappa", "rebase", "cost", "compare"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    # 이 머신은 GPU 학습이 같이 도는 일이 잦다. load average 가 코어 수에 근접하면
    # workers 를 늘려도 처리량이 안 오르고 학습만 느려진다. 기본을 4 로 둔다.
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit]
    fn = {"kappa": _task_kappa, "rebase": _task_rebase, "cost": _task_cost}.get(a.task)
    if a.task == "compare":
        _compare(dirs)
        return
    t_wall = time.perf_counter()
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(fn, dirs, chunksize=4) if r is not None]
    t_wall = time.perf_counter() - t_wall
    print(f"\n=== [{a.split}] task={a.task}  시나리오 {len(res):,} / {len(dirs):,}  "
          f"workers={a.workers}  벽시계 {t_wall:.1f}s ===")

    if a.task == "kappa":
        kb = np.concatenate([r[0] for r in res]); ka = np.concatenate([r[1] for r in res])
        db = np.concatenate([r[2] for r in res]); da = np.concatenate([r[3] for r in res])
        nrt = sum(r[4] for r in res); nfb = sum(r[5] for r in res)
        print(f"경로 {nrt:,}개 (fallback {nfb}개 = {100*nfb/max(nrt,1):.3f}%)\n")
        print("[적합 전 — 원시 중심선 유한차분]")
        _q(kb, "|kappa|", "1/m"); _q(db, "|dkappa/ds|", "1/m^2")
        print("[적합 후 — 3차 평활 스플라인 tol=0.02]")
        _q(ka, "|kappa|", "1/m"); _q(da, "|dkappa/ds|", "1/m^2")
        print("\n반경으로 본 물리적 타당성 (승용차 최소회전반경 ~5 m):")
        print(f"  {'|kappa| 임계':>16s} {'반경':>7s} {'적합 전':>10s} {'적합 후':>10s}")
        for t in (0.05, 0.1, 0.2, 0.5, 1.0):
            print(f"  {'> '+format(t,'.2f'):>16s} {'< '+format(1/t,'.1f')+' m':>7s} "
                  f"{100*np.mean(kb>t):9.4f}% {100*np.mean(ka>t):9.4f}%")

    elif a.task == "rebase":
        dd = np.concatenate([r[0] for r in res]); dk = np.concatenate([r[1] for r in res])
        ds = np.concatenate([r[2] for r in res])
        d0 = np.concatenate([r[3] for r in res]); d1 = np.concatenate([r[4] for r in res])
        print("기준을 '직선보간 폴리라인' -> '적합 스플라인' 으로 바꿀 때 라벨의 이동량\n")
        _q(dd, "|Δd| 횡오프셋", "m"); _q(dk, "|Δθ| 잔차각", "deg"); _q(ds, "|Δs| 호길이", "m")
        print("\n참고 — 라벨 자체의 크기 (이동량이 이것에 비해 얼마나 작은가):")
        _q(d0, "|d| 적합 전", "m"); _q(d1, "|d| 적합 후", "m")
        print(f"\n  |Δd| <= 0.05 m 인 step {100*np.mean(dd<=0.05):.1f}%   "
              f"|Δθ| <= 1 deg 인 step {100*np.mean(dk<=1.0):.1f}%")
        t0 = np.concatenate([r[5] for r in res]); t1 = np.concatenate([r[6] for r in res])
        j0 = np.concatenate([r[7] for r in res]); j1 = np.concatenate([r[8] for r in res])
        print("\n재기준화가 잡음 제거인가 단순 이동인가 — theta 와 그 스텝간 떨림")
        print("  (h 를 위치차분으로 정한 step 만. 저속 보조 step 은 h 자체가 못 믿을 값이다)")
        _q(t0, "|θ| 적합 전", "deg", (50, 90, 95, 99)); _q(t1, "|θ| 적합 후", "deg", (50, 90, 95, 99))
        _q(j0, "|Δθ/step| 적합 전", "deg", (50, 90, 95, 99))
        _q(j1, "|Δθ/step| 적합 후", "deg", (50, 90, 95, 99))

    elif a.task == "cost":
        A = np.array([r[:3] for r in res]); nrt = np.array([r[3] for r in res])
        npt = np.array([r[4] for r in res])
        print(f"경로 {nrt.mean():.1f}개/시나리오   테이블 {npt.mean():,.0f}점/시나리오\n")
        for i, nm in enumerate(("지도 읽기 + parquet", "경로 열거 (build_routes)",
                                "스플라인 적합 (이 파일)")):
            print(f"  {nm:26s} 평균 {A[:,i].mean():7.2f} ms  중앙 {np.median(A[:,i]):7.2f}  "
                  f"p95 {np.percentile(A[:,i],95):7.2f}  비중 {100*A[:,i].mean()/A.sum(1).mean():5.1f}%")
        tot = A.sum(1).mean()
        print(f"  {'합계':26s} 평균 {tot:7.2f} ms")
        print(f"\ntrain {199908:,} 시나리오 추정 (측정 벽시계 {t_wall:.1f}s / {len(res)} 시나리오):")
        per = t_wall / len(res)
        for w in (4, 8, 16, 32):
            scale = a.workers / w
            print(f"  workers={w:<3d}  {199908*per*scale/3600:6.2f} 시간"
                  + ("   <- 이번 측정과 같은 조건" if w == a.workers else ""))
        print(f"  ※ 위는 이번 측정 조건(load average 높음)의 선형 외삽이다. "
              f"스플라인 적합만 떼면 {199908*A[:,2].mean()/1000/a.workers/3600:.2f} 시간 "
              f"(workers={a.workers}).")


def _compare(dirs):
    """곡선 종류 비교 — 모듈 docstring 의 표를 다시 뽑는다."""
    from scipy.interpolate import make_smoothing_spline
    from scipy.spatial import cKDTree
    from lane_frame import build_routes

    def dev(F, Q):
        """점-선분 거리. 0.25 m 샘플 최근접으로 재면 평균 0.0625 m 의 가짜 바닥이 생긴다."""
        _, j = cKDTree(F).query(Q)
        idx = np.clip(j[:, None] + np.arange(-2, 2)[None, :], 0, len(F) - 2)
        A, B = F[idx], F[idx + 1]
        AB = B - A
        L2 = (AB ** 2).sum(-1); L2[L2 < 1e-12] = 1e-12
        t = np.clip(((Q[:, None, :] - A) * AB).sum(-1) / L2, 0, 1)
        return np.sqrt(((Q[:, None, :] - (A + t[..., None] * AB)) ** 2).sum(-1).min(1))

    def pen(P, lam):                                  # 벌점평활 — 경계에서 f''=0
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        if len(P) < 5 or u[-1] < 2:
            return None
        bx = make_smoothing_spline(u, P[:, 0], lam=lam)
        by = make_smoothing_spline(u, P[:, 1], lam=lam)
        n = max(64, int(u[-1] / ARC_STEP) + 1); ud = np.linspace(0, u[-1], n)
        S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(bx(ud)), np.diff(by(ud))))])
        s = np.append(np.arange(0, S[-1], STEP), S[-1]); ui = np.interp(s, S, ud)
        dx, dy = bx(ui, 1), by(ui, 1); ddx, ddy = bx(ui, 2), by(ui, 2)
        sp = np.hypot(dx, dy); sp[sp < 1e-12] = 1e-12
        return {"pts": np.stack([bx(ui), by(ui)], 1), "s": s,
                "kappa": (dx * ddy - dy * ddx) / sp ** 3}

    cfg = ([(f"보간 k={k}", lambda P, k=k: fit_polyline(P, tol=0.0, k=k)) for k in (3, 5)]
           + [(f"FITPACK k=3 tol={t:.2f}", lambda P, t=t: fit_polyline(P, tol=t))
              for t in (0.01, 0.02, 0.05, 0.20)]
           + [(f"FITPACK k=5 tol={t:.2f}", lambda P, t=t: fit_polyline(P, tol=t, k=5))
              for t in (0.02, 0.05)]
           + [(f"벌점평활 lam={l:g}", lambda P, l=l: pen(P, l)) for l in (0.1, 1.0, 10.0)])

    samples = []
    for d in dirs:
        r = _routes_of(d)
        if r is None:
            continue
        g, routes, _ = r
        samples.extend((route_vertices(g, rt), rt["pts"]) for rt in routes)
    print(f"\n=== 곡선 종류 비교 (시나리오 {len(dirs)}, 경로 {len(samples):,}) ===")
    print(f"{'설정':20s} {'실패':>4s} {'ms':>5s} | {'정점편차 평균/p95/max':>21s} | "
          f"{'폴리라인편차 평균/p95/max':>21s} | {'|k| p99/max':>14s} | "
          f"{'|dk/ds| p95/p99/max':>21s} | 끝에서 거리별 평균 |kappa|")
    bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    for name, f in cfg:
        dv, dp, kk, dk, fail = [], [], [], [], 0
        acc = {b: [] for b in bins}
        t0 = time.perf_counter()
        for V, P in samples:
            o = f(V)
            if o is None or not np.isfinite(o["kappa"]).all():
                fail += 1
                continue
            F, S, K = o["pts"], o["s"], o["kappa"]
            dv.append(dev(F, V)); dp.append(dev(F, P[::4]))
            kk.append(np.abs(K)); dk.append(np.abs(np.diff(K)) / STEP)
            e = np.minimum(S, S[-1] - S)
            for b in bins:
                m = (e >= b[0]) & (e < b[1])
                if m.any():
                    acc[b].append(np.abs(K[m]))
        ms = (time.perf_counter() - t0) / len(samples) * 1e3
        if not kk:
            print(f"{name:20s} {fail:4d}  전부 실패")
            continue
        dv = np.concatenate(dv); dp = np.concatenate(dp)
        kk = np.concatenate(kk); dk = np.concatenate(dk)
        prof = " ".join(f"{np.concatenate(acc[b]).mean():.5f}" if acc[b] else "  -  " for b in bins)
        print(f"{name:20s} {fail:4d} {ms:5.2f} | {dv.mean():6.3f} {np.percentile(dv,95):6.3f} "
              f"{dv.max():7.3f} | {dp.mean():6.3f} {np.percentile(dp,95):6.3f} {dp.max():7.3f} | "
              f"{np.percentile(kk,99):6.4f} {kk.max():6.3f} | {np.percentile(dk,95):6.4f} "
              f"{np.percentile(dk,99):6.4f} {dk.max():6.3f} | {prof}")
    print("  (끝거리 구간: 0~1  1~2  2~5  5~10  10~20  20+ m — 벌점평활만 앞 구간이 무너진다)")


if __name__ == "__main__":
    _main()
