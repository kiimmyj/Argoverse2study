#!/usr/bin/env python
"""cleanse_census.py - '맵 이탈'과 'heading 오기입'이 **몇 건인지** 전수로 센다.

왜
--
데이터를 버릴지 보정할지는 양이 정한다. 0.1% 면 버리면 되고 15% 면 버릴 수 없다.
지금까지 나온 수치(경로 없음 3.78%, 커버리지 실패 14.1%, 뒤집힘 8.5%)는 표본 2~3천 개에서
나온 것이고 모집단·정의가 문서마다 달랐다. 여기서 정의를 한 곳에 못박고 train+val 전수로 센다.

무엇을 재나 (정의는 아래 상수와 one() 주석에 있다)
--------------------------------------------------
A. 맵 이탈
  A1  후보 경로 0개                     dataset_lane 과 같은 절차로 열거해 build_routes 가 빈 경우
  A2  중심선 5 m 밖                     focal 110스텝의 최근접 중심선 거리 > 5 m — 주행가능영역 안/밖으로 가름
  A3  정답이 후보 밴드 밖               ≤6 슬롯 중 어느 것도 정답 6초를 규칙 밴드로 못 덮음 (= 커버리지 실패)
  A4  A3 의 원인                        경로없음 / 나란한 후보 없음 / U턴 / 후진 / 지도 밖 / 밴드만 초과

B. heading 오기입
  B1  AV2 heading 이 이동방향과 180°    트랙 단위, 총 이동거리 구간별, focal / 주변 / 차량여부로 가름
  B2  우리 전처리(align_ref)가 뒤집음   관측 50스텝 기준. 게이트판(align_ref_gated)과 나란히 센다
  B3  ±pi 점프 · |dh| > 7.3°/step      AV2 필드와 우리 h 를 같은 기준으로
  B4  방향이 한 번도 정의되지 않는 트랙  위치차분이 v_min(1.0 m/s)을 한 번도 못 넘음
  B5  속도 필드 vs 위치차분 속력 불일치  창 가장자리 램프 포함

읽기 전용이다. 원본도 캐시도 건드리지 않는다.

  python src/cleanse_census.py --split val  --workers 16 --out viz/v4/cleanse/data
  python src/cleanse_census.py --split train --workers 16 --limit 50000
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
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import lane_frame as lf                                            # noqa: E402
from lane_graph import LaneGraph, REACH_MARGIN_M                   # noqa: E402
from heading_decomp import (build_heading, guard, align_ref, align_ref_gated,  # noqa: E402
                            lane_field, wrap, V_MIN, JUMP_DEG)

ROOT = Path("/data/argoverse2/motion_forecasting")
REPO = SRC.parent

# ---------------------------------------------------------------- 상수 = 판정 기준
OBS, FUT, T_ALL = 50, 60, 110
DT = 0.1
PRED_SEC = 6.0
LAT_FAR_M = 5.0          # 중심선에서 이보다 멀면 '차로 없음' (lane_graph.CAND_RADIUS_M · heading_decomp.MAX_LAT_M 과 같은 값)
NEAR_M = 150.0           # 차로 밭을 만들 때 focal 예측시작점에서 이 반경 안의 차로만 (6초 최대 주행 ~120 m 를 덮는다)
DEDUP_M = 2.5            # dataset_lane.DEDUP_M — 지평에서 이보다 가까운 두 경로는 같은 분기
N_MODES = 6              # dataset_lane.N_MODES
ALIGN_DEG = 30.0         # viz_v4_common.ROUTE_ALIGN_DEG — |theta_정답| 최대가 이 미만이면 '나란한 경로'
MOVE_V = 1.0             # [m/s] 진행방향 기반 양은 이 속력 이상 스텝만 (viz_v4_common.MOVE_V)
UTURN_DEG = 135.0        # 6초 순 진행방향 변화가 이보다 크면 U턴
REVERSE_MOVE_M = 2.0     # 후진 판정에 필요한 최소 6초 이동거리
DH_LIMIT_DEG = 7.3       # 정상값 상한 p99.99 (heading_decomp.guard 주석 · viz_v4_common.LABEL_DTHETA_DEG)
PI_JUMP_DEG = 90.0       # 이보다 크면 사실상 ±pi 점프
DV_BIG = 1.0             # [m/s] 속도 필드와 위치차분 속력 차가 이보다 크면 '크게 어긋남'
DV_HUGE = 2.0
PATH_BINS = np.array([0.0, 2.0, 5.0, 20.0, 50.0, 1e9])   # heading_decomp.PATH_BINS 와 같다
PATH_LBL = ["0–2 m", "2–5 m", "5–20 m", "20–50 m", "50 m+"]
VEH_TYPES = {"vehicle", "bus", "truck", "truck_cab", "box_truck", "large_vehicle",
             "motorcyclist", "articulated_bus", "school_bus"}      # heading_decomp.run_flip 의 VEH
AGENT_TYPES = {"vehicle", "bus", "motorcyclist", "cyclist"}        # prep_agents_heading.AGENT_TYPES
# focal 의 종류 — v4 의 출력 공간(차로 경로 위의 s, d)이 애초에 성립하는 대상인지가 여기서 갈린다
TYPE_CODE = {"vehicle": 1, "pedestrian": 2, "cyclist": 3, "motorcyclist": 4, "bus": 5,
             "static": 6, "background": 7, "construction": 8, "riderless_bicycle": 9,
             "unknown": 10}
TYPE_NAME = {v: k for k, v in TYPE_CODE.items()} | {0: "기타"}
CAUSE = ["덮음", "경로없음", "나란한후보없음", "U턴", "후진", "지도밖", "밴드만초과"]

# 스칼라 필드 이름 -> dtype. one() 이 돌려주는 dict 의 키와 1:1 이다.
FIELDS = {
    # 기본
    "n_tracks": "i4", "n_veh": "i4", "v0": "f4", "move6": "f4", "move_obs": "f4",
    "dh6": "f4", "fwd6": "f4", "f_type": "u1", "f_gaps": "u1",
    # A1
    "n_starts": "i2", "n_starts_course": "i2", "n_routes": "i2", "n_slots": "i2",
    "route_none": "u1",
    # A2  (any = 방향 무시 최근접, dir = 진행방향과 같은 쪽 차로만)
    "far_obs": "i2", "far_fut": "i2", "far_obs_dir": "i2", "far_fut_dir": "i2",
    "far_off_da_obs": "i2", "far_off_da_fut": "i2", "off_da_obs": "i2", "off_da_fut": "i2",
    "lat_max_obs": "f4", "lat_max_fut": "f4", "lat_med": "f4",
    # A3 / A4
    "coverage": "u1", "cov_175": "u1", "cov_36": "u1",
    "n_inband": "i2", "n_aligned": "i2", "gt_meand": "f4", "gt_maxd": "f4",
    "gt_maxth": "f4", "frac_out": "f4", "n_out": "i2", "cause": "u1",
    # B — focal
    "f_path": "f4", "f_judgeable": "u1", "f_flip_av2": "u1", "f_undef": "u1",
    "f_align_flip": "u1", "f_align_flip_gated": "u1", "f_gated_judged": "u1",
    "f_align_judgeable": "u1",
    "f_ref_steps": "i2", "f_ref_anti": "i2", "f_lane_steps": "i2", "f_anti_lane": "i2",
    "f_anti_lane_av2": "i2",
    "f_dh_gt73": "i2", "f_dh_gt90": "i2", "f_dh_gt73_av2": "i2", "f_dh_gt90_av2": "i2",
    "f_dv_big": "i2", "f_dv_huge": "i2", "f_n_dv": "i2",
    "f_ramp_head": "f4", "f_ramp_tail": "f4", "f_ramp_ok": "u1",
    # B — 주변 트랙 (시나리오 합계)
    "a_tracks": "i4", "a_undef": "i4", "a_align_flip": "i4", "a_align_flip_gated": "i4",
    "a_gated_judged": "i4", "a_align_judgeable": "i4",
    # 뒤집힘 판정이 차로 방향과 맞는가 — 보조로 메운 스텝(src==2)에서만 본다
    "a_chk": "i4", "a_chk_flip": "i4", "a_chk_flip_anti": "i4", "a_chk_noflip_anti": "i4",
}
# 이동거리 구간 x (차량/그외) x (트랙수/판정가능/뒤집힘) — B1 집계용
BIN_FIELDS = [f"b_{'veh' if v else 'oth'}_{c}_{b}"
              for v in (1, 0) for c in ("n", "j", "f") for b in range(5)]
for k in BIN_FIELDS:
    FIELDS[k] = "i4"


def _at(route, s_query):
    """dataset_lane._at 와 같다 — 경로 위 호길이 s 지점의 좌표."""
    S = route["s"]
    t = float(np.clip(s_query, 0.0, S[-1]))
    return np.array([np.interp(t, S, route["pts"][:, 0]),
                     np.interp(t, S, route["pts"][:, 1])])


def load_tracks(sdir):
    """parquet 을 pyarrow 로 직접 읽어 트랙별 (위치·heading·속도·관측) 배열로 편다.

    av2 의 load_argoverse_scenario_parquet 은 object_state 를 dataclass 로 하나씩 만들어
    22~86 ms 가 든다. 같은 값을 pyarrow 로 3.6 ms 에 읽는다 (--check-loader 로 대조한다).
    """
    t = pq.read_table(sdir / f"scenario_{sdir.name}.parquet")
    col = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
    fid = str(col["focal_track_id"][0])
    tid = col["track_id"].astype(str)
    ts = col["timestep"].astype(np.int64)
    order = np.argsort(tid, kind="stable")
    tid, ts = tid[order], ts[order]
    P = np.stack([col["position_x"][order], col["position_y"][order]], axis=1)
    V = np.stack([col["velocity_x"][order], col["velocity_y"][order]], axis=1)
    H = col["heading"][order].astype(np.float64)
    OT = col["object_type"][order].astype(str)
    bnd = np.flatnonzero(np.concatenate([[True], tid[1:] != tid[:-1], [True]]))
    out = []
    for a, b in zip(bnd[:-1], bnd[1:]):
        k = ts[a:b]
        m = (k >= 0) & (k < T_ALL)
        if not m.any():
            continue
        pos = np.full((T_ALL, 2), np.nan)
        hav = np.full(T_ALL, np.nan)
        vel = np.full((T_ALL, 2), np.nan)
        obs = np.zeros(T_ALL, bool)
        kk = k[m]
        pos[kk], hav[kk], vel[kk], obs[kk] = P[a:b][m], H[a:b][m], V[a:b][m], True
        out.append({"id": tid[a], "type": OT[a], "pos": pos, "hav2": hav, "vel": vel,
                    "obs": obs, "focal": tid[a] == fid})
    return out, fid


def drivable_paths(raw):
    """log_map_archive 의 drivable_areas 폴리곤 -> matplotlib Path 목록."""
    from matplotlib.path import Path as MPath
    out = []
    for a in raw.get("drivable_areas", {}).values():
        b = np.array([[p["x"], p["y"]] for p in a["area_boundary"]], dtype=np.float64)
        if len(b) >= 3:
            out.append(MPath(np.vstack([b, b[:1]]), closed=True))
    return out


def in_drivable(paths, pts):
    """점들이 주행가능영역 폴리곤 **하나라도** 안에 있는가 (bool (N,))."""
    if not paths or len(pts) == 0:
        return np.zeros(len(pts), bool)
    hit = np.zeros(len(pts), bool)
    for p in paths:
        hit |= p.contains_points(pts)
    return hit


def track_flip(tr, v_min=V_MIN):
    """트랙 하나의 (총 이동거리, 판정가능, AV2 뒤집힘, 방향 미정의).

    heading_decomp.one_flip 과 같은 규칙이다 — 위치차분으로 정해진 스텝(src==1)이 5개
    이상일 때만 판정하고, 그 스텝에서 median|wrap(h_av2 - h_course)| > 90° 면 뒤집힘.
    """
    h, src = build_heading(tr["pos"], obs=tr["obs"], h_ref=None, v_min=v_min)
    m = tr["obs"] & (src == 1)
    o = np.flatnonzero(tr["obs"])
    path = (float(np.linalg.norm(np.diff(tr["pos"][o], axis=0), axis=1).sum())
            if len(o) > 1 else 0.0)
    if m.sum() < 5:
        return path, False, False, bool(m.sum() == 0)
    flip = bool(np.median(np.abs(wrap(tr["hav2"][m] - h[m]))) > np.pi / 2)
    return path, True, flip, False


def course_of(pos, obs, dt=DT, v_min=V_MIN):
    """build_heading 의 **첫 블록**만 그대로 — 위치차분 방향(정의 안 되면 NaN)과 구간속력.

    align_ref 는 build_heading 안에서 구멍을 메우기 **전**의 이 배열을 본다. 메운 배열을
    넘기면(정지 트랙은 전부 0.0 으로 채워진다) 뒤집힘 판정이 동전던지기가 되어 비율이 부풀려진다.
    """
    pos = np.asarray(pos, dtype=np.float64)
    T = len(pos)
    idx = np.flatnonzero(obs)
    course = np.full(T, np.nan)
    v_seg = np.zeros(T)
    for a, b in zip(idx[:-1], idx[1:]):
        d = pos[b] - pos[a]
        n = float(np.linalg.norm(d))
        v_seg[a] = n / ((b - a) * dt)
        if v_seg[a] >= v_min:
            course[a] = np.arctan2(d[1], d[0])
    if len(idx) >= 2 and np.isfinite(course[idx[-2]]):
        course[idx[-1]] = course[idx[-2]]
    if len(idx) >= 2:
        v_seg[idx[-1]] = v_seg[idx[-2]]
    return course, v_seg


def align_flags(pos, hav2, obs):
    """우리 전처리가 이 구간에서 AV2 heading 을 뒤집는가 (align_ref / align_ref_gated).

    build_heading 안에서 벌어지는 일을 그대로 다시 한다: guard -> align_ref(원시 course).
    게이트판은 heading_decomp.align_ref_gated (구간속력 >= 2 m/s · 관측 이동 >= 5 m).
    """
    course, v_seg = course_of(pos, obs)
    hr = guard(np.asarray(hav2, dtype=np.float64), JUMP_DEG)
    _, flip = align_ref(hr, course, obs)
    _, flip_g, judged = align_ref_gated(hr, course, obs, v_seg, pos=pos)
    judgeable = bool((np.isfinite(course) & np.isfinite(hr) & obs).sum() >= 3)
    return bool(flip), bool(flip_g), bool(judged), judgeable


def one(sdir, do_map=True):
    try:
        return _one(sdir, do_map)
    except Exception as e:                       # 실패는 세고 넘어간다 (사유는 --debug 로)
        return {"sid": sdir.name, "err": f"{type(e).__name__}: {e}"}


def _one(sdir, do_map=True):
    tracks, fid = load_tracks(sdir)
    r = {k: 0 for k in FIELDS}
    r["sid"] = sdir.name
    r["err"] = ""
    r["n_tracks"] = len(tracks)
    ftr = next(t for t in tracks if t["focal"])
    pos, hav2, vel, obs = ftr["pos"], ftr["hav2"], ftr["vel"], ftr["obs"]
    r["f_type"] = TYPE_CODE.get(ftr["type"], 0)
    r["f_gaps"] = int(not obs.all())        # focal 이 110스텝 내내 관측되지 않는 경우
    if obs.sum() < 3:
        raise ValueError("focal 관측 스텝이 3개 미만")
    v_fld = np.linalg.norm(vel, axis=1)

    # 우리 h (감사용이라 110스텝 전체로 만든다 — 학습 입력은 관측 50스텝만 쓴다)
    h_all, src_all = build_heading(pos, obs=obs, h_ref=hav2)
    dp = np.diff(pos, axis=0)
    v_pos = np.linalg.norm(dp, axis=1) / DT
    moving = np.concatenate([v_pos >= MOVE_V, [False]])

    # ---------------- 기본량
    r["v0"] = float(v_fld[OBS - 1])
    r["move6"] = float(np.linalg.norm(np.diff(pos[OBS - 1:], axis=0), axis=1).sum())
    r["move_obs"] = float(np.linalg.norm(np.diff(pos[:OBS], axis=0), axis=1).sum())
    a_, b_ = OBS - 1, T_ALL - 1
    r["dh6"] = float(np.degrees(wrap(h_all[b_] - h_all[a_])))
    fwd = np.array([np.cos(h_all[a_]), np.sin(h_all[a_])])
    r["fwd6"] = float((pos[b_] - pos[a_]) @ fwd)
    r["n_veh"] = int(sum(1 for t in tracks if t["type"] in VEH_TYPES))

    # ---------------- B: focal
    p_, j_, fl_, ud_ = track_flip(ftr)
    r["f_path"], r["f_judgeable"], r["f_flip_av2"], r["f_undef"] = p_, j_, fl_, ud_
    af, afg, judged, jable = align_flags(pos[:OBS], hav2[:OBS], obs[:OBS])
    r["f_align_flip"], r["f_align_flip_gated"] = af, afg
    r["f_gated_judged"], r["f_align_judgeable"] = judged, jable
    for nm, hh in (("", h_all), ("_av2", guard(hav2, 180.0))):    # AV2 는 가드 없이 본다(180 = 사실상 해제)
        dh = np.degrees(np.abs(wrap(np.diff(hh))))
        dh = dh[np.isfinite(dh)]
        r[f"f_dh_gt73{nm}"] = int((dh > DH_LIMIT_DEG).sum())
        r[f"f_dh_gt90{nm}"] = int((dh > PI_JUMP_DEG).sum())
    dv = np.abs(v_pos - v_fld[:-1])
    r["f_dv_big"], r["f_dv_huge"], r["f_n_dv"] = int((dv > DV_BIG).sum()), int((dv > DV_HUGE).sum()), len(dv)
    mid = v_fld[10:100]
    ok = float(np.median(mid)) >= 3.0            # 램프는 실제로 달린 차량에서만 의미가 있다
    r["f_ramp_ok"] = ok
    if ok:
        r["f_ramp_head"] = float(v_pos[0] / max(np.median(v_pos[10:20]), 1e-6))
        r["f_ramp_tail"] = float(v_pos[-1] / max(np.median(v_pos[-20:-10]), 1e-6))

    # ---------------- B: 주변 트랙
    agents = []
    for t in tracks:
        if t["focal"]:
            continue
        p2, j2, f2, u2 = track_flip(t)
        veh = 1 if t["type"] in VEH_TYPES else 0
        b = int(np.searchsorted(PATH_BINS, p2, side="right")) - 1
        b = min(max(b, 0), 4)
        tag = "veh" if veh else "oth"
        r[f"b_{tag}_n_{b}"] += 1
        if j2:
            r[f"b_{tag}_j_{b}"] += 1
            if f2:
                r[f"b_{tag}_f_{b}"] += 1
        if t["type"] not in AGENT_TYPES or not t["obs"][OBS - 1]:
            continue                              # 모델이 실제로 입력에 넣는 후보만 (prep_agents_heading 기준)
        r["a_tracks"] += 1
        r["a_undef"] += int(u2)
        af2, afg2, jd2, jb2 = align_flags(t["pos"][:OBS], t["hav2"][:OBS], t["obs"][:OBS])
        r["a_align_flip"] += int(af2)
        r["a_align_flip_gated"] += int(afg2)
        r["a_gated_judged"] += int(jd2)
        r["a_align_judgeable"] += int(jb2)
        agents.append((float(np.linalg.norm(t["pos"][OBS - 1] - pos[OBS - 1])), af2, t))

    if not do_map:
        return r

    # ---------------- 지도
    raw = json.loads((sdir / f"log_map_archive_{sdir.name}.json").read_text())
    g = LaneGraph.from_json_dict(raw)             # centerline="api" — dataset_lane 기본값
    origin = pos[OBS - 1]
    th0 = float(hav2[OBS - 1])                    # dataset_lane 은 AV2 heading 으로 방향 필터를 건다
    d0 = np.array([np.cos(th0), np.sin(th0)])
    speed = float(v_fld[OBS - 1])
    starts = g.candidate_lanes(origin, d0, path=pos[:OBS])
    r["n_starts"] = len(starts)
    dc = np.array([np.cos(h_all[OBS - 1]), np.sin(h_all[OBS - 1])])
    r["n_starts_course"] = len(g.candidate_lanes(origin, dc, path=pos[:OBS]))
    reach = g.reachable(starts, max(20.0, speed * PRED_SEC) + REACH_MARGIN_M) if starts else set()
    rts = lf.build_routes(g, starts, reach, v0=speed) if starts else []
    r["n_routes"] = len(rts)
    r["route_none"] = int(not rts)

    # ---------------- A2: 중심선까지의 거리
    fld = lane_field(g, near_xy=origin, near_m=NEAR_M)
    dap = drivable_paths(raw)
    if fld is not None:
        P, A, _ = fld
        d2 = ((P[None, :, :] - pos[:, None, :]) ** 2).sum(axis=2)
        lat_any = np.sqrt(d2.min(axis=1))
        cs = np.cos(A[None, :] - h_all[:, None]) > 0          # heading_decomp.decompose 와 같은 대향차로 제외
        lat_dir = np.sqrt(np.where(cs, d2, np.inf).min(axis=1))
    else:
        lat_any = lat_dir = np.full(T_ALL, np.inf)
    far_any, far_dir = lat_any > LAT_FAR_M, lat_dir > LAT_FAR_M

    # ---------------- B2: 진행방향이 '앉아 있는 차로'와 반대인가
    # 방향 필터 없이 최근접 차로 점을 잡고 |wrap(h − k)| > 90° 면 대향이다.
    # (heading_decomp 의 v3 주석이 말하는 '차로 반대 비율' 과 같은 정의)
    if fld is not None:
        near = d2.argmin(axis=1)
        k_near = A[near]
        on_lane = lat_any <= LAT_FAR_M
        anti = np.abs(wrap(h_all - k_near)) > np.pi / 2
        anti_av2 = np.abs(wrap(hav2 - k_near)) > np.pi / 2
        r["f_lane_steps"] = int(on_lane.sum())
        r["f_anti_lane"] = int((on_lane & anti).sum())
        r["f_anti_lane_av2"] = int((on_lane & anti_av2).sum())
        ref_m = on_lane[:OBS] & (src_all[:OBS] == 2)      # 보조(AV2)로 메운 스텝 = 뒤집힘 판정이 실제로 효과를 내는 곳
        r["f_ref_steps"], r["f_ref_anti"] = int(ref_m.sum()), int((ref_m & anti[:OBS]).sum())

        P4, A4 = P[::4], A[::4]                          # 주변 트랙은 2 m 간격 축약 밭으로 (방향 판정에는 충분)
        agents.sort(key=lambda z: z[0])
        for _, af2, t in agents[:8]:
            h2, s2 = build_heading(t["pos"][:OBS], obs=t["obs"][:OBS], h_ref=t["hav2"][:OBS])
            m2 = t["obs"][:OBS] & (s2 == 2)
            if m2.sum() < 3:
                continue
            q = t["pos"][:OBS][m2]
            dd = ((P4[None, :, :] - q[:, None, :]) ** 2).sum(axis=2)
            jn = dd.argmin(axis=1)
            near_ok = np.sqrt(dd[np.arange(len(q)), jn]) <= LAT_FAR_M
            if near_ok.sum() < 3:
                continue
            an = (np.abs(wrap(h2[m2] - A4[jn])) > np.pi / 2)[near_ok]
            r["a_chk"] += 1
            bad = bool(an.mean() > 0.5)
            if af2:
                r["a_chk_flip"] += 1
                r["a_chk_flip_anti"] += int(bad)
            else:
                r["a_chk_noflip_anti"] += int(bad)
    inda = in_drivable(dap, pos)
    for nm, sl in (("obs", slice(0, OBS)), ("fut", slice(OBS, T_ALL))):
        r[f"far_{nm}"] = int(far_any[sl].sum())
        r[f"far_{nm}_dir"] = int(far_dir[sl].sum())
        r[f"far_off_da_{nm}"] = int((far_any[sl] & ~inda[sl]).sum())
        r[f"off_da_{nm}"] = int((~inda[sl]).sum())
        r[f"lat_max_{nm}"] = float(np.nan_to_num(lat_any[sl].max(), posinf=999.0))
    r["lat_med"] = float(np.nan_to_num(np.median(lat_any), posinf=999.0))

    # ---------------- A3: 후보 슬롯(중복 제거 ≤6) 과 정답 밴드
    if rts:
        order = np.argsort([lf.to_frame(pos[:OBS], rr)[2] for rr in rts])
        rts = [rts[i] for i in order]
        hz = max(10.0, speed * PRED_SEC)
        keep = []
        for rr in rts:
            q = np.array([_at(rr, hz * f) for f in (0.5, 1.0)])
            if all(np.linalg.norm(q - np.array([_at(o, hz * f) for f in (0.5, 1.0)]),
                                  axis=1).max() > DEDUP_M for o in keep):
                keep.append(rr)
            if len(keep) >= N_MODES:
                break
        rts = keep
    slots = []
    if rts:
        for rr in rts:
            ll, lr = lf.rule_band(g, rr)
            slots.append((rr, ll, lr))
    else:
        # dataset_lane 의 폴백(직진 1개, 밴드 ±3.6). 커버리지 정의를 viz_v4_dump 와 맞춘다.
        ln = max(30.0, speed * PRED_SEC + 20.0)
        M = 64
        arc = np.linspace(0.0, ln, M)
        xy = (np.stack([np.ones(M - 1), np.zeros(M - 1)], 1).cumsum(0) * (ln / (M - 1)))
        pts = np.vstack([np.zeros((1, 2)), xy])
        Rm = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
        pts = pts @ Rm.T + origin
        tan = np.repeat(d0[None, :], M, axis=0)
        rr = {"lanes": (), "pts": pts, "s": arc, "tan": tan}
        slots.append((rr, np.full(M, lf.WIDE_HALF_W), np.full(M, lf.WIDE_HALF_W)))
    r["n_slots"] = len(slots)

    fut = pos[OBS:]
    mv_fut = moving[OBS:]
    inband, meand, maxd, maxth, nout = [], [], [], [], []
    ib175 = ib36 = False
    for rr, ll, lr in slots:
        s_, d_, md = lf.to_frame(fut, rr)
        j = np.clip(np.searchsorted(rr["s"], s_), 0, len(rr["s"]) - 1)
        k_ = np.arctan2(rr["tan"][j, 1], rr["tan"][j, 0])
        th = wrap(h_all[OBS:] - k_)
        bad = (d_ > ll[j]) | (-d_ > lr[j])
        mx = float(np.abs(d_).max())
        ib175 |= mx <= lf.LANE_HALF_W          # 밴드를 평평하게 ±1.75 / ±3.6 로 뒀을 때의 대조군
        ib36 |= mx <= lf.WIDE_HALF_W
        inband.append(not bad.any())
        nout.append(int(bad.sum()))
        meand.append(float(np.abs(d_).mean()))
        maxd.append(float(np.abs(d_).max()))
        maxth.append(float(np.degrees(np.abs(th[mv_fut])).max()) if mv_fut.any() else 0.0)
    inband = np.array(inband); meand = np.array(meand)
    maxd = np.array(maxd); maxth = np.array(maxth); nout = np.array(nout)
    aligned = maxth < ALIGN_DEG
    rr_ = np.arange(len(slots))
    for m_ in (inband & aligned, aligned, inband, np.ones(len(slots), bool)):
        if m_.any():
            cand = rr_[m_]
            break
    gi = int(cand[np.argmin(meand[cand])])        # viz_v4_dump 와 같은 '정답 기준 경로' 규칙
    r["coverage"] = int(inband.any())
    r["cov_175"], r["cov_36"] = int(ib175), int(ib36)
    r["n_inband"], r["n_aligned"] = int(inband.sum()), int(aligned.sum())
    r["gt_meand"], r["gt_maxd"], r["gt_maxth"] = meand[gi], maxd[gi], maxth[gi]
    r["n_out"] = int(nout[gi])
    r["frac_out"] = float(nout[gi] / FUT)

    # ---------------- A4: 커버리지 실패 원인 (앞에서 걸리면 뒤는 안 본다)
    if r["coverage"]:
        c = 0
    elif r["route_none"]:
        c = 1
    elif abs(r["dh6"]) > UTURN_DEG:
        c = 3
    elif r["fwd6"] < 0 and r["move6"] > REVERSE_MOVE_M:
        c = 4
    elif far_any[OBS:].mean() > 0.5:
        c = 5
    elif not aligned.any():
        c = 2
    else:
        c = 6
    r["cause"] = c
    return r


# ---------------------------------------------------------------- 실행
def run(a):
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()]
    if a.limit:
        dirs = dirs[:a.limit]
    t0 = time.perf_counter()
    rows = []
    with Pool(a.workers) as pool:
        for i, rr in enumerate(pool.imap(partial(one, do_map=not a.no_map), dirs, chunksize=8), 1):
            rows.append(rr)
            if i % 5000 == 0 or i == len(dirs):
                el = time.perf_counter() - t0
                print(f"[{a.split}] {i:,}/{len(dirs):,}  {el:.0f}s  "
                      f"ETA {el / i * (len(dirs) - i):.0f}s  ({i / el:.1f}/s)", flush=True)
    err = [x for x in rows if x.get("err")]
    good = [x for x in rows if not x.get("err")]
    print(f"[{a.split}] 완료 {len(good):,} / 실패 {len(err):,}  {time.perf_counter() - t0:.0f}s")
    for e in err[:5]:
        print("   실패:", e["sid"], e["err"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arr = {k: np.array([x[k] for x in good], dtype=v) for k, v in FIELDS.items()}
    arr["sid"] = np.array([x["sid"] for x in good])
    np.savez_compressed(out / f"census_{a.split}.npz", **arr)
    meta = {"split": a.split, "n_dirs": len(dirs), "n_ok": len(good), "n_err": len(err),
            "errors": [{"sid": e["sid"], "err": e["err"]} for e in err[:200]],
            "limit": a.limit, "no_map": bool(a.no_map),
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "code_sha12": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
            "consts": {"LAT_FAR_M": LAT_FAR_M, "NEAR_M": NEAR_M, "DEDUP_M": DEDUP_M,
                       "ALIGN_DEG": ALIGN_DEG, "UTURN_DEG": UTURN_DEG, "MOVE_V": MOVE_V,
                       "DH_LIMIT_DEG": DH_LIMIT_DEG, "PI_JUMP_DEG": PI_JUMP_DEG,
                       "DV_BIG": DV_BIG, "V_MIN": V_MIN, "PATH_BINS": PATH_BINS.tolist()}}
    (out / f"census_{a.split}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("wrote", out / f"census_{a.split}.npz")


def check_loader(a):
    """pyarrow 판이 av2 로더와 같은 값을 주는지 대조한다 (전수 집계의 전제)."""
    from av2.datasets.motion_forecasting import scenario_serialization
    dirs = [p for p in sorted((ROOT / a.split).iterdir()) if p.is_dir()][:a.limit or 50]
    bad = 0
    for d in dirs:
        tracks, fid = load_tracks(d)
        s = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
        ok = fid == s.focal_track_id
        f = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        st = sorted(f.object_states, key=lambda x: x.timestep)
        P = np.array([x.position for x in st]); H = np.array([x.heading for x in st])
        V = np.array([x.velocity for x in st])
        mine = next(t for t in tracks if t["focal"])
        k = np.array([x.timestep for x in st])
        ok &= bool(np.array_equal(mine["pos"][k], P) and np.array_equal(mine["hav2"][k], H)
                   and np.array_equal(mine["vel"][k], V))
        ok &= len(tracks) == len(s.tracks)
        ok &= mine["type"] == str(getattr(f, "object_type", "?")).split(".")[-1].lower()
        mt = sorted(t["type"] for t in tracks)
        ok &= mt == sorted(str(getattr(t, "object_type", "?")).split(".")[-1].lower()
                           for t in s.tracks)
        bad += (not ok)
        if not ok:
            print("불일치", d.name)
    print(f"로더 대조 {len(dirs)} 시나리오 중 불일치 {bad} 건 "
          f"({'통과' if bad == 0 else '실패'})")


VAL_CACHE = Path("/data/argoverse2/cache/v4/val_e67373005962be8a_n24988")


def check_cache(a):
    """A1·A3 을 **다른 코드 경로**로 다시 센다 — 원본이 아니라 학습이 실제로 먹는 캐시로.

    다른 점이 셋이라 우연히 같아질 수 없다: (1) 경로가 원본 해상도(0.25 m)가 아니라 64점 등간격,
    (2) 밴드를 보간으로 읽음, (3) 정답을 미래 60스텝이 아니라 110스텝 전체로 투영(전진 대응의 시작점이 다름).
    이 방식은 viz_v4_dump.py 의 coverage 정의와 같다.
    """
    c = Path(a.cache)
    sid = np.array(json.loads((c / "scenario_id.json").read_text()))
    W = {k: np.load(c / f"{k}.npy", mmap_mode="r") for k in
         ("routes", "route_tan", "route_band", "route_len", "n_distinct", "route_fallback", "y")}
    n = len(sid) if not a.limit else min(a.limit, len(sid))
    cov = np.zeros(n, bool)
    fb = np.asarray(W["route_fallback"][:n]).astype(bool).ravel()
    for i in range(n):
        y = np.asarray(W["y"][i], np.float64)
        nd = max(int(W["n_distinct"][i]), 1)
        for rix in range(nd):
            ln = float(W["route_len"][i, rix])
            rd = {"pts": np.asarray(W["routes"][i, rix], np.float64),
                  "tan": np.asarray(W["route_tan"][i, rix], np.float64),
                  "s": np.linspace(0.0, ln, W["routes"].shape[2])}
            s_, d_, _ = lf.to_frame(y, rd)
            bd = np.asarray(W["route_band"][i, rix], np.float64)
            M = bd.shape[0]
            idx = np.clip(s_ / max(ln, 1e-3) * (M - 1), 0.0, M - 1 - 1e-4)
            i0 = np.floor(idx).astype(int)
            f = (idx - i0)[:, None]
            b = bd[i0] * (1 - f) + bd[np.minimum(i0 + 1, M - 1)] * f
            if np.all(d_ <= b[:, 0]) and np.all(-d_ <= b[:, 1]):
                cov[i] = True
                break
        if (i + 1) % 5000 == 0:
            print(f"  캐시 {i + 1:,}/{n:,}", flush=True)
    print(f"\n[캐시 {c.name}]  n={n:,}")
    print(f"  폴백(경로 0개)      {fb.mean() * 100:6.2f}%   ({fb.sum():,}건)")
    print(f"  커버리지 실패       {(~cov).mean() * 100:6.2f}%   ({(~cov).sum():,}건)")

    f = Path(a.out) / "census_val.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        pos = {s: j for j, s in enumerate(d["sid"])}
        j = np.array([pos.get(s, -1) for s in sid[:n]])
        ok = j >= 0
        mine_fb = d["route_none"][j[ok]].astype(bool)
        mine_cov = d["coverage"][j[ok]].astype(bool)
        print(f"\n  전수집계와 대조 (공통 {int(ok.sum()):,}건)")
        print(f"  경로 0개        캐시 {fb[ok].mean() * 100:6.2f}%  vs  census {mine_fb.mean() * 100:6.2f}%"
              f"   불일치 {(fb[ok] != mine_fb).sum():,}건")
        print(f"  커버리지 실패    캐시 {(~cov[ok]).mean() * 100:6.2f}%  vs  census "
              f"{(~mine_cov).mean() * 100:6.2f}%   불일치 {(cov[ok] != mine_cov).sum():,}건")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(REPO / "viz/v4/cleanse/data"))
    ap.add_argument("--no-map", action="store_true", help="지도 항목(A)을 건너뛴다")
    ap.add_argument("--check-loader", action="store_true", help="pyarrow 로더 대조만")
    ap.add_argument("--check-cache", action="store_true",
                    help="A1·A3 을 학습 캐시(다른 코드 경로)로 다시 세어 대조")
    ap.add_argument("--cache", default=str(VAL_CACHE))
    a = ap.parse_args()
    if a.check_loader:
        check_loader(a)
    elif a.check_cache:
        check_cache(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
