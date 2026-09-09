"""
dataset_lane.py - 차로 이동 규칙(lane_graph)을 피처로 얹은 Dataset.

dataset_map.py 는 지도를 '가까운 차선 20개의 중심선 (20,10,2)' 로만 준다. 좌표뿐이라
모델은 어느 차선이 대향차로인지, 넘을 수 있는 실선인지, 애초에 갈 수 있는 곳인지 모른다.

여기서는 같은 20개 차선에 규칙 피처 (20,10) 을 나란히 붙인다.

  [0:2] 진행방향 단위벡터 (focal 기준 회전)   ← 역주행 판단의 핵심
  [2]   focal 과 같은 방향인가
  [3]   교차로 내부인가
  [4:7] 차로 종류 (VEHICLE / BUS / BIKE)
  [7]   왼쪽으로 차선변경 가능한가
  [8]   오른쪽으로 차선변경 가능한가
  [9]   규칙상 도달 가능한 차로인가        ← 예측 공간을 절반으로 줄여주는 신호

궤적(x, y)은 dataset_map.py 와 비트 단위로 같고, 차선(lanes)도 밀리미터 이하(중앙값 3e-5 m)로
일치한다 — float32 연산 순서 차이일 뿐이다. 규칙 피처를 넣은 효과만 따로 볼 수 있다.
그러면서 ArgoverseStaticMap 객체를 만들지 않고, 20개를 고른 뒤 리샘플링해서 조금 더 빠르다.

  python src/dataset_lane.py        # 동작 확인 + dataset_map 대비 속도
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting import scenario_serialization

from dataset_map import OBS_LEN, PRED_LEN, N_LANES, N_PTS, _resample, _rotation_matrix
import lane_frame as lf
from heading_decomp import build_heading, wrap
from lane_graph import LaneGraph, REACH_MARGIN_M

import json

PRED_SEC = PRED_LEN / 10.0        # 예측 구간 6초
N_MODES = 6                       # L3: 후보 경로 = 예측 모드의 개수 (기존 K=6 과 맞춤)
N_RPTS = 64                       # 경로 하나를 호길이 등간격 몇 점으로 표현할지


class Av2LaneRuleDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 limit: Optional[int] = None, with_rules: bool = True,
                 centerline: str = "api", select: str = "centroid",
                 theta_ch: bool = False, h_src: str = "av2",
                 routes: bool = False, n_modes: int = N_MODES):
        """centerline / select 기본값은 dataset_map.py 와 숫자까지 일치하도록 맞춰져 있다.
        (select="nearest" 는 차선까지의 최단거리로 고르는 개선안이지만 기존 실험과 달라진다)

        routes=True 면 L3 용으로 후보 경로 K개를 함께 준다 (아래 _routes 참고).

        theta_ch=True 면 입력에 3채널을 **덧붙인다** (기존 5채널은 그대로 둔다):
            [sin theta, cos theta, valid]      theta = h - k  (차로 대비 잔차각)
        대체가 아니라 추가인 이유는 이전 실험(커밋 89372e6)에서 방향 채널을 대체했을 때
        나빠졌기 때문이다. h_src="build" 면 h 를 AV2 heading 대신 위치차분으로 만든다.
        """
        self.root = Path(data_root) / split
        self.with_rules = with_rules
        self.theta_ch, self.h_src = theta_ch, h_src
        self.routes, self.n_modes = routes, n_modes
        self.centerline, self.select = centerline, select
        self.dirs = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / f"scenario_{d.name}.parquet").exists():
                self.dirs.append(d)
        if limit is not None:
            self.dirs = self.dirs[:limit]

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        sdir = self.dirs[idx]
        sid = sdir.name

        # --- 궤적 (dataset_map.py 와 동일) ---
        s = scenario_serialization.load_argoverse_scenario_parquet(
            sdir / f"scenario_{sid}.parquet")
        focal = next(t for t in s.tracks if t.track_id == s.focal_track_id)
        st = sorted(focal.object_states, key=lambda x: x.timestep)
        pos = np.array([x.position for x in st], dtype=np.float32)
        vel = np.array([x.velocity for x in st], dtype=np.float32)
        head = np.array([x.heading for x in st], dtype=np.float32)

        origin = pos[OBS_LEN - 1].copy()
        theta = head[OBS_LEN - 1].copy()
        R = _rotation_matrix(-theta)
        pos_n = (pos - origin) @ R.T
        x = np.concatenate([pos_n, vel @ R.T, (head - theta).reshape(-1, 1)], axis=1)[:OBS_LEN]
        y = pos_n[OBS_LEN:]

        # --- 지도: 원본 JSON에서 직접 그래프를 만든다 ---
        # av2 API 를 거치면 중심선을 10점으로 다시 만들어 주는데, 원본(중앙값 12점)을
        # 그대로 쓰는 편이 정보 손실도 적고 빠르다.
        raw = json.loads((sdir / f"log_map_archive_{sid}.json").read_text())
        g = LaneGraph.from_json_dict(raw, centerline=self.centerline)

        ids = list(g.lanes)
        norm = {i: _resample((g.lanes[i].centerline.astype(np.float32) - origin) @ R.T, N_PTS)
                for i in ids}
        if self.select == "centroid":       # dataset_map.py 와 동일한 기준
            d = np.array([np.linalg.norm(norm[i].mean(axis=0)) for i in ids])
        else:                               # 차선까지의 최단거리 (더 합리적이지만 기준선이 달라짐)
            d = np.array([np.min(np.linalg.norm(norm[i], axis=1)) for i in ids])
        sel = [ids[o] for o in np.argsort(d)[:N_LANES]]

        lane_arr = np.zeros((N_LANES, N_PTS, 2), dtype=np.float32)
        lane_mask = np.zeros(N_LANES, dtype=np.float32)
        for i, lid in enumerate(sel):
            lane_arr[i] = norm[lid]
            lane_mask[i] = 1.0

        if self.theta_ch:
            # h = k + theta 분해를 입력에 얹는다. k 는 이미 뽑아 놓은 20개 차로의
            # 리샘플 점에서 구하므로(50 x 200 비교) 비용이 거의 없다.
            #   h : 진행방향 (focal 정규화 프레임). AV2 heading 은 차체 방향이라 이동방향과
            #       슬립각만큼 어긋나고, 그 값으로 적분하면 복원이 1.69 m 어긋난다.
            #   k : 진행방향과 같은 쪽을 향하는 가장 가까운 중심선 점의 접선각
            #   valid : 5 m 안에 그런 점이 없으면 0 — "여기서는 차로를 믿지 마라"
            if self.h_src == "build":
                h_city, _ = build_heading(pos.astype(np.float64), h_ref=head.astype(np.float64))
                h_n = wrap(h_city - theta)[:OBS_LEN]
            else:
                h_n = wrap((head - theta).astype(np.float64))[:OBS_LEN]
            P = np.concatenate([norm[l] for l in sel]).astype(np.float64)
            G = np.concatenate([np.gradient(norm[l].astype(np.float64), axis=0) for l in sel])
            n = np.linalg.norm(G, axis=1, keepdims=True); n[n < 1e-9] = 1.0
            A = np.arctan2(*(G / n)[:, ::-1].T)
            th = np.zeros(OBS_LEN); vd = np.zeros(OBS_LEN)
            for t in range(OBS_LEN):
                d2 = ((P - pos_n[t]) ** 2).sum(axis=1)
                d2 = np.where(np.cos(A - h_n[t]) > 0, d2, np.inf)   # 대향차로 제외
                j = int(d2.argmin())
                if d2[j] <= 25.0:                                   # (5 m)^2
                    th[t], vd[t] = wrap(h_n[t] - A[j]), 1.0
            x = np.concatenate(
                [x, np.stack([np.sin(th) * vd, np.cos(th) * vd, vd], axis=1)],
                axis=1).astype(np.float32)

        out = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "lanes": torch.from_numpy(lane_arr),
            "lane_mask": torch.from_numpy(lane_mask),
            "origin": torch.from_numpy(origin),
            "theta": torch.tensor(theta, dtype=torch.float32),
            "scenario_id": sid,
        }
        if not (self.with_rules or self.routes):
            return out

        # --- 도달 가능 차로 (규칙 피처와 후보 경로가 함께 쓴다) ---
        h0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        # 예측 시점에는 미래 주행거리를 모르니 현재 속도로 추정한다 (6초 × 현재 속력).
        speed = float(np.linalg.norm(vel[OBS_LEN - 1]))
        budget = max(20.0, speed * PRED_SEC) + REACH_MARGIN_M
        path_city = pos_n[:OBS_LEN].astype(np.float64) @ R + origin
        starts = g.candidate_lanes(origin.astype(np.float64), h0, path=path_city)
        reach = g.reachable(starts, budget) if starts else set()

        if self.with_rules:
            feat = np.zeros((N_LANES, LaneGraph.N_FEAT + 1), dtype=np.float32)
            for i, lid in enumerate(sel):
                feat[i] = g.lane_features(lid, h0, reach)
            out["lane_feat"] = torch.from_numpy(feat)
            out["n_reachable"] = torch.tensor(len(reach), dtype=torch.float32)

        if self.routes:
            out.update(self._routes(g, starts, reach, speed, path_city, origin, R))
        return out

    def _routes(self, g, starts, reach, speed, path_city, origin, R):
        """L3: 후보 경로 K개를 예측 모드로 쓰기 위한 텐서들.

        순위는 **관측된 과거와의 적합도**로 매긴다 — 미래를 안 보므로 추론 때도 쓸 수 있고,
        실측에서 호길이 정렬 대비 recall@1 이 43.1% -> 63.9% 로 크게 오른다(recall@5 는 비슷).

        밴드(좌/우 횡오프셋 상한)를 함께 준다. succ-only 경로에서는 차선변경이 잔차 d 로
        표현되므로 ±1.75 로 자르면 표현력이 recall@40 79.7% 에서 막히고, 그렇다고 평평하게
        ±3.6 으로 열면 밴드의 27.6% 가 대향차로가 된다. 규칙상 갈 수 있는 쪽만 넓힌다.

        경로가 하나도 없으면(차로 후보 0개 — 주차장 등) **직진 가상 경로** 하나를 준다.
        모델이 항상 최소 한 모드는 갖게 하기 위한 것이고, 그런 샘플은 route_mask 의
        합이 1 이고 밴드가 반폭이라 밖에서 구별된다.
        """
        K, M = self.n_modes, N_RPTS
        rts = lf.build_routes(g, starts, reach, v0=speed) if starts else []
        if rts:
            order = np.argsort([lf.to_frame(path_city, r)[2] for r in rts])
            rts = [rts[i] for i in order[:K]]

        P = np.zeros((K, M, 2), np.float32); T = np.zeros((K, M, 2), np.float32)
        B = np.zeros((K, M, 2), np.float32); L = np.zeros(K, np.float32)
        SD = np.zeros((K, 2), np.float32)          # 현재 위치의 (s0, d0)
        mask = np.zeros(K, np.float32)
        now = path_city[-1:]                        # 예측 시작점 (city 좌표)
        for i, r in enumerate(rts):
            q = lf.resample_route(r, M)
            ll, lr = lf.rule_band(g, r)
            j = np.clip((q["s"] / max(q["len"], 1e-6) * (len(r["s"]) - 1)).astype(int),
                        0, len(r["s"]) - 1)
            P[i] = ((q["pts"] - origin) @ R.T).astype(np.float32)
            T[i] = (q["tan"] @ R.T).astype(np.float32)
            B[i] = np.stack([ll[j], lr[j]], axis=1).astype(np.float32)
            L[i] = q["len"]; mask[i] = 1.0
            # L0 적분기의 시작 상태. 전진 대응을 쓰는 to_frame 과 같은 기준으로 잡아야
            # 첫 스텝에서 궤적이 튀지 않는다.
            s0, d0, _ = lf.to_frame(np.concatenate([path_city, now]), r)
            SD[i] = (s0[-1], d0[-1])
        if not rts:                       # 폴백: 현재 진행방향으로 뻗는 직진 경로
            ln = max(30.0, speed * PRED_SEC + 20.0)
            P[0, :, 0] = np.linspace(0.0, ln, M); T[0, :, 0] = 1.0
            B[0] = lf.LANE_HALF_W; L[0] = ln; mask[0] = 1.0
        return {"routes": torch.from_numpy(P), "route_tan": torch.from_numpy(T),
                "route_band": torch.from_numpy(B), "route_len": torch.from_numpy(L),
                "route_sd0": torch.from_numpy(SD), "route_mask": torch.from_numpy(mask),
                "v0": torch.tensor(speed, dtype=torch.float32),
                "route_fallback": torch.tensor(0.0 if rts else 1.0)}


if __name__ == "__main__":
    import time
    from dataset_map import Av2MapDataset
    root = "/data/argoverse2/motion_forecasting"
    ds = Av2LaneRuleDataset(root, "val", limit=30)
    b = ds[0]
    print(f"x {tuple(b['x'].shape)}  y {tuple(b['y'].shape)}  "
          f"lanes {tuple(b['lanes'].shape)}  lane_feat {tuple(b['lane_feat'].shape)}")
    f = b["lane_feat"].numpy()
    m = b["lane_mask"].numpy().astype(bool)
    print(f"\n실제 차선 {int(m.sum())}개 중")
    print(f"  focal 과 같은 방향   {int(f[m,2].sum()):3d}개")
    print(f"  교차로 내부          {int(f[m,3].sum()):3d}개")
    print(f"  좌 차선변경 가능      {int(f[m,7].sum()):3d}개")
    print(f"  우 차선변경 가능      {int(f[m,8].sum()):3d}개")
    print(f"  규칙상 도달 가능      {int(f[m,9].sum()):3d}개  (지도 전체 도달 {int(b['n_reachable'])}개)")

    old = Av2MapDataset(root, "val", limit=30)
    for name, d in (("dataset_map (기존)", old), ("dataset_lane (규칙 포함)", ds)):
        d[0]
        t = time.perf_counter()
        for i in range(20):
            d[i]
        print(f"{name:26} {(time.perf_counter()-t)/20*1000:6.1f} ms/샘플")
