"""
lane_graph.py - AV2 HD map에서 '차로 간 이동 가능 규칙'을 복원한다.

dataset_map.py 는 지도를 '주변 차선 20개의 중심선'으로만 쓴다. 그래서 모델 입장에서는
대향차로와 같은 방향 차로가 구분되지 않고, 실선/점선도 보이지 않는다.
역주행하거나 넘을 수 없는 실선을 넘는 예측이 나오는 이유다.

지도는 이미 답을 갖고 있다. 여기서 규칙 네 가지로 복원한다.

  SUCC   진행    successors. 중심선 방향을 따라서만 (역주행 금지의 뿌리)
  LEFT   좌변경  left_neighbor  중 같은 방향이면서 표시를 넘을 수 있는 것만
  RIGHT  우변경  right_neighbor 중 같은 방향이면서 표시를 넘을 수 있는 것만
  UTURN  U턴     교차로를 거쳐 진행방향이 뒤집히는 경로

val 120개 시나리오(차로 8,030개)에서 실측해 정한 규칙이다.

  - left_neighbor 의 **45.2%가 대향차로**다 (right_neighbor 는 0.1%). 우측통행이라 그렇다.
    '인접차로 = 차선변경 가능' 으로 두면 좌변경의 절반이 역주행이 된다.
  - 대향 인접차로의 표시는 96%가 노란색 계열 아니면 NONE, 같은 방향은 78%가 흰색 계열이다.
    다만 NONE 이 양쪽에 다 나오므로(대향 37.7% / 동일 17.9%) **표시만으로는 판별할 수 없고
    방향 검사가 1차 근거**다.
  - successors 가 방향을 뒤집는 경우는 8,201쌍 중 2건뿐. U턴은 교차로를 경유해 찾아야 한다.

실제 미래 궤적으로 검증한 결과 (src/lane_graph_check.py):

              적중률   후보 차로        역주행 차로
  튜닝 300개   98.1%   31.5 → 14.4개   12.92 → 1.35개
  미사용 400개  97.7%   30.5 → 15.0개   11.85 → 1.63개   ← 처음 보는 시나리오

튜닝에 쓴 표본과 안 쓴 표본이 거의 같아 과적합은 아니다.
(상수를 반경 12m / 8개로 키우면 튜닝 표본에서 100%가 나오지만, 미사용 표본에서는
 98.3%로 0.6%p 개선에 그치면서 역주행 차로가 1.63 → 2.38개로 늘어난다. 그래서 5m / 4개를 둔다.)

즉 **맞는 답은 거의 다 남기면서 역주행 선택지를 지운다.** 남은 1.5개는 좌회전을 두 번
이어붙이면 반대 방향이 되는 정상 경로라 역주행이 아니다.

사용:
  g = LaneGraph.from_json_dict(raw)
  g.movable(lane_id)                    # 이 차로에서 갈 수 있는 (차로, 이동종류)
  S = g.candidate_lanes(xy, heading)    # 현재 위치의 차로 후보 (1개로 찍으면 안 된다)
  g.reachable(S, dist + REACH_MARGIN_M) # 도달 가능한 차로
  g.lane_features(lane_id, ref, R)      # 모델에 넣을 규칙 피처
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# --------------------------------------------------------------------------- 표시 종류 → 넘을 수 있는가
_DASH = {"DASHED_WHITE", "DASHED_YELLOW", "DOUBLE_DASH_WHITE", "DOUBLE_DASH_YELLOW"}
_SOLID = {"SOLID_WHITE", "SOLID_YELLOW", "DOUBLE_SOLID_WHITE", "DOUBLE_SOLID_YELLOW",
          "SOLID_BLUE"}
# 한쪽만 점선인 표시. 두 토큰 중 **앞이 관측 차로에 가까운 선**이라고 본다.
# av2 문서에 명시되어 있지 않아 상수로 빼두었다. 해석을 뒤집으려면 두 집합을 맞바꾸면 된다.
# (전체 인접쌍의 2.6%라 영향은 작지만, 미국 관행상 점선 쪽 차량이 넘어갈 수 있다.)
_NEAR_DASH = {"DASH_SOLID_WHITE", "DASH_SOLID_YELLOW"}
_NEAR_SOLID = {"SOLID_DASH_WHITE", "SOLID_DASH_YELLOW"}
# NONE/UNKNOWN 은 '도색이 없음'이지 '막혔음'이 아니다. 교차로 내부나 미도색 구간에서 나온다.
_UNMARKED = {"NONE", "UNKNOWN"}


def mark_crossable(mark: str) -> bool:
    """이 표시를 넘어 차선변경을 할 수 있는가. 방향 검사와 반드시 함께 써야 한다."""
    if mark in _DASH or mark in _NEAR_DASH or mark in _UNMARKED:
        return True
    if mark in _SOLID or mark in _NEAR_SOLID:
        return False
    return True                       # 모르는 값은 막지 않는다 (규칙이 과하게 좁아지지 않도록)


class Move(str, Enum):
    SUCC = "succ"
    LEFT = "left"
    RIGHT = "right"
    UTURN = "uturn"


LANE_TYPES = ("VEHICLE", "BUS", "BIKE")

# 같은 방향으로 볼 각도 한계. 실측 분포가 0.3° 아니면 180° 근처로 완전히 갈려서
# 90~135° 사이에는 표본이 하나도 없었다. 어디를 잘라도 같은 결과가 나온다.
SAME_DIR_COS = 0.0
# 차선변경 한 번이 소비하는 종방향 거리. reachable() 의 예산에서 깎는다.
LANE_CHANGE_COST_M = 15.0
# 도달 예산에 얹는 여유. 30m 로 두면 적중률이 7%p 떨어진다(차로 길이 단위로 예산을 깎기 때문).
REACH_MARGIN_M = 60.0
# 차로 후보 반경/개수. 실측으로 정한 값 — 후보를 1개로 찍으면 적중률이 98.3% → 78.2% 로
# 무너진다. 교차로에서 회전 차로들이 공간적으로 겹치기 때문에 하나만 고르면 분기를 잘못 탄다.
CAND_RADIUS_M = 5.0
CAND_TOP_K = 4


@dataclass
class Lane:
    id: int
    centerline: np.ndarray            # (M,2) city 좌표
    direction: np.ndarray             # (2,) 시작→끝 단위벡터
    length: float
    is_intersection: bool
    lane_type: str
    left_mark: str
    right_mark: str
    left_neighbor: Optional[int]
    right_neighbor: Optional[int]
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)


@dataclass
class LaneGraph:
    lanes: Dict[int, Lane]
    edges: Dict[int, List[Tuple[int, Move]]]
    # 규칙에 걸려 잘린 인접 연결. 왜 막혔는지 확인·집계용.
    rejected: Dict[int, List[Tuple[int, Move, str]]] = field(default_factory=dict)

    # ----------------------------------------------------------------- 생성
    @classmethod
    def from_static_map(cls, amap) -> "LaneGraph":
        lanes: Dict[int, Lane] = {}
        for lid, seg in amap.vector_lane_segments.items():
            cl = np.asarray(amap.get_lane_segment_centerline(lid))[:, :2]
            step = np.linalg.norm(np.diff(cl, axis=0), axis=1)
            v = cl[-1] - cl[0]
            n = float(np.linalg.norm(v))
            lanes[int(lid)] = Lane(
                id=int(lid), centerline=cl,
                direction=(v / n if n > 1e-6 else np.array([1.0, 0.0])),
                length=float(step.sum()),
                is_intersection=bool(seg.is_intersection),
                lane_type=str(getattr(seg.lane_type, "value", seg.lane_type)),
                left_mark=str(getattr(seg.left_mark_type, "value", seg.left_mark_type)),
                right_mark=str(getattr(seg.right_mark_type, "value", seg.right_mark_type)),
                left_neighbor=seg.left_neighbor_id,
                right_neighbor=seg.right_neighbor_id,
                successors=[int(s) for s in seg.successors],
                predecessors=[int(p) for p in seg.predecessors],
            )
        return cls._build(lanes)

    @classmethod
    def from_json_dict(cls, raw: dict, centerline: str = "api") -> "LaneGraph":
        """원본 JSON에서 바로 만든다. ArgoverseStaticMap 객체를 만들지 않아 더 빠르다.

        centerline="api"  좌/우 경계선에서 중심선을 10점으로 다시 계산 (av2 API와 동일한 값).
                          dataset_map.py 와 숫자까지 맞추려면 이쪽이어야 한다.
        centerline="json" JSON에 저장된 중심선(중앙값 12점)을 그대로 사용. 정보 손실이 적다.
        """
        from av2.geometry.interpolate import compute_midpoint_line
        lanes: Dict[int, Lane] = {}
        for k, s in raw["lane_segments"].items():
            if centerline == "api":
                lb = np.array([[p["x"], p["y"], p["z"]] for p in s["left_lane_boundary"]])
                rb = np.array([[p["x"], p["y"], p["z"]] for p in s["right_lane_boundary"]])
                cl = np.asarray(compute_midpoint_line(lb, rb, 10)[0])[:, :2].astype(np.float64)
            else:
                cl = np.array([[p["x"], p["y"]] for p in s["centerline"]], dtype=np.float64)
            step = np.linalg.norm(np.diff(cl, axis=0), axis=1)
            v = cl[-1] - cl[0]
            n = float(np.linalg.norm(v))
            lanes[int(k)] = Lane(
                id=int(k), centerline=cl,
                direction=(v / n if n > 1e-6 else np.array([1.0, 0.0])),
                length=float(step.sum()),
                is_intersection=bool(s["is_intersection"]),
                lane_type=s["lane_type"],
                left_mark=s["left_lane_mark_type"], right_mark=s["right_lane_mark_type"],
                left_neighbor=s["left_neighbor_id"], right_neighbor=s["right_neighbor_id"],
                successors=[int(x) for x in s["successors"]],
                predecessors=[int(x) for x in s["predecessors"]],
            )
        return cls._build(lanes)

    @classmethod
    def _build(cls, lanes: Dict[int, Lane]) -> "LaneGraph":
        edges: Dict[int, List[Tuple[int, Move]]] = {k: [] for k in lanes}
        rejected: Dict[int, List[Tuple[int, Move, str]]] = {k: [] for k in lanes}

        for lid, ln in lanes.items():
            # 진행: successors 를 그대로 쓴다. 지도가 방향까지 담아 만든 연결이라
            # 여기서는 존재 여부만 확인한다. predecessors 는 절대 이동으로 쓰지 않는다(=역주행 금지).
            for sid in ln.successors:
                if sid in lanes:
                    edges[lid].append((sid, Move.SUCC))

            # 차선변경: 방향이 같아야 하고, 맞닿은 표시를 넘을 수 있어야 한다.
            for nid, move, mark in ((ln.left_neighbor, Move.LEFT, ln.left_mark),
                                    (ln.right_neighbor, Move.RIGHT, ln.right_mark)):
                if nid is None or nid not in lanes:
                    continue
                if float(ln.direction @ lanes[nid].direction) <= SAME_DIR_COS:
                    rejected[lid].append((nid, move, "대향차로"))
                    continue
                if not mark_crossable(mark):
                    rejected[lid].append((nid, move, f"실선({mark})"))
                    continue
                edges[lid].append((nid, move))

        g = cls(lanes=lanes, edges=edges, rejected=rejected)
        g._add_uturns()
        return g

    def _add_uturns(self, max_hops: int = 3, max_gap_m: float = 60.0) -> None:
        """교차로를 거쳐 진행방향이 뒤집히는 경로를 U턴으로 표시한다.

        successors 가 곧장 방향을 뒤집는 경우는 거의 없고(8,201쌍 중 2건), 대부분
        '좌회전 차로 → 교차로 내부 → 반대 방향 본선' 처럼 두세 단계로 나뉘어 있다.
        """
        for lid, ln in self.lanes.items():
            seen = {lid}
            q = deque([(lid, 0)])
            while q:
                cur, hop = q.popleft()
                if hop >= max_hops:
                    continue
                for nxt, mv in self.edges[cur]:
                    if mv is not Move.SUCC or nxt in seen:
                        continue
                    seen.add(nxt)
                    tgt = self.lanes[nxt]
                    if float(ln.direction @ tgt.direction) < -0.7:      # 135° 이상 꺾임
                        gap = float(np.linalg.norm(tgt.centerline[0] - ln.centerline[-1]))
                        if gap <= max_gap_m:
                            self.edges[lid].append((nxt, Move.UTURN))
                            continue
                    q.append((nxt, hop + 1))

    # ----------------------------------------------------------------- 질의
    def movable(self, lane_id: int) -> List[Tuple[int, Move]]:
        """이 차로에서 규칙상 갈 수 있는 (차로, 이동종류)."""
        return list(self.edges.get(lane_id, []))

    def reachable(self, start: Iterable[int] | int, budget_m: float) -> Set[int]:
        """주행거리 예산 안에서 도달 가능한 차로 집합. 진행은 차로 길이만큼,
        차선변경은 LANE_CHANGE_COST_M 만큼 예산을 깎는다."""
        starts = [start] if isinstance(start, (int, np.integer)) else list(start)
        best: Dict[int, float] = {}
        q = deque()
        for s in starts:
            if s in self.lanes:
                best[int(s)] = budget_m
                q.append(int(s))
        while q:
            cur = q.popleft()
            rest = best[cur]
            for nxt, mv in self.edges[cur]:
                # UTURN 은 SUCC 여러 개를 건너뛴 '표식'이라 도달성 계산에서는 제외한다.
                # 같은 목적지가 이미 SUCC 사슬로 도달 가능하고, UTURN 을 한 칸 비용으로
                # 치면 실제보다 싸져서 도달 집합이 부풀어 오른다.
                if mv is Move.UTURN:
                    continue
                cost = self.lanes[cur].length if mv is Move.SUCC else LANE_CHANGE_COST_M
                left = rest - cost
                if left < 0:
                    continue
                if left > best.get(nxt, -1.0) + 1e-6:
                    best[nxt] = left
                    q.append(nxt)
        return set(best)

    def candidate_lanes(self, xy: np.ndarray, heading_dir: Optional[np.ndarray] = None,
                        radius: float = CAND_RADIUS_M, top_k: int = CAND_TOP_K,
                        path: Optional[np.ndarray] = None) -> List[int]:
        """한 점이 속할 수 있는 차로 후보를 가까운 순으로 돌려준다.

        차로 매칭은 원래 애매하다. 특히 교차로에서는 좌/우/직진 차로가 공간적으로 겹쳐서
        가장 가까운 하나만 고르면 엉뚱한 분기에 갇힌다. 검증에서 후보를 1개로 제한하면
        적중률이 78.2%까지 떨어졌고, 4개로 늘리면 98.3%가 됐다.

        path 를 주면(최근 궤적) 그 궤적과의 평균 거리까지 더해 정렬한다.
        """
        out = []
        for lid, ln in self.lanes.items():
            if heading_dir is not None and float(ln.direction @ heading_dir) <= SAME_DIR_COS:
                continue
            d0 = float(np.min(np.linalg.norm(ln.centerline - xy, axis=1)))
            if d0 > radius:
                continue
            if path is None:
                score = d0
            else:
                dm = np.linalg.norm(path[:, None, :] - ln.centerline[None, :, :], axis=2)
                score = d0 + float(dm.min(axis=1).mean())
            out.append((score, lid))
        return [l for _, l in sorted(out)[:top_k]]

    def nearest_lane(self, xy: np.ndarray, heading_dir: Optional[np.ndarray] = None,
                     max_dist: float = CAND_RADIUS_M) -> Optional[int]:
        """가장 가까운 차로 하나. 편의용이며, 도달성 판정에는 candidate_lanes 를 써야 한다."""
        c = self.candidate_lanes(xy, heading_dir, max_dist, 1)
        return c[0] if c else None

    # ----------------------------------------------------------------- 모델 입력용
    N_FEAT = 9

    def lane_features(self, lane_id: int, ref_dir: np.ndarray,
                      reachable: Optional[Set[int]] = None) -> np.ndarray:
        """차로 하나의 규칙 피처 (9,).

        [dx, dy]        focal 기준으로 회전한 진행방향 단위벡터  ← 역주행 판단의 핵심
        [same_dir]      focal 과 같은 방향인가
        [is_inter]      교차로 내부인가
        [veh, bus, bike] 차로 종류
        [x_left]        왼쪽으로 차선변경 가능한가
        [x_right]       오른쪽으로 차선변경 가능한가
        (reachable 을 주면 마지막에 도달가능 플래그가 하나 더 붙어 (10,) 이 된다)
        """
        ln = self.lanes[lane_id]
        c, s = ref_dir[0], ref_dir[1]
        R = np.array([[c, s], [-s, c]])                 # focal 진행방향을 +x 로 돌리는 회전
        d = R @ ln.direction
        moves = {m for _, m in self.edges[lane_id]}
        f = [d[0], d[1],
             1.0 if float(ln.direction @ ref_dir) > SAME_DIR_COS else 0.0,
             1.0 if ln.is_intersection else 0.0,
             *(1.0 if ln.lane_type == t else 0.0 for t in LANE_TYPES),
             1.0 if Move.LEFT in moves else 0.0,
             1.0 if Move.RIGHT in moves else 0.0]
        if reachable is not None:
            f.append(1.0 if lane_id in reachable else 0.0)
        return np.array(f, dtype=np.float32)

    def summary(self) -> dict:
        n_rej = sum(len(v) for v in self.rejected.values())
        by = {}
        for v in self.rejected.values():
            for _, mv, why in v:
                by[f"{mv.value}:{why}"] = by.get(f"{mv.value}:{why}", 0) + 1
        cnt = {m.value: 0 for m in Move}
        for v in self.edges.values():
            for _, mv in v:
                cnt[mv.value] += 1
        return {"lanes": len(self.lanes), "edges": cnt, "rejected": n_rej, "reasons": by}


if __name__ == "__main__":
    import json
    from pathlib import Path
    root = Path("/data/argoverse2/motion_forecasting/val")
    d = sorted(p for p in root.iterdir() if p.is_dir())[0]
    g = LaneGraph.from_json_dict(json.loads((d / f"log_map_archive_{d.name}.json").read_text()))
    print(f"시나리오 {d.name}")
    print(g.summary())
    lid = next(iter(g.lanes))
    print(f"\n차로 {lid} 에서 갈 수 있는 곳: {g.movable(lid)}")
    print(f"60m 안에 도달 가능한 차로 {len(g.reachable(lid, 60.0))}개")
