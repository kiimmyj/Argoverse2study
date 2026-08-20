"""
hdmap_render.py - AV2 HD map을 '도로처럼' 그린다 (animation/animate_focal.py 방식).

기존 시각화는 차선 하나하나를 회색 폴리곤(테두리 포함)으로 그려서
차선 구분선 · 도로 경계 · 차선 조각의 이음매가 전부 같은 회색 선으로 보였다.
여기서는 애니메이션 렌더러와 같은 규칙을 쓴다:

  ① 도로 면 : drivable area + 차선 폴리곤을 '테두리 없이' 한 색으로 깔아 하나의 아스팔트 면
  ② 노면표시: HD map의 실제 mark_type(점선/실선, 흰색/노란색/이중선)만 선으로 그림
  ③ 교차로 / 자전거도로 / 횡단보도는 색을 달리해 구분

좌표는 focal 기준 정규화 좌표(원점=focal 마지막 관측, +x=진행방향)를 그대로 쓴다.
→ 모든 그림에서 focal이 중앙, 도로 방향이 수평으로 정렬된다(애니메이션 카메라와 동일).

사용:
    scene = build_scene(sdir / f"log_map_archive_{sid}.json", origin, theta)
    draw_scene(ax, scene, xlim, ylim)
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from av2.map.map_api import ArgoverseStaticMap

# --- 색 (animate_focal.py 와 동일 계열) ---------------------------------------
BG          = "#16171b"   # 도로 밖
DRIVABLE    = "#2b2d32"   # 주행가능영역(갓길 포함)
ASPHALT     = "#34363c"   # 차로
ASPHALT_INT = "#3b3d44"   # 교차로 안 차로
BIKE        = "#31403b"   # 자전거도로
CURB        = "#4a4d55"   # 도로 바깥 경계(연석)
MARK_W      = "#d8d8d0"   # 흰색 노면표시
MARK_Y      = "#e0c85a"   # 노란색 노면표시
MARK_B      = "#5aa9e0"   # 파란색 노면표시
CROSSWALK   = "#b9bcc4"   # 횡단보도
TEXT        = "#c9ccd2"

DASH_M, GAP_M = 3.0, 9.0   # 점선 3m / 공백 9m (미국 기준)
DOUBLE_OFF = 0.18          # 이중선 간격 [m]

# mark_type -> [(스타일, 색, 중심선에서의 offset[m]), ...]
_MARK_STYLE = {
    "DASHED_WHITE":        [("dash",  MARK_W, 0.0)],
    "DASHED_YELLOW":       [("dash",  MARK_Y, 0.0)],
    "SOLID_WHITE":         [("solid", MARK_W, 0.0)],
    "SOLID_YELLOW":        [("solid", MARK_Y, 0.0)],
    "SOLID_BLUE":          [("solid", MARK_B, 0.0)],
    "DOUBLE_SOLID_WHITE":  [("solid", MARK_W, -DOUBLE_OFF), ("solid", MARK_W, DOUBLE_OFF)],
    "DOUBLE_SOLID_YELLOW": [("solid", MARK_Y, -DOUBLE_OFF), ("solid", MARK_Y, DOUBLE_OFF)],
    "DOUBLE_DASH_WHITE":   [("dash",  MARK_W, -DOUBLE_OFF), ("dash",  MARK_W, DOUBLE_OFF)],
    "DOUBLE_DASH_YELLOW":  [("dash",  MARK_Y, -DOUBLE_OFF), ("dash",  MARK_Y, DOUBLE_OFF)],
    "DASH_SOLID_WHITE":    [("dash",  MARK_W, -DOUBLE_OFF), ("solid", MARK_W, DOUBLE_OFF)],
    "DASH_SOLID_YELLOW":   [("dash",  MARK_Y, -DOUBLE_OFF), ("solid", MARK_Y, DOUBLE_OFF)],
    "SOLID_DASH_WHITE":    [("solid", MARK_W, -DOUBLE_OFF), ("dash",  MARK_W, DOUBLE_OFF)],
    "SOLID_DASH_YELLOW":   [("solid", MARK_Y, -DOUBLE_OFF), ("dash",  MARK_Y, DOUBLE_OFF)],
    # NONE / UNKNOWN 은 실제로 도색이 없는 구간(주로 교차로 안)이라 그리지 않는다.
}


# --- 기하 유틸 ---------------------------------------------------------------
def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def normalize(pts: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    """city 좌표 (N,2) → focal 기준 정규화 좌표 (dataset_map.py 와 동일)."""
    return (np.asarray(pts, dtype=np.float32)[:, :2] - origin) @ _rot(-theta).T


def _cumdist(poly: np.ndarray):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _sample(poly: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def _dashes(poly: np.ndarray, dash=DASH_M, gap=GAP_M):
    """폴리라인을 데이터(m) 단위로 점선 조각들로 자른다. 축척·dpi와 무관하게 3m/9m."""
    s = _cumdist(poly)
    L = float(s[-1])
    if L < 1e-6:
        return []
    out, a = [], 0.0
    while a < L:
        b = min(a + dash, L)
        n = max(2, int((b - a) / 0.5) + 2)
        out.append(_sample(poly, s, np.linspace(a, b, n)))
        a += dash + gap
    return out


def _offset(poly: np.ndarray, d: float) -> np.ndarray:
    """폴리라인을 법선 방향으로 d[m] 평행이동 (이중선용)."""
    if d == 0.0 or len(poly) < 2:
        return poly
    tan = np.gradient(poly, axis=0)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln < 1e-9] = 1.0
    return poly + d * nrm / ln


def _key(poly: np.ndarray):
    """인접 차선이 공유하는 경계선을 한 번만 그리기 위한 중복 제거 키."""
    r = np.round(poly, 2)
    fwd = tuple(map(tuple, r))
    rev = tuple(map(tuple, r[::-1]))
    return min(fwd, rev)


def _bbox(poly: np.ndarray):
    return poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()


# --- 장면 구성 ---------------------------------------------------------------
@dataclass
class RoadScene:
    """정규화 좌표로 변환해 둔 HD map 요소들 (그리기 전용)."""
    drivable: list = field(default_factory=list)     # (N,2) 폴리곤
    lanes: list = field(default_factory=list)        # (폴리곤, 색)
    marks: list = field(default_factory=list)        # (폴리라인, mark_type)
    crossings: list = field(default_factory=list)    # (edge1(2,2), edge2(2,2))


def build_scene(map_json: Path, origin: np.ndarray, theta: float) -> RoadScene:
    amap = ArgoverseStaticMap.from_json(Path(map_json))
    sc = RoadScene()

    for da in amap.get_scenario_vector_drivable_areas():
        sc.drivable.append(normalize(da.xyz, origin, theta))

    seen = set()
    for lane in amap.vector_lane_segments.values():
        left = normalize(lane.left_lane_boundary.xyz, origin, theta)
        right = normalize(lane.right_lane_boundary.xyz, origin, theta)
        if str(lane.lane_type).endswith("BIKE"):
            color = BIKE
        elif lane.is_intersection:
            color = ASPHALT_INT
        else:
            color = ASPHALT
        sc.lanes.append((np.concatenate([left, right[::-1]], axis=0), color))

        for bnd, mtype in ((left, lane.left_mark_type), (right, lane.right_mark_type)):
            name = str(mtype).split(".")[-1]
            if name not in _MARK_STYLE:
                continue
            k = _key(bnd)
            if k in seen:          # 옆 차선과 공유하는 경계선 → 한 번만
                continue
            seen.add(k)
            sc.marks.append((bnd, name))

    for pc in amap.get_scenario_ped_crossings():
        e1, e2 = pc.get_edges_2d()
        sc.crossings.append((normalize(e1, origin, theta), normalize(e2, origin, theta)))

    return sc


# --- 그리기 ------------------------------------------------------------------
def _visible(poly, box):
    if box is None:
        return True
    x0, x1, y0, y1 = box
    a, b, c, d = _bbox(poly)
    return a <= x1 and b >= x0 and c <= y1 and d >= y0


def draw_scene(ax, scene: RoadScene, xlim=None, ylim=None, mark_lw=1.3):
    """ax 에 도로를 그린다. xlim/ylim 을 주면 시야 밖 요소는 건너뛴다."""
    box = None
    if xlim is not None and ylim is not None:
        pad = 0.1 * (xlim[1] - xlim[0])
        box = (xlim[0] - pad, xlim[1] + pad, ylim[0] - pad, ylim[1] + pad)

    ax.set_facecolor(BG)

    # ① 주행가능영역(갓길 포함) — 바깥 경계만 연석 색으로
    for poly in scene.drivable:
        if _visible(poly, box):
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=DRIVABLE,
                                    edgecolor=CURB, lw=1.0, zorder=1))
    # ② 차로 면 — 테두리 없음 → 조각들이 이어져 하나의 도로로 보인다
    for poly, color in scene.lanes:
        if _visible(poly, box):
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=color,
                                    edgecolor="none", zorder=2))
    # ③ 횡단보도 (얼룩말 무늬)
    for e1, e2 in scene.crossings:
        if not _visible(np.concatenate([e1, e2]), box):
            continue
        n = max(2, int(np.linalg.norm(e1[1] - e1[0]) / 0.9))
        t = np.linspace(0, 1, 2 * n + 1)
        p1 = e1[0] + np.outer(t, e1[1] - e1[0])
        p2 = e2[0] + np.outer(t, e2[1] - e2[0])
        for i in range(0, 2 * n, 2):
            quad = np.array([p1[i], p1[i + 1], p2[i + 1], p2[i]])
            ax.add_patch(MplPolygon(quad, closed=True, facecolor=CROSSWALK,
                                    edgecolor="none", alpha=0.5, zorder=3))
    # ④ 노면표시
    for bnd, name in scene.marks:
        if not _visible(bnd, box):
            continue
        for style, color, off in _MARK_STYLE[name]:
            line = _offset(bnd, off)
            pieces = _dashes(line) if style == "dash" else [line]
            for p in pieces:
                ax.plot(p[:, 0], p[:, 1], color=color, lw=mark_lw,
                        solid_capstyle="butt", zorder=4)


def draw_vehicle(ax, x, y, yaw=0.0, length=4.8, width=2.0,
                 color="#ff5c33", edge="white", zorder=9):
    """차량을 실제 크기 박스로 (애니메이션의 focal 표시와 동일)."""
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2, width / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    R = np.array([[c, -s], [s, c]])
    ax.add_patch(MplPolygon(corners @ R.T + [x, y], closed=True, facecolor=color,
                            edgecolor=edge, lw=1.0, zorder=zorder))


def draw_scale_bar(ax, xlim, ylim, color=TEXT):
    """좌하단 축척 막대 (10/20/50/100m 중 시야에 맞는 값)."""
    span = xlim[1] - xlim[0]
    length = min([m for m in (10, 20, 50, 100, 200) if m >= span / 6], default=200)
    x0 = xlim[0] + 0.06 * span
    y0 = ylim[0] + 0.06 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.0, zorder=10,
            solid_capstyle="butt")
    for xe in (x0, x0 + length):
        ax.plot([xe, xe], [y0 - 0.012 * span, y0 + 0.012 * span], color=color,
                lw=2.0, zorder=10)
    ax.text(x0 + length / 2, y0 + 0.02 * span, f"{length} m", color=color,
            fontsize=8, ha="center", va="bottom", zorder=10)
