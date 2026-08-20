"""
traffic_data.py - NGSIM / HighD를 하나의 공통 포맷으로 읽어들인다.

두 데이터셋은 단위(피트/미터), 좌표 정의(앞면중앙/bbox 좌상단), 샘플링(10Hz/25Hz),
주행방향(단방향/양방향)이 전부 다르다. 렌더러가 이걸 신경쓰지 않도록 여기서 흡수한다.

공통 좌표계
  x : 주행방향 [m], 항상 + 방향으로 진행
  y : 횡방향 [m], 차량 왼쪽이 +  (즉 1차로가 화면 위쪽)
  yaw : 진행방향 각도 [rad], 0 = +x

공통 tracks 컬럼
  track_id frame t x y vx vy speed yaw length width lane_id cls
  (x, y 는 차량 '중심'. 데이터셋별 기준점 차이는 여기서 보정됨)
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except ImportError:                                   # scipy 없으면 이동평균으로 대체
    savgol_filter = None

FT = 0.3048
CLS_CAR, CLS_TRUCK, CLS_MOTO = "car", "truck", "moto"


# --------------------------------------------------------------------------- 도로 기하
@dataclass
class Lane:
    lane_id: int
    y_center: float
    y_left: float
    y_right: float
    x_start: float
    x_end: float
    kind: str = "main"            # main | side (본선 옆의 램프/보조차로)
    label: str = ""


@dataclass
class Road:
    name: str
    lanes: List[Lane] = field(default_factory=list)
    x_min: float = 0.0
    x_max: float = 0.0

    @property
    def main_lanes(self):
        return [l for l in self.lanes if l.kind == "main"]

    @property
    def side_lanes(self):
        return [l for l in self.lanes if l.kind == "side"]

    @property
    def y_top(self):
        return max(l.y_left for l in self.lanes)

    @property
    def y_bottom(self):
        return min(l.y_right for l in self.lanes)


@dataclass
class Scene:
    name: str
    road: Road
    tracks: pd.DataFrame
    focal_id: str
    dt: float                      # 프레임 간격 [s]


# --------------------------------------------------------------------------- 공통 후처리
def _smooth(v, win, poly=3):
    n = len(v)
    if n < 5:
        return v.astype(float)
    w = min(win if win % 2 == 1 else win - 1, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        return v.astype(float)
    if savgol_filter is not None:
        return savgol_filter(v.astype(float), w, poly)
    k = np.ones(w) / w
    return np.convolve(np.pad(v.astype(float), (w // 2, w // 2), mode="edge"), k, mode="valid")[:n]


def _kinematics(g, dt, smooth_win, max_yaw_deg=12.0):
    """트랙 하나의 위치를 다듬고 속도/헤딩을 만든다.

    NGSIM은 영상 판독으로 만든 데이터라 위치 지터가 커서, 그대로 쓰면 박스가 떤다.
    특히 횡방향은 실제 움직임이 느린 데 비해(차선변경 1회에 4초 남짓) 노이즈가 커서
    종방향보다 두 배 넓은 창으로 눌러준다. 그래야 헤딩이 정지 상태에서 흔들리지 않는다.
    """
    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    if smooth_win > 1:
        x = _smooth(x, smooth_win)
        y = _smooth(y, 2 * smooth_win + 1)

    vx = np.gradient(x, dt) if len(x) > 1 else np.zeros_like(x)
    vy = np.gradient(y, dt) if len(y) > 1 else np.zeros_like(y)
    speed = np.hypot(vx, vy)

    # 저속(정체)에서는 gradient 방향이 튄다 -> 직전의 유효한 헤딩을 유지한다.
    yaw = np.arctan2(vy, np.maximum(vx, 1e-6))
    lim = np.deg2rad(max_yaw_deg)
    yaw = np.clip(yaw, -lim, lim)
    ok = speed > 1.0
    if ok.any():
        idx = np.where(ok, np.arange(len(yaw)), -1)
        idx = np.maximum.accumulate(idx)
        idx[idx < 0] = np.argmax(ok)
        yaw = yaw[idx]
    else:
        yaw = np.zeros_like(yaw)

    g = g.copy()
    g["x"], g["y"] = x, y
    g["vx"], g["vy"], g["speed"], g["yaw"] = vx, vy, speed, yaw
    return g


def _apply_tracks(df, dt, smooth_win):
    """트랙별로 스무딩/속도/헤딩을 계산해 다시 합친다."""
    parts = [_kinematics(g, dt, smooth_win) for _, g in df.groupby("track_id", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df


def _split_tracks(df, id_col, time_col, dt_ms, gap_factor=1.5):
    """시간이 끊긴 지점에서 트랙을 나눈다.

    NGSIM의 vehicle_id는 15분짜리 녹화 3개에 걸쳐 재사용되므로 이 처리가 필수다.
    """
    df = df.sort_values([id_col, time_col], kind="mergesort")
    d = df.groupby(id_col, sort=False)[time_col].diff()
    new = (d.isna()) | (d > dt_ms * gap_factor)
    seg = new.groupby(df[id_col]).cumsum().astype(int)
    return df[id_col].astype(str) + "_" + seg.astype(str)


# --------------------------------------------------------------------------- NGSIM
NGSIM_CLS = {1: CLS_MOTO, 2: CLS_CAR, 3: CLS_TRUCK}


def load_ngsim(lanes_df, window_df, location, focal_vehicle_id=None, smooth_win=11):
    dt = 0.1
    df = window_df.rename(columns={"global_time": "t_ms"}).copy()

    df["track_id"] = _split_tracks(df, "vehicle_id", "t_ms", 100)
    df = df.sort_values(["track_id", "t_ms"], kind="mergesort")

    # 피트 -> 미터, NGSIM 좌표 -> 공통 좌표
    df["x"] = df["local_y"] * FT              # 종방향(주행방향)
    df["y"] = -df["local_x"] * FT             # 횡방향, 1차로가 위로 오도록 부호 반전
    df["length"] = df["v_length"] * FT
    df["width"] = df["v_width"] * FT
    df["cls"] = df["v_class"].map(NGSIM_CLS).fillna(CLS_CAR)
    df["t"] = (df["t_ms"] - df["t_ms"].min()) / 1000.0
    df["frame"] = ((df["t_ms"] - df["t_ms"].min()) // 100).astype(int)

    df = _apply_tracks(df, dt, smooth_win)

    # NGSIM 좌표는 차량 '앞면 중앙'이다 -> 길이 절반만큼 뒤로 밀어 중심으로 바꾼다.
    df["x"] = df["x"] - 0.5 * df["length"] * np.cos(df["yaw"])
    df["y"] = df["y"] - 0.5 * df["length"] * np.sin(df["yaw"])

    focal_id = _pick_focal_track(df, focal_vehicle_id)
    road = _ngsim_road(lanes_df, location)
    cols = ["track_id", "frame", "t", "x", "y", "vx", "vy", "speed",
            "yaw", "length", "width", "lane_id", "cls"]
    return Scene(f"NGSIM {location}", road, df[cols].reset_index(drop=True), focal_id, dt)


def _pick_focal_track(df, vehicle_id):
    """분할된 트랙 중 focal을 고른다. id를 지정하면 그 id의 가장 긴 조각."""
    sizes = df.groupby("track_id").size()
    if vehicle_id is not None:
        pref = f"{int(vehicle_id)}_"
        cand = sizes[sizes.index.str.startswith(pref)]
        if not cand.empty:
            return str(cand.idxmax())
        print(f"  ! vehicle_id={vehicle_id}를 시간창에서 못 찾음 -> 가장 긴 트랙 사용")
    return str(sizes.idxmax())


def _ngsim_road(lanes_df, location):
    from ngsim_fetch import LANE_LAYOUT
    layout = LANE_LAYOUT.get(location)
    ld = lanes_df.set_index("lane_id")

    if layout is None:                        # 간선도로(lankershim/peachtree) 등: 일반 처리
        layout = {"main": sorted(ld.index.tolist()), "side": {}}

    main = [i for i in layout["main"] if i in ld.index]
    centers = sorted(((int(i), float(ld.loc[i, "y_center"])) for i in main),
                     key=lambda p: -p[1])     # 위(1차로)부터 아래로
    widths = [abs(centers[k][1] - centers[k + 1][1]) for k in range(len(centers) - 1)]
    lw = float(np.median(widths)) if widths else 3.6

    lanes = []
    for k, (lid, yc) in enumerate(centers):
        y_left = (yc + centers[k - 1][1]) / 2 if k > 0 else yc + lw / 2
        y_right = (yc + centers[k + 1][1]) / 2 if k < len(centers) - 1 else yc - lw / 2
        lanes.append(Lane(lid, yc, y_left, y_right,
                          float(ld.loc[lid, "x_start"]), float(ld.loc[lid, "x_end"]), "main"))

    # 램프/보조차로는 물리적으로 '본선 오른쪽의 같은 차로 대역'이 종방향 구간만 달리하며
    # on-ramp -> aux -> off-ramp로 이어지는 것이다. lane별 평균 횡위치를 그대로 쓰면
    # 60cm 남짓의 단차가 생기므로, 관측량으로 가중평균한 하나의 대역으로 합친다.
    side = {lid: lab for lid, lab in layout["side"].items() if lid in ld.index}
    if side:
        w = ld.loc[list(side), "n"].to_numpy(float)
        yc = float(np.average(ld.loc[list(side), "y_center"].to_numpy(float), weights=w))
        for lid, lab in side.items():
            lanes.append(Lane(lid, yc, yc + lw / 2, yc - lw / 2,
                              float(ld.loc[lid, "x_start"]), float(ld.loc[lid, "x_end"]),
                              "side", lab))

    return Road(location, lanes,
                min(l.x_start for l in lanes), max(l.x_end for l in lanes))


# --------------------------------------------------------------------------- HighD
# 주의: HighD는 levelXdata 라이선스가 필요해 이 환경에 파일이 없다. 아래는 공개 포맷 명세
# (tracks / tracksMeta / recordingMeta)에 맞춰 작성했으나 실제 파일로는 아직 검증하지 못했다.
HIGHD_CLS = {"car": CLS_CAR, "truck": CLS_TRUCK}


def load_highd(prefix, focal_vehicle_id=None, direction=None, smooth_win=11):
    """prefix 예시: /path/to/highd/data/01  ->  01_tracks.csv 등을 읽는다."""
    tracks = pd.read_csv(f"{prefix}_tracks.csv")
    meta = pd.read_csv(f"{prefix}_tracksMeta.csv")
    rec = pd.read_csv(f"{prefix}_recordingMeta.csv").iloc[0]

    fps = float(rec["frameRate"])
    dt = 1.0 / fps
    meta.columns = [c.strip() for c in meta.columns]

    if direction is None:                     # focal이 속한 방향의 차로만 그린다
        if focal_vehicle_id is not None:
            direction = int(meta.loc[meta["id"] == int(focal_vehicle_id), "drivingDirection"].iloc[0])
        else:
            direction = int(meta.sort_values("numFrames", ascending=False)["drivingDirection"].iloc[0])

    ids = meta.loc[meta["drivingDirection"] == direction, "id"]
    df = tracks[tracks["id"].isin(ids)].copy()
    df = df.merge(meta[["id", "class", "drivingDirection"]], on="id", how="left")

    # HighD의 x,y는 bbox 좌상단(영상 좌표: x 오른쪽, y 아래쪽). 먼저 중심으로 옮긴다.
    xc = df["x"] + df["width"] / 2
    yc = df["y"] + df["height"] / 2
    if direction == 2:                        # 아래쪽 차로 = 오른쪽으로 주행
        df["x"], df["y"] = xc, -yc
        df["vx"], df["vy"] = df["xVelocity"], -df["yVelocity"]
        markings = rec["lowerLaneMarkings"]
        sign = -1.0
    else:                                     # 위쪽 차로 = 왼쪽으로 주행 -> 뒤집어서 +x 진행으로 통일
        df["x"], df["y"] = -xc, yc
        df["vx"], df["vy"] = -df["xVelocity"], df["yVelocity"]
        markings = rec["upperLaneMarkings"]
        sign = 1.0

    df["length"] = df["width"]                # HighD의 width=진행방향 길이, height=횡폭
    df["width"] = df["height"]
    df["cls"] = df["class"].str.lower().map(HIGHD_CLS).fillna(CLS_CAR)
    df["lane_id"] = df["laneId"]
    df["track_id"] = df["id"].astype(str) + "_0"
    df["frame"] = df["frame"] - df["frame"].min()
    df["t"] = df["frame"] * dt

    df = df.sort_values(["track_id", "frame"], kind="mergesort")
    df = _apply_tracks(df, dt, smooth_win)

    ys = sign * np.array([float(v) for v in str(markings).split(";")])
    ys = np.sort(ys)[::-1]                    # 위에서 아래로
    lanes = [Lane(k + 1, (ys[k] + ys[k + 1]) / 2, ys[k], ys[k + 1],
                  float(df["x"].min()), float(df["x"].max()), "main")
             for k in range(len(ys) - 1)]

    focal_id = _pick_focal_track(df, focal_vehicle_id)
    road = Road(f"highD {int(rec['id'])}", lanes,
                float(df["x"].min()), float(df["x"].max()))
    cols = ["track_id", "frame", "t", "x", "y", "vx", "vy", "speed",
            "yaw", "length", "width", "lane_id", "cls"]
    return Scene(f"highD rec{int(rec['id'])} dir{direction}", road,
                 df[cols].reset_index(drop=True), focal_id, dt)
