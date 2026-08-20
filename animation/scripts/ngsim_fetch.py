"""
ngsim_fetch.py - NGSIM 궤적 데이터를 US DOT ITS DataHub(Socrata) API에서 필요한 만큼만 받아온다.

전체 데이터는 location 하나가 450만 행이라 통째로 받을 이유가 없다.
focal 한 대를 따라가는 애니메이션에는 "그 차가 살아있는 시간창"만 있으면 되므로 3단계로 나눠 받는다.

  1) 차선 기하   : lane_id별 local_x 평균 / local_y 범위        (요청 1회)
  2) 차량 인덱스 : vehicle_id별 프레임 수 / 시간범위            (요청 1회)
  3) 시간창 원본 : focal이 존재하는 구간의 모든 차량 행         (요청 몇 회)

받은 결과는 animation/data/ 아래 csv로 캐시한다(재실행 시 재다운로드 안 함).

사용:
  python ngsim_fetch.py --location us-101           # focal 자동 선택 후 시간창까지 캐시
  python ngsim_fetch.py --location i-80 --list 20   # focal 후보만 출력
"""
import argparse
import io
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

RESOURCE = "https://data.transportation.gov/resource/8ect-6jqj.csv"
FT = 0.3048                       # feet -> meter
DT_MS = 100                       # NGSIM 샘플 주기 (10 Hz)
PAGE = 50000                      # Socrata 한 번에 받을 행 수

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

LOCATIONS = ["us-101", "i-80", "lankershim", "peachtree"]

# NGSIM 4개 구간의 차로 구성(공식 문서 기준). 데이터에서 뽑은 lane별 local_y 범위와도 일치한다.
#   us-101 : 본선 1-5, 6=aux(on-ramp와 off-ramp를 잇는 보조차로), 7=on-ramp, 8=off-ramp
#   i-80   : 본선 1-6, 7=on-ramp
#   arterial 2곳은 교차로가 있어 차로 구성이 구간별로 달라 별도 취급하지 않는다.
LANE_LAYOUT = {
    "us-101": {"main": [1, 2, 3, 4, 5],
               "side": {7: "ON-RAMP", 6: "AUX", 8: "OFF-RAMP"}},
    "i-80": {"main": [1, 2, 3, 4, 5, 6],
             "side": {7: "ON-RAMP"}},
}


def _fetch(params, tries=4):
    """Socrata에 SoQL 질의를 던지고 DataFrame으로 돌려준다."""
    url = RESOURCE + "?" + urllib.parse.urlencode(params)
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return pd.read_csv(io.BytesIO(r.read()))
        except Exception as e:            # 네트워크/스로틀링은 그냥 잠깐 쉬고 재시도
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"NGSIM API 요청 실패: {url}\n{last}")


def _cache(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


# --------------------------------------------------------------------------- 1) 차선 기하
def fetch_lane_geometry(location, refresh=False):
    """lane_id별 (횡방향 중심, 종방향 존재구간)을 구한다. 도로를 그리는 근거가 된다."""
    path = _cache(f"ngsim_{location}_lanes.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    df = _fetch({
        "$select": "lane_id,count(1) as n,avg(local_x) as x_c,min(local_y) as y0,max(local_y) as y1",
        "$where": f"location='{location}'",
        "$group": "lane_id",
        "$order": "lane_id",
        "$limit": 1000,
    })
    # NGSIM은 피트 단위. local_y=종방향(주행방향), local_x=횡방향(좌측 끝에서 잰 거리).
    df["y_center"] = -df["x_c"] * FT      # 화면 좌표: 1차로(좌측)가 위로 오도록 부호 반전
    df["x_start"] = df["y0"] * FT
    df["x_end"] = df["y1"] * FT
    df = df[["lane_id", "n", "y_center", "x_start", "x_end"]]
    df.to_csv(path, index=False)
    print(f"[lanes] {location}: {len(df)}개 차로 -> {path}")
    return df


# --------------------------------------------------------------------------- 2) 차량 인덱스
def fetch_vehicle_index(location, refresh=False):
    """vehicle_id별 프레임 수와 시간범위. focal 후보를 고르는 데 쓴다."""
    path = _cache(f"ngsim_{location}_index.csv")
    if os.path.exists(path) and not refresh:
        cached = pd.read_csv(path)
        if "total_frames" in cached.columns:       # 옛 스키마면 무시하고 다시 받는다
            return cached
        print(f"[index] {location}: 옛 캐시(total_frames 없음) 감지 -> 다시 받음")

    df = _fetch({
        "$select": ("vehicle_id,total_frames,count(1) as n,min(global_time) as t0,"
                    "max(global_time) as t1,min(lane_id) as lane_min,max(lane_id) as lane_max,"
                    "max(v_class) as v_class"),
        "$where": f"location='{location}'",
        "$group": "vehicle_id,total_frames",
        "$order": "vehicle_id",
        "$limit": 200000,
    })
    # NGSIM의 vehicle_id는 15분짜리 녹화 3개에 걸쳐 재사용된다(전역 유일이 아님).
    # total_frames는 "그 녹화 안에서 그 차의 총 프레임 수"라 녹화마다 값이 달라진다.
    # 둘을 묶으면 재사용된 ID도 녹화별 트랙으로 갈라지므로, vehicle_id 하나로 묶을 때처럼
    # 진짜 최장 트랙이 통째로 탈락하는 일이 없어진다.
    span = (df["t1"] - df["t0"]) / DT_MS + 1
    df["contiguous"] = (span - df["n"]).abs() <= 2
    df["duration_s"] = df["n"] * DT_MS / 1000.0
    df.to_csv(path, index=False)
    n_ok = int(df["contiguous"].sum())
    print(f"[index] {location}: 트랙 후보 {len(df)}개 (연속 트랙 {n_ok}개) -> {path}")
    return df


def pick_focal(index, min_lanes_seen=1):
    """구간 안에 가장 오래 머무는 차량을 focal로 고른다."""
    cand = index[index["contiguous"]].copy()
    if min_lanes_seen > 1:                # 차선변경을 요구할 때만 사용
        cand = cand[cand["lane_max"] - cand["lane_min"] >= min_lanes_seen - 1]
    if cand.empty:
        raise RuntimeError("focal 후보가 없습니다.")
    return cand.sort_values("n", ascending=False).iloc[0]


# --------------------------------------------------------------------------- 3) 시간창 원본
COLS = ("vehicle_id,frame_id,global_time,local_x,local_y,v_length,v_width,"
        "v_class,v_vel,lane_id")


def fetch_window(location, t0, t1, refresh=False):
    """[t0, t1] 사이 해당 구간의 모든 차량 행. 이게 애니메이션의 원재료다."""
    path = _cache(f"ngsim_{location}_{int(t0)}_{int(t1)}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    chunks, offset = [], 0
    while True:
        part = _fetch({
            "$select": COLS,
            "$where": f"location='{location}' AND global_time>={int(t0)} AND global_time<={int(t1)}",
            "$order": "global_time,vehicle_id",   # offset 페이징은 정렬이 있어야 안전하다
            "$limit": PAGE,
            "$offset": offset,
        })
        chunks.append(part)
        print(f"  [window] +{len(part)}행 (누적 {sum(len(c) for c in chunks)})")
        if len(part) < PAGE:
            break
        offset += PAGE

    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(path, index=False)
    print(f"[window] {location} {t0}~{t1}: {len(df)}행 -> {path}")
    return df


def prepare(location, track_id=None, pad_s=2.0, refresh=False):
    """차선 기하 + focal이 살아있는 시간창을 통째로 준비해서 경로를 돌려준다."""
    lanes = fetch_lane_geometry(location, refresh)
    index = fetch_vehicle_index(location, refresh)

    if track_id is None:
        row = pick_focal(index)
    else:
        sel = index[index["vehicle_id"] == int(track_id)]
        if sel.empty:
            raise RuntimeError(f"vehicle_id={track_id} 없음")
        if len(sel) > 1:
            print(f"  ! vehicle_id={track_id}는 녹화 {len(sel)}곳에서 재사용된 ID입니다. "
                  f"그중 가장 긴 트랙을 씁니다.")
        ok = sel[sel["contiguous"]]
        row = (ok if not ok.empty else sel).sort_values("n", ascending=False).iloc[0]
        if not row["contiguous"]:
            print(f"  ! 연속성 검사를 통과한 트랙이 없어 시간창이 넓게 잡힙니다.")

    t0 = int(row["t0"]) - int(pad_s * 1000)
    t1 = int(row["t1"]) + int(pad_s * 1000)
    print(f"[focal] vehicle_id={int(row['vehicle_id'])} "
          f"{row['duration_s']:.1f}초 lane {int(row['lane_min'])}~{int(row['lane_max'])}")
    win = fetch_window(location, t0, t1, refresh)
    return lanes, win, int(row["vehicle_id"])


def main():
    p = argparse.ArgumentParser(description="NGSIM 데이터 부분 다운로드")
    p.add_argument("--location", default="us-101", choices=LOCATIONS)
    p.add_argument("--track-id", type=int, default=None)
    p.add_argument("--list", type=int, default=0, help="focal 후보 상위 N개만 출력하고 종료")
    p.add_argument("--pad", type=float, default=2.0, help="focal 앞뒤 여유 시간(초)")
    p.add_argument("--refresh", action="store_true", help="캐시 무시하고 다시 받기")
    a = p.parse_args()

    if a.list:
        idx = fetch_vehicle_index(a.location, a.refresh)
        top = idx[idx["contiguous"]].sort_values("n", ascending=False).head(a.list)
        print(f"\n{'vehicle_id':>10} {'total_frames':>13} {'duration_s':>10} {'lane':>8} {'class':>6}")
        for _, r in top.iterrows():
            print(f"{int(r['vehicle_id']):>10} {int(r['total_frames']):>13} "
                  f"{r['duration_s']:>10.1f} "
                  f"{int(r['lane_min'])}->{int(r['lane_max']):<5} {int(r['v_class']):>6}")
        return

    prepare(a.location, a.track_id, a.pad, a.refresh)


if __name__ == "__main__":
    main()
