"""
animate_focal.py - focal 차량 한 대를 탑뷰로 따라가는 애니메이션을 만든다.

카메라는 도로에 정렬된 채(도로가 항상 수평) focal을 화면 중앙에 고정한다.
화면에는 차선/램프가 그려진 도로, 주변 차량(실제 크기 박스), 하단 HUD가 나온다.

  python animate_focal.py                                  # NGSIM us-101, 가장 오래 머무는 차
  python animate_focal.py --location i-80
  python animate_focal.py --track-id 2113                  # 차선변경 하는 차 지정
  python animate_focal.py --dataset highd --highd-prefix /path/to/01

인코딩은 ffmpeg가 있으면 H.264로, 없으면 OpenCV(mp4v)로 떨어진다.
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import traffic_data as td

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

BG = "#16171b"            # 도로 밖
ASPHALT = "#34363c"
ASPHALT_RAMP = "#2b2d32"
MARK = "#d8d8d0"          # 차선 표시
EDGE_L = "#e0c85a"        # 중앙분리대쪽 실선(미국식 노란 실선)
EDGE_R = "#e8e8e0"
CAR_COLORS = {td.CLS_CAR: "#7d8896", td.CLS_TRUCK: "#b8874a", td.CLS_MOTO: "#69a89c"}
FOCAL = "#ff5c33"
TEXT = "#c9ccd2"
MAX_VEHICLES = 400        # 재사용할 사각형 패치 개수


# --------------------------------------------------------------------------- 도로
def _merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def draw_road(ax, road, pt_per_m):
    """도로는 매 프레임 다시 그릴 필요가 없으므로 한 번만 그리고, 카메라는 xlim만 옮긴다."""
    main = sorted(road.main_lanes, key=lambda l: -l.y_center)
    side = road.side_lanes
    # 3m 표시 / 9m 공백 (미국 기준). matplotlib의 dash는 point 단위라 축척으로 환산한다.
    dash = (0, (3.0 * pt_per_m, 9.0 * pt_per_m))

    for lane in road.lanes:
        ax.add_patch(Rectangle(
            (lane.x_start, lane.y_right), lane.x_end - lane.x_start,
            lane.y_left - lane.y_right,
            facecolor=ASPHALT if lane.kind == "main" else ASPHALT_RAMP,
            edgecolor="none", zorder=1))

    for a, b in zip(main[:-1], main[1:]):          # 본선 차선 (점선)
        y = (a.y_right + b.y_left) / 2
        ax.plot([min(a.x_start, b.x_start), max(a.x_end, b.x_end)], [y, y],
                color=MARK, lw=1.4, ls=dash, zorder=3)

    top, bot = main[0], main[-1]
    ax.plot([top.x_start, top.x_end], [top.y_left] * 2, color=EDGE_L, lw=2.0, zorder=3)

    # 본선 우측 가장자리: 램프/보조차로가 붙어있는 구간은 점선, 나머지는 실선
    spans = _merge_intervals([(l.x_start, l.x_end) for l in side]) if side else []
    edges, x = [], bot.x_start
    for a, b in spans:
        a, b = max(a, bot.x_start), min(b, bot.x_end)
        if b <= a:
            continue
        if a > x:
            edges.append((x, a, False))
        edges.append((max(a, x), b, True))
        x = max(x, b)
    if x < bot.x_end:
        edges.append((x, bot.x_end, False))
    for x0, x1, dashed in edges:
        ax.plot([x0, x1], [bot.y_right] * 2, color=EDGE_R, lw=1.8,
                ls=dash if dashed else "-", zorder=3)

    if side:                                        # 램프/보조차로 대역: 바깥 실선 + 양끝 마감
        yl, yr = side[0].y_left, side[0].y_right
        for a, b in spans:
            ax.plot([a, b], [yr, yr], color=EDGE_R, lw=1.8, zorder=3)
            for xe in (a, b):
                ax.plot([xe, xe], [yr, yl], color=EDGE_R, lw=1.4, zorder=3)
        for lane in side:
            ax.text((lane.x_start + lane.x_end) / 2, lane.y_center, lane.label,
                    color="#9aa0aa", fontsize=7.5, ha="center", va="center", zorder=4)


def _find_ffmpeg():
    """PATH에 없더라도 지금 쓰는 파이썬 옆(conda 환경 bin)은 한 번 더 본다."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    cand = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    return cand if os.path.exists(cand) else None


class VideoOut:
    """프레임(RGBA 배열)을 받아 mp4로 인코딩한다.

    ffmpeg가 있으면 원시 프레임을 파이프로 밀어넣어 H.264로 뽑는다. 없으면 OpenCV의
    mp4v로 떨어지는데, 같은 화면이라도 mp4v는 파일이 3배쯤 커진다.
    """

    def __init__(self, path, w, h, fps, crf=20):
        self.w, self.h, self.proc, self.cv = w, h, None, None
        exe = _find_ffmpeg()
        if exe:
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", f"{fps:g}",
                 "-i", "-", "-an",
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", path],
                stdin=subprocess.PIPE)
            self.kind = f"ffmpeg H.264 (crf {crf})"
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not self.cv.isOpened():
                raise RuntimeError("cv2.VideoWriter 열기 실패")
            self.kind = "OpenCV mp4v (ffmpeg 없음)"

    def write(self, rgba):
        a = rgba[:self.h, :self.w]
        if self.proc is not None:
            self.proc.stdin.write(a.tobytes())     # tobytes()는 항상 C 순서라 슬라이스여도 안전
        else:
            self.cv.write(cv2.cvtColor(a, cv2.COLOR_RGBA2BGR))

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg 인코딩 실패")
        else:
            self.cv.release()


def set_rect(rect, x, y, length, width, yaw):
    """중심/헤딩으로 회전 사각형을 배치한다. Rectangle은 xy(=좌하단)를 축으로 회전한다."""
    c, s = np.cos(yaw), np.sin(yaw)
    corner_x = x - 0.5 * length * c + 0.5 * width * s
    corner_y = y - 0.5 * length * s - 0.5 * width * c
    rect.set_xy((corner_x, corner_y))
    rect.set_width(length)
    rect.set_height(width)
    rect.set_angle(np.degrees(yaw))


# --------------------------------------------------------------------------- 본체
def render(scene, ahead, behind, margin, fig_w, dpi, out_path, speed,
           start_seconds, max_seconds, gif, crf=20):
    df = scene.tracks
    focal_all = df[df["track_id"] == scene.focal_id].sort_values("frame")
    if focal_all.empty:
        raise RuntimeError("focal 트랙이 비어 있습니다.")

    f0, f1 = int(focal_all["frame"].iloc[0]), int(focal_all["frame"].iloc[-1])
    if start_seconds:
        f0 = min(f1, f0 + int(start_seconds / scene.dt))
    if max_seconds:
        f1 = min(f1, f0 + int(max_seconds / scene.dt))
    frames = list(range(f0, f1 + 1))

    by_frame = {f: g for f, g in df[df["frame"].between(f0, f1)].groupby("frame")}
    focal_by_frame = focal_all.set_index("frame")

    # --- 화면 크기: x범위와 y범위의 비율을 축 상자 비율과 똑같이 잡아 축척을 1:1로 만든다
    x_span = ahead + behind
    y_span = (scene.road.y_top - scene.road.y_bottom) + 2 * margin
    y_mid = (scene.road.y_top + scene.road.y_bottom) / 2

    pad = 0.015
    ax_w_in = fig_w * (1 - 2 * pad)
    ax_h_in = ax_w_in * y_span / x_span
    hud_in, top_in = 0.72, 0.34
    fig_h = ax_h_in + hud_in + top_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([pad, hud_in / fig_h, 1 - 2 * pad, ax_h_in / fig_h])
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylim(y_mid - y_span / 2, y_mid + y_span / 2)

    pt_per_m = ax_w_in * 72.0 / x_span
    draw_road(ax, scene.road, pt_per_m)

    pool = []
    for _ in range(MAX_VEHICLES):
        r = Rectangle((0, 0), 0, 0, facecolor=CAR_COLORS[td.CLS_CAR],
                      edgecolor="#0f1013", lw=0.6, zorder=5, visible=False)
        ax.add_patch(r)
        pool.append(r)
    focal_rect = Rectangle((0, 0), 0, 0, facecolor=FOCAL, edgecolor="white",
                           lw=1.4, zorder=8)
    ax.add_patch(focal_rect)

    fig.text(pad, 1 - top_in / fig_h * 0.62, "", color=TEXT, fontsize=10,
             va="center", ha="left").set_text(
        f"{scene.name}   focal={scene.focal_id}   view {behind:.0f}m back / {ahead:.0f}m ahead")
    hud = fig.text(pad, hud_in / fig_h * 0.42, "", color=TEXT, fontsize=10.5,
                   va="center", ha="left", family="monospace")
    for k, (lab, col) in enumerate([("moto", CAR_COLORS[td.CLS_MOTO]),
                                    ("truck", CAR_COLORS[td.CLS_TRUCK]),
                                    ("car", CAR_COLORS[td.CLS_CAR]),
                                    ("focal", FOCAL)]):
        fig.text(1 - pad - k * 0.045, hud_in / fig_h * 0.42, lab, color=col,
                 fontsize=9.5, va="center", ha="right", family="monospace")

    writer, gif_frames = None, []
    out_fps = max(1.0, round(1.0 / scene.dt * speed))

    for n, f in enumerate(frames):
        fr = focal_by_frame.loc[f]
        fx, fy = float(fr["x"]), float(fr["y"])
        ax.set_xlim(fx - behind, fx + ahead)

        cur = by_frame.get(f)
        others = cur[cur["track_id"] != scene.focal_id]
        others = others[(others["x"] > fx - behind - 15) & (others["x"] < fx + ahead + 15)]

        k = 0
        for _, v in others.iterrows():
            if k >= MAX_VEHICLES:
                break
            r = pool[k]
            set_rect(r, v["x"], v["y"], v["length"], v["width"], v["yaw"])
            r.set_facecolor(CAR_COLORS.get(v["cls"], CAR_COLORS[td.CLS_CAR]))
            r.set_visible(True)
            k += 1
        for r in pool[k:]:
            r.set_visible(False)
        set_rect(focal_rect, fx, fy, fr["length"], fr["width"], fr["yaw"])

        gap = lead_gap(others, fr)
        hud.set_text(
            f"t={float(fr['t']):6.1f}s   speed={float(fr['speed'])*3.6:5.1f} km/h   "
            f"lane={int(fr['lane_id'])}   x={fx:6.1f}m   neighbors={k:3d}   "
            f"gap={'--' if gap is None else f'{gap:5.1f}m'}")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())

        if writer is None:
            h, w = rgba.shape[:2]
            w, h = w - w % 2, h - h % 2          # H.264/mp4v 모두 짝수 해상도를 요구한다
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            writer = VideoOut(out_path, w, h, out_fps, crf)
            print(f"[render] {w}x{h} @ {out_fps:g}fps, {len(frames)} frames | {writer.kind}")
        writer.write(rgba)
        if gif and n % 3 == 0:
            gif_frames.append(rgba[::2, ::2, :3].copy())

        if n % 100 == 0:
            print(f"  frame {n}/{len(frames)}")

    writer.close()
    plt.close(fig)
    print(f"saved -> {out_path}  ({len(frames)*scene.dt:.1f}s / {len(frames)/out_fps:.1f}s 재생)")

    if gif:
        from PIL import Image
        gp = out_path.replace(".mp4", ".gif")
        imgs = [Image.fromarray(a) for a in gif_frames]
        imgs[0].save(gp, save_all=True, append_images=imgs[1:],
                     duration=int(1000 * 3 / out_fps), loop=0, optimize=True)
        print(f"saved -> {gp}")


def lead_gap(others, fr):
    """같은 차로에서 바로 앞차와의 범퍼 간격[m]."""
    same = others[others["lane_id"] == fr["lane_id"]]
    ahead_v = same[same["x"] > fr["x"]]
    if ahead_v.empty:
        return None
    lead = ahead_v.loc[ahead_v["x"].idxmin()]
    return float((lead["x"] - lead["length"] / 2) - (fr["x"] + fr["length"] / 2))


def main():
    p = argparse.ArgumentParser(description="focal 추종 탑뷰 애니메이션")
    p.add_argument("--dataset", default="ngsim", choices=["ngsim", "highd"])
    p.add_argument("--location", default="us-101", help="NGSIM 구간")
    p.add_argument("--highd-prefix", default=None, help="예: /data/highd/data/01")
    p.add_argument("--track-id", type=int, default=None)
    p.add_argument("--ahead", type=float, default=75.0)
    p.add_argument("--behind", type=float, default=65.0)
    p.add_argument("--margin", type=float, default=2.5, help="도로 위아래 여백[m]")
    p.add_argument("--fig-width", type=float, default=15.0)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    p.add_argument("--start-seconds", type=float, default=0, help="트랙 시작 후 N초부터")
    p.add_argument("--max-seconds", type=float, default=0, help="0이면 트랙 전체")
    p.add_argument("--smooth", type=int, default=11, help="위치 스무딩 창(프레임), 1=끔")
    p.add_argument("--crf", type=int, default=20,
                   help="H.264 화질 (낮을수록 고화질/큰 파일, 18~28 권장)")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.dataset == "ngsim":
        import ngsim_fetch
        lanes, win, vid = ngsim_fetch.prepare(a.location, a.track_id)
        scene = td.load_ngsim(lanes, win, a.location, vid, a.smooth)
        name = f"ngsim_{a.location}_v{vid}"
    else:
        if not a.highd_prefix:
            p.error("--highd-prefix 가 필요합니다 (예: /data/highd/data/01)")
        scene = td.load_highd(a.highd_prefix, a.track_id, smooth_win=a.smooth)
        name = f"highd_{os.path.basename(a.highd_prefix)}_v{scene.focal_id}"

    out = a.out or os.path.join(OUT_DIR, name + ".mp4")
    print(f"[scene] {scene.name}  트랙 {scene.tracks['track_id'].nunique()}개  "
          f"프레임 {scene.tracks['frame'].nunique()}개  focal={scene.focal_id}")
    render(scene, a.ahead, a.behind, a.margin, a.fig_width, a.dpi,
           out, a.speed, a.start_seconds, a.max_seconds, a.gif, a.crf)


if __name__ == "__main__":
    main()
