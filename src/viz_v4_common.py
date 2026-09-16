"""
viz_v4_common.py - v4 시각화·분석 스크립트(viz_v4_*.py)가 함께 쓰는 상수·로더·그리기 도구.

기존 파일은 import 만 한다(수정하지 않는다). 지도 그리기는 hdmap_render.py 의 규칙
(차로 면 + 실제 노면표시)을 **밝은 배경용으로** 옮겨 왔다 — 과제가 '과거 회색 · 정답 검정' 을
요구하는데 기존 렌더러는 어두운 배경이라 검정 선이 안 보이기 때문이다.

정의·임계값은 전부 여기 한 곳에 둔다. 덤프·그림·리포트가 같은 숫자를 쓰게 하려는 것이다.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_ROOT = Path("/data/argoverse2/motion_forecasting")
VAL_DIR = DATA_ROOT / "val"
VAL_CACHE = Path("/data/argoverse2/cache/v4/val_e67373005962be8a_n24988")
RUNS = ROOT / "runs"
VIZ_ROOT = ROOT / "viz" / "v4"
DEFAULT_TAG = "v4_l4nw_ah2_full_sm1_s0"
SEED = 0
DT = 0.1
OBS, FUT = 50, 60                      # 관측 50스텝, 예측 60스텝 (t=49 가 예측 시작 = 0 s)
T_AX = (np.arange(OBS + FUT) - (OBS - 1)) * DT      # 110스텝 시간축 [s], t=49 -> 0
T_PRED = np.arange(1, FUT + 1) * DT                 # 예측 스텝 k -> (k+1)·0.1 s

# ---------------------------------------------------------------- 지표
# 손실 가중치는 상수로 두지 않고 runs/<tag>.json 의 args(offlane · off_nonwinner · smooth)를 쓴다.
MISS_M = 2.0         # miss = minFDE6 > 2 m
LABEL_DTHETA_DEG = 7.3

# ---------------------------------------------------------------- 정답 기반 상황 분류
# 우선순위는 이 리스트 순서다. 앞에서 걸리면 뒤는 안 본다(상호배타).
CLASSES = ["정지", "좌회전", "우회전", "좌차선변경", "우차선변경", "급감속", "급가속", "정속", "기타"]
STOP_VMAX = 1.0      # [m/s]  미래 6초 최대 속력 < 이면 정지
TURN_DEG = 30.0      # [deg]  6초 순 진행방향 변화 |Δh| > 이면 회전 (+ 좌)
LC_MAX_DEG = 15.0    # [deg]  |Δh| < 이면서
LC_LAT_M = 2.5       # [m]    정답 기준 경로의 횡변위 |Δd| > 이면 차선변경 (+ 좌)
LC_MAX_ABS_D = 5.0   # [m]    차선변경은 기준 경로에서 이 이상 멀어지지 않는다 (넘으면 경로가 정답을 안 따라간 것)
LC_MIN_MOVE = 10.0   # [m]    6초 이동거리가 이 이상일 때만 차선변경으로 본다 (저속 횡이동·주차 제외)
ROUTE_ALIGN_DEG = 30.0  # [deg] 움직인 스텝에서 |θ_정답| 최대가 이 미만이면 '진행방향과 나란한 경로'
MOVE_V = 1.0         # [m/s]  진행방향 기반 양(Δh, θ)은 이 속력 이상인 스텝만 쓴다 (저속 방향은 잡음)
DEC_A = -3.0         # [m/s²] 미래 최소 가속도 < 이면 급감속
ACC_A = 2.5          # [m/s²] 미래 최대 가속도 > 이면 급가속
CONST_DV = 1.5       # [m/s]  |v_end − v0| < 이고
CONST_V0 = 2.0       # [m/s]  v0 > 이면 정속
# 종방향 신호: AV2 velocity 필드의 속력 -> median 5 -> Savitzky–Golay(15스텝=1.5 s, 2차).
# 위치 차분을 쓰지 않는 이유: AV2 위치는 11초 창 양 끝 약 0.6 s 에 인공 감속이 있다
# (끝 스텝 속력이 실제의 약 46%). 위치로 가속도를 내면 이동 시나리오의 약 80% 가 '급감속' 이 된다.
MED_K = 5
SG_WIN, SG_POLY = 15, 2
HEAD_AVG = 3         # Δh 양 끝을 이 스텝 수로 평균 (t=49..51 / t=106..108)

# ---------------------------------------------------------------- 앞차 정의
LEAD_TYPES = ("VEHICLE", "BUS", "MOTORCYCLIST")
LEAD_LAT_M = 1.8     # focal 이 실제로 달린 경로(110스텝 + 끝 접선 연장)에서의 횡거리 |lat| <
LEAD_MIN_M = 2.5     # 중심 간 거리가 이보다 가까우면 겹친 추적(중복 트랙)으로 보고 제외
LEAD_MAX_M = 80.0    # 경로를 따라 앞쪽 0 < Δs <= 이 거리
LEAD_HEAD_DEG = 45.0 # 그 지점 경로 접선과 진행방향 차 <
LEAD_EXT_M = 100.0   # 경로 끝 접선 방향 연장 길이 (정지·저속 focal 도 앞을 볼 수 있게)
PATH_STEP_M = 1.0    # 기준 경로 리샘플 간격
THW_MIN_V = 0.5      # focal 속력이 이보다 낮으면 시간간격을 정의하지 않는다('정지 중')

# ---------------------------------------------------------------- 구간·표본 기준
MIN_N = 50           # 칸·막대의 표본이 이보다 적으면 흐리게 그린다
BINS = {
    "v0": ([0, 1, 5, 10, 15, 20, np.inf], ["<1", "1–5", "5–10", "10–15", "15–20", "≥20"]),
    "amax": ([0, 1, 2, 3, 4, np.inf], ["<1", "1–2", "2–3", "3–4", "≥4"]),
    "dh": ([0, 5, 15, 30, 60, 90, np.inf], ["<5", "5–15", "15–30", "30–60", "60–90", "≥90"]),
    "thw": ([0, 1, 2, 4, np.inf], ["<1 s", "1–2 s", "2–4 s", "≥4 s"]),
}

# ---------------------------------------------------------------- 색 (dataviz 기본 팔레트, 밝은 면)
SURF = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
SERIES = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
C_WIN = C_ORANGE     # 승자(끝점 오차 최소) 모드
C_TOP = "#184f95"    # 확률 1위 모드 (파랑 램프의 짙은 쪽 — 지도에서도 확률이 가장 크면 가장 짙다)
C_GT = INK
C_PAST = "#8f8d87"
C_LEAD = C_VIOLET
C_BAND = C_AQUA


def setup_mpl():
    """Agg 백엔드 + 한글 폰트 + 차분한 축. 모든 그림 스크립트가 맨 처음 부른다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": "Noto Sans CJK KR",
        "axes.unicode_minus": False,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
    })
    return plt


def savefig(fig, path, dpi=140):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 경로·태그
def tag_dirs(tag):
    base = VIZ_ROOT / tag
    return {"base": base, "data": base / "data", "cases": base / "cases",
            "stats": base / "stats", "why": base / "why", "div": base / "diversity"}


def run_args(tag):
    d = json.loads((RUNS / f"{tag}.json").read_text())
    return d["args"], d


def training_running():
    """공유 머신 규칙: 학습이 돌면 GPU 를 쓰지 않는다."""
    r = subprocess.run(["pgrep", "-f", "src/train_v4[.]py"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def pick_device(want="auto"):
    import torch
    if want != "auto":
        return want
    if training_running():
        print("[device] train_v4 가 돌고 있다 -> CPU 로 추론한다", flush=True)
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(tag, device):
    """runs/<tag>.json 의 args 와 체크포인트에서 V4Net 을 복원한다 (compare_v4_full.py 와 같은 방식)."""
    import torch
    from model_v4 import V4Net
    args, meta = run_args(tag)
    sd = torch.load(RUNS / f"lstm_{tag}.pth", map_location="cpu")
    in_dim = sd["traj_encoder.weight_ih_l0"].shape[1]
    lane_in = sd["lane_encoder.0.weight"].shape[1]
    th0 = args.get("th0", "current")
    m = V4Net(in_dim=in_dim, lane_in=lane_in, level="l0", th0_mode=th0).to(device)
    m.load_state_dict(sd)
    m.eval()
    info = {"in_dim": int(in_dim), "lane_in": int(lane_in), "th0": th0,
            "offlane": float(args.get("offlane", 0.0)), "off_nonwinner": bool(args.get("off_nonwinner", 0)),
            "smooth": float(args.get("smooth", 0.0)), "input": args.get("input", "raw5"),
            "fallback": args.get("fallback"), "train_n": args.get("limit")}
    return m, info


def expected_scores(tag):
    """runs/v4_full_compare.json 의 val 24,988 재채점 값 (재현 확인 기준)."""
    p = RUNS / "v4_full_compare.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["runs"].get(tag, {}).get("val24988")


def git_head():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def route_dict(pts, tan, length):
    """캐시의 64점 등간격 경로 -> lane_frame.to_frame 이 받는 dict."""
    n = len(pts)
    return {"pts": np.asarray(pts, np.float64), "tan": np.asarray(tan, np.float64),
            "s": np.linspace(0.0, float(length), n)}


def wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def bin_labels(x, key):
    edges, labels = BINS[key]
    idx = np.digitize(np.asarray(x, float), edges[1:-1], right=False)
    out = np.array(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)]
    out[~np.isfinite(np.asarray(x, float))] = None
    return out


# ---------------------------------------------------------------- 밝은 지도 렌더러 (hdmap_render 규칙)
L_BG = "#f4f3ef"          # 도로 밖
L_DRIVE = "#e6e4dd"       # 주행가능영역
L_LANE = "#dcdad2"        # 차로 면
L_LANE_INT = "#d2cfc6"    # 교차로 안 차로
L_BIKE = "#d9e3dc"
L_CURB = "#bdbab0"
L_MARK_W = "#9d9a91"      # 흰 노면표시 -> 밝은 배경에서는 회색으로
L_MARK_Y = "#c99a1c"      # 노란 노면표시
L_MARK_B = "#6d9fcc"
L_CROSS = "#c9c6bd"


def build_scene(sid, origin, theta):
    import hdmap_render as hr
    return hr.build_scene(VAL_DIR / sid / f"log_map_archive_{sid}.json",
                          np.asarray(origin, np.float32), float(theta))


def draw_scene_light(ax, scene, xlim, ylim, mark_lw=1.0):
    """hdmap_render.draw_scene 와 같은 순서(주행면 -> 차로 면 -> 횡단보도 -> 노면표시), 밝은 색."""
    import hdmap_render as hr
    from matplotlib.patches import Polygon as MplPolygon
    pad = 0.1 * (xlim[1] - xlim[0])
    box = (xlim[0] - pad, xlim[1] + pad, ylim[0] - pad, ylim[1] + pad)
    ax.set_facecolor(L_BG)
    for poly in scene.drivable:
        if hr._visible(poly, box):
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=L_DRIVE, edgecolor=L_CURB,
                                    lw=0.8, zorder=1))
    for poly, color in scene.lanes:
        if hr._visible(poly, box):
            fc = {hr.BIKE: L_BIKE, hr.ASPHALT_INT: L_LANE_INT}.get(color, L_LANE)
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=fc, edgecolor="none", zorder=2))
    for e1, e2 in scene.crossings:
        if not hr._visible(np.concatenate([e1, e2]), box):
            continue
        n = max(2, int(np.linalg.norm(e1[1] - e1[0]) / 0.9))
        t = np.linspace(0, 1, 2 * n + 1)
        p1 = e1[0] + np.outer(t, e1[1] - e1[0])
        p2 = e2[0] + np.outer(t, e2[1] - e2[0])
        for i in range(0, 2 * n, 2):
            quad = np.array([p1[i], p1[i + 1], p2[i + 1], p2[i]])
            ax.add_patch(MplPolygon(quad, closed=True, facecolor=L_CROSS, edgecolor="none",
                                    alpha=0.7, zorder=3))
    cmap = {hr.MARK_W: L_MARK_W, hr.MARK_Y: L_MARK_Y, hr.MARK_B: L_MARK_B}
    for bnd, name in scene.marks:
        if not hr._visible(bnd, box):
            continue
        for style, color, off in hr._MARK_STYLE[name]:
            line = hr._offset(bnd, off)
            pieces = hr._dashes(line) if style == "dash" else [line]
            for p in pieces:
                ax.plot(p[:, 0], p[:, 1], color=cmap.get(color, L_MARK_W), lw=mark_lw,
                        solid_capstyle="butt", zorder=4)


def draw_box(ax, x, y, yaw, color, length=4.6, width=1.9, alpha=1.0, zorder=9, ec="white", lw=0.8):
    from matplotlib.patches import Polygon as MplPolygon
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2, width / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    R = np.array([[c, -s], [s, c]])
    ax.add_patch(MplPolygon(corners @ R.T + [x, y], closed=True, facecolor=color, edgecolor=ec,
                            lw=lw, alpha=alpha, zorder=zorder))


def band_polygon(pts, tan, band):
    """경로 좌우 밴드 폐다각형 (visualize_v4_modes.band_polygon 과 같은 정의)."""
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    left = pts + nrm * band[:, 0:1]
    right = pts - nrm * band[:, 1:2]
    return np.concatenate([left, right[::-1]], axis=0)


def scale_bar(ax, xlim, ylim, color=INK2):
    span = xlim[1] - xlim[0]
    length = min([m for m in (5, 10, 20, 50, 100, 200) if m >= span / 6], default=200)
    x0 = xlim[0] + 0.05 * span
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=1.8, zorder=30, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.015 * span, f"{length} m", color=color, fontsize=7.5,
            ha="center", va="bottom", zorder=30)


def prob_color(p):
    """확률 -> 파랑 램프 색 (서수 램프라 밝은 쪽은 step 250 부터)."""
    ramp = BLUE_RAMP[3:]
    i = int(np.clip(round(float(p) * (len(ramp) - 1)), 0, len(ramp) - 1))
    return ramp[i]


def faded(n):
    return n < MIN_N


def annotate_n(ax, xs, ns, y=None, fontsize=7.2):
    """막대·상자 위(또는 축 아래)에 표본 수를 적는다."""
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.transData, ax.transAxes)
    for x, n in zip(xs, ns):
        ax.text(x, 1.0 if y is None else y, f"n={int(n):,}", transform=tr, ha="center", va="bottom",
                fontsize=fontsize, color=MUTED if faded(n) else INK2)
