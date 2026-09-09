"""
visualize_v4_modes.py - 학습된 v4 의 예측 6개가 **어느 후보 경로에 붙어 있는지**를 그린다.

v4 는 예측 모드를 '학습된 익명 슬롯 K=6' 에서 '지도에서 열거된 후보 경로' 로 바꾼 설계다.
그래서 이 그림이 답해야 하는 질문은 minADE 가 몇이냐가 아니라 두 가지다:

  ① 지도가 이 장면에서 **구별되는 분기를 몇 개** 주었나 (n_distinct)
     — 분기가 1개면 모드 6개는 전부 같은 경로 위의 종방향(속도) 다중성이다.
  ② 예측이 그 경로의 **밴드(허용 횡오프셋) 안에 있나**
     — level="l0" 는 출력이 (a, theta) 이고 Frenet 적분으로 궤적을 만들므로, 궤적이 어디까지
       갈 수 있는지가 곧 밴드다. 밴드를 칠해 두면 '출력 공간이 무엇을 허용하는가'가 눈에 보인다.
       (L4 벌점 없이 학습하면 밴드는 아직 **경계가 아니라 기준선**이다 — 벗어난 step 수를
        칸 제목에 같이 적어 두는 이유가 그것이다.)

  python src/visualize_v4_modes.py --ckpt runs/lstm_v4_l0_s0.pth --level l0 --n 9 --out v4_modes.png
  python src/visualize_v4_modes.py --level l3 --limit 60        # 체크포인트 없이 레이아웃만 확인
"""
import argparse, sys
from pathlib import Path
sys.path.append("src")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from torch.utils.data import DataLoader

try:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

import hdmap_render as hr
from dataset_lane import Av2LaneRuleDataset
from model_v4 import V4Net, N_PTS, N_RULE

DATA_ROOT = "/data/argoverse2/motion_forecasting"
MAP = Path(DATA_ROOT) / "val"
# train_v4.py 와 같은 목록이지만 여기서 다시 적는다 — 시각화가 학습 스크립트의
# import 에 묶이면 학습 쪽을 건드릴 때마다 그림이 같이 죽는다.
ROUTE_KEYS = ("routes", "route_tan", "route_band", "route_len", "route_sd0",
              "route_mask", "route_sub", "v0")

PAST, GT, PRED, BEST = "#c9ccd2", "#31b0ff", "#ffa53d", "#5ce08a"
ROUTE, ROUTE_HI, BAND = "#7e858f", "#b9c0ca", "#4f7fa8"


def _o(lw):
    """궤적이 노면표시와 섞이지 않도록 어두운 테두리를 두른다 (visualize_map.py 와 동일)."""
    return [pe.Stroke(linewidth=lw + 1.8, foreground="#101115"), pe.Normal()]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def turn_deg(gt):
    """정답 궤적의 회전각 [deg]. postprocess_rules.py 의 gt_turn_deg 와 같은 정의."""
    d0, d1 = _unit(gt[2] - gt[0]), _unit(gt[-1] - gt[-3])
    # np.cross 는 2차원 벡터에서 deprecated 라 외적 z 성분을 직접 쓴다
    return float(np.degrees(np.arctan2(d0[0] * d1[1] - d0[1] * d1[0], np.dot(d0, d1))))


def band_offset(traj, pts, tan, band):
    """예측 (T,2) 을 경로에 투영해 (횡오프셋 d, 그 지점의 밴드) 를 낸다 -> (T,), (T,2).

    l0 은 적분기가 d 를 상태로 직접 들고 있어 이 함수가 필요 없지만, l3 는 위치를 자유
    회귀하므로 '경로 기준으로 얼마나 벗어났나'를 밖에서 재야 한다. 두 레벨의 칸 제목에
    같은 숫자를 적으려면 재는 방법이 하나여야 해서 투영을 쓴다.
    경로는 호길이 등간격이라 최근접 점 인덱스만 찾으면 되고, 부호는 접선 기준 외적으로 낸다.
    """
    j = np.argmin(((traj[:, None, :] - pts[None]) ** 2).sum(-1), axis=1)
    r = traj - pts[j]
    d = tan[j, 0] * r[:, 1] - tan[j, 1] * r[:, 0]      # 좌측 법선 성분 = d
    return d, band[j]


def n_out_of_band(d, band):
    """밴드 밖으로 나간 step 수. 좌/우 상한이 다르므로 양쪽을 따로 본다."""
    return int(((d > band[..., 0]) | (-d > band[..., 1])).sum())


def band_polygon(pts, tan, band):
    """경로 좌우 밴드를 채울 폐다각형. 좌측 상한은 +N, 우측 상한은 -N 쪽이다."""
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    left = pts + nrm * band[:, 0:1]
    right = pts - nrm * band[:, 1:2]
    return np.concatenate([left, right[::-1]], axis=0)


def collect(model, ds, args, device):
    """풀 전체를 한 번 훑어 그림에 필요한 것만 CPU 로 모은다."""
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    rows = []
    for b in dl:
        kw = dict(x=b["x"].to(device), lanes=b["lanes"].to(device),
                  lane_mask=b["lane_mask"].to(device),
                  lane_feat=b["lane_feat"].to(device) if "lane_feat" in b else None)
        kw.update({k: b[k].to(device) for k in ROUTE_KEYS})
        with torch.no_grad():
            traj, logits, aux = model(**kw)
        y = b["y"].to(device)
        mm = b["route_mask"].to(device)
        dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
        big = torch.full_like(dist[:, :, 0], float("inf"))
        ade = torch.where(mm > 0, dist.mean(2), big)           # 없는 경로는 승자가 될 수 없다
        for i in range(y.size(0)):
            r = {"sid": b["scenario_id"][i],
                 "hist": b["x"][i, :, :2].numpy(),
                 "gt": b["y"][i].numpy(),
                 "pred": traj[i].cpu().numpy(),
                 "ade": ade[i].cpu().numpy(),
                 "prob": torch.softmax(logits[i], 0).cpu().numpy(),
                 "origin": b["origin"][i].numpy(),
                 "theta": float(b["theta"][i]),
                 "n_dist": int(b["n_distinct"][i]),
                 "fallback": float(b["route_fallback"][i]) > 0.5}
            for k in ("routes", "route_tan", "route_band", "route_mask"):
                r[k] = b[k][i].numpy()
            if aux:                       # l0: 적분기가 들고 있는 d/band 가 정답이다
                r["d"] = aux["d"][i].cpu().numpy()
                r["band"] = aux["band"][i].cpu().numpy()
            else:                         # l3: 밖에서 투영해 잰다
                dd, bb = [], []
                for m in range(r["pred"].shape[0]):
                    a, c = band_offset(r["pred"][m], r["routes"][m], r["route_tan"][m],
                                       r["route_band"][m])
                    dd.append(a); bb.append(c)
                r["d"], r["band"] = np.stack(dd), np.stack(bb)
            r["turn"] = turn_deg(r["gt"])
            r["disp"] = float(np.linalg.norm(r["gt"][-1] - r["gt"][0]))
            r["best"] = int(np.argmin(r["ade"]))
            r["minade"] = float(r["ade"][r["best"]])
            rows.append(r)
    return rows


def pick(rows, args):
    """좌회전 / 우회전 / 직진을 섞어 고른다 (visualize_v4_lane.py 와 같은 방식).

    회전 케이스는 **분기가 실제로 갈라진 장면**을 우선한다 — n_distinct 가 1 이면
    '모드가 후보 경로'라는 이 그림의 주장 자체를 보일 수 없다.
    이동거리 하한을 두는 이유: 회전각을 궤적 양끝 방향차로 재는데 정지 차량은 그 방향이
    노이즈라, 조건이 없으면 '안 움직인 차'가 좌회전으로 뽑힌다.
    """
    ok = np.array([(not r["fallback"]) and r["disp"] >= args.min_disp for r in rows])
    turn = np.array([r["turn"] for r in rows])
    ndist = np.array([r["n_dist"] for r in rows])
    minade = np.array([r["minade"] for r in rows])
    disp = np.array([r["disp"] for r in rows])
    nl, nr, ns = (int(v) for v in args.mix.split(","))

    picks = []
    for lab, mask, k in [("좌회전", ok & (turn > 30) & (ndist >= 2), nl),
                         ("우회전", ok & (turn < -30) & (ndist >= 2), nr),
                         ("직진", ok & (np.abs(turn) < 10), ns)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            pass
        elif lab == "직진":     # 직진은 평균에 가까운 것 (대표 케이스)
            idx = idx[np.argsort(np.abs(minade[idx] - np.median(minade[ok])))]
        else:                   # 회전은 분기가 많고 많이 움직인 것부터 (그림이 잘 보인다)
            idx = idx[np.lexsort((-disp[idx], -ndist[idx]))]
        idx = [int(i) for i in idx if int(i) not in picks][:k]
        picks += list(idx)
        print(f"  {lab:4} 후보 {int(mask.sum()):3d}개 중 {len(idx)}개 선택"
              f"  분기 {ndist[idx].tolist()}  이동 {np.round(disp[idx], 0).tolist()} m")
    if len(picks) < args.n:      # 부족하면 분기가 많은 순으로 채운다
        rest = [int(i) for i in np.lexsort((-disp, -ndist)) if ok[i] and int(i) not in picks]
        print(f"  보충 {min(args.n - len(picks), len(rest))}개 (분기 많은 순)")
        picks += rest[:args.n - len(picks)]
    return picks[:args.n]


def draw_cell(ax, r, level, show_band):
    sid = r["sid"]
    scene = hr.build_scene(MAP / sid / f"log_map_archive_{sid}.json", r["origin"], r["theta"])
    valid = np.where(r["route_mask"] > 0)[0]
    nd = max(r["n_dist"], 1)

    # 시야는 궤적으로만 잡는다 — 경로는 100 m 넘게 뻗어 있어 넣으면 장면이 통째로 작아진다.
    xy = np.concatenate([r["hist"], r["gt"], r["pred"][valid].reshape(-1, 2)], axis=0)
    xy = xy[~np.isnan(xy).any(1)]
    lo, hi = xy.min(0), xy.max(0)
    cx, cy = (lo + hi) / 2
    hw = max(*(hi - lo) * 0.62, 30.0)
    xlim, ylim = (cx - hw, cx + hw), (cy - hw, cy + hw)
    hr.draw_scene(ax, scene, xlim, ylim, mark_lw=1.4)

    # ① 밴드 — 출력 공간이 허용하는 폭. 중복 슬롯은 같은 경로라 구별되는 것만 칠한다.
    if show_band:
        for k in valid[:nd]:
            poly = band_polygon(r["routes"][k], r["route_tan"][k], r["route_band"][k])
            ax.fill(poly[:, 0], poly[:, 1], color=BAND, alpha=0.17, lw=0, zorder=2.6)
    # ② 후보 경로 중심선. best 가 탄 경로만 밝게 해서 조건이 무엇이었는지 보이게 한다.
    bj = r["best"] % nd
    for k in valid[:nd]:
        p = r["routes"][k]
        hi_k = (k == bj)
        ax.plot(p[:, 0], p[:, 1], color=ROUTE_HI if hi_k else ROUTE,
                lw=1.9 if hi_k else 1.3, alpha=0.95 if hi_k else 0.6, zorder=4.5)

    # ③ 예측 6개. 각 예측의 끝점에 붙은 경로 번호를 적고, 그 경로까지 가는 실을 그린다.
    for m in valid:
        t, is_best = r["pred"][m], m == r["best"]
        col = BEST if is_best else PRED
        ax.plot(t[:, 0], t[:, 1], color=col, lw=2.4 if is_best else 2.1,
                ls=(0, (4.5, 3.5)) if is_best else "-", alpha=1.0 if is_best else 0.9,
                zorder=13 if is_best else 5, path_effects=_o(2.4) if is_best else None)
        rj = m % nd
        q = r["routes"][rj]
        j = int(np.argmin(((q - t[-1]) ** 2).sum(1)))
        ax.plot([t[-1, 0], q[j, 0]], [t[-1, 1], q[j, 1]], color=col, lw=0.8,
                ls=":", alpha=0.75, zorder=6)
        ax.text(t[-1, 0], t[-1, 1], f"{rj}", color=col, fontsize=7.5, zorder=14,
                ha="center", va="center", path_effects=_o(0.6))

    ax.plot(r["hist"][:, 0], r["hist"][:, 1], color=PAST, lw=2.4, zorder=7,
            path_effects=_o(2.4))
    ax.plot(r["gt"][:, 0], r["gt"][:, 1], color=GT, lw=2.8, zorder=11,
            path_effects=[pe.Stroke(linewidth=5.0, foreground="#0b1520"), pe.Normal()],
            solid_capstyle="round")
    hr.draw_vehicle(ax, 0, 0, 0, color="#ff5c33")
    hr.draw_scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3a3d44")

    ob = n_out_of_band(r["d"][r["best"]], r["band"][r["best"]])
    oa = sum(n_out_of_band(r["d"][m], r["band"][m]) for m in valid)
    tg = r["turn"]
    kind = "좌회전" if tg > 30 else ("우회전" if tg < -30 else "직진")
    ax.set_title(f"{sid[:8]}  {kind}({tg:+.0f}°)  구별 분기 {r['n_dist']}개 / 모드 {len(valid)}\n"
                 f"minADE6 {r['minade']:.2f} m   "
                 f"밴드이탈 best {ob}/60 step · 6모드 합 {oa}",
                 fontsize=10.5, color=hr.TEXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="없으면 랜덤 초기화로 그린다 (레이아웃 확인용)")
    ap.add_argument("--level", default="l0", choices=["l3", "l0"])
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--limit", type=int, default=200, help="풀로 쓸 val 시나리오 수")
    ap.add_argument("--mix", default="3,3,3", help="좌회전,우회전,직진 개수")
    ap.add_argument("--min-disp", dest="min_disp", type=float, default=10.0,
                    help="정답 궤적의 최소 이동거리 [m]")
    ap.add_argument("--band", default="auto", choices=["auto", "on", "off"],
                    help="auto = l0 에서만 밴드를 칠한다")
    ap.add_argument("--rules", type=int, default=1, help="ckpt 가 없을 때만 쓰인다")
    ap.add_argument("--theta", type=int, default=0, help="ckpt 가 없을 때만 쓰인다")
    ap.add_argument("--h-src", dest="h_src", default="build", choices=["av2", "build"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3,
                    help="이 머신은 학습이 돌고 있다 — 4 이하로 둔다")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="v4_modes.png")
    args = ap.parse_args()

    # 입력 채널 수는 ckpt 에서 읽는다. 학습 때 --theta 0 이었는지 1 이었는지를 손으로
    # 맞추다 틀리면 모델이 조용히 이상한 예측을 내므로 사람이 정할 값이 아니다.
    sd = None
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        in_dim = sd["traj_encoder.weight_ih_l0"].shape[1]
        lane_in = sd["lane_encoder.0.weight"].shape[1]
    else:
        in_dim = 5 + (3 if args.theta else 0)
        lane_in = N_PTS * 2 + (N_RULE if args.rules else 0)
    use_rules = lane_in > N_PTS * 2

    ds = Av2LaneRuleDataset(DATA_ROOT, "val", args.limit, with_rules=use_rules,
                            theta_ch=in_dim > 5, h_src=args.h_src, routes=True)
    model = V4Net(in_dim=in_dim, lane_in=lane_in, level=args.level).to(args.device)
    if sd is not None:
        model.load_state_dict(sd)
    model.eval()
    print(f"[{args.level}] ckpt {args.ckpt or '없음(랜덤 초기화)'} | in_dim {in_dim} "
          f"| lane_in {lane_in} | {args.device} | 풀 {len(ds)} 시나리오", flush=True)

    rows = collect(model, ds, args, args.device)
    picks = pick(rows, args)
    show_band = args.band == "on" or (args.band == "auto" and args.level == "l0")

    fig, axes = plt.subplots(3, 3, figsize=(17, 17.6), facecolor=hr.BG)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, i in zip(axes.flat, picks):
        ax.set_visible(True)
        draw_cell(ax, rows[i], args.level, show_band)

    lines = [plt.Line2D([], [], color=PAST, lw=2.3),
             plt.Line2D([], [], color=GT, lw=2.8),
             plt.Line2D([], [], color=BEST, lw=2.4, ls=(0, (4, 3))),
             plt.Line2D([], [], color=PRED, lw=2.1),
             plt.Line2D([], [], color=ROUTE_HI, lw=1.9),
             plt.Line2D([], [], color=ROUTE, lw=1.3, alpha=0.6)]
    labs = ["과거 5초 (관측 구간)", "실제 차량이 간 길 (정답)",
            f"{args.level} 예측 · best of 6", f"{args.level} 예측 · 나머지",
            "best 가 조건으로 쓴 후보 경로", "그 밖의 후보 경로"]
    if show_band:
        lines.append(plt.Rectangle((0, 0), 1, 1, facecolor=BAND, alpha=0.35, lw=0))
        labs.append("밴드 (규칙이 허용하는 횡오프셋)")
    leg = axes.flat[0].legend(lines, labs, fontsize=9.5, loc="upper left",
                              facecolor="#22242a", edgecolor="#3a3d44", framealpha=0.92)
    for t in leg.get_texts():
        t.set_color(hr.TEXT)

    tag = "랜덤 초기화 (레이아웃 확인용)" if sd is None else Path(args.ckpt).name
    fig.suptitle(f"v4 예측 모드 = 지도에서 열거된 후보 경로  ·  level={args.level}  ·  {tag}",
                 fontsize=18, color="#e8ecf2", y=0.995, va="top")
    sub = ("예측 끝점의 숫자는 그 모드가 조건으로 쓴 경로 번호다"
           "   ·   구별 분기가 1개면 모드 6개는 같은 경로 위의 종방향 다중성이다")
    if show_band:
        sub += "\n칠해진 밴드가 l0 의 출력 공간이 허용하는 횡방향 범위다"
    fig.text(0.5, 0.9755, sub, ha="center", va="top", fontsize=12, color="#9aa3ad")
    fig.tight_layout(rect=[0.006, 0.012, 0.994, 0.9635 if not show_band else 0.958])
    fig.savefig(args.out, dpi=120, facecolor=hr.BG)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
