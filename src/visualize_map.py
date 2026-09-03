"""
visualize_map.py - v3(궤적+지도 attention) 예측을 HD map 위에 그린다.

도로 렌더링은 animation/scripts/animate_focal.py 와 같은 방식(hdmap_render.py):
  - 아스팔트 면 + 실제 노면표시(점선/실선, 흰/노랑) + 교차로/횡단보도 구분
  - focal 기준 정규화 좌표로 그려서 항상 focal이 중앙, 진행방향이 화면 오른쪽
  - 모델 입력은 근처 차선 20개지만, 그림은 그 장면의 '모든' 차선을 그린다

사용:
  python src/visualize_map.py                  # hard / average 3x3 두 장
  python src/visualize_map.py --pool 500
  python src/visualize_map.py --single 0       # FDE 가장 작은 케이스 한 장 크게
"""
import argparse
import sys
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from torch.utils.data import DataLoader

import hdmap_render as hr
from dataset_map import Av2MapDataset
from model_map import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"
CKPT = "lstm_map.pth"
OBS_LEN = 50

PAST = "#c9ccd2"          # 과거 궤적
GT = "#4da3ff"            # 실제 미래
PRED = "#ff5c33"          # 예측 6개
BEST = "#5ce08a"          # 6개 중 최선
FOCAL = "#ff5c33"


def _outline(lw):
    """궤적이 노면표시와 섞이지 않도록 어두운 테두리를 두른다."""
    return [pe.Stroke(linewidth=lw + 1.8, foreground="#101115"), pe.Normal()]


def draw_case(ax, r, label, aspect=1.0, min_span=60.0, mark_lw=1.3, lw_scale=1.0):
    """한 칸: 도로 + 과거/정답/예측 6개. aspect=축 가로/세로 비율(시야를 축에 맞춤)."""
    xy = np.concatenate([r["hist"], r["gt"]] + r["trajs"], axis=0)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (lo + hi) / 2
    xr, yr = (hi - lo) * 1.2 + 14.0
    hw = max(xr / 2, min_span / 2, (yr / 2) * aspect)
    hh = hw / aspect
    xlim, ylim = (cx - hw, cx + hw), (cy - hh, cy + hh)

    hr.draw_scene(ax, r["scene"], xlim, ylim, mark_lw=mark_lw)

    h, gt = r["hist"], r["gt"]
    lw_h, lw_g, lw_b = 2.4 * lw_scale, 3.0 * lw_scale, 2.4 * lw_scale
    ax.plot(h[:, 0], h[:, 1], color=PAST, lw=lw_h, zorder=6, label="past 5s",
            path_effects=_outline(lw_h))
    ax.plot(gt[:, 0], gt[:, 1], color=GT, lw=lw_g, zorder=7, label="ground truth",
            path_effects=_outline(lw_g))
    for k, tr in enumerate(r["trajs"]):
        if k == r["best"]:
            continue
        ax.plot(tr[:, 0], tr[:, 1], color=PRED, lw=1.5 * lw_scale, ls="--",
                alpha=0.35 + 0.55 * float(r["probs"][k]), zorder=5)
    ax.plot([], [], color=PRED, lw=1.5 * lw_scale, ls="--", label="pred (K=6)")
    b = r["trajs"][r["best"]]
    ax.plot(b[:, 0], b[:, 1], color=BEST, lw=lw_b, ls="--", zorder=8,
            label="best of 6", path_effects=_outline(lw_b))
    hr.draw_vehicle(ax, 0.0, 0.0, 0.0, color=FOCAL)
    hr.draw_scale_bar(ax, xlim, ylim)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3a3d44")
    ax.set_title(f"{label} | minADE={r['min_ade']:.1f}m  minFDE={r['min_fde']:.1f}m",
                 fontsize=11, color=hr.TEXT)


def style_legend(ax, fontsize=9):
    leg = ax.legend(fontsize=fontsize, loc="upper left", facecolor="#22242a",
                    edgecolor="#3a3d44", framealpha=0.9)
    for t in leg.get_texts():
        t.set_color(hr.TEXT)


def draw_grid(records, idxs, labels, title, out):
    fig, axes = plt.subplots(3, 3, figsize=(17, 17), facecolor=hr.BG)
    for ax, i, lab in zip(axes.flat, idxs, labels):
        draw_case(ax, records[i], lab)
    style_legend(axes.flat[0])
    fig.suptitle(title, fontsize=15, color=hr.TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=hr.BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def draw_single(r, label, out):
    w, h = 15.0, 8.5
    fig, ax = plt.subplots(figsize=(w, h), facecolor=hr.BG)
    draw_case(ax, r, label, aspect=w / h, mark_lw=1.8, lw_scale=1.3)
    style_legend(ax, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=hr.BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def collect(args, device):
    use_lane = args.lane > 0
    if use_lane:
        from dataset_lane import Av2LaneRuleDataset
        from model_lane import LSTMMapRule
        ds = Av2LaneRuleDataset(DATA_ROOT, args.split, limit=args.pool,
                                with_rules=(args.lane == 30))
        model = LSTMMapRule(lane_in=args.lane).to(device)
    else:
        ds = Av2MapDataset(DATA_ROOT, args.split, limit=args.pool)
        model = LSTMSeq2Seq().to(device)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    records = []
    with torch.no_grad():
        for idx, b in enumerate(loader):
            x = b["x"].to(device)
            lanes_t = b["lanes"].to(device); mask = b["lane_mask"].to(device)
            y = b["y"][0].numpy()
            if use_lane:
                lf = b["lane_feat"].to(device) if args.lane == 30 else None
                traj, logits = model(x, lanes_t, mask, lf)
            else:
                traj, logits = model(x, lanes_t, mask)
            traj = traj[0].cpu().numpy()
            probs = F.softmax(logits[0], dim=0).cpu().numpy()
            origin = b["origin"][0].numpy(); theta = float(b["theta"][0])

            fde_k = np.linalg.norm(traj[:, -1] - y[-1], axis=1)
            ade_k = np.linalg.norm(traj - y[None], axis=2).mean(axis=1)
            best = int(fde_k.argmin())        # 그림에서 강조할 모드
            # 지표는 AV2 규약대로 각각 독립적으로 최소 (학습 evaluate() 와 동일)

            sdir = ds.dirs[idx]; sid = sdir.name
            # 그림은 정규화 좌표 그대로 (focal 중앙 / 진행방향 +x)
            scene = hr.build_scene(sdir / f"log_map_archive_{sid}.json", origin, theta)

            records.append({
                "sid": sid,
                "min_ade": float(ade_k.min()), "min_fde": float(fde_k.min()),
                "hist": x[0, :, :2].cpu().numpy(),
                "gt": y,
                "trajs": [traj[k] for k in range(6)],
                "scene": scene, "probs": probs, "best": best,
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--pool", type=int, default=300, help="평가/후보 시나리오 수")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--lane", type=int, default=-1,
                    help="차로규칙 모델: lane_encoder 입력(20=규칙없음, 30=규칙포함). -1이면 dataset_map")
    ap.add_argument("--title", default="v3 on HD map")
    ap.add_argument("--prefix", default="prediction_map",
                    help="출력 파일 이름 앞부분")
    ap.add_argument("--single", type=int, default=None,
                    help="FDE 오름차순 등수 하나만 크게 저장 (0=가장 정확)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = collect(args, device)
    records.sort(key=lambda r: r["min_fde"])
    n = len(records)

    if args.single is not None:
        i = max(0, min(n - 1, args.single))
        draw_single(records[i], f"{records[i]['sid'][:8]} (rank {i+1}/{n})",
                    f"{args.prefix}_single.png")
    else:
        draw_grid(records, list(range(n - 9, n)), ["hard"] * 9,
                  f"{args.title} - HARD cases (high FDE)", f"{args.prefix}_hard.png")
        mid = n // 2
        draw_grid(records, list(range(mid - 4, mid + 5)), ["avg"] * 9,
                  f"{args.title} - AVERAGE cases", f"{args.prefix}_avg.png")

    print(f"pool={n}  minADE6={np.mean([r['min_ade'] for r in records]):.2f}m  "
          f"minFDE6={np.mean([r['min_fde'] for r in records]):.2f}m")


if __name__ == "__main__":
    main()
