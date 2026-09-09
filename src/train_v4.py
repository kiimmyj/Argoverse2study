"""
train_v4.py - v4 를 레벨별로 누적 학습한다.

  L2 : python src/train_v4.py --level l2                       (= 기준선 재현)
  L3 : python src/train_v4.py --level l3
  L0 : python src/train_v4.py --level l0
  L4 : python src/train_v4.py --level l0 --offlane 1.0

손실·평가·하이퍼파라미터는 train_lane.py(기준선 minADE6 1.292) 와 같다.
다른 것은 모델과 Dataset 이 주는 필드뿐이라, 지표 차이가 나면 레벨 때문이다.

L4 를 왜 모든 모드에 거나
-------------------------
WTA 는 정답에 가장 가까운 하나에만 거리 손실을 건다. 나머지 K-1 개는 gradient 를 전혀
못 받는데, 도로를 벗어나는 예측은 사실상 전부 그 **버려진 모드들**에 있다.
L4 는 모든 모드에 개별로 벌점을 주므로 버려진 모드를 감독하는 유일한 항이다.
"""
import argparse, json, os, sys, time
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_lane import Av2LaneRuleDataset
from model_v4 import V4Net, N_PTS, N_RULE

DATA_ROOT = "/data/argoverse2/motion_forecasting"
ROUTE_KEYS = ("routes", "route_tan", "route_band", "route_len", "route_sd0",
              "route_mask", "route_sub", "v0")


def to_dev(b, device, level, use_rules):
    kw = dict(x=b["x"].to(device), lanes=b["lanes"].to(device),
              lane_mask=b["lane_mask"].to(device),
              lane_feat=b["lane_feat"].to(device) if use_rules else None)
    if level != "l2":
        kw.update({k: b[k].to(device) for k in ROUTE_KEYS})
    return kw


def loss_fn(traj, logits, y, mode_mask, aux, w_off, nonwinner_only=False):
    """WTA 거리 손실 + 모드 확률 + (선택) 차로 이탈 벌점.

    nonwinner_only=True 면 벌점을 **버려진 모드에만** 건다. 근거는 실측이다:
    기준 경로를 하나 고르면 정답 궤적의 14.05% step(시나리오의 28.7%)이 밴드를 벗어나
    벌점을 받는다. 승자 모드는 정의상 정답에 가장 가까우므로 거리 손실이 이미 옳게
    감독하고 있고, 거기에 벌점을 더하면 오탐만 얹힌다. 문서가 L4 에 부여한 목적도
    '버려진 K-1 개 감독' 이다 — 승자는 애초에 대상이 아니다.
    """
    B, Kk = traj.shape[:2]
    fde = torch.norm(traj[:, :, -1] - y.unsqueeze(1)[:, :, -1], dim=-1)
    fde = fde.masked_fill(mode_mask == 0, float("inf"))       # 없는 경로는 승자가 될 수 없다
    best = fde.argmin(dim=1)
    idx = best.view(B, 1, 1, 1).expand(B, 1, traj.size(2), traj.size(3))
    best_traj = traj.gather(1, idx).squeeze(1)
    loss = F.smooth_l1_loss(best_traj, y) + F.cross_entropy(logits, best)
    off = torch.zeros((), device=traj.device)
    if w_off > 0 and aux:
        # 밴드 밖으로 나간 만큼만 벌준다 (단측 hinge). 안쪽은 어디에 있든 0 이다 —
        # '중심에 가까울수록 좋다'로 쓰면 편도 2차선에서 중앙선이 최적해가 된다.
        d, band = aux["d"], aux["band"]
        over = torch.relu(d - band[..., 0]) + torch.relu(-d - band[..., 1])
        m = mode_mask
        if nonwinner_only:
            m = m * (1.0 - F.one_hot(best, m.size(1)).float())
        m = m.unsqueeze(-1)
        off = (over * m).sum() / m.sum().clamp(min=1) / over.size(2)
        loss = loss + w_off * off
    return loss, off.detach()


@torch.no_grad()
def evaluate(model, loader, device, level, use_rules):
    model.eval()
    ade = fde = off = n = 0.0
    dth = []                       # 스텝간 |dtheta| — 궤적이 지그재그인지 재는 계기.
    for b in loader:
        y = b["y"].to(device)
        traj, _, aux = model(**to_dev(b, device, level, use_rules))
        mm = (b["route_mask"].to(device) if level != "l2"
              else torch.ones(traj.shape[:2], device=device))
        dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
        big = torch.full_like(dist[:, :, 0], float("inf"))
        a = torch.where(mm > 0, dist.mean(2), big).min(1).values
        f = torch.where(mm > 0, dist[:, :, -1], big).min(1).values
        ade += a.sum().item(); fde += f.sum().item(); n += y.size(0)
        if aux:
            o = (torch.relu(aux["d"] - aux["band"][..., 0])
                 + torch.relu(-aux["d"] - aux["band"][..., 1]))
            off += ((o > 0).float() * mm.unsqueeze(-1)).sum().item() / max(o.size(2), 1)
            # 실제 차량의 스텝간 방향 변화는 p99.99 가 7.3° 다(라벨 전수조사).
            # 이보다 크면 물리적으로 못 내는 요레이트다.
            th = aux["theta"]
            dth.append((th[:, :, 1:] - th[:, :, :-1]).abs().flatten() * 180.0 / np.pi)
    q = (float(torch.cat(dth).median()), float(torch.cat(dth).quantile(0.99))) if dth else (0.0, 0.0)
    return ade / n, fde / n, off / n, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="l3", choices=["l2", "l3", "l0"])
    ap.add_argument("--offlane", type=float, default=0.0, help="L4 벌점 가중치")
    ap.add_argument("--off-nonwinner", dest="off_nonwinner", type=int, default=0,
                    help="1 이면 L4 를 버려진 모드에만 건다 (승자는 거리 손실이 감독)")
    ap.add_argument("--rules", type=int, default=1)
    ap.add_argument("--theta", type=int, default=1)
    ap.add_argument("--h-src", dest="h_src", default="build", choices=["av2", "build"])
    ap.add_argument("--fallback", default="fan", choices=["straight1", "straight6", "fan"],
                    help="지도가 경로를 못 주는 시나리오를 무엇으로 채울지. "
                         "straight1=직진1개(mask 1) / straight6=직진6복제(모드예산만) / fan=부채꼴6개")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    use_rules = bool(args.rules)
    tag = args.tag or f"v4_{args.level}" + (f"_off{args.offlane:g}" if args.offlane else "") \
        + f"_s{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    lim = None if args.limit == 0 else args.limit
    dkw = dict(with_rules=use_rules, theta_ch=bool(args.theta), h_src=args.h_src,
               routes=args.level != "l2", fallback=args.fallback)
    tr = Av2LaneRuleDataset(DATA_ROOT, "train", lim, **dkw)
    va = Av2LaneRuleDataset(DATA_ROOT, "val", args.val_limit, **dkw)
    g = torch.Generator(); g.manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, generator=g,
                    num_workers=args.workers, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, persistent_workers=True)

    lane_in = N_PTS * 2 + (N_RULE if use_rules else 0)
    in_dim = 5 + (3 if args.theta else 0)
    model = V4Net(in_dim=in_dim, lane_in=lane_in, level=args.level).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] {device} | level {args.level} | offlane {args.offlane} | "
          f"fallback {args.fallback} | in_dim {in_dim} "
          f"| params {npar:,} | train {len(tr)} val {len(va)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist, best = [], (1e9, 1e9)
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = seen = 0.0; roff = 0.0
        for b in tl:
            y = b["y"].to(device)
            traj, logits, aux = model(**to_dev(b, device, args.level, use_rules))
            mm = (b["route_mask"].to(device) if args.level != "l2"
                  else torch.ones(traj.shape[:2], device=device))
            loss, off = loss_fn(traj, logits, y, mm, aux, args.offlane,
                                bool(args.off_nonwinner))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * y.size(0); roff += float(off) * y.size(0); seen += y.size(0)
        ade, fde, ooff, (dth50, dth99) = evaluate(model, vl, device, args.level, use_rules)
        hist.append({"epoch": epoch, "loss": run / seen, "offlane": roff / seen,
                     "minADE6": ade, "minFDE6": fde, "val_offlane_steps": ooff,
                     "dtheta_p50_deg": dth50, "dtheta_p99_deg": dth99,
                     "sec": time.time() - t0})
        if ade < best[0]:
            best = (ade, fde)
            torch.save(model.state_dict(), f"{args.outdir}/lstm_{tag}.pth")
        print(f"[{tag}] {epoch:02d}/{args.epochs} loss {run/seen:.4f} | minADE6 {ade:.3f} | "
              f"minFDE6 {fde:.3f} | 이탈 {ooff:.2f} | dθ p99 {dth99:.1f}° | "
              f"{time.time()-t0:.0f}s", flush=True)

    json.dump({"tag": tag, "level": args.level, "offlane": args.offlane,
               "off_nonwinner": bool(args.off_nonwinner),
               "lane_in": lane_in, "in_dim": in_dim, "params": npar,
               "args": vars(args), "history": hist,
               "best_minADE6": best[0], "best_minFDE6": best[1]},
              open(f"{args.outdir}/{tag}.json", "w"), indent=2)
    print(f"[{tag}] BEST minADE6 {best[0]:.3f} | minFDE6 {best[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
