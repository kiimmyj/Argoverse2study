"""
train_lane.py - 차로 이동 규칙 피처를 넣어 v3 를 학습한다.

train_map.py(v3) 와 손실·평가·하이퍼파라미터가 동일하고, 다른 것은
Dataset 이 규칙 피처를 함께 주고 lane_encoder 입력이 20 → 30 이라는 점뿐이다.

  python src/train_lane.py --rules 1   # 규칙 포함
  python src/train_lane.py --rules 0   # 같은 Dataset, 규칙 없음 (대조군)
"""
import argparse, json, os, sys, time
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_lane import Av2LaneRuleDataset
from model_lane import LSTMMapRule, N_PTS, N_RULE

DATA_ROOT = "/data/argoverse2/motion_forecasting"


def multimodal_loss(traj, logits, y):
    B = traj.shape[0]
    fde = torch.norm(traj[:, :, -1] - y.unsqueeze(1)[:, :, -1], dim=-1)
    best = fde.argmin(dim=1)
    idx = best.view(B, 1, 1, 1).expand(B, 1, traj.size(2), traj.size(3))
    best_traj = traj.gather(1, idx).squeeze(1)
    return F.smooth_l1_loss(best_traj, y) + F.cross_entropy(logits, best)


@torch.no_grad()
def evaluate(model, loader, device, use_rules):
    model.eval()
    tot_ade = tot_fde = n = 0
    for b in loader:
        x, y = b["x"].to(device), b["y"].to(device)
        lf = b["lane_feat"].to(device) if use_rules else None
        traj, _ = model(x, b["lanes"].to(device), b["lane_mask"].to(device), lf)
        d = torch.norm(traj - y.unsqueeze(1), dim=-1)
        tot_ade += d.mean(dim=2).min(dim=1).values.sum().item()
        tot_fde += d[:, :, -1].min(dim=1).values.sum().item()
        n += x.size(0)
    return tot_ade / n, tot_fde / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=int, default=1)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    use_rules = bool(args.rules)
    tag = f"lane_{'rules' if use_rules else 'norule'}_s{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    lim = None if args.limit == 0 else args.limit
    tr = Av2LaneRuleDataset(DATA_ROOT, "train", lim, with_rules=use_rules)
    va = Av2LaneRuleDataset(DATA_ROOT, "val", args.val_limit, with_rules=use_rules)
    g = torch.Generator(); g.manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, generator=g,
                    num_workers=args.workers, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, persistent_workers=True)

    lane_in = N_PTS * 2 + (N_RULE if use_rules else 0)
    model = LSTMMapRule(lane_in=lane_in).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] device {device} | lane_in {lane_in} | params {npar:,} | "
          f"train {len(tr)} val {len(va)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist, best = [], (1e9, 1e9)
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = seen = 0
        for b in tl:
            x, y = b["x"].to(device), b["y"].to(device)
            lf = b["lane_feat"].to(device) if use_rules else None
            traj, logits = model(x, b["lanes"].to(device), b["lane_mask"].to(device), lf)
            loss = multimodal_loss(traj, logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)
        ade, fde = evaluate(model, vl, device, use_rules)
        hist.append({"epoch": epoch, "loss": run / seen, "minADE6": ade,
                     "minFDE6": fde, "sec": time.time() - t0})
        if ade < best[0]:
            best = (ade, fde)
            torch.save(model.state_dict(), f"{args.outdir}/lstm_{tag}.pth")
        print(f"[{tag}] {epoch:02d}/{args.epochs} loss {run/seen:.4f} | "
              f"minADE6 {ade:.3f} | minFDE6 {fde:.3f} | {time.time()-t0:.0f}s", flush=True)

    json.dump({"tag": tag, "rules": use_rules, "lane_in": lane_in, "params": npar,
               "args": vars(args), "history": hist,
               "best_minADE6": best[0], "best_minFDE6": best[1]},
              open(f"{args.outdir}/{tag}.json", "w"), indent=2)
    print(f"[{tag}] BEST minADE6 {best[0]:.3f} | minFDE6 {best[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
