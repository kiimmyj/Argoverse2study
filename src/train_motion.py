"""
train_motion.py - 궤적 입력 표현만 바꿔가며 v3 를 학습/비교한다.

train_map.py 와 손실·평가·하이퍼파라미터가 동일하고, 다른 것은 x 의 채널뿐이다.
같은 시드를 쓰므로 가중치 초기화와 배치 순서도 표현끼리 동일하다.

  python src/train_motion.py --repr raw5   --limit 20000 --epochs 8
  python src/train_motion.py --repr vah3   --limit 20000 --epochs 8
  python src/train_motion.py --repr ah0_2  --limit 20000 --epochs 8
"""
import argparse, json, os, sys, time
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_motion import Av2MotionDataset, in_dim
from model_motion import LSTMMapAttn

DATA_ROOT = "/data/argoverse2/motion_forecasting"


def multimodal_loss(traj, logits, y):
    """winner-takes-all: 6개 중 정답에 가장 가까운 것만 회귀 학습 + 어느 것이 맞았는지 분류."""
    B = traj.shape[0]
    fde = torch.norm(traj[:, :, -1] - y.unsqueeze(1)[:, :, -1], dim=-1)   # (B,K)
    best = fde.argmin(dim=1)
    idx = best.view(B, 1, 1, 1).expand(B, 1, traj.size(2), traj.size(3))
    best_traj = traj.gather(1, idx).squeeze(1)
    return F.smooth_l1_loss(best_traj, y) + F.cross_entropy(logits, best)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_ade = tot_fde = n = 0
    for b in loader:
        x = b["x"].to(device); y = b["y"].to(device)
        traj, _ = model(x, b["lanes"].to(device), b["lane_mask"].to(device))
        d = torch.norm(traj - y.unsqueeze(1), dim=-1)      # (B,K,60)
        tot_ade += d.mean(dim=2).min(dim=1).values.sum().item()
        tot_fde += d[:, :, -1].min(dim=1).values.sum().item()
        n += x.size(0)
    return tot_ade / n, tot_fde / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", default="vah3")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    lim = None if args.limit == 0 else args.limit
    train_ds = Av2MotionDataset(DATA_ROOT, "train", lim, args.repr)
    val_ds = Av2MotionDataset(DATA_ROOT, "val", args.val_limit, args.repr)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g,
                              num_workers=args.workers, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, persistent_workers=True)

    model = LSTMMapAttn(in_dim=in_dim(args.repr)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.repr}] device {device} | in_dim {in_dim(args.repr)} | params {n_par:,} | "
          f"train {len(train_ds)} val {len(val_ds)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist, best = [], (1e9, 1e9)
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = seen = 0
        for b in train_loader:
            x = b["x"].to(device); y = b["y"].to(device)
            traj, logits = model(x, b["lanes"].to(device), b["lane_mask"].to(device))
            loss = multimodal_loss(traj, logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)

        ade, fde = evaluate(model, val_loader, device)
        hist.append({"epoch": epoch, "loss": run / seen, "minADE6": ade, "minFDE6": fde,
                     "sec": time.time() - t0})
        if ade < best[0]:
            best = (ade, fde)
            torch.save(model.state_dict(), f"{args.outdir}/lstm_{args.repr}.pth")
        print(f"[{args.repr}] {epoch:02d}/{args.epochs} loss {run/seen:.4f} | "
              f"minADE6 {ade:.3f} | minFDE6 {fde:.3f} | {time.time()-t0:.0f}s", flush=True)

    json.dump({"repr": args.repr, "in_dim": in_dim(args.repr), "params": n_par,
               "args": vars(args), "history": hist,
               "best_minADE6": best[0], "best_minFDE6": best[1]},
              open(f"{args.outdir}/{args.repr}.json", "w"), indent=2)
    print(f"[{args.repr}] BEST minADE6 {best[0]:.3f} | minFDE6 {best[1]:.3f} "
          f"-> {args.outdir}/{args.repr}.json", flush=True)


if __name__ == "__main__":
    main()
