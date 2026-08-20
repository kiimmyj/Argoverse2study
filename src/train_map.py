"""
train_map.py - 궤적+지도(attention) K=6 모델 학습 (v3).

K=6 학습과 동일하되 lanes/lane_mask 를 모델에 함께 전달.
사용: python src/train_map.py --limit 50000 --val-limit 2000 --epochs 15 --lr 5e-4
"""
import argparse, sys, time
sys.path.append("src")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_map import Av2MapDataset
from model_map import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"


def multimodal_loss(traj, logits, y):
    B, K = traj.shape[0], traj.shape[1]
    y_exp = y.unsqueeze(1)
    fde = torch.norm(traj[:, :, -1] - y_exp[:, :, -1], dim=-1)   # (B,K)
    best = fde.argmin(dim=1)
    idx = best.view(B, 1, 1, 1).expand(B, 1, traj.size(2), traj.size(3))
    best_traj = traj.gather(1, idx).squeeze(1)
    reg = F.smooth_l1_loss(best_traj, y)
    cls = F.cross_entropy(logits, best)
    return reg + cls


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_ade = tot_fde = n = 0
    for b in loader:
        x = b["x"].to(device); y = b["y"].to(device)
        lanes = b["lanes"].to(device); mask = b["lane_mask"].to(device)
        traj, _ = model(x, lanes, mask)
        d = torch.norm(traj - y.unsqueeze(1), dim=-1)   # (B,K,60)
        tot_ade += d.mean(dim=2).min(dim=1).values.sum().item()
        tot_fde += d[:, :, -1].min(dim=1).values.sum().item()
        n += x.size(0)
    return tot_ade / n, tot_fde / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    lim = None if args.limit == 0 else args.limit
    train_ds = Av2MapDataset(DATA_ROOT, "train", limit=lim)
    val_ds = Av2MapDataset(DATA_ROOT, "val", limit=args.val_limit)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = LSTMSeq2Seq().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = seen = 0
        for b in train_loader:
            x = b["x"].to(device); y = b["y"].to(device)
            lanes = b["lanes"].to(device); mask = b["lane_mask"].to(device)
            traj, logits = model(x, lanes, mask)
            loss = multimodal_loss(traj, logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)

        ade, fde = evaluate(model, val_loader, device)
        print(f"[{epoch:02d}/{args.epochs}] train_loss {run/seen:.4f} | "
              f"minADE6 {ade:.3f} | minFDE6 {fde:.3f} | {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), "lstm_map.pth")
    print("saved -> lstm_map.pth")


if __name__ == "__main__":
    main()
