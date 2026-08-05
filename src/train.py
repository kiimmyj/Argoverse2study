"""
train.py - LSTM 멀티모달(K=6) 학습.

loss = winner-takes-all 회귀(6개 중 최선만) + 모드 분류(어느 게 최선인지)
평가 = minADE_6 / minFDE_6 (6개 중 최선)
사용: python src/train.py --limit 50000 --val-limit 2000 --epochs 15 --lr 5e-4
"""
import argparse, sys, time
sys.path.append("src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Av2FocalDataset
from model import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"


def multimodal_loss(traj, logits, y):
    """
    traj:(B,K,60,2)  logits:(B,K)  y:(B,60,2)
    winner-takes-all: 각 샘플에서 최종점 오차가 가장 작은 모드를 정답 모드로.
    """
    B, K = traj.shape[0], traj.shape[1]
    y_exp = y.unsqueeze(1)                         # (B,1,60,2)
    # 각 모드의 최종점(FDE) 거리로 winner 선택
    fde = torch.norm(traj[:, :, -1] - y_exp[:, :, -1], dim=-1)   # (B,K)
    best = fde.argmin(dim=1)                        # (B,)  최선 모드 인덱스

    # 회귀 손실: 최선 모드만 (smooth L1, 전 구간)
    idx = best.view(B, 1, 1, 1).expand(B, 1, traj.size(2), traj.size(3))
    best_traj = traj.gather(1, idx).squeeze(1)      # (B,60,2)
    reg = F.smooth_l1_loss(best_traj, y)

    # 분류 손실: 어느 모드가 최선인지 맞히기
    cls = F.cross_entropy(logits, best)
    return reg + cls, reg.item(), cls.item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_ade = tot_fde = n = 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        traj, _ = model(x)                          # (B,K,60,2)
        d = torch.norm(traj - y.unsqueeze(1), dim=-1)   # (B,K,60)
        ade_k = d.mean(dim=2)                        # (B,K) 각 모드 ADE
        fde_k = d[:, :, -1]                          # (B,K) 각 모드 FDE
        tot_ade += ade_k.min(dim=1).values.sum().item()   # minADE_6
        tot_fde += fde_k.min(dim=1).values.sum().item()   # minFDE_6
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
    train_ds = Av2FocalDataset(DATA_ROOT, "train", limit=lim)
    val_ds = Av2FocalDataset(DATA_ROOT, "val", limit=args.val_limit)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = LSTMSeq2Seq().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = seen = 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            traj, logits = model(x)
            loss, _, _ = multimodal_loss(traj, logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)

        ade, fde = evaluate(model, val_loader, device)
        print(f"[{epoch:02d}/{args.epochs}] train_loss {run/seen:.4f} | "
              f"minADE6 {ade:.3f} | minFDE6 {fde:.3f} | {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), "lstm_multimodal.pth")
    print("saved -> lstm_multimodal.pth")


if __name__ == "__main__":
    main()
