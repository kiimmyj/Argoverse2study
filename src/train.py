"""
train.py - LSTMSeq2Seq focal 궤적 예측 baseline 학습.

먼저 작은 데이터(--limit)로 loss 가 줄어드는지 확인 후 전체로 확대.
사용:
  python src/train.py                          # 기본: train limit=2000, 10 epoch
  python src/train.py --limit 0 --epochs 30    # 전체 데이터, 30 epoch (tmux 권장)
"""
import argparse
import sys
import time
sys.path.append("src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Av2FocalDataset
from model import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"


@torch.no_grad()
def evaluate(model, loader, device):
    """val loss + minADE/minFDE (정규화 좌표 기준, K=1)."""
    model.eval()
    tot_loss, tot_ade, tot_fde, n = 0.0, 0.0, 0.0, 0
    loss_fn = nn.SmoothL1Loss()
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        pred = model(x, teacher_forcing=False)          # 추론: 자기예측
        tot_loss += loss_fn(pred, y).item() * x.size(0)
        # 시점별 유클리드 거리
        dist = torch.norm(pred - y, dim=-1)             # (B, 60)
        tot_ade += dist.mean(dim=1).sum().item()        # 전 구간 평균
        tot_fde += dist[:, -1].sum().item()             # 마지막 시점
        n += x.size(0)
    return tot_loss / n, tot_ade / n, tot_fde / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="train 시나리오 수 (0=전체)")
    ap.add_argument("--val-limit", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    lim = None if args.limit == 0 else args.limit
    train_ds = Av2FocalDataset(DATA_ROOT, "train", limit=lim)
    val_ds = Av2FocalDataset(DATA_ROOT, "val", limit=args.val_limit)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4)

    model = LSTMSeq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)

            pred = model(x, y, teacher_forcing=True)    # 학습: teacher forcing
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 폭발 방지
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        tr_loss = run_loss / seen
        val_loss, ade, fde = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
              f"minADE {ade:.3f} | minFDE {fde:.3f} | {dt:.1f}s")

    # 가중치 저장
    torch.save(model.state_dict(), "lstm_seq2seq.pth")
    print("saved -> lstm_seq2seq.pth")


if __name__ == "__main__":
    main()
