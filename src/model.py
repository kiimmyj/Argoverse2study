"""
LSTMMultimodal - focal 궤적 예측 (K=6 멀티모달, 방향 B 확장).

Encoder: 입력 (B, 50, 5) → LSTM → hidden 요약
Head    : hidden → 궤적 6개 (B,6,60,2) + 각 궤적 확률 (B,6)
          한 번에 출력(오차 누적 없음), teacher forcing 불필요.
"""
import torch
import torch.nn as nn

OBS_LEN, PRED_LEN = 50, 60
IN_DIM, OUT_DIM = 5, 2
K = 6                       # 예측 모드 개수


class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM, hidden=128,
                 num_layers=2, pred_len=PRED_LEN, k=K):
        super().__init__()
        self.pred_len = pred_len
        self.out_dim = out_dim
        self.k = k

        # Encoder: (B, 50, 5) -> hidden
        self.encoder = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)

        # 공통 backbone
        self.backbone = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

        # 궤적 헤드: 6개 궤적 * 60 * 2 = 720 을 한 번에
        self.traj_head = nn.Linear(hidden, k * pred_len * out_dim)
        # 확률 헤드: 6개 모드의 점수(logit)
        self.prob_head = nn.Linear(hidden, k)

    def forward(self, x, y=None, teacher_forcing=True):
        """
        x : (B, 50, 5)
        return: traj (B, K, 60, 2),  logits (B, K)
        """
        B = x.size(0)
        _, (h, c) = self.encoder(x)          # h: (num_layers, B, hidden)
        feat = self.backbone(h[-1])          # (B, hidden)

        traj = self.traj_head(feat)          # (B, K*60*2)
        traj = traj.view(B, self.k, self.pred_len, self.out_dim)  # (B,K,60,2)
        logits = self.prob_head(feat)        # (B, K)  확률 점수(softmax 전)
        return traj, logits


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from torch.utils.data import DataLoader
    from dataset import Av2FocalDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = Av2FocalDataset("/data/argoverse2/motion_forecasting", "train", limit=64)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = LSTMSeq2Seq().to(device)
    print(f"model params : {sum(p.numel() for p in model.parameters()):,}")

    batch = next(iter(loader))
    x, y = batch["x"].to(device), batch["y"].to(device)
    traj, logits = model(x)
    print(f"x     : {tuple(x.shape)}      (16, 50, 5) 기대")
    print(f"traj  : {tuple(traj.shape)}   (16, 6, 60, 2) 기대")
    print(f"logits: {tuple(logits.shape)}      (16, 6) 기대")
    ok = traj.shape == (x.size(0), 6, 60, 2) and logits.shape == (x.size(0), 6)
    print("forward OK" if ok else "shape mismatch!")
