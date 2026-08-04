"""
LSTMEncoderMLP - focal 궤적 예측 baseline (방향 B: 미래 60 step 한 번에 출력).

Encoder: 입력 (B, 50, 5) 를 LSTM 으로 읽어 hidden state 로 압축
Head    : hidden(128) → Linear → 미래 (B, 60, 2) 를 통째로 출력
          (한 step 씩 이어붙이지 않음 → 오차 누적/발산 없음, teacher forcing 불필요)
"""
import torch
import torch.nn as nn

OBS_LEN, PRED_LEN = 50, 60
IN_DIM, OUT_DIM = 5, 2          # 입력 피처 5 / 출력 위치 2


class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM,
                 hidden=128, num_layers=2, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.out_dim = out_dim

        # Encoder: (B, 50, 5) -> hidden state
        self.encoder = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)

        # Head: 마지막 hidden(128) -> 미래 60*2=120 을 한 번에
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pred_len * out_dim),
        )

    def forward(self, x, y=None, teacher_forcing=True):
        """
        x : (B, 50, 5)  입력
        y, teacher_forcing : 이전 인터페이스 호환용(사용 안 함)
        return: (B, 60, 2)  예측
        """
        B = x.size(0)

        # 1) Encoder: 과거를 hidden state 로 압축
        _, (h, c) = self.encoder(x)          # h: (num_layers, B, hidden)
        last_h = h[-1]                       # 마지막 층의 hidden: (B, hidden)

        # 2) 미래 60 step 을 한 번에 출력 후 (B, 60, 2) 로 reshape
        out = self.head(last_h)              # (B, 120)
        return out.view(B, self.pred_len, self.out_dim)   # (B, 60, 2)


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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params : {n_params:,}")

    batch = next(iter(loader))
    x, y = batch["x"].to(device), batch["y"].to(device)
    print(f"x (input)  : {tuple(x.shape)}   (16, 50, 5) 기대")
    print(f"y (target) : {tuple(y.shape)}   (16, 60, 2) 기대")

    pred = model(x)
    print(f"pred       : {tuple(pred.shape)}   (16, 60, 2) 기대")
    print("forward OK" if pred.shape == y.shape else "shape mismatch!")
