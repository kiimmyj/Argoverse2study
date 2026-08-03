"""
LSTMSeq2Seq - focal 궤적 예측 baseline (K=1, teacher forcing).

Encoder: 입력 (B, 50, 5) 를 LSTM 으로 읽어 hidden state 로 압축
Decoder: 그 state 에서 미래 (B, 60, 2) 위치를 한 step 씩 생성
         teacher forcing: 각 step 입력 = 이전 시점의 '정답' 위치
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

        # Decoder: 이전 위치(2) -> 다음 위치. encoder state 를 이어받음
        self.decoder = nn.LSTM(out_dim, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)   # hidden -> (x, y)

    def forward(self, x, y=None, teacher_forcing=True):
        """
        x : (B, 50, 5)  입력
        y : (B, 60, 2)  정답 (teacher forcing 시 필요)
        return: (B, 60, 2)  예측
        """
        B = x.size(0)

        # 1) Encoder: 과거를 hidden state 로 압축
        _, (h, c) = self.encoder(x)          # h,c: (num_layers, B, hidden)

        # 2) Decoder 첫 입력 = t=49 위치 (정규화 좌표라 (0,0))
        dec_in = torch.zeros(B, 1, self.out_dim, device=x.device)  # (B,1,2)

        outputs = []
        for t in range(self.pred_len):
            out, (h, c) = self.decoder(dec_in, (h, c))   # out: (B,1,hidden)
            pred = self.head(out)                        # (B,1,2)
            outputs.append(pred)

            # 다음 step 입력 결정
            if teacher_forcing and y is not None:
                dec_in = y[:, t:t+1, :]        # 정답 이전 위치(커닝)
            else:
                dec_in = pred                 # 자기 예측(추론)

        return torch.cat(outputs, dim=1)      # (B, 60, 2)


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from torch.utils.data import DataLoader
    from dataset import Av2FocalDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # DataLoader: Dataset 을 배치로 묶기 (focal 은 길이 50/60 고정이라 그냥 쌓임)
    ds = Av2FocalDataset("/data/argoverse2/motion_forecasting", "train", limit=64)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = LSTMSeq2Seq().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params : {n_params:,}")

    # 배치 하나 통과시켜 shape 확인
    batch = next(iter(loader))
    x, y = batch["x"].to(device), batch["y"].to(device)
    print(f"x (input)  : {tuple(x.shape)}   (16, 50, 5) 기대")
    print(f"y (target) : {tuple(y.shape)}   (16, 60, 2) 기대")

    pred = model(x, y, teacher_forcing=True)
    print(f"pred       : {tuple(pred.shape)}   (16, 60, 2) 기대")
    print("forward OK" if pred.shape == y.shape else "shape mismatch!")
