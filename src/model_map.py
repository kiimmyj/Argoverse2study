"""
LSTMMapAttn - 궤적 + HD map(차선) 을 attention 으로 결합한 K=6 예측 (v3).

① 궤적 인코더: (B,50,5) → LSTM → 궤적 특징 (B,128)
② 지도 인코더: (B,20,10,2) → flatten → MLP → 차선 특징 (B,20,128)
③ Attention: 궤적(query)이 차선들(key/value)에 집중 → 지도 반영된 특징 (B,128)
④ K=6 head: → 궤적 6개 (B,6,60,2) + 확률 (B,6)
"""
import torch
import torch.nn as nn

OBS_LEN, PRED_LEN = 50, 60
IN_DIM, OUT_DIM = 5, 2
N_LANES, N_PTS = 20, 10
K = 6
HID = 128


class LSTMSeq2Seq(nn.Module):   # 이름 유지(호환), 실제는 map-attention 모델
    def __init__(self, hid=HID, k=K, pred_len=PRED_LEN, out_dim=OUT_DIM):
        super().__init__()
        self.k, self.pred_len, self.out_dim = k, pred_len, out_dim

        # ① 궤적 인코더 (LSTM)
        self.traj_encoder = nn.LSTM(IN_DIM, hid, num_layers=2, batch_first=True)

        # ② 지도 인코더 (각 차선 10*2=20 → 128)
        self.lane_encoder = nn.Sequential(
            nn.Linear(N_PTS * 2, hid), nn.ReLU(), nn.Linear(hid, hid))

        # ③ Attention (궤적 query, 차선 key/value)
        self.attn = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)

        # 궤적특징 + attention결과 결합
        self.fuse = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU())

        # ④ K=6 head
        self.traj_head = nn.Linear(hid, k * pred_len * out_dim)
        self.prob_head = nn.Linear(hid, k)

    def forward(self, x, lanes, lane_mask, y=None, teacher_forcing=True):
        """
        x:(B,50,5)  lanes:(B,20,10,2)  lane_mask:(B,20)
        return: traj (B,6,60,2), logits (B,6)
        """
        B = x.size(0)

        # ① 궤적 특징
        _, (h, c) = self.traj_encoder(x)
        traj_feat = h[-1]                                  # (B,128)

        # ② 차선 특징
        lanes_flat = lanes.reshape(B, N_LANES, N_PTS * 2)  # (B,20,20)
        lane_feat = self.lane_encoder(lanes_flat)          # (B,20,128)

        # ③ Attention: query=궤적(1개), key/value=차선(20개)
        q = traj_feat.unsqueeze(1)                         # (B,1,128)
        key_padding = (lane_mask == 0)                     # (B,20) True=무시
        attn_out, _ = self.attn(q, lane_feat, lane_feat,
                                key_padding_mask=key_padding)  # (B,1,128)
        attn_out = attn_out.squeeze(1)                     # (B,128)

        # 궤적특징 + 지도특징 결합
        fused = self.fuse(torch.cat([traj_feat, attn_out], dim=1))  # (B,128)

        # ④ K=6 출력
        traj = self.traj_head(fused).view(B, self.k, self.pred_len, self.out_dim)
        logits = self.prob_head(fused)
        return traj, logits


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from torch.utils.data import DataLoader
    from dataset_map import Av2MapDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = Av2MapDataset("/data/argoverse2/motion_forecasting", "train", limit=32)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = LSTMSeq2Seq().to(device)
    print(f"model params : {sum(p.numel() for p in model.parameters()):,}")

    b = next(iter(loader))
    x = b["x"].to(device)
    lanes = b["lanes"].to(device)
    mask = b["lane_mask"].to(device)
    traj, logits = model(x, lanes, mask)
    print(f"x     : {tuple(x.shape)}       (8, 50, 5)")
    print(f"lanes : {tuple(lanes.shape)}  (8, 20, 10, 2)")
    print(f"traj  : {tuple(traj.shape)}   (8, 6, 60, 2) 기대")
    print(f"logits: {tuple(logits.shape)}      (8, 6) 기대")
    ok = traj.shape == (x.size(0), 6, 60, 2) and logits.shape == (x.size(0), 6)
    print("forward OK" if ok else "shape mismatch!")
