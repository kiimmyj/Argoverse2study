"""
model_lane.py - v3 구조에 차로 이동 규칙 피처를 얹은 모델.

model_map.py(v3) 와 레이어 정의가 같고, 다른 것은 **lane_encoder 의 입력 차원** 하나다.

    규칙 없음 : 차선 좌표 10점 × 2 = 20
    규칙 포함 : 좌표 20 + 규칙 피처 10 = 30

규칙 피처 (dataset_lane.py) 10개:
  [0:2] 진행방향 단위벡터   [2] focal 과 같은 방향인가   [3] 교차로 내부인가
  [4:7] 차로 종류           [7] 좌변경 가능              [8] 우변경 가능
  [9]   규칙상 도달 가능

궤적 인코더·attention·K=6 head 는 v3 그대로라, 성능 차이가 나면 규칙 피처 때문이다.
"""
import torch
import torch.nn as nn

OBS_LEN, PRED_LEN = 50, 60
IN_DIM, OUT_DIM = 5, 2
N_LANES, N_PTS = 20, 10
N_RULE = 10
K = 6
HID = 128


class LSTMMapRule(nn.Module):
    def __init__(self, in_dim=IN_DIM, lane_in=N_PTS * 2 + N_RULE, hid=HID,
                 k=K, pred_len=PRED_LEN, out_dim=OUT_DIM):
        super().__init__()
        self.k, self.pred_len, self.out_dim = k, pred_len, out_dim
        self.lane_in = lane_in
        self.traj_encoder = nn.LSTM(in_dim, hid, num_layers=2, batch_first=True)
        self.lane_encoder = nn.Sequential(
            nn.Linear(lane_in, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.attn = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU())
        self.traj_head = nn.Linear(hid, k * pred_len * out_dim)
        self.prob_head = nn.Linear(hid, k)

    def forward(self, x, lanes, lane_mask, lane_feat=None):
        B = x.size(0)
        _, (h, _) = self.traj_encoder(x)
        traj_feat = h[-1]

        lane_in = lanes.reshape(B, N_LANES, N_PTS * 2)          # (B,20,20)
        if self.lane_in > N_PTS * 2:
            if lane_feat is None:
                raise ValueError("규칙 피처(lane_feat)가 필요한 모델이다")
            lane_in = torch.cat([lane_in, lane_feat], dim=2)     # (B,20,30)
        lane_emb = self.lane_encoder(lane_in)

        attn_out, _ = self.attn(traj_feat.unsqueeze(1), lane_emb, lane_emb,
                                key_padding_mask=(lane_mask == 0))
        fused = self.fuse(torch.cat([traj_feat, attn_out.squeeze(1)], dim=1))
        traj = self.traj_head(fused).view(B, self.k, self.pred_len, self.out_dim)
        return traj, self.prob_head(fused)


if __name__ == "__main__":
    for lane_in in (20, 30):
        m = LSTMMapRule(lane_in=lane_in)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(4, OBS_LEN, IN_DIM)
        lanes = torch.randn(4, N_LANES, N_PTS, 2)
        lf = torch.randn(4, N_LANES, N_RULE)
        t, l = m(x, lanes, torch.ones(4, N_LANES), lf if lane_in == 30 else None)
        print(f"lane_in={lane_in:2d}  params {n:,}  traj {tuple(t.shape)}  logits {tuple(l.shape)}")
