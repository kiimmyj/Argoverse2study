"""
model_motion.py - model_map.py(v3) 와 완전히 동일한 구조. 입력 채널 수만 인자로 받는다.

궤적 인코더의 입력 차원(in_dim)을 빼면 v3 와 한 글자도 다르지 않다.
표현 비교 실험에서 '표현 말고는 아무것도 안 바뀌었다' 를 보장하기 위함.
"""
import torch
import torch.nn as nn

OBS_LEN, PRED_LEN = 50, 60
OUT_DIM = 2
N_LANES, N_PTS = 20, 10
K = 6
HID = 128


class LSTMMapAttn(nn.Module):
    def __init__(self, in_dim=5, hid=HID, k=K, pred_len=PRED_LEN, out_dim=OUT_DIM):
        super().__init__()
        self.k, self.pred_len, self.out_dim = k, pred_len, out_dim
        self.traj_encoder = nn.LSTM(in_dim, hid, num_layers=2, batch_first=True)
        self.lane_encoder = nn.Sequential(
            nn.Linear(N_PTS * 2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.attn = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU())
        self.traj_head = nn.Linear(hid, k * pred_len * out_dim)
        self.prob_head = nn.Linear(hid, k)

    def forward(self, x, lanes, lane_mask):
        B = x.size(0)
        _, (h, _) = self.traj_encoder(x)
        traj_feat = h[-1]
        lane_feat = self.lane_encoder(lanes.reshape(B, N_LANES, N_PTS * 2))
        attn_out, _ = self.attn(traj_feat.unsqueeze(1), lane_feat, lane_feat,
                                key_padding_mask=(lane_mask == 0))
        fused = self.fuse(torch.cat([traj_feat, attn_out.squeeze(1)], dim=1))
        traj = self.traj_head(fused).view(B, self.k, self.pred_len, self.out_dim)
        return traj, self.prob_head(fused)


if __name__ == "__main__":
    for d in (2, 3, 4, 5):
        m = LSTMMapAttn(in_dim=d)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(4, OBS_LEN, d)
        t, l = m(x, torch.randn(4, N_LANES, N_PTS, 2), torch.ones(4, N_LANES))
        print(f"in_dim={d}  params {n:,}  traj {tuple(t.shape)}  logits {tuple(l.shape)}")
