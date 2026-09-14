"""
model_v4.py - v4 모델. L2 / L3 / L0 을 **누적으로** 켠다.

    level="l2"   지도 cross-attention + 학습된 익명 슬롯 K=6, 위치 직접 회귀   (= 기준선)
    level="l3"   모드가 익명 슬롯이 아니라 **지도에서 열거된 후보 경로**, 위치 직접 회귀
    level="l0"   l3 + 출력을 **액션 (a, theta)** 로 바꾸고 Frenet 적분기로 궤적 복원

왜 L3 가 대체 불가인가
----------------------
모드가 익명 슬롯이면 갈림길에서 모델이 **액션의 평균**을 뱉는다. T자에서 좌회전과 직진이
반반이면 평균 곡률로 교차로를 48° 비스듬히 가로지르는 궤적이 나오는데, 이건 매끄럽고
물리적으로 실행 가능해서 **모든 검사를 통과한다**. 다중모드를 만드는 것은 정보가 아니라
**이산 조건 변수**이고, 그 조건 변수가 지도에서 실제로 열거된 경로여야 한다.

L0 을 왜 Frenet 으로 적분하나
-----------------------------
데카르트로 적분하면(p += v·dt·(cos h, sin h)) 횡오프셋 d 가 명시적으로 안 나와서
L4 벌점을 걸려면 예측 궤적을 다시 경로에 투영해야 한다 — 미분 가능하게 만들기 까다롭다.
경로 좌표계에서 적분하면 속도벡터가 차로 접선과 이루는 각이 곧 theta 이므로

    ds = v·cos(theta)·dt        dd = v·sin(theta)·dt

로 **d 가 상태 변수로 직접 나온다**. L4 는 그 d 에 hinge 를 거는 것으로 끝나고,
h = k(s) + theta 도 정의 그대로 성립한다. 시작 상태 (s0, d0) 는 Dataset 이 준다
(현재 위치를 경로에 투영한 값 — 복원 왕복오차 0.000 m 확인).

출력 범위
---------
theta 는 tanh 로 ±90° 에 가둔다(실측 |theta| p99 = 82°). a 는 ±8 m/s² —
승용차 최대 감속이 대략 -8, 가속이 +5 이므로 감속 쪽에 맞춘 보수적 상한이다.
속도는 relu 로 음수를 막는다(후진은 이 표현이 다루지 않는다).
"""
import numpy as np
import torch
import torch.nn as nn

OBS_LEN, PRED_LEN = 50, 60
N_LANES, N_PTS = 20, 10
N_RULE = 10
K, HID = 6, 128
N_RPTS = 64
DT = 0.1

XY_SCALE = 50.0          # 경로 좌표는 100 m 를 넘어가므로 입력 전에 줄인다
LEN_SCALE = 100.0
BAND_SCALE = 3.6
THETA_MAX = np.pi / 2
# 스텝당 방향 변화 상한 [rad]. **이것이 실현가능성을 세우는 유일한 장치다.**
# 곡률 파라미터화는 방향이 유계 곡률의 적분이라 실현가능성이 공짜로 따라오지만,
# 각도 레벨에서 theta 를 스텝마다 독립으로 내면 그 보장이 사라진다 — 제한 없이 학습한
# L0 은 실측 |dtheta| 중앙 17.9°/step, p99 174°(= 1741°/s)로 궤적의 71.4% 가 실제
# 차량이 못 내는 것이었다. 라벨 전수조사의 정상값 상한이 7.3°/step (p99.99) 이라 8° 로 둔다.
DTHETA_MAX = np.radians(8.0)
A_SCALE = 8.0
V_MAX = 40.0


def interp1d(arr: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """arr (B,K,M,C) 를 idx (B,K,T) 위치에서 선형보간 -> (B,K,T,C).

    경로는 호길이 **등간격**으로 리샘플돼 있으므로 idx = s / len * (M-1) 로 바로 얻는다.
    idx 에 대해 미분 가능해야 한다 — s 가 액션 a 에서 나오므로 여기서 gradient 가 끊기면
    적분기 전체가 학습되지 않는다.
    """
    M, C = arr.size(2), arr.size(3)
    idx = idx.clamp(0.0, M - 1 - 1e-4)
    i0 = idx.floor()
    f = (idx - i0).unsqueeze(-1)
    i0 = i0.long().unsqueeze(-1).expand(-1, -1, -1, C)
    a = torch.gather(arr, 2, i0)
    b = torch.gather(arr, 2, (i0 + 1).clamp(max=M - 1))
    return a * (1 - f) + b * f


def route_point(routes: torch.Tensor, route_tan: torch.Tensor, s: torch.Tensor,
                route_len: torch.Tensor) -> torch.Tensor:
    """호길이 s 에 해당하는 중심선 위 점 (B,K,T,2). **경로 양 끝 밖은 끝 접선 방향으로 직선 연장**한다.

    interp1d 만 쓰면 idx 가 [0, M-1] 로 잘린다. 그러면
      - s0 < 0 인 모드(시작 차로가 차량보다 앞에서 시작한다 — lane_graph.candidate_lanes 의 탐색 반경이
        5 m 라서)는 s 가 0 을 넘을 때까지 경로 시작점에 얼어붙고,
      - 경로 길이보다 멀리 가는 모드는 끝점에 붙어 멈춘다.
    얼마나 자주 일어나는지는 여기 숫자로 적지 않고 runs/v4_reeval_s0.json 의 geometry·rollout_clamp_l0b 에
    둔다(src/reeval_v4.py 가 쓴다) — 주석과 측정이 서로 다른 숫자를 들고 있지 않게.
    연장하면 두 경우 모두 위치가 실제 진행대로 움직이고, 잘린 구간에서 끊기던 s 의 gradient 도 산다.
    접선·밴드 조회는 끝값을 유지하는 interp1d 를 그대로 쓴다 — 연장 구간은 직선이기 때문이다.
    """
    M = routes.size(2)
    L = route_len.clamp(min=1e-3).unsqueeze(-1)                 # (B,K,1)
    P = interp1d(routes, s / L * (M - 1))
    t0 = route_tan[:, :, :1, :]
    t1 = route_tan[:, :, -1:, :]
    t0 = t0 / t0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    under = torch.clamp(s, max=0.0).unsqueeze(-1)               # 시작점보다 뒤인 만큼 (음수)
    over = torch.relu(s - L).unsqueeze(-1)                      # 끝점을 지난 만큼
    return P + t0 * under + t1 * over


class V4Net(nn.Module):
    def __init__(self, in_dim=5, lane_in=N_PTS * 2 + N_RULE, hid=HID, k=K,
                 pred_len=PRED_LEN, out_dim=2, route_pts=N_RPTS, level="l3"):
        super().__init__()
        assert level in ("l2", "l3", "l0")
        self.level, self.k, self.pred_len, self.out_dim = level, k, pred_len, out_dim
        self.lane_in = lane_in

        # --- L2: 궤적 인코더 + 지도 cross-attention (기준선과 동일) ---
        self.traj_encoder = nn.LSTM(in_dim, hid, num_layers=2, batch_first=True)
        self.lane_encoder = nn.Sequential(
            nn.Linear(lane_in, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.attn = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU())

        if level == "l2":
            self.traj_head = nn.Linear(hid, k * pred_len * out_dim)
            self.prob_head = nn.Linear(hid, k)
        else:
            # --- L3: 후보 경로 하나를 임베딩 (좌표 2 + 접선 2 + 밴드 2 = 점당 6채널) ---
            self.route_encoder = nn.Sequential(
                nn.Linear(route_pts * 6 + 1, hid), nn.ReLU(), nn.Linear(hid, hid))
            # 같은 경로에 배정된 슬롯끼리 구별시키는 임베딩. 지도가 갈림길을 주지 않는
            # 직선 도로에서 모드가 통째로 붕괴하는 것을 막는 축이다 (종방향 다중성).
            self.sub_emb = nn.Embedding(k, hid)
            self.mode = nn.Sequential(nn.Linear(hid * 3, hid), nn.ReLU())
            self.traj_head = nn.Linear(hid, pred_len * (2 if level == "l0" else out_dim))
            self.prob_head = nn.Linear(hid, 1)

    # ------------------------------------------------------------------ L0 적분기
    def rollout(self, act, routes, route_tan, route_len, route_sd0, v0):
        """액션 (a, theta) -> 궤적. 순수 함수이고 미분 가능하다.

        누적합만으로 끝나 순차 루프가 없다 — 학습 경로와 추론 경로가 같아서
        exposure bias 가 구조적으로 0 이다.
        """
        B, Kk, T, _ = act.shape
        a = A_SCALE * torch.tanh(act[..., 0])                 # (B,K,T) 종방향 가속도
        # 잔차각은 값이 아니라 **변화율**로 낸다. 스텝당 |dtheta| <= DTHETA_MAX 이므로
        # 어떤 출력을 내도 요레이트가 물리 한계 안에 있다. 시작값 theta_0 는 현재
        # 진행방향과 경로 접선의 차인데, 정규화 프레임에서 진행방향이 0 이라 -k(s0) 다.
        M0 = routes.size(2)
        idx0 = route_sd0[..., 0:1] / route_len.clamp(min=1e-3).unsqueeze(-1) * (M0 - 1)
        t0 = interp1d(route_tan, idx0)
        th0 = -torch.atan2(t0[..., 1], t0[..., 0])            # (B,K,1)
        dth = DTHETA_MAX * torch.tanh(act[..., 1])
        th = (th0 + torch.cumsum(dth, dim=2)).clamp(-THETA_MAX, THETA_MAX)
        # 속도: 음수를 막는다(후진 미지원). clamp 대신 relu 라 gradient 가 살아 있다.
        v = torch.relu(v0.view(B, 1, 1) + torch.cumsum(a * DT, dim=2)).clamp(max=V_MAX)
        s = route_sd0[..., 0:1] + torch.cumsum(v * torch.cos(th) * DT, dim=2)
        d = route_sd0[..., 1:2] + torch.cumsum(v * torch.sin(th) * DT, dim=2)

        M = routes.size(2)
        idx = s / route_len.clamp(min=1e-3).unsqueeze(-1) * (M - 1)
        P = route_point(routes, route_tan, s, route_len)   # (B,K,T,2) 중심선 위 점, 양 끝은 직선 연장
        Tg = interp1d(route_tan, idx)
        Tg = Tg / Tg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        N = torch.stack([-Tg[..., 1], Tg[..., 0]], dim=-1)    # 좌측 법선
        traj = P + N * d.unsqueeze(-1)
        k_ang = torch.atan2(Tg[..., 1], Tg[..., 0])
        return traj, {"a": a, "theta": th, "dtheta": dth, "v": v, "s": s, "d": d,
                      "idx": idx, "h": k_ang + th}

    # ------------------------------------------------------------------
    def forward(self, x, lanes, lane_mask, lane_feat=None, routes=None,
                route_tan=None, route_band=None, route_len=None,
                route_sd0=None, route_mask=None, route_sub=None, v0=None):
        B = x.size(0)
        _, (h, _) = self.traj_encoder(x)
        traj_feat = h[-1]

        lane_in = lanes.reshape(B, N_LANES, N_PTS * 2)
        if self.lane_in > N_PTS * 2:
            if lane_feat is None:
                raise ValueError("규칙 피처(lane_feat)가 필요한 모델이다")
            lane_in = torch.cat([lane_in, lane_feat], dim=2)
        lane_emb = self.lane_encoder(lane_in)
        attn_out, _ = self.attn(traj_feat.unsqueeze(1), lane_emb, lane_emb,
                                key_padding_mask=(lane_mask == 0))
        fused = self.fuse(torch.cat([traj_feat, attn_out.squeeze(1)], dim=1))

        if self.level == "l2":
            traj = self.traj_head(fused).view(B, self.k, self.pred_len, self.out_dim)
            return traj, self.prob_head(fused), {}

        # --- L3: 경로별 모드 ---
        Kk, M = routes.size(1), routes.size(2)
        rin = torch.cat([(routes / XY_SCALE).reshape(B, Kk, M * 2),
                         route_tan.reshape(B, Kk, M * 2),
                         (route_band / BAND_SCALE).reshape(B, Kk, M * 2),
                         (route_len / LEN_SCALE).unsqueeze(-1)], dim=2)
        route_emb = self.route_encoder(rin)
        if route_sub is None:
            route_sub = torch.zeros(B, Kk, dtype=torch.long, device=routes.device)
        mode = self.mode(torch.cat([fused.unsqueeze(1).expand(-1, Kk, -1),
                                    route_emb, self.sub_emb(route_sub)], dim=2))
        head = self.traj_head(mode).view(B, Kk, self.pred_len, 2)
        logits = self.prob_head(mode).squeeze(-1)
        logits = logits.masked_fill(route_mask == 0, float("-inf"))

        if self.level == "l3":
            return head, logits, {}
        traj, aux = self.rollout(head, routes, route_tan, route_len, route_sd0, v0)
        aux["band"] = interp1d(route_band, aux["idx"])
        return traj, logits, aux


if __name__ == "__main__":
    B = 4
    b = dict(x=torch.randn(B, OBS_LEN, 8), lanes=torch.randn(B, N_LANES, N_PTS, 2),
             lane_mask=torch.ones(B, N_LANES), lane_feat=torch.randn(B, N_LANES, N_RULE),
             routes=torch.randn(B, K, N_RPTS, 2).cumsum(2),
             route_tan=torch.zeros(B, K, N_RPTS, 2), route_band=torch.full((B, K, N_RPTS, 2), 1.75),
             route_len=torch.full((B, K), 80.0), route_sd0=torch.zeros(B, K, 2),
             route_mask=torch.ones(B, K), route_sub=torch.arange(K).repeat(B, 1),
             v0=torch.full((B,), 10.0))
    b["route_tan"][..., 0] = 1.0
    for lv in ("l2", "l3", "l0"):
        m = V4Net(in_dim=8, level=lv)
        t, lg, aux = m(**b)
        n = sum(p.numel() for p in m.parameters())
        extra = ""
        if aux:
            extra = (f"  |d| max {aux['d'].abs().max():.2f}  v [{aux['v'].min():.1f},"
                     f"{aux['v'].max():.1f}]  |theta| max {aux['theta'].abs().max()*57.3:.0f}°")
        print(f"{lv:3} params {n:>9,}  traj {tuple(t.shape)}  logits {tuple(lg.shape)}{extra}")
    # 적분기 gradient 가 액션까지 흐르는지
    m = V4Net(in_dim=8, level="l0"); t, lg, aux = m(**b); t.sum().backward()
    g = m.traj_head.weight.grad
    print(f"적분기 gradient: traj_head.grad |max| {g.abs().max():.4f}  (0 이면 끊긴 것)")
