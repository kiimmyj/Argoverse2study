"""
train_v4.py - v4 를 레벨별로 누적 학습한다.

  L2 : python src/train_v4.py --level l2                       (= 기준선 재현)
  L3 : python src/train_v4.py --level l3
  L0 : python src/train_v4.py --level l0                       (흔들림 벌점 포함 — 아래)
  L4 : python src/train_v4.py --level l0 --offlane 1.0

L0 는 '액션 출력 + Frenet 적분기 + 흔들림 벌점'이다 (2026-09-16 사용자 결정).

**기본 벌점은 좌표 기준이다 (2026-10-06 사용자 결정)** — `--smooth-mode xy --smooth-xy 1.0`.
예측 좌표에서 복원한 진행방향의 스텝 변화가 7.3°/step 을 넘는 분량만 힌지로 벌한다(jitter_xy()).
8조건 × 3시드 = 24판 결과: 액션 벌점(jitter())을 쓰면 minADE6 1.440 / 좌표 위반율 0.76%,
좌표 벌점 1.0 은 **1.359 / 2.28%** 로 정확도 손해가 검출되지 않으면서 위반율은 벌점 없음(8.47%)의 0.29배다.
액션 벌점의 대가는 회전에 몰린다(직진 +0.055 vs 회전 +0.164 m) — 실제 회전까지 누른다.
자세한 근거는 docs/v4_notion.md 2.16절.

예전 판을 재현하려면 **모드를 명시한다** — 액션 벌점판은 `--smooth-mode action --smooth 1.0`,
벌점 없는 판은 `--smooth-mode action --smooth 0`.

손실·평가·하이퍼파라미터는 train_lane.py(기준선 minADE6 1.292) 와 같다.
다른 것은 모델과 Dataset 이 주는 필드뿐이라, 지표 차이가 나면 레벨 때문이다.

L4 를 왜 모든 모드에 거나
-------------------------
WTA 는 정답에 가장 가까운 하나에만 거리 손실을 건다. 나머지 K-1 개는 gradient 를 전혀
못 받는데, 도로를 벗어나는 예측은 사실상 전부 그 **버려진 모드들**에 있다.
L4 는 모든 모드에 개별로 벌점을 주므로 버려진 모드를 감독하는 유일한 항이다.
"""
import argparse, json, os, sys, time
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_lane import Av2LaneRuleDataset
from model_v4 import V4Net, N_PTS, N_RULE, A_SCALE, DTHETA_MAX

LABEL_DTHETA_DEG = 7.3     # 실제 차량의 스텝간 방향 변화 p99.99 (라벨 전수조사). 넘으면 못 내는 요레이트다
SMOOTH_L0_DEFAULT = 0.0    # 액션 벌점 기본 가중치. 2026-10-06 부터 기본 벌점은 좌표 기준(--smooth-mode xy)이라 0 이다
DATA_ROOT = "/data/argoverse2/motion_forecasting"
CACHE_ROOT = "/data/argoverse2/cache/v4"      # prepare_v4.py 가 캐시를 굽는 곳
ROUTE_KEYS = ("routes", "route_tan", "route_band", "route_len", "route_sd0",
              "route_mask", "route_sub", "v0", "h0")


def to_dev(b, device, level, use_rules):
    kw = dict(x=b["x"].to(device), lanes=b["lanes"].to(device),
              lane_mask=b["lane_mask"].to(device),
              lane_feat=b["lane_feat"].to(device) if use_rules else None)
    if level != "l2":
        kw.update({k: b[k].to(device) for k in ROUTE_KEYS})
    if "route_hist" in b and level != "l2":
        kw["route_hist"] = b["route_hist"].to(device)
    if "agents" in b:
        kw["agents"] = b["agents"].to(device)
        kw["agents_mask"] = b["agents_mask"].to(device)
    return kw


def loss_fn(traj, logits, y, mode_mask, aux, w_off, nonwinner_only=False):
    """WTA 거리 손실 + 모드 확률 + (선택) 차로 이탈 벌점.

    nonwinner_only=True 면 벌점을 **버려진 모드에만** 건다. 근거는 실측이다:
    기준 경로를 하나 고르면 정답 궤적의 14.05% step(시나리오의 28.7%)이 밴드를 벗어나
    벌점을 받는다. 승자 모드는 정의상 정답에 가장 가까우므로 거리 손실이 이미 옳게
    감독하고 있고, 거기에 벌점을 더하면 오탐만 얹힌다. 문서가 L4 에 부여한 목적도
    '버려진 K-1 개 감독' 이다 — 승자는 애초에 대상이 아니다.
    """
    B, Kk = traj.shape[:2]
    fde = torch.norm(traj[:, :, -1] - y.unsqueeze(1)[:, :, -1], dim=-1)
    fde = fde.masked_fill(mode_mask == 0, float("inf"))       # 없는 경로는 승자가 될 수 없다
    best = fde.argmin(dim=1)
    idx = best.view(B, 1, 1, 1).expand(B, 1, traj.size(2), traj.size(3))
    best_traj = traj.gather(1, idx).squeeze(1)
    loss = F.smooth_l1_loss(best_traj, y) + F.cross_entropy(logits, best)
    off = torch.zeros((), device=traj.device)
    if w_off > 0 and aux:
        # 밴드 밖으로 나간 만큼만 벌준다 (단측 hinge). 안쪽은 어디에 있든 0 이다 —
        # '중심에 가까울수록 좋다'로 쓰면 편도 2차선에서 중앙선이 최적해가 된다.
        d, band = aux["d"], aux["band"]
        over = torch.relu(d - band[..., 0]) + torch.relu(-d - band[..., 1])
        m = mode_mask
        if nonwinner_only:
            m = m * (1.0 - F.one_hot(best, m.size(1)).float())
        m = m.unsqueeze(-1)
        off = (over * m).sum() / m.sum().clamp(min=1) / over.size(2)
        loss = loss + w_off * off
    return loss, off.detach()


def jitter_steps(aux):
    """(B,K,T-1) 스텝별 흔들림 = Δ(a / A_SCALE)² + Δ(dθ / DTHETA_MAX)²."""
    ua = aux["a"] / A_SCALE
    ut = aux["dtheta"] / DTHETA_MAX
    return (ua[..., 1:] - ua[..., :-1]) ** 2 + (ut[..., 1:] - ut[..., :-1]) ** 2


LANE_W_M = 3.42          # AV2 차로 폭 중앙값 — 횡 목표 오프셋의 크기


def lat_target_loss(aux, mode_mask, route_sub, lane_w=LANE_W_M):
    """같은 경로에 배정된 **둘째·셋째 슬롯**을 옆 차로 쪽으로 밀어 두는 보조 손실 (2026-09-30).

    왜 — 차선변경 분석(docs/v4_lane_change.md): 같은 경로의 하위 모드들이 끝 s 는 11.32 m 퍼지는데
    끝 d 는 **0.31 m** 밖에 안 퍼진다. 모드 축이 종방향으로만 서 있어서, 차선변경이 표현 가능한
    밴드(±3.6 m) 안에 있어도 모델이 옆으로 가는 모드를 아예 만들지 않는다(1위 적중 12.4%).
    sub 임베딩은 이미 있지만 '무엇을 뜻하는지'를 아무도 알려 주지 않는다 — 여기서 알려 준다.

    sub=0 은 건드리지 않는다(정답이 차로 유지인 경우가 대부분이라 그쪽을 흔들면 손해다).
    sub>=1 은 ±lane_w 로 번갈아 목표를 준다. 힌지가 아니라 제곱이지만 가중치를 작게 준다.
    """
    d_end = aux["d"][..., -1]                                  # (B,K) 끝 횡오프셋
    # 방향은 **슬롯 번호**로 번갈아 준다. sub 로 나누면 구별 경로가 3개일 때 sub 가 0·1 뿐이라
    # 둘째 묶음 전체가 같은 쪽(왼쪽)만 겨냥하게 된다.
    slot = torch.arange(d_end.size(1), device=d_end.device).view(1, -1).expand_as(d_end)
    sgn = torch.where(slot % 2 == 0, 1.0, -1.0)
    tgt = torch.where(route_sub >= 1, sgn * lane_w, torch.zeros_like(sgn))
    m = mode_mask * (route_sub >= 1).float()
    return (((d_end - tgt) / lane_w) ** 2 * m).sum() / m.sum().clamp(min=1)


def jitter_xy(traj, mode_mask, thresh_deg=LABEL_DTHETA_DEG, v_min=1.0, dt=0.1):
    """**예측 좌표**에서 복원한 진행방향의 스텝 변화가 라벨 상한을 넘는 만큼만 벌한다 (2026-09-30).

    왜 이 형태인가 — 2026-09-29 측정: 지금 벌점이 누르는 dθ 를 통째로 지워도 좌표 기준 위반율이
    0.735% → 0.733% 로 **0.3% 밖에 안 움직인다**. 좌표 방향 변화는 Δψ = Δθ + Δk(경로 곡률) + 기하 보정이고
    주항은 기하다. 즉 지금 벌점은 궤적에 거의 나타나지 않는 변수를 누르고 있다.
    문헌도 같은 방향이다 — 매끄러움·실현성 지표는 전부 좌표에서 나온다(WOSAC, PTNet, MultiPath++ TRI-c,
    Greer YawLoss). 모델 내부 출력에 건 지표는 구성상 항등식이 되기 쉽다(MultiPath++ TRI-h 0.00% ↔ TRI-c 1.22%).

    정지 구간은 방향이 정의되지 않으므로 **양쪽 스텝 속력이 v_min 이상일 때만** 센다(평가 지표와 같은 규칙).
    힌지라 임계 안에서는 0 이다 — 실제 회전은 누르지 않는다.
    """
    d = traj[..., 1:, :] - traj[..., :-1, :]                      # (B,K,T-1,2)
    sp = d.norm(dim=-1) / dt
    psi = torch.atan2(d[..., 1], d[..., 0])
    dpsi = torch.remainder(psi[..., 1:] - psi[..., :-1] + np.pi, 2 * np.pi) - np.pi
    lim = float(np.radians(thresh_deg))
    over = (dpsi.abs() - lim).clamp(min=0) / lim                  # 임계 초과분만, 무차원
    ok = ((sp[..., 1:] >= v_min) & (sp[..., :-1] >= v_min)).float()
    m = mode_mask.unsqueeze(-1) * ok
    return (over ** 2 * m).sum() / m.sum().clamp(min=1)


LANE_YAW_TOL_DEG = 15.0    # 정답 라벨 p95 (경로가 3 m 안인 경우) — src/lane_yaw_stats.py 측정
LANE_YAW_D_MAX = 3.0       # |d| 가 이보다 크면 그 모드는 자기 경로 위에 있지 않다 — 접선각 비교가 무의미하다
LANE_YAW_CLAMP = 2.0       # 허용 오차의 몇 배까지 셀지 — 180° 이상치가 gradient 를 독점하지 않게 막는다


def lane_yaw_loss(traj, aux, mode_mask, tol_deg=LANE_YAW_TOL_DEG, d_max=LANE_YAW_D_MAX,
                  v_min=1.0, dt=0.1, clamp=LANE_YAW_CLAMP):
    """**예측 좌표**의 진행방향이 그 모드가 타는 경로의 접선각에서 벗어난 만큼을 힌지로 벌한다 (2026-10-06).

    출처 — Greer et al., "Trajectory Prediction in Autonomous Driving with a Lane Heading Auxiliary Loss"
    (arXiv:2011.06679) 의 YawLoss. 세 가지를 그대로 따른다: ① 연속한 **예측 좌표 두 점**의 arctan 으로
    진행방향을 만들고 ② 허용 오차 안에서는 0 인 힌지로 벌하고 ③ 승자뿐 아니라 **모든 살아있는 모드**에 건다
    (그 논문이 근거를 들어 주장하는 성질이다 — 버려지는 모드는 거리 손실의 감독을 못 받는다).

    왜 넣나 — 24판 측정에서 모델의 |Δψ| 중앙값이 직진 0.06° → 회전 0.16° 로 2.7배밖에 안 커진다
    (정답은 0.12° → 1.10°, 9배). jitter_xy 는 '급변 금지'만 걸 뿐 **어디로 돌아야 하는지**는 말하지 않는다.
    경로 접선각은 회전에서 실제로 돌아가므로, 거기에 정렬시키면 회전 구조가 생긴다.
    적분기 안에서는 h = k(s) + θ 라 정의상 정렬돼 있지만 **좌표에서 복원한 ψ 는 다르다** — 그 차이(기하 항)가
    위반의 주항이었다(기여 75%, 2.14절). 이 손실은 그 항을 직접 겨냥한다.

    허용 오차 15° 는 정답 라벨 분포에서 왔다 — 경로가 3 m 안일 때 |ψ_gt − k| 의 p95 다
    (`src/lane_yaw_stats.py`: 전체 p50 1.11° / p90 10.41° / p95 21.37°). 실제 차선변경·코너 커팅을
    벌하지 않으려면 이 정도가 필요하다.
    """
    d = traj[..., 1:, :] - traj[..., :-1, :]                      # (B,K,T-1,2)
    sp = d.norm(dim=-1) / dt
    psi = torch.atan2(d[..., 1], d[..., 0])
    k = aux["k"]                                                  # (B,K,T) 경로 접선각
    # 세그먼트의 접선각 = 양 끝 각의 원형 평균 (YawLoss 는 중점의 차선 heading 을 쓴다)
    kc = torch.cos(k[..., 1:]) + torch.cos(k[..., :-1])
    ks = torch.sin(k[..., 1:]) + torch.sin(k[..., :-1])
    km = torch.atan2(ks, kc)
    err = torch.remainder(psi - km + np.pi, 2 * np.pi) - np.pi
    lim = float(np.radians(tol_deg))
    over = ((err.abs() - lim).clamp(min=0) / lim).clamp(max=clamp)
    on_route = (aux["d"][..., 1:].abs() <= d_max) & (aux["d"][..., :-1].abs() <= d_max)
    ok = (sp[..., :] >= v_min) & on_route
    m = mode_mask.unsqueeze(-1) * ok.float()
    return (over ** 2 * m).sum() / m.sum().clamp(min=1)


def jitter(aux, mode_mask):
    """흔들림 벌점 — 액션 (a, dθ) 의 이웃 스텝 차이 제곱 평균 (살아있는 모든 모드).

    DTHETA_MAX 레이트 상한은 **크기만** 막고, 부호가 매 스텝 뒤집히는 진동은 막지 않는다.
    dθ 가 ±8° 로 번갈아 나오거나 a 가 가속·감속을 번갈아 내도 위치는 수 cm 밖에 안 흔들려서
    거리 손실이 이 방향을 감독하지 못한다. 전체 데이터(스텝 4배)로 학습한 (a, h) 판에서 두 채널이
    모두 상한까지 포화한 채 지그재그했고, 승자 모드도 마찬가지였다 — 그래서 L4 와 달리 승자를 빼지 않는다.

    두 채널을 각자 출력 상한으로 나눠 단위를 없앤다. 눈금: 1°/step² 나 1 m/s²/step(저크 10 m/s³) 은
    채널당 (1/8)² ≈ 0.016 이고, 상한끼리 뒤집는 지그재그는 채널당 4 다.
    """
    j = jitter_steps(aux)
    m = mode_mask.unsqueeze(-1)
    return (j * m).sum() / m.sum().clamp(min=1) / j.size(2)


@torch.no_grad()
def evaluate(model, loader, device, level, use_rules):
    model.eval()
    ade = fde = off = n = 0.0
    dth = []                       # 스텝간 |dtheta| — 궤적이 지그재그인지 재는 계기.
    exc = live = jit = 0.0         # 살아있는 모드만: 라벨 상한 초과 step 수, step 수, 흔들림 합
    for b in loader:
        y = b["y"].to(device)
        traj, _, aux = model(**to_dev(b, device, level, use_rules))
        mm = (b["route_mask"].to(device) if level != "l2"
              else torch.ones(traj.shape[:2], device=device))
        dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
        big = torch.full_like(dist[:, :, 0], float("inf"))
        a = torch.where(mm > 0, dist.mean(2), big).min(1).values
        f = torch.where(mm > 0, dist[:, :, -1], big).min(1).values
        ade += a.sum().item(); fde += f.sum().item(); n += y.size(0)
        if aux:
            o = (torch.relu(aux["d"] - aux["band"][..., 0])
                 + torch.relu(-aux["d"] - aux["band"][..., 1]))
            off += ((o > 0).float() * mm.unsqueeze(-1)).sum().item() / max(o.size(2), 1)
            # dθ p99 는 빈 슬롯까지 섞인 기존 계기라 그대로 둔다. 실현가능성 판정은 아래의
            # 살아있는 모드 기준 초과 비율로 한다 — p99 는 초과가 1% 를 넘으면 상한에 붙어 더 안 움직인다.
            th = aux["theta"]
            dd = (th[:, :, 1:] - th[:, :, :-1]).abs() * 180.0 / np.pi
            dth.append(dd.flatten())
            lm = mm.unsqueeze(-1)
            exc += ((dd > LABEL_DTHETA_DEG).float() * lm).sum().item()
            live += lm.sum().item() * dd.size(2)
            jit += (jitter_steps(aux) * lm).sum().item()
    q = (float(torch.cat(dth).median()), float(torch.cat(dth).quantile(0.99))) if dth else (0.0, 0.0)
    feas = {"dtheta_over_label_pct": 100.0 * exc / max(live, 1.0), "jitter": jit / max(live, 1.0)}
    return ade / n, fde / n, off / n, q, feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="l3", choices=["l2", "l3", "l0"])
    ap.add_argument("--offlane", type=float, default=0.0, help="L4 벌점 가중치")
    ap.add_argument("--off-nonwinner", dest="off_nonwinner", type=int, default=0,
                    help="1 이면 L4 를 버려진 모드에만 건다 (승자는 거리 손실이 감독)")
    ap.add_argument("--rules", type=int, default=1)
    ap.add_argument("--theta", type=int, default=1)
    ap.add_argument("--lane-smooth", dest="lane_smooth", type=float, default=0.0,
                    help="중심선 평활 sigma [m] (0=끔). k 가 원시 폴리라인 중앙차분이라 계단이다")
    ap.add_argument("--route-hist", dest="route_hist", type=int, default=0,
                    help="경로별 관측 이력 스텝 수 (0=끔, 10=관측 5초를 2 Hz 로). "
                         "route encoder 에 (H x 5) 를 덧붙인다 — d, 종진행, sin/cos θ, valid")
    ap.add_argument("--h-src", dest="h_src", default="build", choices=["av2", "build"])
    ap.add_argument("--fallback", default="straight1", choices=["straight1", "straight6", "fan"],
                    help="지도가 경로를 못 주는 시나리오를 무엇으로 채울지. "
                         "straight1=직진1개(mask 1) / straight6=직진6복제(모드예산만) / fan=부채꼴6개")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-sched", dest="lr_sched", default="const", choices=["const", "cosine"],
                    help="학습률 스케줄. const = 끝까지 --lr 고정(기존). cosine = 스텝마다 코사인으로 --lr-min 까지 낮춘다 "
                         "— 고정 학습률로는 에폭 사이 val 요동이 남아 수렴한 가중치를 못 얻는다")
    ap.add_argument("--lr-min", dest="lr_min", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--smooth-mode", dest="smooth_mode", default="xy",
                    choices=["action", "xy", "both"],
                    help="흔들림 벌점을 무엇에 걸지 — xy=예측 좌표의 방향 변화 힌지(기본, 2026-10-06), "
                         "action=액션 (a, dθ) 차분(옛 기본), both=둘 다")
    ap.add_argument("--smooth-xy", dest="smooth_xy", type=float, default=1.0,
                    help="좌표 기준 벌점 가중치. 1.0 = 정확도 손해 없이 위반율 0.29배, 3.0 = 1.13%% (24판 측정)")
    ap.add_argument("--lane-yaw", dest="lane_yaw", type=float, default=0.0,
                    help="차선 heading 정렬 보조 손실 가중치 (YawLoss, arXiv:2011.06679) — 예측 좌표의 진행방향을 "
                         "그 모드 경로의 접선각에 %g° 안으로 맞춘다. 0 이면 끈다" % LANE_YAW_TOL_DEG)
    ap.add_argument("--lat-modes", dest="lat_modes", type=float, default=0.0,
                    help="횡 모드 축 보조 손실 가중치 — 같은 경로의 둘째·셋째 슬롯을 ±차로폭으로 민다")
    ap.add_argument("--cleanse", type=int, default=0,
                    help="1 이면 전처리 결함 세 가지를 고친다 — 차로 거리(점-선분), 뒤집힌 focal heading, 뒤집힘 판정 게이트")
    ap.add_argument("--agents", type=int, default=0,
                    help="1 이면 주변 차량(가까운 32대 과거 궤적)을 입력에 붙인다")
    ap.add_argument("--agents-radius", dest="agents_radius", type=float, default=30.0,
                    help="주변 차량 반경 [m] — 사용자가 고른 단순 기준")
    ap.add_argument("--agents-step", dest="agents_step", type=int, default=5,
                    help="주변 차량 과거를 몇 스텝마다 뽑을지 (5 = 2 Hz)")
    ap.add_argument("--input", default="raw5", choices=["raw5", "ah2", "ah2_2hz", "ah3", "ah3_2hz", "ah3n", "ah3s", "ah3p", "ah3r"],
                    help="raw5=(x,y,vx,vy,h_AV2) / ah2=(a, h) 2채널, h 는 위치차분 진행방향 / "
                         "ah2_2hz=같은 (a, h) 를 위치 평활 뒤 2 Hz 로 뽑은 10스텝, "
                         "ah3/ah3_2hz=평활판(가우시안 σ0.25 s + 속력 가중 혼합, 램프 4스텝 버림)")
    ap.add_argument("--th0", default="current", choices=["current", "guard"],
                    help="적분기 시작 잔차각. guard=wrap(h0-k(s0)), |값|>90° 면 current")
    ap.add_argument("--smooth", type=float, default=None,
                    help="**액션** 흔들림 벌점 가중치 — (a, dθ) 의 스텝간 변화 제곱 (jitter). "
                         "기본 0 이다(기본 벌점은 좌표 기준). --smooth-mode action|both 와 함께 줄 때만 쓴다")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-every", dest="save_every", type=int, default=1,
                    help="N 에폭마다 체크포인트를 <outdir>/ckpt/<tag>/epNN.pth 로 남긴다 — 에폭별 시각화"
                         "(모델이 어느 방향으로 학습되는지)에 쓴다. 0 이면 끈다. best 는 따로 lstm_<tag>.pth")
    ap.add_argument("--cache", action="store_true",
                    help="prepare_v4.py 가 구운 전처리 캐시로 학습한다 (원본과 bit-exact, 에폭마다 하던 전처리가 사라진다)")
    ap.add_argument("--cache-root", dest="cache_root", default=CACHE_ROOT)
    args = ap.parse_args()
    if args.smooth is None:
        args.smooth = SMOOTH_L0_DEFAULT if args.level == "l0" else 0.0

    use_rules = bool(args.rules)
    tag = args.tag or f"v4_{args.level}" + (f"_off{args.offlane:g}" if args.offlane else "") \
        + (f"_sm{args.smooth:g}" if args.smooth and args.smooth_mode in ("action", "both") else "") \
        + (f"_smxy{args.smooth_xy:g}" if args.smooth_xy and args.smooth_mode in ("xy", "both") else "") \
        + (f"_ly{args.lane_yaw:g}" if args.lane_yaw else "") \
        + (f"_{args.lr_sched}{args.epochs}" if args.lr_sched != "const" else "") + f"_s{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    lim = None if args.limit == 0 else args.limit
    # **prepare_v4.dataset_kwargs() 와 글자 그대로 같아야 한다.** 캐시 키는 이 dict 의 JSON 이라
    # 한쪽에만 키가 늘면 (값이 기본값이어도) 해시가 달라져 모든 --cache 판이 '캐시 없음' 으로 죽는다.
    # 2026-10-06 에 route_hist·lane_smooth 를 prepare_v4 에만 넣어 실제로 그렇게 됐다.
    dkw = dict(with_rules=use_rules, theta_ch=bool(args.theta), h_src=args.h_src,
               routes=args.level != "l2", fallback=args.fallback, input_repr=args.input,
               cleanse=args.cleanse, route_hist=args.route_hist, lane_smooth=args.lane_smooth)
    if args.cache:
        # prepare_v4.py 와 같은 규칙(전처리 설정 + 소스 해시)으로 캐시를 찾는다. 소스를 고쳤으면 키가
        # 달라져 못 찾으므로, 낡은 캐시로 학습하는 일은 구조적으로 막힌다.
        from dataset_cached import CachedV4Dataset, open_cache
        src = os.path.dirname(os.path.abspath(__file__))
        cdir = {sp: open_cache(args.cache_root, sp, dkw, src, n)
                for sp, n in (("train", args.limit), ("val", args.val_limit))}
        miss = [sp for sp, d in cdir.items() if d is None]
        if miss:
            extra = f" --input {args.input}" if getattr(args, "input", None) else ""
            raise SystemExit(f"캐시 없음 {miss} — 먼저 같은 인자로 prepare_v4.py 를 돌려라: "
                             f"python src/prepare_v4.py --level {args.level} --theta {args.theta} "
                             f"--rules {args.rules} --h-src {args.h_src} --fallback {args.fallback} "
                             f"--limit {args.limit} --val-limit {args.val_limit}{extra}")
        tr, va = CachedV4Dataset(cdir["train"]), CachedV4Dataset(cdir["val"])
        print(f"[cache] train {cdir['train']}", flush=True)
        print(f"[cache] val   {cdir['val']}", flush=True)
    else:
        tr = Av2LaneRuleDataset(DATA_ROOT, "train", lim, **dkw)
        va = Av2LaneRuleDataset(DATA_ROOT, "val", args.val_limit, **dkw)
    agents_in = 0
    if args.agents:
        # 주변 차량을 붙인다 (2026-09-30). 전처리는 이미 구워 둔 캐시를 쓴다 —
        # t=0 에 관측되는 가까운 32대 · 과거 50스텝 · focal 프레임 · 미래 미사용.
        from dataset_agents import attach_agents
        tr, dtr = attach_agents(tr, "train", len(tr), radius_m=args.agents_radius, step=args.agents_step)
        va, dva = attach_agents(va, "val", len(va), radius_m=args.agents_radius, step=args.agents_step)
        agents_in = len(tr.store.idx) * 5
        print(f"[agents] train {dtr}\n[agents] val   {dva}\n"
              f"[agents] 반경 {args.agents_radius:g} m · {len(tr.store.idx)}스텝 · 채널 5 → {agents_in}", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, generator=g,
                    num_workers=args.workers, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, persistent_workers=True)

    lane_in = N_PTS * 2 + (N_RULE if use_rules else 0)
    in_dim = (2 if args.input.startswith(("ah2", "ah3")) else 5) + (3 if args.theta else 0)
    # route_hist 는 경로마다 (H x 5) 라 route encoder 입력이 그만큼 늘어난다.
    rh_in = args.route_hist * 5 if args.route_hist else 0
    model = V4Net(in_dim=in_dim, lane_in=lane_in, level=args.level, th0_mode=args.th0,
                  agents_in=agents_in, route_hist_in=rh_in).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] {device} | level {args.level} | offlane {args.offlane} | "
          f"fallback {args.fallback} | input {args.input} | th0 {args.th0} | in_dim {in_dim} "
          f"| route_hist {args.route_hist} | lane_smooth {args.lane_smooth:g} "
          f"| params {npar:,} | train {len(tr)} val {len(va)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = None
    if args.lr_sched == "cosine":      # 스텝 단위. const 면 스케줄러를 만들지 않아 기존과 수치가 같다
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs * len(tl), eta_min=args.lr_min)
    hist, best = [], (1e9, 1e9)
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = seen = 0.0; roff = 0.0; rsm = 0.0
        for b in tl:
            y = b["y"].to(device)
            traj, logits, aux = model(**to_dev(b, device, args.level, use_rules))
            mm = (b["route_mask"].to(device) if args.level != "l2"
                  else torch.ones(traj.shape[:2], device=device))
            loss, off = loss_fn(traj, logits, y, mm, aux, args.offlane,
                                bool(args.off_nonwinner))
            if args.smooth > 0 and aux and args.smooth_mode in ("action", "both"):
                sm = jitter(aux, mm)
                loss = loss + args.smooth * sm
                rsm += float(sm.detach()) * y.size(0)
            if args.lane_yaw > 0 and aux:
                loss = loss + args.lane_yaw * lane_yaw_loss(traj, aux, mm)
            if args.lat_modes > 0 and aux:
                loss = loss + args.lat_modes * lat_target_loss(aux, mm, b["route_sub"].to(device))
            if args.smooth_mode in ("xy", "both") and args.smooth_xy > 0:
                # 기본 벌점 (2026-10-06). 액션이 아니라 **예측 좌표**에서 복원한 방향을 누른다 —
                # 액션 벌점은 회전까지 눌러 회전 시나리오에서 +0.164 m 를 내준다(24판 측정, 2.16절).
                loss = loss + args.smooth_xy * jitter_xy(traj, mm)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sched is not None:
                sched.step()
            run += loss.item() * y.size(0); roff += float(off) * y.size(0); seen += y.size(0)
        ade, fde, ooff, (dth50, dth99), feas = evaluate(model, vl, device, args.level, use_rules)
        hist.append({"epoch": epoch, "loss": run / seen, "offlane": roff / seen,
                     "smooth": rsm / seen,
                     "minADE6": ade, "minFDE6": fde, "val_offlane_steps": ooff,
                     "dtheta_p50_deg": dth50, "dtheta_p99_deg": dth99,
                     "val_dtheta_over_label_pct": feas["dtheta_over_label_pct"],
                     "val_jitter": feas["jitter"],
                     "lr": opt.param_groups[0]["lr"],
                     "sec": time.time() - t0})
        if ade < best[0]:
            best = (ade, fde)
            torch.save(model.state_dict(), f"{args.outdir}/lstm_{tag}.pth")
        if args.save_every and (epoch % args.save_every == 0 or epoch == args.epochs):
            # 저장은 학습 수치(RNG·가중치)를 건드리지 않는다. 판당 1.8 MB × 에폭 수
            ck = f"{args.outdir}/ckpt/{tag}"
            os.makedirs(ck, exist_ok=True)
            torch.save(model.state_dict(), f"{ck}/ep{epoch:02d}.pth")
        print(f"[{tag}] {epoch:02d}/{args.epochs} loss {run/seen:.4f} | minADE6 {ade:.3f} | "
              f"minFDE6 {fde:.3f} | 이탈 {ooff:.2f} | dθ p99 {dth99:.1f}° | "
              f"{LABEL_DTHETA_DEG:g}°초과 {feas['dtheta_over_label_pct']:.2f}% | "
              f"흔들림 {feas['jitter']:.3f} | lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s", flush=True)

    json.dump({"tag": tag, "level": args.level, "offlane": args.offlane,
               "off_nonwinner": bool(args.off_nonwinner),
               "lane_in": lane_in, "in_dim": in_dim, "params": npar,
               "args": vars(args), "history": hist,
               "best_minADE6": best[0], "best_minFDE6": best[1]},
              open(f"{args.outdir}/{tag}.json", "w"), indent=2)
    print(f"[{tag}] BEST minADE6 {best[0]:.3f} | minFDE6 {best[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
