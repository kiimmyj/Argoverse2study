"""compare_v4_ah2.py - (a, h) 판 학습 후 분석 — 학습 로그 비교표 + 체크포인트 직접 재측정.

비교 대상 주의: v4_l0b / v4_l4_nw 는 5채널 입력 · θ₀ current · route_point 이전 적분기로 학습됐다.
(a, h) 판과는 입력·θ₀ 규칙·적분기 기하가 한꺼번에 달라서 입력 효과만 따로 읽으면 안 된다.
재측정 표는 네 체크포인트를 모두 현재 코드(route_point 포함)로 굴린 값이라 적분기 기하만은 맞춘 비교다.
실현가능성은 로그의 dθ p99 로 판정하지 않는다 — 초과 step 비율을 직접 센다('0%' 오기 재발 방지).

사용: python src/compare_v4_ah2.py [val 시나리오 수=300] [workers=2]
"""
import sys, json, statistics as st
import numpy as np, torch
sys.path.insert(0, "/home/user/Argoverse2study/src")
from torch.utils.data import DataLoader
from dataset_lane import Av2LaneRuleDataset
from model_v4 import V4Net
from train_v4 import to_dev

ROOT = "/home/user/Argoverse2study"
RUNS = [("v4_l0b_s0", "L0 5채널"), ("v4_l4_nw_s0", "L4 5채널"),
        ("v4_l0_ah2_s0", "L0 (a,h)"), ("v4_l4nw_ah2_s0", "L4 (a,h)")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2

print("=== 학습 로그 (val 2,000) ===")
print(f"{'run':10} {'best ADE':>8} {'best FDE':>8} {'최근5 ADE':>9} {'최근5 FDE':>9} {'최근5 이탈':>9} "
      f"{'dθp99(끝)':>9} {'에폭 중앙':>8} {'workers':>7}")
for tag, name in RUNS:
    d = json.load(open(f"{ROOT}/runs/{tag}.json")); h = d["history"]; l5 = h[-5:]
    print(f"{name:10} {d['best_minADE6']:8.3f} {d['best_minFDE6']:8.3f} "
          f"{st.mean(r['minADE6'] for r in l5):9.3f} {st.mean(r['minFDE6'] for r in l5):9.3f} "
          f"{st.mean(r['val_offlane_steps'] for r in l5):9.2f} {h[-1].get('dtheta_p99_deg', float('nan')):8.1f}° "
          f"{st.median(r['sec'] for r in h):7.0f}s {d['args']['workers']:7d}")
print("(에폭 시간은 workers·동시 실행 수가 판마다 달라 벽시계 비교용이 아니다)")

print(f"\n=== 체크포인트 재측정 — 현재 코드, val {N}, workers {W} ===")
print(f"{'run':10} {'minADE6':>8} {'minFDE6':>8} {'dθ>7.3° step':>13} {'dθ max':>7} {'밴드 이탈':>8} {'첫스텝 θ=±90°':>13}")
for tag, name in RUNS:
    sd = torch.load(f"{ROOT}/runs/lstm_{tag}.pth", map_location="cpu")
    in_dim = sd["traj_encoder.weight_ih_l0"].shape[1]; ah2 = in_dim == 2
    ds = Av2LaneRuleDataset("/data/argoverse2/motion_forecasting", "val", N, with_rules=True, routes=True,
                            fallback="straight1", input_repr="ah2" if ah2 else "raw5")
    m = V4Net(in_dim=in_dim, lane_in=30, level="l0", th0_mode="guard" if ah2 else "current").cuda()
    m.load_state_dict(sd); m.eval()
    ade = fde = off = 0.0; n = 0; D = []; slide = 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=16, num_workers=W):
            traj, _, aux = m(**to_dev(b, "cuda", "l0", True))
            y = b["y"].cuda(); mm = b["route_mask"].cuda(); live = mm > 0
            dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
            big = torch.full_like(dist[:, :, 0], float("inf"))
            ade += torch.where(live, dist.mean(2), big).min(1).values.sum().item()
            fde += torch.where(live, dist[:, :, -1], big).min(1).values.sum().item()
            o = torch.relu(aux["d"] - aux["band"][..., 0]) + torch.relu(-aux["d"] - aux["band"][..., 1])
            off += ((o > 0).float() * mm.unsqueeze(-1)).sum().item() / o.size(2)
            th = aux["theta"]; dd = (th[:, :, 1:] - th[:, :, :-1]).abs() * 180 / np.pi
            D.append(dd[live.unsqueeze(-1).expand_as(dd)].cpu())
            slide += int(((th[:, :, 0].abs() >= np.pi / 2 - 1e-4) & live).sum())
            n += y.size(0)
    D = torch.cat(D).numpy()
    print(f"{name:10} {ade/n:8.4f} {fde/n:8.4f} {(D > 7.3).mean()*100:12.2f}% {D.max():6.2f}° "
          f"{off/n:8.2f} {slide:13d}")
print("(밴드 이탈 = 시나리오당 살아있는 모드별 '밴드 밖 step 비율'의 합. 첫스텝 θ=±90° = 옆으로 미끄러지며 출발한 모드 수)")
