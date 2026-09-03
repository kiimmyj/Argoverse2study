"""
export_trajectories.py - 학습된 v3 모델로 궤적을 뽑아 파일로 저장한다.

시각화(PNG)와 별개로, 예측 궤적 자체를 다른 작업에서 쓸 수 있게 npz 로 떨군다.
정규화 좌표(focal 기준)와 city 좌표 복원에 필요한 origin/theta 를 함께 저장한다.

사용:
  python src/export_trajectories.py                          # v3, val 2000개
  python src/export_trajectories.py --ckpt runs/lstm_raw5.pth --limit 500
  python src/export_trajectories.py --split val --out runs/traj_v3.npz
"""
import argparse, sys, time
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_map import Av2MapDataset, _rotation_matrix
from model_map import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"


def denorm(arr_n, origin, theta):
    """정규화 좌표 → city 좌표. arr_n:(...,2)"""
    R = _rotation_matrix(theta)
    return arr_n @ R.T + origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lstm_map.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/traj_v3.npz")
    ap.add_argument("--lane", type=int, default=-1,
                    help="차로규칙 모델: lane_encoder 입력 차원(20=규칙없음, 30=규칙포함). -1이면 dataset_map 사용")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_lane = args.lane > 0
    if use_lane:
        from dataset_lane import Av2LaneRuleDataset
        from model_lane import LSTMMapRule
        ds = Av2LaneRuleDataset(DATA_ROOT, args.split, limit=args.limit,
                                with_rules=(args.lane == 30))
        model = LSTMMapRule(lane_in=args.lane).to(device)
    else:
        ds = Av2MapDataset(DATA_ROOT, args.split, limit=args.limit)
        model = LSTMSeq2Seq().to(device)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"ckpt {args.ckpt} | {args.split} {len(ds)}개 | device {device}", flush=True)

    H, G, P, PR, OR, TH, SID = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            if use_lane:
                lf = b["lane_feat"].to(device) if args.lane == 30 else None
                traj, logits = model(x, b["lanes"].to(device), b["lane_mask"].to(device), lf)
            else:
                traj, logits = model(x, b["lanes"].to(device), b["lane_mask"].to(device))
            H.append(b["x"][:, :, :2].numpy())            # (B,50,2) 과거
            G.append(b["y"].numpy())                      # (B,60,2) 정답
            P.append(traj.cpu().numpy())                  # (B,6,60,2) 예측
            PR.append(F.softmax(logits, dim=1).cpu().numpy())
            OR.append(b["origin"].numpy()); TH.append(b["theta"].numpy())
            SID += list(b["scenario_id"])
    hist, gt, pred = np.concatenate(H), np.concatenate(G), np.concatenate(P)
    probs, origin, theta = np.concatenate(PR), np.concatenate(OR), np.concatenate(TH)

    d = np.linalg.norm(pred - gt[:, None], axis=3)        # (N,6,60)
    ade_k, fde_k = d.mean(axis=2), d[:, :, -1]
    n = len(gt)
    best = fde_k.argmin(axis=1)          # 시각화용 '최선 모드' (끝점 기준)
    # AV2 규약: minADE 와 minFDE 는 각각 독립적으로 K 중 최소를 취한다
    # (train_map.py / train_motion.py 의 evaluate() 와 같은 정의)
    min_ade = ade_k.min(axis=1)
    min_fde = fde_k.min(axis=1)

    np.savez_compressed(
        args.out, scenario_id=np.array(SID), hist=hist.astype(np.float32),
        gt=gt.astype(np.float32), pred=pred.astype(np.float32),
        probs=probs.astype(np.float32), origin=origin.astype(np.float32),
        theta=theta.astype(np.float32), best=best.astype(np.int16),
        min_ade=min_ade.astype(np.float32), min_fde=min_fde.astype(np.float32))

    print(f"\n=== v3 궤적 추출 완료 ({time.time()-t0:.0f}s) ===")
    print(f"  시나리오        {n}")
    print(f"  minADE6        {min_ade.mean():.3f} m   (중앙 {np.median(min_ade):.3f})")
    print(f"  minFDE6        {min_fde.mean():.3f} m   (중앙 {np.median(min_fde):.3f})")
    print(f"  MissRate@2m    {(min_fde > 2.0).mean()*100:.1f}%")
    print(f"  최선 모드 분포   {np.bincount(best, minlength=6)}")
    print(f"\n  저장 -> {args.out}")
    print(f"    hist (N,50,2) / gt (N,60,2) / pred (N,6,60,2) / probs (N,6)")
    print(f"    origin (N,2), theta (N,)  ← city 좌표 복원용")
    print(f"    denorm(arr, origin[i], theta[i]) 로 city 좌표 변환")


if __name__ == "__main__":
    main()
