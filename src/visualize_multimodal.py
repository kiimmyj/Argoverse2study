"""
visualize_multimodal.py - K=6 멀티모달 예측 시각화 (6개 궤적 다 그림).

각 칸: 과거(회색) + 실제 미래(파랑 굵게) + 예측 6개(빨강 계열)
  - 6개 궤적을 확률에 따라 진하기 다르게 (확률 높을수록 진함)
  - 6개 중 정답에 가장 가까운 것(best)은 초록 강조
제목: minADE6 / minFDE6 (6개 중 최선 기준)
사용: python src/visualize_multimodal.py
"""
import sys
sys.path.append("src")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import Av2FocalDataset, _rotation_matrix
from model import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"
CKPT = "lstm_multimodal.pth"
N_POOL = 300


def denorm(arr_n, origin, theta):
    R = _rotation_matrix(theta)
    return arr_n @ R.T + origin


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Av2FocalDataset(DATA_ROOT, "val", limit=N_POOL)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = LSTMSeq2Seq().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    records = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"][0].numpy()                    # (60,2)
            traj, logits = model(x)                      # (1,6,60,2),(1,6)
            traj = traj[0].cpu().numpy()                 # (6,60,2)
            probs = F.softmax(logits[0], dim=0).cpu().numpy()   # (6,)
            origin = batch["origin"][0].numpy()
            theta = float(batch["theta"][0])

            fde_k = np.linalg.norm(traj[:, -1] - y[-1], axis=1)   # (6,)
            ade_k = np.linalg.norm(traj - y[None], axis=2).mean(axis=1)  # (6,)
            best = int(fde_k.argmin())

            records.append({
                "min_fde": float(fde_k[best]),
                "min_ade": float(ade_k[best]),
                "hist": denorm(x[0, :, :2].cpu().numpy(), origin, theta),
                "gt": denorm(y, origin, theta),
                "trajs": [denorm(traj[k], origin, theta) for k in range(traj.shape[0])],
                "probs": probs,
                "best": best,
            })

    records.sort(key=lambda r: r["min_fde"])
    n = len(records)
    idxs = [0, 1, 2, n // 2 - 1, n // 2, n // 2 + 1, n - 3, n - 2, n - 1]
    picks = [records[i] for i in idxs]
    labels = ["good"] * 3 + ["medium"] * 3 + ["bad"] * 3

    mean_ade = np.mean([r["min_ade"] for r in records])
    mean_fde = np.mean([r["min_fde"] for r in records])

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, r, lab in zip(axes.flat, picks, labels):
        h, gt = r["hist"], r["gt"]
        ax.plot(h[:, 0], h[:, 1], color="gray", lw=2, label="past", zorder=2)
        ax.plot(gt[:, 0], gt[:, 1], color="tab:blue", lw=3, label="GT", zorder=4)
        for k, tr in enumerate(r["trajs"]):
            if k == r["best"]:
                ax.plot(tr[:, 0], tr[:, 1], color="tab:green", lw=2.5,
                        ls="--", label="best of 6", zorder=5)
            else:
                a = 0.3 + 0.6 * r["probs"][k]
                ax.plot(tr[:, 0], tr[:, 1], color="tab:red", lw=1.3,
                        ls="--", alpha=a, zorder=3)
        ax.scatter(h[-1, 0], h[-1, 1], color="black", s=30, zorder=6)
        ax.set_title(f"{lab} | ADE={r['min_ade']:.1f}m  FDE={r['min_fde']:.1f}m",
                     fontsize=11)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=8, loc="best")

    fig.suptitle(f"K=6 Multimodal (val)  |  minADE6={mean_ade:.2f}m  minFDE6={mean_fde:.2f}m"
                 f"\ngray=past, blue=GT, green=best of 6, red=other modes (dark=high prob)",
                 fontsize=13)
    fig.tight_layout()
    out = "prediction_multimodal_3x3.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved -> {out}")
    print(f"pool={n}  minADE6={mean_ade:.2f}m  minFDE6={mean_fde:.2f}m")


if __name__ == "__main__":
    main()
