"""
visualize_prediction.py - 학습된 모델의 예측을 실제와 겹쳐 시각화.

val 에서 여러 시나리오를 예측 → FDE 로 정렬 → 잘한 것/중간/못한 것 9개를 3x3 격자로.
각 칸 제목에 ADE, FDE 둘 다 표시.
각 칸: focal 과거(회색) + 실제 미래(파랑) + 예측 미래(빨강 점선).
예측은 정규화 좌표 → origin·theta 로 역변환해 city 좌표로 복원.

사용: python src/visualize_prediction.py
"""
import sys
sys.path.append("src")

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import Av2FocalDataset, _rotation_matrix
from model import LSTMSeq2Seq

DATA_ROOT = "/data/argoverse2/motion_forecasting"
CKPT = "lstm_seq2seq.pth"
N_POOL = 300
OBS_LEN = 50


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
            y = batch["y"][0].numpy()                 # (60,2) 정답
            pred = model(x)[0].cpu().numpy()          # (60,2) 예측
            origin = batch["origin"][0].numpy()
            theta = float(batch["theta"][0])

            dist = np.linalg.norm(pred - y, axis=1)   # (60,) 시점별 거리
            ade = float(dist.mean())                  # 전 구간 평균
            fde = float(dist[-1])                     # 최종 시점
            hist_n = x[0, :, :2].cpu().numpy()
            records.append({
                "ade": ade, "fde": fde,
                "hist": denorm(hist_n, origin, theta),
                "gt": denorm(y, origin, theta),
                "pred": denorm(pred, origin, theta),
                "sid": batch["scenario_id"][0],
            })

    # FDE 로 정렬 후 잘한 것 3 / 중간 3 / 못한 것 3
    records.sort(key=lambda r: r["fde"])
    n = len(records)
    idxs = [0, 1, 2, n // 2 - 1, n // 2, n // 2 + 1, n - 3, n - 2, n - 1]
    picks = [records[i] for i in idxs]
    labels = ["good"] * 3 + ["medium"] * 3 + ["bad"] * 3

    # 전체 풀 평균 (참고용)
    mean_ade = np.mean([r["ade"] for r in records])
    mean_fde = np.mean([r["fde"] for r in records])

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for ax, r, lab in zip(axes.flat, picks, labels):
        h, gt, pr = r["hist"], r["gt"], r["pred"]
        ax.plot(h[:, 0], h[:, 1], color="gray", lw=2, label="past")
        ax.plot(gt[:, 0], gt[:, 1], color="tab:blue", lw=2.5, label="GT future")
        ax.plot(pr[:, 0], pr[:, 1], color="tab:red", lw=2, ls="--", label="pred")
        ax.scatter(h[-1, 0], h[-1, 1], color="black", s=30, zorder=5)
        ax.set_title(f"{lab} | ADE={r['ade']:.1f}m  FDE={r['fde']:.1f}m", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=9, loc="best")

    fig.suptitle(f"Prediction vs GT (val)  |  pool mean: ADE={mean_ade:.2f}m  FDE={mean_fde:.2f}m"
                 f"\ngray=past, blue=GT, red=pred", fontsize=13)
    fig.tight_layout()
    out = "prediction_val_3x3.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved -> {out}")
    print(f"pool={n}  mean ADE={mean_ade:.2f}m  mean FDE={mean_fde:.2f}m")
    print(f"best FDE={records[0]['fde']:.2f}m  worst FDE={records[-1]['fde']:.2f}m")


if __name__ == "__main__":
    main()
