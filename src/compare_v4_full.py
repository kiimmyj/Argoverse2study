"""compare_v4_full.py - (a, h) 전체 데이터 판과 흔들림 벌점 판을 같은 val 로 다시 채점한다 (2026-09-16).

비교 대상
  50k       v4_l0_ah2_s0 / v4_l4nw_ah2_s0                   train 50,000 · best 는 val 2,000 에서 고름
  전체      v4_l0_ah2_full_s0 / v4_l4nw_ah2_full_s0           train 199,908 · best 는 val 24,988 에서 고름
  벌점 1.0  v4_l0_ah2_full_sm1_s0 / v4_l4nw_ah2_full_sm1_s0   전체 + train_v4 --smooth 1.0
  2 Hz      v4_l0_ah2_2hz_full_sm1_s0 / v4_l4nw_ah2_2hz_full_sm1_s0   벌점 1.0 판과 입력만 다름(--input ah2_2hz)
  벌점 0.1  v4_l0_ah2_full_sm0.1_s0                          전체 + --smooth 0.1
  5채널     v4_l0b_s0 / v4_l4_nw_s0                          참고용. 50k · 옛 기하로 학습, val 2,000 만(원본에서 즉석 전처리)
각 판은 자기 val 에서 best 에폭을 골랐으므로 자기 집합 점수가 조금 낙관적이다. 그래서 두 집합을 모두 보인다.

지표 — 살아있는 모드 기준, train_v4.evaluate 와 같은 정의
  minADE6 / minFDE6 / 이탈   이탈 = 시나리오당 '모드별 밴드 밖 step 비율' 의 합 (0~6)
  7.3°초과                   스텝간 |Δθ| 가 라벨 p99.99(7.3°) 를 넘는 step 비율
  흔들림 θ / a               Δ(dθ / 8°)² 와 Δ(a / 8 m/s²)² 의 평균 — train_v4.jitter 의 두 항

사용: python src/compare_v4_full.py      (학습이 돌지 않을 때 — GPU 를 같이 쓰면 에폭 시간 측정이 흐려진다)
출력: runs/v4_full_compare.json
"""
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_cached import CachedV4Dataset
from dataset_lane import Av2LaneRuleDataset
from model_v4 import V4Net, A_SCALE, DTHETA_MAX
from train_v4 import to_dev, LABEL_DTHETA_DEG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/data/argoverse2/motion_forecasting"
# 입력 형태별 val 캐시. 키가 아니라 경로로 고정한다 — 캐시 키는 소스 파일 전체 해시라 주석만 고쳐도 바뀐다.
# 두 캐시 모두 2026-09-17 코드(a8df67b)의 원본 전처리와 48개 대조를 통과했다.
VAL_CACHE = {"ah2": "/data/argoverse2/cache/v4/val_e67373005962be8a_n24988",
             "ah2_2hz": "/data/argoverse2/cache/v4/val_32e2b95fe31293fd_n24988"}
OUT = f"{ROOT}/runs/v4_full_compare.json"
RUNS = [
    ("v4_l0b_s0", "L0 5채널 50k"),
    ("v4_l0_ah2_s0", "L0 50k"),
    ("v4_l0_ah2_full_s0", "L0 전체"),
    ("v4_l0_ah2_full_sm1_s0", "L0 전체·벌점1"),
    ("v4_l0_ah2_full_sm0.1_s0", "L0 전체·벌점0.1"),
    ("v4_l4_nw_s0", "L4 5채널 50k"),
    ("v4_l4nw_ah2_s0", "L4 50k"),
    ("v4_l4nw_ah2_full_s0", "L4 전체"),
    ("v4_l4nw_ah2_full_sm1_s0", "L4 전체·벌점1"),
    ("v4_l0_ah2_2hz_full_sm1_s0", "L0 전체·벌점1·2Hz"),
    ("v4_l4nw_ah2_2hz_full_sm1_s0", "L4 전체·벌점1·2Hz"),
]


@torch.no_grad()
def score(model, ds, workers):
    ade = fde = off = 0.0
    n = 0
    exc = live = jt = ja = 0.0
    for b in DataLoader(ds, batch_size=32, shuffle=False, num_workers=workers):
        traj, _, aux = model(**to_dev(b, "cuda", "l0", True))
        y = b["y"].cuda()
        mm = b["route_mask"].cuda()
        lv = mm > 0
        dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
        big = torch.full_like(dist[:, :, 0], float("inf"))
        ade += torch.where(lv, dist.mean(2), big).min(1).values.sum().item()
        fde += torch.where(lv, dist[:, :, -1], big).min(1).values.sum().item()
        lm = mm.unsqueeze(-1)
        o = torch.relu(aux["d"] - aux["band"][..., 0]) + torch.relu(-aux["d"] - aux["band"][..., 1])
        off += ((o > 0).float() * lm).sum().item() / o.size(2)
        th = aux["theta"]
        dd = (th[:, :, 1:] - th[:, :, :-1]).abs() * 180.0 / np.pi
        exc += ((dd > LABEL_DTHETA_DEG).float() * lm).sum().item()
        live += lm.sum().item() * dd.size(2)
        ut = aux["dtheta"] / DTHETA_MAX
        ua = aux["a"] / A_SCALE
        jt += (((ut[..., 1:] - ut[..., :-1]) ** 2) * lm).sum().item()
        ja += (((ua[..., 1:] - ua[..., :-1]) ** 2) * lm).sum().item()
        n += y.size(0)
    return {"n": n, "minADE6": ade / n, "minFDE6": fde / n, "offlane": off / n,
            "dtheta_over_label_pct": 100.0 * exc / live,
            "jitter_theta": jt / live, "jitter_a": ja / live}


def main():
    t_all = time.time()
    res = {}
    for tag, name in RUNS:
        jpath, cpath = f"{ROOT}/runs/{tag}.json", f"{ROOT}/runs/lstm_{tag}.pth"
        if not (os.path.exists(jpath) and os.path.exists(cpath)):
            print(f"[skip] {tag} — 결과 파일 없음")
            continue
        d = json.load(open(jpath))
        a = d["args"]
        h = d["history"]
        sd = torch.load(cpath, map_location="cpu")
        in_dim = sd["traj_encoder.weight_ih_l0"].shape[1]
        inp = a.get("input", "raw5")          # 5채널 옛 판은 args 에 input 이 없다
        th0 = a.get("th0", "current")
        m = V4Net(in_dim=in_dim, lane_in=30, level="l0", th0_mode=th0).cuda()
        m.load_state_dict(sd)
        m.eval()
        best = min(h, key=lambda x: x["minADE6"])
        r = {"name": name, "input": inp, "th0": th0,
             "train_n": a["limit"], "val_n_selected": a["val_limit"],
             "smooth": a.get("smooth", 0.0), "offlane": a.get("offlane", 0.0),
             "best_epoch": best["epoch"], "logged_best": [d["best_minADE6"], d["best_minFDE6"]],
             "epoch_sec_median": st.median(x["sec"] for x in h),
             "total_min": sum(x["sec"] for x in h) / 60.0}
        t0 = time.time()
        if inp in VAL_CACHE:
            r["val2000"] = score(m, CachedV4Dataset(VAL_CACHE[inp], limit=2000), 2)
            r["val24988"] = score(m, CachedV4Dataset(VAL_CACHE[inp]), 2)
        else:
            ds = Av2LaneRuleDataset(DATA_ROOT, "val", 2000, with_rules=True, routes=True,
                                    fallback="straight1", input_repr="raw5")
            r["val2000"] = score(m, ds, 8)
        print(f"[{tag}] {time.time() - t0:.0f}s", flush=True)
        res[tag] = r

    json.dump({"val_cache": VAL_CACHE, "label_dtheta_deg": LABEL_DTHETA_DEG, "runs": res,
               "sec": time.time() - t_all}, open(OUT, "w"), indent=2, ensure_ascii=False)

    print("\n=== 학습 기록 (best 는 각자 고른 val) ===")
    print(f"{'run':16} {'train':>7} {'벌점':>4} {'best ep':>7} {'minADE6':>8} {'minFDE6':>8} {'에폭 중앙':>8} {'합계':>6}")
    for r in res.values():
        print(f"{r['name']:16} {r['train_n']:7d} {r['smooth']:4g} {r['best_epoch']:7d} "
              f"{r['logged_best'][0]:8.3f} {r['logged_best'][1]:8.3f} {r['epoch_sec_median']:7.0f}s "
              f"{r['total_min']:5.1f}분")
    for key, label in (("val2000", "val 2,000 (50k 판이 best 를 고른 집합)"),
                       ("val24988", "val 24,988 (전체 판이 best 를 고른 집합)")):
        print(f"\n=== 재채점 — {label} ===")
        print(f"{'run':16} {'minADE6':>8} {'minFDE6':>8} {'이탈':>5} {'7.3°초과':>8} {'흔들림θ':>8} {'흔들림a':>8}")
        for r in res.values():
            if key not in r:
                continue
            s = r[key]
            print(f"{r['name']:16} {s['minADE6']:8.3f} {s['minFDE6']:8.3f} {s['offlane']:5.2f} "
                  f"{s['dtheta_over_label_pct']:7.2f}% {s['jitter_theta']:8.4f} {s['jitter_a']:8.4f}")
    print(f"\n소요 {time.time() - t_all:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
