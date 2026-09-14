#!/usr/bin/env python
"""기존 v4 체크포인트를 재학습 없이 다시 평가한다 — 2026-09-14 적분기 기하 수정의 효과 기록.

무엇을 비교하나 (같은 val 데이터, 같은 가중치)
  before            커밋 55c52e5 의 model_v4 · dataset_lane
                    (경로 끝에서 위치가 멈추는 clamp, 폴백 경로 첫 점이 원점보다 ln/M 앞)
  extend            현재 model_v4 (route_point 가 경로 양 끝 밖을 끝 접선으로 직선 연장) + 55c52e5 데이터
  extend+fallback0  현재 model_v4 + 현재 dataset_lane (폴백 경로 첫 점을 원점으로)

왜 이렇게 나누나
  경로 데이터는 그대로 두고 적분기만 고쳤으므로, 같은 가중치로 다시 채점하면 재학습 없이 기하 수정의
  몫만 잴 수 있다. L3 는 적분기를 안 거치므로 before == extend 여야 한다 — 수정 범위의 확인이다.

함께 기록하는 빈도 (model_v4.route_point 주석이 여기를 가리킨다)
  geometry            경로 시작점이 차량보다 앞인 모드(s0 < 0)의 비율. 분모는 **살아있는 모드**
                      (route_mask > 0)다. 폴백 시나리오의 가려진 슬롯까지 분모에 넣으면 비율이 낮게 나온다
                      — 그래서 개수와 두 분모를 모두 남긴다.
  rollout_clamp_l0b   수정 전 적분기로 l0b 를 굴렸을 때 시작점에 얼어붙는 스텝 수와 경로 끝을 넘는 비율.
                      s 는 액션의 적분이라 위치 복원 방식과 무관하게 수정 전후가 같다.

지표는 train_v4.evaluate 를 그대로 쓴다 (학습 로그와 같은 정의).
체크포인트들이 학습될 때 폴백은 직진 1개였으므로 fallback="straight1" 로 평가한다.

주의: before 는 학습 로그의 best 와 정확히 같지 않다(학습 이후 데이터 코드가 조금 바뀌었다).
비교는 이 스크립트가 쓴 json 안의 평가끼리만 한다.

사용:  python src/reeval_v4.py        # runs/v4_reeval_s0.json 을 쓴다
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import train_v4 as T          # noqa: E402  evaluate · to_dev · DATA_ROOT
import model_v4 as MF         # noqa: E402  현재(수정본)
import dataset_lane as DF     # noqa: E402  현재(수정본)

BEFORE = "55c52e5"
CKPTS = [("l0b", "v4_l0b_s0", "l0"), ("l4_nw", "v4_l4_nw_s0", "l0"),
         ("l4_all", "v4_l4_all_s0", "l0"), ("l3b", "v4_l3b_s0", "l3")]


def module_at(commit, rel, name, tmp):
    """커밋 시점의 파일을 별도 모듈 이름으로 불러온다 (현재 모듈과 나란히 쓰려고)."""
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"],
                         capture_output=True, text=True, check=True).stdout
    p = Path(tmp) / f"{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Batches(list):
    """evaluate 는 로더를 순회만 한다. 전처리를 한 번만 하고 여러 모델에 재사용하려고 리스트로 둔다."""

    def __init__(self, items, ds):
        super().__init__(items)
        self.dataset = ds


def sha12(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def load(M, level, tag, dev):
    m = M.V4Net(in_dim=5, lane_in=MF.N_PTS * 2 + MF.N_RULE, level=level).to(dev)
    m.load_state_dict(torch.load(REPO / "runs" / f"lstm_{tag}.pth", map_location=dev))
    return m.eval()


def clamp_stats(M, tag, B, dev):
    """수정 전 clamp 가 실제로 걸리는 빈도. θ 가 ±90° 로 잘려 cos θ >= 0 이므로 s 는 단조증가다 —
    그래서 s0 < 0 인 모드만 시작점에 얼어붙고, 얼어붙는 스텝 수는 s < 0 인 스텝 수와 같다."""
    m = load(M, "l0", tag, dev)
    frozen, over_mode, over_step, live, horizon = [], 0, 0, 0, 0
    with torch.no_grad():
        for b in B:
            _, _, aux = m(**T.to_dev(b, dev, "l0", True))
            s = aux["s"]                                               # (B,K,T)
            horizon = s.size(-1)
            alive = b["route_mask"].to(dev) > 0
            neg0 = (b["route_sd0"][..., 0].to(dev) < 0) & alive
            frozen.append((s < 0).sum(-1)[neg0].float().cpu())
            over = (s > b["route_len"].to(dev).unsqueeze(-1)) & alive.unsqueeze(-1)
            over_mode += int(over.any(-1).sum())
            over_step += int(over.sum())
            live += int(alive.sum())
    f = torch.cat(frozen)
    return {"checkpoint": f"runs/lstm_{tag}.pth", "horizon_steps": horizon,
            "s0_negative_modes": int(f.numel()),
            "frozen_start_steps_median": float(f.median()),
            "frozen_start_steps_p90": float(torch.quantile(f, 0.9)),
            "past_route_end_mode_frac": round(over_mode / live, 4),
            "past_route_end_step_frac": round(over_step / (live * horizon), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(REPO / "runs" / "v4_reeval_s0.json"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = dict(with_rules=True, theta_ch=False, h_src="build", routes=True, fallback="straight1")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        MB = module_at(BEFORE, "src/model_v4.py", "model_v4_before", tmp)
        DB = module_at(BEFORE, "src/dataset_lane.py", "dataset_lane_before", tmp)

        def batches(D):
            ds = D.Av2LaneRuleDataset(T.DATA_ROOT, "val", a.val_limit, **kw)
            return Batches(list(DataLoader(ds, batch_size=64, shuffle=False,
                                           num_workers=a.workers)), ds)
        BB, BA = batches(DB), batches(DF)

        mask = torch.cat([b["route_mask"] for b in BB]) > 0
        s0neg = (torch.cat([b["route_sd0"][..., 0] for b in BB]) < 0) & mask
        fb = torch.cat([b["route_fallback"] for b in BB]) > 0.5
        first_before = torch.cat([b["routes"][:, 0, 0] for b in BB])[fb]
        first_after = torch.cat([b["routes"][:, 0, 0] for b in BA])[fb]

        def ev(M, level, tag, B):
            ade, fde, off = T.evaluate(load(M, level, tag, dev), B, dev, level, True)[:3]
            return {"minADE6": round(float(ade), 4), "minFDE6": round(float(fde), 4),
                    "val_offlane_steps": round(float(off), 4)}

        results = {}
        for name, tag, level in CKPTS:
            log = json.loads((REPO / "runs" / f"{tag}.json").read_text())
            r = results[name] = {
                "checkpoint": f"runs/lstm_{tag}.pth",
                "level": level,
                "train_log_best": {"minADE6": round(log["best_minADE6"], 4),
                                   "minFDE6": round(log["best_minFDE6"], 4)},
                "before": ev(MB, level, tag, BB),
                "extend": ev(MF, level, tag, BB),
                "extend+fallback0": ev(MF, level, tag, BA),
            }
            print(f"{name:7s} before {r['before']['minADE6']:.4f}/{r['before']['minFDE6']:.4f}  "
                  f"extend {r['extend']['minADE6']:.4f}/{r['extend']['minFDE6']:.4f}  "
                  f"extend+fallback0 {r['extend+fallback0']['minADE6']:.4f}/"
                  f"{r['extend+fallback0']['minFDE6']:.4f}", flush=True)
        clamp = clamp_stats(MB, "v4_l0b_s0", BB, dev)

    out = {
        "tag": "v4_reeval_s0",
        "what": "기존 체크포인트 재평가(재학습 없음) — 적분기 경로 끝 직선 연장(route_point)과 "
                "폴백 경로 첫 점 원점화의 효과",
        "date": time.strftime("%Y-%m-%d"),
        "before_commit": BEFORE,
        "after_code_sha12": {"model_v4.py": sha12(REPO / "src" / "model_v4.py"),
                             "dataset_lane.py": sha12(REPO / "src" / "dataset_lane.py")},
        "data": {"split": "val", "n": len(BB.dataset), **kw},
        "geometry": {
            "slots_total": int(mask.numel()),
            "live_modes": int(mask.sum()),
            "s0_negative_modes": int(s0neg.sum()),
            "s0_negative_frac_of_live_modes": round(float(s0neg.sum() / mask.sum()), 4),
            "s0_negative_frac_of_all_slots": round(float(s0neg.sum()) / mask.numel(), 4),
            "fallback_scenarios": int(fb.sum()),
            "fallback_first_point_x_median_before_m":
                round(float(first_before[:, 0].median()), 4) if fb.any() else None,
            "fallback_first_point_abs_max_after_m":
                round(float(first_after.abs().max()), 6) if fb.any() else None,
        },
        "rollout_clamp_l0b": clamp,
        "results": results,
        "note": "before 는 학습 로그 best 와 다를 수 있다(학습 이후 데이터 코드 변경). "
                "같은 파일 안의 평가끼리만 비교할 것. 재학습 효과는 포함되지 않는다.",
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("geometry:", out["geometry"])
    print("rollout_clamp_l0b:", clamp)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
