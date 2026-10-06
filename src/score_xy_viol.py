"""
score_xy_viol.py - 완료된 판들을 **예측 좌표에서 복원한 진행방향** 기준으로 다시 채점한다 (2026-10-06).

왜 다시 재나
------------
학습 로그의 "7.3°초과"는 `aux["theta"]`(차로 접선에 대한 잔차각)의 스텝 변화에서 잰 값이다.
theta 는 모델이 직접 내는 출력이라 action 벌점이 누르는 변수와 같고, 실제 진행방향
psi = atan2(dy, dx) 에는 경로 곡률 k(s) 와 프레네 기하 항이 더 들어간다. 2026-09-29 측정에서
모델의 dθ 를 통째로 지워도 좌표 기준 위반율이 0.735% -> 0.733% 밖에 안 움직였다 — 즉 로그 지표와
궤적의 실제 흔들림은 거의 무관하다. 그래서 penalty 비교의 결론은 좌표 기준으로 내야 한다.

재는 것 (정의는 상수로 모아 둔다. train_v4.jitter_xy 의 규칙과 같다)
  over_alive_pct   살아있는 모드 전체에서 |dpsi| > 7.3°/step 인 스텝 비율 [%]
  over_top1_pct    확률 1위 모드만 같은 비율 [%]  (평가에 실제로 쓰이는 궤적)
  over_mean_deg    초과한 스텝들의 평균 |dpsi| [°]
  p99_dpsi_deg     |dpsi| 99 분위 [°]
  theta_over_pct   기존(잔차각) 지표 — 로그 값과 맞는지 대조용
  minADE6/minFDE6  재채점 (로그 best 와 대조)
시나리오별 값도 `runs/v4_xy_viol_per.npz` 에 남긴다 — 차선변경 627건처럼 **임의의 부분집합**을
나중에 따로 집계하려면 전체 평균만으로는 못 하기 때문이다(`src/score_subset.py`).
또 **정답의 총 회전량으로 시나리오를 나눠** 같은 값을 따로 낸다. penalty 가 잡음만 누르는지
실제 회전까지 누르는지는 이 분해로만 갈린다 (직진에서만 위반이 줄고 회전에서 ADE 가 나빠지면
실제 회전을 누른 것이다).
정지 구간은 방향이 정의되지 않으므로 **양쪽 스텝 속력이 1 m/s 이상일 때만** 센다.

  python src/score_xy_viol.py --tags-file tags.txt      # 한 줄에 tag 하나
  python src/score_xy_viol.py --tag v4_l4nw_ah2_full_sm1_cos30_s0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_cached import CachedV4Dataset          # noqa: E402
from model_v4 import V4Net                          # noqa: E402
from train_v4 import to_dev                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DT = 0.1
LABEL_DTHETA_DEG = 7.3      # AV2 라벨 전수조사 p99.99 — 기존 지표와 같은 임계
MOVE_V = 1.0                # m/s 이하 스텝은 위치차분 방향이 잡음이다


def val_cache_of(tag):
    """학습 로그가 적어 둔 val 캐시 경로를 그대로 쓴다 (판마다 소스 해시가 달라 디렉터리명이 다르다)."""
    log = ROOT / "runs" / f"{tag}.log"
    for line in log.read_text(errors="ignore").splitlines():
        if "[cache] val" in line:
            return line.split("[cache] val", 1)[1].strip()
    raise SystemExit(f"{tag}: 로그에 val 캐시 경로가 없다")


def agents_cache_of(tag):
    log = ROOT / "runs" / f"{tag}.log"
    for line in log.read_text(errors="ignore").splitlines():
        if "[agents] val" in line:
            return line.split("[agents] val", 1)[1].strip()
    return None


TURN_BINS = [(0.0, 5.0, "직진 <5°"), (5.0, 30.0, "완만 5-30°"), (30.0, 1e9, "회전 >=30°")]


def gt_turn_deg(y):
    """정답 궤적의 총 진행방향 변화량 [°] — 움직이는 스텝의 dpsi 를 부호 그대로 합친 절댓값."""
    d = y[..., 1:, :] - y[..., :-1, :]
    sp = d.norm(dim=-1) / DT
    psi = torch.atan2(d[..., 1], d[..., 0])
    dp = torch.remainder(psi[..., 1:] - psi[..., :-1] + np.pi, 2 * np.pi) - np.pi
    ok = ((sp[..., 1:] >= MOVE_V) & (sp[..., :-1] >= MOVE_V)).float()
    return (dp * ok).sum(-1).abs() * 180.0 / np.pi


def dpsi_deg(traj):
    """(…,T,2) -> |dpsi| [°] (…,T-2) 와 '양쪽 스텝이 움직인다' 마스크."""
    d = traj[..., 1:, :] - traj[..., :-1, :]
    sp = d.norm(dim=-1) / DT
    psi = torch.atan2(d[..., 1], d[..., 0])
    dp = torch.remainder(psi[..., 1:] - psi[..., :-1] + np.pi, 2 * np.pi) - np.pi
    ok = (sp[..., 1:] >= MOVE_V) & (sp[..., :-1] >= MOVE_V)
    return dp.abs() * 180.0 / np.pi, ok.float()


class Acc:
    """초과 비율·평균 초과량·분위를 모으는 누적기 (분위는 표본을 모아서 낸다)."""

    def __init__(self):
        self.n = self.exc = self.exc_sum = 0.0
        self.samp = []

    def add(self, dp, m):
        self.n += m.sum().item()
        over = (dp > LABEL_DTHETA_DEG).float() * m
        self.exc += over.sum().item()
        self.exc_sum += (dp * over).sum().item()
        v = dp[m > 0]
        if v.numel():
            self.samp.append(v[torch.randint(v.numel(), (min(v.numel(), 20000),), device=v.device)].cpu())

    def out(self):
        s = torch.cat(self.samp) if self.samp else torch.zeros(1)
        return {"steps": int(self.n),
                "over_pct": 100.0 * self.exc / max(self.n, 1.0),
                "over_mean_deg": self.exc_sum / max(self.exc, 1.0),
                "p50_deg": float(s.median()), "p99_deg": float(s.quantile(0.99))}


@torch.no_grad()
def score(tag, batch=32, workers=4, limit=None):
    j = json.loads((ROOT / "runs" / f"{tag}.json").read_text())
    a = j["args"]
    cpath = ROOT / "runs" / f"lstm_{tag}.pth"
    sd = torch.load(cpath, map_location="cpu")
    in_dim = sd["traj_encoder.weight_ih_l0"].shape[1]
    ag_in = sd["agent_encoder.0.weight"].shape[1] if "agent_encoder.0.weight" in sd else 0
    m = V4Net(in_dim=in_dim, lane_in=30, level="l0",
              th0_mode=a.get("th0", "current"), agents_in=ag_in).cuda()
    m.load_state_dict(sd)
    m.eval()

    ds = CachedV4Dataset(val_cache_of(tag), limit=limit)
    if ag_in:
        from dataset_agents import AgentsStore, WithAgents
        ds = WithAgents(ds, AgentsStore(agents_cache_of(tag),
                                        radius_m=a.get("agents_radius", 30.0),
                                        step=a.get("agents_step", 5)))
    al, t1, gt = Acc(), Acc(), Acc()
    per = {"ade": [], "fde": [], "exc": [], "steps": [], "turn": []}   # 시나리오별 (부분집합 집계용)
    by = {lab: {"t1": Acc(), "gt": Acc(), "ade": 0.0, "n": 0.0} for *_, lab in TURN_BINS}
    ade = fde = n = 0.0
    exc_th = live_th = 0.0
    for b in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
        traj, logits, aux = m(**to_dev(b, "cuda", "l0", True))
        y = b["y"].cuda()
        mm = b["route_mask"].cuda()
        lv = mm > 0
        dist = torch.norm(traj - y.unsqueeze(1), dim=-1)
        big = torch.full_like(dist[:, :, 0], float("inf"))
        ade += torch.where(lv, dist.mean(2), big).min(1).values.sum().item()
        fde += torch.where(lv, dist[:, :, -1], big).min(1).values.sum().item()
        n += y.size(0)
        # 기존(잔차각) 지표 — 로그와 대조
        th = aux["theta"]
        dd = (th[:, :, 1:] - th[:, :, :-1]).abs() * 180.0 / np.pi
        exc_th += ((dd > LABEL_DTHETA_DEG).float() * mm.unsqueeze(-1)).sum().item()
        live_th += mm.sum().item() * dd.size(2)
        # 좌표 기준
        dp, ok = dpsi_deg(traj)                       # (B,K,T-2)
        al.add(dp, ok * mm.unsqueeze(-1))
        top = logits.argmax(1)                        # 빈 모드는 -inf 라 뽑히지 않는다
        ar = torch.arange(traj.size(0), device=traj.device)
        t1.add(dp[ar, top], ok[ar, top])
        dpg, okg = dpsi_deg(y)                        # (B,T-2)
        gt.add(dpg, okg)
        # ---- 시나리오별 값 (top-1 기준 위반 step 수와 유효 step 수)
        tn = gt_turn_deg(y)
        a1 = torch.where(lv, dist.mean(2), big).min(1).values
        f1 = torch.where(lv, dist[:, :, -1], big).min(1).values
        dp1, ok1 = dp[ar, top], ok[ar, top]
        per["ade"].append(a1.cpu().numpy())
        per["fde"].append(f1.cpu().numpy())
        per["exc"].append((((dp1 > LABEL_DTHETA_DEG).float() * ok1).sum(-1)).cpu().numpy())
        per["steps"].append(ok1.sum(-1).cpu().numpy())
        per["turn"].append(tn.cpu().numpy())
        for lo, hi, lab in TURN_BINS:
            sel = (tn >= lo) & (tn < hi)
            if not bool(sel.any()):
                continue
            by[lab]["t1"].add(dp[ar, top][sel], ok[ar, top][sel])
            by[lab]["gt"].add(dpg[sel], okg[sel])
            by[lab]["ade"] += a1[sel].sum().item()
            by[lab]["n"] += int(sel.sum())
    per = {k: np.concatenate(v) for k, v in per.items()}
    return per, {"tag": tag, "n": int(n), "minADE6": ade / n, "minFDE6": fde / n,
            "logged_best_minADE6": j["best_minADE6"],
            "theta_over_pct": 100.0 * exc_th / max(live_th, 1.0),
            "xy_alive": al.out(), "xy_top1": t1.out(), "gt": gt.out(),
            "by_turn": {lab: {"n": int(v["n"]), "minADE6": v["ade"] / max(v["n"], 1.0),
                              "xy_top1": v["t1"].out(), "gt": v["gt"].out()}
                        for lab, v in by.items()},
            "args": {k: a.get(k) for k in ("input", "smooth", "smooth_mode", "smooth_xy",
                                           "agents", "lat_modes", "cleanse", "seed")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--tags-file", dest="tags_file")
    ap.add_argument("--out", default=str(ROOT / "runs" / "v4_xy_viol.json"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4, help="공유 머신 규칙: 16 이하")
    ap.add_argument("--limit", type=int, default=None, help="스모크용")
    args = ap.parse_args()
    tags = list(args.tag)
    if args.tags_file:
        tags += [l.strip() for l in Path(args.tags_file).read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for t in tags:
        t0 = time.time()
        try:
            per, r = score(t, args.batch, args.workers, args.limit)
        except Exception as e:                       # 한 판이 깨져도 나머지는 재운다
            print(f"[{t}] 실패: {type(e).__name__}: {e}", flush=True)
            continue
        res[t] = r
        print(f"[{t}] {time.time() - t0:.0f}s  ADE {r['minADE6']:.3f} (로그 {r['logged_best_minADE6']:.3f})  "
              f"theta초과 {r['theta_over_pct']:.2f}%  "
              f"xy초과 alive {r['xy_alive']['over_pct']:.3f}% / top1 {r['xy_top1']['over_pct']:.3f}%  "
              f"(정답 {r['gt']['over_pct']:.3f}%)", flush=True)
        if args.limit is None:
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            pf = out.with_name(out.stem + "_per.npz")
            keep = dict(np.load(pf)) if pf.exists() else {}
            keep.update({f"{t}|{k}": v for k, v in per.items()})
            np.savez_compressed(pf, **keep)
    print("저장", out if args.limit is None else "(스모크 — 저장 안 함)")


if __name__ == "__main__":
    main()
