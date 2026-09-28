"""
viz_v4_labelfit.py - 측정 4: 라벨 재현 하한 (TPK arXiv:2505.06743 식).

"우리 구조(Frenet 적분기 + 액션 상한)로 정답 궤적을 얼마나 재현할 수 있는가"를 잰다.
모델 가중치는 건드리지 않는다 — 시나리오마다 액션열 (a, dθ) 를 자유 변수로 두고 경사하강한다.

단계
  1. 정확 역변환: 정답 궤적 y(60점)를 **정답 기준 경로**(viz_v4_dump 의 gt_route)의 Frenet 좌표
     (s, d) 로 되돌린다. 초기값은 lane_frame.to_frame, 정밀화는 model_v4.route_point 와 같은
     기하로 torch 경사하강. 그 뒤 적분기 식을 거꾸로 풀어 (v, θ, a, dθ) 를 얻는다.
       ds[k] = v[k]·cos θ[k]·dt,  dd[k] = v[k]·sin θ[k]·dt   (s[-1]=s0, d[-1]=d0)
       a[k]  = (v[k] − v[k−1])/dt (v[-1]=v0),  dθ[k] = wrap(θ[k] − θ[k−1]) (θ[-1]=θ0)
     → 이 액션열은 **제약이 없다**. 상한을 넘는 비율이 곧 '라벨이 우리 액션 공간 밖' 이다.
  2. 제약 적합: act 를 자유 변수로 두고 a = 8·tanh(act₀), dθ = 8°·tanh(act₁) (학습과 같은
     파라미터화) 로 model_v4.V4Net.rollout 을 돌려 목적함수를 최소화한다.
       obj=ade        평균 L2 (구조적 정확도 하한)
       obj=sl1        smooth_l1 (학습 거리 손실과 같은 식)
       obj=sl1+jit    smooth_l1 + 1.0·jitter (흔들림 벌점이 강요하는 최소 손해)

출력
  viz/v4/feasibility/data/labelfit.json   숫자
  viz/v4/feasibility/data/gt_actions.npz  정답 액션열 (측정 3 의 기준선이 읽는다)

  python src/viz_v4_labelfit.py --n 2000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

OUT = C.VIZ_ROOT / "feasibility"
DT = C.DT
A_MAX = 8.0                       # model_v4.A_SCALE
DTH_MAX_DEG = 8.0                 # model_v4.DTHETA_MAX
TH_MAX_DEG = 90.0                 # model_v4.THETA_MAX
SAT = 0.99                        # |값| >= SAT·상한 이면 '상한에 붙었다'
REF_TAG = "v4_l4nw_ah2_full_sm1_cos30_s0"   # gt_route·상황 분류를 읽어올 덤프 (모두 같은 값이다)


# --------------------------------------------------------------------------- 1. 정확 역변환
def frenet_invert(y, routes, route_tan, route_len, sd0, device, iters=400, lr=0.05):
    """정답 궤적 y (N,60,2) → 경로 기준 (s, d). model_v4.route_point 와 **같은 기하**를 쓴다.

    lane_frame.to_frame 은 64점 폴리라인의 최근접 점 + 접선 투영이라 적분기의 선형보간 기하와
    조금 다르다. 적분기로 되굴려 y 를 재현해야 하므로 적분기 기하로 다시 맞춘다.
    반환: s (N,60), d (N,60), 잔차 [m] (N,60)
    """
    import torch
    import lane_frame as lf
    from model_v4 import route_point

    N, T = y.shape[:2]
    # --- 초기값: to_frame (원점 + 60스텝, 원점은 버린다)
    s0_np, d0_np = np.asarray(sd0[:, 0], np.float64), np.asarray(sd0[:, 1], np.float64)
    s_init = np.zeros((N, T)); d_init = np.zeros((N, T))
    for i in range(N):
        rd = C.route_dict(routes[i], route_tan[i], route_len[i])
        path = np.concatenate([np.zeros((1, 2)), np.asarray(y[i], np.float64)], 0)
        s_, d_, _ = lf.to_frame(path, rd)
        s_init[i], d_init[i] = s_[1:], d_[1:]

    t = lambda a: torch.as_tensor(np.array(a), dtype=torch.float32, device=device)
    R = t(routes).unsqueeze(1)                 # (N,1,64,2)
    Tn = t(route_tan).unsqueeze(1)
    L = t(route_len).view(N, 1)
    Y = t(y).unsqueeze(1)                      # (N,1,60,2)
    s = t(s_init).unsqueeze(1).clone().requires_grad_(True)
    d = t(d_init).unsqueeze(1).clone().requires_grad_(True)
    opt = torch.optim.Adam([s, d], lr=lr)
    M = R.size(2)
    for it in range(iters):
        opt.zero_grad()
        P = route_point(R, Tn, s, L)
        from model_v4 import interp1d
        Tg = interp1d(Tn, s / L.clamp(min=1e-3).unsqueeze(-1) * (M - 1))
        Tg = Tg / Tg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Nm = torch.stack([-Tg[..., 1], Tg[..., 0]], -1)
        q = P + Nm * d.unsqueeze(-1)
        loss = ((q - Y) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = route_point(R, Tn, s, L)
        from model_v4 import interp1d
        Tg = interp1d(Tn, s / L.clamp(min=1e-3).unsqueeze(-1) * (M - 1))
        Tg = Tg / Tg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Nm = torch.stack([-Tg[..., 1], Tg[..., 0]], -1)
        q = P + Nm * d.unsqueeze(-1)
        res = (q - Y).norm(dim=-1).squeeze(1).cpu().numpy()
    return s.detach().squeeze(1).double().cpu().numpy(), d.detach().squeeze(1).double().cpu().numpy(), res


def theta0_guard(routes, route_tan, route_len, sd0, h0):
    """model_v4.rollout 의 th0 (th0_mode='guard') 를 numpy 로 그대로."""
    M = routes.shape[-2]
    L = np.maximum(np.asarray(route_len, np.float64), 1e-3)
    idx = np.clip(np.asarray(sd0[:, 0], np.float64) / L * (M - 1), 0.0, M - 1 - 1e-4)
    i0 = np.floor(idx).astype(int)
    f = (idx - i0)[:, None]
    tg = route_tan[np.arange(len(idx)), i0] * (1 - f) + \
        route_tan[np.arange(len(idx)), np.minimum(i0 + 1, M - 1)] * f
    k0 = np.arctan2(tg[:, 1], tg[:, 0])
    th0 = -k0
    prop = C.wrap(np.asarray(h0, np.float64) - k0)
    return np.where(np.abs(prop) <= np.radians(TH_MAX_DEG), prop, th0), k0


def gt_actions(s, d, sd0, v0, th0):
    """(s, d) + 시작 상태 → 정답 액션열 (제약 없음). 적분기 식을 그대로 뒤집는다."""
    s_prev = np.concatenate([np.asarray(sd0[:, 0], np.float64)[:, None], s[:, :-1]], 1)
    d_prev = np.concatenate([np.asarray(sd0[:, 1], np.float64)[:, None], d[:, :-1]], 1)
    ds, dd = s - s_prev, d - d_prev
    v = np.hypot(ds, dd) / DT
    th = np.arctan2(dd, ds)
    v_prev = np.concatenate([np.asarray(v0, np.float64)[:, None], v[:, :-1]], 1)
    a = (v - v_prev) / DT
    th_prev = np.concatenate([np.asarray(th0, np.float64)[:, None], th[:, :-1]], 1)
    dth = C.wrap(th - th_prev)
    return {"a": a, "dtheta": dth, "v": v, "theta": th, "s": s, "d": d}


# --------------------------------------------------------------------------- 2. 제약 적합
def fit_actions(batch, y, device, objective, w_smooth=0.0, iters=3000, lr=0.05, init=None,
                log_every=None):
    """액션열을 자유 변수로 두고 목적함수를 최소화한다. 모델 가중치는 쓰지 않는다
    (rollout 은 순수 함수 — th0_mode 만 본다)."""
    import torch
    import torch.nn.functional as F
    from model_v4 import V4Net, A_SCALE, DTHETA_MAX

    log_every = log_every or max(1, iters // 6)
    net = V4Net(in_dim=5, lane_in=30, level="l0", th0_mode="guard").to(device)
    net.eval()
    N = y.shape[0]
    t = lambda a: torch.as_tensor(np.array(a), dtype=torch.float32, device=device)
    R, Tn = t(batch["routes"]).unsqueeze(1), t(batch["route_tan"]).unsqueeze(1)
    L, SD0 = t(batch["route_len"]).view(N, 1), t(batch["route_sd0"]).view(N, 1, 2)
    V0, H0 = t(batch["v0"]).view(N), t(batch["h0"]).view(N)
    Y = t(y)
    if init is None:
        act = torch.zeros(N, 1, C.FUT, 2, device=device)
    else:
        # 정확 역변환 값에서 시작한다 (atanh 로 되돌림, 상한 안으로 자름)
        a0 = np.clip(init["a"] / A_SCALE, -0.999, 0.999)
        t0 = np.clip(init["dtheta"] / DTHETA_MAX, -0.999, 0.999)
        act = torch.stack([t(np.arctanh(a0)), t(np.arctanh(t0))], -1).unsqueeze(1)
    act = act.clone().requires_grad_(True)
    mm = torch.ones(N, 1, device=device)
    opt = torch.optim.Adam([act], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.02)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        traj, aux = net.rollout(act, R, Tn, L, SD0, V0, h0=H0)
        dist = (traj - Y.unsqueeze(1)).norm(dim=-1)                    # (N,1,60)
        if objective == "ade":
            data = dist.mean()
        elif objective == "sl1":
            data = F.smooth_l1_loss(traj.squeeze(1), Y)
        else:
            raise ValueError(objective)
        loss = data
        if w_smooth > 0:
            ua, ut = aux["a"] / A_SCALE, aux["dtheta"] / DTHETA_MAX
            j = (ua[..., 1:] - ua[..., :-1]) ** 2 + (ut[..., 1:] - ut[..., :-1]) ** 2
            loss = loss + w_smooth * j.mean()
        loss.backward()
        opt.step()
        sch.step()
        if (it + 1) % log_every == 0 or it == 0:
            hist.append({"it": it + 1, "loss": float(loss.detach()), "ade": float(dist.mean()),
                         "fde": float(dist[..., -1].mean())})
            print(f"    [{objective}{'+jit' if w_smooth else ''}] {it+1}/{iters} "
                  f"loss {float(loss):.5f} ADE {float(dist.mean()):.4f} "
                  f"FDE {float(dist[...,-1].mean()):.4f}", flush=True)
    with torch.no_grad():
        traj, aux = net.rollout(act, R, Tn, L, SD0, V0, h0=H0)
        dist = (traj - Y.unsqueeze(1)).norm(dim=-1).squeeze(1)
        a_v = aux["a"].squeeze(1).cpu().numpy()
        dth_v = aux["dtheta"].squeeze(1).cpu().numpy()
        ua, ut = aux["a"] / A_SCALE, aux["dtheta"] / DTHETA_MAX
        jj = ((ua[..., 1:] - ua[..., :-1]) ** 2 + (ut[..., 1:] - ut[..., :-1]) ** 2).squeeze(1)
        out = {"ade": dist.mean(1).cpu().numpy(), "fde": dist[:, -1].cpu().numpy(),
               "a": a_v, "dtheta": dth_v, "jitter": jj.mean(1).cpu().numpy(),
               "traj": traj.squeeze(1).cpu().numpy(), "hist": hist}
    return out


def sat_stats(a, dth):
    """제약 포화. a 는 [m/s²], dth 는 [rad]."""
    sa = np.abs(a) >= SAT * A_MAX
    st = np.abs(dth) >= SAT * np.radians(DTH_MAX_DEG)
    both = sa | st
    return {"a_step_pct": float(100.0 * sa.mean()), "dtheta_step_pct": float(100.0 * st.mean()),
            "any_step_pct": float(100.0 * both.mean()),
            "a_scen_pct": float(100.0 * sa.any(1).mean()),
            "dtheta_scen_pct": float(100.0 * st.any(1).mean()),
            "any_scen_pct": float(100.0 * both.any(1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="val 앞 N 시나리오")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--inv-iters", dest="inv_iters", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ref-tag", dest="ref_tag", default=REF_TAG)
    ap.add_argument("--also-top1-route", dest="also_top1", type=int, default=1,
                    help="모델이 고른 top-1 경로로도 같은 적합을 돌린다 ('정답 경로를 안다' 가정의 값)")
    a = ap.parse_args()
    import pandas as pd
    import torch
    from dataset_cached import CachedV4Dataset

    t_all = time.time()
    dev = C.pick_device(a.device)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    ds = CachedV4Dataset(C.VAL_CACHE)
    N = min(a.n, len(ds))
    df = pd.read_parquet(C.tag_dirs(a.ref_tag)["data"] / "scenarios.parquet").iloc[:N]
    g = df["gt_route"].to_numpy().astype(int)
    ar = np.arange(N)

    raw = {k: np.asarray(ds.raw(k)[:N]) for k in ("routes", "route_tan", "route_len", "route_sd0",
                                                  "v0", "h0", "y", "n_distinct", "route_fallback",
                                                  "route_mask")}
    # 정답 기준 경로 슬롯 g 의 배열만 뽑는다 (슬롯 r 은 경로 r % n_distinct 이므로 r < nd 면 경로 r)
    B = {"routes": raw["routes"][ar, g], "route_tan": raw["route_tan"][ar, g],
         "route_len": raw["route_len"][ar, g], "route_sd0": raw["route_sd0"][ar, g],
         "v0": raw["v0"], "h0": raw["h0"]}
    y = raw["y"]
    print(f"[setup] n={N} device={dev} 정답경로 슬롯 분포 {np.bincount(g, minlength=6).tolist()}",
          flush=True)

    # 시작 상태 (s0,d0) 결함 — 경로 g 에서 원점과 얼마나 떨어져 있나
    off0 = C.start_offset(B["routes"], B["route_tan"], B["route_len"], B["route_sd0"])
    print(f"[start] |P(s0)+N·d0| 평균 {off0.mean():.3f} m  p99 {np.percentile(off0,99):.3f} m  "
          f">0.5 m {100.0*(off0>0.5).mean():.2f}%", flush=True)

    # ---- 1. 정확 역변환
    t0 = time.time()
    s, d, res = frenet_invert(y, B["routes"], B["route_tan"], B["route_len"], B["route_sd0"],
                              dev, iters=a.inv_iters)
    th0, k0 = theta0_guard(B["routes"], B["route_tan"], B["route_len"], B["route_sd0"], B["h0"])
    GA = gt_actions(s, d, B["route_sd0"], B["v0"], th0)
    print(f"[invert] {time.time()-t0:.0f}s  잔차 평균 {res.mean():.4f} m  p99 {np.percentile(res,99):.4f} m "
          f"최대 {res.max():.4f} m", flush=True)

    # 되굴림 확인: 제약 없이 (a, dθ) 를 그대로 적분하면 y 를 재현해야 한다.
    # 단 rollout 은 v 에 relu, θ 에 ±90° clamp 를 건다 — 그 둘이 안 걸린 시나리오에서만 항등이다.
    ab = np.abs(GA["a"]); tb = np.degrees(np.abs(GA["dtheta"])); thb = np.degrees(np.abs(GA["theta"]))
    v_raw = B["v0"][:, None] + np.cumsum(GA["a"] * DT, 1)
    clamped = (thb > TH_MAX_DEG).any(1) | (v_raw < 0).any(1)
    v_re = np.maximum(v_raw, 0.0)
    th_re = np.clip(th0[:, None] + np.cumsum(GA["dtheta"], 1), -np.radians(TH_MAX_DEG),
                    np.radians(TH_MAX_DEG))
    s_re = B["route_sd0"][:, 0][:, None] + np.cumsum(v_re * np.cos(th_re) * DT, 1)
    d_re = B["route_sd0"][:, 1][:, None] + np.cumsum(v_re * np.sin(th_re) * DT, 1)
    err_all = np.abs(np.stack([s_re - s, d_re - d])).max(0).max(1)
    sd_err = float(err_all.max())
    sd_err_free = float(err_all[~clamped].max()) if (~clamped).any() else float("nan")
    print(f"[invert] (a,dθ) → (s,d) 되굴림 최대오차 전체 {sd_err:.2e} m / "
          f"clamp 안 걸린 {int((~clamped).sum())}개 {sd_err_free:.2e} m "
          f"(clamp 걸린 시나리오 {100.0*clamped.mean():.1f}%)", flush=True)

    # 정답 라벨의 저속 잡음 분리: AV2 속도 필드 평활값 >= MOVE_V 인 스텝만 따로 센다
    vf = np.load(C.tag_dirs(a.ref_tag)["data"] / "raw.npz")["v_fld"][:N, C.OBS:]   # (N,60)
    mv = vf >= C.MOVE_V
    def _pct(m, sel=None):
        return float(100.0 * (m if sel is None else m[sel]).mean())
    anyb = (ab > A_MAX) | (tb > DTH_MAX_DEG) | (thb > TH_MAX_DEG)
    oob = {
        "a_over_8_step_pct": _pct(ab > A_MAX),
        "dtheta_over_8deg_step_pct": _pct(tb > DTH_MAX_DEG),
        "theta_over_90deg_step_pct": _pct(thb > TH_MAX_DEG),
        "v_negative_step_pct": _pct(v_raw < 0),
        "any_step_pct": _pct(anyb),
        "a_over_8_scen_pct": _pct((ab > A_MAX).any(1)),
        "dtheta_over_8deg_scen_pct": _pct((tb > DTH_MAX_DEG).any(1)),
        "theta_over_90deg_scen_pct": _pct((thb > TH_MAX_DEG).any(1)),
        "any_scen_pct": _pct(anyb.any(1)),
        "a_p99": float(np.percentile(ab, 99)), "a_max": float(ab.max()),
        "dtheta_p99_deg": float(np.percentile(tb, 99)), "dtheta_max_deg": float(tb.max()),
        "theta_p99_deg": float(np.percentile(thb, 99)),
        # 움직이는 스텝만 (v_fld >= 1 m/s) — 정지 차량 위치 잡음을 뺀 값
        "moving_step_frac_pct": _pct(mv),
        "mv_a_over_8_step_pct": _pct(ab > A_MAX, mv),
        "mv_dtheta_over_8deg_step_pct": _pct(tb > DTH_MAX_DEG, mv),
        "mv_theta_over_90deg_step_pct": _pct(thb > TH_MAX_DEG, mv),
        "mv_any_step_pct": _pct(anyb, mv),
        "mv_any_scen_pct": _pct((anyb & mv).any(1)),
        "mv_a_p99": float(np.percentile(ab[mv], 99)) if mv.any() else float("nan"),
        "mv_dtheta_p99_deg": float(np.percentile(tb[mv], 99)) if mv.any() else float("nan"),
        "clamped_scen_pct": _pct(clamped),
    }
    print("[oob] 정답 액션열이 우리 상한 밖: " +
          "  ".join(f"{k} {v:.3f}" for k, v in oob.items() if k.endswith("pct")), flush=True)

    # ---- 2. 제약 적합
    fits = {}
    runs = [("ade", "ade", 0.0), ("sl1", "sl1", 0.0), ("sl1_jit1", "sl1", 1.0)]
    for name, obj, ws in runs:
        print(f"[fit] {name}", flush=True)
        t0 = time.time()
        r = fit_actions(B, y, dev, obj, w_smooth=ws, iters=a.iters, init=GA)
        r["sec"] = time.time() - t0
        fits[name] = r

    # ---- 2b. 모델이 고른 경로로 같은 적합 ('정답 경로를 안다' 가정이 얼마나 낙관적인가)
    route_cmp = None
    if a.also_top1:
        pz = np.load(C.tag_dirs(a.ref_tag)["data"] / "pred.npz")
        top1 = pz["top1"][:N].astype(int)
        nd = np.maximum(raw["n_distinct"].astype(int), 1)
        t1_slot = top1 % nd
        gt_all = pd.read_parquet(C.tag_dirs(a.ref_tag)["data"] / "scenarios.parquet")["gt_route"].to_numpy()
        agree_N = float(100.0 * (t1_slot == g).mean())
        agree_all = float(100.0 * ((pz["top1"].astype(int) % np.maximum(
            np.asarray(ds.raw("n_distinct")).astype(int), 1)) == gt_all).mean())
        B1 = {"routes": raw["routes"][ar, t1_slot], "route_tan": raw["route_tan"][ar, t1_slot],
              "route_len": raw["route_len"][ar, t1_slot], "route_sd0": raw["route_sd0"][ar, t1_slot],
              "v0": raw["v0"], "h0": raw["h0"]}
        print(f"[route] top-1 경로 == 정답 경로: 앞 {N} 에서 {agree_N:.1f}% / 전체 24,988 에서 {agree_all:.1f}%",
              flush=True)
        r1 = fit_actions(B1, y, dev, "sl1", w_smooth=0.0, iters=a.iters, init=None)
        r2 = fit_actions(B1, y, dev, "sl1", w_smooth=1.0, iters=a.iters, init=None)
        route_cmp = {
            "top1_eq_gt_route_pct_n": agree_N, "top1_eq_gt_route_pct_all": agree_all,
            "gt_route_sl1_ADE": float(fits["sl1"]["ade"].mean()),
            "top1_route_sl1_ADE": float(r1["ade"].mean()),
            "gt_route_sl1_FDE": float(fits["sl1"]["fde"].mean()),
            "top1_route_sl1_FDE": float(r1["fde"].mean()),
            "top1_route_sl1_jit1_ADE": float(r2["ade"].mean()),
            "top1_route_delta_ADE": float(r2["ade"].mean() - r1["ade"].mean()),
        }
        print("[route] " + json.dumps(route_cmp, ensure_ascii=False), flush=True)
        del pz

    # ---- 3. 집계
    cov_fail = ~df["coverage"].to_numpy()
    fb = df["fallback"].to_numpy() if "fallback" in df else raw["route_fallback"] > 0.5
    cls = df["cls"].to_numpy()
    res_json = {
        "n": int(N), "device": dev, "git_head": C.git_head(), "ref_tag": a.ref_tag,
        "val_cache": str(ds.dir), "iters": a.iters, "inv_iters": a.inv_iters,
        "limits": {"a_max_mps2": A_MAX, "dtheta_max_deg": DTH_MAX_DEG, "theta_max_deg": TH_MAX_DEG,
                   "sat_frac": SAT, "dt_s": DT},
        "start_offset_m": {"mean": float(off0.mean()), "p99": float(np.percentile(off0, 99)),
                           "over_0.5m_pct": float(100.0 * (off0 > 0.5).mean())},
        "invert_residual_m": {"mean": float(res.mean()), "p99": float(np.percentile(res, 99)),
                              "max": float(res.max()), "sd_roundtrip_max_m": sd_err,
                              "sd_roundtrip_max_m_no_clamp": sd_err_free,
                              "n_no_clamp": int((~clamped).sum())},
        "gt_action_out_of_bounds": oob,
        "coverage": {"fail_pct": float(100.0 * cov_fail.mean()),
                     "fallback_pct": float(100.0 * np.asarray(fb).mean()),
                     "n_fail": int(cov_fail.sum())},
        "fits": {}, "by_class": {}, "by_coverage": {}, "route_comparison": route_cmp,
    }
    for name, r in fits.items():
        h = r["hist"]
        # 마지막 두 기록 사이 ADE 변화가 1e-3 m 를 넘으면 아직 수렴 중이다
        conv = (abs(h[-1]["ade"] - h[-2]["ade"]) < 1e-3) if len(h) >= 2 else None
        res_json["fits"][name] = {
            "converged": conv,
            "ade_last_delta": (abs(h[-1]["ade"] - h[-2]["ade"]) if len(h) >= 2 else None),
            "ADE": float(r["ade"].mean()), "FDE": float(r["fde"].mean()),
            "ADE_p50": float(np.median(r["ade"])), "ADE_p90": float(np.percentile(r["ade"], 90)),
            "jitter_mean": float(r["jitter"].mean()),
            "saturation": sat_stats(r["a"], r["dtheta"]), "sec": r["sec"],
            "hist": r["hist"],
        }
    base = fits["sl1"]["ade"].mean()
    res_json["penalty_cost"] = {
        "ADE_sl1": float(base), "ADE_sl1_jit1": float(fits["sl1_jit1"]["ade"].mean()),
        "delta_ADE": float(fits["sl1_jit1"]["ade"].mean() - base),
        "FDE_sl1": float(fits["sl1"]["fde"].mean()),
        "FDE_sl1_jit1": float(fits["sl1_jit1"]["fde"].mean()),
        "delta_FDE": float(fits["sl1_jit1"]["fde"].mean() - fits["sl1"]["fde"].mean()),
    }
    for cname in list(C.CLASSES) + ["전체"]:
        m = np.ones(N, bool) if cname == "전체" else (cls == cname)
        if m.sum() == 0:
            continue
        row = {"n": int(m.sum())}
        for name, r in fits.items():
            row[name + "_ADE"] = float(r["ade"][m].mean())
            row[name + "_FDE"] = float(r["fde"][m].mean())
        row["oob_any_scen_pct"] = float(100.0 * ((np.abs(GA["a"]) > A_MAX) |
                                                 (np.degrees(np.abs(GA["dtheta"])) > DTH_MAX_DEG) |
                                                 (np.degrees(np.abs(GA["theta"])) > TH_MAX_DEG)
                                                 ).any(1)[m].mean())
        row["sat_any_scen_pct"] = float(100.0 * ((np.abs(fits["sl1"]["a"]) >= SAT * A_MAX) |
                                                 (np.abs(fits["sl1"]["dtheta"]) >=
                                                  SAT * np.radians(DTH_MAX_DEG))).any(1)[m].mean())
        res_json["by_class"][cname] = row
    for lab, m in (("경로 커버리지 성공", ~cov_fail), ("경로 커버리지 실패", cov_fail),
                   ("폴백 경로", np.asarray(fb, bool)), ("지도 경로", ~np.asarray(fb, bool))):
        if m.sum() == 0:
            continue
        res_json["by_coverage"][lab] = {
            "n": int(m.sum()),
            **{f"{k}_ADE": float(v["ade"][m].mean()) for k, v in fits.items()},
            **{f"{k}_FDE": float(v["fde"][m].mean()) for k, v in fits.items()}}

    (OUT / "data" / "labelfit.json").write_text(json.dumps(res_json, indent=2, ensure_ascii=False))
    np.savez(OUT / "data" / "gt_actions.npz", sid=df["sid"].to_numpy().astype("U36"),
             gt_route=g, cls=cls.astype("U8"), coverage=~cov_fail, fallback=np.asarray(fb, bool),
             v0=raw["v0"], start_off=off0, invert_res=res.astype(np.float32),
             v_fld=vf.astype(np.float32), moving=mv, y=y.astype(np.float32),
             **{k: v.astype(np.float32) for k, v in GA.items()},
             th0=th0.astype(np.float32), k0=k0.astype(np.float32),
             fit_ade=np.stack([fits[k]["ade"] for k in fits]).astype(np.float32),
             fit_fde=np.stack([fits[k]["fde"] for k in fits]).astype(np.float32),
             fit_names=np.array(list(fits)),
             fit_a=np.stack([fits[k]["a"] for k in fits]).astype(np.float32),
             fit_dtheta=np.stack([fits[k]["dtheta"] for k in fits]).astype(np.float32),
             fit_traj=np.stack([fits[k]["traj"] for k in fits]).astype(np.float32))
    print(json.dumps({k: res_json[k] for k in ("gt_action_out_of_bounds", "penalty_cost",
                                               "coverage")}, indent=2, ensure_ascii=False))
    print(f"[done] {OUT/'data'}  {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
