"""
lane_yaw_stats.py - 차선 heading 정렬 손실의 **허용 오차(tolerance)** 를 데이터에서 정한다 (2026-10-06).

왜 재나
-------
Greer et al. "Trajectory Prediction with a Lane Heading Auxiliary Loss"(arXiv:2011.06679) 의 YawLoss 는
예측 좌표에서 복원한 진행방향이 가장 가까운 차선의 heading 과 벌어진 만큼을 벌하되, **허용 오차 안에서는 0** 이다.
그 허용 오차를 임의로 정하면 실제 차선변경·회전을 벌하게 된다. 그래서 7.3°/step 임계를 정했을 때와 같은 방식으로
**정답 라벨의 분포에서** 정한다.

무엇을 재나 — 정답 궤적이 따라간 경로(후보 중 평균거리 최소)를 기준으로
    |wrap(psi_gt - k_lane)|   psi_gt = 정답 좌표의 진행방향, k_lane = 그 지점의 경로 접선각
정지 구간은 방향이 정의되지 않으므로 **속력 1 m/s 이상** 스텝만 센다.
정답의 총 회전량으로 구간을 나눠 따로 낸다 — 직진에서 큰 값이 나오면 그건 차선변경이다.

  python src/lane_yaw_stats.py                  # val 전체
  python src/lane_yaw_stats.py --limit 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VAL_CACHE = "/data/argoverse2/cache/v4/val_65ba9f4dcc91f9b5_n24988"
DT = 0.1
MOVE_V = 1.0
TURN_BINS = [(0.0, 5.0, "직진 <5°"), (5.0, 30.0, "완만 5-30°"), (30.0, 1e9, "회전 >=30°")]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def route_roughness(Y, R, T, M, SD, L):
    """정답의 속도 프로파일로 **경로 중심선을 정확히 따라갔을 때** 나오는 |Δψ| [°/step].

    왜 — 적분기는 h = k(s) + θ 로 궤적을 만든다. k 가 거칠면 모델이 무엇을 내든 그 거칠기가 궤적에 실린다.
    그래서 '모델 탓'과 '지도 탓'을 가르려면 **θ 를 0 으로 둔 궤적**의 방향 변화를 먼저 봐야 한다.
    k 는 model_v4.interp1d 와 같은 방식(접선 벡터를 선형보간한 뒤 atan2)으로 뽑는다 — 모델이 실제로 보는 값이다.
    비교 대상은 같은 시나리오 정답의 |Δψ| 다.
    """
    n, K, Mp, _ = R.shape
    dv = Y[:, 1:] - Y[:, :-1]
    sp = np.linalg.norm(dv, axis=-1) / DT                                    # (n,59) 정답 속력
    dist = np.linalg.norm(Y[:, None, :, None, :] - R[:, :, None, :, :], axis=-1)
    score = np.where(M > 0, dist.min(axis=3).mean(axis=2), np.inf)
    win = score.argmin(axis=1)
    ar = np.arange(n)
    Tw, Lw = T[ar, win], np.maximum(L[ar, win], 1e-3)
    s_t = SD[ar, win, 0][:, None] + np.cumsum(sp * DT, axis=1)               # (n,59) 호길이 진행
    idx = np.clip(s_t / Lw[:, None] * (Mp - 1), 0, Mp - 1 - 1e-4)
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    tx = np.take_along_axis(Tw[..., 0], i0, 1) * (1 - f) + np.take_along_axis(Tw[..., 0], i0 + 1, 1) * f
    ty = np.take_along_axis(Tw[..., 1], i0, 1) * (1 - f) + np.take_along_axis(Tw[..., 1], i0 + 1, 1) * f
    k = np.arctan2(ty, tx)
    dk = np.abs(np.degrees(wrap(k[:, 1:] - k[:, :-1])))                      # (n,58)
    ok = (sp[:, 1:] >= MOVE_V) & (sp[:, :-1] >= MOVE_V)
    return dk, ok


def route_roughness_all(Y, R, T, M, SD, L):
    """route_roughness 의 **전체 후보 경로** 판 (2026-10-06, 감독 세션 추가).

    왜 나눠 보나 — 승자 경로만 재면 정답과 같은 수준으로 매끄럽게 나온다(p99 4.79 vs 4.51°).
    승자는 '과거 적합도 1위'라 보통 차가 실제로 올라타 있는 멀쩡한 차로 사슬이기 때문이다.
    그런데 **흔들림 벌점과 L4 는 승자가 아니라 살아있는 모든 모드에 걸린다.** 거친 후보 위에서
    모델은 따라가서 벌점을 받거나 벗어나서 힌지를 받거나 둘 중 하나다. 그래서 '지도 탓'의
    크기를 보려면 후보 전체를 재야 한다.

    실측(val 500, 살아있는 경로 전부, 정답 속도로 실제 주행 구간만):
        평활 전      p99 6.80°, 7.3° 초과 0.813%
        sigma 1.0 m  p99 5.43°, 7.3° 초과 0.468%   (-42%)
        정답 궤적    p99 4.38°, 7.3° 초과 0.338%
    p99.9 은 정답이 더 두껍다(16.27 vs 14.15) — 극단은 실제 급기동이지 경로 탓이 아니다.

    반환은 route_roughness 와 같은 (dk, ok) 이고 모드 축을 평탄화해 (n*K_live, 58) 로 준다.
    """
    n, K, Mp, _ = R.shape
    dv = Y[:, 1:] - Y[:, :-1]
    sp = np.linalg.norm(dv, axis=-1) / DT                                    # (n,59)
    s_t = SD[..., 0][..., None] + np.cumsum(sp * DT, axis=1)[:, None, :]      # (n,K,59)
    Lk = np.maximum(L, 1e-3)[..., None]
    idx = np.clip(s_t / Lk * (Mp - 1), 0, Mp - 1 - 1e-4)
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    tx = np.take_along_axis(T[..., 0], i0, 2) * (1 - f) + np.take_along_axis(T[..., 0], i0 + 1, 2) * f
    ty = np.take_along_axis(T[..., 1], i0, 2) * (1 - f) + np.take_along_axis(T[..., 1], i0 + 1, 2) * f
    k = np.arctan2(ty, tx)                                                   # (n,K,59)
    dk = np.abs(np.degrees(wrap(k[..., 1:] - k[..., :-1])))                  # (n,K,58)
    ok = ((sp[:, 1:] >= MOVE_V) & (sp[:, :-1] >= MOVE_V))[:, None, :] & (M > 0)[..., None]
    return dk.reshape(-1, dk.shape[-1]), ok.reshape(-1, ok.shape[-1])


def block(Y, R, T, M):
    """한 묶음의 |psi_gt - k_lane| [°], 이동 마스크, 총 회전량 [°], 정답-경로 평균거리 [m]."""
    n = Y.shape[0]
    dv = Y[:, 1:] - Y[:, :-1]
    sp = np.linalg.norm(dv, axis=-1) / DT
    psi = np.arctan2(dv[..., 1], dv[..., 0])
    mid = 0.5 * (Y[:, 1:] + Y[:, :-1])                                           # 세그먼트 중점 (YawLoss 와 같은 짝)
    dist = np.linalg.norm(Y[:, None, :, None, :] - R[:, :, None, :, :], axis=-1)  # (n,6,60,64)
    near = dist.min(axis=3)
    score = np.where(M > 0, near.mean(axis=2), np.inf)
    win = score.argmin(axis=1)
    ar = np.arange(n)
    Rw, Tw = R[ar, win], T[ar, win]
    dm = np.linalg.norm(mid[:, :, None, :] - Rw[:, None, :, :], axis=-1)          # (n,59,64)
    j = dm.argmin(axis=2)
    tg = np.take_along_axis(Tw[:, None, :, :], j[..., None, None], axis=2)[:, :, 0]
    k = np.arctan2(tg[..., 1], tg[..., 0])
    err = np.abs(np.degrees(wrap(psi - k)))
    ok = sp >= MOVE_V
    turn = np.degrees(np.abs((wrap(psi[:, 1:] - psi[:, :-1]) * (ok[:, 1:] & ok[:, :-1])).sum(1)))
    return err, ok, turn, near[ar, win].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=VAL_CACHE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "runs" / "lane_yaw_stats.json"))
    a = ap.parse_args()
    d = Path(a.cache)
    n = a.limit or int(np.load(d / "y.npy", mmap_mode="r").shape[0])
    mmY, mmR = np.load(d / "y.npy", mmap_mode="r"), np.load(d / "routes.npy", mmap_mode="r")
    mmT, mmM = np.load(d / "route_tan.npy", mmap_mode="r"), np.load(d / "route_mask.npy", mmap_mode="r")
    # (N,6,60,64) 거리 텐서는 전체를 한 번에 잡으면 수 GB 라 청크로 돈다
    errs, oks, turns, fars = [], [], [], []
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        e, o, t, f = block(np.asarray(mmY[lo:hi], np.float64), np.asarray(mmR[lo:hi], np.float64),
                           np.asarray(mmT[lo:hi], np.float64), np.asarray(mmM[lo:hi], np.float64))
        errs.append(e); oks.append(o); turns.append(t); fars.append(f)
    err = np.concatenate(errs); ok = np.concatenate(oks)
    turn = np.concatenate(turns); far = np.concatenate(fars)

    def stat(sel):
        v = err[sel]
        if v.size == 0:
            return None
        return {"steps": int(v.size), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99)), "p99.9": float(np.percentile(v, 99.9)),
                "mean": float(v.mean()), "max": float(v.max())}

    res = {"n": n, "cache": str(d), "all": stat(ok), "by_turn": {}}
    print(f"정답의 |psi - 차선 heading| [°]  (n={n:,}, 속력 >= {MOVE_V} m/s 스텝만)")
    print(f"{'구간':14s} {'스텝수':>10s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'p99.9':>8s} {'최대':>7s}")
    s = res["all"]
    print(f"{'전체':14s} {s['steps']:10,d} {s['p50']:7.2f} {s['p90']:7.2f} {s['p95']:7.2f} {s['p99']:7.2f} {s['p99.9']:8.2f} {s['max']:7.1f}")
    for lo, hi, lab in TURN_BINS:
        sel = ok & ((turn >= lo) & (turn < hi))[:, None]
        st = stat(sel)
        res["by_turn"][lab] = st
        if st:
            print(f"{lab:14s} {st['steps']:10,d} {st['p50']:7.2f} {st['p90']:7.2f} {st['p95']:7.2f} "
                  f"{st['p99']:7.2f} {st['p99.9']:8.2f} {st['max']:7.1f}")
    # 경로 자체가 정답에서 멀면 접선각 비교가 무의미하다 — 얼마나 되는지 같이 적는다
    res["route_dist_m"] = {"p50": float(np.percentile(far, 50)), "p90": float(np.percentile(far, 90)),
                           "over_3m_pct": float(100.0 * (far > 3.0).mean())}
    print(f"\n정답-경로 평균거리: 중앙 {res['route_dist_m']['p50']:.2f} m, p90 {res['route_dist_m']['p90']:.2f} m, "
          f"3 m 초과 {res['route_dist_m']['over_3m_pct']:.1f}%")
    sel = ok & (far <= 3.0)[:, None]
    res["route_close"] = stat(sel)
    s = res["route_close"]
    print(f"경로가 3 m 안인 경우만: p90 {s['p90']:.2f} p95 {s['p95']:.2f} p99 {s['p99']:.2f} p99.9 {s['p99.9']:.2f}°")
    # ---------- 경로 중심선 자체의 거칠기 (θ=0 궤적)
    rk, rok = [], []
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        dk, ok2 = route_roughness(np.asarray(mmY[lo:hi], np.float64), np.asarray(mmR[lo:hi], np.float64),
                                  np.asarray(mmT[lo:hi], np.float64), np.asarray(mmM[lo:hi], np.float64),
                                  np.asarray(np.load(d / "route_sd0.npy", mmap_mode="r")[lo:hi], np.float64),
                                  np.asarray(np.load(d / "route_len.npy", mmap_mode="r")[lo:hi], np.float64))
        rk.append(dk); rok.append(ok2)
    rk, rok = np.concatenate(rk), np.concatenate(rok)
    v = rk[rok]
    gt_dpsi = []
    dvv = np.asarray(mmY[:n], np.float64)
    dd = dvv[:, 1:] - dvv[:, :-1]
    spp = np.linalg.norm(dd, axis=-1) / DT
    pp = np.arctan2(dd[..., 1], dd[..., 0])
    gp = np.abs(np.degrees(wrap(pp[:, 1:] - pp[:, :-1])))
    gok = (spp[:, 1:] >= MOVE_V) & (spp[:, :-1] >= MOVE_V)
    gv = gp[gok]
    res["route_roughness"] = {
        "설명": "정답 속도로 경로 중심선을 정확히 따라갔을 때의 |Δψ| (θ=0). k 는 model_v4.interp1d 와 같은 선형보간",
        "경로": {"steps": int(v.size), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)), "p99.9": float(np.percentile(v, 99.9)),
                "over_7.3_pct": float(100.0 * (v > 7.3).mean())},
        "정답": {"steps": int(gv.size), "p50": float(np.percentile(gv, 50)), "p90": float(np.percentile(gv, 90)),
                "p99": float(np.percentile(gv, 99)), "p99.9": float(np.percentile(gv, 99.9)),
                "over_7.3_pct": float(100.0 * (gv > 7.3).mean())}}
    print("\n경로 중심선을 그대로 따라갔을 때 vs 정답 궤적 — |Δψ| [°/step]")
    print(f"{'':10s} {'p50':>7s} {'p90':>7s} {'p99':>7s} {'p99.9':>8s} {'7.3°초과':>9s}")
    for lab in ("경로", "정답"):
        q = res["route_roughness"][lab]
        print(f"{lab:10s} {q['p50']:7.2f} {q['p90']:7.2f} {q['p99']:7.2f} {q['p99.9']:8.2f} {q['over_7.3_pct']:8.2f}%")
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("저장", a.out)


if __name__ == "__main__":
    main()
