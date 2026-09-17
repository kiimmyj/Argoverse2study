"""
viz_v4_compare.py - 네 판 비교: 15에폭 상수 학습률 vs 30에폭 코사인 × 10 Hz vs 2 Hz 입력 (2026-09-17).

판 (모두 L4nw · (a, h) · 전체 데이터 199,908 · 흔들림 벌점 1.0 · θ0 guard · 시드 0 · best 체크포인트)
  h10_c15  v4_l4nw_ah2_full_sm1_s0            10 Hz · 15에폭 · 학습률 5e-4 고정
  h2_c15   v4_l4nw_ah2_2hz_full_sm1_s0        2 Hz  · 15에폭 · 학습률 5e-4 고정
  h10_c30  v4_l4nw_ah2_full_sm1_cos30_s0      10 Hz · 30에폭 · 코사인 5e-4 → 1e-5 (스텝 단위)
  h2_c30   v4_l4nw_ah2_2hz_full_sm1_cos30_s0  2 Hz  · 30에폭 · 코사인
입력  runs/<tag>.json (history) · viz/v4/<tag>/data/{scenarios.parquet, pred.npz, raw.npz} (viz_v4_dump.py 결과)
색 규칙 (모든 그림 같음)  색 = 입력 주기 (10 Hz 파랑 · 2 Hz 주황), 스케줄 = 선·채움
  (30에폭 코사인 진한 실선·채운 점, 15에폭 상수 옅은 점선·빈 점). 어두운 지도에서는 10 Hz 초록 · 2 Hz 분홍.

절
  a  학습 곡선 (a1) · 최근 5에폭 요동 (a2)
  b  15 상수 → 30 코사인: 조건별 Δ (b1) · 오차 출처·손실·선택·다양성 (b2) · 상황별 값 (b3) · 묶음 조건별 값 (b4)
     · 분기 선택 오류의 성격 (b5: 옆 차로 / 다른 방향, 옆 차로 슬롯인데 궤적은 정답 차로에 머무는가 — 정답 경로 투영)
  c  10 Hz vs 2 Hz (둘 다 30): ΔminFDE6 분포 (c1) · 조건별 Δ (c2) · 결정 트리 (c3) · 램프·최근 동역학 점검 (c4)
     · 다른 학습 쌍(15에폭) 재현성과 가설 H3(10 Hz focal heading 뒤집힘) (c5)
  d  실현가능성 (d1: 각도 초과·흔들림·저크·이탈 · 끝 속력비 · 평균 a(t) · 1~2 s 출렁임(원 출력 / 정답과 같은 평활) · 끝 감속)
  e  평균 사례 궤적 — 10 Hz·30 판 평균값 기준 9개에서 네 판의 best of 6 (e1, 칸 제목 = 승자 모드 ADE) · 확률 1위 (e2)
  f  고정 시나리오 14개의 네 판 (f0) · 발견별 대표 시나리오 (f1–f8, d 칸은 예측을 정답 경로에 투영,
     상황 분류 인공물 후보는 선정에서 빼고 각주를 단다)
그리기 규칙  원점 -> 첫 예측점(t = 0.1 s)은 가는 점선(캐시 출발점 결함이 꺾인 선처럼 보이지 않게) · 과거에 1 s 점과
     2 Hz 판 입력 시점(빈 원) · 어두운 지도의 정답은 예측 위에 가늘게
  v  숫자 교차 검증 (다른 코드 경로)
출력  viz/v4/cos30_compare/*.png · viz/v4/cos30_compare/data/summary.json (+ tree_rules.txt)

  python src/viz_v4_compare.py              # 전부
  python src/viz_v4_compare.py --only a,b   # 일부 절만 (summary.json 은 기존 것과 합쳐 쓴다)
"""
import argparse
import json
import statistics
import textwrap
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C

OUT = C.VIZ_ROOT / "cos30_compare"
DATA = OUT / "data"
RUNS = [("h10_c15", "v4_l4nw_ah2_full_sm1_s0", "10 Hz · 15에폭 상수"),
        ("h2_c15", "v4_l4nw_ah2_2hz_full_sm1_s0", "2 Hz · 15에폭 상수"),
        ("h10_c30", "v4_l4nw_ah2_full_sm1_cos30_s0", "10 Hz · 30에폭 코사인"),
        ("h2_c30", "v4_l4nw_ah2_2hz_full_sm1_cos30_s0", "2 Hz · 30에폭 코사인")]
ORDER = ["h10_c15", "h10_c30", "h2_c15", "h2_c30"]          # 그림 순서: 입력별로 15 → 30
TAG = {k: t for k, t, _ in RUNS}
LAB = {k: l for k, _, l in RUNS}
SHORT = {"h10_c15": "10Hz·15", "h10_c30": "10Hz·30", "h2_c15": "2Hz·15", "h2_c30": "2Hz·30"}
REF = "h10_c30"                                              # 모델과 무관한 표·평균 사례 기준 판
LIGHT_BLUE, LIGHT_ORANGE = "#7fb0ea", "#f29a70"
COL = {"h10_c15": LIGHT_BLUE, "h10_c30": C.C_BLUE, "h2_c15": LIGHT_ORANGE, "h2_c30": C.C_ORANGE}
LS = {"h10_c15": (0, (4, 2)), "h2_c15": (0, (4, 2)), "h10_c30": "-", "h2_c30": "-"}
MK = {"h10_c15": "o", "h10_c30": "o", "h2_c15": "s", "h2_c30": "s"}
FILL = {"h10_c15": False, "h2_c15": False, "h10_c30": True, "h2_c30": True}
# 어두운 지도용 (viz_v4_gallery 비교 그림과 같은 색: 10 Hz 초록, 2 Hz 분홍)
DCOL = {"h10_c30": "#62f08e", "h10_c15": "#b4f8c9", "h2_c30": "#ff6fd2", "h2_c15": "#ffb8ea"}
DLS = {"h10_c30": "-", "h2_c30": "-", "h10_c15": (0, (3.0, 1.7)), "h2_c15": (0, (3.0, 1.7))}
# 같은 길을 가면 뒤에 그린 선이 앞 선을 다 가린다 -> 10 Hz 는 굵게 아래에, 2 Hz 는 가늘게 위에 (겹쳐도 테두리로 보인다)
DLW = {"h10_c30": 4.6, "h2_c30": 1.9, "h10_c15": 3.2, "h2_c15": 1.5}
DPE = {"h10_c30": 1.4, "h10_c15": 1.2, "h2_c30": 0.8, "h2_c15": 0.7}      # 외곽선 두께 (선 굵기에 더함)
DMS = {"h10_c30": 36, "h10_c15": 24, "h2_c30": 15, "h2_c15": 11}          # 1초 점 크기
C_DIFF_SCHED = C.C_VIOLET      # Δ 막대: 30 − 15 는 입력색, 2 Hz − 10 Hz 는 보라(30) · 회색(15)
PRED_FIELDS = ("traj", "prob", "alive", "v", "a", "theta", "dtheta", "d", "h", "band", "jit_t_sum", "jit_a_sum",
               "exc_cnt", "off_frac")
RAW_FIELDS = ("pos", "v_pos", "v_fld", "a_fld", "h", "lead_dist", "gt_d_g", "y", "route_meand", "route_maxth")
SUM = {}
TIE_M = 0.5            # ΔminFDE6 |차| < 0.5 m 는 '비김'
TREE_DEPTH, TREE_LEAF, TREE_SEED = 4, 200, 0


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def note(fig, text, y=0.006, fs=8.0):
    fig.text(0.01, y, text, fontsize=fs, color=C.INK2, ha="left", va="bottom")


def run_handles(keys=ORDER, lw=2.0):
    from matplotlib.lines import Line2D
    return ([Line2D([], [], color=COL[k], ls=LS[k], lw=lw, marker=MK[k], ms=6, mfc=COL[k] if FILL[k] else "white",
                    mec=COL[k], mew=1.3) for k in keys], [LAB[k] for k in keys])


def bar_kw(k):
    return dict(color=COL[k] if FILL[k] else "white", edgecolor=COL[k], linewidth=1.6,
                hatch=None if FILL[k] else "////")


# =========================================================================== 데이터
class Data:
    """네 판의 표·예측 배열과 모델과 무관한 원본 파생량(기준 판 raw.npz)."""

    def __init__(self):
        import pandas as pd
        import viz_v4_stats as S
        t0 = time.time()
        self.df, self.P, self.meta = {}, {}, {}
        for k in ORDER:
            d = C.tag_dirs(TAG[k])["data"]
            self.df[k] = pd.read_parquet(d / "scenarios.parquet")
            pr = np.load(d / "pred.npz")
            self.P[k] = {f: pr[f] for f in PRED_FIELDS}
            self.meta[k] = json.loads((d / "meta.json").read_text())
            if not self.meta[k]["repro"]["ok"]:
                raise SystemExit(f"{k}: 덤프 재현 확인이 통과하지 않았다")
        raw = np.load(C.tag_dirs(TAG[REF])["data"] / "raw.npz")
        self.R = {f: raw[f] for f in RAW_FIELDS}
        # 모델과 무관한 열·배열이 네 덤프에서 같은지 (같은 val 순서·같은 분류)
        same_cols = ["sid", "cls", "dh6", "dd6", "v0", "v0_f", "amax_f", "amin_f", "gt_route", "n_distinct",
                     "n_lanes30", "n_reachable", "fallback", "coverage", "gap49", "thw49", "has_lead49", "n_alive"]
        chk = {}
        for k in ORDER:
            a, b = self.df[REF][same_cols], self.df[k][same_cols]
            eq = all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) or
                     np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True, atol=0)
                     if a[c].dtype != object else np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in same_cols)
            rw = np.load(C.tag_dirs(TAG[k])["data"] / "raw.npz")
            pos_d = float(np.abs(rw["pos"] - self.R["pos"]).max())
            chk[k] = {"table_same": bool(eq), "pos_max_abs_diff": pos_d}
            if not eq or pos_d > 0:
                raise SystemExit(f"{k}: 모델과 무관한 열이 기준 판과 다르다 {chk[k]}")
        SUM["model_independent_check"] = chk
        self.N = len(self.df[REF])
        T = S.prep(self.df[REF])
        T["nd_lab"] = np.where(T["fallback"], "폴백", T["n_distinct"].astype(str))
        self.T = T
        self._features()
        print(f"[data] 네 판 {self.N:,} 시나리오 · {time.time() - t0:.0f}s", flush=True)

    def _features(self):
        """램프·최근 동역학 등 시나리오 특성 (정답·원본 위치에서 계산, 모델과 무관)."""
        T, R = self.T, self.R
        pos = R["pos"].astype(np.float64)
        sp = np.linalg.norm(np.diff(pos, axis=1), axis=2) / C.DT          # (N,109) 스텝 t -> t+1 속력
        ref = sp[:, 10:15].mean(1)                                          # 1.0–1.5 s 구간
        moving = ref > 3.0
        T["ramp_ratio"] = np.where(moving, sp[:, 0] / np.maximum(ref, 1e-6), np.nan)
        # 10 Hz 입력 a 채널이 첫 0.5 s 에 담는 가짜 가속 [m/s²] (위치차분, 관측 안)
        T["ramp_acc"] = np.where(moving, (sp[:, 5] - sp[:, 0]) / 0.5, np.nan)
        # 관측 마지막 1초의 동역학 (관측만 사용): 속력 변화 [m/s²], 진행방향 변화율 [°/s]
        T["a_last"] = (sp[:, 44:49].mean(1) - sp[:, 39:44].mean(1)) / 0.5
        h = R["h"].astype(np.float64)
        yaw = np.degrees(h[:, 48] - h[:, 38]) / 1.0
        T["yaw_last"] = np.where(sp[:, 38:48].min(1) >= C.MOVE_V, yaw, 0.0)
        T["abs_yaw_last"] = T["yaw_last"].abs()
        T["abs_a_last"] = T["a_last"].abs()
        # θ0 (슬롯 0 = 경로 0, 첫 예측 스텝의 θ — 모델 출력이지만 |dθ| ≤ 8° 라 입력의 θ0 와 8° 안)
        T["theta0"] = np.degrees(np.abs(self.P[REF]["theta"][:, 0, 0].astype(np.float64)))
        T["v_pos49"] = sp[:, 48]
        # 상황 분류 인공물 후보 (분류기는 바꾸지 않는다 — 대표 사례 선정에서 빼고 그림에 각주만 단다)
        #  · 회전 라벨인데 위치로는 직진 (C.turn_straight_by_position, 검토 r10 과 같은 규칙: 3,932 중 107)
        #  · 회전 라벨인데 6초 끝 횡변위 < 1 m (정지 직전 방향 잡음 등, 선정에서만 뺀다)
        #  · 급감속 라벨인데 최대 가속도도 급가속 임계 초과 (경계 사례)
        flag, alt = C.turn_straight_by_position(pos, T["cls"].to_numpy())
        T["turn_alt"] = alt
        T["turn_straight"] = flag
        path6 = np.linalg.norm(np.diff(pos[:, C.OBS - 1:], axis=1), axis=2).sum(1)
        turn = T["cls"].isin(["좌회전", "우회전"]).to_numpy()
        T["turn_suspect"] = flag | (turn & (np.abs(pos[:, -1, 1]) < 1.0) & (path6 >= 4.0))
        T["dec_acc_border"] = (T["cls"] == "급감속").to_numpy() & (T["amax_f"].to_numpy() > C.ACC_A)
        T["y_end"] = pos[:, -1, 1]

    # 판별 열
    def col(self, k, c):
        return self.df[k][c].to_numpy()


def paired(a, b):
    """b − a 의 평균과 95% 신뢰 반폭 (시나리오 짝)."""
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan, 0
    h = 1.96 * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return float(d.mean()), float(h), n


def run_metrics(D, k):
    """판 하나의 시나리오별 지표 (b·c·d 공통)."""
    df, P = D.df[k], D.P[k]
    out = {"minade": df["minade"].to_numpy(np.float64), "minfde": df["minfde"].to_numpy(np.float64),
           "miss": df["miss"].to_numpy(float), "top1_ade": df["top1_ade"].to_numpy(np.float64),
           "top1_fde": df["top1_fde"].to_numpy(np.float64),
           "abs_ds": df["ds_end"].abs().to_numpy(np.float64), "abs_dd": df["dd_end"].abs().to_numpy(np.float64),
           "win_ne_top1": (~df["win_eq_top1"]).to_numpy(float),
           "gt_top1": np.where(df["n_distinct"] >= 2, (df["gt_route_rank"] == 1).astype(float), np.nan),
           "route_err": np.where(df["n_distinct"] >= 2, df["route_err"].astype(float), np.nan),
           "win_route_err": np.where(df["n_distinct"] >= 2, df["win_route_err"].astype(float), np.nan),
           "l_sl1": df["l_sl1"].to_numpy(np.float64), "l_ce": df["l_ce"].to_numpy(np.float64),
           "l_hinge": df["l_hinge"].to_numpy(np.float64), "l_jit": df["l_jit"].to_numpy(np.float64),
           "spread": np.where(df["n_alive"] >= 2, df["spread"], np.nan).astype(np.float64),
           "eff_br": np.where(df["n_distinct"] >= 2, df["eff_branches"], np.nan).astype(np.float64),
           "n_clusters": np.where(df["n_alive"] >= 2, df["n_end_clusters"], np.nan).astype(np.float64),
           "offlane": df["offlane"].to_numpy(np.float64), "exc_pct": df["exc_pct"].to_numpy(np.float64)}
    return out


# =========================================================================== a 학습 곡선
def sec_a(D=None):
    plt = C.setup_mpl()
    from matplotlib.lines import Line2D
    H, A = {}, {}
    for k in ORDER:
        j = json.loads((C.RUNS / f"{TAG[k]}.json").read_text())
        H[k], A[k] = j["history"], j["args"]
    for k in ORDER:
        for h in H[k]:
            h["slce"] = h["loss"] - A[k]["offlane"] * h["offlane"] - A[k]["smooth"] * h["smooth"]
            h.setdefault("lr", A[k]["lr"])          # 상수 판은 history 에 lr 이 없다 -> 인자 값
    panels = [("minADE6", "minADE6 [m]"), ("minFDE6", "minFDE6 [m]"), ("val_offlane_steps", "이탈 (시나리오당 0~6)"),
              ("val_dtheta_over_label_pct", "7.3°초과 [%]"), ("val_jitter", "흔들림 θ + a (무차원)"),
              ("slce", "학습 손실 중 SL1 + CE (train)"), ("lr", "학습률 (에폭 끝, log)"), ("sec", "에폭 시간 [s]")]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.4), sharex=True)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.84, bottom=0.08, wspace=0.26, hspace=0.3)
    for ax, (key, lab) in zip(axes.flat, panels):
        for k in ORDER:
            ep = [h["epoch"] for h in H[k]]
            ys = [h[key] for h in H[k]]
            if key == "lr":
                # 두 입력의 학습률은 같은 값이다 — 10 Hz 를 굵게 밑에, 2 Hz 를 가늘게 위에 그려 둘 다 보이게
                wide = k.startswith("h10")
                ax.plot(ep, ys, color=COL[k], ls=LS[k], lw=5.0 if wide else 1.6, marker=MK[k],
                        ms=7.5 if wide else 3.6, mfc=COL[k] if FILL[k] else "white", mec=COL[k], mew=1.2,
                        zorder=2 if wide else 4)
                continue
            ax.plot(ep, ys, color=COL[k], ls=LS[k], lw=2.0, marker=MK[k], ms=4.6,
                    mfc=COL[k] if FILL[k] else "white", mec=COL[k], mew=1.2, zorder=3)
            b = min(H[k], key=lambda h: h["minADE6"])
            ax.scatter([b["epoch"]], [b[key]], s=150, facecolors="none", edgecolors=COL[k], linewidths=1.6, zorder=4)
        ax.axvspan(10.5, 15.5, color=C.GRID, alpha=0.5, lw=0, zorder=0)
        ax.axvspan(25.5, 30.5, color=C.GRID, alpha=0.9, lw=0, zorder=0)
        if key == "lr":
            ax.set_yscale("log")
            tk = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4]
            ax.set_yticks(tk)
            ax.set_yticklabels([f"{t:.0e}".replace("e-0", "e-") for t in tk])
            ax.minorticks_off()
            ax.text(0.02, 0.04, "두 입력의 학습률은 같다\n(10 Hz 굵게 밑 · 2 Hz 가늘게 위)", transform=ax.transAxes,
                    fontsize=8, color=C.INK2, va="bottom")
        ax.set_title(lab, loc="left", fontsize=10.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlim(0.3, 30.7)
    for ax in axes[1]:
        ax.set_xlabel("에폭")
    hs, ls = run_handles()
    hs.append(Line2D([], [], color=C.INK2, marker="o", ms=11, mfc="none", ls="none"))
    ls.append("best 에폭 (그 판의 val minADE6 최소)")
    from matplotlib.patches import Patch
    hs += [Patch(facecolor=C.GRID, alpha=0.5), Patch(facecolor=C.GRID, alpha=0.9)]
    ls += ["에폭 11–15 (15에폭 판의 최근 5)", "에폭 26–30 (30에폭 판의 최근 5)"]
    fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.045, 0.935), ncol=4, fontsize=9.2)
    fig.suptitle("a1 · 학습 곡선 — 네 판의 에폭별 val 평가 (val 24,988, 학습 로그 그대로)", x=0.01, y=0.985,
                 ha="left", fontsize=13.5)
    note(fig, "색 = 입력 주기, 선 = 스케줄(실선 30에폭 코사인 5e-4→1e-5 스텝 단위 / 점선 15에폭 5e-4 고정). "
              "SL1 + CE = 학습 손실 − 1.0·이탈 hinge − 1.0·흔들림. 흔들림 = train_v4.evaluate 의 θ·a 두 항 합. 시드 1개. "
              "스케줄과 에폭 수를 함께 바꿨고 대조판(상수 30에폭 · 코사인 15에폭)이 없다. 30에폭 두 판은 동시에 학습했다(에폭 시간은 부하를 나눠 쓴 값).")
    C.savefig(fig, OUT / "a1_learning_curves.png")

    # ---- 요동 (최근 5 에폭 표준편차)
    mets = [("minADE6", "minADE6 [m]", "{:.3f}"), ("minFDE6", "minFDE6 [m]", "{:.3f}"),
            ("val_offlane_steps", "이탈", "{:.3f}"), ("val_dtheta_over_label_pct", "7.3°초과 [%]", "{:.3f}"),
            ("val_jitter", "흔들림 θ + a", "{:.5f}")]
    fl = {}
    for k in ORDER:
        fl[k] = {}
        n_ep = len(H[k])
        for key, _, _ in mets:
            ys = np.array([h[key] for h in H[k]], float)
            last = ys[n_ep - 5:]
            mid = ys[10:15]
            fl[k][key] = {"last5_epochs": [n_ep - 4, n_ep], "last5_mean": float(last.mean()),
                          "last5_std": float(last.std(ddof=1)),
                          "last5_absdiff_median": float(np.median(np.abs(np.diff(last)))),
                          "ep11_15_mean": float(mid.mean()), "ep11_15_std": float(mid.std(ddof=1)),
                          "ep11_15_absdiff_median": float(np.median(np.abs(np.diff(mid))))}
        fl[k]["lr_last5"] = [float(h["lr"]) for h in H[k][n_ep - 5:]]
        fl[k]["lr_ep11_15"] = [float(h["lr"]) for h in H[k][10:15]]
        b = min(H[k], key=lambda h: h["minADE6"])
        fl[k]["best"] = {"epoch": b["epoch"], "minADE6": b["minADE6"], "minFDE6": b["minFDE6"]}
        fl[k]["final"] = {"epoch": H[k][-1]["epoch"], "minADE6": H[k][-1]["minADE6"],
                          "minFDE6": H[k][-1]["minFDE6"]}
        fl[k]["epoch_sec_median"] = float(np.median([h["sec"] for h in H[k]]))
        fl[k]["total_min"] = float(sum(h["sec"] for h in H[k]) / 60)
        fl[k]["slce_last5"] = float(np.mean([h["slce"] for h in H[k][n_ep - 5:]]))
        fl[k]["loss_last5"] = float(np.mean([h["loss"] for h in H[k][n_ep - 5:]]))
    SUM["a_fluct"] = fl
    # 같은 에폭 구간(1–15)에서 두 스케줄 비교: 에폭별 minADE6 차
    SUM["a_same_epochs"] = {inp: {"ep1_15_minade_c30_minus_c15": [
        float(H[f"{inp}_c30"][e]["minADE6"] - H[f"{inp}_c15"][e]["minADE6"]) for e in range(15)]}
        for inp in ("h10", "h2")}
    fig, axes = plt.subplots(1, 5, figsize=(21, 5.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.76, bottom=0.2, wspace=0.3)
    slots = [("h10_c15", "last5"), ("h10_c30", "ep11_15"), ("h10_c30", "last5"),
             ("h2_c15", "last5"), ("h2_c30", "ep11_15"), ("h2_c30", "last5")]
    xl = ["10Hz·15\n11–15", "10Hz·30\n11–15", "10Hz·30\n26–30",
          "2Hz·15\n11–15", "2Hz·30\n11–15", "2Hz·30\n26–30"]
    xs = np.array([0, 1, 2, 3.4, 4.4, 5.4])
    for ax, (key, lab, fmt) in zip(axes, mets):
        for x, (k, win) in zip(xs, slots):
            v = fl[k][key][f"{win}_std"]
            kw = bar_kw(k)
            if win == "ep11_15" and k.endswith("c30"):
                kw = dict(color="white", edgecolor=COL[k], linewidth=1.6, hatch="..")
            ax.bar(x, v, width=0.8, zorder=3, **kw)
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=7.6, color=C.INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels(xl, fontsize=7.8)
        ax.set_xlabel("판 · 에폭 창", fontsize=8.5)
        ax.set_title(f"{lab}\n5에폭 창 표준편차", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.suptitle("a2 · 에폭 사이 요동 — 5에폭 창의 표준편차(ddof=1)", x=0.01, y=0.97, ha="left", fontsize=13)
    from matplotlib.patches import Patch
    hs = [Patch(facecolor="white", edgecolor=C.INK2, hatch="////"), Patch(facecolor="white", edgecolor=C.INK2, hatch=".."),
          Patch(facecolor=C.INK2)]
    fig.legend(hs + [Patch(facecolor=C.C_BLUE), Patch(facecolor=C.C_ORANGE)],
               ["15에폭 상수 · 최근 5 (에폭 11–15)", "30에폭 코사인 · 에폭 11–15 (학습률 3.5e-4→2.5e-4)",
                "30에폭 코사인 · 최근 5 (에폭 26–30, 학습률 3.1e-5→1e-5)", "10 Hz", "2 Hz"],
               loc="upper left", bbox_to_anchor=(0.04, 0.925), ncol=5, fontsize=9)
    lr11 = fl["h10_c30"]["lr_ep11_15"]
    lr26 = fl["h10_c30"]["lr_last5"]
    note(fig, f"30에폭 판 학습률(에폭 끝): 에폭 11–15 {lr11[0]:.1e}→{lr11[-1]:.1e}, 에폭 26–30 {lr26[0]:.1e}→{lr26[-1]:.1e}. "
              "15에폭 판은 5e-4 고정. val 24,988 평가라 요동에는 학습의 확률성만 들어 있다(같은 val 집합). 시드 1개. "
              "창마다 5점뿐이고, 학습률 차이는 에폭 순서(학습 진행)와 겹쳐 있어 요동 감소의 원인을 가를 수 없다.")
    C.savefig(fig, OUT / "a2_fluctuation.png")
    print("[a] 학습 곡선·요동", flush=True)
    return H


# =========================================================================== b·c 조건별 Δ
GROUPINGS = [("cls", C.CLASSES, "상황 (정답 기반)"),
             ("v0_bin", C.BINS["v0"][1], "v0 [m/s]"),
             ("nd_lab", ["폴백", "1", "2", "3", "4", "5", "6"], "구별 분기 수"),
             ("reach_bin", ["0", "1–5", "6–10", "11–20", "21–30", ">30"], "도달 가능 차로 수"),
             ("lane30_bin", ["0–5", "6–10", "11–15", "16–20"], "30 m 안 차로 수")]
DMETS = [("minade", "ΔminADE6 [m]", 1.0), ("minfde", "ΔminFDE6 [m]", 1.0), ("miss", "Δmiss [%p]", 100.0)]


def delta_table(D, M, pairs):
    out = {}
    for lab, ka, kb, _ in pairs:
        out[lab] = {}
        for mk, _, sc in DMETS:
            a_all, b_all = M[ka][mk], M[kb][mk]
            m_all, h_all, n_all = paired(a_all, b_all)
            out[lab][mk] = {"all": {"n": n_all, "a": float(a_all.mean() * sc), "b": float(b_all.mean() * sc),
                                    "delta": m_all * sc, "ci95": h_all * sc}}
            for gk, levels, _ in GROUPINGS:
                g = D.T[gk].to_numpy()
                cell = {}
                for lv in levels:
                    m = g == lv
                    if not m.any():
                        continue
                    dm, hh, n = paired(a_all[m], b_all[m])
                    cell[str(lv)] = {"n": n, "a": float(a_all[m].mean() * sc), "b": float(b_all[m].mean() * sc),
                                     "delta": dm * sc, "ci95": hh * sc,
                                     "contrib_pct": 100.0 * n * dm / (n_all * m_all) if m_all else np.nan}
                out[lab][mk][gk] = cell
    return out


def fig_delta_grid(D, tab, pairs, title, foot, path):
    plt = C.setup_mpl()
    from matplotlib.patches import Patch
    nr, nc = len(DMETS), len(GROUPINGS)
    W = 24
    H = 3.3 * nr + 1.9
    fig, axes = plt.subplots(nr, nc, figsize=(W, H), sharex="row",
                             gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.05, 1.0, 0.9]))
    fig.subplots_adjust(left=0.075, right=0.99, top=1 - 1.45 / H, bottom=1.0 / H, wspace=0.42, hspace=0.36)
    npair = len(pairs)
    off = (np.arange(npair) - (npair - 1) / 2) * 0.34
    for i, (mk, mlab, _) in enumerate(DMETS):
        for j, (gk, levels, glab) in enumerate(GROUPINGS):
            ax = axes[i, j]
            cells = tab[pairs[0][0]][mk][gk]
            lv = [str(x) for x in levels if str(x) in cells]
            ys = np.arange(len(lv))
            for p, (lab, _, _, col) in enumerate(pairs):
                cc = tab[lab][mk][gk]
                for y, l in zip(ys, lv):
                    c = cc[l]
                    a_ = 0.35 if C.faded(c["n"]) else 1.0
                    ax.barh(y + off[p], c["delta"], height=0.32, color=col, alpha=a_, zorder=3)
                    if np.isfinite(c["ci95"]):
                        ax.plot([c["delta"] - c["ci95"], c["delta"] + c["ci95"]], [y + off[p]] * 2, color=C.INK,
                                lw=0.9, alpha=a_, zorder=4)
                al = tab[lab][mk]["all"]
                ax.axvline(al["delta"], color=col, lw=1.2, ls=(0, (4, 2)), zorder=2)
            ax.axvline(0, color=C.INK2, lw=0.9, zorder=2)
            ax.set_yticks(ys)
            ax.set_yticklabels([f"{l}  n={cells[l]['n']:,}" for l in lv], fontsize=8.2)
            for t, l in zip(ax.get_yticklabels(), lv):
                if C.faded(cells[l]["n"]):
                    t.set_color(C.MUTED)
            ax.set_ylim(len(lv) - 0.45, -0.55)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(glab, loc="left", fontsize=11)
            ax.set_xlabel(mlab, fontsize=9)
    hs = [Patch(facecolor=col) for _, _, _, col in pairs]
    labs = [f"{lab}  (전체 ΔminADE6 {tab[lab]['minade']['all']['delta']:+.3f} · ΔminFDE6 "
            f"{tab[lab]['minfde']['all']['delta']:+.3f} m · Δmiss {tab[lab]['miss']['all']['delta']:+.1f}%p = 점선)"
            for lab, _, _, _ in pairs]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(0.075, 1 - 0.62 / H), ncol=1, fontsize=9.5)
    fig.suptitle(title, x=0.01, y=1 - 0.1 / H, ha="left", fontsize=13.5)
    note(fig, foot)
    C.savefig(fig, path)


def sec_b(D, M):
    pairs = [("10 Hz: 30 코사인 − 15 상수", "h10_c15", "h10_c30", C.C_BLUE),
             ("2 Hz: 30 코사인 − 15 상수", "h2_c15", "h2_c30", C.C_ORANGE)]
    tab = delta_table(D, M, pairs)
    SUM["b_delta"] = tab
    fig_delta_grid(D, tab, pairs,
                   "b1 · 더 학습해서(15에폭 상수 → 30에폭 코사인) 무엇이 좋아졌나 — 조건별 평균 변화 (음수 = 좋아짐)",
                   "같은 시나리오 짝 차이의 평균 · 가는 선 = 95% 신뢰구간 · 점선 = 전체 평균 변화 · 흐린 칸 n < 50 · val 24,988 · "
                   "각 판 best 체크포인트 · 상황은 정답 기반(viz_v4_common 임계값) · 시드 1개.",
                   OUT / "b1_delta_sched_by_condition.png")

    # ---- b2 오차 출처·손실·선택·다양성 (전체)
    plt = C.setup_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(21, 10.2))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.08, wspace=0.24, hspace=0.42)
    nd2 = D.T["n_distinct"].to_numpy() >= 2
    fb = D.T["fallback"].to_numpy()
    val = {}
    for k in ORDER:
        m = M[k]
        wre = D.df[k]["win_route_err"].to_numpy()
        fde = m["minfde"]
        N = len(fde)
        val[k] = {"abs_ds_mean": float(m["abs_ds"].mean()), "abs_dd_mean": float(m["abs_dd"].mean()),
                  "abs_ds_median": float(np.median(m["abs_ds"])), "abs_dd_median": float(np.median(m["abs_dd"])),
                  "lon_dominant": float((m["abs_ds"] > m["abs_dd"]).mean()),
                  "gt_top1": float(np.nanmean(m["gt_top1"]) * 100), "win_route_ok": float(100 - np.nanmean(m["win_route_err"]) * 100),
                  "win_eq_top1": float(100 - m["win_ne_top1"].mean() * 100),
                  "l_sl1": float(m["l_sl1"].mean()), "l_ce": float(m["l_ce"].mean()),
                  "l_hinge": float(m["l_hinge"].mean()), "l_jit": float(m["l_jit"].mean()),
                  "spread_median": float(np.nanmedian(m["spread"])), "n_clusters_mean": float(np.nanmean(m["n_clusters"])),
                  "eff_br_mean": float(np.nanmean(m["eff_br"])),
                  "top1_ade": float(m["top1_ade"].mean()), "top1_fde": float(m["top1_fde"].mean()),
                  # minFDE6 = Σ 그룹 비중 × 그룹 평균 (그룹: 승자 경로 = 정답 / ≠ 정답 / 분기 1개·폴백)
                  "fde_part_ok": float(fde[nd2 & ~wre].sum() / N), "fde_part_err": float(fde[nd2 & wre].sum() / N),
                  "fde_part_nd1": float(fde[~nd2].sum() / N),
                  "n_err": int((nd2 & wre).sum()), "fde_mean_err": float(fde[nd2 & wre].mean()),
                  "fde_mean_ok": float(fde[nd2 & ~wre].mean()), "fde_mean_nd1": float(fde[~nd2].mean())}
    SUM["b_overall"] = val
    w = 0.19
    offs = {k: (j - 1.5) * w for j, k in enumerate(ORDER)}

    def grouped(ax, items, fmt="{:.2f}", scale=None):
        for x, (key, _) in enumerate(items):
            for k in ORDER:
                v = val[k][key] * (scale[x] if scale else 1.0)
                ax.bar(x + offs[k], v, width=w * 0.92, zorder=3, **bar_kw(k))
                # 5자 이상(예: 1.455)은 막대 폭보다 넓어 옆 숫자와 붙는다 -> 세로로 쓴다
                s = fmt.format(v)
                rot = len(s) >= 5
                ax.text(x + offs[k], v, (" " + s) if rot else s, ha="center", va="bottom", fontsize=7.2,
                        color=C.INK2, rotation=90 if rot else 0)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([t for _, t in items], fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    grouped(axes[0, 0], [("abs_ds_mean", "종방향 |Δs| 평균"), ("abs_dd_mean", "횡방향 |Δd| 평균"),
                         ("abs_ds_median", "|Δs| 중앙"), ("abs_dd_median", "|Δd| 중앙")])
    axes[0, 0].set_ylabel("[m]")
    axes[0, 0].set_title("① 승자 모드 끝점 오차의 종·횡 성분 (승자 경로 Frenet)", loc="left", fontsize=10.5)
    grouped(axes[0, 1], [("gt_top1", "정답 경로 = 경로 확률 1위\n(구별 분기 ≥ 2)"),
                         ("win_route_ok", "승자 경로 = 정답 경로\n(구별 분기 ≥ 2)"),
                         ("win_eq_top1", "승자 모드 = 확률 1위\n(전체)")], fmt="{:.1f}")
    axes[0, 1].set_ylabel("[%]")
    axes[0, 1].set_title("② 분기·모드 선택 (높을수록 좋음)", loc="left", fontsize=10.5)
    grouped(axes[0, 2], [("l_sl1", "smoothL1 (승자)"), ("l_ce", "CE"), ("l_hinge", "이탈 hinge ×1"),
                         ("l_jit", "흔들림 ×1 (×100 표시)")], fmt="{:.3f}", scale=[1, 1, 1, 100])
    axes[0, 2].set_title("③ 손실 항 — 학습식을 시나리오마다 적용한 평균", loc="left", fontsize=10.5)
    grouped(axes[1, 0], [("spread_median", "끝점 퍼짐 중앙 [m]\n(살아있는 모드 ≥ 2)"),
                         ("n_clusters_mean", "끝점 군집 수 평균\n(2.5 m 안 = 같은 군집)"),
                         ("eff_br_mean", "유효 분기 수 평균\n(구별 분기 ≥ 2)")])
    axes[1, 0].set_title("④ 모드 다양성", loc="left", fontsize=10.5)
    grouped(axes[1, 1], [("top1_ade", "확률 1위 모드 ADE"), ("top1_fde", "확률 1위 모드 FDE")])
    axes[1, 1].set_ylabel("[m]")
    axes[1, 1].set_title("⑤ 확률만 믿고 한 궤적을 고를 때의 오차", loc="left", fontsize=10.5)
    ax = axes[1, 2]
    parts = [("fde_part_nd1", "분기 1개·폴백", C.AXIS), ("fde_part_ok", "승자 경로 = 정답", C.C_AQUA),
             ("fde_part_err", "승자 경로 ≠ 정답", C.C_RED)]
    for x, k in enumerate(ORDER):
        bot = 0.0
        for key, lab, col in parts:
            v = val[k][key]
            ax.bar(x, v, bottom=bot, width=0.6, color=col, edgecolor=COL[k], linewidth=1.8, zorder=3,
                   label=lab if x == 0 else None)
            ax.text(x, bot + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=7.8,
                    color="white" if col == C.C_RED else C.INK)
            bot += v
        ax.text(x, bot, f"{bot:.3f}", ha="center", va="bottom", fontsize=8.2, color=C.INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([SHORT[k] for k in ORDER])
    ax.set_ylabel("minFDE6 기여 [m] = 그룹 비중 × 그룹 평균")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3, fontsize=8.5)
    ax.set_title("⑥ minFDE6 분해 — 막대 테두리 색 = 판", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    hs, ls = run_handles()
    fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.05, 0.945), ncol=4, fontsize=9.5)
    fig.suptitle("b2 · 오차 출처 · 손실 항 · 선택 · 다양성 — 네 판 (val 24,988, best 체크포인트)", x=0.01, y=0.99,
                 ha="left", fontsize=13.5)
    note(fig, "Δs·Δd = 승자 모드가 탄 후보 경로의 Frenet 좌표 끝점 차(예측 적분기 상태 − 정답 투영). 막대: 채움 = 30에폭 코사인, "
              "빗금 = 15에폭 상수. 경로 확률 = 같은 경로 슬롯 확률 합. hinge 는 버려진 모드만 평균.")
    C.savefig(fig, OUT / "b2_sources_loss_selection.png")

    # ---- b3 상황별 값
    cls_mets = [("abs_ds", "종방향 |Δs| 평균 [m]", np.nanmean, 1.0), ("abs_dd", "횡방향 |Δd| 평균 [m]", np.nanmean, 1.0),
                ("gt_top1", "정답 경로 = 경로 확률 1위 [%]\n(같은 경로 슬롯 합, 구별 분기 ≥ 2)", np.nanmean, 100.0),
                ("win_ne_top1", "승자 ≠ 확률 1위 [%]", np.nanmean, 100.0),
                ("l_ce", "CE 평균", np.nanmean, 1.0), ("spread", "끝점 퍼짐 중앙 [m]", np.nanmedian, 1.0)]
    SUM["b_by_class"] = fig_value_grid(D, M, cls_mets, [("cls", C.CLASSES, "상황")],
                                       "b3 · 상황별 오차 출처 · 선택 · 손실 · 다양성 — 네 판의 값",
                                       OUT / "b3_by_class_values.png")
    # ---- b4 묶음 조건 (필수 요건 5)
    T = D.T
    T["theta0_bin"] = C.bin_labels(T["theta0"], "dh")          # 같은 구간 (<5, 5–15, …)
    SUM["b_bundles"] = fig_value_grid(
        D, M,
        [("abs_ds", "종방향 |Δs| 평균 [m]", np.nanmean, 1.0), ("minfde", "minFDE6 평균 [m]", np.nanmean, 1.0),
         ("route_err", "분기 선택 오류 [%]\n(확률 1위 모드의 경로 ≠ 정답, 분기 ≥ 2)", np.nanmean, 100.0)],
        [("v0_bin", C.BINS["v0"][1], "v0 [m/s]"), ("amax_bin", C.BINS["amax"][1], "미래 최대 |a| [m/s²]"),
         ("dh_bin", C.BINS["dh"][1], "6초 |Δh| [°]"), ("theta0_bin", C.BINS["dh"][1], "|θ0| [°] (10Hz·30 슬롯 0)")],
        "b4 · 묶음 조건별 값 — (속력·가속·종방향 오차) × (방향변화·θ0·분기 선택)", OUT / "b4_bundles.png",
        by_rows=True)
    # ---- b5 분기 선택 오류의 성격 (옆 차로인가, 다른 방향인가)
    SUM["b_route_nature"] = fig_route_nature(D, OUT / "b5_route_error_nature.png")
    print("[b] 스케줄 비교", flush=True)
    return tab


SAME_LANE_M = 1.75     # 정답 경로 기준 끝 횡위치 차 < 차로 폭(3.5 m)의 절반이면 '정답 차로에 머묾'
PROJ_WORKERS = 12      # 공유 머신: 16 이하
_PJ = {}
PROJ_CHECK = []        # f 그림: 정답 경로 위 모드의 '투영 d' 와 '모델 d' 최대 차 (교차 검증)


def _proj_one(args):
    """예측 궤적(60,2)을 정답 기준 경로의 Frenet 좌표로 투영한 d(60,). 정답과 같게 과거 50스텝부터 이어서
    투영한다(lane_frame.to_frame 은 직전 점보다 앞에서만 대응점을 찾으므로 출발점이 같아야 한다)."""
    import lane_frame as lf
    i, g, trj = args
    rd = C.route_dict(_PJ["routes"][i, g], _PJ["tan"][i, g], _PJ["rlen"][i, g])
    _, d, _ = lf.to_frame(np.vstack([_PJ["pos"][i, :C.OBS].astype(np.float64), np.asarray(trj, np.float64)]), rd)
    return d[C.OBS:]


def proj_gt_route(D, rows, trajs):
    """rows 의 시나리오마다 trajs[k](60,2) 를 정답 기준 경로에 투영 -> (len(rows), 60) d."""
    from multiprocessing import get_context
    from dataset_cached import CachedV4Dataset
    if not _PJ:
        ds = CachedV4Dataset(C.VAL_CACHES["ah2"])
        _PJ.update(routes=np.asarray(ds.raw("routes")), tan=np.asarray(ds.raw("route_tan")),
                   rlen=np.asarray(ds.raw("route_len")), pos=D.R["pos"])
    g = D.T["gt_route"].to_numpy().astype(int)
    jobs = [(int(i), int(g[i]), t) for i, t in zip(rows, trajs)]
    if len(jobs) < 64:
        return np.array([_proj_one(j) for j in jobs]).reshape(len(jobs), -1)
    with get_context("fork").Pool(PROJ_WORKERS) as pool:
        out = pool.map(_proj_one, jobs, chunksize=64)
    return np.array(out)


def route_nature(D, k):
    """1위 경로가 정답 경로가 아닐 때, 고른 경로가 정답 궤적과 나란한 옆 차로인가(정답 대비 최대 |θ| < 30° 이고
    평균 |d| < 5 m) 다른 방향 분기인가. 구별 분기 ≥ 2 만."""
    ar = np.arange(D.N)
    df = D.df[k]
    tr = df["top1_route"].to_numpy()
    err = df["route_err"].to_numpy() & (D.T["n_distinct"].to_numpy() >= 2)
    md = D.R["route_meand"][ar, tr]
    mth = D.R["route_maxth"][ar, tr]
    par = err & (mth < C.ROUTE_ALIGN_DEG) & (md < 5.0)
    return err, par, err & ~par, md


def fig_route_nature(D, path):
    plt = C.setup_mpl()
    T = D.T
    nd2 = T["n_distinct"].to_numpy() >= 2
    vb = T["v0_bin"].to_numpy()
    levels = C.BINS["v0"][1]
    fig, axes2 = plt.subplots(2, 3, figsize=(21, 11.0))
    fig.subplots_adjust(left=0.045, right=0.99, top=0.855, bottom=0.085, wspace=0.24, hspace=0.62)
    axes = list(axes2.flat)          # flat 은 한 번 쓰면 끝나는 반복자라 판마다 zip 하려면 목록이어야 한다
    out = {"levels": levels, "n": [int((nd2 & (vb == l)).sum()) for l in levels], "same_lane_m": SAME_LANE_M}
    x = np.arange(len(levels))
    gt0 = (T["gt_route"].to_numpy() == 0)
    ar = np.arange(D.N)
    t0 = time.time()
    rows_nd2 = np.flatnonzero(nd2)
    lane_txt = []
    endlat = {}
    for k in ORDER:
        err, par, oth, md = route_nature(D, k)
        fde = D.df[k]["top1_fde"].to_numpy(np.float64)
        top = D.df[k]["top1"].to_numpy()
        top0 = D.df[k]["top1_route"].to_numpy() == 0
        # 구별 분기 ≥ 2 전체의 1위 궤적을 정답 경로에 투영 -> 끝 횡위치가 정답과 SAME_LANE_M 안이면 '정답 차로'
        dp_all = proj_gt_route(D, rows_nd2, D.P[k]["traj"][rows_nd2, top[rows_nd2]])
        lane_ok = np.zeros(D.N, bool)
        endlat[k] = np.abs(dp_all[:, -1] - D.R["gt_d_g"][rows_nd2, -1].astype(np.float64))
        lane_ok[rows_nd2] = endlat[k] < SAME_LANE_M
        # 옆 차로 슬롯을 고른 1위 모드: 정답 경로에 투영한 끝 횡위치가 정답과 얼마나 다른가 + 자기 경로 기준 끝 |d|
        rows = np.flatnonzero(par)
        pos_in = np.searchsorted(rows_nd2, rows)
        dp = dp_all[pos_in]
        stay_r = lane_ok[rows]
        stay = np.zeros(D.N, bool)
        stay[rows] = stay_r
        own_d = np.abs(D.P[k]["d"][ar, top, -1].astype(np.float64))
        r = {"par": [], "oth": [], "fde_gap": [], "top0": [], "sel_meand_err_median": [], "n_par": [],
             "par_stay": [], "par_own_d_median": [], "n_gap_min": [], "n_err": [], "n_ok": []}
        for l in levels:
            m = nd2 & (vb == l)
            r["par"].append(100 * par[m].mean())
            r["oth"].append(100 * oth[m].mean())
            r["fde_gap"].append(float(fde[m & err].mean() - fde[m & ~err].mean()))
            # ③ 은 두 집단(틀림 · 맞음)의 평균 차 -> 흐림은 두 집단 중 작은 쪽 n 으로 정한다
            r["n_err"].append(int((m & err).sum()))
            r["n_ok"].append(int((m & ~err).sum()))
            r["n_gap_min"].append(min(r["n_err"][-1], r["n_ok"][-1]))
            r["top0"].append(100 * top0[m].mean())
            r["sel_meand_err_median"].append(float(np.median(md[m & err])))
            mp = m & par
            r["n_par"].append(int(mp.sum()))
            r["par_stay"].append(100 * float(stay[mp].mean()) if mp.any() else np.nan)
            r["par_own_d_median"].append(float(np.median(own_d[mp])) if mp.any() else np.nan)
        m_all = nd2
        r["all"] = {"err": 100 * err[m_all].mean(), "par": 100 * par[m_all].mean(), "oth": 100 * oth[m_all].mean(),
                    "par_share_of_err": float(par[m_all].sum() / max(err[m_all].sum(), 1)),
                    "fde_err": float(fde[m_all & err].mean()), "fde_ok": float(fde[m_all & ~err].mean()),
                    "n_par": int(par.sum()), "par_stay": 100 * float(stay_r.mean()),
                    "par_stay_fde": float(fde[stay].mean()), "par_leave_fde": float(fde[par & ~stay].mean()),
                    "par_own_d_median": float(np.median(own_d[par])),
                    "par_stay_endlat_median": float(np.median(np.abs(dp[:, -1] - D.R["gt_d_g"][rows, -1]))),
                    # 경로 '슬롯' 기준과 '궤적이 정답 차로에 있나' 기준 (구별 분기 ≥ 2)
                    "route_ok": 100 * float((~err[m_all]).mean()),
                    "lane_ok": 100 * float(lane_ok[m_all].mean()),
                    "lane_ok_given_route_ok": 100 * float(lane_ok[m_all & ~err].mean()),
                    "lane_ok_given_route_err": 100 * float(lane_ok[m_all & err].mean()),
                    "lane_ok_by_v0": [100 * float(lane_ok[nd2 & (vb == l)].mean()) for l in levels]}
        lane_txt.append(f"{SHORT[k]} {r['all']['lane_ok']:.1f}% (슬롯 기준 {r['all']['route_ok']:.1f}%)")
        out[k] = r
        for ax, key in zip(axes, ("par", "oth", "fde_gap", "top0", "par_stay", "par_own_d_median")):
            ax.plot(x, r[key], color=COL[k], ls=LS[k], lw=2.0, zorder=3)
            fade = ([C.faded(n) for n in r["n_par"]] if key in ("par_stay", "par_own_d_median")
                    else [C.faded(n) for n in r["n_gap_min"]] if key == "fde_gap"
                    else [C.faded(n) for n in out["n"]])
            for xx, yy, f_ in zip(x, r[key], fade):
                ax.plot(xx, yy, marker=MK[k], ms=6.5, mfc=COL[k] if FILL[k] else "white", mec=COL[k], mew=1.4,
                        alpha=0.35 if f_ else 1.0, zorder=4, ls="none")
    print(f"[b5] 1위 궤적 정답 경로 투영 {len(rows_nd2) * len(ORDER):,}개 · {time.time() - t0:.0f}s", flush=True)
    # 차로 기준의 민감도: 임계값 · 6초 이동 부분집합 (짝 차이는 %p, 95% 반폭)
    mv = D.T["move6"].to_numpy(np.float64)[rows_nd2]
    every = np.ones(len(rows_nd2), bool)
    sens = []
    for lab, thr, msk in (("1.0 m", 1.0, every), ("1.75 m (기본)", SAME_LANE_M, every), ("2.5 m", 2.5, every),
                          ("1.75 m · 6초 이동 ≥ 20 m", SAME_LANE_M, mv >= 20.0),
                          ("1.75 m · 6초 이동 < 5 m", SAME_LANE_M, mv < 5.0)):
        ok_ = {k: (endlat[k][msk] < thr).astype(np.float64) for k in ORDER}
        row = {"label": lab, "thr_m": thr, "n": int(msk.sum())} | {k: 100 * float(ok_[k].mean()) for k in ORDER}
        for nm_, a_, b_ in (("10Hz 30-15", "h10_c15", "h10_c30"), ("2Hz 30-15", "h2_c15", "h2_c30"),
                            ("30 2Hz-10Hz", "h10_c30", "h2_c30"), ("15 2Hz-10Hz", "h10_c15", "h2_c15")):
            m_, h_, _ = paired(ok_[a_], ok_[b_])
            row[nm_] = [100 * m_, 100 * h_]
        sens.append(row)
    out["lane_sensitivity"] = sens
    out["gt_route0"] = [100 * gt0[nd2 & (vb == l)].mean() for l in levels]
    axes[3].plot(x, out["gt_route0"], color=C.C_GT, lw=2.0, marker="D", ms=6, zorder=4, label="정답 경로 = 경로 0")
    axes[3].legend(loc="lower left", fontsize=9)
    ttl = [("① 나란한 옆 차로를 1위로 고른 비율 [%]", "정답 대비 최대 |θ| < 30° · 평균 |d| < 5 m 인 경로"),
           ("② 다른 방향 분기를 1위로 고른 비율 [%]", "①이 아닌 분기 선택 오류"),
           ("③ 경로를 틀렸을 때 1위 FDE 손해 [m]", "1위 FDE 평균 (틀림 − 맞음)"),
           ("④ 1위 경로가 경로 0 인 비율 [%]", "경로 0 = 관측 과거와 가장 잘 맞는 후보 (데이터셋 정렬)"),
           ("⑤ ①인데 1위 궤적은 정답 차로에 머문 비율 [%]",
            f"1위 궤적을 정답 경로에 투영한 6초 끝 횡위치가 정답과 {SAME_LANE_M:g} m 안"),
           ("⑥ ①인 1위 모드의 '자기 경로 기준' 끝 |d| 중앙 [m]",
            "≈ 차로 폭이면 옆 차로 경로를 골라 놓고 d 로 되돌아온 것")]
    for q, (ax, (t1, t2)) in enumerate(zip(axes, ttl)):
        ax.set_xticks(x)
        if q == 2:   # ③ 두 집단 n (10Hz·30): 틀림 / 맞음
            ax.set_xticklabels([f"{l}\n틀림{a:,}/맞음{b:,}" for l, a, b in
                                zip(levels, out[REF]["n_err"], out[REF]["n_ok"])], fontsize=7.4)
        elif q < 4:
            ax.set_xticklabels([f"{l}\nn={n:,}" for l, n in zip(levels, out["n"])], fontsize=8.2)
        else:   # ⑤⑥ 의 모집단 = ① (판마다 다르다) -> 10Hz·30 의 n 을 적고 흐림은 판마다
            ax.set_xticklabels([f"{l}\n①n={n:,} (10Hz·30)" for l, n in zip(levels, out[REF]["n_par"])], fontsize=7.6)
        ax.set_xlabel("v0 [m/s]")
        ax.set_title(f"{t1}\n{t2}", loc="left", fontsize=10)
    axes[2].axhline(0, color=C.INK2, lw=0.8)
    axes[5].axhline(3.5, color=C.MUTED, lw=1.0, ls=(0, (4, 2)))
    axes[5].set_ylim(0, 4.2)
    axes[5].text(0.02, 3.62, "차로 폭 3.5 m", transform=axes[5].get_yaxis_transform(), va="bottom", ha="left",
                 fontsize=8, color=C.INK2, bbox=dict(fc=C.SURF, ec="none", pad=0.5))
    hs, ls = run_handles()
    fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.045, 0.955), ncol=4, fontsize=9.5)
    fig.suptitle("b5 · 분기 선택 오류의 성격 — 속력별 (구별 분기 ≥ 2, 분기 선택 오류 = 확률 1위 모드의 경로 ≠ 정답 기준 경로)",
                 x=0.01, y=0.99, ha="left", fontsize=13.5)
    fig.text(0.4, 0.958, f"1위 궤적이 정답 차로에 있는 비율(끝 횡위치 {SAME_LANE_M:g} m 안, n={int(nd2.sum()):,}): "
             + " · ".join(lane_txt), fontsize=9.2, color=C.INK2)
    note(fig, "선택 경로와 정답 사이 거리는 viz_v4_dump 의 경로별 정답 평균 |d|·최대 |θ| (모델과 무관). 흐린 점 n < 50"
              "(③ 은 틀림·맞음 중 작은 쪽, ⑤⑥ 은 그 판의 ① 개수, x 라벨 n 은 10Hz·30). "
              "투영 = lane_frame.to_frame(과거 50 + 예측 60) · 정답 d 도 같은 함수(viz_v4_dump). val 24,988 · best 체크포인트 · 시드 1개.")
    C.savefig(fig, path)
    return out


def fig_value_grid(D, M, mets, groupings, title, path, by_rows=False):
    """칸마다 (조건 수준 × 네 판) 점. mets × groupings 격자. 반환: 숫자 표."""
    plt = C.setup_mpl()
    nr, nc = (len(mets), len(groupings))
    if not by_rows and len(groupings) == 1:
        nr, nc = 1, len(mets)
    W = 4.0 * nc + 1.6
    H = 3.9 * nr + 2.3 if by_rows else 6.2
    fig, axes = plt.subplots(nr, nc, figsize=(W, H), squeeze=False, sharex="row" if by_rows else False)
    fig.subplots_adjust(left=0.09 if not by_rows else 0.075, right=0.99, top=1 - 1.35 / H,
                        bottom=(1.25 if by_rows else 0.62) / H,
                        wspace=0.35 if by_rows else 0.12, hspace=0.35)
    tab = {}
    first_ns = {}
    cells = [(i, j) for i in range(nr) for j in range(nc)]
    for (i, j) in cells:
        mi, gi = (i, j) if by_rows else (j, 0)
        mk, mlab, fn, sc = mets[mi]
        gk, levels, glab = groupings[gi]
        ax = axes[i, j]
        g = D.T[gk].to_numpy()
        lv = [l for l in levels if (g == l).any()]
        ys = np.arange(len(lv))
        ns = []
        for y, l in zip(ys, lv):
            m = g == l
            vals_n = int(np.isfinite(M[REF][mk][m]).sum())
            ns.append(vals_n)
            for q, k in enumerate(ORDER):
                v = fn(M[k][mk][m]) * sc if vals_n else np.nan
                tab.setdefault(mk, {}).setdefault(gk, {}).setdefault(str(l), {"n": vals_n})[k] = float(v)
                a_ = 0.35 if C.faded(vals_n) else 1.0
                ax.plot(v, y + (q - 1.5) * 0.17, ls="none", marker=MK[k], ms=6.5, mfc=COL[k] if FILL[k] else "white",
                        mec=COL[k], mew=1.5, alpha=a_, zorder=3)
            # 입력별로 15 → 30 을 잇는다
            for inp in ("h10", "h2"):
                a0, a1 = [fn(M[f"{inp}_{s}"][mk][m]) * sc if vals_n else np.nan for s in ("c15", "c30")]
                q0, q1 = ORDER.index(f"{inp}_c15"), ORDER.index(f"{inp}_c30")
                ax.annotate("", xy=(a1, y + (q1 - 1.5) * 0.17), xytext=(a0, y + (q0 - 1.5) * 0.17),
                            arrowprops=dict(arrowstyle="-", color=COL[f"{inp}_c30"], lw=0.8, alpha=0.6), zorder=2)
        allv = fn(M[REF][mk]) * sc
        ax.set_yticks(ys)
        show_lab = by_rows or j == 0
        if show_lab:
            first_ns[(i, gi)] = ns
        elif ns != first_ns.get((0, gi), ns):
            # 이 칸의 모집단이 행 라벨(첫 칸)의 n 과 다르다 (예: 구별 분기 ≥ 2 만) -> 칸 안 오른쪽에 그 n 을 적는다
            for y, n in zip(ys, ns):
                ax.text(0.995, y - 0.40, f"n={n:,}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=6.8, color=C.MUTED if C.faded(n) else C.INK2, zorder=5,
                        bbox=dict(fc=C.SURF, ec="none", pad=0.4, alpha=0.85))
        ax.set_yticklabels([f"{l}  n={n:,}" for l, n in zip(lv, ns)] if show_lab else [""] * len(lv), fontsize=8.3)
        if show_lab:
            for t, n in zip(ax.get_yticklabels(), ns):
                if C.faded(n):
                    t.set_color(C.MUTED)
        ax.set_ylim(len(lv) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        for q in ax.get_yticks()[:-1]:
            ax.axhline(q + 0.5, color=C.GRID, lw=0.6, zorder=1)
        if by_rows:
            if i == 0:
                ax.set_title(glab, loc="left", fontsize=11)
            ax.set_xlabel(mlab, fontsize=9)
        else:
            ax.set_title(mlab, loc="left", fontsize=10)
            ax.axvline(allv, color=C.INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    hs, ls = run_handles()
    fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.075, 1 - 0.55 / H), ncol=4, fontsize=9.5)
    fig.suptitle(title, x=0.01, y=1 - 0.1 / H, ha="left", fontsize=13.5)
    note(fig, "점 = 조건 수준 안의 평균(퍼짐은 중앙값) · 가는 선 = 같은 입력의 15 → 30 · 흐린 줄 n < 50 · val 24,988 · 시드 1개"
              + ("" if by_rows else " · 점선 = 10Hz·30 전체값 · 행 라벨 n = 상황 전체, 칸 안 n = 그 칸의 모집단"
                                    "(정답 경로 칸은 구별 분기 ≥ 2, 퍼짐 칸은 살아있는 모드 ≥ 2)"))
    C.savefig(fig, path)
    return tab


# =========================================================================== c 10 Hz vs 2 Hz
TREE_FEATS = [("v0", "v0 [m/s]"), ("absdh", "|Δh| [°]"), ("absa", "최대|a| [m/s²]"), ("n_distinct", "분기 수"),
              ("n_lanes30", "30m 차로 수"), ("n_reachable", "도달 차로 수"), ("fallback", "폴백"),
              ("has_lead49", "앞차 있음"), ("thw_f", "시간간격 [s]"), ("theta0", "|θ0| [°] (10Hz·30 출력)"),
              ("abs_a_last", "관측 끝 |a| [m/s²]"), ("abs_yaw_last", "관측 끝 |yaw| [°/s]"),
              ("ramp_acc", "시작 램프 가속 [m/s²]"), ("move6", "6초 이동 [m]")]


def ramp_div(v, lo, hi):
    """Δ = 2 Hz − 10 Hz: 음수(2 Hz 우세) 주황, 양수(10 Hz 우세) 파랑, 0 근처 옅게."""
    ORANGE = ["#fdeee6", "#fbd5c1", "#f7b596", "#f2916a", "#eb6834", "#c24f22"]
    BLUE = ["#eef4fc", C.BLUE_RAMP[0], C.BLUE_RAMP[3], C.BLUE_RAMP[6], C.BLUE_RAMP[8], C.BLUE_RAMP[10]]
    if v < 0:
        q = min(1.0, v / lo) if lo < 0 else 0.0
        c = ORANGE[int(round(q * (len(ORANGE) - 1)))]
        return c, ("white" if q > 0.75 else C.INK)
    q = min(1.0, v / hi) if hi > 0 else 0.0
    c = BLUE[int(round(q * (len(BLUE) - 1)))]
    return c, ("white" if q > 0.75 else C.INK)


def sec_c(D, M):
    plt = C.setup_mpl()
    T = D.T
    a10, a2 = M["h10_c30"], M["h2_c30"]
    dF = a2["minfde"] - a10["minfde"]
    dA = a2["minade"] - a10["minade"]
    T["dF"] = dF
    T["dA"] = dA
    T["dF15"] = M["h2_c15"]["minfde"] - M["h10_c15"]["minfde"]
    q = np.percentile(dF, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    better2, worse2 = (dF < -TIE_M), (dF > TIE_M)
    SUM["c_dist"] = {"n": len(dF), "mean": float(dF.mean()), "median": float(np.median(dF)),
                     "pct": dict(zip(["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"], q.tolist())),
                     "share_2hz_better": float(better2.mean()), "share_10hz_better": float(worse2.mean()),
                     "share_tie": float((~better2 & ~worse2).mean()), "tie_m": TIE_M,
                     "sum_pos": float(dF[dF > 0].sum() / len(dF)), "sum_neg": float(dF[dF < 0].sum() / len(dF)),
                     "tail_share_abs_gt5": float((np.abs(dF) > 5).mean()),
                     "tail_contrib_abs_gt5": float(dF[np.abs(dF) > 5].sum() / dF.sum()),
                     "mean_dA": float(dA.mean()), "corr_dF_dF15": float(np.corrcoef(dF, T["dF15"])[0, 1])}
    # ---- c1 분포
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), gridspec_kw=dict(width_ratios=[1.15, 1.1, 1.0]))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.83, bottom=0.13, wspace=0.3)
    ax = axes[0]
    edges = np.concatenate([-np.logspace(np.log10(40), -2, 26), [0], np.logspace(-2, np.log10(40), 26)])
    cnt, _ = np.histogram(np.clip(dF, -39.9, 39.9), edges)
    xs = np.arange(len(cnt))
    cols = [C.C_ORANGE if (edges[b] + edges[b + 1]) / 2 < -TIE_M else
            (C.C_BLUE if (edges[b] + edges[b + 1]) / 2 > TIE_M else C.AXIS) for b in range(len(cnt))]
    ax.bar(xs, cnt, width=0.9, color=cols, zorder=3)
    tick_v = [-20, -5, -1, -0.1, 0, 0.1, 1, 5, 20]
    tick_x = [np.interp(v, edges, np.arange(len(edges))) - 0.5 for v in tick_v]
    ax.set_xticks(tick_x)
    ax.set_xticklabels([f"{v:g}" for v in tick_v])
    ax.set_yscale("log")
    ax.set_ylim(0.7, cnt.max() * 40)          # 위쪽 글상자 자리 (막대를 가리지 않게)
    ax.set_xlabel("ΔminFDE6 = 2 Hz − 10 Hz [m] (대칭 로그 구간, ±40 m 에서 자름)")
    ax.set_ylabel("로그 폭 구간마다 센 시나리오 수 (log)\n구간 폭이 바깥으로 갈수록 넓다 — 밀도가 아니다")
    ax.set_title(f"(a) 분포 — 2 Hz 우세(< −{TIE_M:g} m) {better2.mean() * 100:.1f}% · 비김 "
                 f"{(~better2 & ~worse2).mean() * 100:.1f}% · 10 Hz 우세 {worse2.mean() * 100:.1f}%", loc="left", fontsize=10)
    ax.text(0.02, 0.97, f"평균 {dF.mean():+.3f} m · 중앙 {np.median(dF):+.3f} m\n"
                        f"p5 {q[1]:+.2f} · p25 {q[3]:+.2f} · p75 {q[5]:+.2f} · p95 {q[7]:+.2f} m\n"
                        f"|Δ| > 5 m 인 {SUM['c_dist']['tail_share_abs_gt5'] * 100:.1f}% 가 평균 차의 "
                        f"{SUM['c_dist']['tail_contrib_abs_gt5'] * 100:.0f}% 를 만든다",
            transform=ax.transAxes, va="top", fontsize=8.8, color=C.INK2,
            bbox=dict(fc="white", ec=C.GRID, pad=4))
    ax.grid(axis="x", visible=False)
    # (b) 상황별 우세 비율
    ax = axes[1]
    cls = T["cls"].to_numpy()
    rows = []
    for c in C.CLASSES:
        m = cls == c
        rows.append((c, int(m.sum()), better2[m].mean() * 100, (~better2 & ~worse2)[m].mean() * 100,
                     worse2[m].mean() * 100, dF[m].mean()))
    for y, (c, n, b, t, w_, mu) in enumerate(rows):
        ax.barh(y, b, color=C.C_ORANGE, height=0.62, zorder=3)
        ax.barh(y, t, left=b, color=C.GRID, height=0.62, zorder=3)
        ax.barh(y, w_, left=b + t, color=C.C_BLUE, height=0.62, zorder=3)
        ax.text(b / 2, y, f"{b:.0f}", ha="center", va="center", fontsize=7.8, color="white")
        ax.text(b + t + w_ / 2, y, f"{w_:.0f}", ha="center", va="center", fontsize=7.8, color="white")
        ax.text(101, y, f"평균 {mu:+.2f} m", va="center", fontsize=8, color=C.INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c}  n={n:,}" for c, n, *_ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 122)
    ax.set_xlabel("[%]")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"(b) 상황별 — 주황 2 Hz 우세 · 회색 비김(|Δ| < {TIE_M:g} m) · 파랑 10 Hz 우세", loc="left", fontsize=10)
    SUM["c_dist"]["by_class"] = {r[0]: {"n": r[1], "better2": r[2], "tie": r[3], "better10": r[4], "mean": r[5]}
                                 for r in rows}
    # (c) 두 판 minFDE6 산점
    ax = axes[2]
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("b", C.BLUE_RAMP)
    x_, y_ = np.log10(np.clip(a10["minfde"], 0.05, 60)), np.log10(np.clip(a2["minfde"], 0.05, 60))
    hb = ax.hexbin(x_, y_, gridsize=55, cmap=cm, bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax, label="시나리오 수 (log)")
    ax.plot([-1.3, 1.8], [-1.3, 1.8], color=C.INK2, lw=1, ls=(0, (4, 2)))
    tk = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(np.log10(tk)); ax.set_xticklabels([f"{t:g}" for t in tk])
    ax.set_yticks(np.log10(tk)); ax.set_yticklabels([f"{t:g}" for t in tk])
    ax.set_xlabel("10 Hz · 30 minFDE6 [m] (log)")
    ax.set_ylabel("2 Hz · 30 minFDE6 [m] (log)")
    r_ = float(np.corrcoef(x_, y_)[0, 1])
    ax.set_title(f"(c) 시나리오별 두 판 — 로그 상관 {r_:.2f} · 점선 = 같음", loc="left", fontsize=10)
    SUM["c_dist"]["log_corr"] = r_
    fig.suptitle("c1 · 10 Hz vs 2 Hz (둘 다 30에폭 코사인) — 시나리오별 ΔminFDE6 = 2 Hz − 10 Hz", x=0.01, y=0.975,
                 ha="left", fontsize=13.5)
    note(fig, f"val 24,988 · best 체크포인트(10 Hz 에폭 29, 2 Hz 에폭 25) · 음수 = 2 Hz 가 더 가깝다. "
              f"15에폭 판끼리의 같은 차와의 시나리오 상관 {SUM['c_dist']['corr_dF_dF15']:.2f}. 시드 1개라 판 사이 차에는 학습 확률성이 섞인다.")
    C.savefig(fig, OUT / "c1_delta_distribution.png")

    # ---- c2 조건별
    pairs = [("30 코사인: 2 Hz − 10 Hz", "h10_c30", "h2_c30", C.C_VIOLET),
             ("15 상수: 2 Hz − 10 Hz", "h10_c15", "h2_c15", C.MUTED)]
    tab = delta_table(D, M, pairs)
    SUM["c_delta"] = tab
    fig_delta_grid(D, tab, pairs,
                   "c2 · 10 Hz vs 2 Hz 격차가 어디서 나오나 — 조건별 평균 차 (양수 = 2 Hz 가 나쁨)",
                   "같은 시나리오 짝 차이의 평균 · 가는 선 = 95% 신뢰구간 · 점선 = 전체 평균 차 · 흐린 칸 n < 50 · val 24,988 · "
                   "best 체크포인트 · 회색 = 같은 비교를 15에폭 상수 판끼리 (패턴이 재현되는지 보려고) · 시드 1개.",
                   OUT / "c2_delta_input_by_condition.png")
    # ---- c3 결정 트리
    tree_c(D, dF)
    # ---- c4 램프·최근 동역학
    fig_c4(D, dF, T["dF15"].to_numpy())
    # ---- c5 재현성 (다른 학습 쌍) · 가설 H3 (10 Hz focal heading 뒤집힘, 노션 2.11절)
    fig_c5(D, M, dF, T["dF15"].to_numpy())
    print("[c] 10 Hz vs 2 Hz", flush=True)


HEADING_SCAN = C.VIZ_ROOT / "heading_prep" / "data" / "scan.npz"   # viz_v4_heading_prep.py 결과 (fflip: 판정 재구현)


def fig_c5(D, M, dF, dF15):
    """격차의 어느 부분이 학습 쌍을 바꿔도 남는가 + 10 Hz 입력의 heading 뒤집힘 결함이 격차를 설명하나."""
    plt = C.setup_mpl()
    T = D.T
    fig, axes = plt.subplots(1, 4, figsize=(23, 6.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.78, bottom=0.2, wspace=0.3)
    res = {}
    # (a) 트리 잎
    ax = axes[0]
    lv = SUM["c_tree"]["leaves"]
    xs = np.array([r["mean"] for r in lv])
    ys = np.array([r["mean_15const"] for r in lv])
    ns = np.array([r["n"] for r in lv])
    same = np.array([r["same_sign_15const"] for r in lv])
    ax.scatter(xs, ys, s=18 + ns / 60, c=[C.C_VIOLET if s_ else C.MUTED for s_ in same], alpha=0.85, zorder=3,
               edgecolors="white", linewidths=0.8)
    lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=C.INK2, lw=0.9, ls=(0, (4, 2)))
    ax.axhline(0, color=C.AXIS, lw=0.8); ax.axvline(0, color=C.AXIS, lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    # 원점 근처(|x|,|y| < 0.45)는 번호만 겹치지 않게 놓고, 값은 글상자로
    far = [q for q in range(len(lv)) if max(abs(xs[q]), abs(ys[q])) >= 0.45]
    near = [q for q in range(len(lv)) if q not in far]
    C.place_labels(ax, [(xs[q], ys[q]) for q in far], [f"잎{lv[q]['leaf']} n={lv[q]['n']:,}" for q in far])
    C.place_labels(ax, [(xs[q], ys[q]) for q in near], [f"{lv[q]['leaf']}" for q in near], fontsize=7.0)
    if near:
        ax.text(0.02, 0.02, "원점 근처 잎 — 번호: n · 30쌍 / 15쌍 평균 Δ [m]\n" + "\n".join(
            f"잎{lv[q]['leaf']}: n={lv[q]['n']:,} · {lv[q]['mean']:+.2f} / {lv[q]['mean_15const']:+.2f}" for q in near),
            transform=ax.transAxes, fontsize=7.0, color=C.INK2, va="bottom", ha="left", linespacing=1.35,
            bbox=dict(fc=C.SURF, ec=C.GRID, pad=2.5, alpha=0.92), zorder=6)
    ax.set_xlabel("잎 평균 Δ — 30에폭 코사인 두 판 [m]")
    ax.set_ylabel("같은 잎의 평균 Δ — 15에폭 상수 두 판 [m]")
    w_same = float(ns[same].sum() / ns.sum())
    ax.set_title(f"(a) c3 트리 잎 — 보라 = 부호 재현 {int(same.sum())}/{len(lv)}개 (시나리오 {w_same * 100:.0f}%)",
                 loc="left", fontsize=10)
    res["leaves_same_sign"] = int(same.sum())
    res["leaves_same_sign_scen_share"] = w_same
    res["leaf_corr_weighted"] = float(np.cov(xs, ys, aweights=ns)[0, 1]
                                      / np.sqrt(np.cov(xs, aweights=ns)[()] * np.cov(ys, aweights=ns)[()]))
    # (b) 조건 칸 (c2 의 표)
    ax = axes[1]
    tab = SUM["c_delta"]
    a30, a15 = tab["30 코사인: 2 Hz − 10 Hz"]["minfde"], tab["15 상수: 2 Hz − 10 Hz"]["minfde"]
    marks = {"cls": "o", "v0_bin": "s", "nd_lab": "^", "reach_bin": "D", "lane30_bin": "v"}
    # 판 색(파랑·주황)과 섞이지 않게 조건 종류는 무채색 계열 + 모양으로 가른다
    gcol = {"cls": C.INK, "v0_bin": "#6f6c64", "nd_lab": C.C_AQUA, "reach_bin": "#b08a2e", "lane30_bin": C.C_MAGENTA}
    per = {}
    for gk, _, glab in GROUPINGS:
        cx_, cy_, cn_ = [], [], []
        for l, c in a30[gk].items():
            if c["n"] < C.MIN_N:
                continue
            cx_.append(c["delta"]); cy_.append(a15[gk][l]["delta"]); cn_.append(c["n"])
        cx_, cy_, cn_ = map(np.asarray, (cx_, cy_, cn_))
        ax.scatter(cx_, cy_, s=12 + cn_ / 80, marker=marks[gk], color=gcol[gk], alpha=0.8, zorder=3, label=glab,
                   edgecolors="white", linewidths=0.6)
        per[gk] = {"n_cells": int(len(cx_)), "same_sign": int((np.sign(cx_) == np.sign(cy_)).sum()),
                   "corr": float(np.corrcoef(cx_, cy_)[0, 1]) if len(cx_) > 2 else None}
    res["cells"] = per
    lim = 1.1
    ax.plot([-lim, lim], [-lim, lim], color=C.INK2, lw=0.9, ls=(0, (4, 2)))
    ax.axhline(0, color=C.AXIS, lw=0.8); ax.axvline(0, color=C.AXIS, lw=0.8)
    ax.axvline(a30["all"]["delta"], color=C.C_VIOLET, lw=1.0, ls=(0, (2, 2)))
    ax.axhline(a15["all"]["delta"], color=C.MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.set_xlim(-0.6, lim); ax.set_ylim(-0.95, lim)
    ax.set_xlabel("조건 칸 평균 ΔminFDE6 — 30 코사인 [m]")
    ax.set_ylabel("같은 칸 — 15 상수 [m]")
    ax.legend(loc="lower right", fontsize=7.6, title="c2 의 조건 (n ≥ 50 칸)", title_fontsize=7.8)
    ax.set_title(f"(b) c2 조건 칸 — 전체 평균(점선) 30 {a30['all']['delta']:+.3f} · 15 {a15['all']['delta']:+.3f} m",
                 loc="left", fontsize=10)
    # (c) 시나리오 하나하나
    ax = axes[2]
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("b", C.BLUE_RAMP)
    cl = 12
    hb = ax.hexbin(np.clip(dF, -cl, cl), np.clip(dF15, -cl, cl), gridsize=48, cmap=cm, bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax, label="시나리오 수 (log)")
    ax.axhline(0, color=C.AXIS, lw=0.8); ax.axvline(0, color=C.AXIS, lw=0.8)
    ax.set_xlabel(f"시나리오 ΔminFDE6 — 30 코사인 [m] (±{cl} m 자름)")
    ax.set_ylabel("같은 시나리오 — 15 상수 [m]")
    agree = float(((dF > TIE_M) & (dF15 > TIE_M)).mean() + ((dF < -TIE_M) & (dF15 < -TIE_M)).mean())
    opp = float(((dF > TIE_M) & (dF15 < -TIE_M)).mean() + ((dF < -TIE_M) & (dF15 > TIE_M)).mean())
    res["scen_corr"] = float(np.corrcoef(dF, dF15)[0, 1])
    res["scen_same_winner_share"] = agree
    res["scen_opposite_winner_share"] = opp
    ax.set_title(f"(c) 시나리오별 — 상관 {res['scen_corr']:.2f} · 같은 쪽 우세 {agree * 100:.0f}% · 반대 {opp * 100:.0f}%",
                 loc="left", fontsize=10)
    # (d) H3: 10 Hz 입력 focal heading 뒤집힘
    ax = axes[3]
    h3 = {"available": HEADING_SCAN.exists()}
    if h3["available"]:
        z = np.load(HEADING_SCAN)
        ff = z["fflip"]
        sids = json.loads((C.VAL_CACHES["ah2"] / "scenario_id.json").read_text())
        h3["order_same"] = bool(len(sids) == D.N and list(T["sid"]) == sids[:D.N])
        if not h3["order_same"]:
            raise SystemExit("heading 스캔 순서가 덤프와 다르다")
        f10, f2 = ff[:D.N, 0] == 1, ff[:D.N, 2] == 1
        groups = [("10 Hz 입력만 뒤집힘", f10 & ~f2), ("두 입력 모두 뒤집힘", f10 & f2),
                  ("2 Hz 입력만 뒤집힘", ~f10 & f2), ("뒤집힘 없음", ~f10 & ~f2)]
        wbar = 0.19
        for gi, (glab, m) in enumerate(groups):
            n_ = int(m.sum())
            row = {"n": n_}
            for q, k in enumerate(ORDER):
                v = float(M[k]["minfde"][m].mean()) if n_ else np.nan
                row[k] = v
                kw = bar_kw(k)
                ax.bar(gi + (q - 1.5) * wbar, v, width=wbar * 0.92, zorder=3, alpha=0.35 if C.faded(n_) else 1.0, **kw)
                ax.text(gi + (q - 1.5) * wbar, v, f" {v:.1f}", rotation=90, ha="center", va="bottom", fontsize=7.2,
                        color=C.INK2)
            row["d30"] = float(dF[m].mean()) if n_ else np.nan
            row["d15"] = float(dF15[m].mean()) if n_ else np.nan
            row["contrib_d30"] = float(dF[m].sum() / D.N)
            h3[glab] = row
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\nn={h3[g]['n']:,}\nΔ30 {h3[g]['d30']:+.2f} · Δ15 {h3[g]['d15']:+.2f}"
                            for g, _ in groups], fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("minFDE6 평균 [m]")
        g1 = h3["10 Hz 입력만 뒤집힘"]
        ax.set_title(f"(d) H3 heading 뒤집힘(노션 2.11)\n10 Hz 입력만 뒤집힌 {g1['n']}개가 30 쌍 평균 차(+{dF.mean():.3f} m)에 "
                     f"보탠 몫 {g1['contrib_d30']:+.4f} m", loc="left", fontsize=10)
        hs_, ls_ = run_handles()
        ax.legend(hs_, ls_, loc="upper right", fontsize=7.6)
    SUM["c_replicate"] = res
    SUM["c_h3_flip"] = h3
    fig.suptitle("c5 · 격차는 학습 쌍을 바꿔도 남나 — 30에폭 코사인 쌍 vs 15에폭 상수 쌍 (Δ = 2 Hz − 10 Hz minFDE6) · 가설 H3",
                 x=0.01, y=0.975, ha="left", fontsize=13.5)
    pc = res["cells"]
    fig.text(0.01, 0.9, "시드 1개짜리 판 네 개라, 두 입력의 차를 '다른 학습 쌍' 에서 한 번 더 재는 것이 유일한 재현 점검이다. "
                        f"평균 차(30쌍 {dF.mean():+.3f} / 15쌍 {dF15.mean():+.3f} m)는 같은 방향이지만, 조건 칸의 부호 일치는 "
                        + ", ".join(f"{glab} {pc[gk]['same_sign']}/{pc[gk]['n_cells']}" for gk, _, glab in GROUPINGS)
                        + f", 시나리오별 상관은 {res['scen_corr']:.2f} 다.",
             fontsize=9.6, color=C.INK2)
    note(fig, "(d) 뒤집힘 = build_heading 의 180° 교정(align_ref) 판정 재구현 (viz_v4_heading_prep scan.npz fflip, 판정 −1 은 '뒤집힘 없음'). "
              "흐린 막대 n < 50. val 24,988 · best 체크포인트.", fs=8.2)
    C.savefig(fig, OUT / "c5_replication_h3.png")
    print(f"[c5] 잎 부호 재현 {res['leaves_same_sign']}/{len(lv)} · 시나리오 상관 {res['scen_corr']:.3f}", flush=True)


def tree_features(D):
    T = D.T
    X = {}
    for c, _ in TREE_FEATS:
        if c == "thw_f":
            X[c] = np.where(np.isfinite(T["thw49"]), T["thw49"], 99.0)
        elif c == "ramp_acc":
            X[c] = np.where(np.isfinite(T["ramp_acc"]), T["ramp_acc"], -99.0)
        else:
            X[c] = T[c].to_numpy(np.float64)
    for c in C.CLASSES:
        X[f"cls_{c}"] = (T["cls"].to_numpy() == c).astype(float)
    names = list(X)
    labels = {c: l for c, l in TREE_FEATS} | {f"cls_{c}": f"상황={c}" for c in C.CLASSES}
    return np.column_stack([X[n] for n in names]), names, labels


def tree_c(D, dF):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score
    plt = C.setup_mpl()
    X, names, labels = tree_features(D)
    lo, hi = np.percentile(dF, [1, 99])
    y = np.clip(dF, lo, hi)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=TREE_SEED)
    est = DecisionTreeRegressor(max_depth=TREE_DEPTH, min_samples_leaf=TREE_LEAF, random_state=TREE_SEED)
    est.fit(X[tr], y[tr])
    r2_tr, r2_te = r2_score(y[tr], est.predict(X[tr])), r2_score(y[te], est.predict(X[te]))
    r2_te_raw = r2_score(dF[te], est.predict(X[te]))
    # 다른 학습 쌍(15에폭 상수 두 판)의 같은 차에 이 트리를 그대로 대면 — 조건이 격차를 '계통적으로' 가르는가
    dF15 = D.T["dF15"].to_numpy()
    lo15, hi15 = np.percentile(dF15, [1, 99])
    r2_rep = r2_score(np.clip(dF15, lo15, hi15), est.predict(X))
    is_te = np.zeros(len(y), bool)
    is_te[te] = True
    # 비교 기준: 상황만 쓴 트리(깊이 같음)
    cls_cols = [j for j, n in enumerate(names) if n.startswith("cls_")]
    est_c = DecisionTreeRegressor(max_depth=TREE_DEPTH, min_samples_leaf=TREE_LEAF, random_state=TREE_SEED)
    est_c.fit(X[tr][:, cls_cols], y[tr])
    r2_cls = r2_score(y[te], est_c.predict(X[te][:, cls_cols]))
    # 노드 통계는 전체 24,988 (원 Δ, 자르지 않은 값)으로 다시 잰다
    path = est.decision_path(X).tocsc()
    t = est.tree_
    stats = {}
    for j in range(t.node_count):
        rows = path[:, j].nonzero()[0]
        v = dF[rows]
        rt = rows[is_te[rows]]
        stats[j] = {"n": len(rows), "mean": float(v.mean()), "median": float(np.median(v)),
                    "mean_clip": float(y[rows].mean()),
                    "n_test": int(len(rt)), "mean_test": float(dF[rt].mean()) if len(rt) else np.nan,
                    "mean_15": float(dF15[rows].mean()),
                    "better2": float((v < -TIE_M).mean()), "better10": float((v > TIE_M).mean()), "rows": rows}
    lo_c = min(s["mean_clip"] for s in stats.values())
    hi_c = max(s["mean_clip"] for s in stats.values())
    bound = max(abs(lo_c), abs(hi_c))

    BIN_LAB = {"fallback": ("폴백 아님", "폴백"), "has_lead49": ("앞차 없음", "앞차 있음")}
    BIN_LAB.update({f"cls_{c}": (f"상황 ≠ {c}", f"상황 = {c}") for c in C.CLASSES})

    def fmt_thr(name, x):
        if abs(x) < 0.001:
            return f"{x:.1e}"
        if abs(x) < 0.01:
            return f"{x:.4f}"
        if abs(x) < 0.1:
            return f"{x:.3f}"
        if abs(x) < 1:
            return f"{x:.2f}"
        return f"{x:.1f}" if abs(x) < 100 else f"{x:.0f}"

    def build(j, rule, rx=()):
        s = stats[j]
        fc, tc = ramp_div(s["mean_clip"], -bound, bound)
        leaf = t.children_left[j] == -1
        rep = "재현" if np.sign(s["mean_15"]) == np.sign(s["mean"]) else "반대"
        nd = {"id": j, "rule": list(rule), "rule_exact": [list(z) for z in rx], "n": s["n"], "color": fc, "tc": tc,
              "fs": 9.0 if leaf else 9.8,
              "text": f"n={s['n']:,}\n평균 {s['mean']:+.2f} m\n2Hz 우세 {s['better2'] * 100:.0f}%"
                      f" · 10Hz {s['better10'] * 100:.0f}%"
                      + (f"\n검증30% {s['mean_test']:+.2f} · 15쌍 {s['mean_15']:+.2f} ({rep})" if leaf else
                         f"\n15쌍 {s['mean_15']:+.2f}")}
        if t.children_left[j] != -1:
            nm = names[t.feature[j]]
            if nm in BIN_LAB:
                l_lab, r_lab = BIN_LAB[nm]
            else:
                thr = fmt_thr(nm, float(t.threshold[j]))
                l_lab, r_lab = f"{labels[nm]} ≤ {thr}", f"{labels[nm]} > {thr}"
                if nm == "v0" and float(t.threshold[j]) < 0.05:
                    # AV2 속도 필드: 정지 차량은 1e-19 수준 -> 이 분기는 사실상 't=0 에 정지' 인가
                    l_lab, r_lab = l_lab + " (정지)", r_lab + " (움직임)"
            th = float(t.threshold[j])
            nd["children"] = [(l_lab, build(int(t.children_left[j]), rule + [l_lab], rx + ((nm, "<=", th),))),
                              (r_lab, build(int(t.children_right[j]), rule + [r_lab], rx + ((nm, ">", th),)))]
            nd["split"] = (nm, float(t.threshold[j]))
        return nd

    root = build(0, [])
    leaves = C.tree_leaves(root)
    # 잎 대표 시나리오: 잎 중앙값 Δ 에 가장 가까운 시나리오 (폴백 제외, 같은 거리면 앞 인덱스)
    sids = D.T["sid"].to_numpy()
    fb = D.T["fallback"].to_numpy()
    leaf_rows = []
    for lf in leaves:
        rows = stats[lf["id"]]["rows"]
        cand = rows[~fb[rows]] if (~fb[rows]).any() else rows
        rep = int(cand[np.argmin(np.abs(dF[cand] - stats[lf["id"]]["median"]))])
        leaf_rows.append({"leaf": lf["id"], "rule": " · ".join(lf["rule"]), "n": lf["n"],
                          "mean": stats[lf["id"]]["mean"], "median": stats[lf["id"]]["median"],
                          "better2": stats[lf["id"]]["better2"], "better10": stats[lf["id"]]["better10"],
                          "contrib_pct": 100.0 * lf["n"] * stats[lf["id"]]["mean"] / (len(dF) * dF.mean()),
                          "mean_15const": float(D.T["dF15"].to_numpy()[rows].mean()),
                          "same_sign_15const": bool(np.sign(stats[lf["id"]]["mean_15"]) == np.sign(stats[lf["id"]]["mean"])),
                          "n_test": stats[lf["id"]]["n_test"], "mean_test": stats[lf["id"]]["mean_test"],
                          "ci95_test": float(1.96 * dF[rows[is_te[rows]]].std(ddof=1) / np.sqrt(max(stats[lf["id"]]["n_test"], 2))),
                          "rep_idx": rep, "rep_sid": str(sids[rep]), "rep_dF": float(dF[rep]),
                          "rep_cls": str(D.T["cls"].iloc[rep]), "rule_exact": lf["rule_exact"]})
    # 검증: 잎 규칙을 표 조건식으로 다시 적용 (decision_path 와 다른 경로)
    check = []
    for lf, lr in zip(leaves, leaf_rows):
        m = np.ones(len(dF), bool)
        node = root
        for step in lf["rule"]:
            nm, thr = node["split"]
            col = X[:, names.index(nm)]
            left = node["children"][0][0] == step
            m &= (col <= thr) if left else (col > thr)
            node = node["children"][0][1] if left else node["children"][1][1]
        check.append({"leaf": lf["id"], "n_mask": int(m.sum()), "n_path": lr["n"],
                      "mean_mask": float(dF[m].mean()), "mean_path": lr["mean"],
                      "ok": bool(int(m.sum()) == lr["n"] and abs(float(dF[m].mean()) - lr["mean"]) < 1e-9)})
    imp = sorted(zip(names, est.feature_importances_), key=lambda z: -z[1])
    write_tree_rules(DATA / "tree_rules.txt", root, t, stats,
                     head=[f"c3 결정 트리 — 목표 ΔminFDE6 = 2Hz·30 − 10Hz·30 [m] (양수 = 10 Hz 가 가깝다)",
                           f"sklearn DecisionTreeRegressor 깊이 {TREE_DEPTH} · 잎 ≥ {TREE_LEAF} · 학습 70% (시드 {TREE_SEED})",
                           f"학습값 = 학습 70% 에서 p1–p99 ({lo:+.2f} ~ {hi:+.2f} m)로 자른 목표의 평균 (sklearn tree_.value)",
                           "전체값 = val 24,988 전부의 자르지 않은 Δ 평균 (그림 노드 글과 같다)",
                           "그림 노드 색 = 전체의 '자른 Δ' 평균, 노드 글 = '원 Δ' 평균 — 둘의 크기·부호가 다를 수 있다",
                           "검증 30% = 트리 학습에 쓰지 않은 표본의 원 Δ 평균. 전체값에는 분기를 고른 학습 표본이 섞여 있다",
                           "특성 |θ₀| 는 입력이 아니라 10Hz·30 모델 출력(첫 예측 스텝 θ)이다"
                           + ("" if "theta0" in {names[f] for f in t.feature if f >= 0} else " — 이 트리의 분기에는 쓰이지 않았다"),
                           "임계값은 유효숫자 4자리. 서술용 — 인과가 아니다."])
    SUM["c_tree"] = {"target": "ΔminFDE6 = 2Hz·30 − 10Hz·30 [m], 학습은 p1–p99 로 자른 값", "clip": [float(lo), float(hi)],
                     "depth": TREE_DEPTH, "min_leaf": TREE_LEAF, "seed": TREE_SEED, "split": "학습 70% / 검증 30%",
                     "features": {n: labels[n] for n in names},
                     "r2_train": float(r2_tr), "r2_test": float(r2_te), "r2_test_raw_target": float(r2_te_raw),
                     "r2_test_class_only_tree": float(r2_cls),
                     "r2_replicate_15const_pair": float(r2_rep),
                     "replicate_note": "같은 트리 예측을 15에폭 상수 두 판의 ΔminFDE6(p1–p99 자름)에 댄 R² — 다른 학습 쌍 재현성",
                     "root_threshold": float(t.threshold[0]), "root_feature": names[t.feature[0]],
                     "importance": [(labels[n], float(v)) for n, v in imp if v > 0],
                     "leaves": leaf_rows, "leaf_check": check, "leaf_check_ok": all(c["ok"] for c in check),
                     "note": "서술용 — 조건과 격차의 동시 발생을 요약할 뿐 인과가 아니다. 특성 중 |Δh|·최대|a|·6초 이동은 정답(미래)에서 온다."}
    # 그림
    n_leaf = len(leaves)
    fig = plt.figure(figsize=(max(24, 2.45 * n_leaf), 14.0))
    ax = fig.add_axes([0.005, 0.07, 0.99, 0.79])
    C.draw_tree(ax, root, fontsize=9.8, edge_fs=9.4)
    n_rep = sum(1 for r in leaf_rows if r["same_sign_15const"])
    fig.suptitle("c3 · 결정 트리 — 어떤 시나리오에서 2 Hz 가 이기나/지나 (ΔminFDE6 = 2 Hz − 10 Hz, 둘 다 30에폭 코사인)",
                 x=0.01, y=0.985, ha="left", fontsize=14)
    fig.text(0.01, 0.93, f"sklearn DecisionTreeRegressor · 깊이 {TREE_DEPTH} · 잎 ≥ {TREE_LEAF} · 학습 70% (시드 {TREE_SEED}) · "
                         f"목표값은 p1–p99 ({lo:+.1f} ~ {hi:+.1f} m)로 자름 · 설명력 R² 학습 {r2_tr:.3f} / 검증 {r2_te:.3f} "
                         f"(상황만 쓴 트리 검증 {r2_cls:.3f}) · 같은 트리를 15에폭 두 판의 Δ 에 대면 R² {r2_rep:.3f}. "
                         f"노드 글 = 전체 24,988 의 원 Δ 평균·우세 비율.", fontsize=10, color=C.INK2)
    corr_pair = SUM["c_dist"]["corr_dF_dF15"]
    fig.text(0.01, 0.908, f"잎 넷째 줄: 검증 30% 만의 평균 · 15쌍 = 같은 잎의 15에폭 상수 두 판 Δ 평균 (부호가 같으면 '재현', "
                          f"잎 {n_leaf}개 중 {n_rep}개). R² 가 0 에 가깝다 = 시나리오 하나하나의 차는 거의 예측되지 않는다 "
                          f"(c1: 두 학습 쌍의 시나리오별 Δ 상관 {corr_pair:.2f}).",
             fontsize=10, color=C.INK2)
    fig.text(0.01, 0.886, "색: 주황 = 음수(2 Hz 가 가깝다) · 파랑 = 양수(10 Hz 가 가깝다), 진할수록 크다 — 색은 전체의 '자른 Δ' 평균, "
                          "글자는 '원 Δ' 평균이라 둘이 다를 수 있다(잎 9 등). "
                          f"우세 = |Δ| > {TIE_M:g} m. 서술용 요약이지 인과가 아니다.", fontsize=10, color=C.INK2)
    note(fig, "특성: " + " · ".join(labels[n] for n in names if not n.startswith("cls_")) + " · 상황(원-핫). "
              "|Δh|·최대|a|·6초 이동은 정답(미래)에서, 관측 끝 |a|·|yaw|·시작 램프는 관측 위치에서 계산. 시드 1개.", fs=8.2)
    C.savefig(fig, OUT / "c3_tree_input_gap.png")
    print(f"[c3] 트리 R² 학습 {r2_tr:.3f} 검증 {r2_te:.3f} (상황만 {r2_cls:.3f}) · 15쌍 {r2_rep:.3f} · 잎 {n_leaf} "
          f"(15쌍 부호 재현 {n_rep}) · 대조 {'통과' if SUM['c_tree']['leaf_check_ok'] else '실패'}", flush=True)


def sig4(x):
    """임계값 표기: 유효숫자 4자리 (0.00004134 -> 4.134e-05, 13.9259 -> 13.93)."""
    x = float(x)
    if x != 0 and abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.4g}"


def write_tree_rules(path, root, t, stats, head):
    """export_text 대신 쓴다 — 임계값을 유효숫자로, 잎마다 학습값·전체값·n 을 함께 적는다."""
    lines = [f"# {h}" for h in head]

    def rec(nd, depth):
        j = nd["id"]
        s = stats[j]
        if not nd.get("children"):
            lines.append("|   " * depth + f"|--- 잎 {j}: 학습값 {float(t.value[j][0][0]):+.3f} (n_학습 {int(t.n_node_samples[j])}) · "
                         f"전체값 {s['mean']:+.3f} (n {s['n']}) · 검증 30% {s['mean_test']:+.3f} (n {s['n_test']}) · "
                         f"15쌍 {s['mean_15']:+.3f}")
            return
        nm, th = nd["split"]
        for (lab, ch), op in zip(nd["children"], ("<=", "> ")):
            lines.append("|   " * depth + f"|--- {nm} {op} {sig4(th)}   [{lab}]")
            rec(ch, depth + 1)

    rec(root, 0)
    path.write_text("\n".join(lines) + "\n")


def binned_mean(ax, x, ys, edges, labels, cols, names, xlabel, title, min_n=C.MIN_N, lss=None):
    """x 구간별 여러 Δ 의 평균 ± 95% CI. 반환: 표. lss = 선 모양 (30 쌍 실선 · 15 쌍 점선, 색 규칙)."""
    b = np.digitize(x, edges[1:-1])
    ok = np.isfinite(x)
    out = {}
    lss = lss or ["-"] * len(ys)
    for q, (y, col, nm, lsq) in enumerate(zip(ys, cols, names, lss)):
        mu, hw, ns = [], [], []
        for j in range(len(labels)):
            m = ok & (b == j)
            v = y[m]
            ns.append(int(m.sum()))
            mu.append(float(v.mean()) if len(v) else np.nan)
            hw.append(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan)
        xs = np.arange(len(labels)) + (q - (len(ys) - 1) / 2) * 0.22
        for x_, m_, h_, n_ in zip(xs, mu, hw, ns):
            a_ = 0.35 if n_ < min_n else 1.0
            ax.plot([x_, x_], [m_ - h_, m_ + h_], color=col, lw=1.2, alpha=a_, zorder=3)
            ax.plot(x_, m_, marker="o", ms=6.5, color=col, alpha=a_, zorder=4, mec=C.SURF, mew=1.0)
        ax.plot(xs, mu, color=col, lw=1.3, ls=lsq, alpha=0.7, zorder=2, label=nm)
        out[nm] = {"mean": mu, "ci95": hw, "n": ns}
    ax.axhline(0, color=C.INK2, lw=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\nn={n:,}" for l, n in zip(labels, out[names[0]]["n"])], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10)
    out["bins"] = labels
    return out


def fig_c4(D, dF, dF15):
    plt = C.setup_mpl()
    T = D.T
    fig, axes = plt.subplots(2, 3, figsize=(21, 10.4))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.08, wspace=0.22, hspace=0.5)
    ys, cols, nms = [dF, dF15], [C.C_VIOLET, C.MUTED], ["30 코사인: 2 Hz − 10 Hz", "15 상수: 2 Hz − 10 Hz"]
    lss = ["-", (0, (4, 2))]         # 색 규칙: 30에폭 실선 · 15에폭 점선 (Δ 는 보라 · 회색)
    import functools
    bm = functools.partial(binned_mean, lss=lss)
    res = {}
    res["ramp_ratio"] = bm(axes[0, 0], T["ramp_ratio"].to_numpy(), ys, [0, 0.4, 0.5, 0.6, 0.8, np.inf],
                                    ["<0.4", "0.4–0.5", "0.5–0.6", "0.6–0.8", "≥0.8"], cols, nms,
                                    "첫 스텝 속력 ÷ 1.0–1.5 s 속력 (작을수록 램프가 강함)",
                                    "① 시작 램프 강도 (1.0–1.5 s 속력 > 3 m/s 만)")
    res["ramp_acc"] = bm(axes[0, 1], T["ramp_acc"].to_numpy(), ys, [-np.inf, 2, 4, 6, 10, np.inf],
                                  ["<2", "2–4", "4–6", "6–10", "≥10"], cols, nms,
                                  "첫 0.5 s 위치차분 가속 [m/s²] (10 Hz 입력 a 채널이 담는 가짜 가속)",
                                  "② 램프가 10 Hz 입력에 넣는 가짜 가속의 크기 (①과 같은 모집단)")
    res["v0"] = bm(axes[0, 2], T["v0"].to_numpy(), ys, C.BINS["v0"][0], C.BINS["v0"][1], cols, nms,
                            "v0 [m/s]", "③ 현재 속력 (램프 가속은 속력에 비례)")
    res["a_last"] = bm(axes[1, 0], T["a_last"].to_numpy(), ys, [-np.inf, -2, -0.5, 0.5, 2, np.inf],
                                ["≤−2", "−2–−0.5", "±0.5", "0.5–2", "≥2"], cols, nms,
                                "관측 마지막 1초 속력 변화율 [m/s²] (위치차분, 관측만)",
                                "④ 예측 직전의 가감속 (10 Hz 가 0.1 s 해상도로 보는 것)")
    res["yaw_last"] = bm(axes[1, 1], T["abs_yaw_last"].to_numpy(), ys, [0, 1, 3, 6, 12, np.inf],
                                  ["<1", "1–3", "3–6", "6–12", "≥12"], cols, nms,
                                  "관측 마지막 1초 |진행방향 변화율| [°/s] (1 m/s 미만 구간이 있으면 0)",
                                  "⑤ 예측 직전의 회전 시작")
    res["thw"] = bm(axes[1, 2], T["thw49"].to_numpy(), ys, C.BINS["thw"][0], C.BINS["thw"][1], cols, nms,
                             "t=0 앞차 시간간격 [s] (앞차 있고 v0 ≥ 0.5 m/s)", "⑥ 앞차 시간간격")
    for ax in axes.flat:
        ax.set_ylabel("ΔminFDE6 평균 [m]")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle("c4 · 격차의 후보 설명 점검 — 창 가장자리 램프 · 예측 직전 동역학 · 앞차 (양수 = 2 Hz 가 나쁨)",
                 x=0.01, y=0.975, ha="left", fontsize=13.5)
    rp10, rp50, rp90 = np.nanpercentile(T["ramp_ratio"].to_numpy(), [10, 50, 90])
    fig.text(0.01, 0.915, "가설 H1 램프: 10 Hz 입력은 인덱스 0~4 의 인공 감속(가짜 가속 약 1 g)을 담고, 2 Hz 입력은 인덱스 4 부터 평활해 피한다 "
                          "→ 램프가 강할수록 10 Hz 가 불리하면 Δ 가 음수 쪽으로 가야 한다.\n"
                          "가설 H2 해상도: 10 Hz 는 마지막 0.5 s 를 5스텝으로, 2 Hz 는 1스텝(평활)으로 본다 → 예측 직전 동역학이 클수록 "
                          "Δ 가 양수 쪽으로 가야 한다. 둘 다 관측으로 따진 서술이고, 입력만 바꾼 재학습(예: 첫 0.5 s 를 가린 10 Hz 입력) 없이 "
                          "원인은 확정할 수 없다. "
                          f"램프는 이동 시나리오 대부분에 있어(① p10–p90 {rp10:.2f}–{rp90:.2f}) 전체에 고르게 작용하는 효과는 이 그림으로 배제할 수 없다.",
             fontsize=9.3, color=C.INK2)
    res["ramp_ratio_pct"] = [float(rp10), float(rp50), float(rp90)]
    note(fig, "점 = 구간 평균 · 세로선 = 95% 신뢰구간 · 흐린 점 n < 50 · 보라 실선 = 30에폭 코사인 두 판의 차 · 회색 점선 = 15에폭 상수 두 판의 차 · "
              "val 24,988 · best 체크포인트 · 시드 1개.")
    C.savefig(fig, OUT / "c4_gap_hypotheses.png")
    SUM["c_hypo"] = res


# =========================================================================== d 실현가능성
OSC_WIN, OSC_K0, OSC_K1 = 15, 8, 45     # 1.5 s 중앙 이동평균, 평가 구간 예측 스텝 8..44 (0.9–4.5 s)


def osc_rms(a):
    """가속도 시계열(...,60)에서 1.5 s 중앙 이동평균을 뺀 잔차의 RMS (0.9–4.5 s) — 1~2 s 주기 출렁임의 크기.
    가장자리는 이동평균 창이 60스텝 안에 들어오는 구간만 쓴다(8−7 ≥ 0, 44+7 ≤ 59)."""
    a = np.asarray(a, np.float64)
    h = OSC_WIN // 2
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1)   # c[j] = Σ a[:j]
    k = np.arange(OSC_K0, OSC_K1)
    ma = (c[..., k + h + 1] - c[..., k - h]) / OSC_WIN                                      # 평균 a[k−h..k+h]
    r = a[..., k] - ma
    return np.sqrt((r ** 2).mean(-1))


def smooth_like_gt(v_past, v_fut):
    """정답 a_fld 와 같은 평활: 속력(관측부 = 속도 필드, 예측부 = 모델) -> median 5 -> SG 15/2 미분 -> 예측부.
    viz_v4_dump._smooth_speed 를 행마다 부르는 대신 축을 맞춰 한 번에 건다 (같은 식, sec_v 에서 대조)."""
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    ser = np.concatenate([np.asarray(v_past, np.float64), np.asarray(v_fut, np.float64)], axis=1)
    vm = median_filter(ser, size=(1, C.MED_K), mode="nearest")
    a = savgol_filter(vm, C.SG_WIN, C.SG_POLY, deriv=1, delta=C.DT, axis=1, mode="interp")
    return a[:, C.OBS:]


def sec_d(D, M):
    plt = C.setup_mpl()
    from matplotlib.lines import Line2D
    R = D.R
    N = D.N
    ar = np.arange(N)
    pos = R["pos"].astype(np.float64)
    step = np.linalg.norm(np.diff(pos, axis=1), axis=2) / C.DT
    fut = step[:, C.OBS - 1:]                                    # k: 49+k -> 50+k
    vref = fut[:, 40:50].mean(1)
    ok = vref > 5.0
    V = {}
    for k in ORDER:
        P, df = D.P[k], D.df[k]
        top, win = df["top1"].to_numpy(), df["winner"].to_numpy()
        V[k] = {"vt": P["v"][ar, top].astype(np.float64), "vw": P["v"][ar, win].astype(np.float64),
                "at": P["a"][ar, top].astype(np.float64)}
        ok &= (V[k]["vt"][:, 40:50].mean(1) > 1.0) & (V[k]["vw"][:, 40:50].mean(1) > 1.0)
    vf = R["v_fld"][:, C.OBS - 1:].astype(np.float64)

    def deficit(s, r):
        return ((r[:, None] - s[:, 50:60]) * C.DT).sum(1)

    dg = deficit(fut[ok], vref[ok])
    dfield = deficit(vf[ok], vf[ok][:, 40:50].mean(1))
    feas = {"end_pop": {"n": int(ok.sum()), "n_moving": int((vref > 5.0).sum()),
                        "gt_deficit_median": float(np.median(dg)), "field_deficit_median": float(np.median(dfield))}}
    kk = np.arange(30, 60)
    for k in ORDER:
        P, df = D.P[k], D.df[k]
        al = P["alive"]
        top = df["top1"].to_numpy()
        live = al.sum() * (C.FUT - 1)
        a = P["a"].astype(np.float64)
        jerk = np.diff(a, axis=2) / C.DT                               # m/s³
        dth = np.degrees(P["dtheta"].astype(np.float64)) / C.DT         # °/s (dθ 는 스텝당)
        yacc = np.diff(dth, axis=2) / C.DT                              # °/s²
        m3 = al[:, :, None]
        jt = jerk[ar, top]
        vt, vw = V[k]["vt"][ok], V[k]["vw"][ok]
        rt, rw = vt[:, 40:50].mean(1), vw[:, 40:50].mean(1)
        dt_, dw_ = deficit(vt, rt), deficit(vw, rw)
        feas[k] = {
            "exc_pct": float(100 * np.where(al, P["exc_cnt"], 0).sum() / live),
            "jit_theta": float(np.where(al, P["jit_t_sum"], 0).astype(np.float64).sum() / live),
            "jit_a": float(np.where(al, P["jit_a_sum"], 0).astype(np.float64).sum() / live),
            "jerk_rms_all": float(np.sqrt((jerk ** 2 * m3).sum() / live)),
            "jerk_rms_top1": float(np.sqrt((jt ** 2).mean())),
            "jerk_p99_abs_all": float(np.percentile(np.abs(jerk[np.broadcast_to(m3, jerk.shape)]), 99)),
            "yawacc_rms_all": float(np.sqrt((yacc ** 2 * m3).sum() / live)),
            "top_deficit_median": float(np.median(dt_)), "win_deficit_median": float(np.median(dw_)),
            "top_follow": float(np.median(dt_) / np.median(dg)), "win_follow": float(np.median(dw_) / np.median(dg)),
            "top_a_last6": float(V[k]["at"][ok][:, 54:60].mean()),
            "top_ratio": np.median(vt[:, kk] / rt[:, None], axis=0).tolist(),
            "top_a_mean_t": V[k]["at"][ok].mean(0).tolist(),
            "offlane": float(M[k]["offlane"].mean()),
        }
        # 1~2 s 주기 가속도 출렁임: 평균 곡선(위상이 맞는 부분)과 시나리오별(1위 모드)
        osc_s = osc_rms(V[k]["at"])
        D.T[f"osc_{k}"] = osc_s
        feas[k]["osc_mean_curve_rms"] = float(osc_rms(V[k]["at"][ok].mean(0)))
        feas[k]["osc_scen_median"] = float(np.median(osc_s[ok]))
        feas[k]["osc_scen_p90"] = float(np.percentile(osc_s[ok], 90))
        # 정답과 같은 평활(속력 -> median 5 -> SG 1.5 s 미분)을 모델 1위 속력에도 건 값 — 정답과 비교할 때는 이것만 쓴다
        osc_m = osc_rms(smooth_like_gt(R["v_fld"][:, :C.OBS], V[k]["vt"]))
        D.T[f"osc_sm_{k}"] = osc_m
        feas[k]["osc_sm_median"] = float(np.median(osc_m[ok]))
        feas[k]["osc_sm_p90"] = float(np.percentile(osc_m[ok], 90))
    feas["osc_def"] = (f"가속도에서 {OSC_WIN * C.DT:g} s 중앙 이동평균을 뺀 잔차 RMS, 예측 {(OSC_K0 + 1) * C.DT:.1f}–"
                       f"{OSC_K1 * C.DT:.1f} s, 1위 모드, 끝 감속 모집단. osc_sm = 모델 1위 속력(관측부는 속도 필드)에 정답과 같은 "
                       f"평활(median {C.MED_K} -> SG {C.SG_WIN}/{C.SG_POLY} 미분)을 건 뒤 같은 지표")
    # 정답 가속도(속도 필드 median 5 -> SG 1.5 s 미분)의 같은 지표 — 모델은 osc_sm 과만 비교한다
    ga = R["a_fld"][:, C.OBS:].astype(np.float64)
    D.T["osc_gt"] = osc_rms(ga)
    D.T["end_pop"] = ok
    feas["osc_gt_field"] = {"mean_curve_rms": float(osc_rms(ga[ok].mean(0))),
                            "scen_median": float(np.median(osc_rms(ga[ok]))),
                            "scen_p90": float(np.percentile(osc_rms(ga[ok]), 90))}
    feas["gt_ratio"] = np.median(fut[ok][:, kk] / vref[ok, None], axis=0).tolist()
    vfr = vf[ok][:, 40:50].mean(1)
    feas["field_ratio"] = np.median(vf[ok][:, kk] / vfr[:, None], axis=0).tolist()
    feas["gt_a_fld_mean_t"] = R["a_fld"][ok][:, C.OBS:].astype(np.float64).mean(0).tolist()
    # 정답 가감속 (위치 차분, 같은 모집단) — 끝 0.5 초 인공 감속이 여기서 보인다
    feas["gt_a_pos_mean_t"] = (np.diff(fut[ok], axis=1) / C.DT).mean(0).tolist()
    SUM["d_feas"] = feas

    fig = plt.figure(figsize=(23, 11.4))
    gs = fig.add_gridspec(2, 1, left=0.04, right=0.99, top=0.86, bottom=0.08, hspace=0.42)
    gtop = gs[0].subgridspec(1, 6, wspace=0.45)
    gbot = gs[1].subgridspec(1, 4, wspace=0.28, width_ratios=[1.0, 1.25, 1.0, 1.0])
    bars = [("exc_pct", "7.3°초과 [%]\n(살아있는 모드 step)", "{:.2f}"),
            ("jit_theta", "흔들림 θ\n(Δ(dθ/8°)² 평균)", "{:.5f}"),
            ("jit_a", "흔들림 a\n(Δ(a/8)² 평균)", "{:.5f}"),
            ("jerk_rms_all", "저크 RMS [m/s³]\n(살아있는 모든 모드)", "{:.2f}"),
            ("yawacc_rms_all", "잔차각 가속 RMS [°/s²]\n(Δ(dθ/Δt)/Δt, 모든 모드)", "{:.0f}"),
            ("offlane", "이탈\n(시나리오당 0~6)", "{:.3f}")]
    for j, (key, lab, fmt) in enumerate(bars):
        ax = fig.add_subplot(gtop[0, j])
        for x, k in enumerate(ORDER):
            v = feas[k][key]
            ax.bar(x, v, width=0.7, zorder=3, **bar_kw(k))
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=7.8, color=C.INK2)
        ax.set_xticks(range(4))
        ax.set_xticklabels([SHORT[k] for k in ORDER], rotation=40, ha="right", fontsize=8.2)
        ax.set_title(lab, loc="left", fontsize=9.6)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    t = (kk + 1) * C.DT
    ax = fig.add_subplot(gbot[0, 0])
    ax.axvspan(5.45, 6.05, color=C.GRID, alpha=0.8, lw=0, zorder=0)
    ax.plot(t, feas["gt_ratio"], color=C.C_GT, lw=2.0, zorder=5, **C.step_kw())
    ax.plot(t, feas["field_ratio"], color=C.MUTED, lw=1.4, ls=(0, (3, 2)), zorder=5)
    for k in ORDER:
        ax.plot(t, feas[k]["top_ratio"], color=COL[k], ls=LS[k], lw=1.9, zorder=4, **C.step_kw())
    ax.axhline(1, color=C.INK2, lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel("예측 시간 [s] · 점 = 0.1 s 스텝")
    ax.set_ylabel("속력 ÷ (4.1–5.0 s 평균) — 중앙값")
    ax.set_title("① 끝 3초 속력 비 — 확률 1위 모드", loc="left", fontsize=10.5)
    ax.legend([Line2D([], [], color=C.C_GT, lw=2, marker="o", ms=3), Line2D([], [], color=C.MUTED, lw=1.4, ls=(0, (3, 2)))],
              ["정답 (위치 차분)", "정답 (AV2 속도 필드·평활)"], loc="lower left", fontsize=8.5)
    ax = fig.add_subplot(gbot[0, 1])
    tt = np.arange(1, 61) * C.DT
    ax.axvspan(5.45, 6.05, color=C.GRID, alpha=0.8, lw=0, zorder=0)
    ax.plot(tt, feas["gt_a_fld_mean_t"], color=C.MUTED, lw=1.4, ls=(0, (3, 2)), zorder=5)
    ax.plot(np.arange(1, 60) * C.DT, feas["gt_a_pos_mean_t"], color=C.C_GT, lw=1.6, zorder=5, **C.step_kw())
    for k in ORDER:
        ax.plot(tt, feas[k]["top_a_mean_t"], color=COL[k], ls=LS[k], lw=1.9, zorder=4, **C.step_kw())
    ax.axhline(0, color=C.INK2, lw=0.8)
    ax.set_ylim(-3.5, 1.5)
    ax.set_xlabel("예측 시간 [s] · 점 = 0.1 s 스텝")
    ax.set_ylabel("평균 가속도 [m/s²]")
    ax.set_title("② 확률 1위 모드의 평균 가속도 a(t) (−3.5 ~ 1.5 밖은 잘림)", loc="left", fontsize=10.5)
    ax.legend([Line2D([], [], color=C.C_GT, lw=1.6, marker="o", ms=3), Line2D([], [], color=C.MUTED, lw=1.4, ls=(0, (3, 2)))],
              ["정답 (위치 차분의 차분)", "정답 (속도 필드 평활)"], loc="lower left", fontsize=8.5)
    ax.text(0.02, 0.975, "평균 곡선의 1~2 s 파동은 2Hz·30 에만 있다 (4.5 s 봉우리는 네 판 모두)", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=C.INK2, bbox=dict(fc=C.SURF, ec="none", pad=1.0, alpha=0.9), zorder=8)
    # ③ 1~2 s 출렁임: 원 출력 / 정답과 같은 평활 — 정답 기준선은 같은 평활 쪽에만 긋는다
    ax = fig.add_subplot(gbot[0, 2])
    og = feas["osc_gt_field"]
    groups = [("osc_scen_p90", "osc_scen_median", "원 출력\n(모델 a 그대로)"),
              ("osc_sm_p90", "osc_sm_median", "정답과 같은 평활\n(속력 → median 5 → SG 1.5 s 미분)")]
    for x, (kp, km, lab) in enumerate(groups):
        for q, k in enumerate(ORDER):
            v, vm = feas[k][kp], feas[k][km]
            xx = x + (q - 1.5) * 0.19
            ax.bar(xx, v, width=0.17, zorder=3, **bar_kw(k))
            ax.plot([xx - 0.07, xx + 0.07], [vm, vm], color=C.INK, lw=1.6, zorder=5)
            ax.text(xx, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4, color=C.INK2)
    ax.plot([0.55, 1.45], [og["scen_p90"]] * 2, color=C.C_GT, lw=1.8, zorder=4)
    ax.text(1.45, og["scen_p90"], f" 정답 p90 {og['scen_p90']:.2f}\n (중앙 {og['scen_median']:.2f})", va="center",
            ha="left", fontsize=8, color=C.INK)
    ax.set_xlim(-0.5, 1.95)
    ax.set_ylim(0, max(ax.get_ylim()[1], og["scen_p90"] * 1.25))
    ax.set_xticks(range(2))
    ax.set_xticklabels([g[2] for g in groups], fontsize=8.2)
    ax.set_ylabel("1.5 s 이동평균 잔차 RMS [m/s²]")
    ax.set_title("③ 1~2 s 주기 가속 출렁임 — 막대 p90 · 검은 가로선 중앙값\n(1위 모드, 0.9–4.5 s, 원 출력은 정답과 비교하지 않는다)",
                 loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gbot[0, 3])
    items = [("top_deficit_median", "확률 1위"), ("win_deficit_median", "승자")]
    for x, (key, lab) in enumerate(items):
        for q, k in enumerate(ORDER):
            v = feas[k][key]
            xx = x + (q - 1.5) * 0.19
            ax.bar(xx, v, width=0.17, zorder=3, **bar_kw(k))
            ax.text(xx, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=C.INK2)
    ax.axhline(feas["end_pop"]["gt_deficit_median"], color=C.C_GT, lw=1.6, zorder=2)
    ax.axhline(feas["end_pop"]["field_deficit_median"], color=C.MUTED, lw=1.4, ls=(0, (3, 2)), zorder=2)
    tr_ = ax.get_yaxis_transform()
    # 라벨은 막대가 없는 오른쪽 끝에 (정답 위치 선 위 / 속도 필드 선 아래)
    ax.text(0.995, feas["end_pop"]["gt_deficit_median"] + 0.03, f"정답(위치) {feas['end_pop']['gt_deficit_median']:.2f} m",
            va="bottom", ha="right", fontsize=8.4, transform=tr_)
    ax.text(0.995, feas["end_pop"]["field_deficit_median"] - 0.03,
            f"정답(속도 필드) {feas['end_pop']['field_deficit_median']:.2f} m", va="top", ha="right", fontsize=8.4,
            color=C.INK2, transform=tr_)
    ax.set_ylim(min(ax.get_ylim()[0], feas["end_pop"]["field_deficit_median"] - 0.22),
                max(ax.get_ylim()[1], feas["end_pop"]["gt_deficit_median"] + 0.2))
    ax.set_xlim(-0.5, 1.75)
    ax.set_xticks(range(2))
    ax.set_xticklabels([l for _, l in items])
    ax.set_ylabel("끝 1초 부족 거리 [m] (중앙값)")
    ax.set_title("④ 등속 대비 끝 1초에 덜 간 거리\n라벨 인공 감속 추종", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    hs, ls = run_handles()
    fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.04, 0.945), ncol=4, fontsize=9.5)
    fig.suptitle("d1 · 실현가능성 — 각도 초과 · 흔들림 · 저크 · 1~2 s 출렁임 · 끝 감속 (val 24,988, best 체크포인트)",
                 x=0.01, y=0.99, ha="left", fontsize=13.5)
    fig.text(0.4, 0.955, "흔들림 벌점은 0.1 s 스텝 사이 차이만 본다 → 1~2 s 주기의 느린 출렁임은 거의 벌하지 않는다. "
                         "③ 30에폭 판에서 커진 것은 p90 꼬리이고(같은 평활 기준 ×"
                         f"{feas['h10_c30']['osc_sm_p90'] / feas['h10_c15']['osc_sm_p90']:.1f} · "
                         f"×{feas['h2_c30']['osc_sm_p90'] / feas['h2_c15']['osc_sm_p90']:.1f}), 같은 평활로 보면 네 판 모두 정답보다 작다",
             fontsize=9.2, color=C.INK2)
    note(fig, f"③④ 모집단(끝 감속 모집단) = 4.1–5.0 s 정답(위치 차분) 속력 > 5 m/s 이고 네 판 모두 1위·승자 모드의 같은 구간 속력 > 1 m/s "
              f"({feas['end_pop']['n_moving']:,} 중 {feas['end_pop']['n']:,}). 부족 거리 = Σ_(5.1–6.0 s)(기준 속력 − 속력)·0.1 s. "
              "저크·잔차각 가속은 모델 출력 a·dθ 의 0.1 s 차분(두 입력 모두 출력은 10 Hz).", fs=7.9)
    C.savefig(fig, OUT / "d1_feasibility.png")
    print("[d] 실현가능성", flush=True)


# =========================================================================== e 평균 사례 궤적 (어두운 지도)
def sec_e(D):
    import viz_v4_gallery as G
    plt = C.setup_mpl()
    G.dark_rc()
    runs = {k: G.Run(TAG[k]) for k in ORDER}
    ref = runs[REF]
    idxs, dist, st = G.pick_mean(ref.df)
    stats = {k: G.pick_mean(runs[k].df)[2] for k in ORDER}
    rec = {"rule": "10Hz·30 판 (minADE6, minFDE6) 가 그 판 평균에 가장 가까운 9개 (viz_v4_gallery.pick_mean)",
           "idx": idxs, "sid": [str(ref.df['sid'].iloc[i]) for i in idxs], "stats": stats, "panels": {}}
    for mode in ("best", "top1"):
        rec["panels"][mode] = draw_e(runs, idxs, st, stats, mode, OUT / f"e{1 if mode == 'best' else 2}_avg_{mode}_4runs.png")
    G_facts = {}
    for i in idxs:
        f = {}
        for k in ORDER:
            r = runs[k].df.iloc[i]
            f[k] = {"minade": float(r["minade"]), "minfde": float(r["minfde"]), "top1_ade": float(r["top1_ade"]),
                    "top1_fde": float(r["top1_fde"]), "top1_prob": float(runs[k].prob[i][int(r["top1"])]),
                    "win_eq_top1": bool(r["win_eq_top1"]), "route_err": bool(r["route_err"]), "cls": r["cls"]}
        G_facts[str(ref.df["sid"].iloc[i])] = f
    rec["facts"] = G_facts
    # 검증: 그림에 쓴 궤적으로 지표 재계산
    rec["metric_check"] = {k: G.metric_check(runs[k], idxs)["max_abs_diff_m"] for k in ORDER}
    SUM["e_gallery"] = rec
    print(f"[e] 평균 사례 네 판 (지표 대조 최대 차 {max(rec['metric_check'].values()):.1e} m)", flush=True)


def draw_e(runs, idxs, st, stats, mode, path):
    import viz_v4_gallery as G
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    header_in = 0.28 + 0.52 + 0.30 * 3 + 0.12 + 0.34 * 3 + 0.45
    foot = G.def_lines(G.PAST_VIEW_S, compare=True)[:3] + [
        "점 = 1초 간격(1~6 s) · 색 = 입력 주기(초록 10 Hz, 분홍 2 Hz — 어두운 지도용, 밝은 그림의 파랑·주황과 같은 뜻) · "
        "선 = 스케줄(실선 30에폭 코사인, 점선 15에폭 상수) · 10 Hz 는 굵게 아래에, 2 Hz 는 가늘게 위에, 정답(파랑)은 맨 위에 가늘게",
        "과거 선의 작은 점 = 1초 간격, 빈 원 = 2 Hz 판의 입력 시점(0.5 s) · 원점→첫 예측점(t = 0.1 s)은 가는 점선 · "
        + ("best of 6 = 끝점 오차 최소(승자) 모드, 칸 제목 ADE 는 그 모드의 ADE" if mode == "best" else
           "확률 1위 모드 (칸 제목 괄호 = 그 모드 확률, ○/× = 그 모드의 경로가 정답 기준 경로인가)") + " · 모델마다 시드 1개"]
    fig, axes, W, H, pw = G.grid_figure(header_in, footer_in=0.12 + 0.27 * len(foot) + 0.12, title_lines=5)
    ref = runs[REF]
    rows = []
    for ax, i in zip(axes, idxs):
        r0 = ref.df.iloc[i]
        scene, inter = ref.scene(i)
        past = ref.pos[i, :C.OBS]
        gt = G.with_origin(ref.y[i])
        tr = {}
        for k in ORDER:
            r = runs[k].df.iloc[i]
            m = int(r["winner"] if mode == "best" else r["top1"])
            tr[k] = (runs[k].traj[i, m].astype(np.float64), m, r)
        xlim, ylim = G.view([G.past_for_view(past, G.PAST_VIEW_S), gt] + [t for t, _, _ in tr.values()])
        G.draw_map_dark(ax, scene, xlim, ylim)
        C.draw_box(ax, 0, 0, 0, G.C_FOCAL, zorder=7, ec="white", lw=1.0)
        ax.plot(past[:, 0], past[:, 1], color=G.C_PAST, lw=2.6, zorder=10, path_effects=G._pe(2.6))
        C.past_dots(ax, past, 2, G.C_PAST, zorder=10.5, dark=True)      # 1 s 점 + 2 Hz 판 입력 시점(빈 원)
        for z, k in enumerate(ORDER):
            u = np.asarray(tr[k][0], np.float64)
            C.origin_join(ax, u[0], DCOL[k], lw=DLW[k], zorder=13.5 + z)
            ax.plot(u[:, 0], u[:, 1], color=DCOL[k], lw=DLW[k], ls=DLS[k], zorder=14 + z,
                    path_effects=G._pe(DLW[k], DPE[k]))
            ax.scatter(u[9::10, 0], u[9::10, 1], s=DMS[k], color=DCOL[k],
                       edgecolors=G.OUTLINE, linewidths=0.6, zorder=14.5 + z)
        # 정답은 예측 위에 가늘게 — 예측이 정답과 겹쳐도 정답이 가려지지 않는다
        ax.plot(gt[:, 0], gt[:, 1], color=G.C_GT, lw=2.4, zorder=20, path_effects=G._pe(2.4, 1.6))
        ax.scatter(gt[10::10, 0], gt[10::10, 1], s=22, color=G.C_GT, edgecolors=G.OUTLINE, linewidths=0.9, zorder=20.5)
        G.finish_axes(ax, xlim, ylim)
        lines = [G.title_line1(r0, inter)]
        for k in ORDER:
            u, m, r = tr[k]
            if mode == "best":
                # 그린 선 = 승자(끝점 오차 최소) 모드 -> 그 모드의 ADE (minADE6 와 다를 수 있다) / FDE (= minFDE6)
                lines.append(f"{SHORT[k]}  승자 ADE {float(r['win_ade']):.2f} / FDE {float(r['minfde']):.2f} m · "
                             f"1위{'=' if r['win_eq_top1'] else '≠'}best")
            else:
                ok = "" if int(r["n_distinct"]) < 2 else (" ○" if not r["route_err"] else " ×")
                lines.append(f"{SHORT[k]}  ADE {float(r['top1_ade']):.2f} / FDE {float(r['top1_fde']):.2f} m "
                             f"({float(runs[k].prob[i][m]):.2f}){ok}")
        G.panel_title(ax, lines, pw, colors=[G.TXT] + [DCOL[k] for k in ORDER], sizes=[12.6] + [11.0] * 4)
        rows.append({"idx": int(i), "sid": str(r0["sid"])})
    title = ("v4 예측 궤적 — 네 판 비교 · " + ("best of 6" if mode == "best" else "확률 1위 모드")
             + " (같은 시나리오 · 10 Hz·30 판 평균값 기준 9개)")
    s1 = "모집단 val 24,988  ·  평균 minADE6 / minFDE6:  " + "  ·  ".join(
        f"{SHORT[k]} {stats[k]['mean_ade']:.3f} / {stats[k]['mean_fde']:.3f}" for k in ORDER) + " m"
    s2 = ("공통: L4nw · (a, h) · 전체 데이터 · 흔들림 벌점 1.0 · θ0 guard · best 체크포인트  —  "
          "15 = 15에폭 학습률 5e-4 고정, 30 = 30에폭 코사인 5e-4→1e-5")
    s3 = G.composition(ref, idxs, [ref.scene(i)[1] for i in idxs])
    y = G.header(fig, W, H, title, [s1, s2, s3])
    h = [Line2D([], [], color=G.C_PAST, lw=2.6, marker="o", ms=4, mec=G.OUTLINE),
         Line2D([], [], color=G.C_GT, lw=2.4, marker="o", ms=5, mec=G.OUTLINE),
         Patch(facecolor=G.C_FOCAL, edgecolor="white")]
    lab = ["과거 5초 (관측, ● 1초 · ○ 2 Hz 입력 시점)", "실제 차량이 간 길 (정답 6초, 맨 위 · ● 1초 간격)", "focal 차량 (t = 0 s)"]
    # 열 우선 3열: [공통 3] [10 Hz 두 판 + 빈칸] [2 Hz 두 판 + 빈칸]
    for grp in (("h10_c15", "h10_c30"), ("h2_c15", "h2_c30")):
        for k in grp:
            h.append(Line2D([], [], color=DCOL[k], lw=DLW[k], ls=DLS[k], marker="o", ms=5, mec=G.OUTLINE))
            lab.append(f"{LAB[k]} ({'실선' if k.endswith('c30') else '점선'})")
        h.append(Line2D([], [], lw=0))
        lab.append("")
    G.legend(fig, W, H, y - 0.10, h, lab, ncol=3)
    G.footer(fig, H, foot)
    C.savefig(fig, path, dpi=140)
    return rows


# =========================================================================== f 대표 시나리오 (밝은 지도 + 시계열)
class CaseCtx:
    def __init__(self, D):
        from dataset_cached import CachedV4Dataset
        ds = CachedV4Dataset(C.VAL_CACHES["ah2"])
        self.cache = {k: ds.raw(k) for k in ("routes", "route_tan", "route_band", "route_len", "route_mask",
                                             "origin", "theta")}
        self.D = D
        self._scene = {}

    def scene(self, i):
        import viz_v4_cases as VC
        if i not in self._scene:
            sid = self.D.T["sid"].iloc[i]
            o, th = self.cache["origin"][i], float(self.cache["theta"][i])
            self._scene[i] = (C.build_scene(sid, o, th), VC.others_at_obs(sid, o, th))
        return self._scene[i]

    def gt_route_frame(self, i):
        """정답 기준 경로에서 정답의 (s, d, 밴드, k) — viz_v4_dump.raw_one 과 같은 계산."""
        import lane_frame as lf
        import viz_v4_dump as VD
        g = int(self.D.T["gt_route"].iloc[i])
        rd = C.route_dict(self.cache["routes"][i, g], self.cache["route_tan"][i, g], self.cache["route_len"][i, g])
        ln = float(self.cache["route_len"][i, g])
        pos = self.D.R["pos"][i].astype(np.float64)
        s, d, _ = lf.to_frame(pos, rd)
        bd = VD._band_at(np.asarray(self.cache["route_band"][i, g], np.float64), ln, s)
        tk = VD._band_at(np.asarray(self.cache["route_tan"][i, g], np.float64), ln, s)
        return s, d, bd, np.arctan2(tk[:, 1], tk[:, 0])


def draw_case_map(ax, cx, i, mode="top1", compact=False, keys=ORDER, fit=False):
    D = cx.D
    T = D.T
    nd = int(T["n_distinct"].iloc[i])
    g = int(T["gt_route"].iloc[i])
    pos = D.R["pos"][i]
    tr = {}
    for k in keys:
        r = D.df[k].iloc[i]
        tr[k] = D.P[k]["traj"][i, int(r["top1"] if mode == "top1" else r["winner"])]
    pts = np.concatenate([pos[25:]] + list(tr.values()))
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    ratio = (bb.height * fh) / max(bb.width * fw, 1e-6)
    if fit:
        # 내용에 맞춘 시야 — 축 상자가 시야 비율로 줄어든다(긴 직선 도로에서 빈 지도를 줄인다)
        hx, hy = max((hi - lo)[0] * 0.58 + 4, 14.0), max((hi - lo)[1] * 0.58 + 4, 11.0)
        hy = max(hy, hx * 0.3)             # 너무 납작하지 않게
    else:
        hx, hy = max((hi - lo)[0] * 0.6, 14.0), max((hi - lo)[1] * 0.6, 14.0 * min(ratio, 1.0))
        if hy / hx < ratio:
            hy = hx * ratio
        else:
            hx = hy / ratio
    xlim, ylim = (c[0] - hx, c[0] + hx), (c[1] - hy, c[1] + hy)
    scene, others = cx.scene(i)
    C.draw_scene_light(ax, scene, xlim, ylim, mark_lw=0.8 if compact else 1.0)
    rm = cx.cache["route_mask"][i]
    for k_ in range(nd):
        if not rm[k_]:
            continue
        rp = cx.cache["routes"][i][k_]
        on = k_ == g
        if on:
            poly = C.band_polygon(rp, cx.cache["route_tan"][i][k_], cx.cache["route_band"][i][k_])
            ax.fill(poly[:, 0], poly[:, 1], color=C.C_BAND, alpha=0.08, lw=0, zorder=5)
        ax.plot(rp[:, 0], rp[:, 1], color=C.INK2 if on else C.MUTED, lw=1.3 if on else 0.8,
                ls="-" if on else (0, (3, 2)), alpha=0.9, zorder=6)
        if not compact:
            inside = np.where((rp[:, 0] > xlim[0] + 2) & (rp[:, 0] < xlim[1] - 2)
                              & (rp[:, 1] > ylim[0] + 2) & (rp[:, 1] < ylim[1] - 2))[0]
            if len(inside):
                q = inside[int(len(inside) * (0.5 + 0.09 * k_)) % len(inside)]
                ax.text(rp[q, 0], rp[q, 1], f"r{k_}" + (" 정답" if on else ""), fontsize=7, color=C.INK2, zorder=7,
                        clip_on=True, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    for x, y, yaw, typ in others:
        if xlim[0] - 5 < x < xlim[1] + 5 and ylim[0] - 5 < y < ylim[1] + 5 and typ in ("vehicle", "bus", "motorcyclist"):
            C.draw_box(ax, x, y, yaw, "#b9b6ad", alpha=0.9, zorder=8,
                       length=2.2 if typ == "motorcyclist" else (11.0 if typ == "bus" else 4.6),
                       width=0.9 if typ == "motorcyclist" else (2.5 if typ == "bus" else 1.9))
    import matplotlib.patheffects as pe
    halo = lambda lw: [pe.Stroke(linewidth=lw + 2.2, foreground="white"), pe.Normal()]
    ax.plot(pos[:C.OBS, 0], pos[:C.OBS, 1], color=C.C_PAST, lw=2.2, zorder=10)
    C.past_dots(ax, pos[:C.OBS], 2, C.C_PAST, zorder=10.5)            # 1 s 점 + 2 Hz 판 입력 시점(빈 원)
    ax.plot(pos[C.OBS - 1:, 0], pos[C.OBS - 1:, 1], color=C.C_GT, lw=2.4, zorder=14, path_effects=halo(2.4))
    ax.scatter(pos[C.OBS + 9::10, 0], pos[C.OBS + 9::10, 1], s=18, color=C.C_GT, zorder=15, edgecolors="white",
               linewidths=0.7)
    for z, k in enumerate(keys):
        t = np.asarray(tr[k], np.float64)
        C.origin_join(ax, t[0], COL[k], lw=2.2, zorder=10.8)       # 원점 -> t=0.1 s 는 가는 점선
        ax.plot(t[:, 0], t[:, 1], color=COL[k], ls=LS[k], lw=2.2, zorder=11 + z * 0.1,
                path_effects=halo(2.2))
        ax.scatter(t[9::10, 0], t[9::10, 1], s=17, marker=MK[k], color=COL[k] if FILL[k] else "white",
                   edgecolors=COL[k], linewidths=1.0, zorder=12 + z * 0.1)
    C.draw_box(ax, 0, 0, 0, "none", zorder=16, ec=C.INK, lw=1.4)
    C.scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    if fit:
        ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_color(C.AXIS)


def artifact_note(T, i):
    """대표 사례 각주 — 상황 분류 인공물 후보."""
    r = T.iloc[i]
    if bool(r["turn_straight"]):
        return (f"※ 분류 인공물 후보: '{r['cls']}' 라벨이지만 위치로는 직진이다(시작·끝 방향 차 {r['turn_alt']:+.0f}°). "
                "정지 직전의 방향 잡음이 Δh 를 만든 경우다")
    if bool(r["turn_suspect"]):
        return f"※ 분류 인공물 후보: '{r['cls']}' 라벨이지만 6초 끝 횡변위가 {T['y_end'].iloc[i]:+.1f} m 로 거의 직진이다"
    if bool(r["dec_acc_border"]):
        return (f"※ 경계 사례: '급감속'(최소 a {r['amin_f']:+.2f} m/s²) 이지만 최대 a {r['amax_f']:+.2f} m/s² 로 급가속 임계도 넘는다")
    return ""


def case_figure(cx, i, heading, why, path, mode="top1"):
    plt = C.setup_mpl()
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    D = cx.D
    T = D.T
    r0 = T.iloc[i]
    fig = plt.figure(figsize=(22, 12.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 2.0], left=0.012, right=0.99, top=0.85, bottom=0.05,
                  wspace=0.06)
    left = gs[0, 0].subgridspec(3, 1, height_ratios=[2.3, 0.62, 0.95], hspace=0.08)
    axm = fig.add_subplot(left[0])
    draw_case_map(axm, cx, i, mode=mode, fit=True)
    axm.apply_aspect()                   # fit 시야로 줄어든 지도 상자의 실제 위치
    hs = [Line2D([], [], color=C.C_PAST, lw=2.2, marker="o", ms=3.5, mec="white"),
          Line2D([], [], color=C.C_GT, lw=2.4, marker="o", ms=4),
          Line2D([], [], color=C.C_PAST, lw=0, marker="o", ms=6, mfc="none"),
          Line2D([], [], color=C.INK2, lw=0.9, ls=(0, (1.0, 1.4)))]
    labs = ["과거 5초 (● 1초 간격)", "정답 6초 (● 1초 간격)", "2 Hz 판 입력 시점 (0.5 s)", "원점 → 첫 예측점 (t = 0.1 s)"]
    h2, l2 = run_handles()
    axl = fig.add_subplot(left[1])
    axl.axis("off")
    axl.legend(hs + [Line2D([], [], color=C.INK2, lw=1.3)] + h2,
               labs + ["정답 기준 경로 (청록 = 밴드)"] + [f"{l} · {'확률 1위' if mode == 'top1' else 'best of 6'} (● 1초)"
                                                    for l in l2],
               loc="center", fontsize=8.6, frameon=True, facecolor="white", edgecolor=C.GRID, ncol=2)
    # 표
    axt = fig.add_subplot(left[2])
    axt.axis("off")
    cols = ["판", "minADE6 / FDE6", "1위 ADE / FDE", "1위: 확률 · 경로", "best: 경로", "정답 경로 확률(순위)",
            "SL1 · CE · hinge"]
    cells, colors = [], []
    nd = int(r0["n_distinct"])
    g0 = int(r0["gt_route"])

    def ok_mark(m_):
        return "" if nd < 2 else (" ○" if m_ % nd == g0 else " ×")

    for k in ORDER:
        r = D.df[k].iloc[i]
        top, win = int(r["top1"]), int(r["winner"])
        gtp = (f"{float(r['gt_route_prob']):.2f} ({int(r['gt_route_rank'])}위)" if nd >= 2 else "— (분기 1)")
        cells.append([SHORT[k], f"{r['minade']:.2f} / {r['minfde']:.2f}", f"{r['top1_ade']:.2f} / {r['top1_fde']:.2f}",
                      f"{D.P[k]['prob'][i, top]:.2f} · r{top % max(nd, 1)}{ok_mark(top)}",
                      f"r{win % max(nd, 1)}{ok_mark(win)}" + (" (=1위)" if win == top else ""), gtp,
                      f"{r['l_sl1']:.2f} · {r['l_ce']:.2f} · {r['l_hinge']:.2f}"])
        colors.append(COL[k])
    tb = axt.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[0.09, 0.15, 0.15, 0.155, 0.14, 0.155, 0.16])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.3)
    tb.scale(1, 1.5)
    for (rr, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(C.GRID)
        if rr == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(color=C.INK2, weight="bold")
        elif cc == 0:
            cell.set_facecolor(colors[rr - 1])
            cell.set_text_props(color="white", weight="bold")
    axt.text(0.0, 0.02, "○/× = 그 모드(슬롯)의 후보 경로가 정답 기준 경로인가 · 정답 경로 확률 = 그 경로에 배정된 슬롯 확률의 합, "
                        "순위 = 경로 확률 순위\n(두 기준이 다를 수 있다: 1위 모드가 옆 경로여도 정답 경로의 슬롯 합이 1위일 수 있다) · "
                        "손실 = 학습식(train_v4.loss_fn)을 이 시나리오에만 적용",
             transform=axt.transAxes, fontsize=7.4, color=C.INK2, va="bottom", linespacing=1.45)
    # 지도가 가로로 길어 위로 줄었으면 범례·표를 지도 바로 아래로 올린다 (빈칸 줄이기)
    pm, pl, pt = axm.get_position(), axl.get_position(), axt.get_position()
    if pm.y0 > pl.y1 + 0.02:
        top = pm.y0 - 0.012
        axl.set_position([pl.x0, top - pl.height, pl.width, pl.height])
        axt.set_position([pt.x0, top - pl.height - 0.012 - pt.height, pt.width, pt.height])
    # 시계열
    s_g, d_g, bd_g, k_g = cx.gt_route_frame(i)
    lc = C.lc_interval(D.R["gt_d_g"][i], r0["cls"])
    tax, tp = C.T_AX, C.T_PRED
    R = D.R
    hg = R["h"][i].astype(np.float64)
    hg = hg - 2 * np.pi * np.round(hg[C.OBS - 1] / (2 * np.pi))
    series = [("v", "속도 v [m/s] — 정답: 검정 위치 차분 · 회색 점선 AV2 속도 필드(평활) · 회색 띠 = 창 양 끝 램프"),
              ("a", "가속도 a [m/s²] — 정답(검정) = AV2 속도 필드 평활의 변화율"),
              ("h", "진행방향 h [°]"), ("theta", "잔차각 θ [°] (각 판은 자기 모드 경로 기준, 정답은 정답 경로 기준)"),
              ("d", "정답 경로 기준 횡오프셋 d [m] (+좌) — 예측도 정답 경로에 투영"), ("lead", "앞차 거리 [m]")]
    sub = gs[0, 1].subgridspec(3, 2, hspace=0.42, wspace=0.16)      # 3행 × 2열
    ga = int(r0["gt_route"])
    for q, (key, ylab) in enumerate(series):
        ax = fig.add_subplot(sub[q // 2, q % 2])
        C.shade_lc(ax, lc, label=(key == "d"))
        ax.axvline(0, color=C.INK2, lw=0.8, zorder=1)
        if key in ("v", "a"):
            ax.axvspan(5.45, 6.0, color=C.GRID, alpha=0.8, lw=0, zorder=0)
            ax.axvspan(-4.9, -4.45, color=C.GRID, alpha=0.8, lw=0, zorder=0)
        vals = []
        if key == "v":
            ax.plot(tax, R["v_pos"][i], color=C.C_GT, lw=1.6, zorder=5, **C.step_kw())
            ax.plot(tax, R["v_fld"][i], color=C.MUTED, lw=1.1, ls=(0, (3, 2)), zorder=5)
        elif key == "a":
            ax.plot(tax, R["a_fld"][i], color=C.C_GT, lw=1.6, zorder=5, **C.step_kw())
            vals.append(R["a_fld"][i])
        elif key == "h":
            ax.plot(tax, np.degrees(hg), color=C.C_GT, lw=1.6, zorder=5, **C.step_kw())
            vals.append(np.degrees(hg[5:]))       # 범위는 창 시작 0.5 s(램프·저속 방향 잡음)를 빼고 잡는다
        elif key == "theta":
            th = np.degrees(C.wrap(hg - k_g))
            th[R["v_fld"][i] < C.MOVE_V] = np.nan
            ax.plot(tax, th, color=C.C_GT, lw=1.3, zorder=5, **C.step_kw())
        elif key == "d":
            ax.plot(tax, bd_g[:, 0], color=C.C_BAND, lw=1.0, ls=(0, (4, 2)))
            ax.plot(tax, -bd_g[:, 1], color=C.C_BAND, lw=1.0, ls=(0, (4, 2)))
            ax.axhline(0, color=C.AXIS, lw=0.8)
            ax.plot(tax, d_g, color=C.C_GT, lw=1.6, zorder=5, **C.step_kw())
            vals += [d_g[C.OBS - 10:], bd_g[C.OBS:, 0], -bd_g[C.OBS:, 1]]
        elif key == "lead":
            ld = R["lead_dist"][i]
            if np.isfinite(ld).any():
                ax.plot(tax, ld, color=C.C_LEAD, lw=1.6, **C.step_kw())
                ax.set_ylim(0, min(C.LEAD_MAX_M, np.nanmax(ld) * 1.2 + 2))
            else:
                ax.text(0.5, 0.5, "앞차 없음", transform=ax.transAxes, ha="center", va="center", color=C.MUTED)
        if key not in ("lead",):
            for k in ORDER:
                r = D.df[k].iloc[i]
                m = int(r["top1"] if mode == "top1" else r["winner"])
                P = D.P[k]
                if key == "v":
                    ys = P["v"][i, m]
                elif key == "a":
                    ys = P["a"][i, m]
                elif key == "h":
                    hh = np.unwrap(P["h"][i, m].astype(np.float64))
                    ys = np.degrees(hh - 2 * np.pi * np.round((hh[0] - hg[C.OBS - 1]) / (2 * np.pi)))
                elif key == "theta":
                    ys = np.degrees(P["theta"][i, m])
                else:
                    # 모델 d 는 그 모드가 탄 경로 기준이라, 정답 경로 기준으로 보려면 궤적을 정답 경로에 투영한다
                    ys = proj_gt_route(D, [i], [P["traj"][i, m]])[0]
                    if (m % max(nd, 1)) == ga:       # 같은 경로면 모델 d 와 같아야 한다 (교차 검증)
                        PROJ_CHECK.append(float(np.abs(ys - P["d"][i, m].astype(np.float64)).max()))
                ax.plot(tp, ys, color=COL[k], ls=LS[k], lw=1.7, zorder=4, **C.step_kw())
                vals.append(np.asarray(ys, float))
        if key == "a":
            lim = max(2.0, min(9.0, 1.15 * float(np.nanmax(np.abs(np.concatenate(vals))))))
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color=C.AXIS, lw=0.8)
        if key == "d" and vals:
            yy = np.concatenate(vals)
            ax.set_ylim(max(-12, np.nanmin(yy) - 0.8), min(12, np.nanmax(yy) + 0.8))
        if key == "h" and vals:
            yy = np.concatenate(vals)
            lo_, hi_ = float(np.nanmin(yy)), float(np.nanmax(yy))
            pad = max(2.0, 0.08 * (hi_ - lo_))
            ax.set_ylim(lo_ - pad, hi_ + pad)
            h_all = np.degrees(hg[:5])
            if np.any((h_all < lo_ - pad) | (h_all > hi_ + pad)):
                ax.text(0.01, 0.03, "창 시작 0.5 s 의 정답 값 일부가 범위 밖(잘림)", transform=ax.transAxes, fontsize=7.2,
                        color=C.INK2, va="bottom")
        ax.set_xlim(-5, 6)
        ax.set_title(ylab, loc="left", fontsize=9.2)
        if q >= 4:
            ax.set_xlabel("시간 [s] (0 = 예측 시작) · 점 = 0.1 s 스텝")
    fig.suptitle(heading, x=0.012, y=0.985, ha="left", fontsize=13.5)
    lines = [f"{r0['sid']} · 상황 {r0['cls']} · 6초 Δh {r0['dh6']:+.0f}° · Δd {r0['dd6']:+.1f} m · v0 {r0['v0']:.1f} m/s · "
             f"a 최소/최대 {r0['amin_f']:+.1f}/{r0['amax_f']:+.1f} m/s² · 구별 분기 {nd} · 정답 경로 r{int(r0['gt_route'])} · "
             f"앞차 {'없음' if not np.isfinite(r0['gap49']) else format(r0['gap49'], '.0f') + ' m'}",
             why]
    note_ = artifact_note(T, i)
    if note_:
        lines.append(note_)
    fig.text(0.012, 0.935, "\n".join(lines), fontsize=9.6, color=C.INK2, va="top", linespacing=1.55)
    C.savefig(fig, path)


def sec_f(D, M):
    cx = CaseCtx(D)
    T = D.T
    picks = {}
    # f0 고정 시나리오 14개 (작업 B 와 같은 ID) — 네 판의 확률 1위
    pk = json.loads((C.RUNS / "v4_epochs_v4_l4nw_ah2_2hz_full_sm1_s0_picks.json").read_text())["picks"]
    sid2i = {s: j for j, s in enumerate(T["sid"])}
    idx14 = [sid2i[p["sid"]] for p in pk]
    fixed_overview(cx, pk, idx14, OUT / "f0_fixed14_4runs_top1.png")
    SUM["f_fixed14"] = {p["sid"]: {k: {"minade": float(D.df[k]["minade"].iloc[j]),
                                       "top1_ade": float(D.df[k]["top1_ade"].iloc[j]),
                                       "route_err": bool(D.df[k]["route_err"].iloc[j])} for k in ORDER}
                        for p, j in zip(pk, idx14)}
    def near_median(mask, score, name, q=50):
        """mask 안에서 score 가 그 모집단 q 백분위(기본 중앙값)에 가장 가까운 시나리오 (폴백 제외)."""
        m = mask & ~T["fallback"].to_numpy()
        pool = np.where(m)[0]
        s = score[pool]
        ref = float(np.percentile(s, q))
        j = pool[np.argsort(np.abs(s - ref), kind="mergesort")[0]]
        picks[name] = {"idx": int(j), "sid": str(T["sid"].iloc[j]), "pool": int(len(pool)),
                       "score": float(score[j]), "pool_median": float(np.median(s)), "q": q, "pool_q": ref}
        return int(j)

    cls = T["cls"].to_numpy()
    d_sched10 = M["h10_c30"]["minfde"] - M["h10_c15"]["minfde"]
    d_in = M["h2_c30"]["minfde"] - M["h10_c30"]["minfde"]
    # F1: 30 코사인이 회전을 개선 — 좌·우회전 중 10 Hz minFDE6 이 1 m 넘게 좋아진 시나리오의 중앙 개선
    sus = T["turn_suspect"].to_numpy()
    base1 = np.isin(cls, ["좌회전", "우회전"]) & ~sus & (d_sched10 < -1.0) & ~T["fallback"].to_numpy()
    # 모집단의 다수(커버리지 성공 · 구별 분기 ≥ 2)에서 고른다 — 소수 집단(경로 밖 회전 등)이 대표가 되지 않게
    major1 = base1 & T["coverage"].to_numpy().astype(bool) & (T["n_distinct"].to_numpy() >= 2)
    j = near_median(major1, d_sched10, "f1_turn_sched")
    picks["f1_turn_sched"] |= {"pool_all": int(base1.sum()), "pool_all_median": float(np.median(d_sched10[base1]))}
    p1 = picks["f1_turn_sched"]
    case_figure(cx, j, "f1 · 더 학습한 효과 (회전) — 10 Hz 15 → 30 에서 minFDE6 가 1 m 넘게 좋아진 회전의 중앙 사례",
                f"선정: 좌·우회전(분류 인공물 후보·폴백 제외) 중 10Hz·30 − 10Hz·15 minFDE6 < −1 m 인 {p1['pool_all']:,}개(중앙값 "
                f"{p1['pool_all_median']:+.2f} m) → 그중 다수인 커버리지 성공·구별 분기 ≥ 2 {p1['pool']:,}개"
                f"({100 * p1['pool'] / p1['pool_all']:.0f}%)에서 그 차가 중앙값({p1['pool_median']:+.2f} m)에 가장 가까운 시나리오 · "
                "그림 = best of 6 (선정 기준인 minFDE6 의 모드, 확률 1위는 표)",
                OUT / "f1_turn_sched_gain.png", mode="best")
    # F2: 분기 선택이 틀렸다가 맞게 — 평균이 좋아진 2 Hz 에서 (b2: 정답 경로 1위 52.6 → 56.3%).
    # '정답 경로 = 경로 확률 1위'(슬롯 합)와 '1위 모드의 경로 = 정답'(슬롯 하나) 두 정의가 모두 바뀐 시나리오만 쓴다.
    re15, re30 = M["h2_c15"]["route_err"], M["h2_c30"]["route_err"]
    g15, g30 = M["h2_c15"]["gt_top1"], M["h2_c30"]["gt_top1"]
    m2 = (g15 == 0) & (re15 == 1) & (g30 == 1) & (re30 == 0)
    fl = {"h10": {}, "h2": {}}
    for inp in ("h10", "h2"):
        a_, b_ = M[f"{inp}_c15"], M[f"{inp}_c30"]
        both_ok = lambda mm, g_: (mm["gt_top1"] == g_) & (mm["route_err"] == (1 - g_))
        fl[inp] = {"wrong_to_right": int((both_ok(a_, 0) & both_ok(b_, 1)).sum()),
                   "right_to_wrong": int((both_ok(a_, 1) & both_ok(b_, 0)).sum()),
                   "n_nd2": int((T["n_distinct"] >= 2).sum())}
    SUM["f_route_flips"] = fl
    j = near_median(m2, M["h2_c30"]["top1_fde"], "f2_route_fixed")
    case_figure(cx, j, "f2 · 더 학습한 효과 (분기 선택, 2 Hz) — 15 에서 틀린 분기를 1위로 골랐다가 30 에서 정답 분기를 1위로 고른 사례",
                f"선정: 2 Hz 에서 두 기준(경로 확률 1위 · 1위 모드의 경로)이 모두 틀림 → 맞음으로 바뀐 {picks['f2_route_fixed']['pool']:,}개"
                f"(구별 분기 ≥ 2, 반대로 바뀐 것 {fl['h2']['right_to_wrong']:,}개)에서 2Hz·30 확률 1위 FDE 가 중앙값에 가장 가까운 것 · "
                "그림 = 확률 1위 모드", OUT / "f2_route_fixed.png", mode="top1")
    # F3: 2 Hz 가 크게 진 사례 (트리의 가장 나쁜 잎에서)
    lv = SUM.get("c_tree", {}).get("leaves")
    if lv:
        worst = max(lv, key=lambda z: z["mean"])
        best = min(lv, key=lambda z: z["mean"])
        for tagname, leaf, head in (("f3_2hz_loses", worst, "2 Hz 가 가장 불리한 잎"), ("f4_2hz_wins", best, "2 Hz 가 가장 유리한 잎")):
            j = leaf["rep_idx"]
            ci0 = abs(leaf["mean_test"]) <= leaf["ci95_test"]
            rep = ("잎 평균 부호가 같음" if leaf["same_sign_15const"] else "잎 평균 부호 반대") + \
                  (" · 검증 30% 신뢰구간이 0 을 포함하는 후보" if ci0 else " · 검증 30% 신뢰구간이 0 을 넘음")
            picks[tagname] = {"idx": int(j), "sid": leaf["rep_sid"], "leaf_rule": leaf["rule"], "leaf_n": leaf["n"],
                              "leaf_mean": leaf["mean"], "leaf_mean_test": leaf["mean_test"],
                              "leaf_mean_15const": leaf["mean_15const"], "score": float(d_in[j]),
                              "score_15const": float(T["dF15"].iloc[j]), "leaf_ci95_test": leaf["ci95_test"],
                              "leaf_test_ci_has_zero": bool(ci0)}
            case_figure(cx, j, f"{tagname[:2]} · 10 Hz vs 2 Hz — 결정 트리에서 {head}의 대표 시나리오 (잎 중앙값 Δ 에 가장 가까움) · "
                               f"15에폭 쌍에서 {rep}",
                        f"잎 규칙: {leaf['rule']} · n={leaf['n']:,} · 잎 평균 ΔminFDE6 {leaf['mean']:+.2f} m (검증 30% "
                        f"{leaf['mean_test']:+.2f} ± {leaf['ci95_test']:.2f}, 15에폭 쌍 {leaf['mean_15const']:+.2f})\n"
                        f"이 시나리오 Δ {float(d_in[j]):+.2f} m "
                        f"(15에폭 쌍 {float(T['dF15'].iloc[j]):+.2f}) · 그림 = best of 6",
                        OUT / f"{tagname}.png", mode="best")
    # F5: 끝 감속 추종 — 고속 이동에서 정답 끝 감속과 모델 끝 감속
    v = D.R["v_fld"][:, C.OBS + 40:C.OBS + 50].mean(1)
    m5 = (cls == "정속") & (v > 15)
    j = near_median(m5, M["h10_c30"]["minfde"], "f5_end_decel")
    case_figure(cx, j, "f5 · 라벨 끝 인공 감속 — 고속 정속 주행에서 정답(위치)과 네 판의 끝 1초",
                f"선정: 정속 · 4.1–5.0 s 속도 필드 > 15 m/s 인 {picks['f5_end_decel']['pool']:,}개에서 10Hz·30 minFDE6 가 중앙값에 "
                "가장 가까운 것 · 그림 = 확률 1위 모드 · 회색 띠 = 정답 끝 0.5초", OUT / "f5_end_decel.png", mode="top1")
    # F6: 고속에서 10 Hz·30 이 나란한 옆 차로를 1위로 고른 사례 (15 는 맞췄음)
    e15, _, _, _ = route_nature(D, "h10_c15")
    e30, p30, _, _ = route_nature(D, "h10_c30")
    m6 = (T["v0"].to_numpy() >= 10) & ~e15 & p30
    j = near_median(m6, M["h10_c30"]["top1_fde"], "f6_parallel_lane")
    case_figure(cx, j, "f6 · 고속 분기 선택 — 10 Hz·15 는 정답 차로를 1위로, 10 Hz·30 은 나란한 옆 차로를 1위로 고른 사례",
                f"선정: v0 ≥ 10 m/s · 그런 {picks['f6_parallel_lane']['pool']:,}개에서 10Hz·30 확률 1위 FDE 가 중앙값에 가장 가까운 것 · "
                "그림 = 확률 1위 모드", OUT / "f6_parallel_lane.png", mode="top1")
    # F7: 30에폭 판의 1~2 s 주기 가속도 출렁임 (d1 ③ 의 p90 꼬리) — 정답 가속이 매끈한 정속 주행에서
    #     10Hz·30 과 2Hz·30 이 모두 그 판 분포의 p90 근처인 사례 (백분위 순위 |r10 − 90| + |r2 − 90| 최소)
    ar = np.arange(D.N)
    osc = {k: osc_rms(D.P[k]["a"][ar, D.df[k]["top1"].to_numpy()]) for k in ORDER}
    osc_sm = {k: osc_rms(smooth_like_gt(D.R["v_fld"][:, :C.OBS],
                                         D.P[k]["v"][ar, D.df[k]["top1"].to_numpy()].astype(np.float64))) for k in ORDER}
    osc_gt = osc_rms(D.R["a_fld"][:, C.OBS:].astype(np.float64))
    base7 = (cls == "정속") & (T["v0"].to_numpy() >= 5)
    pool7 = base7 & (osc_gt <= np.median(osc_gt[base7])) & ~T["fallback"].to_numpy()
    idx7 = np.flatnonzero(pool7)

    def prank(v):
        o = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
        return 100.0 * (o + 0.5) / len(v)

    r10, r2 = prank(osc["h10_c30"][idx7]), prank(osc["h2_c30"][idx7])
    cost = np.abs(r10 - 90) + np.abs(r2 - 90)
    q7 = int(np.argsort(cost, kind="mergesort")[0])
    j = int(idx7[q7])
    picks["f7_accel_wobble"] = {
        "idx": j, "sid": str(T["sid"].iloc[j]), "pool": int(len(idx7)),
        "rule": "정속 · v0 ≥ 5 · 정답 출렁임 ≤ 중앙 · 폴백 제외에서 |순위(10Hz·30) − 90| + |순위(2Hz·30) − 90| 최소",
        "pct_rank_this": {"h10_c30": float(r10[q7]), "h2_c30": float(r2[q7])},
        "osc_this": {k: float(osc[k][j]) for k in ORDER} | {"gt_field": float(osc_gt[j])},
        "osc_sm_this": {k: float(osc_sm[k][j]) for k in ORDER},
        "osc_pool_median": {k: float(np.median(osc[k][pool7])) for k in ORDER},
        "osc_pool_p90": {k: float(np.percentile(osc[k][pool7], 90)) for k in ORDER}}
    pk7 = picks["f7_accel_wobble"]
    case_figure(cx, j, "f7 · 30에폭 판의 1~2 s 주기 가속도 출렁임 (p90 꼬리) — 정답 가속은 매끈한데 확률 1위 모드의 a(t) 가 오르내린다 (d1 ③)",
                f"선정: 정속 · v0 ≥ 5 m/s · 정답(속도 필드) 출렁임이 중앙 이하인 {pk7['pool']:,}개에서 두 30에폭 판이 모두 p90 근처인 것 "
                f"(순위 10Hz·30 {pk7['pct_rank_this']['h10_c30']:.0f} · 2Hz·30 {pk7['pct_rank_this']['h2_c30']:.0f} 백분위)\n"
                "출렁임(원 출력) [m/s²]: " + " · ".join(f"{SHORT[k]} {osc[k][j]:.2f}" for k in ORDER)
                + " (모집단 p90 " + " · ".join(f"{pk7['osc_pool_p90'][k]:.2f}" for k in ORDER) + ") · "
                f"정답과 같은 평활: 10Hz·30 {osc_sm['h10_c30'][j]:.2f} · 2Hz·30 {osc_sm['h2_c30'][j]:.2f} · 정답 {osc_gt[j]:.2f} · "
                "그림 = 확률 1위 모드", OUT / "f7_accel_wobble.png", mode="top1")
    # F8: 차선변경 — 더 학습해도 평균이 좋아지지 않은 상황(b1)의 중앙 사례, 차선변경 구간(노란 띠)이 보이는 것
    lc_ok = np.array([C.lc_interval(D.R["gt_d_g"][q], c) is not None if c in ("좌차선변경", "우차선변경") else False
                      for q, c in enumerate(cls)])
    m8 = lc_ok
    j = near_median(m8, d_sched10, "f8_lane_change")
    dl = {c: paired(M["h10_c15"]["minfde"][cls == c], M["h10_c30"]["minfde"][cls == c]) for c in ("좌차선변경", "우차선변경")}
    picks["f8_lane_change"]["class_delta_10hz"] = {c: {"mean": v[0], "ci95": v[1], "n": v[2]} for c, v in dl.items()}
    case_figure(cx, j, "f8 · 차선변경 — 더 학습해도 평균이 좋아지지 않은 상황의 중앙 사례 "
                       f"(10 Hz 30 − 15 ΔminFDE6: 좌 {dl['좌차선변경'][0]:+.2f} ± {dl['좌차선변경'][1]:.2f} · "
                       f"우 {dl['우차선변경'][0]:+.2f} ± {dl['우차선변경'][1]:.2f} m)",
                f"선정: 좌·우 차선변경 중 차선변경 구간이 잡히는 {picks['f8_lane_change']['pool']:,}개(폴백 제외)에서 "
                f"10Hz·30 − 10Hz·15 minFDE6 가 중앙값({picks['f8_lane_change']['pool_median']:+.2f} m)에 가장 가까운 것 · "
                "노란 띠 = 차선변경 구간(정답 경로 d 변화 10–90%) · 그림 = 확률 1위 모드",
                OUT / "f8_lane_change.png", mode="top1")
    SUM["f_picks"] = picks
    SUM["f_proj_check"] = {"n": len(PROJ_CHECK), "max_abs_m": float(max(PROJ_CHECK)) if PROJ_CHECK else None,
                           "median_abs_m": float(np.median(PROJ_CHECK)) if PROJ_CHECK else None,
                           "what": "f 그림 d 칸: 정답 경로 위 모드의 투영 d(lane_frame.to_frame) vs 모델 출력 d 의 최대 차"}
    print(f"[f] 대표 시나리오 {len(picks)}개 + 고정 14개 · 투영 d 대조 {SUM['f_proj_check']}", flush=True)


def fixed_overview(cx, pk, idx14, path):
    plt = C.setup_mpl()
    from matplotlib.lines import Line2D
    D = cx.D
    ncol = 4
    nrow = int(np.ceil((len(pk) + 1) / ncol))
    H = 5.3 * nrow + 1.5
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, H), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.45 / H, bottom=0.01, wspace=0.05, hspace=0.3)
    for ax in axes.flat[len(pk):]:
        ax.axis("off")
    for ax, p, i in zip(axes.flat, pk, idx14):
        draw_case_map(ax, cx, i, mode="top1", compact=True)
        vals = " · ".join(f"{D.df[k]['top1_ade'].iloc[i]:.1f}" for k in ORDER)
        mark = " ※" if artifact_note(D.T, i) else ""
        ax.set_title(f"{p['cls']} {p['k']} · {p['sid'][:8]}{mark}\n1위 ADE {vals} m", fontsize=9.3)
    ax = axes.flat[len(pk)]
    hs, ls = run_handles()
    ax.legend([Line2D([], [], color=C.C_PAST, lw=2.2, marker="o", ms=3.5, mec="white"),
               Line2D([], [], color=C.C_PAST, lw=0, marker="o", ms=6, mfc="none"),
               Line2D([], [], color=C.C_GT, lw=2.4, marker="o", ms=4),
               Line2D([], [], color=C.INK2, lw=0.9, ls=(0, (1.0, 1.4)))] + hs,
              ["과거 5초 (● 1초)", "2 Hz 판 입력 시점 (0.5 s)", "정답 6초 (● 1초 간격)", "원점 → 첫 예측점 (t = 0.1 s)"]
              + [f"{l} · 확률 1위" for l in ls], loc="upper center", fontsize=8.8,
              frameon=True, facecolor="white", edgecolor=C.GRID)
    notes = [f"{p['sid'][:8]}: {artifact_note(D.T, i)[2:]}" for p, i in zip(pk, idx14) if artifact_note(D.T, i)]
    ax.text(0.5, 0.02, "칸 제목 숫자 = 1위 ADE [m]:\n" + " · ".join(SHORT[k] for k in ORDER)
            + ("\n\n※ " + "\n※ ".join(textwrap.fill(n_, 46) for n_ in notes) if notes else ""),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=C.INK2)
    fig.suptitle("f0 · 고정 시나리오 14개 (작업 B 와 같은 ID) — 네 판의 확률 1위 모드 (best 체크포인트)", x=0.01,
                 y=1 - 0.25 / H, ha="left", fontsize=14)
    C.savefig(fig, path)


# =========================================================================== v 교차 검증
def sec_v(D, M, H=None):
    """리포트 숫자를 다른 코드 경로로 다시 계산해 대조한다 (필수 요건 9b)."""
    out = {}
    cmp_json = json.loads((C.RUNS / "v4_full_compare.json").read_text())["runs"]
    for k in ORDER:
        P = D.P[k]
        y = D.R["y"].astype(np.float64)
        al = P["alive"]
        dist = np.linalg.norm(P["traj"].astype(np.float64) - y[:, None], axis=-1)
        ade = np.where(al, dist.mean(2), np.inf).min(1)
        fde = np.where(al, dist[:, :, -1], np.inf).min(1)
        exp = cmp_json[TAG[k]]["val24988"]
        # 저크 RMS 두 경로: a 직접 차분 vs 흔들림 a 에서 환산 (80·√jit_a)
        feas = SUM.get("d_feas", {}).get(k, {})
        out[k] = {"minade_recalc": float(ade.mean()), "minade_table": float(D.df[k]["minade"].mean()),
                  "minade_compare_json": exp["minADE6"], "minfde_recalc": float(fde.mean()),
                  "minfde_compare_json": exp["minFDE6"],
                  "miss_recalc_pct": float((fde > C.MISS_M).mean() * 100),
                  "jerk_rms_from_jit": float(80.0 * np.sqrt(exp["jitter_a"])),
                  "jerk_rms_direct": feas.get("jerk_rms_all"),
                  "yawacc_rms_from_jit": float(800.0 * np.sqrt(exp["jitter_theta"])),
                  "yawacc_rms_direct": feas.get("yawacc_rms_all"),
                  "exc_compare_json": exp["dtheta_over_label_pct"], "exc_direct": feas.get("exc_pct")}
    # 상황별 평균 두 경로 (pandas groupby vs numpy bincount)
    cls_codes = {c: j for j, c in enumerate(C.CLASSES)}
    code = np.array([cls_codes[c] for c in D.T["cls"]])
    grp = {}
    for k in ("h10_c30", "h2_c30"):
        gb = D.df[k].groupby("cls")["minfde"].mean()
        bc = np.bincount(code, weights=D.df[k]["minfde"].to_numpy(np.float64)) / np.bincount(code)
        grp[k] = max(abs(gb[c] - bc[cls_codes[c]]) for c in C.CLASSES)
    out["class_mean_groupby_vs_bincount_maxdiff"] = grp
    # 조건별 Δ 표의 한 칸을 다시 계산 (pandas 경로)
    import pandas as pd
    dd = pd.DataFrame({"cls": D.T["cls"], "d": M["h2_c30"]["minfde"] - M["h10_c30"]["minfde"]})
    re_ = dd.groupby("cls")["d"].mean()
    tab = SUM.get("c_delta", {}).get("30 코사인: 2 Hz − 10 Hz", {}).get("minfde", {}).get("cls", {})
    out["c_delta_cls_maxdiff"] = max(abs(re_[c] - tab[c]["delta"]) for c in tab) if tab else None
    # 학습 곡선 요동: statistics.stdev 로 다시
    fl = SUM.get("a_fluct", {})
    rr = {}
    for k in ORDER:
        h = json.loads((C.RUNS / f"{TAG[k]}.json").read_text())["history"]
        v = [x["minADE6"] for x in h][-5:]
        rr[k] = abs(statistics.stdev(v) - fl[k]["minADE6"]["last5_std"]) if fl else None
    out["fluct_stdev_diff"] = rr
    # 코사인 학습률 식 (에폭 끝 lr) — 스텝 수 = ceil(199908/32)
    steps = int(np.ceil(199908 / 32))
    for k in ("h10_c30", "h2_c30"):
        h = json.loads((C.RUNS / f"{TAG[k]}.json").read_text())["history"]
        pred = [1e-5 + (5e-4 - 1e-5) * (1 + np.cos(np.pi * e * steps / (30 * steps))) / 2 for e in range(1, 31)]
        out[f"lr_formula_maxdiff_{k}"] = float(max(abs(a - x["lr"]) for a, x in zip(pred, h)))
    # 정답 경로 투영 d vs 모델 d — 1위 모드가 정답 경로 위인 시나리오 무작위 400개 (b5·f 그림의 투영이 맞는가)
    rng = np.random.default_rng(C.SEED)
    k = REF
    top = D.df[k]["top1"].to_numpy()
    on = np.flatnonzero((D.df[k]["top1_route"].to_numpy() == D.T["gt_route"].to_numpy()) & ~D.T["fallback"].to_numpy())
    pick = np.sort(rng.choice(on, min(400, len(on)), replace=False))
    dp = proj_gt_route(D, pick, D.P[k]["traj"][pick, top[pick]])
    dm = D.P[k]["d"][pick, top[pick]].astype(np.float64)
    ad = np.abs(dp - dm)
    out["proj_vs_model_d"] = {"run": k, "n": int(len(pick)), "max_abs_m": float(ad.max()),
                              "p99_abs_m": float(np.percentile(ad, 99)), "median_abs_m": float(np.median(ad)),
                              "end_max_abs_m": float(ad[:, -1].max()),
                              "end_share_lt_same_lane": float((ad[:, -1] < SAME_LANE_M).mean())}
    # 출렁임 지표: 누적합 경로(osc_rms) vs np.convolve 경로
    fe = SUM.get("d_feas", {})
    if fe:
        ker = np.ones(OSC_WIN) / OSC_WIN
        oc = {}
        for k in ORDER:
            a_ = np.asarray(fe[k]["top_a_mean_t"], np.float64)
            ma = np.convolve(a_, ker, mode="same")
            oc[k] = abs(float(np.sqrt(((a_ - ma)[OSC_K0:OSC_K1] ** 2).mean())) - fe[k]["osc_mean_curve_rms"])
        out["osc_convolve_vs_cumsum_absdiff"] = oc
    # 정답과 같은 평활: 축 한 번(smooth_like_gt) vs viz_v4_dump._smooth_speed 를 행마다 (무작위 300개, 10Hz·30 1위)
    import viz_v4_dump as VD
    k = REF
    top = D.df[k]["top1"].to_numpy()
    q = np.sort(rng.choice(D.N, 300, replace=False))
    vt = D.P[k]["v"][q, top[q]].astype(np.float64)
    a_vec = smooth_like_gt(D.R["v_fld"][q, :C.OBS], vt)
    a_row = np.stack([VD._smooth_speed(np.concatenate([D.R["v_fld"][i_, :C.OBS].astype(np.float64), vt[r_]]))[1][C.OBS:]
                      for r_, i_ in enumerate(q)])
    out["smooth_like_gt_vec_vs_row_maxdiff"] = float(np.abs(a_vec - a_row).max())
    # 정답 a_fld 가 같은 함수(원 속도 필드 -> _smooth_speed)의 결과인지는 덤프가 보장한다 — 같은 osc 로 p90 을 다시 잰다
    if fe and "end_pop" in D.T:
        # 같은 모집단(끝 감속 모집단)에서 행마다 np.convolve 로 다시 잰 정답 p90 (d1 값과 같아야 한다)
        ker = np.ones(OSC_WIN) / OSC_WIN
        rows_ = np.flatnonzero(D.T["end_pop"].to_numpy())
        vals_ = []
        for i_ in rows_:
            a_ = D.R["a_fld"][i_, C.OBS:].astype(np.float64)
            vals_.append(np.sqrt(((a_ - np.convolve(a_, ker, mode="same"))[OSC_K0:OSC_K1] ** 2).mean()))
        out["osc_gt_p90_recalc"] = {"n": int(len(rows_)), "p90_convolve": float(np.percentile(vals_, 90)),
                                    "p90_d1": fe["osc_gt_field"]["scen_p90"],
                                    "absdiff": abs(float(np.percentile(vals_, 90)) - fe["osc_gt_field"]["scen_p90"])}
    SUM["verify"] = out
    for k in ORDER:
        o = out[k]
        print(f"[v] {k}: minADE6 재계산 {o['minade_recalc']:.6f} · 표 {o['minade_table']:.6f} · json "
              f"{o['minade_compare_json']:.6f} | 저크 RMS 직접 {o['jerk_rms_direct']} · 환산 {o['jerk_rms_from_jit']:.4f} | "
              f"각가속 직접 {o['yawacc_rms_direct']} · 환산 {o['yawacc_rms_from_jit']:.2f}", flush=True)
    print(f"[v] 상황 평균 경로 차 {grp} · Δ 표 대조 {out['c_delta_cls_maxdiff']} · 요동 {rr} · "
          f"lr 식 {out['lr_formula_maxdiff_h10_c30']:.2e}", flush=True)
    print(f"[v] 투영 d vs 모델 d {out['proj_vs_model_d']} · 출렁임 두 경로 {out.get('osc_convolve_vs_cumsum_absdiff')}",
          flush=True)


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b,c,d,e,f,v")
    a = ap.parse_args()
    todo = a.only.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    sp = DATA / "summary.json"
    if sp.exists():
        SUM.update(json.loads(sp.read_text()))
    t0 = time.time()
    if "a" in todo:
        sec_a()
    need = [s for s in todo if s in ("b", "c", "d", "e", "f", "v")]
    if need:
        D = Data()
        M = {k: run_metrics(D, k) for k in ORDER}
        if "b" in todo:
            sec_b(D, M)
        if "c" in todo:
            sec_c(D, M)
        if "d" in todo:
            sec_d(D, M)
        if "e" in todo:
            sec_e(D)
        if "f" in todo:
            if "c" not in todo:
                D.T["dF"] = M["h2_c30"]["minfde"] - M["h10_c30"]["minfde"]
                D.T["dF15"] = M["h2_c15"]["minfde"] - M["h10_c15"]["minfde"]
            sec_f(D, M)
        if "v" in todo:
            sec_v(D, M)
    SUM["runs"] = {k: {"tag": TAG[k], "label": LAB[k]} for k in ORDER}
    SUM["made"] = time.strftime("%Y-%m-%d %H:%M:%S")
    SUM["git_head"] = C.git_head()
    sp.write_text(json.dumps(SUM, indent=2, ensure_ascii=False, default=_json))
    print(f"[done] {sp}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
