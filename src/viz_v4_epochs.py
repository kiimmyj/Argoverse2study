"""
viz_v4_epochs.py - 에폭별 체크포인트로 '모델이 어느 방향으로 학습되고 있는지' 본다 (표준 작업 B).

입력   runs/ckpt/<tag>/epNN.pth (train_v4 --save-every, 2026-09-17 부터)
       runs/<tag>.json — 없으면(학습 중) 로그의 cmd 줄과 에폭 줄을 읽고, --input/--th0 로 덮을 수 있다
평가   그 판의 --input 에 맞는 val 캐시(viz_v4_common.VAL_CACHES)의 앞 N 개 (기본 2,000)

계산
  1. 기준표 (에폭과 무관, 한 번만) — viz_v4_dump.raw_one 을 그대로 쓴다: 상황 분류, 정답 기준 경로,
     110스텝 위치·속도(위치 차분 / AV2 속도 필드 평활)·가속도. data/epoch_base_n<N>.{parquet,npz}.
     같은 조건으로 더 큰 N 을 만든 것이 있으면 앞부분을 잘라 쓴다.
  2. 에폭마다 추론 — viz_v4_dump.run_inference(첫 배치 손실 분해 대조 포함)·derived·score_like_compare.
     시나리오별 값은 data/epoch_scen_n<N>.npz 에 모으고, 체크포인트·장치·코드가 같은 에폭은 다시 계산하지 않는다.
  3. N 이 학습 로그의 val 크기와 같으면 로그 history 와 대조한다 (재현 확인).
  숫자 요약은 data/epochs_n<N>.json.

그림 (viz/v4/<tag>/epochs/ — N 이 2,000 이 아니면 epochs_n<N>/)
  e1 / e2   상황별 지표 곡선 (정확도·모드 선택 / 실현가능성·다양성). 한 칸에 상황 2~3개 + 전체(점선)
  e3        학습 로그 곡선 — 이 판과 비교 판의 history, 이 판을 val 앞 N 개로 다시 잰 값(점선)
  e4        무엇이 먼저 배워지나 — 에폭 1 → 최종 변화의 90% 에 처음 닿는 에폭, 에폭 사이 요동
  e5        끝 1초 감속 추종 — 라벨 인공물(정답 끝 0.5초 감속)을 에폭에 따라 얼마나 따라가나
  panel/    고정 시나리오 (시드 0, ID 는 data/epoch_picks_n<N>.json) — 에폭 1·3·5·10·15 의
            확률 1위(실선)·승자(점선), 에폭별 모드 확률, 1위 모드 v·a. panel_overview.png 는 지도만 모은 것

  python src/viz_v4_epochs.py --tag v4_l4nw_ah2_2hz_full_sm1_s0                    # 기본 (val 앞 2,000)
  python src/viz_v4_epochs.py --tag <tag> --n-val 24988 --epochs 15                 # 재현 확인 (학습 로그와 같은 집합)
  python src/viz_v4_epochs.py --tag <tag> --device cpu --n-val 200 --input ah2_2hz --th0 guard   # 학습 중 개발

정의·임계값은 viz_v4_common.py 상수와 아래 METRICS 에 있다. 학습이 돌면 GPU 를 쓰지 않는다.
"""
import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_v4_common as C
import viz_v4_dump as D

PANEL_CLASSES = ["좌회전", "우회전", "좌차선변경", "우차선변경", "급감속", "정속", "정지"]
CLASS_FILE = {"좌회전": "left_turn", "우회전": "right_turn", "좌차선변경": "lc_left", "우차선변경": "lc_right",
              "급감속": "hard_brake", "정속": "cruise", "정지": "stop"}
FACETS = [("회전", ["좌회전", "우회전"]),
          ("차선변경", ["좌차선변경", "우차선변경"]),
          ("정지 · 급감속 · 급가속", ["정지", "급감속", "급가속"]),
          ("정속 · 기타", ["정속", "기타"])]
DEFAULT_N = 2000
LAST_K = 3            # 최종값 = 마지막 3 에폭 평균 (val 요동을 줄이려고)
REACH = 0.9           # '90% 도달' = 에폭 처음 → 최종 변화량의 90% 에 처음 닿는 에폭
MOVE_REF_V = 5.0      # 끝 감속 모집단: 4.1–5.0 s 정답(위치 차분) 속력 > 5 m/s (viz_v4_stats W5 와 같다)
MODEL_REF_V = 1.0     # 그리고 평가한 모든 에폭에서 1위·승자 모드의 4.1–5.0 s 속력 > 1 m/s (모집단을 에폭끼리 고정)
K_REF = slice(40, 50)   # 예측 스텝 40..49 = 4.1–5.0 s
K_END = slice(50, 60)   # 예측 스텝 50..59 = 5.1–6.0 s (끝 1초)
K_LAST6 = slice(54, 60)  # 5.5–6.0 s (끝 0.6초 평균 가속도)
SPEED_BINS = [5, 10, 15, 20, 40]
THRESH_KEYS = ("CLASSES", "STOP_VMAX", "TURN_DEG", "LC_MAX_DEG", "LC_LAT_M", "LC_MAX_ABS_D", "LC_MIN_MOVE",
               "ROUTE_ALIGN_DEG", "MOVE_V", "DEC_A", "ACC_A", "CONST_DV", "CONST_V0", "MED_K", "SG_WIN",
               "SG_POLY", "HEAD_AVG", "LEAD_TYPES", "LEAD_LAT_M", "LEAD_MIN_M", "LEAD_MAX_M", "LEAD_HEAD_DEG",
               "LEAD_EXT_M", "PATH_STEP_M", "THW_MIN_V", "MISS_M", "MIN_N")

# (키, 그림 라벨, 표 형식, 좋은 방향 -1 낮을수록 / +1 높을수록 / 0 없음, 모집단)
METRICS = [
    ("minade", "minADE6 [m]", "{:.3f}", -1, "전체"),
    ("minfde", "minFDE6 [m]", "{:.3f}", -1, "전체"),
    ("miss", "miss [%]", "{:.1f}", -1, "전체"),
    ("win_ne_top1", "승자 ≠ 확률 1위 [%]", "{:.1f}", -1, "전체"),
    ("ce", "CE (승자 모드)", "{:.3f}", -1, "전체"),
    ("gt_top1", "정답 경로 = 확률 1위 [%]", "{:.1f}", +1, "구별 분기 ≥ 2"),
    ("offlane", "이탈 (시나리오당 0~6)", "{:.3f}", -1, "전체"),
    ("jit_theta", "흔들림 θ", "{:.5f}", -1, "살아있는 모드 step"),
    ("jit_a", "흔들림 a", "{:.5f}", -1, "살아있는 모드 step"),
    ("exc_pct", "7.3°초과 [%]", "{:.3f}", -1, "살아있는 모드 step"),
    ("spread", "모드 끝점 퍼짐 [m] (중앙값)", "{:.2f}", 0, "살아있는 모드 ≥ 2"),
    ("eff_br", "유효 분기 수 (평균)", "{:.3f}", 0, "구별 분기 ≥ 2"),
]
MET = {m[0]: m for m in METRICS}
POP_N = {"전체": "n", "구별 분기 ≥ 2": "n_nd2", "살아있는 모드 ≥ 2": "n_alive2", "살아있는 모드 step": "n"}
DEFINITIONS = {
    "minade/minfde": "살아있는 모드 중 최소 ADE / FDE (train_v4.evaluate 와 같다)",
    "miss": f"minFDE6 > {C.MISS_M:g} m 인 시나리오 비율",
    "win_ne_top1": "승자(끝점 오차 최소 = 학습 WTA) ≠ 확률 1위 인 비율",
    "ce": "cross_entropy(logits, 승자) — 학습 CE 항을 시나리오별로",
    "gt_top1": "정답 기준 경로(viz_v4_dump.raw_one)의 경로 확률(같은 경로 슬롯 합)이 1위인 비율, 구별 분기 ≥ 2 만",
    "offlane": "살아있는 모드별 '밴드 밖 step 비율'의 합 (시나리오 평균, 0~6)",
    "jit_theta/jit_a": "Δ(dθ/8°)², Δ(a/8 m/s²)² 의 합 ÷ 살아있는 모드 step 수 (상황 안에서 step 가중)",
    "exc_pct": "살아있는 모드의 |Δθ| > 7.3° step 비율 (step 가중)",
    "spread": "살아있는 모드 끝점의 쌍별 평균 거리의 중앙값, 살아있는 모드 ≥ 2 만",
    "eff_br": "exp(경로 확률 엔트로피)의 평균, 구별 분기 ≥ 2 만",
    "final": f"최종값 = 평가한 마지막 {LAST_K} 에폭 평균",
    "e90": f"(값 − 첫 에폭 값) / (최종 − 첫 에폭 값) ≥ {REACH:g} 에 처음 닿는 에폭. |최종 − 첫| ≤ 요동이면 정하지 않음",
    "e90_stay": f"유지 에폭 = 그 에폭에 진행률 ≥ {REACH:g} 이고 그 뒤 끝까지 진행률 ≥ {REACH:g} − 요동/|최종 − 첫| "
                "(요동 한 번만큼 여유)인 첫 에폭",
    "e90_stay_strict": f"엄격 유지 에폭 = 그 에폭부터 끝까지 진행률 ≥ {REACH:g} 인 첫 에폭 (여유 없음)",
    "fluct": "요동 = 에폭 번호가 중앙 이상인 연속 에폭 쌍의 |차이| 중앙값",
    "end_deficit": "끝 1초 부족 거리 = Σ_{5.1–6.0 s} (4.1–5.0 s 평균 속력 − 속력)·0.1 s — 등속 대비 덜 간 거리",
}


# --------------------------------------------------------------------------- 도구
def sha_bytes(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


def fig_dir(tag, n, out_root=None):
    base = Path(out_root) if out_root else C.tag_dirs(tag)["base"]
    return base / ("epochs" if n == DEFAULT_N else f"epochs_n{n}"), base / "data"


def choose_device(want):
    import torch
    busy = C.training_running()
    if want == "auto":
        dev = "cpu" if busy or not torch.cuda.is_available() else "cuda"
        if busy:
            print("[device] train_v4 가 돌고 있다 -> CPU", flush=True)
        return dev
    if want == "cuda" and busy:
        raise SystemExit("train_v4 가 돌고 있다 — 공유 머신 규칙상 GPU 추론을 하지 않는다 (--device cpu 로 작게)")
    return want


# --------------------------------------------------------------------------- 1. 기준표 (에폭과 무관)
def base_key(cache_dir):
    th = {k: getattr(C, k) for k in THRESH_KEYS}
    src = "".join(inspect.getsource(f) for f in (D.raw_one, D._lead, D._smooth_speed, D._band_at))
    files = b"".join((C.SRC / f).read_bytes() for f in ("heading_decomp.py", "lane_frame.py", "dataset_map.py"))
    return sha_bytes(json.dumps(th, sort_keys=True, default=str, ensure_ascii=False), src, files,
                     Path(cache_dir).name)[:16]


def build_base(cache_dir, n, workers):
    import pandas as pd
    from dataset_cached import CachedV4Dataset
    ds = CachedV4Dataset(cache_dir, limit=n)
    sids = ds.sids[:n]
    rows, arrs = [None] * n, [None] * n
    t0 = time.time()
    # 승자 경로 자리는 0 번 경로로 채우고(raw_one 은 승자 경로 기준 (s, d) 도 낸다), 그 필드는 버린다 — 모델과 무관한 표만 남긴다
    with Pool(workers, initializer=D._init_worker, initargs=(str(cache_dir),)) as pool:
        for k, (row, arr) in enumerate(pool.imap(D.raw_one, [(i, sids[i], 0) for i in range(n)], chunksize=16)):
            rows[k], arrs[k] = row, arr
            if (k + 1) % 5000 == 0:
                print(f"[base] {k + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    drop_row = ("gt_s_end_w", "gt_d_end_w")
    drop_arr = ("gt_s_w", "gt_d_w", "gt_band_w", "gt_k_w")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in drop_row} for r in rows])
    df.insert(1, "sid", sids)
    if df["y_err"].max() > 1e-3:
        raise SystemExit(f"원본 정답과 캐시 y 가 다르다 (최대 {df['y_err'].max():.2e} m) — 인덱스 정렬이 어긋났다")
    df["n_distinct"] = np.maximum(np.asarray(ds.raw("n_distinct")[:n]).astype(int), 1)
    df["n_reachable"] = np.asarray(ds.raw("n_reachable")[:n]).astype(int)
    df["fallback"] = np.asarray(ds.raw("route_fallback")[:n]) > 0.5
    df["v0"] = np.asarray(ds.raw("v0")[:n]).astype(np.float32)
    df["n_alive"] = (np.asarray(ds.raw("route_mask")[:n]) > 0).sum(1)
    R = {k: np.stack([a[k] for a in arrs]).astype(bool if k == "route_inband" else np.float32)
         for k in arrs[0] if k not in drop_arr}
    print(f"[base] {n} 시나리오 {time.time() - t0:.0f}s (workers {workers}), y 최대차 {df['y_err'].max():.1e} m",
          flush=True)
    return df, R


def load_or_build_base(cache_dir, n, workers, data_dir, fresh):
    import pandas as pd
    key = base_key(cache_dir)
    if not fresh:
        for mj in sorted(data_dir.glob("epoch_base_n*.json")):
            m = json.loads(mj.read_text())
            if m.get("key") != key or m.get("n", 0) < n:
                continue
            stem = mj.with_suffix("")
            df = pd.read_parquet(stem.with_suffix(".parquet")).iloc[:n].reset_index(drop=True)
            z = np.load(stem.with_suffix(".npz"))
            R = {k: z[k][:n] for k in z.files}
            print(f"[base] 캐시 사용 {stem.name} (n={m['n']} 중 앞 {n})", flush=True)
            return df, R, key
    df, R = build_base(cache_dir, n, workers)
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = data_dir / f"epoch_base_n{n}"
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    np.savez(stem.with_suffix(".npz"), **R)
    stem.with_suffix(".json").write_text(json.dumps(
        {"key": key, "n": n, "val_cache": str(cache_dir), "workers": workers,
         "thresholds": {k: getattr(C, k) for k in THRESH_KEYS},
         "class_counts": df["cls"].value_counts().reindex(C.CLASSES).fillna(0).astype(int).to_dict()},
        indent=2, ensure_ascii=False, default=float))
    return df, R, key


# --------------------------------------------------------------------------- 고정 시나리오
def pick_fixed(df, per_class, seed=C.SEED):
    """상황마다 per_class 개 (시드 고정). 폴백(살아있는 모드 1개)은 모드 확률 막대가 뜻이 없어 뺀다."""
    rng = np.random.default_rng(seed)
    out = []
    for c in PANEL_CLASSES:
        pool = np.where((df["cls"] == c).to_numpy() & ~df["fallback"].to_numpy()
                        & (df["n_alive"].to_numpy() >= 2))[0]
        if len(pool) == 0:
            continue
        ids = np.sort(rng.choice(pool, min(per_class, len(pool)), replace=False))
        for k, i in enumerate(ids, 1):
            out.append({"cls": c, "k": k, "idx": int(i), "sid": str(df["sid"].iloc[i]), "pool": int(len(pool))})
    return out


def load_or_pick(df, n, per_class, data_dir, picks_from=None):
    path = data_dir / f"epoch_picks_n{n}.json"
    sid2i = {s: i for i, s in enumerate(df["sid"])}
    src = Path(picks_from) if picks_from else (path if path.exists() else None)
    if src is not None:
        d = json.loads(src.read_text())
        # pool_n = 고를 때의 모집단 크기(val 앞 N) — 다른 N 으로 다시 그려도 그림 설명이 고른 조건을 적게
        picks = [dict(p, idx=sid2i[p["sid"]], pool_n=int(p.get("pool_n", d.get("n_val", n))))
                 for p in d["picks"] if p["sid"] in sid2i]
        if picks_from or d.get("per_class") == per_class:
            if len(picks) < len(d["picks"]):
                print(f"[picks] {src} 의 {len(d['picks']) - len(picks)}개는 val 앞 {n} 밖이라 뺐다", flush=True)
            print(f"[picks] 고정 시나리오 {len(picks)}개를 {src.name} 에서 읽었다", flush=True)
            if picks_from:
                path.write_text(json.dumps(dict(d, picks=picks, copied_from=str(src)), indent=2, ensure_ascii=False))
            return picks
    picks = [dict(p, pool_n=int(n)) for p in pick_fixed(df, per_class)]
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": C.SEED, "n_val": n, "per_class": per_class,
                                "rule": "정답 기반 상황별 무작위, 폴백·살아있는 모드 1개 제외", "picks": picks},
                               indent=2, ensure_ascii=False))
    print(f"[picks] 고정 시나리오 {len(picks)}개를 새로 골랐다 (시드 {C.SEED}) -> {path.name}", flush=True)
    return picks


# --------------------------------------------------------------------------- 2. 에폭별 추론
PANEL_FIELDS = ("traj", "prob", "alive", "v", "a", "ade", "fde")
SCEN_FIELDS = ("minade", "minfde", "winner", "top1", "ce", "offlane", "jt_sum", "ja_sum", "exc_sum", "spread",
               "eff_br", "n_clusters", "gt_rank", "gt_prob", "top1_prob", "top1_ade", "top1_fde",
               "v_top", "v_win", "a_top", "a_win")
SCORE_KEYS = ("minADE6", "minFDE6", "offlane", "dtheta_over_label_pct", "jitter_theta", "jitter_a")


def infer_epoch(tag, ckpt, cfg, device, batch, lw, n, cache_dir, cache_raw, gt_route, picks_idx):
    """체크포인트 하나 -> (시나리오별 값, 고정 시나리오 배열, compare 방식 전체 점수)."""
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    model, in_dim, lane_in = C.build_model(sd, cfg.get("th0", "current"), device)
    info = C.model_info(cfg, in_dim, lane_in)
    P, S, info, _ = D.run_inference(tag, device, batch, lw, limit=n, model=model, info=info, cache_dir=cache_dir)
    Dd = D.derived(P, S, cache_raw)
    score = D.score_like_compare(P, S)
    al = P["alive"]
    ar = np.arange(n)
    win, top = S["winner"].astype(int), S["top1"].astype(int)
    rp = Dd["rprob"]
    gp = rp[ar, gt_route]
    f16 = np.float16
    E = {"minade": S["minade"], "minfde": S["minfde"],
         "winner": S["winner"].astype(np.int8), "top1": S["top1"].astype(np.int8), "ce": S["l_ce"],
         "offlane": Dd["offlane"].astype(np.float32),
         "jt_sum": np.where(al, P["jit_t_sum"], 0).astype(np.float64).sum(1),
         "ja_sum": np.where(al, P["jit_a_sum"], 0).astype(np.float64).sum(1),
         "exc_sum": np.where(al, P["exc_cnt"], 0).astype(np.float64).sum(1),
         "spread": Dd["spread"].astype(np.float32), "eff_br": Dd["eff_branches"].astype(np.float32),
         "n_clusters": Dd["n_end_clusters"].astype(np.int8),
         "gt_rank": (1 + (rp > gp[:, None] + 1e-9).sum(1)).astype(np.int8), "gt_prob": gp.astype(np.float32),
         "top1_prob": Dd["top1_prob"].astype(np.float32),
         "top1_ade": Dd["top1_ade"].astype(np.float32), "top1_fde": Dd["top1_fde"].astype(np.float32),
         "v_top": P["v"][ar, top].astype(f16), "v_win": P["v"][ar, win].astype(f16),
         "a_top": P["a"][ar, top].astype(f16), "a_win": P["a"][ar, win].astype(f16)}
    pk = np.asarray(picks_idx, int)
    PN = {k: P[k][pk] for k in PANEL_FIELDS}
    del P, S, Dd, model
    gc.collect()
    return E, PN, score


def calc_key():
    files = b"".join((C.SRC / f).read_bytes() for f in ("model_v4.py", "train_v4.py", "dataset_cached.py"))
    src = "".join(inspect.getsource(f) for f in (infer_epoch, D.run_inference, D.derived, D.score_like_compare))
    return sha_bytes(files, src)[:10]


def run_epochs(tag, cfg, ckpts, epochs, device, a, n, cache_dir, df, picks, data_dir):
    """평가 결과를 data/epoch_scen_n<N>.npz 에 모은다. 키(체크포인트 해시·장치·N·고정 시나리오·코드)가 같은 에폭은 재사용."""
    from dataset_cached import CachedV4Dataset
    path = data_dir / f"epoch_scen_n{n}.npz"
    picks_idx = [p["idx"] for p in picks]
    ck = calc_key()
    pk_hash = sha_bytes(np.asarray(picks_idx, np.int64).tobytes())[:8]
    old = {}
    if path.exists() and not a.fresh:
        z = np.load(path, allow_pickle=False)
        for r, (e, key) in enumerate(zip(z["epochs"], z["keys"])):
            old[int(e)] = (str(key), {f: z[f][r] for f in SCEN_FIELDS},
                           {f: z["pn_" + f][r] for f in PANEL_FIELDS},
                           {k: float(z["sc_" + k][r]) for k in SCORE_KEYS} | {"n": n})
    ds = CachedV4Dataset(cache_dir, limit=n)
    cache_raw = {k: ds.raw(k)[:n] for k in ("n_distinct", "n_reachable", "route_fallback", "v0", "lanes",
                                          "lane_mask")}
    gt_route = df["gt_route"].to_numpy().astype(int)
    rows = {}
    t_all = time.time()
    for e in epochs:
        key = "|".join([sha_bytes(ckpts[e].read_bytes())[:16], device, str(n), pk_hash, ck, Path(cache_dir).name])
        if e in old and old[e][0] == key:
            rows[e] = old[e]
            print(f"[ep{e:02d}] 저장된 결과 재사용", flush=True)
            continue
        t0 = time.time()
        E, PN, score = infer_epoch(tag, ckpts[e], cfg, device, a.batch, a.lw, n, cache_dir, cache_raw,
                                   gt_route, picks_idx)
        rows[e] = (key, E, PN, score)
        print(f"[ep{e:02d}] minADE6 {score['minADE6']:.4f}  minFDE6 {score['minFDE6']:.4f}  "
              f"이탈 {score['offlane']:.3f}  7.3°초과 {score['dtheta_over_label_pct']:.3f}%  "
              f"흔들림 θ {score['jitter_theta']:.5f} a {score['jitter_a']:.5f}  ({time.time() - t0:.1f}s)",
              flush=True)
    # 이번에 평가하지 않은 옛 에폭도 버리지 않는다 — 고정 시나리오가 같을 때만 (배열 모양이 같아야 한다).
    # 그림에는 이번에 고른 에폭만 쓴다. 키 = 체크포인트|장치|N|고정 시나리오|코드|캐시
    keep = dict(rows)
    for e, v in old.items():
        if e not in keep and v[0].split("|")[3] == pk_hash and v[2]["prob"].shape[0] == len(picks_idx):
            keep[e] = v
    es = sorted(keep)
    out = {"epochs": np.array(es), "keys": np.array([keep[e][0] for e in es]), "picks": np.array(picks_idx)}
    for f in SCEN_FIELDS:
        out[f] = np.stack([keep[e][1][f] for e in es])
    for f in PANEL_FIELDS:
        out["pn_" + f] = np.stack([keep[e][2][f] for e in es])
    for k in SCORE_KEYS:
        out["sc_" + k] = np.array([keep[e][3][k] for e in es], np.float64)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[epochs] {len(rows)} 에폭 {time.time() - t_all:.0f}s -> {path.name}", flush=True)
    return {e: rows[e] for e in epochs}


PANEL_EXTRA_ALL = ("traj", "prob", "v", "a", "h", "theta", "d", "band")


def panel_extra(tag, cfg, ckpts, epochs, picks, cache_dir, device, data_dir, z, batch=64):
    """고정 시나리오만 다시 추론해 h·θ·d·밴드를 모은다 (에폭 결과 npz 에는 궤적·확률·v·a 만 있다).
    키 = 체크포인트 해시·장치·고정 시나리오·이 함수 소스·calc_key·캐시. 같은 출력인 궤적·확률·v·a 를 저장된 pn_* 와
    대조한다(요건 9b) — 배치 구성이 달라도 같은 값이어야 한다."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from dataset_cached import CachedV4Dataset
    from train_v4 import to_dev
    idx = [int(p["idx"]) for p in picks]
    pk_hash = sha_bytes(np.asarray(idx, np.int64).tobytes())[:8]
    src_h = sha_bytes(inspect.getsource(panel_extra))[:8]
    ck = calc_key()
    path = data_dir / f"epoch_panel_extra_{pk_hash}.npz"
    keys = {e: "|".join([sha_bytes(ckpts[e].read_bytes())[:16], device, pk_hash, src_h, ck, Path(cache_dir).name])
            for e in epochs}
    old = {}
    if path.exists():
        zz = np.load(path, allow_pickle=False)
        for r, (e, k) in enumerate(zip(zz["epochs"], zz["keys"])):
            if keys.get(int(e)) == str(k):
                old[int(e)] = {f: zz[f][r] for f in PANEL_EXTRA_ALL}
    ds = CachedV4Dataset(cache_dir)
    sub = Subset(ds, idx)
    out = {}
    t0 = time.time()
    n_new = 0
    for e in epochs:
        if e in old:
            out[e] = old[e]
            continue
        sd = torch.load(ckpts[e], map_location="cpu")
        model, _, _ = C.build_model(sd, cfg.get("th0", "current"), device)
        rec = {f: [] for f in PANEL_EXTRA_ALL}
        with torch.no_grad():
            for b in DataLoader(sub, batch_size=batch, shuffle=False, num_workers=0):
                kw = to_dev(b, device, "l0", True)
                traj, logits, aux = model(**kw)
                cpu = lambda t: t.detach().float().cpu().numpy()
                rec["traj"].append(cpu(traj))
                rec["prob"].append(cpu(torch.softmax(logits, 1)))
                for f in ("v", "a", "h", "theta", "d", "band"):
                    rec[f].append(cpu(aux[f]))
        out[e] = {f: np.concatenate(v).astype(np.float32) for f, v in rec.items()}
        n_new += 1
        del model, sd
    if n_new:
        es = sorted(out)
        np.savez(path, epochs=np.array(es), keys=np.array([keys[e] for e in es]),
                 **{f: np.stack([out[e][f] for e in es]) for f in PANEL_EXTRA_ALL})
    # 대조: 저장된 에폭 결과(pn_*)와 같은 출력인가 (살아있는 모드만)
    row = {int(e): r for r, e in enumerate(z["epochs"])}
    chk = {"n_epochs": len(epochs), "n_new": n_new, "device": device, "sec": round(time.time() - t0, 1), "max_abs": {}}
    for f in ("traj", "prob", "v", "a"):
        m = 0.0
        for e in epochs:
            al = z["pn_alive"][row[e]].astype(bool)
            dd = np.abs(out[e][f].astype(np.float64) - z["pn_" + f][row[e]].astype(np.float64))
            if dd.ndim > 2:
                dd = dd.reshape(dd.shape[0], dd.shape[1], -1).max(-1)
            m = max(m, float(dd[al].max()))
        chk["max_abs"][f] = m
    chk["ok"] = bool(max(chk["max_abs"].values()) < 1e-3)
    print(f"[panel] 고정 시나리오 재추론 {n_new}/{len(epochs)} 에폭 ({chk['sec']} s) · 저장 결과와 최대 차 "
          + " · ".join(f"{k} {v:.1e}" for k, v in chk["max_abs"].items()), flush=True)
    return out, chk


# --------------------------------------------------------------------------- 3. 집계
def agg(E, m, nd, n_alive, live):
    """한 에폭·한 모집단(m) 의 지표. 정의는 DEFINITIONS."""
    n = int(m.sum())
    md, ms = m & (nd >= 2), m & (n_alive >= 2)
    out = {"n": n, "n_nd2": int(md.sum()), "n_alive2": int(ms.sum())}
    if n == 0:
        return out | {k: float("nan") for k in MET}
    lv = float(live[m].sum())
    out.update({
        "minade": float(E["minade"][m].astype(np.float64).mean()),
        "minfde": float(E["minfde"][m].astype(np.float64).mean()),
        "miss": 100.0 * float((E["minfde"][m] > C.MISS_M).mean()),
        "win_ne_top1": 100.0 * float((E["winner"][m] != E["top1"][m]).mean()),
        "ce": float(E["ce"][m].astype(np.float64).mean()),
        "gt_top1": 100.0 * float((E["gt_rank"][md] == 1).mean()) if md.any() else float("nan"),
        "offlane": float(E["offlane"][m].astype(np.float64).mean()),
        "jit_theta": float(E["jt_sum"][m].sum() / lv),
        "jit_a": float(E["ja_sum"][m].sum() / lv),
        "exc_pct": 100.0 * float(E["exc_sum"][m].sum() / lv),
        "spread": float(np.nanmedian(E["spread"][ms])) if ms.any() else float("nan"),
        "eff_br": float(E["eff_br"][md].astype(np.float64).mean()) if md.any() else float("nan"),
        "top1_ade": float(E["top1_ade"][m].astype(np.float64).mean()),
        "gt_prob": float(E["gt_prob"][md].astype(np.float64).mean()) if md.any() else float("nan"),
    })
    return out


def aggregate(rows, df):
    nd = df["n_distinct"].to_numpy()
    n_alive = df["n_alive"].to_numpy()
    live = n_alive.astype(np.float64) * (C.FUT - 1)
    cls = df["cls"].to_numpy()
    masks = {"전체": np.ones(len(df), bool)} | {c: cls == c for c in C.CLASSES}
    return {e: {c: agg(rows[e][1], m, nd, n_alive, live) for c, m in masks.items()} for e in rows}


def order_stats(epochs, ys):
    """무엇이 먼저 배워지나 — 한 지표의 에폭 곡선에서 90% 도달 에폭·요동."""
    ep = np.asarray(epochs)
    y = np.asarray(ys, float)
    if len(y) < 3 or not np.isfinite(y).all():
        return None
    first, final = float(y[0]), float(y[-min(LAST_K, len(y)):].mean())
    d = final - first
    diffs = np.abs(np.diff(y))
    late = ep[:-1] >= np.median(ep)
    fl = float(np.median(diffs[late])) if late.any() else float(np.median(diffs))
    prog = (y - first) / d if d != 0 else np.full(len(y), np.nan)
    resolved = bool(abs(d) > fl)
    e90 = next((int(e) for e, p in zip(ep, prog) if p >= REACH), None) if resolved else None
    # 유지 에폭: 그 에폭부터 끝까지 진행률이 (90% − 요동 한 번) 아래로 내려가지 않는 첫 에폭 — 한 번 튄 값과 구별한다
    e90s = e90x = None
    if resolved:
        floor = REACH - fl / abs(d)
        for k in range(len(ep)):
            if np.all(prog[k:] >= floor) and prog[k] >= REACH:
                e90s = int(ep[k])
                break
        # 엄격: 여유 없이 끝까지 90% 이상
        for k in range(len(ep)):
            if np.all(prog[k:] >= REACH):
                e90x = int(ep[k])
                break
    return {"first": first, "final": final, "change": d,
            "change_pct": 100.0 * d / abs(first) if first else float("nan"),
            "fluct": fl, "fluct_pct": 100.0 * fl / abs(final) if final else float("nan"),
            "snr": abs(d) / fl if fl > 0 else float("inf"), "resolved": resolved, "e90": e90, "e90_stay": e90s,
            "e90_stay_strict": e90x,
            "progress": [float(p) for p in prog]}


def learning_order(res, epochs):
    cls_all = ["전체"] + C.CLASSES
    return {c: {k: order_stats(epochs, [res[e][c][k] for e in epochs]) for k in MET} for c in cls_all}


def end_decel(rows, R, epochs):
    """끝 1초 감속 추종. 모집단은 모든 평가 에폭에 공통(에폭끼리 같은 시나리오를 비교한다)."""
    pos = R["pos"].astype(np.float64)
    step = np.linalg.norm(np.diff(pos, axis=1), axis=2) / C.DT          # (N,109) t -> t+1
    fut = step[:, C.OBS - 1:]                                             # k: 49+k -> 50+k
    vref = fut[:, K_REF].mean(1)
    vf = R["v_fld"][:, C.OBS - 1:].astype(np.float64)                    # 속도 필드(평활), viz_v4_stats W5 와 같다
    ok = vref > MOVE_REF_V
    V = {e: {k: rows[e][1][k].astype(np.float64) for k in ("v_top", "v_win", "a_top", "a_win")} for e in epochs}
    for e in epochs:
        ok &= (V[e]["v_top"][:, K_REF].mean(1) > MODEL_REF_V) & (V[e]["v_win"][:, K_REF].mean(1) > MODEL_REF_V)
    if ok.sum() < 5:
        return None

    def deficit(s, r):
        return ((r[:, None] - s[:, K_END]) * C.DT).sum(1)

    kk = np.arange(30, 60)
    dg = deficit(fut[ok], vref[ok])
    vfr = vf[ok][:, K_REF].mean(1)
    dfield = deficit(vf[ok], vfr)
    v_ok = vref[ok]
    bins = list(zip(SPEED_BINS[:-1], SPEED_BINS[1:]))
    out = {"n": int(ok.sum()), "n_moving": int((vref > MOVE_REF_V).sum()), "t": ((kk + 1) * C.DT).tolist(),
           "gt_ratio": np.median(fut[ok][:, kk] / vref[ok, None], axis=0).tolist(),
           "field_ratio": np.median(vf[ok][:, kk] / vfr[:, None], axis=0).tolist(),
           "gt_deficit": float(np.median(dg)), "field_deficit": float(np.median(dfield)),
           "gt_a_last6": float(((fut[ok][:, 59] - fut[ok][:, 53]) / (6 * C.DT)).mean()),
           "field_a_last6": float(((vf[ok][:, 59] - vf[ok][:, 53]) / (6 * C.DT)).mean()),
           "speed_bins": [f"{a_}-{b_}" for a_, b_ in bins],
           "speed_bin_n": [int(((v_ok >= a_) & (v_ok < b_)).sum()) for a_, b_ in bins],
           "gt_deficit_by_speed": [float(np.median(dg[(v_ok >= a_) & (v_ok < b_)])) if ((v_ok >= a_) & (v_ok < b_)).any()
                                   else None for a_, b_ in bins],
           "epoch": {}}
    for e in epochs:
        vt, vw = V[e]["v_top"][ok], V[e]["v_win"][ok]
        rt, rw = vt[:, K_REF].mean(1), vw[:, K_REF].mean(1)
        dt_, dw_ = deficit(vt, rt), deficit(vw, rw)
        out["epoch"][e] = {
            "top_ratio": np.median(vt[:, kk] / rt[:, None], axis=0).tolist(),
            "win_ratio": np.median(vw[:, kk] / rw[:, None], axis=0).tolist(),
            "top_deficit": float(np.median(dt_)), "win_deficit": float(np.median(dw_)),
            "top_follow": float(np.median(dt_) / np.median(dg)), "win_follow": float(np.median(dw_) / np.median(dg)),
            "top_minus_gt_median": float(np.median(dt_ - dg)), "win_minus_gt_median": float(np.median(dw_ - dg)),
            "top_a_last6": float(V[e]["a_top"][ok][:, K_LAST6].mean()),
            "win_a_last6": float(V[e]["a_win"][ok][:, K_LAST6].mean()),
            "top_end_ratio": float(np.median(vt[:, 59] / rt)),
            "top_deficit_by_speed": [float(np.median(dt_[(v_ok >= a_) & (v_ok < b_)]))
                                     if ((v_ok >= a_) & (v_ok < b_)).any() else None for a_, b_ in bins]}
    return out


def repro_check(tag, rows, hist, hist_src, cfg, n):
    """N 이 학습 로그의 val 크기와 같을 때만 대조한다 (로그 값은 val 24,988 평가)."""
    vl = cfg.get("val_limit")
    if n != vl:
        return {"applicable": False, "reason": f"평가 N={n} ≠ 학습 로그 val {vl} — 대조하지 않는다"}
    tol = 5e-4 if hist_src == "json" else 1e-3        # 로그 줄은 소수 2~3자리로 반올림돼 있다
    hmap = {h["epoch"]: h for h in hist}
    out = {"applicable": True, "history_source": hist_src, "tol": tol, "epochs": {}}
    pairs = (("minADE6", "minADE6", 1.0), ("minFDE6", "minFDE6", 1.0), ("offlane", "val_offlane_steps", 1.0),
             ("dtheta_over_label_pct", "val_dtheta_over_label_pct", 1.0), ("jitter", "val_jitter", 1.0))
    for e, (_, _, _, sc) in rows.items():
        h = hmap.get(e)
        if h is None:
            continue
        got = dict(sc, jitter=sc["jitter_theta"] + sc["jitter_a"])
        d = {k: {"got": got[k], "log": h[hk], "abs_diff": abs(got[k] - h[hk])} for k, hk, _ in pairs if hk in h}
        # 로그 줄의 이탈·7.3°초과는 소수 2자리라 ±0.005 까지 반올림 차가 난다
        t = {k: (tol if hist_src == "json" or k in ("minADE6", "minFDE6", "jitter") else 5.1e-3) for k in d}
        out["epochs"][e] = {"diff": d, "ok": all(v["abs_diff"] <= t[k] for k, v in d.items()),
                            "same_3dp": {k: round(v["got"], 3) == round(v["log"], 3) for k, v in d.items()}}
    out["ok"] = all(v["ok"] for v in out["epochs"].values()) if out["epochs"] else None
    return out


def best_ckpt_check(tag, ckpts, hist):
    """best(lstm_<tag>.pth) 가 history 의 minADE6 최소 에폭 체크포인트와 같은 가중치인지."""
    import torch
    p = C.RUNS / f"lstm_{tag}.pth"
    if not hist or not p.exists():
        return None
    be = min(hist, key=lambda h: h["minADE6"])["epoch"]
    if be not in ckpts:
        return {"best_epoch": be, "same": None, "reason": "그 에폭 체크포인트가 없다"}
    a_, b_ = torch.load(p, map_location="cpu"), torch.load(ckpts[be], map_location="cpu")
    same = a_.keys() == b_.keys() and all(torch.equal(a_[k], b_[k]) for k in a_)
    return {"best_epoch": int(be), "same": bool(same)}


# --------------------------------------------------------------------------- 그림 도구
def end_labels(ax, items, dx=5.0, gap=10.0, fs=7.3):
    """선 끝 직접 라벨. 세로로 gap(pt) 보다 가까우면 벌리고, 옮긴 라벨은 가는 선으로 잇는다.
    라벨 뒤는 면 색으로 칠해 기준선이 글자를 긋지 않게 한다."""
    to_pt = 72.0 / ax.figure.dpi
    pts = []
    for x, y, t in items:
        if np.isfinite(y):
            yd = ax.transData.transform((x, y))[1] * to_pt
            pts.append([yd, yd, x, y, t])
    if not pts:
        return
    pts.sort(key=lambda p: p[0])
    for k in range(1, len(pts)):
        if pts[k][1] - pts[k - 1][1] < gap:
            pts[k][1] = pts[k - 1][1] + gap
    shift = sum(p[1] - p[0] for p in pts) / len(pts)
    for p in pts:
        p[1] -= shift
    for tgt, placed, x, y, t in pts:
        dy = placed - tgt
        moved = abs(dy) > 0.8
        ax.annotate(t, xy=(x, y), xytext=(dx + (7 if moved else 0), dy), textcoords="offset points",
                    ha="left", va="center", fontsize=fs, color=C.INK2, annotation_clip=False, zorder=20,
                    bbox=dict(boxstyle="square,pad=0.08", fc=C.SURF, ec="none"),
                    arrowprops=dict(arrowstyle="-", color=C.AXIS, lw=0.6, shrinkA=0, shrinkB=1.5) if moved else None)


def epoch_axis(ax, epochs):
    """눈금은 평가한 에폭 안에서만 (라벨 자리로 늘린 x 범위에 없는 에폭 눈금이 찍히지 않게)."""
    ep = sorted(set(int(e) for e in epochs))
    step = max(1, int(np.ceil(len(ep) / 8)))
    ticks = ep[::step]
    if ep[-1] - ticks[-1] >= step:
        ticks.append(ep[-1])
    ax.set_xticks(ticks)


def note(fig, text, y=0.006, fs=7.8):
    fig.text(0.01, y, text, fontsize=fs, color=C.INK2, ha="left", va="bottom")


def last_finite(ys):
    ys = np.asarray(ys, float)
    idx = np.where(np.isfinite(ys))[0]
    return (idx[-1], ys[idx[-1]]) if len(idx) else (None, np.nan)


# --------------------------------------------------------------------------- e1 / e2 상황별 곡선
def fig_class_curves(res, epochs, n_cls, metrics, title, foot, path, best_ep=None, n_val=None):
    plt = C.setup_mpl()
    from matplotlib.lines import Line2D
    nr, nc = len(metrics), len(FACETS)
    W, H = 4.25 * nc + 1.2, 2.25 * nr + 2.0
    fig, axes = plt.subplots(nr, nc, figsize=(W, H), sharex=True, sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=1 - 1.55 / H, bottom=0.86 / H, wspace=0.13, hspace=0.2)
    ep = np.asarray(epochs)
    span = max(ep.max() - ep.min(), 1)
    todo = []
    for i, (key, lab, fmt, better, pop) in enumerate(metrics):
        allv = np.array([res[e]["전체"][key] for e in epochs], float)
        for j, (fname, classes) in enumerate(FACETS):
            ax = axes[i, j]
            if best_ep is not None and ep.min() <= best_ep <= ep.max():
                ax.axvline(best_ep, color=C.AXIS, lw=1.0, zorder=1)
            ax.plot(ep, allv, color=C.INK2, lw=1.3, ls=(0, (4, 2)), zorder=2)
            items = [(ep[-1], allv[-1], "전체")]
            for c in classes:
                ys = np.array([res[e][c][key] for e in epochs], float)
                if not np.isfinite(ys).any():
                    continue
                npop = res[epochs[-1]][c][POP_N[pop]]
                faint = C.faded(npop)
                ax.plot(ep, ys, color=C.CLASS_COLOR[c], lw=2.0, alpha=0.42 if faint else 1.0, marker="o", ms=4.2,
                        mec=C.SURF, mew=1.0, zorder=3)
                k_, y_ = last_finite(ys)
                items.append((ep[k_], y_, c + (f" (n={npop})" if faint else "")))
            todo.append((ax, items))
            ax.set_xlim(ep.min() - 0.5, ep.max() + 0.34 * span + 0.6)
            epoch_axis(ax, epochs)
            if i == 0:
                ax.set_title(fname, loc="left", fontsize=11)
            if j == 0:
                ax.set_ylabel(lab + ("" if pop == "전체" else f"\n({pop})"), fontsize=9)
            if i == nr - 1:
                ax.set_xlabel("에폭")
    # 범례는 칸 순서대로 (열 우선으로 채워지므로 2행 × 6열이면 열마다 한 쌍)
    order = [c for _, cl in FACETS for c in cl]
    hs = [Line2D([], [], color=C.CLASS_COLOR[c], lw=2.0, marker="o", ms=4.2, mec=C.SURF,
                 alpha=0.42 if C.faded(n_cls[c]) else 1.0) for c in order]
    hs += [Line2D([], [], color=C.INK2, lw=1.3, ls=(0, (4, 2)))]
    labs = [f"{c}  n={n_cls[c]:,}" for c in order] + [f"전체  n={n_cls['전체']:,}"]
    if best_ep is not None:
        hs.append(Line2D([], [], color=C.AXIS, lw=1.0))
        labs.append(f"best 에폭 {best_ep} (학습 로그 minADE6 최소)")
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(0.09, 1 - 0.5 / H), ncol=6, fontsize=8.6,
               handlelength=2.2, columnspacing=1.6)
    fig.suptitle(title, x=0.01, y=1 - 0.12 / H, ha="left", fontsize=13)
    fig.canvas.draw()
    for ax, items in todo:
        end_labels(ax, items)
    note(fig, foot)
    return C.savefig(fig, path)


# --------------------------------------------------------------------------- e3 학습 로그
LIGHT = {C.C_BLUE: "#7fb0ea", C.C_ORANGE: "#f29a70"}


def run_style(cfg):
    """판 비교 그림(viz_v4_compare)과 같은 규칙 (필수 요건 6): 색 = 입력 주기(10 Hz 파랑 · 2 Hz 주황),
    선 = 학습률 스케줄(코사인 = 진한 실선 · 상수 = 옅은 점선)."""
    col = C.C_ORANGE if str(cfg.get("input", "")).endswith("2hz") else C.C_BLUE
    cos = cfg.get("lr_sched") == "cosine"
    return (col if cos else LIGHT[col]), ("-" if cos else (0, (4, 2)))


def run_styles(tags):
    """판마다 (색, 선)과 규칙 사용 여부. 규칙으로 둘 이상이 같은 모양이 되면 예전처럼 SERIES 색 + 실선으로 되돌린다."""
    st = {t: run_style(C.run_config(t)[0]) for t in tags}
    if len(set(st.values())) < len(st):
        return {t: (C.SERIES[k % len(C.SERIES)], "-") for k, t in enumerate(tags)}, False
    return st, True


N_CACHE = None      # main 이 val 캐시 크기로 채운다


def pop_desc(n, cfg=None):
    return f"val 전체 {n:,}" if N_CACHE is not None and n >= N_CACHE else f"val 앞 {n:,}"


def fig_train_log(tag, hist, hist_src, refs, res, epochs, n_val, cfg, path, styles=None, rule=False):
    plt = C.setup_mpl()
    from matplotlib.lines import Line2D
    styles = styles or {}
    col0, ls0 = styles.get(tag, (C.C_BLUE, "-"))
    panels = [("minADE6", "minADE6 [m]", lambda r: r["minade"]),
              ("minFDE6", "minFDE6 [m]", lambda r: r["minfde"]),
              ("val_offlane_steps", "이탈 (시나리오당 0~6)", lambda r: r["offlane"]),
              ("val_jitter", "흔들림 (θ + a)", lambda r: r["jit_theta"] + r["jit_a"]),
              ("val_dtheta_over_label_pct", "7.3°초과 [%]", lambda r: r["exc_pct"]),
              ("loss", "학습 손실 (train, 가중합)", None),
              ("sec", "에폭 시간 [s] (train + val 평가)", None)]
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 9.0))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.855, bottom=0.08, wspace=0.26, hspace=0.36)
    runs = [(tag, hist, col0, ls0, "이 판")] + [(t, h, col, ls, "비교") for (t, h, col, ls) in refs]
    for ax, (key, lab, fsub) in zip(axes.flat, panels):
        for t, h, col, ls, _ in runs:
            hh = [r for r in h if key in r]
            if not hh:
                continue
            ax.plot([r["epoch"] for r in hh], [r[key] for r in hh], color=col, ls=ls, lw=2.0, marker="o", ms=4.2,
                    mec=C.SURF, mew=1.0, zorder=3)
            b = min(h, key=lambda r: r["minADE6"])
            if key in b:
                ax.scatter([b["epoch"]], [b[key]], s=120, facecolors="none", edgecolors=col, linewidths=1.6,
                           zorder=4)
        if fsub is not None and res:
            # 다시 잰 값: 로그 점을 감싸는 빈 고리 (겹치면 재현된 것)
            ax.plot(epochs, [fsub(res[e]["전체"]) for e in epochs], color=col0, lw=0.9, ls=(0, (1, 1.6)),
                    marker="o", ms=6.2, mfc="none", mec=C.INK, mew=0.9, zorder=5)
        ax.set_title(lab, loc="left", fontsize=10.5)
        ax.set_xlabel("에폭")
        epoch_axis(ax, sorted(set(epochs) | {r["epoch"] for _, h, _, _, _ in runs for r in h}))
    ax = axes.flat[7]
    ax.axis("off")
    lines = []
    for t, h, _, _, _ in runs:
        if not h:
            lines.append(f"{t}: history 없음")
            continue
        b = min(h, key=lambda r: r["minADE6"])
        secs = sorted(r["sec"] for r in h)
        lines += [t,
                  f"  에폭 {len(h)} · best 에폭 {b['epoch']} · minADE6 {b['minADE6']:.3f} / minFDE6 {b['minFDE6']:.3f}",
                  f"  에폭 시간 중앙 {np.median(secs):.0f} s · 합 {sum(secs) / 60:.1f} 분"]
    lines += ["", f"이 판 history 출처: {hist_src}" + (" (소수 2~3자리)" if hist_src == "로그" else ""),
              f"학습 로그의 val 평가 = val {cfg.get('val_limit', '?'):,} 개" if isinstance(cfg.get('val_limit'), int)
              else "", f"검은 빈 고리 = 이 판 에폭 체크포인트를 {pop_desc(n_val, cfg)} 개로 다시 잰 값",
              "흔들림 = train_v4.evaluate 의 θ·a 두 항 합",
              "색 = 입력 주기(10 Hz 파랑 · 2 Hz 주황), 선 = 스케줄(코사인 실선 · 상수 옅은 점선)" if rule else ""]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.6, color=C.INK2, transform=ax.transAxes,
            linespacing=1.5)
    hs = [Line2D([], [], color=col, ls=ls, lw=2.0, marker="o", ms=4.2, mec=C.SURF) for _, _, col, ls, _ in runs]
    labs = [f"{t} ({'이 판' if k == 0 else '비교'})" for k, (t, _, _, _, _) in enumerate(runs)]
    hs += [Line2D([], [], color=col0, lw=0.9, ls=(0, (1, 1.6)), marker="o", ms=6.2, mfc="none", mec=C.INK, mew=0.9),
           Line2D([], [], color=C.INK2, marker="o", ms=10, mfc="none", ls="none")]
    labs += [f"이 판 · {pop_desc(n_val, cfg)} 재계산", "best 에폭 (그 판의 로그 minADE6 최소)"]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(0.05, 0.935), ncol=4, fontsize=9)
    fig.suptitle("학습 로그 곡선 — 에폭마다 val 평가값 (선마다 한 판)", x=0.01, y=0.985, ha="left", fontsize=13)
    return C.savefig(fig, path)


# --------------------------------------------------------------------------- e4 무엇이 먼저 배워지나
def fig_learning_order(LO, res, epochs, path, tag=""):
    plt = C.setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    cmap = LinearSegmentedColormap.from_list("v4blue", C.BLUE_RAMP)
    ep = list(epochs)
    allo = LO["전체"]
    keys = [k for k in MET if allo.get(k)]
    keys.sort(key=lambda k: (not allo[k]["resolved"], allo[k]["e90"] or 99, -allo[k]["snr"]))
    fig = plt.figure(figsize=(21, 14.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.6, 1.0], height_ratios=[1.0, 0.92], left=0.165, right=0.975,
                  top=0.9, bottom=0.075, wspace=0.42, hspace=0.42)
    # (a) 진행률 열지도 — 요동에 묻힌 지표는 칠하지 않는다
    ax = fig.add_subplot(gs[0, 0])
    M = np.array([[np.clip(p, 0, 1) if allo[k]["resolved"] else np.nan for p in allo[k]["progress"]] for k in keys])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ep)))
    ax.set_xticklabels(ep)
    ax.set_yticks(range(len(keys)))
    ylabs = []
    for k in keys:
        s, fmt = allo[k], MET[k][2]
        ylabs.append(f"{MET[k][1]}\n{fmt.format(s['first'])} → {fmt.format(s['final'])}")
    ax.set_yticklabels(ylabs, fontsize=8.6)
    ax.grid(False)
    for i, k in enumerate(keys):
        s = allo[k]
        if not s["resolved"]:
            ax.add_patch(Rectangle((-0.5, i - 0.5), len(ep), 1, facecolor="#f0efec", edgecolor=C.SURF, lw=1.5))
            ax.text((len(ep) - 1) / 2, i, "변화 ≤ 요동 — 방향을 말할 수 없음", ha="center", va="center", fontsize=8,
                    color=C.INK2)
            continue
        for j, p in enumerate(s["progress"]):
            dark = np.clip(p, 0, 1) > 0.55
            if p < -0.05:
                ax.text(j, i, "−", ha="center", va="center", fontsize=10, color="white" if dark else C.INK)
            elif p > 1.15:
                ax.text(j, i, "+", ha="center", va="center", fontsize=10, color="white" if dark else C.INK)
        if s["e90"] is not None:
            j = ep.index(s["e90"])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=C.C_ORANGE, lw=2.4))
        if s.get("e90_stay") is not None:
            j = ep.index(s["e90_stay"])
            ax.scatter([j], [i + 0.28], s=26, color="white", edgecolors=C.INK, linewidths=1.0, zorder=5)
    ax.set_xlabel("에폭")
    ax.set_title("(a) 전체 — 첫 에폭 → 최종 변화 중 어디까지 왔나 (행 라벨 = 첫 에폭 값 → 최종값)", loc="left",
                 fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.015)
    cb.set_label("진행률 = (값 − 첫 에폭 값) / (최종 − 첫 에폭 값)")
    ax.legend([Patch(facecolor="none", edgecolor=C.C_ORANGE, lw=2.4),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=C.INK, ms=6),
               Line2D([], [], marker="$-$", ls="none", color=C.INK, ms=7),
               Line2D([], [], marker="$+$", ls="none", color=C.INK, ms=7)],
              [f"{int(REACH * 100)}% 에 처음 닿은 에폭", "그 뒤로 유지되는 첫 에폭 (요동 한 번 허용)",
               "첫 에폭보다 반대쪽", "최종을 15% 넘게 지나침"],
              loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=4, fontsize=8.2, frameon=False)
    # (b) 총 변화 / 요동
    ax = fig.add_subplot(gs[0, 1])
    snr = [float(np.clip(allo[k]["snr"], 1e-2, 1e3)) for k in keys]
    ax.barh(range(len(keys)), snr, height=0.5, color=[C.C_BLUE if allo[k]["resolved"] else C.AXIS for k in keys],
            zorder=3)
    ax.axvline(1.0, color=C.INK2, lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax.set_xscale("log")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([""] * len(keys))
    ax.set_ylim(len(keys) - 0.5, -0.5)
    for i, k in enumerate(keys):
        s = allo[k]
        better = MET[k][3]
        if not s["resolved"] or better == 0 or s["change"] == 0:
            way = "" if s["resolved"] else " (요동 이하)"
        else:
            way = " 개선" if s["change"] * better > 0 else " 악화"
        ax.text(snr[i] * 1.1, i, f"{s['change_pct']:+.0f}%{way} · 요동 {s['fluct_pct']:.1f}%", va="center",
                fontsize=8, color=C.INK2)
    ax.set_xlim(min(0.3, min(snr) * 0.7), max(snr) * 14)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|최종 − 첫 에폭 값| ÷ 에폭 간 요동 (log)")
    ax.set_title("(b) 총 변화가 요동의 몇 배인가 — 점선 = 1배\n    글자 = 첫 에폭 대비 변화율 · 요동(최종 대비)",
                 loc="left", fontsize=10.5)
    # (c) 상황 × 지표 90% 도달 에폭 — 모집단 n < MIN_N 인 칸은 칠하지 않는다
    ax = fig.add_subplot(gs[1, :])
    rows_c = ["전체"] + C.CLASSES
    last = epochs[-1]
    E90 = np.full((len(rows_c), len(keys)), np.nan)
    for i, c in enumerate(rows_c):
        for j, k in enumerate(keys):
            s = LO[c].get(k)
            if s and s["e90"] is not None and not C.faded(res[last][c][POP_N[MET[k][4]]]):
                E90[i, j] = s["e90"]
    ax.imshow(np.ma.masked_invalid(E90), cmap=cmap, vmin=min(ep), vmax=max(ep), aspect="auto")
    for i, c in enumerate(rows_c):
        for j, k in enumerate(keys):
            s = LO[c].get(k)
            faint = C.faded(res[last][c][POP_N[MET[k][4]]])
            if s is None:
                txt, col = "·", C.MUTED
            elif s["e90"] is None:
                txt, col = "—", C.MUTED
            else:
                dark = (s["e90"] - min(ep)) / max(max(ep) - min(ep), 1) > 0.5
                txt, col = str(s["e90"]), ("white" if dark and not faint else C.INK)
            if faint:
                col = C.MUTED
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([MET[k][1].replace(" [", "\n[").replace(" (", "\n(") for k in keys], fontsize=8.3)
    ax.set_yticks(range(len(rows_c)))
    ax.set_yticklabels([f"{c}  n={res[last][c]['n']:,}" for c in rows_c], fontsize=9)
    ax.grid(False)
    ax.set_title(f"(c) 상황별 90% 도달 에폭 — 작을수록 먼저 배운다 · — = 변화 ≤ 요동 · 칠하지 않은 회색 숫자 = 그 지표 "
                 f"모집단 n < {C.MIN_N}", loc="left", fontsize=10.5)
    fig.suptitle(f"무엇이 먼저 배워지나 — 지표마다 '최종값까지 변화의 90%' 에 처음 닿는 에폭과 에폭 사이 요동 · {tag}",
                 x=0.01, y=0.975, ha="left", fontsize=13)
    note(fig, f"최종 = 평가한 마지막 {LAST_K} 에폭 평균. 요동 = 에폭 번호가 중앙 이상인 연속 에폭 쌍의 |차이| 중앙값. "
              "첫 에폭 체크포인트는 이미 한 에폭(약 6,250 스텝)을 학습한 뒤라 그 안의 순서는 보이지 않는다. "
              "행 순서 = 전체의 90% 도달 에폭 순.", fs=8)
    return C.savefig(fig, path)


# --------------------------------------------------------------------------- e5 끝 1초 감속
def fig_end_decel(ED, epochs, panel_epochs, path, tag=""):
    plt = C.setup_mpl()
    from matplotlib.lines import Line2D
    ec = C.epoch_colors(panel_epochs)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.8, bottom=0.14, wspace=0.22)
    t = np.asarray(ED["t"])
    a = ax[0]
    a.axvspan(5.45, 6.05, color=C.GRID, alpha=0.8, lw=0, zorder=0)
    a.plot(t, ED["gt_ratio"], color=C.C_GT, lw=2.0, marker="o", ms=3, zorder=4)
    a.plot(t, ED["field_ratio"], color=C.MUTED, lw=1.4, ls=(0, (3, 2)), zorder=4)
    for e in panel_epochs:
        a.plot(t, ED["epoch"][e]["top_ratio"], color=ec[e], lw=1.9, zorder=3, **C.step_kw(ms=2.6))
    a.axhline(1, color=C.INK2, lw=0.8, ls=(0, (4, 2)))
    a.set_xlabel("예측 시간 [s] · 점 = 0.1 s 스텝")
    a.set_ylabel("속력 ÷ (4.1–5.0 s 평균) — 중앙값")
    a.set_title("(a) 끝 3초 속력 비 — 확률 1위 모드, 에폭별", loc="left", fontsize=10.5)
    hs = [Line2D([], [], color=C.C_GT, lw=2, marker="o", ms=3), Line2D([], [], color=C.MUTED, lw=1.4, ls=(0, (3, 2)))]
    hs += [Line2D([], [], color=ec[e], lw=1.9, marker="o", ms=2.6) for e in panel_epochs]
    a.legend(hs, ["정답 (위치 차분)", "정답 (AV2 속도 필드·평활)"] + [f"에폭 {e}" for e in panel_epochs],
             loc="lower left", fontsize=8)
    a.text(5.75, 0.98, "라벨 끝\n인공 감속", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=7.5,
           color=C.INK2)
    ep = np.asarray(epochs)
    for axx, keys, gt_key, fld_key, ylab, ttl in (
            (ax[1], ("top_deficit", "win_deficit"), "gt_deficit", "field_deficit", "끝 1초 부족 거리 [m] (중앙값)",
             "(b) 등속 대비 끝 1초에 덜 간 거리"),
            (ax[2], ("top_a_last6", "win_a_last6"), "gt_a_last6", "field_a_last6", "5.5–6.0 s 평균 가속도 [m/s²]",
             "(c) 끝 0.6초 평균 가속도")):
        axx.axhline(ED[gt_key], color=C.C_GT, lw=1.6, zorder=2)
        axx.axhline(ED[fld_key], color=C.MUTED, lw=1.4, ls=(0, (3, 2)), zorder=2)
        for k_, col in zip(keys, (C.C_TOP, C.C_WIN)):
            axx.plot(ep, [ED["epoch"][e][k_] for e in epochs], color=col, lw=2.0, marker="o", ms=4.2, mec=C.SURF,
                     mew=1.0, zorder=3)
        axx.set_xlabel("에폭")
        axx.set_ylabel(ylab)
        axx.set_title(ttl, loc="left", fontsize=10.5)
        epoch_axis(axx, epochs)
        span = max(ep.max() - ep.min(), 1)
        axx.set_xlim(ep.min() - 0.5, ep.max() + 0.3 * span + 0.5)
        fig.canvas.draw()
        end_labels(axx, [(ep[-1], ED["epoch"][epochs[-1]][keys[0]], "확률 1위"),
                         (ep[-1], ED["epoch"][epochs[-1]][keys[1]], "승자"),
                         (ep[-1], ED[gt_key], "정답 (위치)"), (ep[-1], ED[fld_key], "정답 (속도 필드)")])
    fl = ED["epoch"]
    k3 = min(LAST_K, len(epochs))
    first, last = epochs[:k3], epochs[-k3:]
    mean_ = lambda es, key: float(np.mean([fl[e][key] for e in es]))
    sd_ = lambda es, key: float(np.std([fl[e][key] for e in es], ddof=1)) if len(es) > 1 else float("nan")
    early = epochs[:min(10, len(epochs))]
    late = epochs[-min(10, len(epochs)):]
    fig.suptitle(f"끝 1초 감속 추종 — 정답의 끝 0.5초 인공 감속을 모델이 에폭에 따라 얼마나 따라가나 "
                 f"(이동 시나리오 n={ED['n']:,}) · {tag}", x=0.01, y=0.975, ha="left", fontsize=13)
    fig.text(0.01, 0.885, f"추종 비율(모델 부족 거리 ÷ 정답 부족 거리, 중앙값끼리), 처음 {k3} → 마지막 {k3} 에폭 평균: "
                          f"1위 {mean_(first, 'top_follow'):.2f} → {mean_(last, 'top_follow'):.2f} · "
                          f"승자 {mean_(first, 'win_follow'):.2f} → {mean_(last, 'win_follow'):.2f} "
                          f"(단일 에폭 {epochs[0]} → {epochs[-1]}: 1위 {fl[epochs[0]]['top_follow']:.2f} → "
                          f"{fl[epochs[-1]]['top_follow']:.2f}). 1위 추종의 에폭 간 표준편차: 처음 {len(early)} 에폭 "
                          f"{sd_(early, 'top_follow'):.2f} · 마지막 {len(late)} 에폭 {sd_(late, 'top_follow'):.2f}.  "
                          f"정답(위치) {ED['gt_deficit']:.2f} m · 정답(속도 필드) {ED['field_deficit']:.2f} m",
             fontsize=9.2, color=C.INK2)
    note(fig, f"모집단 = 4.1–5.0 s 정답(위치 차분) 속력 > {MOVE_REF_V:g} m/s 이고 평가한 모든 에폭에서 1위·승자 모드의 같은 구간 "
              f"속력 > {MODEL_REF_V:g} m/s ({ED['n_moving']:,} 중 {ED['n']:,}). 부족 거리 = Σ_(5.1–6.0 s) (기준 속력 − 속력)·0.1 s.")
    return C.savefig(fig, path)


# --------------------------------------------------------------------------- e6 뒤반 변화 결정 트리 (요건 11)
LT_MID, LT_DEPTH, LT_LEAF, LT_SEED = (11, 15), 3, 200, 0
LT_FEATS = [("v0", "v0 [m/s]"), ("absdh", "|Δh| [°]"), ("absa", "최대|a| [m/s²]"), ("n_distinct", "분기 수"),
            ("n_reachable", "도달 차로 수"), ("fallback", "폴백"), ("has_lead49", "앞차 있음"),
            ("thw_f", "시간간격 [s]"), ("move6", "6초 이동 [m]"), ("coverage", "경로 커버리지")]


def late_tree(rows, df, epochs, R, path, suspect=None, tag=""):
    """시나리오별 minADE6 의 뒤반 변화(에폭 11–15 평균 → 마지막 5 에폭 평균)를 시나리오 특성으로 가르는 얕은 트리.
    서술용이다 — 같은 체크포인트 순서라 학습률 감소와 추가 학습이 섞여 있다."""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    ep = list(epochs)
    mid = [e for e in ep if LT_MID[0] <= e <= LT_MID[1]]
    late = ep[-5:]
    if len(ep) < 20 or len(mid) < 5 or set(mid) & set(late):
        return None
    A = lambda es: np.mean([rows[e][1]["minade"].astype(np.float64) for e in es], axis=0)
    y_mid, y_late = A(mid), A(late)
    dy = y_late - y_mid
    X = {}
    X["v0"] = df["v0"].to_numpy(np.float64)
    X["absdh"] = np.nan_to_num(np.abs(df["dh6"].to_numpy(np.float64)))
    X["absa"] = np.maximum(np.abs(df["amax_f"].to_numpy(np.float64)), np.abs(df["amin_f"].to_numpy(np.float64)))
    X["n_distinct"] = df["n_distinct"].to_numpy(np.float64)
    X["n_reachable"] = df["n_reachable"].to_numpy(np.float64)
    X["fallback"] = df["fallback"].to_numpy(np.float64)
    X["has_lead49"] = df["has_lead49"].to_numpy(np.float64)
    X["thw_f"] = np.where(np.isfinite(df["thw49"].to_numpy(np.float64)), df["thw49"].to_numpy(np.float64), 99.0)
    X["move6"] = df["move6"].to_numpy(np.float64)
    X["coverage"] = df["coverage"].to_numpy(np.float64)
    for c in C.CLASSES:
        X[f"cls_{c}"] = (df["cls"].to_numpy() == c).astype(float)
    names = list(X)
    labels = dict(LT_FEATS) | {f"cls_{c}": f"상황={c}" for c in C.CLASSES}
    XM = np.column_stack([X[n] for n in names])
    lo, hi = np.percentile(dy, [1, 99])
    yc = np.clip(dy, lo, hi)
    idx = np.arange(len(dy))
    tr, te = train_test_split(idx, test_size=0.3, random_state=LT_SEED)
    is_te = np.zeros(len(dy), bool)
    is_te[te] = True
    est = DecisionTreeRegressor(max_depth=LT_DEPTH, min_samples_leaf=LT_LEAF, random_state=LT_SEED)
    est.fit(XM[tr], yc[tr])
    r2_tr, r2_te = r2_score(yc[tr], est.predict(XM[tr])), r2_score(yc[te], est.predict(XM[te]))
    t = est.tree_
    dpath = est.decision_path(XM).tocsc()
    stats = {}
    for j in range(t.node_count):
        rr = dpath[:, j].nonzero()[0]
        v = dy[rr]
        rt = rr[is_te[rr]]
        stats[j] = {"n": len(rr), "mean": float(v.mean()), "median": float(np.median(v)),
                    "mean_clip": float(yc[rr].mean()), "mid": float(y_mid[rr].mean()), "late": float(y_late[rr].mean()),
                    "share_better": float((v < -0.1).mean()), "share_worse": float((v > 0.1).mean()),
                    "n_test": int(len(rt)), "mean_test": float(dy[rt].mean()) if len(rt) else float("nan"),
                    "ci_test": float(1.96 * dy[rt].std(ddof=1) / np.sqrt(len(rt))) if len(rt) > 1 else float("nan"),
                    "rows": rr}
    bound = max(abs(v["mean_clip"]) for v in stats.values()) or 1.0
    NEG = ["#eef4fc", C.BLUE_RAMP[0], C.BLUE_RAMP[3], C.BLUE_RAMP[6], C.BLUE_RAMP[8], C.BLUE_RAMP[10]]
    POS = ["#fdeee6", "#fbd5c1", "#f7b596", "#f2916a", "#eb6834", "#c24f22"]

    def col(v):
        q = min(1.0, abs(v) / bound)
        ramp = NEG if v < 0 else POS
        k = int(round(q * (len(ramp) - 1)))
        return ramp[k], ("white" if q > 0.75 else C.INK)

    def thr_txt(x):
        return f"{x:.3e}" if 0 < abs(x) < 1e-3 else f"{x:.4g}"

    BIN = {"fallback": ("폴백 아님", "폴백"), "has_lead49": ("앞차 없음", "앞차 있음"),
           "coverage": ("커버리지 실패", "커버리지 있음")} | {f"cls_{c}": (f"상황 ≠ {c}", f"상황 = {c}") for c in C.CLASSES}

    def edge_labels(nm, th):
        if nm in BIN:
            return BIN[nm]
        if nm == "v0" and th < 0.05:
            return "v0 = 0 (정지 출발)", "v0 > 0"
        return f"{labels[nm]} ≤ {thr_txt(th)}", f"{labels[nm]} > {thr_txt(th)}"

    def simple_rule(rx):
        """같은 특성의 조건을 합쳐 한 줄로 (lo < 특성 ≤ hi)."""
        out, seen = [], []
        for nm, _, _ in rx:
            if nm not in seen:
                seen.append(nm)
        for nm in seen:
            ups = [th for n2, op, th in rx if n2 == nm and op == "<="]
            los = [th for n2, op, th in rx if n2 == nm and op == ">"]
            if nm in BIN:
                out.append(BIN[nm][1] if los else BIN[nm][0])
            elif nm == "v0" and ups and min(ups) < 0.05 and not los:
                out.append("v0 = 0")
            elif nm == "v0" and los and max(los) < 0.05 and not ups:
                out.append("v0 > 0")
            else:
                lab = labels[nm]
                lo = (0.0 if nm == "v0" and max(los) < 0.05 else max(los)) if los else None
                hi = min(ups) if ups else None
                if lo is not None and hi is not None:
                    out.append(f"{thr_txt(lo)} < {lab} ≤ {thr_txt(hi)}")
                elif lo is not None:
                    out.append(f"{lab} > {thr_txt(lo)}")
                else:
                    out.append(f"{lab} ≤ {thr_txt(hi)}")
        return " · ".join(out)

    def build(j, rule, rx):
        st = stats[j]
        fc, tc = col(st["mean_clip"])
        leaf = t.children_left[j] == -1
        nd = {"id": j, "rule": list(rule), "rule_exact": [list(z) for z in rx], "n": st["n"], "color": fc, "tc": tc,
              "fs": 9.0 if leaf else 9.6,
              "text": f"n={st['n']:,}\nΔ 평균 {st['mean']:+.3f} m\n{st['mid']:.2f} → {st['late']:.2f} m"
                      + (f"\n검증30% {st['mean_test']:+.3f} ± {st['ci_test']:.3f}" if leaf else "")}
        if not leaf:
            nm = names[t.feature[j]]
            th = float(t.threshold[j])
            l_lab, r_lab = edge_labels(nm, th)
            nd["children"] = [(l_lab, build(int(t.children_left[j]), rule + [l_lab], rx + ((nm, "<=", th),))),
                              (r_lab, build(int(t.children_right[j]), rule + [r_lab], rx + ((nm, ">", th),)))]
        return nd

    root = build(0, [], ())
    leaves = C.tree_leaves(root)
    sids = df["sid"].to_numpy()
    fb = df["fallback"].to_numpy()
    sus = np.zeros(len(dy), bool) if suspect is None else np.asarray(suspect, bool)
    out_leaves, check = [], []
    for lf in leaves:
        st = stats[lf["id"]]
        rr = st["rows"]
        cand = rr[~fb[rr] & ~sus[rr]] if (~fb[rr] & ~sus[rr]).any() else rr
        rep = int(cand[np.argmin(np.abs(dy[cand] - st["median"]))])
        m = np.ones(len(dy), bool)
        for nm, op, th in lf["rule_exact"]:
            cc = XM[:, names.index(nm)]
            m &= (cc <= th) if op == "<=" else (cc > th)
        check.append(bool(int(m.sum()) == st["n"] and abs(float(dy[m].mean()) - st["mean"]) < 1e-9))
        out_leaves.append({"leaf": lf["id"], "rule": simple_rule(lf["rule_exact"]), "rule_path": " · ".join(lf["rule"]),
                           "rule_exact": lf["rule_exact"],
                           "n": st["n"], "mean": st["mean"], "median": st["median"], "mid": st["mid"],
                           "late": st["late"], "share_better": st["share_better"], "share_worse": st["share_worse"],
                           "n_test": st["n_test"], "mean_test": st["mean_test"], "ci_test": st["ci_test"],
                           "value_train_clip": float(t.value[lf["id"]][0][0]),
                           "rep_idx": rep, "rep_sid": str(sids[rep]), "rep_cls": str(df["cls"].iloc[rep]),
                           "rep_dy": float(dy[rep])})
    imp = sorted(zip(names, est.feature_importances_), key=lambda q: -q[1])
    res = {"target": f"시나리오별 minADE6: 마지막 5 에폭({late[0]}–{late[-1]}) 평균 − 에폭 {mid[0]}–{mid[-1]} 평균 [m] "
                     "(음수 = 뒤반에 좋아짐). 학습은 p1–p99 로 자른 값",
           "clip": [float(lo), float(hi)], "depth": LT_DEPTH, "min_leaf": LT_LEAF, "seed": LT_SEED,
           "r2_train": float(r2_tr), "r2_test": float(r2_te),
           "mean_dy": float(dy.mean()), "corr_mid_level_vs_dy": float(np.corrcoef(y_mid, dy)[0, 1]),
           "importance": [(labels[n], float(v)) for n, v in imp if v > 0],
           "leaves": out_leaves, "leaf_check_ok": all(check),
           "note": "서술용 — 인과가 아니다. |Δh|·최대|a|·6초 이동·커버리지는 정답(미래)에서 온다. "
                   "중간 오차가 큰 시나리오일수록 많이 줄어드는 평균으로의 회귀가 섞여 있다."}
    # 그림
    plt = C.setup_mpl()
    n_leaf = len(leaves)
    fig = plt.figure(figsize=(max(22, 2.9 * n_leaf), 15.5))
    ax = fig.add_axes([0.005, 0.37, 0.99, 0.49])
    C.draw_tree(ax, root, fontsize=10.0, edge_fs=9.6)
    fig.suptitle(f"e6 · 뒤반 변화 결정 트리 — 시나리오별 minADE6 이 에폭 {mid[0]}–{mid[-1]} 평균 → {late[0]}–{late[-1]} 평균 "
                 f"사이에 얼마나 변했나 (n={len(dy):,}) · {tag}", x=0.01, y=0.985, ha="left", fontsize=14.5)
    fig.text(0.01, 0.948, f"sklearn DecisionTreeRegressor · 깊이 {LT_DEPTH} · 잎 ≥ {LT_LEAF} · 학습 70% / 검증 30% (시드 {LT_SEED}) · "
                          f"목표는 p1–p99 ({lo:+.2f} ~ {hi:+.2f} m) 로 잘라 학습 · 설명력 R² 학습 {r2_tr:.3f} / 검증 {r2_te:.3f} "
                          f"(시나리오 하나하나의 변화는 거의 설명되지 않는다) · 전체 평균 Δ {dy.mean():+.3f} m",
             fontsize=10.4, color=C.INK2)
    fig.text(0.01, 0.925, "노드 글 = 그 칸 전체의 (자르지 않은) Δ 평균 · 에폭 11–15 → 마지막 5 에폭 minADE6 평균 · (잎) 검증 30% 평균 ± 95% 반폭.  "
                          "색 = 자른 Δ 평균 (파랑 = 뒤반에 줄었다, 주황 = 늘었다).",
             fontsize=10.2, color=C.INK2)
    fig.text(0.01, 0.903, f"서술용 요약이지 인과가 아니다 — 뒤반에는 학습률 감소와 추가 학습이 함께 있다 (대조 학습 없음). "
                          f"에폭 11–15 오차가 큰 시나리오일수록 Δ 가 음수로 크다 (상관 {res['corr_mid_level_vs_dy']:+.2f}): "
                          "평균으로의 회귀와 '어려운 칸이 뒤늦게 줄었다' 가 섞여 있다.",
             fontsize=10.2, color=C.INK2)
    axt = fig.add_axes([0.01, 0.045, 0.98, 0.3])
    axt.axis("off")
    cols = ["잎", "규칙 (같은 특성은 합침)", "n", "Δ 평균 / 중앙 [m]", "11–15 → 뒤반 [m]", "줄어듦 / 늘어남 (|Δ|>0.1)",
            "검증 30% Δ ± 95%", "대표 시나리오 (중앙값에 가장 가까운)"]
    cells = []
    for r in out_leaves:
        cells.append([str(r["leaf"]), r["rule"], f"{r['n']:,}", f"{r['mean']:+.3f} / {r['median']:+.3f}",
                      f"{r['mid']:.2f} → {r['late']:.2f}", f"{r['share_better'] * 100:.0f}% / {r['share_worse'] * 100:.0f}%",
                      f"{r['mean_test']:+.3f} ± {r['ci_test']:.3f}", f"{r['rep_sid'][:8]} · {r['rep_cls']} · Δ {r['rep_dy']:+.2f}"])
    tb = axt.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[0.035, 0.33, 0.06, 0.11, 0.1, 0.11, 0.11, 0.145])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10.0)
    tb.scale(1, 1.75)
    for (rr, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(C.GRID)
        if rr == 0:
            cell.set_facecolor("#eef0f3")
            cell.set_text_props(weight="bold")
        elif cc == 3:
            fc, tc = col(out_leaves[rr - 1]["mean"])
            cell.set_facecolor(fc)
            cell.set_text_props(color=tc)
        if cc == 1 and rr > 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    note(fig, "특성: " + " · ".join(labels[n] for n in names if not n.startswith("cls_")) + " · 상황 9개(원-핫). "
              "|Δh|·최대|a|·6초 이동·커버리지는 정답(미래)에서 온다. 대표 시나리오는 폴백·위치 기준 직진 의심 회전 라벨을 뺀 중에서 고름. "
              "시드 1개 · 체크포인트 1벌이라 에폭 사이 흔들림과 구분하지 못한다.", fs=9.0)
    C.savefig(fig, path)
    print(f"[e6] 뒤반 트리 R² 학습 {r2_tr:.3f} 검증 {r2_te:.3f} · 잎 {n_leaf} · 대조 {'통과' if all(check) else '실패'}",
          flush=True)
    return res


# --------------------------------------------------------------------------- 고정 시나리오 패널
class PanelCtx:
    def __init__(self, df, R, cache_dir, scen, panel_epochs, hz=10, extra=None, tag=""):
        from dataset_cached import CachedV4Dataset
        ds = CachedV4Dataset(cache_dir)
        self.df, self.R = df, R
        self.cache = {k: ds.raw(k) for k in ("routes", "route_tan", "route_band", "route_len", "route_mask",
                                             "origin", "theta")}
        self.z = scen
        self.row = {int(e): r for r, e in enumerate(scen["epochs"])}
        self.pe = [e for e in panel_epochs if e in self.row]
        self.ec = C.epoch_colors(self.pe)
        self.hz = hz
        self.tag = tag
        self.extra = extra or {}          # {에폭: {"h","theta","d","band",...} (고정 시나리오, 6모드, 60스텝)}
        self._scene = {}
        # 상황 분류 인공물 후보 (분류기는 그대로, 패널 제목에만 표시)
        self.turn_straight, self.turn_alt = C.turn_straight_by_position(R["pos"], df["cls"].to_numpy())
        pos = np.asarray(R["pos"], np.float64)
        path6 = np.linalg.norm(np.diff(pos[:, C.OBS - 1:], axis=1), axis=2).sum(1)
        turn = df["cls"].isin(["좌회전", "우회전"]).to_numpy()
        self.turn_flat = turn & (np.abs(pos[:, -1, 1]) < 1.0) & (path6 >= 4.0) & ~self.turn_straight
        self.dec_acc = (df["cls"] == "급감속").to_numpy() & (df["amax_f"].to_numpy() > C.ACC_A)

    def artifact_note(self, i):
        r = self.df.iloc[i]
        if self.turn_straight[i]:
            return (f"※ 분류 인공물 후보: '{r['cls']}' 라벨이지만 위치로는 직진(시작·끝 방향 차 {self.turn_alt[i]:+.0f}°)")
        if self.turn_flat[i]:
            e = self.R["pos"][i, -1]
            return (f"※ 분류 인공물 후보: '{r['cls']}' 라벨(6초 Δh {r['dh6']:+.0f}°)이지만 끝점 ({e[0]:.1f}, {e[1]:+.1f}) m 로 "
                    "거의 직진 — 정지 직전 방향 잡음이 Δh 를 만든 것으로 보인다")
        if self.dec_acc[i]:
            return (f"※ 경계 사례: '급감속'(최소 a {r['amin_f']:+.2f} m/s²)이지만 최대 a {r['amax_f']:+.2f} m/s² 로 "
                    "급가속 임계도 넘는다")
        return ""

    def gt_route_frame(self, i):
        """정답 기준 경로에서 정답의 (d, 밴드, 경로 접선각) — viz_v4_dump.raw_one 과 같은 계산."""
        import lane_frame as lf
        g = int(self.df["gt_route"].iloc[i])
        ch = self.cache
        rd = C.route_dict(ch["routes"][i, g], ch["route_tan"][i, g], ch["route_len"][i, g])
        ln = float(ch["route_len"][i, g])
        pos = np.asarray(self.R["pos"][i], np.float64)
        s, d, _ = lf.to_frame(pos, rd)
        bd = D._band_at(np.asarray(ch["route_band"][i, g], np.float64), ln, s)
        tk = D._band_at(np.asarray(ch["route_tan"][i, g], np.float64), ln, s)
        return d, bd, np.arctan2(tk[:, 1], tk[:, 0]), rd

    def proj_gt(self, i, rd, traj):
        """예측 궤적(60,2)을 정답 기준 경로에 투영한 d — 정답과 같게 과거 50스텝부터 잇는다."""
        import lane_frame as lf
        q = np.vstack([np.asarray(self.R["pos"][i, :C.OBS], np.float64), np.asarray(traj, np.float64)])
        return lf.to_frame(q, rd)[1][C.OBS:]

    def scene(self, i):
        import viz_v4_cases as VC
        if i not in self._scene:
            sid = self.df["sid"].iloc[i]
            o, th = self.cache["origin"][i], float(self.cache["theta"][i])
            self._scene[i] = (C.build_scene(sid, o, th), VC.others_at_obs(sid, o, th))
        return self._scene[i]


def halo(lw):
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw + 2.4, foreground="white"), pe.Normal()]


def draw_panel_map(ax, cx, j, i, compact=False):
    z, R, df = cx.z, cx.R, cx.df
    nd = int(df["n_distinct"].iloc[i])
    g = int(df["gt_route"].iloc[i])
    pos = R["pos"][i]
    trajs = []
    for e in cx.pe:
        r = cx.row[e]
        top, win = int(z["top1"][r, i]), int(z["winner"][r, i])
        trajs.append((e, z["pn_traj"][r, j, top], z["pn_traj"][r, j, win] if win != top else None))
    pts = np.concatenate([pos[20:]] + [t for _, t, _ in trajs] + [w for _, _, w in trajs if w is not None])
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    ratio = (bb.height * fh) / max(bb.width * fw, 1e-6)
    hx, hy = max((hi - lo)[0] * 0.6, 16.0), max((hi - lo)[1] * 0.6, 16.0 * min(ratio, 1.0))
    if hy / hx < ratio:
        hy = hx * ratio
    else:
        hx = hy / ratio
    xlim, ylim = (c[0] - hx, c[0] + hx), (c[1] - hy, c[1] + hy)
    scene, others = cx.scene(i)
    C.draw_scene_light(ax, scene, xlim, ylim, mark_lw=0.8 if compact else 1.0)
    rm = cx.cache["route_mask"][i]
    for k in range(nd):
        if not rm[k]:
            continue
        rp = cx.cache["routes"][i][k]
        on = k == g
        ax.plot(rp[:, 0], rp[:, 1], color=C.INK2 if on else C.MUTED, lw=1.3 if on else 0.8,
                ls="-" if on else (0, (3, 2)), alpha=0.9, zorder=6)
        if not compact:
            inside = np.where((rp[:, 0] > xlim[0] + 2) & (rp[:, 0] < xlim[1] - 2)
                              & (rp[:, 1] > ylim[0] + 2) & (rp[:, 1] < ylim[1] - 2))[0]
            if len(inside):
                q = inside[int(len(inside) * (0.5 + 0.09 * k)) % len(inside)]
                ax.text(rp[q, 0], rp[q, 1], f"r{k}" + (" 정답" if on else ""), fontsize=7, color=C.INK2, zorder=7,
                        clip_on=True, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    for x, y, yaw, typ in others:
        if xlim[0] - 5 < x < xlim[1] + 5 and ylim[0] - 5 < y < ylim[1] + 5 and typ in ("vehicle", "bus",
                                                                                          "motorcyclist"):
            C.draw_box(ax, x, y, yaw, "#b9b6ad", alpha=0.9, zorder=8,
                       length=2.2 if typ == "motorcyclist" else (11.0 if typ == "bus" else 4.6),
                       width=0.9 if typ == "motorcyclist" else (2.5 if typ == "bus" else 1.9))
    ax.plot(pos[:C.OBS, 0], pos[:C.OBS, 1], color=C.C_PAST, lw=2.2, zorder=10)
    C.past_dots(ax, pos[:C.OBS], cx.hz, C.C_PAST, zorder=10.3)
    for e, tt, ww in trajs:
        col = cx.ec[e]
        if ww is not None:
            C.origin_join(ax, ww[0], col, lw=1.4, zorder=10.6)
            ax.plot(ww[:, 0], ww[:, 1], color=col, lw=1.4, ls=(0, (3, 2)), zorder=11, path_effects=halo(1.4))
        C.origin_join(ax, tt[0], col, lw=2.2, zorder=10.8)
        ax.plot(tt[:, 0], tt[:, 1], color=col, lw=2.2, zorder=12, path_effects=halo(2.2))
        ax.scatter(*tt[-1], s=26, color=col, edgecolors="white", linewidths=1.0, zorder=13)
    ax.plot(pos[C.OBS - 1:, 0], pos[C.OBS - 1:, 1], color=C.C_GT, lw=1.7, zorder=14)
    ax.scatter(*pos[-1], s=30, marker="s", color=C.C_GT, edgecolors="white", linewidths=1.0, zorder=15)
    C.draw_box(ax, 0, 0, 0, "none", zorder=16, ec=C.INK, lw=1.4)
    C.scale_bar(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(C.AXIS)


def map_legend(ax, cx, loc="upper left", fs=7.6, ncol=1):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=C.C_PAST, lw=2.2, marker="o", ms=3.5, mec=C.SURF),
          Line2D([], [], color=C.C_GT, lw=1.7, marker="s", ms=5)]
    hs += [Line2D([], [], color=cx.ec[e], lw=2.2) for e in cx.pe]
    hs += [Line2D([], [], color=C.INK2, lw=1.4, ls=(0, (3, 2))), Line2D([], [], color=C.INK2, lw=1.3),
           Line2D([], [], color=C.MUTED, lw=0.8, ls=(0, (3, 2))), Line2D([], [], color=C.INK2, lw=0.9, ls=(0, (1.0, 1.4)))]
    labs = ["과거 5초 (· 1초" + (" · ○ 2 Hz 입력 시점)" if cx.hz == 2 else ")"), "정답 6초 (■ 끝)"]
    labs += [f"에폭 {e} 확률 1위 (● 끝)" for e in cx.pe]
    labs += ["점선 = 그 에폭의 승자 (1위와 다를 때)", "정답 기준 경로", "다른 후보 경로", "원점 → 첫 예측점 (t = 0.1 s)"]
    leg = ax.legend(hs, labs, loc=loc, fontsize=fs, frameon=True, facecolor="white", edgecolor=C.GRID,
                    framealpha=0.92, ncol=ncol, handlelength=2.2)
    leg.set_zorder(100)          # 지도 요소 위에
    return leg


def panel_figure(cx, j, pick, path):
    """고정 시나리오 한 장: 지도(에폭별 1위·승자) | 모드 확률·요약표 | 시계열 6칸(v · a · h · θ · d · 앞차 거리, 필수 요건 1)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    i = pick["idx"]
    z, R, df = cx.z, cx.R, cx.df
    r_ = df.iloc[i]
    nd, g = int(r_["n_distinct"]), int(r_["gt_route"])
    fig = plt.figure(figsize=(27, 12.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 0.95, 2.0], left=0.01, right=0.99, top=0.87, bottom=0.06,
                  wspace=0.14)
    axm = fig.add_subplot(gs[0, 0])
    draw_panel_map(axm, cx, j, i)
    map_legend(axm, cx)
    mid = gs[0, 1].subgridspec(2, 1, hspace=0.32, height_ratios=[1.0, 1.1])
    # 모드 확률 막대
    axp = fig.add_subplot(mid[0])
    x = np.arange(6)
    alive = z["pn_alive"][cx.row[cx.pe[-1]], j]
    w = min(0.8 / max(len(cx.pe), 1), 0.16)
    top_v = 0.0
    for q, e in enumerate(cx.pe):
        r = cx.row[e]
        pr = z["pn_prob"][r, j]
        xs = x + (q - (len(cx.pe) - 1) / 2) * w
        axp.bar(xs[alive], pr[alive], width=w * 0.86, color=cx.ec[e], zorder=3)
        wv = int(z["winner"][r, i])
        axp.text(xs[wv], pr[wv] + 0.01, "★", ha="center", va="bottom", fontsize=7.5, color=C.INK, zorder=4)
        top_v = max(top_v, float(pr[alive].max()))
    labels = []
    for m in x:
        if not alive[m]:
            labels.append(f"m{m}\n(없음)")
        else:
            labels.append(f"m{m} · r{m % nd}" + ("\n정답 경로" if m % nd == g else ""))
    axp.set_xticks(x)
    axp.set_xticklabels(labels, fontsize=8)
    axp.set_ylim(0, max(top_v * 1.22, 0.05))
    axp.set_ylabel("모드 확률")
    axp.grid(axis="x", visible=False)
    axp.set_title("모드 확률 — 막대 색 = 에폭 (왼쪽부터 에폭 순)\n★ = 그 에폭의 승자", loc="left", fontsize=10)
    # 에폭 요약 표
    axt = fig.add_subplot(mid[1])
    axt.axis("off")
    cols = ["에폭", "확률 1위 (확률)", "승자", "minADE6", "1위 ADE", "정답 경로 확률"]
    cells, colors = [], []
    for e in cx.pe:
        r = cx.row[e]
        t, wv = int(z["top1"][r, i]), int(z["winner"][r, i])
        cells.append([f"{e}", f"m{t} · r{t % nd} ({z['top1_prob'][r, i]:.2f})", f"m{wv} · r{wv % nd}",
                      f"{z['minade'][r, i]:.2f} m", f"{z['top1_ade'][r, i]:.2f} m",
                      f"{z['gt_prob'][r, i]:.2f}" + ("" if nd >= 2 else " (분기 1)")])
        colors.append(cx.ec[e])
    tb = axt.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[0.09, 0.27, 0.15, 0.15, 0.15, 0.19])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.6)
    tb.scale(1, 1.7)
    for (rr, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(C.GRID)
        cell.set_linewidth(0.6)
        if rr == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(color=C.INK2, weight="bold")
        elif cc == 0:
            cell.set_facecolor(colors[rr - 1])
            cell.set_text_props(color="white", weight="bold")
    axt.set_title("에폭별 요약 — 1위·승자 = 슬롯 m · 후보 경로 r", loc="left", fontsize=10)
    lines = [f"상황 {r_['cls']} · 6초 Δh {r_['dh6']:+.0f}° · 경로 횡변위 Δd {r_['dd6']:+.1f} m · v0 {r_['v0']:.1f} m/s",
             f"미래 가속도 최소 {r_['amin_f']:+.1f} · 최대 {r_['amax_f']:+.1f} m/s² (속도 필드 평활) · 6초 이동 {r_['move6']:.0f} m",
             f"구별 분기 {nd} · 정답 기준 경로 r{g} ({r_['gt_route_q']}) · 경로 커버리지 "
             f"{'있음' if r_['coverage'] else '실패'} · 앞차 "
             + (f"{r_['gap49']:.0f} m" if np.isfinite(r_['gap49']) else "없음")]
    axt.text(0.0, 0.0, "\n".join(lines), transform=axt.transAxes, fontsize=8.6, color=C.INK2, va="bottom",
             linespacing=1.6)
    # 시계열 6칸
    t_ax, tp = C.T_AX, C.T_PRED
    d_g, bd_g, k_g, rd = cx.gt_route_frame(i)
    lc = C.lc_interval(R["gt_d_g"][i], r_["cls"]) if "gt_d_g" in R else None
    hg = np.asarray(R["h"][i], np.float64)
    hg = hg - 2 * np.pi * np.round(hg[C.OBS - 1] / (2 * np.pi))
    right = gs[0, 2].subgridspec(3, 2, hspace=0.46, wspace=0.18)
    series = [("v", "속도 v [m/s] — 회색 띠 = 창 양 끝 램프(라벨 끝 0.5 s 인공 감속)"),
              ("a", "가속도 a [m/s²] — 정답 = 속도 필드 평활의 변화율"),
              ("h", "진행방향 h [°]"),
              ("theta", "잔차각 θ [°] — 에폭 선은 자기 모드 경로 기준, 정답은 정답 경로 기준"),
              ("d", "정답 경로 기준 횡오프셋 d [m] (+좌) — 에폭 선은 정답 경로에 투영, 청록 = 밴드"),
              ("lead", "앞차 거리 [m] (경로 따라 중심 간)")]
    have_extra = all(e in cx.extra for e in cx.pe)
    for q, (key, ttl) in enumerate(series):
        ax = fig.add_subplot(right[q // 2, q % 2])
        ax.axvline(0, color=C.INK2, lw=0.8, zorder=1)
        C.shade_lc(ax, lc, label=(key == "d"))
        if key in ("v", "a"):
            ax.axvspan(5.45, 6.0, color=C.GRID, alpha=0.8, lw=0, zorder=0)
            ax.axvspan(-4.9, -4.45, color=C.GRID, alpha=0.8, lw=0, zorder=0)
        vals = []
        if key == "v":
            ax.plot(t_ax, R["v_pos"][i], color=C.C_GT, lw=1.7, zorder=4, **C.step_kw())
            ax.plot(t_ax, R["v_fld"][i], color=C.MUTED, lw=1.1, ls=(0, (3, 2)), zorder=4)
        elif key == "a":
            ax.plot(t_ax, R["a_fld"][i], color=C.C_GT, lw=1.7, zorder=4, **C.step_kw())
            vals.append(np.asarray(R["a_fld"][i], np.float64))
        elif key == "h":
            ax.plot(t_ax, np.degrees(hg), color=C.C_GT, lw=1.6, zorder=4, **C.step_kw())
            vals.append(np.degrees(hg[5:]))
        elif key == "theta":
            th = np.degrees(C.wrap(hg - k_g))
            th[np.asarray(R["v_fld"][i]) < C.MOVE_V] = np.nan
            ax.plot(t_ax, th, color=C.C_GT, lw=1.3, zorder=4, **C.step_kw())
        elif key == "d":
            ax.plot(t_ax, bd_g[:, 0], color=C.C_BAND, lw=1.0, ls=(0, (4, 2)), zorder=2)
            ax.plot(t_ax, -bd_g[:, 1], color=C.C_BAND, lw=1.0, ls=(0, (4, 2)), zorder=2)
            ax.axhline(0, color=C.AXIS, lw=0.8)
            ax.plot(t_ax, d_g, color=C.C_GT, lw=1.6, zorder=4, **C.step_kw())
            vals += [d_g[C.OBS - 10:], bd_g[C.OBS:, 0], -bd_g[C.OBS:, 1]]
        elif key == "lead":
            ld = np.asarray(R["lead_dist"][i], np.float64)
            if np.isfinite(ld).any():
                ax.plot(t_ax, ld, color=C.C_LEAD, lw=1.6, zorder=4, **C.step_kw())
                ax.set_ylim(0, min(C.LEAD_MAX_M, np.nanmax(ld) * 1.2 + 2))
            else:
                ax.text(0.5, 0.5, "앞차 없음", transform=ax.transAxes, ha="center", va="center", color=C.MUTED)
        if key != "lead":
            for e in cx.pe:
                r = cx.row[e]
                t = int(z["top1"][r, i])
                if key in ("v", "a"):
                    ys = z["pn_" + key][r, j, t]
                elif key == "d":
                    ys = cx.proj_gt(i, rd, z["pn_traj"][r, j, t])
                elif not have_extra:
                    continue
                elif key == "h":
                    hh = np.unwrap(np.asarray(cx.extra[e]["h"][j, t], np.float64))
                    ys = np.degrees(hh - 2 * np.pi * np.round((hh[0] - hg[C.OBS - 1]) / (2 * np.pi)))
                else:
                    ys = np.degrees(np.asarray(cx.extra[e]["theta"][j, t], np.float64))
                ax.plot(tp, ys, color=cx.ec[e], lw=1.7, zorder=3, **C.step_kw())
                vals.append(np.asarray(ys, np.float64))
        if key == "a" and vals:
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
        ax.set_xlim(-5, 6)
        ax.set_title(ttl, loc="left", fontsize=9.2)
        if q >= 4:
            ax.set_xlabel("시간 [s]  (0 = 예측 시작) · 점 = 0.1 s 스텝")
        if key == "v":
            hs = [Line2D([], [], color=C.C_GT, lw=1.7, marker="o", ms=C.STEP_MS_10HZ),
                  Line2D([], [], color=C.MUTED, lw=1.1, ls=(0, (3, 2)))]
            ax.legend(hs + [Line2D([], [], color=cx.ec[e], lw=1.7) for e in cx.pe],
                      ["정답 (위치 차분)", "정답 (AV2 속도 필드·평활)"] + [f"에폭 {e} 확률 1위" for e in cx.pe],
                      loc="best", fontsize=7.2, ncol=2)
    e0, e1 = cx.pe[0], cx.pe[-1]
    art = cx.artifact_note(i)
    fig.suptitle(f"[{pick['cls']} {pick['k']}{' ※' if art else ''}] {pick['sid']}  ·  {cx.tag} (입력 {cx.hz} Hz)  ·  minADE6 에폭 {e0} "
                 f"{z['minade'][cx.row[e0], i]:.2f} → 에폭 {e1} {z['minade'][cx.row[e1], i]:.2f} m  ·  1위 ADE "
                 f"{z['top1_ade'][cx.row[e0], i]:.2f} → {z['top1_ade'][cx.row[e1], i]:.2f} m", x=0.01, y=0.985,
                 ha="left", fontsize=13.5)
    pn = int(pick.get("pool_n", len(df)))
    sub = (f"고정 시나리오 (val 앞 {pn:,} 중 상황 '{pick['cls']}' {pick['pool']:,}개에서 시드 {C.SEED} 무작위, 폴백 제외). "
           "지도 = 정규화 프레임(원점 = 현재 위치, x = 현재 진행방향)."
           + (f" 지표 평가는 {pop_desc(len(df))}." if pn != len(df) else "")
           + ("" if have_extra else " (h·θ 에폭 선 없음: 고정 시나리오 재추론을 건너뛰었다)"))
    fig.text(0.01, 0.945, sub + ("\n" + art if art else ""), fontsize=9, color=C.INK2, va="top", linespacing=1.5)
    return C.savefig(fig, path)


def panel_overview(cx, picks, path):
    import matplotlib.pyplot as plt
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    arts = [f"{p['cls']} {p['k']} · {p['sid'][:8]}: {cx.artifact_note(p['idx'])[2:]}" for p in picks
            if cx.artifact_note(p["idx"])]
    foot = 0.2 * len(arts) + (0.15 if arts else 0.0)
    H = 4.9 * nrow + 1.35 + foot
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, H), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.45 / H, bottom=0.01 + foot / H, wspace=0.04,
                        hspace=0.22)
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    e0, e1 = cx.pe[0], cx.pe[-1]
    for j, (ax, p) in enumerate(zip(axes.flat, picks)):
        i = p["idx"]
        draw_panel_map(ax, cx, j, i, compact=True)
        z = cx.z
        mark = " ※" if cx.artifact_note(i) else ""
        ax.set_title(f"{p['cls']} {p['k']} · {p['sid'][:8]}{mark}\nminADE6 에폭{e0} {z['minade'][cx.row[e0], i]:.2f} → "
                     f"에폭{e1} {z['minade'][cx.row[e1], i]:.2f} m", fontsize=9.5)
    if len(picks) < nrow * ncol:
        map_legend(axes.flat[-1], cx, loc="center", fs=9)
    else:
        map_legend(axes.flat[0], cx, fs=6.8)
    fig.suptitle(f"고정 시나리오 {len(picks)}개 — 에폭 {', '.join(map(str, cx.pe))} 의 확률 1위(실선)·승자(점선) · "
                 f"{cx.tag} (입력 {cx.hz} Hz)", x=0.01, y=1 - 0.35 / H, ha="left", va="top", fontsize=14)
    if arts:
        fig.text(0.01, 0.004, "※ " + "\n※ ".join(arts), fontsize=8.6, color=C.INK2, ha="left", va="bottom")
    return C.savefig(fig, path)


def print_summary(S):
    """리포트용 핵심 숫자를 한 번에 본다 (json 과 같은 값)."""
    ep = S["epochs"]
    e0, e1 = ep[0], ep[-1]
    be = S["by_epoch"]
    print(f"\n=== 요약 · {S['tag']} · val 앞 {S['n_val']:,} · 에폭 {e0}–{e1} ===")
    LO = S.get("learning_order") or {}
    if LO:
        print(f"{'지표':28} {'첫':>9} {'최종':>9} {'변화%':>7} {'요동%':>7} {'배':>6} {'90%':>4} {'유지':>4}")
        for k, lab, fmt, _, _ in METRICS:
            s = LO["전체"].get(k)
            if not s:
                continue
            print(f"{lab:28} {fmt.format(s['first']):>9} {fmt.format(s['final']):>9} {s['change_pct']:+7.1f} "
                  f"{s['fluct_pct']:7.1f} {s['snr']:6.1f} {str(s['e90'] or '—'):>4} {str(s['e90_stay'] or '—'):>4}")
    print(f"\n{'상황':8} {'n':>5} " + " ".join(f"{h:>15}" for h in ("minADE6", "miss%", "승자≠1위%", "정답경로1위%",
                                                                   "흔들림a", "7.3°초과%", "90%:ADE/θ/a/1위")))
    for c in ["전체"] + C.CLASSES:
        r0, r1 = be[e0][c], be[e1][c]
        lo = LO.get(c, {})
        e90 = "/".join(str((lo.get(k) or {}).get("e90") or "—") for k in ("minade", "jit_theta", "jit_a", "gt_top1"))
        cells = [f"{r0[k]:.3g}→{r1[k]:.3g}" for k in ("minade", "miss", "win_ne_top1", "gt_top1", "jit_a", "exc_pct")]
        print(f"{c:8} {r1['n']:5d} " + " ".join(f"{x:>15}" for x in cells) + f" {e90:>15}")
    ED = S.get("end_decel")
    if ED:
        f0, f1 = ED["epoch"][e0], ED["epoch"][e1]
        print(f"\n끝 감속 n={ED['n']:,}: 정답(위치) {ED['gt_deficit']:.2f} m · 속도필드 {ED['field_deficit']:.2f} m · "
              f"정답 끝 0.6s a {ED['gt_a_last6']:.1f} (필드 {ED['field_a_last6']:.2f})")
        for e in ep:
            f = ED["epoch"][e]
            print(f"  에폭 {e:2d}: 1위 {f['top_deficit']:.2f} m (추종 {f['top_follow']:.2f}, a {f['top_a_last6']:.2f}) · "
                  f"승자 {f['win_deficit']:.2f} m (추종 {f['win_follow']:.2f}, a {f['win_a_last6']:.2f}) · "
                  f"1위 끝 속력비 {f['top_end_ratio']:.2f}")
        print(f"  속력 구간 {ED['speed_bins']} n {ED['speed_bin_n']} 정답 {ED['gt_deficit_by_speed']} · "
              f"1위 에폭 {e0} {f0['top_deficit_by_speed']} → {e1} {f1['top_deficit_by_speed']}")
    rep = S.get("repro") or {}
    if rep.get("applicable"):
        for e, v in rep["epochs"].items():
            print(f"재현 에폭 {e}: " + "  ".join(f"{k} {x['got']:.5f}/{x['log']:.5f} (차 {x['abs_diff']:.1e})"
                                              for k, x in v["diff"].items()) + f" -> {'통과' if v['ok'] else '어긋남'}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", default="all", help="평가할 에폭 — all 또는 1,3,5,10,15")
    ap.add_argument("--panel-epochs", dest="panel_epochs", default="1,3,5,10,15",
                    help="고정 시나리오·끝 감속 곡선에 겹칠 에폭 (평가한 것만 쓴다)")
    ap.add_argument("--n-val", dest="n_val", type=int, default=DEFAULT_N)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=32, help="compare_v4_full.py 와 같게 32")
    ap.add_argument("--loader-workers", dest="lw", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8, help="원본 parquet 병렬 (학습 중 8 이하, 공유 머신 16 이하)")
    ap.add_argument("--threads", type=int, default=8, help="CPU 추론 torch 스레드 수")
    ap.add_argument("--input", default=None, help="runs/<tag>.json 이 없을 때(학습 중)만 쓰는 입력 형태")
    ap.add_argument("--th0", default=None, choices=[None, "current", "guard"], help="같은 조건의 θ0 규칙")
    ap.add_argument("--ref-tags", dest="ref_tags", default="auto",
                    help="학습 로그 곡선에 겹칠 판 (쉼표). auto = 태그에서 '_2hz' 를 뺀 판이 있으면 그것")
    ap.add_argument("--per-class", dest="per_class", type=int, default=2, help="상황마다 고정 시나리오 수")
    ap.add_argument("--picks-from", dest="picks_from", default=None, help="다른 판의 epoch_picks json 을 그대로 쓴다")
    ap.add_argument("--out-root", dest="out_root", default=None, help="기본 viz/v4/<tag> (개발 시험은 스크래치에)")
    ap.add_argument("--fresh", action="store_true", help="저장된 기준표·에폭 결과를 쓰지 않는다")
    ap.add_argument("--no-figs", dest="no_figs", action="store_true")
    a = ap.parse_args()
    if a.workers > 16:
        raise SystemExit("workers 는 16 이하 (공유 머신)")
    import torch
    t_all = time.time()
    tag = a.tag
    cfg, cfg_src = C.run_config(tag, {"input": a.input, "th0": a.th0})
    hist, hist_src = C.run_history(tag)
    print(f"[config] {cfg_src}: input {cfg.get('input')} · th0 {cfg.get('th0')} · offlane {cfg.get('offlane')} "
          f"(버려진 모드만 {bool(cfg.get('off_nonwinner'))}) · smooth {cfg.get('smooth')} · val {cfg.get('val_limit')}"
          f" | history {hist_src} {len(hist)} 에폭", flush=True)
    cache_dir = C.val_cache_for(cfg["input"])
    ckpts = C.epoch_ckpts(tag)
    if not ckpts:
        raise SystemExit(f"에폭 체크포인트가 없다: {C.RUNS / 'ckpt' / tag}")
    want = sorted(ckpts) if a.epochs == "all" else [int(x) for x in a.epochs.split(",")]
    miss = [e for e in want if e not in ckpts]
    if miss:
        print(f"[epochs] 체크포인트가 아직 없는 에폭은 뺀다: {miss}", flush=True)
    epochs = [e for e in want if e in ckpts]
    if not epochs:
        raise SystemExit("평가할 에폭이 없다")
    device = choose_device(a.device)
    if device == "cpu":
        torch.set_num_threads(a.threads)
    n_cache = json.loads((cache_dir / "meta.json").read_text())["n"]
    n = min(a.n_val, n_cache)
    global N_CACHE
    N_CACHE = n_cache
    figs, data_dir = fig_dir(tag, n, a.out_root)
    print(f"[run] {tag} · 에폭 {epochs} · val 앞 {n:,} ({cache_dir.name}) · {device}", flush=True)

    df, R, bkey = load_or_build_base(cache_dir, n, a.workers, data_dir, a.fresh)
    picks = load_or_pick(df, n, a.per_class, data_dir, a.picks_from)
    rows = run_epochs(tag, cfg, ckpts, epochs, device, a, n, cache_dir, df, picks, data_dir)
    res = aggregate(rows, df)
    # 전체 집계가 compare 방식 점수와 같은지 (같은 값의 두 계산 경로)
    for e in epochs:
        sc, r0 = rows[e][3], res[e]["전체"]
        d = max(abs(sc["minADE6"] - r0["minade"]), abs(sc["offlane"] - r0["offlane"]),
                abs(sc["dtheta_over_label_pct"] - r0["exc_pct"]), abs(sc["jitter_theta"] - r0["jit_theta"]))
        if d > 1e-5:
            raise SystemExit(f"에폭 {e}: 집계({r0['minade']:.6f})와 compare 점수({sc['minADE6']:.6f})가 다르다 ({d:.2e})")
    LO = learning_order(res, epochs) if len(epochs) >= 3 else None
    ED = end_decel(rows, R, epochs)
    rep = repro_check(tag, rows, hist, hist_src, cfg, n)
    bck = best_ckpt_check(tag, ckpts, hist)
    cmp_ = None
    cj = C.RUNS / "v4_full_compare.json"
    if cj.exists() and bck and bck.get("best_epoch") in rows:
        ref = json.loads(cj.read_text())["runs"].get(tag, {}).get(f"val{n}")
        if ref:
            sc = rows[bck["best_epoch"]][3]
            cmp_ = {"best_epoch": bck["best_epoch"], "expected": ref, "got": sc,
                    "max_abs_diff": max(abs(sc[k] - ref[k]) for k in SCORE_KEYS)}
    n_cls = {"전체": len(df)} | {c: int((df["cls"] == c).sum()) for c in C.CLASSES}
    if rep.get("applicable"):
        for e, v in rep["epochs"].items():
            dd = "  ".join(f"{k} {x['got']:.4f}/{x['log']:.4f}" for k, x in v["diff"].items())
            print(f"[repro] 에폭 {e} (계산/로그) {dd} -> {'통과' if v['ok'] else '어긋남'}", flush=True)
    else:
        print(f"[repro] {rep['reason']}", flush=True)
    if bck:
        print(f"[best] lstm_{tag}.pth = ep{bck['best_epoch']:02d}.pth : {bck['same']}", flush=True)
    if cmp_:
        print(f"[compare] best 에폭 val{n} vs v4_full_compare.json 최대 차 {cmp_['max_abs_diff']:.2e}", flush=True)

    panel_epochs = [e for e in (int(x) for x in a.panel_epochs.split(",")) if e in rows]
    if len(panel_epochs) < 2:
        panel_epochs = epochs if len(epochs) <= 5 else [epochs[0], epochs[len(epochs) // 2], epochs[-1]]
    summary = {
        "tag": tag, "n_val": n, "device": device, "epochs": epochs, "panel_epochs": panel_epochs,
        "val_cache": str(cache_dir), "ckpt_dir": str(C.RUNS / "ckpt" / tag), "config_source": cfg_src,
        "config": {k: cfg.get(k) for k in ("level", "input", "th0", "offlane", "off_nonwinner", "smooth", "limit",
                                           "val_limit", "epochs", "fallback", "seed")},
        "history_source": hist_src, "git_head": C.git_head(), "base_key": bkey, "calc_key": calc_key(),
        "definitions": DEFINITIONS, "thresholds": {k: getattr(C, k) for k in THRESH_KEYS},
        "constants": {"LAST_K": LAST_K, "REACH": REACH, "MOVE_REF_V": MOVE_REF_V, "MODEL_REF_V": MODEL_REF_V,
                      "facets": FACETS},
        "class_counts": n_cls, "picks": picks,
        "by_epoch": {e: res[e] for e in epochs},
        "score": {e: rows[e][3] for e in epochs},
        "learning_order": LO, "end_decel": ED, "repro": rep, "best_ckpt": bck, "compare_json": cmp_,
    }
    best_ep = min(hist, key=lambda h: h["minADE6"])["epoch"] if hist else None

    if not a.no_figs and len(epochs) >= 2:
        plt = C.setup_mpl()
        pop = f"{pop_desc(n)} 시나리오 · 상황은 정답 기반(viz_v4_common 임계값) · 흐린 선 = 그 지표 모집단 n < {C.MIN_N}"
        for fname, ttl, mets in (("e1_class_curves_accuracy", "상황별 지표 곡선 ① 정확도·모드 선택", METRICS[:6]),
                                 ("e2_class_curves_feasibility", "상황별 지표 곡선 ② 실현가능성·다양성", METRICS[6:])):
            fig_class_curves(res, epochs, n_cls, mets, f"{ttl} — {tag}, 에폭마다 한 점", pop, figs / f"{fname}.png",
                             best_ep=best_ep, n_val=n)
        refs = []
        ref_tags = ([tag.replace("_2hz", "")] if "_2hz" in tag else []) if a.ref_tags == "auto" else \
            [t for t in a.ref_tags.split(",") if t]
        ref_tags = [t for t in ref_tags if C.run_history(t)[0]]
        styles, rule = run_styles([tag] + ref_tags)
        for t in ref_tags:
            h, src = C.run_history(t)
            refs.append((t, h, *styles[t]))
        fig_train_log(tag, hist, hist_src, refs, res, epochs, n, cfg, figs / "e3_train_log.png", styles=styles,
                      rule=rule)
        if LO:
            fig_learning_order(LO, res, epochs, figs / "e4_learning_order.png", tag=tag)
        if ED:
            fig_end_decel(ED, epochs, panel_epochs, figs / "e5_end_decel.png", tag=tag)
        sus, _ = C.turn_straight_by_position(R["pos"], df["cls"].to_numpy())
        summary["late_tree"] = late_tree(rows, df, epochs, R, figs / "e6_tree_late_change.png", suspect=sus, tag=tag)
        z = dict(np.load(data_dir / f"epoch_scen_n{n}.npz"))
        extra, xchk = ({}, None)
        if picks:
            extra, xchk = panel_extra(tag, cfg, ckpts, panel_epochs, picks, cache_dir, device, data_dir, z)
            if not xchk["ok"]:
                print("[panel] 재추론이 저장된 결과와 다르다 — h·θ 에폭 선을 그리지 않는다", flush=True)
                extra = {}
        summary["panel_extra_check"] = xchk
        cx = PanelCtx(df, R, cache_dir, z, panel_epochs, hz=C.input_hz(cfg.get("input")), extra=extra, tag=tag)
        for j, p in enumerate(picks):
            panel_figure(cx, j, p, figs / "panel" / f"{j + 1:02d}_{CLASS_FILE[p['cls']]}_{p['k']}_{p['sid'][:8]}.png")
            plt.close("all")
        if picks:
            panel_overview(cx, picks, figs / "panel_overview.png")
        print(f"[figs] -> {figs}", flush=True)
    summary["sec"] = time.time() - t_all
    out = data_dir / f"epochs_n{n}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print_summary(summary)
    print(f"[done] {out}  {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
