#!/usr/bin/env python
"""v4 학습 직전까지를 자동화한다 — 전처리를 굽고, 모델을 세우고, 검증하고, 멈춘다.

train_v4.py 는 에폭 루프(:148) 안에서 __getitem__ 을 매번 새로 돈다. 캐시가 없어서
시나리오 하나당 parquet + log_map_archive JSON 을 다시 읽고 LaneGraph 를 다시 만든다.
실측 ~96 ms/샘플이라 train 50,000 기준 1에폭이 단일코어 80분이고, 15에폭이면 같은
계산을 15번 되풀이한다. 에폭 시간의 거의 전부가 이것이고 GPU 는 45% 에서 굶는다.

이 스크립트는 그 전처리를 **필드별 memmap 으로 한 번만 굽는다**. 반환 텐서가 전부
고정 shape 이라 memmap 이 맞다 — 압축도 파싱도 없이 O(1) 임의접근이 된다.

  단계 0  환경·저장소 점검
  단계 1  전처리 캐시 구축 (또는 재사용)
  단계 2  데이터 불변식 검증
  단계 3  모델 완성·검증 (레벨별 파라미터 수 / forward / loss / backward)
  단계 4  학습 직전에서 정지 — 실행할 커맨드만 출력한다

캐시 키는 **전처리 설정 + 소스 해시**다. dataset_lane.py / lane_frame.py / lane_graph.py
중 하나라도 바뀌면 키가 달라져 새 캐시가 생기고, 옛 캐시는 남아 재사용된다.
열거기를 고치고 캐시를 지우는 걸 잊어 낡은 데이터로 학습하는 사고를 막는다.

사용:
  python src/prepare_v4.py --level l3 --limit 50000 --val-limit 2000
  python src/prepare_v4.py --level l3 --verify-only        # 캐시 안 굽고 점검만
  python src/prepare_v4.py --level l3 --rebuild            # 캐시 강제 재생성
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))          # train_v4.py 는 상대경로 append 를 쓴다
from dataset_cached import source_files  # noqa: E402

DATA_ROOT = "/data/argoverse2/motion_forecasting"
CACHE_ROOT = "/data/argoverse2/cache/v4"

# 캐시 키에 들어가는 소스는 dataset_cached.source_files() 가 import 를 따라가 자동으로 모은다.
# (손으로 적은 목록에는 heading_decomp.py · dataset_map.py 가 빠져 있었다.)

# 알려진 파라미터 수 (in_dim=5, lane_in=30 = --theta 0 --rules 1 일 때만)
KNOWN_PARAMS = {"l2": 414_294, "l3": 452_217, "l0": 452_217}

# 문자열이라 memmap 에 못 넣는 키 — 따로 json 으로 뺀다
STR_KEYS = ("scenario_id",)


def log(msg=""):
    print(msg, flush=True)


def rule(title):
    log()
    log("─" * 72)
    log(title)
    log("─" * 72)


# ───────────────────────────── 단계 0 — 점검 ─────────────────────────────

def check_env(args):
    ok = True

    log(f"python      {sys.executable}")
    if "envs/av2" not in sys.executable:
        log("  ! conda env 'av2' 가 아니다. run_*.sh 는 "
            "/home/user/miniforge3/envs/av2/bin/python 를 쓴다.")
        ok = False

    log(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.zeros(1).cuda()                   # CUDA 컨텍스트를 여기서 미리 터뜨린다
        log(f"            {torch.cuda.get_device_name(0)}  "
            f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB free")

    try:
        import av2  # noqa: F401
        from dataset_lane import Av2LaneRuleDataset  # noqa: F401
        from model_v4 import V4Net  # noqa: F401
        log("import      av2 / dataset_lane / model_v4  ok")
    except Exception as e:
        log(f"  ! import 실패: {e}")
        return False

    for split in ("train", "val"):
        p = Path(DATA_ROOT) / split
        if not p.is_dir():
            log(f"  ! 데이터 경로 없음: {p}")
            ok = False

    # 저장소 상태 — 캐시 meta 에 남겨 재현성을 확보한다
    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", "src")
    log(f"git         HEAD {head}")
    if dirty:
        log("            워킹트리 dirty (캐시 meta 에 기록된다):")
        for line in dirty.splitlines():
            log(f"              {line}")

    # 다른 학습이 돌고 있으면 CPU 를 뺏지 않도록 경고한다
    running = ps_train_v4()
    if running:
        log(f"  ! train_v4.py 가 {running} 개 프로세스로 돌고 있다. "
            f"--workers 를 낮게 유지한다 (현재 {args.workers}).")
    return ok


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO), *a],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def ps_train_v4():
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        return sum(1 for l in out.splitlines() if "train_v4.py" in l and "grep" not in l)
    except Exception:
        return 0


# ────────────────────────── 단계 1 — 캐시 구축 ──────────────────────────

def dataset_kwargs(args):
    """train_v4.py:129-132 가 실제로 넘기는 조합을 그대로 복제한다."""
    return dict(with_rules=bool(args.rules),
                theta_ch=bool(args.theta),
                h_src=args.h_src,
                routes=args.level != "l2",
                fallback=args.fallback)


def cache_key(split, dkw):
    """전처리 설정 + 소스 해시. 셋 중 하나라도 바뀌면 새 캐시가 된다."""
    h = hashlib.sha256()
    h.update(json.dumps({"split": split, **dkw}, sort_keys=True).encode())
    src_hashes = {}
    for fn in source_files(REPO / "src"):
        b = (REPO / "src" / fn).read_bytes()
        d = hashlib.sha256(b).hexdigest()
        src_hashes[fn] = d[:12]
        h.update(d.encode())
    return h.hexdigest()[:16], src_hashes


def build_cache(args, split, limit, cache_dir, dkw):
    from torch.utils.data import DataLoader
    from dataset_lane import Av2LaneRuleDataset

    t0 = time.perf_counter()
    log(f"[{split}] Dataset 생성 중 — split 디렉터리를 전수 열거한다 "
        f"(train 199,908 / val 24,988, limit 과 무관하게 든다)")
    ds = Av2LaneRuleDataset(DATA_ROOT, split, limit, **dkw)
    n = len(ds)
    log(f"[{split}] {n:,} 시나리오, 열거 {time.perf_counter() - t0:.1f}s")
    if n == 0:
        raise SystemExit(f"[{split}] 시나리오가 0개다. limit 을 확인하라.")

    probe = ds[0]
    specs = {k: (tuple(v.shape), v.dtype) for k, v in probe.items()
             if k not in STR_KEYS}

    tmp = cache_dir.with_suffix(".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    mm = {k: np.lib.format.open_memmap(
              tmp / f"{k}.npy", mode="w+",
              dtype=np.dtype(str(dt).replace("torch.", "")), shape=(n, *sh))
          for k, (sh, dt) in specs.items()}
    sids = []

    total_bytes = sum(a.nbytes for a in mm.values())
    log(f"[{split}] 캐시 {total_bytes / 2**20:,.0f} MiB, 필드 {len(mm)}개 -> {cache_dir}")

    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_keep_str)
    done, t0 = 0, time.perf_counter()
    for tensors, ids in loader:
        b = len(ids)
        for k, v in tensors.items():
            mm[k][done:done + b] = v.numpy()
        sids.extend(ids)
        done += b
        if done % 3200 == 0 or done == n:
            el = time.perf_counter() - t0
            eta = el / done * (n - done)
            log(f"[{split}] {done:,}/{n:,}  {el:.0f}s 경과  ETA {eta:.0f}s  "
                f"({done / el:.0f} 샘플/s)")

    for a in mm.values():
        a.flush()
    (tmp / "scenario_id.json").write_text(json.dumps(sids))
    key, src_hashes = cache_key(split, dkw)
    (tmp / "meta.json").write_text(json.dumps({
        "split": split, "n": n, "limit": limit, "dataset_kwargs": dkw,
        "cache_key": key, "source_hashes": src_hashes,
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--", "src"),
        "fields": {k: [list(sh), str(dt)] for k, (sh, dt) in specs.items()},
        "built_by": "src/prepare_v4.py",
    }, indent=2, ensure_ascii=False))

    tmp.rename(cache_dir)                     # 원자적 커밋 — 반쯤 구운 캐시를 남기지 않는다
    log(f"[{split}] 완료 {time.perf_counter() - t0:.0f}s")
    return cache_dir


def collate_keep_str(samples):
    """기본 collate 는 문자열을 리스트로 만든다. 텐서와 id 를 분리해 돌려준다."""
    ids = [s["scenario_id"] for s in samples]
    keys = [k for k in samples[0] if k not in STR_KEYS]
    return {k: torch.stack([torch.as_tensor(s[k]) for s in samples]) for k in keys}, ids


def ensure_cache(args, split, limit, dkw):
    key, _ = cache_key(split, dkw)
    cache_dir = Path(args.cache_root) / f"{split}_{key}_n{limit}"
    if cache_dir.exists() and not args.rebuild:
        meta = json.loads((cache_dir / "meta.json").read_text())
        log(f"[{split}] 캐시 재사용 {cache_dir}  (n={meta['n']:,})")
        return cache_dir
    if args.verify_only:
        log(f"[{split}] 캐시 없음 — --verify-only 라 굽지 않는다 ({cache_dir})")
        return None
    if cache_dir.exists():
        log(f"[{split}] --rebuild: 기존 캐시 삭제 {cache_dir}")
        shutil.rmtree(cache_dir)
    return build_cache(args, split, limit, cache_dir, dkw)


# ─────────────────────── 단계 2 — 데이터 불변식 ───────────────────────

def verify_data(args, cache_dir, split):
    """학습 없이 확인 가능한 것만 검사한다. 실패는 세지 되 즉시 죽지 않는다."""
    from dataset_cached import CachedV4Dataset

    ds = CachedV4Dataset(cache_dir)
    meta = ds.meta
    n = len(ds)
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
        log(f"  {'ok  ' if cond else 'FAIL'}  {msg}")

    log(f"[{split}] n={n:,}  필드 {len(meta['fields'])}개")

    s = ds[0]
    exp_in_dim = 5 + (3 if args.theta else 0)
    check(tuple(s["x"].shape) == (50, exp_in_dim), f"x shape (50,{exp_in_dim})")
    check(tuple(s["y"].shape) == (60, 2), "y shape (60,2)")
    check(tuple(s["lanes"].shape) == (20, 10, 2), "lanes shape (20,10,2)")

    if args.level != "l2":
        for k in ("routes", "route_tan", "route_band"):
            check(tuple(s[k].shape) == (6, 64, 2), f"{k} shape (6,64,2)")
        check(s["route_sub"].dtype == torch.int64, "route_sub dtype int64")

    # 전수 통계 — memmap 이라 싸다
    idx = np.arange(n)
    smp = idx if n <= 4000 else np.linspace(0, n - 1, 4000).astype(int)

    y = ds.raw("y")[smp]
    check(np.isfinite(y).all(), "y 에 NaN/Inf 없음")
    x = ds.raw("x")[smp]
    check(np.isfinite(x).all(), "x 에 NaN/Inf 없음")
    # x[:,4] = head - theta 는 wrap 하지 않는다 (dataset_lane.py:106). ±2π 가 정상이다.
    log(f"  info  x[:,4] |max| = {np.abs(x[..., 4]).max():.3f} rad  (wrap 안 함, 정상)")

    if args.level != "l2":
        rm = ds.raw("route_mask")[smp]
        live = rm.sum(1)
        check(np.isfinite(ds.raw("routes")[smp]).all(), "routes 에 NaN/Inf 없음")
        check(((live >= 1) & (live <= 6)).all(), "살아있는 모드 수 ∈ [1,6]")
        sub = ds.raw("route_sub")[smp]
        check(((sub >= 0) & (sub < 6)).all(), "route_sub ∈ [0,5]")

        nd = ds.raw("n_distinct")[smp]
        fb = ds.raw("route_fallback")[smp]
        log(f"  info  살아있는 모드: 6개 {100 * (live == 6).mean():.2f}%  "
            f"1개 {100 * (live == 1).mean():.2f}%")
        log(f"  info  구별 분기 n_distinct: 중앙 {np.median(nd):.0f}  "
            f"≤1 인 비율 {100 * (nd <= 1).mean():.2f}%  "
            f"<6 인 비율 {100 * (nd < 6).mean():.2f}%")
        log(f"  info  폴백(경로 0개) 비율 {100 * fb.mean():.2f}%")
    return fails


# ──────────────────── 단계 3 — 모델 완성·검증 ────────────────────

def verify_model(args, cache_dir):
    from dataset_cached import CachedV4Dataset
    from model_v4 import V4Net, N_PTS, N_RULE
    import train_v4 as T
    from torch.utils.data import DataLoader

    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
        log(f"  {'ok  ' if cond else 'FAIL'}  {msg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_rules = bool(args.rules)
    lane_in = N_PTS * 2 + (N_RULE if use_rules else 0)
    in_dim = 5 + (3 if args.theta else 0)

    for level in (("l2", "l3", "l0") if args.all_levels else (args.level,)):
        log(f"\n[{level}]  in_dim={in_dim} lane_in={lane_in}")
        model = V4Net(in_dim=in_dim, lane_in=lane_in, level=level).to(device)
        npar = sum(p.numel() for p in model.parameters())

        exp = KNOWN_PARAMS.get(level)
        if in_dim == 5 and lane_in == 30 and exp:
            check(npar == exp, f"params {npar:,} == 알려진 값 {exp:,}")
        else:
            log(f"  info  params {npar:,} (in_dim/lane_in 이 기준과 달라 대조 생략)")

        if cache_dir is None:
            log("  skip  캐시가 없어 forward 검증을 건너뛴다")
            continue

        ds = CachedV4Dataset(cache_dir)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        b = next(iter(loader))
        kw = T.to_dev(b, device, level, use_rules)
        traj, logits, aux = model(**kw)

        B = args.batch
        check(tuple(traj.shape) == (B, 6, 60, 2), f"traj shape ({B},6,60,2)")
        check(tuple(logits.shape) == (B, 6), f"logits shape ({B},6)")
        check(torch.isfinite(traj).all(), "traj 유한")
        # l3 은 head 를 그대로 반환한다 (model_v4.py:175-179) — aux 가 비어야 정상.
        # l0 만 rollout(Frenet 적분) 을 거쳐 aux 를 채운다. 파라미터 수는 둘이 같으므로
        # 레벨이 제대로 걸렸는지는 aux 로만 구별된다.
        check((len(aux) == 0) == (level != "l0"),
              f"aux {'비어있음' if level != 'l0' else f'{len(aux)}개 키'} (레벨 반영 확인)")

        y = b["y"].to(device)
        mm = (b["route_mask"].to(device) if level != "l2"
              else torch.ones(y.shape[0], 6, device=device))
        loss, off = T.loss_fn(traj, logits, y, mm, aux, args.offlane,
                              bool(args.off_nonwinner))
        check(torch.isfinite(loss), f"loss 유한 ({loss.item():.4f})")

        loss.backward()
        g = model.traj_head.weight.grad
        check(g is not None and g.abs().max().item() > 0,
              f"traj_head gradient 흐름 (max {g.abs().max().item():.3e})"
              + (" — l0 적분기 단절 검사" if level == "l0" else ""))
    return fails


# ──────────────── 선택 — kappa_lane 부품 (통합 전, 보고용) ────────────────

def check_kappa():
    """다른 세션이 워크트리에서 만드는 중인 부품. 소비자가 아직 없으므로
    학습 인자·Dataset·모델 구성에 절대 개입시키지 않고 존재 여부만 보고한다."""
    wt = REPO / ".claude/worktrees/priceless-cerf-1aa86a"
    src = wt / "src/lane_spline.py"
    if not src.exists():
        log("  kappa_lane 부품 없음 — 건너뛴다")
        return
    head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    log(f"  lane_spline.py 있음 ({src.stat().st_size:,} B, 워크트리 HEAD {head})")
    log("  소비자 없음 — dataset_lane 은 route_kappa 를 만들지 않고 "
        "model_v4.rollout 은 kappa 를 받지 않는다. 통합 전이므로 보고만 한다.")


# ────────────────────────────── main ──────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="v4 전처리를 굽고 모델을 세운 뒤 학습 직전에서 멈춘다")
    p.add_argument("--level", default="l3", choices=("l2", "l3", "l0"))
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--theta", type=int, default=0)      # train_v4 기본은 1, v4 실험은 전부 0
    p.add_argument("--rules", type=int, default=1)
    p.add_argument("--h-src", default="build", choices=("av2", "build"))
    # 경로 0개 시나리오를 채우는 방식. train_v4.py:110 과 같은 기본값이어야 같은 데이터가
    # 되고, 캐시 키에 들어가므로 폴백마다 다른 캐시가 생긴다.
    p.add_argument("--fallback", default="fan", choices=("straight1", "straight6", "fan"))
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--offlane", type=float, default=0.0)
    p.add_argument("--off-nonwinner", type=int, default=0)
    p.add_argument("--workers", type=int, default=8,
                   help="캐시 굽기 병렬도. 다른 학습이 돌면 낮게 유지한다")
    p.add_argument("--cache-root", default=CACHE_ROOT)
    p.add_argument("--rebuild", action="store_true", help="캐시를 강제로 다시 굽는다")
    p.add_argument("--verify-only", action="store_true", help="캐시를 굽지 않고 점검만")
    p.add_argument("--all-levels", action="store_true", help="세 레벨 모두 모델 검증")
    p.add_argument("--with-kappa-check", action="store_true")
    args = p.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers 는 1 이상이어야 한다 "
                         "(DataLoader persistent_workers 제약)")
    if args.val_limit < 1:
        raise SystemExit("--val-limit 0 은 빈 val 셋이 되어 evaluate 에서 "
                         "ZeroDivisionError 로 죽는다")
    if args.limit and args.limit < args.batch:
        raise SystemExit(f"--limit {args.limit} < --batch {args.batch} 이면 "
                         f"drop_last=True 라 train 배치가 0개가 된다")

    t_start = time.perf_counter()
    fails = []

    rule("단계 0 — 환경·저장소 점검")
    if not check_env(args):
        raise SystemExit("환경 점검 실패. 위 항목을 고치고 다시 돌려라.")

    dkw = dataset_kwargs(args)
    log(f"\n전처리 설정 (train_v4.py:129-132 와 동일): {dkw}")

    rule("단계 1 — 전처리 캐시")
    caches = {}
    for split, lim in (("train", args.limit), ("val", args.val_limit)):
        caches[split] = ensure_cache(args, split, lim, dkw)

    rule("단계 2 — 데이터 불변식")
    for split in ("train", "val"):
        if caches[split] is None:
            log(f"[{split}] 캐시 없음 — 건너뛴다")
            continue
        log(f"\n[{split}]")
        fails += verify_data(args, caches[split], split)

    rule("단계 3 — 모델 완성·검증")
    fails += verify_model(args, caches["val"])

    if args.with_kappa_check:
        rule("선택 — kappa_lane 부품 (통합 전)")
        check_kappa()

    rule("단계 4 — 학습 직전에서 정지")
    log(f"소요 {time.perf_counter() - t_start:.0f}s")
    if fails:
        log(f"\n실패 {len(fails)}건 — 학습을 돌리지 마라:")
        for f in fails:
            log(f"  - {f}")
        raise SystemExit(1)

    log("\n모든 점검 통과. 전처리는 캐시에 구워졌고 모델은 세워진다.")
    log("에폭은 돌리지 않았다. 학습하려면:\n")
    tag = f"v4_{args.level}_s0"
    log(f"  PY=/home/user/miniforge3/envs/av2/bin/python")
    log(f"  cd {REPO}")
    log(f"  $PY src/train_v4.py --level {args.level} --theta {args.theta} "
        f"--rules {args.rules} --h-src {args.h_src} --fallback {args.fallback} \\")
    log(f"      --limit {args.limit} --val-limit {args.val_limit} --epochs 15 "
        f"--lr 5e-4 --batch {args.batch} \\")
    log(f"      --workers 24 --outdir runs --tag {tag} > runs/{tag}.log 2>&1")
    log("")
    log("주의 — train_v4.py 는 아직 캐시를 읽지 않는다. 캐시를 쓰려면 "
        "dataset_cached.CachedV4Dataset 로\n"
        "      갈아끼워야 한다 (README 절 참고). 지금 그대로 돌리면 전처리를 "
        "다시 한다.")


if __name__ == "__main__":
    main()
