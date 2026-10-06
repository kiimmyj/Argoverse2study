"""
score_subset.py - 시나리오 부분집합(차선변경 등)에서 판들을 비교한다 (2026-10-06).

왜 필요한가
-----------
차선변경은 val 24,988 중 627건(2.51%)이다. 거기서 minADE6 를 0.15 m 바꿔도 전체 지표로는 0.004 m 라
seed 범위에 완전히 묻힌다. 주변 차량 입력·횡 모드 축·차선 heading 정렬 손실은 모두 "차선변경을 고치려는"
변경이므로 **전용 평가군에서 재야** 판정이 된다.

입력
  runs/v4_xy_viol_per.npz   시나리오별 ade·fde·위반 step 수·유효 step 수·총 회전량 (src/score_xy_viol.py)
  viz/v4/lanechange/data/gt_groups.json   차선변경 분류별 scenario_id 목록 (viz_v4_lanechange.py)
  캐시의 scenario_id.json                 위 두 가지를 잇는 인덱스

  python src/score_subset.py --group D2_횡이동 --base <tag틀> --cond <tag틀> ...
  python src/score_subset.py                      # 기본 묶음 비교
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PER = ROOT / "runs" / "v4_xy_viol_per.npz"
GROUPS = ROOT / "viz" / "v4" / "lanechange" / "data" / "gt_groups.json"
DEFAULT_GROUPS = ["D2_횡이동", "S_예측중새로", "UT_U턴"]


def sid_index(tag):
    """태그가 쓴 val 캐시의 scenario_id 순서 (per.npz 의 행 순서와 같다)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_xy_viol import val_cache_of
    return json.loads((Path(val_cache_of(tag)) / "scenario_id.json").read_text())


def load(per, tag):
    return {k: per[f"{tag}|{k}"] for k in ("ade", "fde", "exc", "steps", "turn")}


def agg(d, sel):
    n = int(sel.sum())
    if n == 0:
        return None
    st = d["steps"][sel].sum()
    return {"n": n, "minADE6": float(d["ade"][sel].mean()), "minFDE6": float(d["fde"][sel].mean()),
            "viol_pct": float(100.0 * d["exc"][sel].sum() / max(st, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", default=str(PER))
    ap.add_argument("--groups", default=str(GROUPS))
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--cond", action="append", default=[],
                    help="'이름=태그틀' — 태그틀에 %%d 가 seed 자리. 여러 번 준다. 첫 번째가 기준판")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    per = np.load(a.per)
    conds = [c.split("=", 1) for c in a.cond]
    if not conds:
        raise SystemExit("--cond '이름=태그틀' 을 최소 둘 주어라 (첫 번째가 기준판)")
    seeds = [int(x) for x in a.seeds.split(",")]
    groups = a.group or DEFAULT_GROUPS

    G = json.loads(Path(a.groups).read_text())["groups"]
    sids = sid_index(conds[0][1] % seeds[0])
    pos = {s: i for i, s in enumerate(sids)}
    N = len(sids)

    sels = {"전체": np.ones(N, bool)}
    for g in groups:
        m = np.zeros(N, bool)
        miss = 0
        for s in G[g]["sids"]:
            i = pos.get(s)
            if i is None:
                miss += 1
            else:
                m[i] = True
        sels[g] = m
        if miss:
            print(f"[경고] {g}: scenario_id {miss}개가 캐시에 없다")

    base_name, base_pat = conds[0]
    for gname, sel in sels.items():
        print(f"\n=== {gname}  (n={int(sel.sum()):,})")
        print(f"{'condition':26s} {'minADE6':>9s} {'범위':>7s} {'위반%':>7s} {'짝 차이':>9s} {'시드별':>26s} {'부호':>6s}")
        base = [agg(load(per, base_pat % s), sel) for s in seeds]
        for name, pat in conds:
            try:
                rs = [agg(load(per, pat % s), sel) for s in seeds]
            except KeyError:
                print(f"{name:26s}  (점수 없음)")
                continue
            ade = [r["minADE6"] for r in rs]
            ds = [r["minADE6"] - b["minADE6"] for r, b in zip(rs, base)]
            sg = "일치" if all(d > 0 for d in ds) or all(d < 0 for d in ds) else "불일치"
            d_s = ", ".join(f"{d:+.3f}" for d in ds)
            print(f"{name:26s} {np.mean(ade):9.3f} {max(ade)-min(ade):7.3f} "
                  f"{np.mean([r['viol_pct'] for r in rs]):7.2f} "
                  f"{np.mean(ds):+9.3f} {d_s:>26s} {'' if pat == base_pat else sg:>6s}")


if __name__ == "__main__":
    main()
