"""R2 선행 측정 정리 — 방향 정확도 · 실현가능성 · 선회율 분포의 **자**를 만든다.

`~/av2_simpl/tools/measure_motion_quality.py` 가 판마다 남긴 npz 를 합친다.

이 지표들은 R2 의 **주 지표**인데 R0·R1 에서 한 번도 잰 적이 없었다. 여기서 세 가지가 선다.
  ① R0 의 값 — R2 가 맞댈 상대
  ② |s43 − R0| — 시드만 바꾼 런 간 폭, 즉 **자**
  ③ 정답(GT)의 값 — **잡음 바닥**. 모델 값을 그 아래로 읽으면 안 된다

⚠️ 지표마다 쓸 수 있는지 없는지가 다르다. 종가속 위반은 정답이 이미 57% 라 **포화**해서
   차이를 못 본다. 그런 지표는 코드가 직접 '못 씀' 으로 표시한다 — 사람 눈으로 거르지 않는다.
"""
import json
from pathlib import Path

import numpy as np

import viz_v4_common as C

SIMPL = Path("/home/user/av2_simpl/tools")
OUT = C.ROOT / "viz" / "motion_quality"
RUNS = ["R0", "R1", "s43"]
# (키, 이름, 좋은 방향, 단위)
METRICS = [("hd_err", "방향 오차", "↓", "°"),
           ("viol_lat", "횡가속 위반 (궤적)", "↓", "%"),
           ("viol_lon", "종가속 위반 (궤적)", "↓", "%"),
           ("flat", "횡가속 위반 (step)", "↓", "%"),
           ("flon", "종가속 위반 (step)", "↓", "%")]
PCT = {"viol_lat", "viol_lon", "flat", "flon"}
# ⚠️ R2 의 적분기가 **구조적으로** 묶는 양. 여기서 좋아지는 것은 학습의 결과가 아니라
#    설계의 결과다 — clip 이 한계 밖을 표현 자체가 불가능하게 만들기 때문이다.
#    DKM 의 "비실현율 26% → 0%" 도 같은 성질이고, 그래서 **판정 근거로 쓸 수 없다.**
#    R2 를 판정하는 것은 적분기가 안 묶는 양이다 — 방향 오차 · 선회율 분포 모양 · minADE.
BY_CONSTRUCTION = {"viol_lat", "viol_lon", "flat", "flon"}


def w1(h1, h2, edges):
    """히스토그램 둘의 1-Wasserstein. CDF 차이의 적분이다."""
    c1, c2 = np.cumsum(h1) / h1.sum(), np.cumsum(h2) / h2.sum()
    return float(np.abs(c1 - c2).sum() * (edges[1] - edges[0]))


R2DOC = (C.ROOT / "docs" / "superpowers" / "specs"
         / "2026-11-25-r2-output-representation-design.md")


def md_block(S):
    """§8-1 에 넣을 표. **지표를 쓸 수 있는지 없는지도 코드가 정한다** — 손으로 거르면
    R2 값을 보고 나서 유리한 지표를 고르게 된다."""
    L = [f"val {S['n_scene']:,}판 · `src/analyze_motion_quality.py`. "
         f"**자 = |s43 − R0|** (시드만 바꾼 런 간 폭) · **효과 = |R1 − R0|**.", ""]
    for who, lab in (("best", "best 모드 (minADE6 가 쓰는 모드)"), ("top1", "top1 모드 (확률 1위)")):
        L += [f"**{lab}**", "",
              "| 지표 | R0 | R1 | s43 | 정답 | 자 | 효과 | 배수 | R2 판정에 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, nm, _, u in METRICS:
            y = S["yardstick"][who].get(k)
            if y is None:
                continue
            gt = f"{y['gt']:.2f}{u}" if y["gt"] is not None else "—"
            rr = f"**{y['ratio']:.1f}배**" if y["ratio"] else "—"
            tag = "✅ 쓴다" if y["usable"] else f"❌ {y['why_not'][0]}"
            L.append(f"| {nm} | **{y['R0']:.2f}{u}** | {y['R1']:.2f}{u} | {y['s43']:.2f}{u} | {gt} "
                     f"| {y['eps']:.2f} | {y['effect']:.2f} | {rr} | {tag} |")
        w = S["turn_w1"][who]
        rr = f"**{w['ratio']:.1f}배**" if w["ratio"] else "—"
        L.append(f"| 선회율 분포 거리 | **{w['R0']:.3f}°** | {w['R1']:.3f}° | {w['s43']:.3f}° | — "
                 f"| {w['eps']:.3f} | {w['effect']:.3f} | {rr} | ✅ 쓴다 |")
        L.append("")
    return "\n".join(L)


def fill(path, marker, body):
    if not path.exists():
        print(f"  (문서 없음 — 건너뜀: {path})")
        return
    txt = path.read_text()
    a, b = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    if a not in txt or b not in txt:
        print(f"  (표식 {marker} 없음 — 건너뜀)")
        return
    head, rest = txt.split(a, 1)
    _, tail = rest.split(b, 1)
    path.write_text(f"{head}{a}\n\n{body}\n{b}{tail}")
    print(f"  문서 갱신 → {path}")


def main():
    D = {r: np.load(SIMPL / f"motion_{r}.npz") for r in RUNS if (SIMPL / f"motion_{r}.npz").exists()}
    if len(D) < len(RUNS):
        raise SystemExit(f"npz 부족: {sorted(D)} — measure_motion_quality.py 를 먼저 돌린다")
    OUT.mkdir(parents=True, exist_ok=True)
    edges = D["R0"]["hist_edges"]

    S = {"n_scene": int(len(D["R0"]["gt.viol"])), "who": {}, "yardstick": {}, "turn_w1": {}}
    for who in ("best", "top1", "gt"):
        S["who"][who] = {}
        for r in RUNS:
            e = {}
            for k, _, _, _ in METRICS:
                key = f"{who}.{k}"
                if key not in D[r].files:
                    continue
                v = float(np.nanmean(D[r][key]))
                e[k] = v * 100 if k in PCT else v
            S["who"][who][r] = e
        # 정답은 판마다 같다(모델과 무관) — 한 번만 적는다
        if who == "gt":
            S["who"]["gt"] = {"value": S["who"]["gt"]["R0"]}

    # 자 = |s43 − R0| · 효과 = |R1 − R0|
    for who in ("best", "top1"):
        S["yardstick"][who] = {}
        for k, _, _, _ in METRICS:
            a, b, c = (S["who"][who][r].get(k) for r in ("R0", "R1", "s43"))
            if a is None:
                continue
            eps, eff = abs(c - a), abs(b - a)
            gt = S["who"]["gt"]["value"].get(k)
            # 못 쓰는 지표 판정 — 코드가 정한다
            why = []
            if k in BY_CONSTRUCTION:
                why.append("R2 적분기가 구조적으로 묶는다(설계의 결과)")
            if gt is not None and gt > 50:
                why.append("정답이 이미 50% 넘게 걸린다(포화)")
            if gt is not None and abs(a - gt) <= eps:
                why.append("R0 가 잡음 바닥과 구별 안 된다(여유 없음)")
            S["yardstick"][who][k] = {"R0": a, "R1": b, "s43": c, "gt": gt,
                                      "eps": eps, "effect": eff,
                                      "ratio": (eff / eps) if eps > 1e-9 else None,
                                      "usable": not why, "why_not": why}
    for who in ("best", "top1"):
        v = {r: w1(D[r][f"hist.{who}"], D[r]["hist.gt"], edges) for r in RUNS}
        eps, eff = abs(v["s43"] - v["R0"]), abs(v["R1"] - v["R0"])
        # 적분기는 꼬리를 자르지만 **분포의 모양**은 안 묶는다 — 그래서 이건 쓸 수 있다
        S["turn_w1"][who] = dict(v, eps=eps, effect=eff,
                                 ratio=(eff / eps) if eps > 1e-9 else None, usable=True)

    print(f"val {S['n_scene']:,}판\n")
    for who in ("best", "top1"):
        print(f"── {who} 모드 " + "─" * 62)
        print(f"{'지표':22s}{'R0':>9s}{'R1':>9s}{'s43':>9s}{'정답':>9s}"
              f"{'자':>8s}{'효과':>8s}{'배수':>7s}  판정")
        for k, nm, _, u in METRICS:
            y = S["yardstick"][who].get(k)
            if y is None:
                continue
            rr = f"{y['ratio']:.1f}x" if y["ratio"] else "—"
            tag = "쓸 수 있다" if y["usable"] else "✗ " + y["why_not"][0]
            # 방향 오차는 정답 자신과 비교할 수 없다 — 정답 대비로 정의된 양이다
            gt = f"{y['gt']:>8.2f}{u}" if y["gt"] is not None else f"{'—':>9s}"
            print(f"{nm:22s}{y['R0']:>8.2f}{u}{y['R1']:>8.2f}{u}{y['s43']:>8.2f}{u}"
                  f"{gt}{y['eps']:>7.2f}{y['effect']:>7.2f}{rr:>7s}  {tag}")
        w = S["turn_w1"][who]
        rr = f"{w['ratio']:.1f}x" if w["ratio"] else "—"
        print(f"{'선회율 분포 거리':22s}" + "".join(f"{w[r]:>8.3f}°" for r in RUNS)
              + f"{'—':>9s}{w['eps']:>7.3f}{w['effect']:>7.3f}{rr:>7s}  쓸 수 있다")
        print()

    (OUT / "numbers.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
    fill(R2DOC, "R2BASE", md_block(S))
    print(f"숫자 → {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
