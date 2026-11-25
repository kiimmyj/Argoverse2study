"""R2 결정 — Frenet 기준선을 **고정**할지 **매 step 재투영**할지, val 전수로 잰다.

왜 이 측정이 필요한가
---------------------
`viz_meeting_1029.m7` 이 좌회전 장면 **하나**에서 방향을 보여 줬다. 거기서 나온 결론은
"θ 로 내면 재투영, Δθ 로 내면 고정" 인데, 장면 하나로 설계를 정할 수는 없다.
이 문서가 하루에 다섯 번 빠진 함정이 그것이다(끝점·짧은 창으로 추세 읽기).

무엇을 재나 — 미래 60 step 에서, 같은 궤적을 두 규칙으로 기술했을 때
  θ    = h − k            모델이 내야 하는 **절대 오프셋**
  Δθ   = Δh − Δk          모델이 내야 하는 **증분**
  고정   k = k[t=49] 하나.  Δk = 0 이므로 **Δθ = Δh — 차선이 식에서 사라진다**
  재투영 k = 매 step 최근접 차로 접선

네 가지를 모은다: |θ| 평균·최대, |Δθ| 평균·최대, **계단성** SD(ΔΔθ), 차로 점프 max|Δk|.
계단성이 핵심이다 — 모델이 배워야 하는 신호가 매끄러운가 톱니인가를 가른다.

⚠️ **v4 캐시의 x.npy/y.npy 를 쓰지 않는다.** 같은 장면인데 프레임도 샘플링도 다르다
(t=0 위치가 24 m, 구간 길이가 0.9 m 어긋난다 — 2026-11-25 확인). 평활·리샘플된 v4 표현이라
원본 궤적이 아니다. 그래서 parquet 에서 직접 읽는다(장면당 170 ms, 그래서 병렬).

    python src/measure_frenet_baseline.py              # val 전수, 워커 8
    python src/measure_frenet_baseline.py --n 500      # 빠른 확인
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import viz_v4_common as C
import viz_meeting_1029 as M

OUT = C.ROOT / "viz" / "frenet_baseline"
V_MIN = 0.5          # [m/s] 이 아래 step 의 위치차분 방향은 믿지 않는다 (ah_feat 과 같은 값)
TURN_DEG = 30.0      # viz_v4_common.TURN_DEG — 미래 순 진행방향 변화가 이 이상이면 회전
STRAIGHT_DEG = 15.0  # viz_v4_common.LC_MAX_DEG — 이 미만이면 직진
MOVE_M = 10.0        # 미래 이동거리가 이 미만이면 '거의 안 움직임' 으로 따로 센다

_SIDS = None
_ORG = _TH = None


def _init():
    """워커마다 한 번 — mmap 과 장면 목록을 연다."""
    global _SIDS, _ORG, _TH
    _SIDS = json.loads((M.CACHE / "scenario_id.json").read_text())
    _ORG = np.load(M.CACHE / "origin.npy", mmap_mode="r")
    _TH = np.load(M.CACHE / "theta.npy", mmap_mode="r")


def _stats(th, dth):
    """한 규칙의 요약. th/dth 는 degree."""
    return dict(th_mean=float(np.abs(th).mean()), th_max=float(np.abs(th).max()),
                dth_mean=float(np.abs(dth).mean()), dth_max=float(np.abs(dth).max()),
                rough=float(np.diff(dth).std()) if len(dth) > 2 else float("nan"))


def one(i):
    """장면 하나 → 두 규칙의 요약. 실패하면 None (지도가 없거나 트랙이 짧은 판)."""
    from av2.map.map_api import ArgoverseStaticMap
    try:
        sid = _SIDS[i]
        origin = np.asarray(_ORG[i], np.float64)
        theta = float(_TH[i])
        ts, pos, head, _ = M._load_focal(sid, origin, theta)
        if len(pos) < C.OBS + 2:
            return None
        # h 는 위치차분 진행방향 — 입력 (a,h) 와 같은 정의로 맞춘다 (AV2 yaw 가 아니다)
        d = np.diff(pos, axis=0)
        spd = np.linalg.norm(d, axis=1) / 0.1
        h = np.arctan2(d[:, 1], d[:, 0])
        h = np.concatenate([h, h[-1:]])                      # 마지막 점은 직전 값 복제
        spd = np.concatenate([spd, spd[-1:]])

        amap = ArgoverseStaticMap.from_json(C.VAL_DIR / sid / f"log_map_archive_{sid}.json")
        pts, tans = M._centerline_cloud(amap, origin, theta)
        if len(pts) < 10:
            return None
        k_mov, _ = M._nearest_lane_angle(pos, h, pts, tans)
        k_fix = k_mov[C.OBS - 1]

        f = slice(C.OBS - 1, len(pos))                        # 미래 60 step (+ 시작점)
        hf, kf, sf = h[f], k_mov[f], spd[f]
        th_mov = np.degrees(C.wrap(hf - kf))
        th_fix = np.degrees(C.wrap(hf - k_fix))
        dth_mov = np.degrees(C.wrap(np.diff(th_mov / 180 * np.pi)))
        dth_fix = np.degrees(C.wrap(np.diff(th_fix / 180 * np.pi)))
        dk = np.degrees(np.abs(C.wrap(np.diff(kf))))

        path = float(np.linalg.norm(np.diff(pos[f], axis=0), axis=1).sum())
        net = float(np.degrees(C.wrap(hf[-1] - hf[0])))
        slow = float((sf < V_MIN).mean())                     # 방향을 못 믿는 step 비율
        cls = ("거의 안 움직임" if path < MOVE_M else
               "회전" if abs(net) > TURN_DEG else
               "직진" if abs(net) < STRAIGHT_DEG else "중간")
        return dict(i=int(i), sid=sid, cls=cls, path=path, net=net, slow=slow,
                    dk_max=float(dk.max()), dk_mean=float(dk.mean()),
                    fix=_stats(th_fix, dth_fix), mov=_stats(th_mov, dth_mov))
    except Exception as e:
        # ⚠️ 조용히 None 을 돌려주면 10% 가 말없이 빠지고, 그게 한쪽으로 치우쳐도 모른다.
        #    사유를 들고 올라가서 main() 이 모아 찍게 한다.
        return {"i": int(i), "fail": f"{type(e).__name__}: {str(e)[:80]}"}


def summarize(rows):
    """클래스별로 두 규칙을 나란히. '거의 안 움직임' 은 방향이 정의가 안 돼 따로 뺀다."""
    out = {}
    groups = [("전체 (움직인 판)", lambda r: r["cls"] != "거의 안 움직임"),
              ("회전", lambda r: r["cls"] == "회전"),
              ("직진", lambda r: r["cls"] == "직진"),
              ("중간", lambda r: r["cls"] == "중간"),
              ("거의 안 움직임", lambda r: r["cls"] == "거의 안 움직임")]
    for lab, sel in groups:
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        e = {"n": len(g), "path": float(np.mean([r["path"] for r in g])),
             "net": float(np.mean([abs(r["net"]) for r in g])),
             "slow": float(np.mean([r["slow"] for r in g])),
             "dk_max": float(np.median([r["dk_max"] for r in g]))}
        for rule in ("fix", "mov"):
            for key in ("th_mean", "th_max", "dth_mean", "dth_max", "rough"):
                v = np.array([r[rule][key] for r in g], float)
                v = v[np.isfinite(v)]
                e[f"{rule}.{key}"] = float(np.median(v))      # 중앙값 — 꼬리가 길다
        out[lab] = e
    return out


def fig(plt, S, rows):
    """두 패널 — ① 증분 축의 거칠기(결정적) ② 절대 오프셋 축(재투영이 이기는 자리)."""
    labs = [l for l in ("회전", "중간", "직진") if l in S]
    xi = np.arange(len(labs))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.5))
    w = 0.36
    for ax, (ka, kb), ylab, title in (
            (axL, ("fix.rough", "mov.rough"), "SD(ΔΔθ) [°/step]  ↓ 작을수록 매끄럽다",
             "① 증분 축 — 재투영은 차 움직임에 없는 톱니를 넣는다"),
            (axR, ("fix.th_mean", "mov.th_mean"), "|θ| 평균 [°]  ↓ 작을수록 내기 쉽다",
             "② 절대 오프셋 축 — 여기서는 재투영이 이긴다")):
        for j, (k, col, lab) in enumerate(((ka, C.C_RED, "고정 — t=49 차로 하나"),
                                           (kb, C.C_AQUA, "재투영 — 매 step 최근접"))):
            v = [S[l][k] for l in labs]
            b = ax.bar(xi + (j - 0.5) * w, v, w, color=col, label=lab, zorder=3)
            ax.bar_label(b, fmt="%.1f", fontsize=8.2, padding=2)
        ax.set_xticks(xi); ax.set_xticklabels([f"{l}\n(n={S[l]['n']:,})" for l in labs])
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10.2, loc="left")
        ax.legend(fontsize=8.4, framealpha=0.93)
        ax.margins(y=0.18)
    fig.tight_layout()
    return C.savefig(fig, OUT / "f1_baseline_rules.png")


DOC = C.ROOT / "docs" / "action_output_head.md"


def md_block(S, n_scene):
    """규격 문서의 표. **손으로 적으면 낡는다** — 실제로 두 번 낡았다(2026-11-26 발견).
    회전의 |θ| 고정값이 300판 예비 측정의 22° 로 남아 있었는데 전수는 35.7° 다."""
    mov = sum(e["n"] for k, e in S.items() if k != "전체 (움직인 판)" and k != "거의 안 움직임")
    L = [f"AV2 val {n_scene:,}판 전수 (`src/measure_frenet_baseline.py`, 미래 60 step, 중앙값):", "",
         "| | n | **신호의 거칠기** 고정 | 재투영 | \\|θ\\| 고정 | 재투영 | **한 step 기준선 점프** |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in ("회전", "직진"):
        e = S[k]
        # ⚠️ 굵게는 **데이터가 정한다**. 열을 고정해 굵게 하면 직진처럼 고정이 이기는
        #    줄에서 거짓말이 된다(2026-11-26 발견 — 재투영 1.5° 가 굵고 고정 1.1° 가 안 굵었다).
        def duo(x, y, f="{:.2f}°"):
            a, b = f.format(x), f.format(y)
            return (f"**{a}**", b) if x <= y else (a, f"**{b}**")
        r1, r2 = duo(e["fix.rough"], e["mov.rough"])
        t1, t2 = duo(e["fix.th_mean"], e["mov.th_mean"], "{:.1f}°")
        L.append(f"| {k} | {e['n']:,} | {r1} | {r2} | {t1} | {t2} | **{e['dk_max']:.1f}°** |")
    tn = S["회전"]["n"] + S["직진"]["n"] + S.get("중간", {"n": 0})["n"]
    L += ["", f"회전은 움직인 판의 **{S['회전']['n'] / tn * 100:.0f}%**, "
              f"직진이 **{S['직진']['n'] / tn * 100:.0f}%** 다."]
    return "\n".join(L)


def fill_doc(S, n_scene):
    if not DOC.exists():
        return
    txt = DOC.read_text()
    a, b = "<!-- FRENET:START -->", "<!-- FRENET:END -->"
    if a not in txt or b not in txt:
        print("  (표식 FRENET 없음 — 문서 갱신 건너뜀)")
        return
    head, rest = txt.split(a, 1)
    _, tail = rest.split(b, 1)
    DOC.write_text(f"{head}{a}\n\n{md_block(S, n_scene)}\n\n{b}{tail}")
    print(f"  문서 갱신 → {DOC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="앞 N 장면만")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--from-json", action="store_true",
                    help="다시 재지 않고 저장된 numbers.json 으로 문서만 갱신")
    a = ap.parse_args()

    if a.from_json:
        d = json.loads((OUT / "numbers.json").read_text())
        fill_doc(d["summary"], d["n_scene"])
        return

    sids = json.loads((M.CACHE / "scenario_id.json").read_text())
    n = a.n or len(sids)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"val {n:,}장면 · 워커 {a.workers}")

    rows, fails = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for j, r in enumerate(ex.map(one, range(n), chunksize=16)):
            if r and "fail" not in r:
                rows.append(r)
            elif r:
                fails.append(r)
            if (j + 1) % 2000 == 0:
                print(f"  {j+1:,} / {n:,} · 성공 {len(rows):,}", flush=True)
    print(f"끝 — 성공 {len(rows):,} / {n:,} · 실패 {len(fails):,}")
    if fails:
        from collections import Counter
        for why, c in Counter(f["fail"].split("(")[0] for f in fails).most_common(5):
            print(f"   {c:>6,}  {why}")
        print()

    S = summarize(rows)
    print(f"{'':16}{'n':>7}{'이동':>7}{'|순회전|':>9}"
          f"{'SD(ΔΔθ) 고정':>13}{'재투영':>8}{'|θ| 고정':>9}{'재투영':>8}{'max|Δk|':>9}")
    for lab, e in S.items():
        print(f"{lab:16}{e['n']:>7,}{e['path']:>6.0f}m{e['net']:>8.0f}°"
              f"{e['fix.rough']:>12.2f}°{e['mov.rough']:>7.2f}°"
              f"{e['fix.th_mean']:>8.0f}°{e['mov.th_mean']:>7.0f}°{e['dk_max']:>8.1f}°")
    print("\n(전부 중앙값 — 꼬리가 길다. SD(ΔΔθ) 는 증분의 step 간 거칠기다.)")

    (OUT / "numbers.json").write_text(json.dumps(
        {"n_scene": len(rows), "summary": S,
         "note": "미래 60 step · h 는 위치차분 · k 는 진행방향 게이트 90° 최근접 차로"},
        ensure_ascii=False, indent=1))
    p = fig(C.setup_mpl(), S, rows)
    fill_doc(S, len(rows))
    print(f"\n그림 → {p}\n숫자 → {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
