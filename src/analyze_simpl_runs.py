"""SIMPL 런들을 같은 자로 재서 비교한다 — R0(시드42) · R1((a,h)) · R0-s43(시드43).

왜 스크립트인가 — 같은 분석을 세 번 한다. ① R1 완주(11-04) ② R0-s43 완주(11-06)
③ 그 뒤 R2. 매번 손으로 세면 매번 다르게 샌다(2026-11-02 에 두 세션이 같은 날
에폭 시간을 각각 다르게 틀렸다). 숫자는 전부 여기서 나오고, 문서는 표식 사이를
이 스크립트가 갈아 끼운다.

  python analyze_simpl_runs.py            # 숫자 + 그림 + 문서 갱신
  python analyze_simpl_runs.py --no-doc   # 숫자만 보고 싶을 때

측정의 함정 셋(로그에서 직접 겪은 것):
  ① `time cost` 는 [Training] 과 [Validation] 에 둘 다 있다 → 앵커 없이 grep 하면 섞인다.
  ② 검증 횟수는 `--val_interval 2` 로 추론할 수 없다 — 에폭 25까지 2에폭마다,
     26부터 매 에폭이라 50에폭에 37회다. 센다.
  ③ 두 런을 비교할 때 양쪽을 같은 규칙으로 잘라야 한다. 한쪽만 ep0 을 빼면 없는 차이가 생긴다.
"""
import argparse
import json
import re
from math import comb
from pathlib import Path

import numpy as np

import viz_v4_common as C

SIMPL = Path("/home/user/av2_simpl")
OUT = C.ROOT / "viz" / "r1_analysis"
DOC = C.ROOT / "docs" / "r1_result_analysis.md"

# 이름 → (로그, 한 줄 설명). 없는 로그는 조용히 건너뛴다(아직 안 돈 런이 있다).
RUNS = [
    ("R0",     SIMPL / "r0.log",    "원본 14채널 · 시드 42"),
    ("R1",     SIMPL / "r1.log",    "(a,h) 11채널 · 시드 42"),
    ("R0-s43", SIMPL / "r0s43.log", "원본 14채널 · 시드 43"),
]
DECAY = (36, 40)          # LR 1e-3 → 1e-4 구간 (R0 에서 확인)

# 사전 등록 경계. **고치지 않는다** — 사후에 바뀐 기준과 구분하려고 원래 값을 남긴다.
# R0 전 구간(1~49, 36쌍)의 인접 검증 |Δ minADE6| p90 이다.
PREREG = 0.0415

# ⚠️ 그런데 이 상수는 비교 구간에 따라 틀린다. R0 는 감쇠 뒤(40~49) |Δ| 가 0.000~0.004 로
# 거의 멈춰 있고, 그 9쌍이 전체 36쌍의 25% 라 p90 을 아래로 끌어내린다. 실측:
#     1~49  p90 0.0415     1~31  p90 0.0637     1~37  p90 0.0511
#     26~35 p90 0.0446     40~49 p90 0.0024
# 그래서 경계는 **상수가 아니라 규칙**이다 — "맞대는 구간과 같은 구간의 R0 p90".
# 끝점끼리 맞대는 최종 판정에는 이것조차 못 쓴다(0.002 를 넘는 모든 차이가 '유의' 가 된다).
# **끝점에 쓸 자는 런 간 폭(|R0-s43 − R0|) 뿐이다.**


# ────────────────────────────────────────────────────────────── 읽기

def read_log(path):
    """에폭별 dict 로 — train/val 손실, lr, train/val minADE6, 각각의 소요 분.

    [Training] 과 [Validation] 을 **앵커로** 구분한다. 한 에폭 안에서 minade_k 줄이
    두 번 나오는데(학습 뒤 / 검증 뒤), 어느 쪽인지는 바로 앞 줄로만 알 수 있다.
    """
    if not path.exists():
        return []
    rows, cur, pend = [], None, None
    for ln in path.read_text(errors="replace").splitlines():
        m = re.match(r"^Epoch (\d+)", ln)
        if m:
            cur = {"ep": int(m.group(1))}
            rows.append(cur)
            pend = None
            continue
        if cur is None:
            continue
        m = re.match(r"^\[Training\] Avg\. loss: ([\d.]+), time cost: ([\d.]+) mins, lr: ([\d.e-]+)", ln)
        if m:
            cur["tr"], cur["tr_min"], cur["lr"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
            pend = "tr"
            continue
        m = re.match(r"^\[Validation\] Avg\. loss: ([\d.]+), time cost: ([\d.]+) mins", ln)
        if m:
            cur["va"], cur["va_min"] = float(m.group(1)), float(m.group(2))
            pend = "va"
            continue
        m = re.search(r"minade_1: ([\d.]+) minfde_1: ([\d.]+) mr_1: ([\d.]+).*"
                      r"minade_k: ([\d.]+) minfde_k: ([\d.]+) mr_k: ([\d.]+)", ln)
        if m and pend:
            cur[f"{pend}_ade1"] = float(m.group(1))
            cur[f"{pend}_fde1"] = float(m.group(2))
            cur[f"{pend}_mr1"] = float(m.group(3))
            cur[f"{pend}_ade"] = float(m.group(4))
            cur[f"{pend}_fde"] = float(m.group(5))
            cur[f"{pend}_mr"] = float(m.group(6))
            pend = None
    return [r for r in rows if "tr" in r]


def vals(rows):
    """검증이 끝난 에폭만 {에폭: minADE6}."""
    return {r["ep"]: r["va_ade"] for r in rows if "va_ade" in r}


# ────────────────────────────────────────────────────────────── 재기

def jitter(rows, upto=None):
    """런 **내부** 인접 검증 사이의 |Δ minADE6|. 이것은 시드 잡음이 아니다.

    `upto` 를 주면 그 에폭까지만 — 두 런을 맞댈 때는 **같은 구간**이어야 한다.
    R0 은 감쇠 뒤 10에폭이 0.795~0.800 으로 거의 평평해서, 전 구간으로 재면
    중앙값이 아래로 눌린다. 진행 중인 런과 그대로 비교하면 없는 차이가 생긴다.
    """
    v = vals(rows)
    ep = [e for e in sorted(v) if upto is None or e <= upto]
    d = np.abs(np.diff([v[e] for e in ep]))
    if not len(d):
        return None
    return {"n": int(len(d)), "median": float(np.median(d)), "mean": float(d.mean()),
            "p90": float(np.percentile(d, 90)), "max": float(d.max())}


def between(a_rows, b_rows):
    """두 런 **사이**의 에폭별 차이. 공통 검증 에폭에서만."""
    a, b = vals(a_rows), vals(b_rows)
    ep = sorted(set(a) & set(b))
    if not ep:
        return None
    d = np.array([b[e] - a[e] for e in ep])
    return {"eps": ep, "diff": d.tolist(), "n": len(ep),
            "median": float(np.median(d)), "mean": float(d.mean()),
            "abs_median": float(np.median(np.abs(d))), "abs_p90": float(np.percentile(np.abs(d), 90)),
            "abs_max": float(np.abs(d).max()),
            "b_better": int((d < 0).sum()), "a_better": int((d > 0).sum()), "tie": int((d == 0).sum())}


def sign_p(k, n):
    """단측 이항검정. ⚠️ 표본이 독립이 아니다 — 한 학습 궤적의 인접 점이라 p 가 과소평가된다."""
    if n == 0:
        return float("nan")
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def timing(rows):
    tr = [r["tr_min"] for r in rows if "tr_min" in r]
    va = [r["va_min"] for r in rows if "va_min" in r]
    if not tr:
        return None
    return {"n_tr": len(tr), "tr_mean": float(np.mean(tr)), "tr_min": min(tr), "tr_max": max(tr),
            "n_va": len(va), "va_mean": float(np.mean(va)) if va else float("nan"),
            "total": float(sum(tr) + sum(va))}


def decay_slope(rows, lo=DECAY[0], hi=DECAY[1]):
    """LR 감쇠 구간에서 얼마나 떨어졌나. R0 의 −0.038 이 비교 기준."""
    v = vals(rows)
    ep = [e for e in sorted(v) if lo <= e <= hi]
    if len(ep) < 2:
        return None
    return {"eps": ep, "first": v[ep[0]], "last": v[ep[-1]], "drop": v[ep[-1]] - v[ep[0]]}


def matched_boundary(upto):
    """비교 구간과 **같은 구간**의 R0 런 내부 p90. 경계는 상수가 아니라 이 규칙이다."""
    j = jitter(read_log(SIMPL / "r0.log"), upto=upto)
    return j["p90"] if j else None


def summarize():
    runs = {}
    for name, path, desc in RUNS:
        rows = read_log(path)
        if not rows:
            continue
        v = vals(rows)
        last = max(v) if v else None
        runs[name] = {
            "desc": desc, "log": str(path), "n_epoch": len(rows), "n_val": len(v),
            "last_val_ep": last,
            "final": {k: rows[-1].get(k) for k in ("va_ade", "va_fde", "va_mr", "va_ade1", "va")}
                     if last is not None else None,
            "best": {"ep": min(v, key=v.get), "ade": min(v.values())} if v else None,
            "jitter": jitter(rows), "timing": timing(rows), "decay": decay_slope(rows),
            "curve": {str(e): v[e] for e in sorted(v)},
        }
    # R1 이 아직 진행 중이면, R0 를 같은 에폭까지만 다시 재서 나란히 둔다(함정 ③).
    if "R1" in runs and "R0" in runs:
        upto = runs["R1"]["last_val_ep"]
        if upto is not None and upto < max(int(k) for k in runs["R0"]["curve"]):
            runs["R0"]["jitter_matched"] = jitter(read_log(SIMPL / "r0.log"), upto=upto)
            runs["R0"]["matched_upto"] = upto
    out = {"runs": runs, "pairs": {}, "prereg_boundary": PREREG, "boundary": {}}
    if "R0" in runs and "R1" in runs:
        pr = between(read_log(SIMPL / "r0.log"), read_log(SIMPL / "r1.log"))
        out["pairs"]["R1-R0"] = pr
        if pr:
            out["boundary"]["R1-R0"] = {"upto": pr["eps"][-1], "p90": matched_boundary(pr["eps"][-1])}
    out["boundary"]["windows"] = {
        f"{lo}~{hi}": (jitter(read_log(SIMPL / "r0.log"), upto=hi) or {}).get("p90")
        for lo, hi in ((1, 31), (1, 37), (1, 49))}
    if "R0" in runs and "R0-s43" in runs:
        out["pairs"]["R0s43-R0"] = between(read_log(SIMPL / "r0.log"), read_log(SIMPL / "r0s43.log"))
    return out


# ────────────────────────────────────────────────────────────── 그림

def fig_curves(plt, S):
    """세 런의 val minADE6 — 감쇠 구간을 띠로."""
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    style = {"R0": (C.C_TOP, "o", "-"), "R1": (C.C_ORANGE, "s", "-"),
             "R0-s43": (C.C_VIOLET, "^", "--")}
    for name, r in S["runs"].items():
        col, mk, ls = style.get(name, (C.MUTED, ".", "-"))
        ep = [int(k) for k in r["curve"]]
        ad = [r["curve"][k] for k in r["curve"]]
        ax.plot(ep, ad, color=col, marker=mk, ms=4.4, ls=ls, lw=1.5, zorder=4)
        ax.text(ep[-1] + 0.4, ad[-1], f" {name}", va="center", fontsize=9, color=col)
    ax.axvspan(DECAY[0] - 0.5, DECAY[1] + 0.5, color=C.C_GREEN, alpha=0.07, zorder=0)
    ax.text(sum(DECAY) / 2, ax.get_ylim()[1], "LR 감쇠", ha="center", va="top",
            fontsize=8.4, color=C.C_GREEN)
    ax.set_xlabel("에폭"); ax.set_ylabel("val minADE6 [m]")
    n = len(S["runs"])
    ax.set_title(f"같은 자로 잰 {'두' if n == 2 else '세'} 런 — 바뀐 것은 입력과 시드뿐이다")
    fig.tight_layout()
    C.savefig(fig, OUT / "a1_curves.png")


def fig_effect_vs_noise(plt, S):
    """핵심 그림 — (R1 − R0) 를 (R0s43 − R0) 잡음 띠 위에 올린다.

    효과가 잡음 띠 밖으로 나가야 '표현 덕분'이라고 말할 수 있다.
    띠가 아직 없으면(s43 미완) 사전 등록 경계만 점선으로 그린다.
    """
    pair = S["pairs"].get("R1-R0")
    if not pair:
        return
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    ep, d = pair["eps"], np.array(pair["diff"])

    noise = S["pairs"].get("R0s43-R0")
    if noise:
        nep, nd = noise["eps"], np.abs(np.array(noise["diff"]))
        band = np.interp(ep, nep, nd)
        ax.fill_between(ep, -band, band, color=C.C_VIOLET, alpha=0.16, zorder=1,
                        label="시드 잡음 |R0s43 − R0| — 이 띠 안은 판정 불가")
    else:
        bd = (S["boundary"].get("R1-R0") or {}).get("p90") or PREREG
        upto = (S["boundary"].get("R1-R0") or {}).get("upto")
        ax.axhspan(-bd, bd, color=C.MUTED, alpha=0.13, zorder=1,
                   label=f"같은 구간(에폭 1~{upto}) R0 런 내부 p90 ±{bd:.3f} — 시드 잡음 아님")
        ax.axhline(PREREG, color=C.MUTED, lw=0.9, ls=":", zorder=2)
        ax.axhline(-PREREG, color=C.MUTED, lw=0.9, ls=":", zorder=2)
        ax.text(ep[0], PREREG, f" 사전 등록 {PREREG:.3f} (전 구간 — 너무 좁다)",
                va="bottom", fontsize=7.8, color=C.MUTED)

    ax.axhline(0, color=C.INK, lw=1.0, zorder=2)
    ax.plot(ep, d, color=C.C_ORANGE, marker="s", ms=4.6, lw=1.5, zorder=4)
    ax.fill_between(ep, 0, d, where=d < 0, color=C.C_GREEN, alpha=0.18, zorder=3)
    ax.fill_between(ep, 0, d, where=d > 0, color=C.C_RED, alpha=0.14, zorder=3)
    ax.axvspan(DECAY[0] - 0.5, DECAY[1] + 0.5, color=C.C_GREEN, alpha=0.07, zorder=0)
    for y, lab, col in ((0.97, "↑ R0 가 낫다", C.C_TOP), (0.03, "↓ R1 이 낫다", C.C_ORANGE)):
        ax.text(0.008, y, lab, transform=ax.transAxes, va="top" if y > 0.5 else "bottom",
                fontsize=8.8, color=col)
    med = float(np.median(d))
    ax.axhline(med, color=C.C_RED, lw=1.0, ls=":", zorder=2)
    ax.text(ep[0], med, f" 중앙 {med:+.3f}", va="bottom", fontsize=8.4, color=C.C_RED)
    ax.text(sum(DECAY) / 2, ax.get_ylim()[0], "LR 감쇠\n(아직 안 옴)", ha="center", va="bottom",
            fontsize=8.2, color=C.C_GREEN)
    ax.set_xlabel("에폭"); ax.set_ylabel("R1 − R0  [m]")
    ax.set_title("효과를 잡음 위에 올려 본다 — 띠 안이면 '차이 없음'이다")
    ax.legend(loc="upper right", fontsize=8.4, framealpha=0.92)
    fig.tight_layout()
    C.savefig(fig, OUT / "a2_effect_vs_noise.png")


# ────────────────────────────────────────────────────────────── 문서

def md_block(S):
    """문서의 <!-- NUMBERS --> 구간에 넣을 표. 숫자는 전부 여기서만 나온다."""
    L = []
    L.append("| 런 | 설명 | 에폭 | 검증 | 최종 minADE6 | minFDE6 | MR6 | minADE1 | 1에폭 평균 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    def num(x, bold=False, fmt="{:.3f}"):
        if x is None:
            return "—"
        s = fmt.format(x)
        return f"**{s}**" if bold else s

    for name, r in S["runs"].items():
        f, t = r["final"] or {}, r["timing"] or {}
        tag = "" if r["n_epoch"] >= 50 else " (진행 중)"
        L.append(f"| **{name}**{tag} | {r['desc']} | {r['n_epoch']} | {r['n_val']} | "
                 f"{num(f.get('va_ade'), bold=True)} | {num(f.get('va_fde'))} | "
                 f"{num(f.get('va_mr'))} | {num(f.get('va_ade1'))} | "
                 f"{num(t.get('tr_mean'), fmt='{:.2f}분')} |")
    L.append("")
    L.append("**런 내부 잡음** — 인접 검증 사이의 |Δ minADE6|. ⚠️ 이것은 시드 잡음이 아니다.")
    L.append("")
    L.append("| 런 | 쌍 | 중앙 | 평균 | p90 | 최대 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for name, r in S["runs"].items():
        for j, tag in ((r.get("jitter"), name),
                       (r.get("jitter_matched"), f"{name} — 에폭 {r.get('matched_upto')}까지 (같은 구간)")):
            if j:
                L.append(f"| {tag} | {j['n']} | {j['median']:.4f} | {j['mean']:.4f} | "
                         f"**{j['p90']:.4f}** | {j['max']:.4f} |")
    for key, lab in (("R0s43-R0", "시드 잡음 — |R0-s43 − R0| (판정의 자)"),
                     ("R1-R0", "효과 후보 — R1 − R0")):
        p = S["pairs"].get(key)
        if not p:
            continue
        # key "R1-R0" 는 diff = R1 − R0 를 뜻한다. between(a=R0, b=R1) 이므로
        # a_better(=d>0 인 에폭 수)는 **뒤쪽 이름(R0)** 의 승수다. 뒤집지 말 것.
        base, other = key.split("-")[0], key.split("-")[1]
        win, k = (other, p["a_better"]) if p["a_better"] >= p["b_better"] else (base, p["b_better"])
        L.append("")
        L.append(f"**{lab}** — 공통 검증 {p['n']}개")
        L.append("")
        L.append("| | 중앙 | 평균 | 절대차 중앙 | 절대차 p90 | 절대차 최대 | 우세 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        L.append(f"| {key} | {p['median']:+.4f} | {p['mean']:+.4f} | {p['abs_median']:.4f} | "
                 f"{p['abs_p90']:.4f} | {p['abs_max']:.4f} | "
                 f"**{win} 가 {k}/{p['n']}** (단측 p={sign_p(k, p['n']):.5f}) |")
        L.append("")
        L.append("⚠️ p 는 **독립 가정**이다 — 한 학습 궤적의 인접 점이라 자기상관으로 **과소평가**된다. "
                 "그리고 부호는 **지속성**만 재지 **원인**은 못 가른다(초기화 제비가 나쁘면 똑같이 나온다).")
    w = S.get("boundary", {}).get("windows") or {}
    if w:
        L.append("")
        L.append("**경계는 상수가 아니라 규칙이다** — 맞대는 구간과 **같은 구간**의 R0 런 내부 p90.")
        L.append("")
        L.append("| R0 를 어디까지 잘랐나 | 런 내부 p90 | 쓰는 곳 |")
        L.append("| --- | --- | --- |")
        use = {"1~31": "지금(R1 에폭 31까지)", "1~37": "에폭 37 판정", "1~49": "사전 등록에 쓴 값 — 너무 좁다"}
        for k, v in w.items():
            if v is not None:
                L.append(f"| 에폭 {k} | **{v:.4f}** | {use.get(k, '')} |")
        L.append("")
        L.append("⚠️ **끝점 판정에는 이 규칙조차 못 쓴다.** R0 는 감쇠 뒤(40~49) 인접 |Δ| 가 "
                 "0.000~0.004 라, 런 내부 경계를 그대로 쓰면 **0.002 를 넘는 모든 차이가 '유의'** 가 된다. "
                 "**끝점에 쓸 자는 런 간 폭(|R0-s43 − R0|) 뿐이다.**")
    for name, r in S["runs"].items():
        if r["decay"]:
            d = r["decay"]
            L.append("")
            L.append(f"**{name} 감쇠 구간 {DECAY[0]}~{DECAY[1]}**: "
                     f"{d['first']:.3f} → {d['last']:.3f}  (**{d['drop']:+.3f}**)")
    return "\n".join(L)


def write_doc(S):
    if not DOC.exists():
        print(f"  (문서 없음 — 건너뜀: {DOC})")
        return
    txt = DOC.read_text()
    a, b = "<!-- NUMBERS:START -->", "<!-- NUMBERS:END -->"
    if a not in txt or b not in txt:
        print("  (표식 없음 — 문서 갱신 건너뜀)")
        return
    head, rest = txt.split(a, 1)
    _, tail = rest.split(b, 1)
    DOC.write_text(f"{head}{a}\n\n{md_block(S)}\n\n{b}{tail}")
    print(f"  문서 갱신 → {DOC}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-doc", action="store_true")
    a = ap.parse_args()
    S = summarize()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    for name, r in S["runs"].items():
        f = r["final"] or {}
        print(f"  {name:7s} 에폭 {r['n_epoch']:2d} · 검증 {r['n_val']:2d} · "
              f"최종 {f.get('va_ade')} · 1에폭 {r['timing']['tr_mean']:.2f}분")
    for k, p in S["pairs"].items():
        print(f"  {k}: 공통 {p['n']}개 · 중앙 {p['median']:+.4f} · 상대 우세 {p['b_better']}/{p['n']}")
    plt = C.setup_mpl()
    fig_curves(plt, S)
    fig_effect_vs_noise(plt, S)
    if not a.no_doc:
        write_doc(S)
    print(f"  숫자 → {OUT / 'numbers.json'}")
