#!/usr/bin/env python
"""viz_cleanse_report.py - cleanse_census 의 전수 결과를 표·요약 json·시나리오 ID 목록으로 만든다.

  python src/viz_cleanse_report.py                       # 표를 찍고 data/summary.json · ids_*.json 을 쓴다

산출물
  viz/v4/cleanse/data/summary.json     항목별 건수·비율 (train/val), 겹침 표, 정의
  viz/v4/cleanse/data/ids_<split>.json 플래그별 시나리오 ID 목록 (나중에 걸러내기용)

그림은 viz_cleanse_figs.py 가 이 json 이 아니라 npz 를 직접 읽어 그린다(같은 정의를 두 번 쓰지 않도록
플래그 정의는 여기 FLAGS 한 곳에만 둔다 — figs 가 import 해서 쓴다).
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cleanse_census import PATH_LBL, CAUSE, FIELDS, TYPE_NAME  # noqa: E402,F401

# focal 이 차로 위를 달리는 대상인가. v4 의 출력 공간(경로 위 s, d + 차로 밴드)은
# 보행자·자전거에는 애초에 성립하지 않는다 — 그래서 A 계열 이탈을 이 축으로 갈라 본다.
ROAD_TYPES = {1, 4, 5}       # vehicle / motorcyclist / bus
WALK_TYPES = {2, 3, 9}       # pedestrian / cyclist / riderless_bicycle

REPO = SRC.parent
OUT = REPO / "viz/v4/cleanse"
DATA = OUT / "data"
OBS, FUT, T_ALL = 50, 60, 110

# ---------------------------------------------------------------- 시나리오 플래그 (한 곳에만 정의한다)
# 이름 -> (설명, 판정식). 전부 **시나리오 단위 bool** 이다.
FLAGS = {
    "A1 경로없음": ("지도가 후보 경로를 하나도 못 준다 (지금은 직진 폴백 1개로 때움)",
                 lambda d: d["route_none"].astype(bool)),
    "A2 차로밖(한 스텝이라도)": ("focal 110스텝 중 중심선 5 m 밖이 1스텝 이상",
                            lambda d: (d["far_obs"] + d["far_fut"]) > 0),
    "A2 차로밖(절반 이상)": ("focal 110스텝의 절반 이상이 중심선 5 m 밖",
                        lambda d: (d["far_obs"] + d["far_fut"]) > T_ALL / 2),
    "A2 도로밖(절반 이상)": ("focal 110스텝의 절반 이상이 주행가능영역 폴리곤 밖",
                        lambda d: (d["off_da_obs"] + d["off_da_fut"]) > T_ALL / 2),
    "A3 커버리지 실패": ("정답 6초가 어느 후보 경로 밴드에도 안 들어감 (폴백 밴드 포함)",
                    lambda d: d["coverage"] == 0),
    "A3 밴드이탈 심함": ("정답 기준 경로에서 미래 60스텝의 25% 이상이 밴드 밖",
                    lambda d: d["frac_out"] > 0.25),
    "B1 focal 라벨 뒤집힘": ("AV2 heading 이 focal 의 이동방향과 180° 어긋남 (판정 가능한 트랙만)",
                        lambda d: d["f_flip_av2"].astype(bool)),
    "B2 focal 전처리 뒤집기": ("align_ref 가 관측 50스텝에서 focal 의 AV2 heading 을 뒤집음",
                         lambda d: d["f_align_flip"].astype(bool)),
    "B2 주변 전처리 뒤집기≥1": ("입력 후보 주변 트랙 중 1대 이상을 align_ref 가 뒤집음",
                          lambda d: d["a_align_flip"] > 0),
    "B3 ±pi 점프(우리 h)": ("우리 h 의 스텝간 변화가 90° 를 넘는 스텝이 있음",
                        lambda d: d["f_dh_gt90"] > 0),
    "B3 7.3°/step 초과 5%↑": ("우리 h 의 |Δh| 가 7.3° 를 넘는 스텝이 5% 이상",
                          lambda d: d["f_dh_gt73"] > 0.05 * (T_ALL - 1)),
    "B3 ±pi 점프(AV2 필드)": ("AV2 heading 필드 자체에 90° 초과 점프가 있음",
                          lambda d: d["f_dh_gt90_av2"] > 0),
    "B4 focal 방향 미정의": ("focal 이 110스텝 내내 1 m/s 를 못 넘어 이동방향을 못 세움",
                       lambda d: d["f_undef"].astype(bool)),
    "B5 속도 불일치 절반↑": ("|위치차분 속력 − 속도 필드| > 1 m/s 인 스텝이 절반 이상",
                       lambda d: d["f_dv_big"] > 0.5 * d["f_n_dv"]),
    "F focal 이 보행자·자전거": ("focal 종류가 pedestrian / cyclist / riderless_bicycle — 차로 출력공간이 성립하지 않음",
                        lambda d: np.isin(d["f_type"], list(WALK_TYPES))),
}
# 겹침 표에 쓸 대표 플래그 (전부 쓰면 15x15 라 안 읽힌다)
KEY_FLAGS = ["F focal 이 보행자·자전거", "A1 경로없음", "A2 차로밖(한 스텝이라도)", "A2 도로밖(절반 이상)",
             "A3 커버리지 실패", "B1 focal 라벨 뒤집힘", "B2 focal 전처리 뒤집기",
             "B3 ±pi 점프(우리 h)", "B4 focal 방향 미정의", "B5 속도 불일치 절반↑"]
# '버릴 수 없는 것' — 데이터 생성 방식에서 오는 것이라 걸러내면 모집단이 통째로 사라진다
NOT_DROPPABLE = ["창 가장자리 속도 램프", "정지·저속 트랙의 방향 미정의", "AV2 heading 의 슬립각"]


def load(split):
    f = DATA / f"census_{split}.npz"
    if not f.exists():
        return None
    return np.load(f, allow_pickle=True)


def pct(x, n):
    return 100.0 * x / max(n, 1)


def flag_table(D):
    """플래그 -> {split: (건수, %)}"""
    out = {}
    for name, (desc, fn) in FLAGS.items():
        row = {"설명": desc}
        for sp, d in D.items():
            m = fn(d)
            row[sp] = {"n": int(m.sum()), "pct": round(float(pct(m.sum(), len(m))), 4),
                       "N": int(len(m))}
        out[name] = row
    return out


def step_rates(D):
    """스텝 단위 비율 (시나리오가 아니라 스텝이 모집단인 항목)."""
    out = {}
    for sp, d in D.items():
        n = len(d["sid"])
        e = {}
        e["중심선 5 m 밖 (관측 50스텝)"] = pct(d["far_obs"].sum(), n * OBS)
        e["중심선 5 m 밖 (정답 60스텝)"] = pct(d["far_fut"].sum(), n * FUT)
        e["같은방향 차로 5 m 밖 (관측)"] = pct(d["far_obs_dir"].sum(), n * OBS)
        e["같은방향 차로 5 m 밖 (정답)"] = pct(d["far_fut_dir"].sum(), n * FUT)
        e["주행가능영역 밖 (관측)"] = pct(d["off_da_obs"].sum(), n * OBS)
        e["주행가능영역 밖 (정답)"] = pct(d["off_da_fut"].sum(), n * FUT)
        far = d["far_obs"].sum() + d["far_fut"].sum()
        off = d["far_off_da_obs"].sum() + d["far_off_da_fut"].sum()
        e["차로밖 중 도로밖 비중"] = pct(off, far)
        e["차로밖 중 주차장·미매핑 비중"] = pct(far - off, far)
        e["|Δh| > 7.3°/step (우리 h)"] = pct(d["f_dh_gt73"].sum(), n * (T_ALL - 1))
        e["|Δh| > 90°/step (우리 h)"] = pct(d["f_dh_gt90"].sum(), n * (T_ALL - 1))
        e["|Δh| > 7.3°/step (AV2 필드)"] = pct(d["f_dh_gt73_av2"].sum(), n * (T_ALL - 1))
        e["|Δh| > 90°/step (AV2 필드)"] = pct(d["f_dh_gt90_av2"].sum(), n * (T_ALL - 1))
        e["속도 불일치 > 1 m/s"] = pct(d["f_dv_big"].sum(), d["f_n_dv"].sum())
        e["속도 불일치 > 2 m/s"] = pct(d["f_dv_huge"].sum(), d["f_n_dv"].sum())
        e["차로 반대 (우리 h)"] = pct(d["f_anti_lane"].sum(), d["f_lane_steps"].sum())
        e["차로 반대 (AV2 필드)"] = pct(d["f_anti_lane_av2"].sum(), d["f_lane_steps"].sum())
        e["보조로 메운 스텝 중 차로 반대"] = pct(d["f_ref_anti"].sum(), d["f_ref_steps"].sum())
        out[sp] = {k: round(float(v), 4) for k, v in e.items()}
    return out


def track_tables(D):
    """트랙 단위 (B1·B2·B4)."""
    out = {}
    for sp, d in D.items():
        e = {"이동거리별 뒤집힘": {}}
        for tag, lab in (("veh", "차량 계열"), ("oth", "그 외")):
            rows = {}
            tot = [0, 0, 0]
            for b in range(5):
                nn = int(d[f"b_{tag}_n_{b}"].sum())
                jj = int(d[f"b_{tag}_j_{b}"].sum())
                ff = int(d[f"b_{tag}_f_{b}"].sum())
                tot = [tot[0] + nn, tot[1] + jj, tot[2] + ff]
                rows[PATH_LBL[b]] = {"트랙": nn, "판정가능": jj, "뒤집힘": ff,
                                     "뒤집힘%": round(pct(ff, jj), 3),
                                     "판정불가%": round(pct(nn - jj, nn), 3)}
            rows["합계"] = {"트랙": tot[0], "판정가능": tot[1], "뒤집힘": tot[2],
                          "뒤집힘%": round(pct(tot[2], tot[1]), 3),
                          "판정불가%": round(pct(tot[0] - tot[1], tot[0]), 3)}
            e["이동거리별 뒤집힘"][lab] = rows
        a_t, a_j = int(d["a_tracks"].sum()), int(d["a_align_judgeable"].sum())
        e["전처리 뒤집기(align_ref)"] = {
            "입력후보 주변트랙": a_t,
            "판정가능": a_j,
            "뒤집음": int(d["a_align_flip"].sum()),
            "뒤집음%(판정가능 대비)": round(pct(d["a_align_flip"].sum(), a_j), 3),
            "게이트판이 판정한 트랙": int(d["a_gated_judged"].sum()),
            "게이트판이 뒤집음": int(d["a_align_flip_gated"].sum()),
            "게이트판 뒤집음%(전체 대비)": round(pct(d["a_align_flip_gated"].sum(), a_t), 3),
            "방향 미정의 트랙%": round(pct(d["a_undef"].sum(), a_t), 3),
        }
        chk, cf = int(d["a_chk"].sum()), int(d["a_chk_flip"].sum())
        e["뒤집기가 차로 방향과 맞나"] = {
            "검사한 트랙": chk, "그중 뒤집힌 트랙": cf,
            "뒤집힌 트랙이 차로 반대%": round(pct(d["a_chk_flip_anti"].sum(), cf), 3),
            "안 뒤집힌 트랙이 차로 반대%": round(pct(d["a_chk_noflip_anti"].sum(), chk - cf), 3),
        }
        out[sp] = e
    return out


def type_table(D):
    """focal 종류별 — A 계열 이탈이 '지도가 나쁜 것'인지 '대상이 차가 아닌 것'인지 가른다."""
    out = {}
    for sp, d in D.items():
        n = len(d["sid"])
        e = {}
        for code in sorted(set(int(x) for x in d["f_type"])):
            m = d["f_type"] == code
            e[TYPE_NAME.get(code, str(code))] = {
                "n": int(m.sum()), "전체%": round(pct(m.sum(), n), 3),
                "경로없음%": round(pct(d["route_none"][m].sum(), m.sum()), 2),
                "커버리지실패%": round(pct((d["coverage"][m] == 0).sum(), m.sum()), 2),
                "차로밖 스텝%": round(pct(d["far_obs"][m].sum() + d["far_fut"][m].sum(),
                                      m.sum() * T_ALL), 2),
                "정지(방향미정의)%": round(pct(d["f_undef"][m].sum(), m.sum()), 2),
                "6초 이동거리 중앙": round(float(np.median(d["move6"][m])), 2)}
        out[sp] = e
    return out


def band_table(D):
    """밴드 폭을 바꾸면 커버리지가 어떻게 되나 (보정 선택지의 비용)."""
    out = {}
    for sp, d in D.items():
        n = len(d["sid"])
        road = np.isin(d["f_type"], list(ROAD_TYPES))
        out[sp] = {
            "전체 n": n, "차량계열 focal n": int(road.sum()),
            "±1.75 고정": round(pct((d["cov_175"] == 0).sum(), n), 3),
            "규칙 밴드(현행)": round(pct((d["coverage"] == 0).sum(), n), 3),
            "±3.6 고정": round(pct((d["cov_36"] == 0).sum(), n), 3),
            "규칙 밴드(차량계열 focal 만)": round(pct((d["coverage"][road] == 0).sum(), road.sum()), 3),
            "±3.6 (차량계열 focal 만)": round(pct((d["cov_36"][road] == 0).sum(), road.sum()), 3)}
    return out


def cause_table(D):
    out = {}
    for sp, d in D.items():
        fail = d["coverage"] == 0
        c = d["cause"][fail]
        n = int(fail.sum())
        out[sp] = {"실패 건수": n, "실패%": round(pct(n, len(d["sid"])), 3),
                   "원인": {CAUSE[i]: {"n": int((c == i).sum()),
                                     "pct": round(pct((c == i).sum(), n), 2)}
                          for i in range(1, len(CAUSE))}}
    return out


# 이미 학습된 판의 시나리오별 점수 — '버리면 무엇을 잃나'를 정확도로 읽기 위해 붙인다.
# 읽기 전용이며 학습하지 않는다. 없으면 이 절은 통째로 건너뛴다.
SCORE_TAG = "v4_l4nw_ah2_2hz_full_sm1_cos30_s0"
SCORE_PARQUET = REPO / "viz/v4" / SCORE_TAG / "data/scenarios.parquet"


def score_join(D):
    if "val" not in D or not SCORE_PARQUET.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(SCORE_PARQUET)[["sid", "minade", "minfde", "cls", "dh6"]]
    d = D["val"]
    c = pd.DataFrame({"sid": d["sid"], "f_type": d["f_type"], "cause": d["cause"],
                      **{k: d[k] for k in ("route_none", "coverage", "f_flip_av2", "f_undef")}})
    m = df.merge(c, on="sid")
    if len(m) == 0:
        return None
    base = float(m.minade.mean())
    groups = {"전체": np.ones(len(m), bool),
              "보행자·자전거 focal": m.f_type.isin(list(WALK_TYPES)).to_numpy(),
              "차량 계열 focal": m.f_type.isin(list(ROAD_TYPES)).to_numpy(),
              "A1 경로없음": (m.route_none == 1).to_numpy(),
              "A3 커버리지 실패": (m.coverage == 0).to_numpy(),
              "A3 커버리지 성공": (m.coverage == 1).to_numpy(),
              "B1 focal 라벨 뒤집힘": (m.f_flip_av2 == 1).to_numpy(),
              "B4 focal 방향 미정의": (m.f_undef == 1).to_numpy()}
    out = {"태그": SCORE_TAG, "n": int(len(m)), "전체 minADE6": round(base, 4), "집단": {}, "빼면": {},
           "커버리지 실패 원인별": {}}
    for k, sel in groups.items():
        s = m[sel]
        out["집단"][k] = {"n": int(len(s)), "minADE6": round(float(s.minade.mean()), 4),
                        "minFDE6": round(float(s.minfde.mean()), 4),
                        "miss%": round(float((s.minfde > 2).mean() * 100), 2)}
        if k != "전체":
            rest = m[~sel]
            out["빼면"][k] = {"남는 n": int(len(rest)),
                             "남는 minADE6": round(float(rest.minade.mean()), 4),
                             "변화": round(float(rest.minade.mean() - base), 4)}
    for i in range(1, len(CAUSE)):
        s = m[(m.coverage == 0) & (m.cause == i)]
        if len(s):
            out["커버리지 실패 원인별"][CAUSE[i]] = {
                "n": int(len(s)), "minADE6": round(float(s.minade.mean()), 4),
                "miss%": round(float((s.minfde > 2).mean() * 100), 2)}
    return out


def overlap(D):
    """대표 플래그끼리의 교집합 (건수와, 행 플래그 대비 비율)."""
    out = {}
    for sp, d in D.items():
        M = {k: FLAGS[k][1](d) for k in KEY_FLAGS}
        n = len(d["sid"])
        tab = {}
        for a in KEY_FLAGS:
            tab[a] = {"n": int(M[a].sum()),
                      "교집합": {b: int((M[a] & M[b]).sum()) for b in KEY_FLAGS}}
        anyf = np.zeros(n, bool)
        for k in KEY_FLAGS:
            anyf |= M[k]
        tab["__하나라도__"] = {"n": int(anyf.sum()), "pct": round(pct(anyf.sum(), n), 3)}
        out[sp] = tab
    return out


def drop_cost(D):
    """'이 기준으로 버리면 몇 %가 빠지나' — 권고 후보 조합별로."""
    combos = {
        "보행자·자전거 focal": ["F focal 이 보행자·자전거"],
        "경로없음만": ["A1 경로없음"],
        "경로없음 + 보행자·자전거": ["A1 경로없음", "F focal 이 보행자·자전거"],
        "경로없음 + 도로밖(절반↑)": ["A1 경로없음", "A2 도로밖(절반 이상)"],
        "커버리지 실패만": ["A3 커버리지 실패"],
        "경로없음 + 커버리지 실패": ["A1 경로없음", "A3 커버리지 실패"],
        "focal 라벨 뒤집힘": ["B1 focal 라벨 뒤집힘"],
        "focal 방향 미정의(정지)": ["B4 focal 방향 미정의"],
        "권고안 A (보행자·자전거 + 경로없음)": ["F focal 이 보행자·자전거", "A1 경로없음"],
        "권고안 B (권고안 A + focal 라벨 뒤집힘 + 차로밖 절반↑)":
            ["F focal 이 보행자·자전거", "A1 경로없음", "B1 focal 라벨 뒤집힘", "A2 차로밖(절반 이상)"],
        "최대안 (권고안 B + 커버리지 실패)":
            ["F focal 이 보행자·자전거", "A1 경로없음", "B1 focal 라벨 뒤집힘",
             "A2 차로밖(절반 이상)", "A3 커버리지 실패"],
    }
    out = {}
    for sp, d in D.items():
        n = len(d["sid"])
        e = {}
        for name, ks in combos.items():
            m = np.zeros(n, bool)
            for k in ks:
                m |= FLAGS[k][1](d)
            e[name] = {"n": int(m.sum()), "pct": round(pct(m.sum(), n), 3),
                       "남는 시나리오": int(n - m.sum())}
        out[sp] = e
    return out


def main():
    D = {sp: d for sp in ("train", "val") if (d := load(sp)) is not None}
    if not D:
        raise SystemExit("census_*.npz 가 없다 — 먼저 src/cleanse_census.py 를 돌려라")
    DATA.mkdir(parents=True, exist_ok=True)

    S = {"모집단": {sp: int(len(d["sid"])) for sp, d in D.items()},
         "시나리오 플래그": flag_table(D),
         "focal 종류별": type_table(D),
         "밴드 폭별 커버리지 실패": band_table(D),
         "스텝 비율": step_rates(D),
         "트랙 집계": track_tables(D),
         "커버리지 실패 원인": cause_table(D),
         "겹침": overlap(D),
         "버릴 때 잃는 양": drop_cost(D),
         "버릴 수 없는 것": NOT_DROPPABLE}
    sc = score_join(D)
    if sc:
        S["학습된 판의 점수 (val)"] = sc
    for sp, d in D.items():
        m = d["f_ramp_ok"].astype(bool)
        S.setdefault("창 가장자리 램프", {})[sp] = {
            "모집단(중간 구간 속력 중앙 ≥ 3 m/s)": int(m.sum()),
            "첫 스텝 속력 / 1~2초 속력 중앙": round(float(np.median(d["f_ramp_head"][m])), 4),
            "끝 스텝 속력 / 9~10초 속력 중앙": round(float(np.median(d["f_ramp_tail"][m])), 4)}
    (DATA / "summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False) + "\n")

    for sp, d in D.items():
        ids = {}
        for name, (_, fn) in FLAGS.items():
            m = fn(d)
            if m.sum() > 50000:      # 걸러낼 후보가 아닌 대량 플래그는 목록을 안 적는다 (npz 로 다시 만들 수 있다)
                ids[name] = {"n": int(m.sum()), "예시": [str(x) for x in d["sid"][m][:200]],
                             "비고": f"5만 건이 넘어 목록 생략 — census_{sp}.npz 로 재현할 수 있다"}
                continue
            ids[name] = [str(x) for x in d["sid"][m]]
        ids["A3 커버리지 실패 원인별"] = {
            CAUSE[i]: [str(x) for x in d["sid"][(d["coverage"] == 0) & (d["cause"] == i)]]
            for i in range(1, len(CAUSE))}
        (DATA / f"ids_{sp}.json").write_text(json.dumps(
            {"split": sp, "n": int(len(d["sid"])),
             "설명": "플래그 -> 시나리오 ID 목록. 정의는 summary.json 의 '시나리오 플래그'.",
             "ids": ids}, indent=1, ensure_ascii=False) + "\n")

    # ---------------- 사람이 읽는 표
    w = max(len(k) for k in FLAGS) + 2
    print("\n=== 시나리오 플래그 (전수) ===")
    print(f"{'항목':<{w}}" + "".join(f"{sp + ' n':>12}{sp + ' %':>9}" for sp in D))
    for name in FLAGS:
        row = f"{name:<{w}}"
        for sp, d in D.items():
            m = FLAGS[name][1](d)
            row += f"{int(m.sum()):>12,}{pct(m.sum(), len(m)):>8.2f}%"
        print(row)
    print("\n=== 스텝 비율 [%] ===")
    keys = list(next(iter(S["스텝 비율"].values())))
    w2 = max(len(k) for k in keys) + 2
    print(f"{'항목':<{w2}}" + "".join(f"{sp:>10}" for sp in D))
    for k in keys:
        print(f"{k:<{w2}}" + "".join(f"{S['스텝 비율'][sp][k]:>9.2f}%" for sp in D))
    print("\n=== focal 종류별 ===")
    for sp in D:
        print(f"[{sp}]")
        for k, v in S["focal 종류별"][sp].items():
            print(f"  {k:<16} n {v['n']:>8,} ({v['전체%']:>5.2f}%)  경로없음 {v['경로없음%']:>5.2f}%  "
                  f"커버리지실패 {v['커버리지실패%']:>5.2f}%  차로밖스텝 {v['차로밖 스텝%']:>5.2f}%  "
                  f"정지 {v['정지(방향미정의)%']:>5.2f}%  이동중앙 {v['6초 이동거리 중앙']:>6.1f} m")
    print("\n=== 밴드 폭별 커버리지 실패 [%] ===")
    for sp in D:
        print(f"[{sp}] " + "  ".join(f"{k} {v}" for k, v in S["밴드 폭별 커버리지 실패"][sp].items()))
    print("\n=== 커버리지 실패 원인 ===")
    for sp in D:
        c = S["커버리지 실패 원인"][sp]
        print(f"[{sp}] 실패 {c['실패 건수']:,}건 ({c['실패%']}%)  " +
              "  ".join(f"{k} {v['n']:,}({v['pct']}%)" for k, v in c["원인"].items()))
    print("\n=== 이동거리별 AV2 heading 뒤집힘 (차량 계열) ===")
    for sp in D:
        t = S["트랙 집계"][sp]["이동거리별 뒤집힘"]["차량 계열"]
        print(f"[{sp}] " + "  ".join(
            f"{k} {v['뒤집힘%']}%(n={v['판정가능']:,})" for k, v in t.items()))
    print("\n=== 버릴 때 잃는 양 ===")
    w3 = max(len(k) for k in S["버릴 때 잃는 양"][list(D)[0]]) + 2
    print(f"{'기준':<{w3}}" + "".join(f"{sp + ' n':>12}{sp + ' %':>9}" for sp in D))
    for k in S["버릴 때 잃는 양"][list(D)[0]]:
        print(f"{k:<{w3}}" + "".join(
            f"{S['버릴 때 잃는 양'][sp][k]['n']:>12,}{S['버릴 때 잃는 양'][sp][k]['pct']:>8.2f}%"
            for sp in D))
    if sc:
        print(f"\n=== 학습된 판({sc['태그']})의 val 점수로 본 '버리면 무엇을 잃나' ===")
        print(f"{'집단':<22}{'n':>8}{'minADE6':>10}{'miss%':>8}   빼면 남는 minADE6")
        for k, v in sc["집단"].items():
            tail = "" if k == "전체" else (f"   {sc['빼면'][k]['남는 minADE6']:.4f} "
                                          f"({sc['빼면'][k]['변화']:+.4f})")
            print(f"{k:<22}{v['n']:>8,}{v['minADE6']:>10.3f}{v['miss%']:>7.1f}%{tail}")
    print(f"\nwrote {DATA / 'summary.json'} / ids_*.json")


if __name__ == "__main__":
    main()
