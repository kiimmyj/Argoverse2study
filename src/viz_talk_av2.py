"""연구실 소개용(10분) 그림 — AV2 를 모르는 사람에게 보여줄 장면과 표현 비교.

왜 따로 그리나 — `viz/r1_analysis/` 와 `viz/v4/` 의 그림은 **분석용**이다. 주석이 많고
가로가 길어 빔에 띄우면 글씨가 안 보인다. 소개용은 **그림 하나에 주장 하나**로 크게 그린다.

무엇을 그리나
  scene   장면 한 장 — 지도 + 주변 차 + 과거 5초 + 정답 6초 + 모델이 낸 후보 6개.
          AV2 를 모르는 사람에게 "문제가 뭔지"를 한 장으로 끝내는 그림이다.
  repr    같은 궤적 하나를 (x, y) 로 적은 것과 (a, h) 로 적은 것을 나란히.
          내 아이디어가 뭔지 말로 설명하지 않고 그림으로 끝내는 그림이다.

  python src/viz_talk_av2.py --find 600      # 장면 후보 고르기 (모델 안 씀)
  python src/viz_talk_av2.py --scene <seq_id>
  python src/viz_talk_av2.py --repr  <seq_id>

⚠️ 좌표계 — SIMPL 캐시는 **agent 마다 자기 프레임**에 궤적을 담는다. 장면 프레임으로 보려면
`TRAJS_VECS` 로 만든 회전을 먹이고 `TRAJS_CTRS` 를 더해야 한다. 안 하면 모든 차가 원점에서
오른쪽을 보고 겹쳐 그려진다. 차선도 같다(`lane_vecs`/`lane_ctrs`).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIMPL = Path("/home/user/av2_simpl")
sys.path.insert(0, str(SIMPL))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import viz_v4_common as C

FEAT = "/data/argoverse2/simpl_features/val/"
FIGSIZE = (12.8, 7.2)          # 빔 화면 16:9
AR = FIGSIZE[0] / FIGSIZE[1]
OUT = ROOT / "viz" / "talk"


def _rot(vec):
    """[...,2] 단위벡터 → 회전행렬 [...,2,2]. agent 프레임 → 장면 프레임."""
    c, s = vec[..., 0], vec[..., 1]
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def to_scene(traj, ctr, vec):
    """[T,2] agent 프레임 궤적 → 장면 프레임."""
    return traj @ _rot(vec).T + ctr


def load_scene(idx_or_id):
    """val 데이터셋에서 한 장면. 정수면 인덱스, 문자열이면 seq_id(UUID)."""
    from simpl.av2_dataset import AV2Dataset
    from utils.utils import from_numpy
    ds = AV2Dataset(FEAT, mode="val", verbose=False)
    if isinstance(idx_or_id, int):
        i = idx_or_id
    else:
        names = [Path(f).stem for f in ds.dataset_files]
        i = names.index(idx_or_id)
    return from_numpy([ds[i]])[0], ds, i


def turn_score(d):
    """focal 이 미래 6초 동안 얼마나 방향을 트나 [deg] + 장면이 얼마나 복잡한가.

    **과거가 아니라 미래**로 잰다 — 소개용 그림은 "답이 하나가 아니다" 를 보여야 하고,
    그게 보이는 장면은 focal 이 앞으로 갈림길에 서는 장면이다.
    """
    fut = d["TRAJS"]["TRAJS_POS_FUT"][0].numpy()          # [60,2] agent 프레임
    v = np.diff(fut, axis=0)
    n = np.linalg.norm(v, axis=1)
    ok = n > 0.05
    if ok.sum() < 10:
        return dict(turn=0.0, disp=0.0, n_agent=0, n_lane=0)
    h = np.unwrap(np.arctan2(v[ok, 1], v[ok, 0]))
    return dict(turn=float(np.degrees(h[-1] - h[0])),
                disp=float(np.linalg.norm(fut[-1] - fut[0])),
                n_agent=int(d["TRAJS"]["TRAJS_CTRS"].shape[0]),
                n_lane=int(d["LANE_GRAPH"]["lane_ctrs"].shape[0]))


def find(n):
    """소개용으로 쓸 만한 장면 후보. 모델을 안 돌리므로 싸다."""
    from simpl.av2_dataset import AV2Dataset
    from utils.utils import from_numpy
    ds = AV2Dataset(FEAT, mode="val", verbose=False)
    n = min(n, len(ds))
    rows = []
    for i in range(n):
        d = from_numpy([ds[i]])[0]
        s = turn_score(d)
        # 보기 좋은 조건: 확실히 틀고(>35°), 충분히 움직이고(>25 m),
        # 너무 붐비지 않고(차 10~45대), 지도가 있는(차선 20개 이상) 장면
        if abs(s["turn"]) > 35 and s["disp"] > 25 and 10 <= s["n_agent"] <= 45 and s["n_lane"] >= 20:
            rows.append((Path(ds.dataset_files[i]).stem, i, s))
        if i % 200 == 0:
            print(f"  {i}/{n} · 후보 {len(rows)}", flush=True)
    rows.sort(key=lambda r: -abs(r[2]["turn"]))
    print(f"\n{n}개 중 후보 {len(rows)}개 — 회전 큰 순 20개\n")
    print(f"{'idx':>6} {'회전[°]':>8} {'이동[m]':>8} {'차':>4} {'차선':>5}  seq_id")
    for sid, i, s in rows[:20]:
        print(f"{i:>6} {s['turn']:>8.0f} {s['disp']:>8.1f} {s['n_agent']:>4} {s['n_lane']:>5}  {sid}")
    return rows


R0_CKPT = SIMPL / "saved_models" / "20261029-100547_Simpl_epoch50.tar"


_NET = {}


def get_net(ckpt=R0_CKPT, mode="orig"):
    """모델·데이터셋을 **한 번만** 올린다. 장면을 수백 개 훑을 때 매번 올리면 그게 비용의 전부다.

    CPU 로 돈다 — GPU 는 전수 측정이 쓰고 있을 수 있고, 한 장면은 CPU 로도 0.2초다.
    """
    key = (str(ckpt), mode)
    if key in _NET:
        return _NET[key]
    os.environ["SIMPL_ACTOR_FEAT"] = mode
    from loader import Loader

    class _A:
        mode = "val"; features_dir = "/data/argoverse2/simpl_features/"
        train_batch_size = 1; val_batch_size = 1
        use_cuda = False; data_aug = False
        adv_cfg_path = "config.simpl_av2_cfg"; model_path = str(ckpt)

    ld = Loader(_A(), torch.device("cpu"), is_ddp=False)
    ld.set_resmue(str(ckpt))
    (_, val_set), net, _, _, _ = ld.load()
    net.eval()
    _NET[key] = (val_set, net)
    return _NET[key]


def predict(idx, ckpt=R0_CKPT, mode="orig"):
    """한 장면 → (장면 dict, [K,60,2] 예측, [K] 확률).

    `traj_pred` 는 **focal 의 agent 프레임**이다(evaluator 가 TRAJS_POS_FUT[0] 과 바로 맞댄다).
    """
    val_set, net = get_net(ckpt, mode)
    data = val_set.collate_fn([val_set[idx]])
    with torch.no_grad():
        post = net.post_process(net(net.pre_process(data)))
    return (data, post["traj_pred"][0, :, :, :2].cpu().numpy(),
            post["prob_pred"][0].cpu().numpy())


def rank_spread(n, top=15):
    """**예측이 갈라지는** 장면을 찾는다. 회전 크기로 고르면 안 된다 — 2026-11-06 에 그랬다가
    모델이 확신하는 우회전을 골라 후보 6개가 한 줄로 겹쳤다. "답이 하나가 아니다" 를
    보이려면 모델 **자신이** 헷갈린 장면이어야 한다.

    점수 = 확률 상위 4개 끝점의 퍼짐[m]. 정답이 그중 어딘가에 있어야 그림이 설득력 있다.
    """
    val_set, net = get_net()
    n = min(n, len(val_set))
    rows = []
    for i in range(n):
        data = val_set.collate_fn([val_set[i]])
        d = data["TRAJS"][0]
        if not (10 <= d["TRAJS_CTRS"].shape[0] <= 45):
            continue
        if data["LANE_GRAPH"][0]["lane_ctrs"].shape[0] < 20:
            continue
        with torch.no_grad():
            post = net.post_process(net(net.pre_process(data)))
        pr = post["traj_pred"][0, :, :, :2].numpy()
        pb = post["prob_pred"][0].numpy()
        k = np.argsort(-pb)[:4]
        ends = pr[k, -1]
        spread = float(np.linalg.norm(ends[:, None] - ends[None], axis=-1).max())
        fut = d["TRAJS_POS_FUT"][0].numpy()
        if np.linalg.norm(fut[-1] - fut[0]) < 20:          # 서 있는 차는 그림이 안 된다
            continue
        err = float(np.linalg.norm(pr[:, -1] - fut[-1], axis=-1).min())
        rows.append((i, spread, err, float(pb.max()),
                     int(d["TRAJS_CTRS"].shape[0]), Path(val_set.dataset_files[i]).stem))
        if i % 100 == 0:
            print(f"  {i}/{n} · 후보 {len(rows)}", flush=True)
    # 많이 갈라지되 **정답을 맞힌** 장면 — 못 맞힌 장면을 소개용으로 쓰면 다른 얘기가 된다
    rows = [r for r in rows if r[2] < 3.0]
    rows.sort(key=lambda r: -r[1])
    print(f"\n퍼짐 큰 순 {top}개 (정답 끝점오차 3 m 이내만)\n")
    print(f"{'idx':>6} {'퍼짐[m]':>9} {'오차[m]':>9} {'1위확률':>8} {'차':>4}  seq_id")
    for i, sp, er, pm, na, sid in rows[:top]:
        print(f"{i:>6} {sp:>9.1f} {er:>9.2f} {pm:>8.2f} {na:>4}  {sid}")
    return rows


def _lanes(g):
    """차선 중심선을 장면 프레임으로. 캐시는 **차선마다 자기 프레임**에 담겨 있다."""
    out = []
    for ctrs, pos, vec in zip(g["node_ctrs"], g["lane_ctrs"], g["lane_vecs"]):
        out.append((ctrs @ _rot(vec).T + pos).numpy())
    return out


def draw_scene(idx, stage, path, title):
    """소개용 장면 그림. stage: 1 = 문제만 · 2 = 정답까지 · 3 = 모델 후보까지."""
    plt = C.setup_mpl()
    data, pred, prob = predict(idx)
    d = data["TRAJS"][0]
    ctrs, vecs, cat = d["TRAJS_CTRS"], d["TRAJS_VECS"], d["TRAJS_CAT"]
    f_ctr, f_vec = ctrs[0], vecs[0]

    # 빔 화면이 16:9 다. 그리고 aspect 는 **datalim** 으로 — box 로 두면 그림이 축 안에서
    # 쪼그라들고 여백만 남는다(2026-11-06 에 그랬다).
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off"); ax.grid(False)

    for ln in _lanes(data["LANE_GRAPH"][0]):
        ax.plot(ln[:, 0], ln[:, 1], color=C.GRID, lw=1.6, zorder=1, solid_capstyle="round")

    # 주변 차 — 과거만, 회색. 개별로 읽히면 안 되고 '상황'으로 읽혀야 한다
    for i in range(1, len(ctrs)):
        if cat[i] == "frag":   # 조각 트랙만 뺀다. unscore 도 실제 차고, 모델이 어텐션으로 본다
            continue
        p = to_scene(d["TRAJS_POS_OBS"][i], ctrs[i], vecs[i]).numpy()
        ax.plot(p[:, 0], p[:, 1], color=C.MUTED, lw=1.6, alpha=0.6, zorder=3)
        C.draw_box(ax, p[-1, 0], p[-1, 1], np.arctan2(vecs[i][1], vecs[i][0]),
                   C.MUTED, alpha=0.8, zorder=4)

    # focal 과거 5초
    past = to_scene(d["TRAJS_POS_OBS"][0], f_ctr, f_vec).numpy()
    ax.plot(past[:, 0], past[:, 1], color=C.C_BLUE, lw=4.0, zorder=8, label="과거 5초 (모델이 보는 것)")
    C.draw_box(ax, past[-1, 0], past[-1, 1], np.arctan2(f_vec[1], f_vec[0]), C.C_BLUE, zorder=10)

    fut = to_scene(d["TRAJS_POS_FUT"][0], f_ctr, f_vec).numpy()
    if stage == 1:
        # ⚠️ 글자를 차 위치에 두면 **차 상자가 글자를 덮는다**(2026-11-06 에 그랬다).
        #    데이터 좌표가 아니라 **화면 오프셋**으로 띄우고 배경 박스를 깐다.
        ax.annotate("6초 뒤 어디?", xy=(fut[len(fut) // 2, 0], fut[len(fut) // 2, 1]),
                    xytext=(-40, 78), textcoords="offset points",
                    fontsize=18, color=C.C_RED, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", fc=C.SURF, ec=C.C_RED, lw=1.2, alpha=0.95),
                    arrowprops=dict(arrowstyle="->,head_width=0.4", color=C.C_RED, lw=2.4,
                                    ls=(0, (4, 3)), shrinkB=6))
    if stage >= 2:
        ax.plot(fut[:, 0], fut[:, 1], color=C.INK, lw=4.0, zorder=9, label="실제로 간 길 (정답)")
        ax.plot(*fut[-1], marker="*", ms=22, color=C.INK, zorder=11, mec="white", mew=1.2)
        ax.annotate("정답", xy=fut[-1], xytext=(10, 10), textcoords="offset points",
                    fontsize=13, color=C.INK, weight="bold", zorder=12)
    if stage >= 3:
        order = np.argsort(-prob)
        for r, k in enumerate(order):
            p = to_scene(torch.as_tensor(pred[k]).float(), f_ctr, f_vec).numpy()
            ax.plot(p[:, 0], p[:, 1], color=C.C_ORANGE, lw=1.6 + 3.4 * float(prob[k]),
                    alpha=0.35 + 0.55 * float(prob[k]), zorder=7,
                    label="모델이 낸 후보 6개" if r == 0 else None)
            ax.plot(*p[-1], marker="o", ms=6, color=C.C_ORANGE,
                    alpha=0.4 + 0.5 * float(prob[k]), zorder=7, mec="white", mew=0.8)

    # 구도는 **그릴 것의 bbox** 에서 낸다. 평균을 중심으로 쓰면 점이 몰린 쪽으로 쏠린다.
    pts = np.concatenate([past, fut] + ([to_scene(torch.as_tensor(pred[k]).float(), f_ctr,
                                                  f_vec).numpy() for k in range(len(pred))]
                                        if stage >= 3 else []))
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    pad = float((hi - lo).max()) * 0.08 + 12
    w, h = float(hi[0] - lo[0]) + 2 * pad, float(hi[1] - lo[1]) + 2 * pad
    # ⚠️ 한 변을 기준으로 정사각 창을 잡으면 16:9 축에서 aspect 를 맞추느라 한쪽이 늘어나
    #    그림이 띠처럼 눌린다. **처음부터 화면 비율로** 창을 잡아야 늘어날 게 없다.
    if w / h < AR:
        w = h * AR
    else:
        h = w / AR
    ax.set_xlim(c[0] - w / 2, c[0] + w / 2)
    ax.set_ylim(c[1] - h / 2, c[1] + h / 2)
    C.scale_bar(ax, ax.get_xlim(), ax.get_ylim())
    ax.set_title(title, fontsize=16, pad=12)
    ax.legend(loc="upper left", fontsize=12.5, frameon=True, framealpha=0.92,
              facecolor=C.SURF, edgecolor=C.GRID)
    fig.tight_layout()
    return C.savefig(fig, path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find", type=int, default=0)
    ap.add_argument("--spread", type=int, default=0, help="예측이 갈라지는 장면 찾기")
    ap.add_argument("--scene", type=int, default=-1, help="val 인덱스")
    ap.add_argument("--repr", type=int, default=-1, help="표현 비교 그림")
    ap.add_argument("--anatomy", type=int, default=-1, help="데이터셋 해부 그림")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.find:
        find(a.find)
    if a.spread:
        rank_spread(a.spread)
    if a.scene >= 0:
        for st, name, title in (
                (1, "t1_problem", "자율주행차가 매 순간 푸는 문제"),
                (2, "t2_answer", "정답은 나중에야 안다"),
                (3, "t3_modes", "그래서 답을 여섯 개 낸다 — 그중 제일 가까운 것으로 채점한다")):
            print("→", draw_scene(a.scene, st, OUT / f"{name}.png", title))
    if a.repr >= 0:
        print("→", draw_repr(a.repr, OUT / "t4_repr.png"))
    if a.anatomy >= 0:
        print("→", draw_anatomy(a.anatomy, OUT / "t0_anatomy.png"))



# ──────────────────────────────────────────────────── 두 표현 비교 (발표 5번 장)

def draw_repr(idx, path):
    """같은 궤적 하나를 **(x, y) 로 적은 것**과 **(a, h) 로 적은 것**을 나란히.

    왜 세 번째 칸(90° 돌리기)이 있나 — 이 아이디어의 핵심은 "좌표는 어디서 보느냐에 따라
    숫자가 통째로 바뀌고, (a, h) 는 거의 안 바뀐다" 다. 말로 하면 안 와닿는데,
    **같은 궤적을 돌려서 두 숫자를 겹쳐 그리면** 한눈에 보인다. 돌린 것이므로 주장이 아니라
    정의상 참이다 — h 는 상수만큼 밀리고 a 는 **그대로**다.
    """
    plt = C.setup_mpl()
    sys.path.insert(0, str(SIMPL))
    from simpl import ah_feat

    data, _, _ = predict(idx)
    d = data["TRAJS"][0]
    # 관측 5초 + 미래 6초를 이어 하나의 궤적으로 본다 — 발표에선 '한 번의 우회전'이 보여야 한다
    p = torch.cat([d["TRAJS_POS_OBS"][0], d["TRAJS_POS_FUT"][0]], 0).float()
    yaw = torch.atan2(d["TRAJS_ANG_OBS"][0][:, 1], d["TRAJS_ANG_OBS"][0][:, 0])
    yaw = torch.cat([yaw, yaw[-1:].repeat(len(p) - len(yaw))])

    # ⚠️ **90° 는 쓰지 않는다.** 직각으로 돌리면 x↔y 가 그대로 맞바뀌어, ② 에서 두 계열이
    #    정확히 포개진다 — "좌표는 통째로 다르다" 가 오히려 "그냥 바뀐 거네" 로 읽힌다.
    #    비스듬한 각이어야 좌표가 섞여서 **정말 다른 숫자**가 된다.
    DEG = 55.0
    th = np.deg2rad(DEG)
    R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).float()
    sets = [("원래", p, yaw, C.C_BLUE), (f"{DEG:.0f}° 돌림", p @ R.T, yaw + th, C.C_ORANGE)]

    # 3칸 — 셋째 칸은 위아래로 쪼갠다(a 와 h 는 단위가 달라 한 축에 못 겹친다).
    fig = plt.figure(figsize=(15.8, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.28)
    axP = fig.add_subplot(gs[:, 0])
    axXY = fig.add_subplot(gs[:, 1])
    axA = fig.add_subplot(gs[0, 2])
    axH = fig.add_subplot(gs[1, 2])
    t = np.arange(len(p)) * 0.1

    axP.set_aspect("equal"); axP.grid(False)
    for lab, q, _, col in sets:
        q = q.numpy()
        axP.plot(q[:, 0], q[:, 1], color=col, lw=3.0, zorder=5, label=lab)
        axP.scatter(q[::5, 0], q[::5, 1], s=16, color=col, zorder=6, ec="white", lw=0.6)
    axP.set_title("① 같은 움직임 — 길이 어디를 향하느냐만 다르다", fontsize=12.5)
    axP.legend(fontsize=10.5, loc="best"); axP.set_xlabel("x [m]"); axP.set_ylabel("y [m]")

    for lab, q, _, col in sets:
        q = q.numpy()
        axXY.plot(t, q[:, 0], color=col, lw=2.4, label=f"{lab} — x")
        axXY.plot(t, q[:, 1], color=col, lw=2.4, ls=(0, (3, 2)), label=f"{lab} — y")
    axXY.set_title("② 좌표로 적으면 — 숫자가 통째로 다르다", fontsize=12.5)
    axXY.set_xlabel("시간 [초]"); axXY.set_ylabel("[m]"); axXY.legend(fontsize=9, loc="best")

    # ⚠️ 두 a 는 **정의상 완전히 같다**(속력은 돌려도 안 변한다). 같은 굵기로 그리면 뒤엣것이
    #    앞엣것을 덮어 "파란 선이 없다" 로 읽힌다. 굵게-가늘게 겹쳐 그리고 말로도 적는다.
    for i, (lab, q, yw, col) in enumerate(sets):
        v, h, _ = ah_feat.speed_and_heading(q[None], yw[None])
        v = v[0].numpy() * 3.6
        a_ = np.diff(v, prepend=v[0]) / 3.0
        axA.plot(t, a_, color=col, lw=4.5 if i == 0 else 1.8,
                 ls="-" if i == 0 else (0, (4, 3)), label=lab, zorder=4 + i)
        axH.plot(t, np.degrees(np.unwrap(h[0].numpy())), color=col, lw=2.6, label=lab)
    axA.set_title("③ a (속도 변화) — 돌려도 그대로", fontsize=12)
    axA.set_ylabel("a"); axA.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.92)
    axA.annotate("두 선이 포개져 있다", xy=(0.30, 0.04), xycoords="axes fraction",
                 ha="center", fontsize=10, color=C.C_GREEN, weight="bold")
    axH.set_title(f"④ h (진행방향) — 통째로 {DEG:.0f}° 밀릴 뿐", fontsize=12)
    axH.set_xlabel("시간 [초]"); axH.set_ylabel("h [°]"); axH.legend(fontsize=8.6, loc="best")

    fig.suptitle("같은 궤적, 두 가지 적는 법 — 내 질문은 \"오른쪽으로 적으면 쉬워지는가\" 다",
                 fontsize=15, y=0.99)
    fig.subplots_adjust(top=0.86, left=0.05, right=0.985, bottom=0.11)
    return C.savefig(fig, path, dpi=150)


# ──────────────────────────────────────────────── 데이터셋 해부 (발표 2번 장)

ROLE = {"focal": ("점수를 받는 한 대", None), "av": ("데이터를 모은 차 (AV)", None),
        "score": ("점수엔 안 들어가지만 끝까지 추적된 차", None),
        "unscore": ("추적은 됐지만 평가 대상 아님", None),
        "frag": ("조각 — 5초 중 일부만 보이다 끊긴 트랙", None)}


def draw_anatomy(idx, path):
    """장면 하나에 **무엇이 들어 있나** — 트랙을 역할별로 칠한다.

    왜 이 그림이 필요한가 — AV2 는 책상에서 만든 데이터가 아니라 **자율주행차가 돌아다니며
    인식 시스템으로 추적한 결과**다. 그래서 깔끔하지 않다: 트랙의 절반 이상이 조각이고,
    관측 구간 step 의 1/5 이 비어 있다. 그런데 **focal 한 대만은 5초가 온전히 보장**된다.
    이 비대칭이 뒤에서 "주변 차 입력이 뭉개졌나" 를 따지는 자리의 출발점이다.
    """
    plt = C.setup_mpl()
    data, _, _ = predict(idx)
    d = data["TRAJS"][0]
    ctrs, vecs, cat = d["TRAJS_CTRS"], d["TRAJS_VECS"], d["TRAJS_CAT"]
    pad = d["PAD_OBS"].numpy().astype(bool)

    col = {"focal": C.C_BLUE, "av": C.C_GREEN, "score": C.C_VIOLET,
           "unscore": C.C_AQUA, "frag": C.MUTED}
    cnt = {k: sum(1 for c in cat if c == k) for k in col}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off"); ax.grid(False)
    for ln in _lanes(data["LANE_GRAPH"][0]):
        ax.plot(ln[:, 0], ln[:, 1], color=C.GRID, lw=1.6, zorder=1, solid_capstyle="round")

    seen, pts = set(), []
    for i in range(len(ctrs)):
        c = cat[i]
        p = to_scene(d["TRAJS_POS_OBS"][i], ctrs[i], vecs[i]).numpy()
        # ⚠️ 조각 트랙은 **보인 step 만** 그려야 한다. 전처리가 결측을 최근접 복사로 메워서
        #    그냥 그리면 한 점에 뭉친 꼬리가 생기고, "조각" 이라는 사실이 안 보인다.
        vis = p[pad[i]]
        lab = None if c in seen else f"{ROLE[c][0]} · {cnt[c]}대"
        seen.add(c)
        w = 4.2 if c == "focal" else (2.4 if c in ("av", "score") else 1.5)
        if len(vis) >= 2:
            ax.plot(vis[:, 0], vis[:, 1], color=col[c], lw=w, alpha=0.95 if c == "focal" else 0.75,
                    zorder=9 if c == "focal" else 4, label=lab)
        elif len(vis) == 1:
            ax.plot(*vis[0], marker="o", ms=4, color=col[c], zorder=4, label=lab)
        C.draw_box(ax, p[-1, 0], p[-1, 1], float(np.arctan2(vecs[i][1], vecs[i][0])),
                   col[c], alpha=0.9 if c == "focal" else 0.55,
                   zorder=10 if c == "focal" else 5)
        if c in ("focal", "av", "score", "unscore"):
            pts.append(p)

    q = np.concatenate(pts)
    lo, hi = q.min(0), q.max(0)
    cen, pad_m = (lo + hi) / 2, float((hi - lo).max()) * 0.08 + 14
    w_, h_ = float(hi[0] - lo[0]) + 2 * pad_m, float(hi[1] - lo[1]) + 2 * pad_m
    if w_ / h_ < AR:
        w_ = h_ * AR
    else:
        h_ = w_ / AR
    ax.set_xlim(cen[0] - w_ / 2, cen[0] + w_ / 2); ax.set_ylim(cen[1] - h_ / 2, cen[1] + h_ / 2)
    C.scale_bar(ax, ax.get_xlim(), ax.get_ylim())
    ax.set_title(f"장면 하나에 들어 있는 것 — 추적 {len(ctrs)}대 · 차선 조각 "
                 f"{len(data['LANE_GRAPH'][0]['lane_ctrs'])}개 · 과거 5초만 표시",
                 fontsize=15, pad=12)
    ax.legend(loc="upper left", fontsize=11.5, frameon=True, framealpha=0.93,
              facecolor=C.SURF, edgecolor=C.GRID)
    fig.tight_layout()
    return C.savefig(fig, path, dpi=150)

if __name__ == "__main__":
    main()
