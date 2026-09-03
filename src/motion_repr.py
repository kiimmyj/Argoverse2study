"""
motion_repr.py - 궤적을 (속도 v / 가속도 a / 방향 h) 표현으로 바꾸고 되돌리는 순수 함수 모음.

기존 파이프라인(dataset_map.py, model_map.py)은 건드리지 않는다.
여기서 표현을 실험해 확정한 뒤에 Dataset 에 붙인다.

--- 배경 ---
모델 입력 (x, y, vx, vy, heading) 5채널을 "액셀/브레이크(a) + 방향(h)" 같은
운전 조작에 가까운 표현으로 바꾸려는 것. AV2 val 실측으로 확인한 사실 3가지:

1. (a, h) 두 개만으로는 위치를 복원할 수 없다. 초기속도 v0 가 빠지기 때문.
   v0 를 주면 복원오차 0.01 m, v0=0 으로 두면 FDE 13.4 m.
2. a 는 위치를 **두 번 미분**한 값이라 노이즈가 크다 (위치차분 기준 std 4.4 m/s^2,
   |a|>3 인 스텝이 19%). v 는 한 번만 미분해서 훨씬 깨끗하다.
3. 방향은 각도(rad/deg)로 넣으면 ±pi 불연속이 생긴다. (sin h, cos h) 2채널이 안전하다.

--- 모드 ---
  mode="vh"   (v, h)      속도 + 방향        v=0 이 곧 '정지'. 적분 1번, 무손실
  mode="ah"   (a, h)      가속도 + 방향      v0 를 따로 줘야 복원 가능
  mode="vah"  (v, a, h)   속도 + 가속도 + 방향  상태(v)와 조작(a)을 둘 다 줌

--- 규약 ---
- pos 는 (T,2) 위치 시퀀스. focal 기준 정규화 좌표를 전제한다.
- dt = 0.1 s (AV2 10 Hz, 실측 확인)
- v, h 는 **위치 차분**에서 만든다 → 되돌리면 오차가 0 이다.
  AV2 의 velocity/heading 필드를 쓰면 부드럽지만 위치와 어긋나 1.31 m 손실이 생긴다
  (heading 필드만 쓰면 0.53 m). heading_field 인자로 선택할 수 있게 열어 뒀다.
- 시간 정렬: 길이 T 유지를 위해 마지막 값을 복제한다.
    v[t] = |pos[t+1]-pos[t]|/dt,  h[t] = atan2(dy,dx),  a[t] = (v[t+1]-v[t])/dt
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DT = 0.1                # AV2 = 10 Hz
MS_TO_KPH = 3.6
KPH_TO_MS = 1.0 / 3.6
STOP_MS = 0.1           # 이보다 느리면 이동방향이 노이즈 → 직전 방향 유지
WHEELBASE = 2.8         # 승용차 축거 [m] (조향각 환산용)
V_FLOOR = 1.0           # 곡률·조향각 계산 시 속력 하한 [m/s] (0으로 나누기 방지)


# ---------------------------------------------------------------- 단위 / 각도
def to_kph(v_ms):
    return np.asarray(v_ms, dtype=np.float64) * MS_TO_KPH


def to_ms(v_kph):
    return np.asarray(v_kph, dtype=np.float64) * KPH_TO_MS


def wrap_pi(a):
    """각도를 (-pi, pi] 로."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def _moving_average(x, w):
    """중심 이동평균 (경계는 가장자리 값으로 패딩). w=1 이면 그대로."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[:len(x)]


# ---------------------------------------------------------------- 궤적 -> 물리량
@dataclass
class Motion:
    """궤적에서 뽑아낸 물리량. 모두 길이 T (마지막 값 복제)."""
    v_ms: np.ndarray        # 속력 [m/s] >= 0        (위치차분, 무손실)
    a_ms2: np.ndarray       # 종방향 가속도 [m/s^2]   + 가속 / - 감속
    h_rad: np.ndarray       # 진행 방향 [rad]         (인코딩용)
    h_exact: np.ndarray     # 복원 전용 이동방향 [rad] (항상 위치차분 기반)
    w_rads: np.ndarray      # 각속도(조향) [rad/s] = dh/dt  (이동방향 기반 — 저속에서 노이즈)
    p0: np.ndarray          # 시작 위치 (2,)
    stopped: np.ndarray     # 정지 여부 (bool, T)
    # AV2 heading 필드 기반 각속도 [rad/s]. 실측상 유일하게 깨끗한 핸들링 신호
    # (lag-1 자기상관 +0.965 vs 이동방향 기반 −0.141). 위치와 정합하지 않아 '입력 전용'.
    w_field: Optional[np.ndarray] = None

    @property
    def v_kph(self):
        return self.v_ms * MS_TO_KPH

    @property
    def a_kphs(self):
        """'초당 몇 km/h 변하는가'. 액셀/브레이크를 체감하기 쉬운 단위."""
        return self.a_ms2 * MS_TO_KPH

    @property
    def h_deg(self):
        return np.degrees(self.h_rad)

    @property
    def w_degs(self):
        return np.degrees(self.w_rads)

    # --- 핸들링(조작) 관점 파생량 ---------------------------------------
    # h(방향)는 '상태'다. 운전자가 직접 만드는 것은 '얼마나 꺾는가'이고,
    # 그것을 표현하는 방식이 아래 넷이다. a(페달)와 짝을 이루는 쪽은 이쪽이다.

    @property
    def kappa(self):
        """곡률 [1/m] = ω/v. '1m 갈 때마다 몇 rad 꺾이는가' — 속도와 무관한 경로 굽힘.
        정지에서 발산하므로 속력에 하한(V_FLOOR)을 둔다."""
        return self.w_rads / np.maximum(self.v_ms, V_FLOOR)

    @property
    def delta_rad(self):
        """조향각 [rad] = atan(L·κ). 자전거모델의 앞바퀴 각도 = 사실상 핸들 각도.
        atan 이 씌워져 있어 곡률이 튀어도 ±90° 안에 갇힌다(자연 클리핑)."""
        return np.arctan(WHEELBASE * self.kappa)

    @property
    def delta_deg(self):
        return np.degrees(self.delta_rad)

    @property
    def kappa_field(self):
        """heading 필드 기반 곡률 [1/m]. 노이즈가 20배 적은 ω_field 를 쓴다."""
        if self.w_field is None:
            return None
        return self.w_field / np.maximum(self.v_ms, V_FLOOR)

    @property
    def delta_field_deg(self):
        """heading 필드 기반 조향각 [deg]. δ 를 공정하게 평가하려면 이 쪽을 써야 한다."""
        k = self.kappa_field
        return None if k is None else np.degrees(np.arctan(WHEELBASE * k))

    @property
    def a_lat(self):
        """횡가속도 [m/s²] = v·ω. 코너에서 몸이 쏠리는 그 힘.
        마찰 한계(약 8 m/s²)를 넘으면 물리적으로 불가능 → 노이즈 판별에 쓸 수 있다."""
        return self.v_ms * self.w_rads

    @property
    def dw_degs(self):
        """h[-1]=0 으로 두고 만든 방향 증분 [deg/s]. 누적하면 h 가 그대로 나온다.
        → 첫 값이 초기 방향 h0 를 싣는다 (dv_kph 와 같은 트릭의 횡방향 버전)."""
        return np.degrees(np.diff(self.h_rad, prepend=0.0)) / 0.1

    @property
    def dv_kph(self):
        """v[-1]=0 으로 두고 만든 속도 증분 [km/h]. 누적합하면 v 가 그대로 나온다.
        → 첫 값이 v0 를 통째로 싣는다 (2채널로 속도를 담는 트릭)."""
        return np.diff(self.v_kph, prepend=0.0)


def traj_to_motion(pos, dt=DT, stop_ms=STOP_MS, heading_field=None,
                   smooth=1, a_clip=None, use_field_heading=False) -> Motion:
    """
    (T,2) 위치 → Motion.

    smooth        : a 를 만들기 전에 속력에 걸 이동평균 창 (홀수). 1=끔.
                    a 는 위치의 2계 미분이라 노이즈가 크다 → 여기서만 완화한다.
                    v 자체는 스무딩하지 않으므로 복원은 계속 무손실이다.
    heading_field : AV2 heading 필드 (T,). 주면 이걸로 w_field(깨끗한 각속도)를 만든다.
                    h_rad 는 그대로 이동방향을 쓴다 — heading 필드는 위치와 정합하지
                    않아서(슬립각) 복원에 쓰면 ADE 0.5 m 를 잃기 때문이다.
    use_field_heading : True 면 h_rad 까지 heading 필드로 덮어쓴다(진단용).
    a_clip        : |a| 상한 [m/s^2]. 노이즈 꼬리를 자를 때.
    """
    pos = np.asarray(pos, dtype=np.float64)
    d = np.diff(pos, axis=0)                        # (T-1,2)
    v = np.linalg.norm(d, axis=1) / dt              # (T-1,)
    h_exact = np.arctan2(d[:, 1], d[:, 0])          # (T-1,)

    # 정지 구간의 이동방향은 노이즈 → 직전 유효 방향 유지 (v≈0 이라 복원엔 거의 영향 없음)
    stopped = v < stop_ms
    h_enc = h_exact.copy()
    last = h_enc[~stopped][0] if (~stopped).any() else 0.0
    for t in range(len(h_enc)):
        if stopped[t]:
            h_enc[t] = last
        else:
            last = h_enc[t]

    a = np.diff(_moving_average(v, smooth)) / dt    # (T-2,)
    if a_clip is not None:
        a = np.clip(a, -a_clip, a_clip)

    # 길이 T 로 맞춤
    v = np.append(v, v[-1])
    h_exact = np.append(h_exact, h_exact[-1])
    h_enc = np.append(h_enc, h_enc[-1])
    a = np.append(a, a[-1] if len(a) else 0.0)
    a = np.append(a, a[-1])[:len(v)]
    stopped = np.append(stopped, stopped[-1])

    w_field = None
    if heading_field is not None:
        hf = wrap_pi(np.asarray(heading_field, dtype=np.float64)[:len(v)])
        w_field = np.append(wrap_pi(np.diff(hf)) / dt, 0.0)[:len(v)]
        if use_field_heading:
            h_enc = hf

    w = np.append(wrap_pi(np.diff(h_enc)) / dt, 0.0)[:len(v)]   # 각속도(조향) [rad/s]

    return Motion(v_ms=v, a_ms2=a, h_rad=h_enc, h_exact=h_exact, w_rads=w,
                  w_field=w_field, p0=pos[0].copy(), stopped=stopped)


# ---------------------------------------------------------------- 물리량 -> 궤적
def vh_to_traj(p0, v_ms, h_rad, dt=DT):
    """(v,h) 적분 → (T,2). 적분 1번이라 오차가 쌓이지 않는다."""
    v = np.asarray(v_ms, float)
    h = np.asarray(h_rad, float)
    step = (v[:, None] * np.stack([np.cos(h), np.sin(h)], axis=1)) * dt
    p0 = np.asarray(p0, float)
    return np.vstack([p0, p0 + np.cumsum(step[:-1], axis=0)])


def ah_to_traj(p0, v0_ms, a_ms2, h_rad, dt=DT, clip_v=True):
    """(a,h) + 초기속도 적분 → (T,2). 적분 2번이라 v0 오차가 위치오차로 자란다."""
    a = np.asarray(a_ms2, float)
    v = np.empty(len(a))
    v[0] = float(v0_ms)
    for t in range(len(a) - 1):
        v[t + 1] = v[t] + a[t] * dt
    if clip_v:
        v = np.maximum(v, 0.0)                      # 후진은 없다고 본다
    return vh_to_traj(p0, v, h_rad, dt)


# ---------------------------------------------------------------- 신경망 입력 인코딩
# 스케일 상수: AV2 val 300개 과거 50 step 실측으로 정함 (나눈 뒤 std ≈ 0.6~1.0, |p99| <= 3)
# (val 300개 과거 50 step, smooth=5 기준. 나눈 뒤 std 0.5~1.0, |p99| <= 3.2)
DEFAULT_SCALES = {
    "v_kph": 20.0,      # 원값 std 17.3 km/h   → 0.87
    "v_ms": 5.0,        # 원값 std  4.81 m/s   → 0.96
    "a_kphs": 15.0,     # 원값 std  9.74 km/h/s → 0.65
    "a_ms2": 4.0,       # 원값 std  2.70 m/s^2  → 0.68
    "h_rad": 1.0,       # 각도 표현은 ±pi 불연속이 있으니 sincos 권장
    "h_deg": 60.0,
    "w_rads": 0.4,      # 각속도 (≈23 deg/s)
    "dv_kph": 3.0,      # 속도 증분 (첫 값만 v0 라 아웃라이어)
    "vxy_kph": 20.0,    # 속도 벡터 성분
    # --- 핸들링 관점 (val 500 실측 반영) ---
    # 실측 결론: 입력으로 쓸 만한 것은 heading 필드 기반 ω 하나뿐이다.
    # κ·δ 는 v 로 나눠서 저속에서 발산하고(|κ|max 746 1/m = 반경 1.3mm),
    # a_lat 는 v 를 곱해 저속 정보를 버린다. 아래 상수는 진단용으로만 남긴다.
    "w_field": 0.1,     # heading 필드 각속도 [rad/s] (실측 std 0.094) ← 권장
    "w_degs": 5.4,      # 이동방향 각속도 [deg/s] — 노이즈가 커서 비권장
    "kappa": 0.1,       # 곡률 [1/m], ±0.2 클립 전제
    "delta_deg": 17.2,  # 조향각 [deg] (=0.3 rad), ±40° 클립 전제
    "a_lat": 3.0,       # 횡가속도 [m/s²], ±8 클립 전제
    "dw_degs": 30.0,    # 방향 증분 (첫 값만 h0 라 아웃라이어)
}
# sin/cos 는 이미 [-1,1] 이라 나누지 않는다 (std 0.22 / 0.41 — 대부분 직진이라 cos≈1).


@dataclass
class ReprConfig:
    """어떤 표현으로 모델에 넣을지."""
    mode: str = "vah"            # "vh" | "ah" | "vah"
    unit: str = "kph"            # "kph" | "ms"
    heading: str = "sincos"      # "sincos"(권장) | "rad" | "deg"
    scale: bool = True
    scales: dict = field(default_factory=lambda: dict(DEFAULT_SCALES))

    def channels(self):
        if self.mode in ("vw", "vxvy", "aw0"):
            return 2
        if self.mode in ("vaw", "vad", "vawf", "vadf"):
            return 3
        if self.mode in ("vahd", "vahw", "vahdf"):
            return 5
        n_h = 2 if self.heading == "sincos" else 1
        return {"vh": 1, "ah": 1, "ah0": 1, "vah": 2}[self.mode] + n_h

    def names(self):
        if self.mode == "aw0":
            return ["dv[km/h] (페달, 첫값=v0)", "dw[deg/s] (핸들, 첫값=h0)"]
        if self.mode == "vaw":
            return ["v[km/h]", "a[km/h/s]", "w[deg/s] 요레이트"]
        if self.mode == "vad":
            return ["v[km/h]", "a[km/h/s]", "delta[deg] 조향각"]
        if self.mode == "vahd":
            return ["v[km/h]", "a[km/h/s]", "sin h", "cos h", "delta[deg] 조향각"]
        if self.mode == "vahw":
            return ["v[km/h]", "a[km/h/s]", "sin h", "cos h", "w_field[rad/s] 요레이트"]
        if self.mode == "vahdf":
            return ["v[km/h]", "a[km/h/s]", "sin h", "cos h", "delta_field[deg] 조향각"]
        if self.mode == "vawf":
            return ["v[km/h]", "a[km/h/s]", "w_field[rad/s] 요레이트"]
        if self.mode == "vadf":
            return ["v[km/h]", "a[km/h/s]", "delta_field[deg] 조향각"]
        if self.mode == "vw":
            return [f"v[{'km/h' if self.unit == 'kph' else 'm/s'}]", "w[rad/s]"]
        if self.mode == "vxvy":
            return ["vx[km/h]", "vy[km/h]"]
        out = []
        if self.mode in ("vh", "vah"):
            out.append(f"v[{'km/h' if self.unit == 'kph' else 'm/s'}]")
        if self.mode in ("ah", "vah"):
            out.append(f"a[{'km/h/s' if self.unit == 'kph' else 'm/s^2'}]")
        if self.mode == "ah0":
            out.append("dv[km/h] (a*dt, 첫값=v0)")
        out += ["sin h", "cos h"] if self.heading == "sincos" else [f"h[{self.heading}]"]
        return out


def encode(motion: Motion, cfg: ReprConfig = None) -> np.ndarray:
    """Motion → (T, C) 모델 입력 배열."""
    cfg = cfg or ReprConfig()

    # --- 2채널 모드 ---
    if cfg.mode == "vw":            # (속도, 각속도) = 유니사이클 제어입력
        v = motion.v_kph if cfg.unit == "kph" else motion.v_ms
        vk = "v_kph" if cfg.unit == "kph" else "v_ms"
        c0 = v / cfg.scales[vk] if cfg.scale else v
        c1 = motion.w_rads / cfg.scales["w_rads"] if cfg.scale else motion.w_rads
        return np.stack([c0, c1], axis=1).astype(np.float32)
    if cfg.mode == "vxvy":          # 속도 벡터 (속도+방향을 한 번에)
        vx = motion.v_kph * np.cos(motion.h_rad)
        vy = motion.v_kph * np.sin(motion.h_rad)
        sc = cfg.scales["vxy_kph"] if cfg.scale else 1.0
        return np.stack([vx / sc, vy / sc], axis=1).astype(np.float32)

    # --- 핸들링(조작) 관점 모드 ---
    if cfg.mode == "aw0":       # (페달, 핸들) 2채널 — 둘 다 누적 방식이라 무손실
        sc = cfg.scale
        c0 = motion.dv_kph / (cfg.scales["dv_kph"] if sc else 1.0)
        c1 = motion.dw_degs / (cfg.scales["dw_degs"] if sc else 1.0)
        return np.stack([c0, c1], axis=1).astype(np.float32)
    if cfg.mode in ("vahw", "vahdf", "vawf", "vadf"):
        # 핸들링 계열 — 조작 신호는 반드시 AV2 heading 필드에서 만든다.
        # 위치차분 기반은 std 81 deg/s 로 노이즈, 필드 기반은 5.4 deg/s (실측).
        sc = cfg.scale
        if motion.w_field is None:
            raise ValueError(f"mode='{cfg.mode}' 는 traj_to_motion(heading_field=...) 이 필요하다")
        if cfg.mode in ("vahw", "vawf"):
            hand = np.clip(motion.w_field, -1.0, 1.0)          # 물리 상한, 실측 0.004% 영향
            hand = hand / (cfg.scales["w_field"] if sc else 1.0)
        else:
            hand = np.clip(motion.delta_field_deg, -40.0, 40.0)  # 실측 1.6% 영향
            hand = hand / (cfg.scales["delta_deg"] if sc else 1.0)
        cols = [motion.v_kph / (cfg.scales["v_kph"] if sc else 1.0),
                motion.a_kphs / (cfg.scales["a_kphs"] if sc else 1.0)]
        if cfg.mode in ("vahw", "vahdf"):                      # 추가형: heading 도 같이
            cols += [np.sin(motion.h_rad), np.cos(motion.h_rad)]
        cols.append(hand)
        return np.stack(cols, axis=1).astype(np.float32)
    if cfg.mode in ("vaw", "vad", "vahd"):
        sc = cfg.scale
        cols = [motion.v_kph / (cfg.scales["v_kph"] if sc else 1.0),
                motion.a_kphs / (cfg.scales["a_kphs"] if sc else 1.0)]
        if cfg.mode == "vahd":
            cols += [np.sin(motion.h_rad), np.cos(motion.h_rad)]
        if cfg.mode == "vaw":
            cols.append(motion.w_degs / (cfg.scales["w_degs"] if sc else 1.0))
        else:
            cols.append(motion.delta_deg / (cfg.scales["delta_deg"] if sc else 1.0))
        return np.stack(cols, axis=1).astype(np.float32)

    cols = []
    if cfg.mode == "ah0":           # 속도 증분: 누적합하면 v (첫 값이 v0)
        dv = motion.dv_kph
        cols.append(dv / cfg.scales["dv_kph"] if cfg.scale else dv)
    if cfg.mode in ("vh", "vah"):
        key = "v_kph" if cfg.unit == "kph" else "v_ms"
        v = motion.v_kph if cfg.unit == "kph" else motion.v_ms
        cols.append(v / cfg.scales[key] if cfg.scale else v)
    if cfg.mode in ("ah", "vah"):
        key = "a_kphs" if cfg.unit == "kph" else "a_ms2"
        a = motion.a_kphs if cfg.unit == "kph" else motion.a_ms2
        cols.append(a / cfg.scales[key] if cfg.scale else a)
    if cfg.heading == "sincos":
        cols += [np.sin(motion.h_rad), np.cos(motion.h_rad)]
    elif cfg.heading == "rad":
        cols.append(motion.h_rad / cfg.scales["h_rad"] if cfg.scale else motion.h_rad)
    else:
        cols.append(motion.h_deg / cfg.scales["h_deg"] if cfg.scale else motion.h_deg)
    return np.stack(cols, axis=1).astype(np.float32)


def decode(feat, cfg: ReprConfig = None, p0=None, v0_ms=None, dt=DT, h_anchor=None):
    """(T,C) → 궤적 (T,2). encode 의 역변환."""
    cfg = cfg or ReprConfig()
    feat = np.asarray(feat, float)
    p0z = np.zeros(2) if p0 is None else p0

    # --- 2채널 모드 ---
    if cfg.mode == "vw":
        vk = "v_kph" if cfg.unit == "kph" else "v_ms"
        v = feat[:, 0] * (cfg.scales[vk] if cfg.scale else 1.0)
        if cfg.unit == "kph":
            v = to_ms(v)
        w = feat[:, 1] * (cfg.scales["w_rads"] if cfg.scale else 1.0)
        # 방향은 각속도 적분 + 앵커. 정규화 좌표에서 t=49 의 방향은 0 이다.
        idx, val = (len(v) - 1, 0.0) if h_anchor is None else h_anchor
        h = np.empty(len(v)); h[idx] = val
        for t in range(idx, len(v) - 1):
            h[t + 1] = h[t] + w[t] * dt
        for t in range(idx, 0, -1):
            h[t - 1] = h[t] - w[t - 1] * dt
        return vh_to_traj(p0z, v, h, dt)
    if cfg.mode == "vxvy":
        sc = cfg.scales["vxy_kph"] if cfg.scale else 1.0
        vxy = to_ms(feat[:, :2] * sc)
        return np.vstack([p0z, p0z + np.cumsum(vxy[:-1] * dt, axis=0)])

    # --- 핸들링 관점 모드 ---
    if cfg.mode == "aw0":
        sc = cfg.scale
        dv = feat[:, 0] * (cfg.scales["dv_kph"] if sc else 1.0)
        dw = feat[:, 1] * (cfg.scales["dw_degs"] if sc else 1.0)
        v = to_ms(np.cumsum(dv))                     # 누적합 = 속력
        h = np.radians(np.cumsum(dw) * dt)           # 누적합 = 방향 (첫 값이 h0)
        return vh_to_traj(p0z, v, h, dt)
    if cfg.mode in ("vaw", "vad", "vahd", "vahw", "vahdf", "vawf", "vadf"):
        sc = cfg.scale
        v = to_ms(feat[:, 0] * (cfg.scales["v_kph"] if sc else 1.0))
        if cfg.mode in ("vahd", "vahw", "vahdf"):   # heading 이 직접 있으면 무손실                       # heading 이 직접 있으면 무손실
            h = np.arctan2(feat[:, 2], feat[:, 3])
            return vh_to_traj(p0z, v, h, dt)
        if cfg.mode == "vaw":
            w = np.radians(feat[:, 2] * (cfg.scales["w_degs"] if sc else 1.0))
        else:                                        # 조향각 → 요레이트: w = v·tan(δ)/L
            d = np.radians(feat[:, 2] * (cfg.scales["delta_deg"] if sc else 1.0))
            w = np.maximum(v, V_FLOOR) * np.tan(d) / WHEELBASE
        idx, val = (len(v) - 1, 0.0) if h_anchor is None else h_anchor
        h = np.empty(len(v)); h[idx] = val
        for t in range(idx, len(v) - 1):
            h[t + 1] = h[t] + w[t] * dt
        for t in range(idx, 0, -1):
            h[t - 1] = h[t] - w[t - 1] * dt
        return vh_to_traj(p0z, v, h, dt)

    i, v, a = 0, None, None
    if cfg.mode == "ah0":
        dv = feat[:, 0] * (cfg.scales["dv_kph"] if cfg.scale else 1.0)
        v = to_ms(np.cumsum(dv))          # 누적합 = 속도
        i = 1
    if cfg.mode in ("vh", "vah"):
        key = "v_kph" if cfg.unit == "kph" else "v_ms"
        v = feat[:, i] * (cfg.scales[key] if cfg.scale else 1.0)
        if cfg.unit == "kph":
            v = to_ms(v)
        i += 1
    if cfg.mode in ("ah", "vah"):
        key = "a_kphs" if cfg.unit == "kph" else "a_ms2"
        a = feat[:, i] * (cfg.scales[key] if cfg.scale else 1.0)
        if cfg.unit == "kph":
            a = to_ms(a)          # km/h/s → m/s^2 도 같은 3.6 배율
        i += 1
    if cfg.heading == "sincos":
        h = np.arctan2(feat[:, i], feat[:, i + 1])
    elif cfg.heading == "rad":
        h = feat[:, i] * (cfg.scales["h_rad"] if cfg.scale else 1.0)
    else:
        h = np.radians(feat[:, i] * (cfg.scales["h_deg"] if cfg.scale else 1.0))

    if v is not None:
        return vh_to_traj(p0z, v, h, dt)            # 속도가 있으면 그걸 쓴다 (무손실)
    if v0_ms is None:
        raise ValueError("mode='ah' 는 v0_ms 가 필요하다 — (a,h)만으로는 위치를 복원할 수 없다")
    return ah_to_traj(p0z, v0_ms, a, h, dt)


# ---------------------------------------------------------------- 검증 도구
def roundtrip_error(pos, cfg: ReprConfig = None, dt=DT, v0=None, h_anchor=None, **motion_kw):
    """궤적 → 표현 → 궤적 왕복 오차 (ADE, FDE) [m]."""
    cfg = cfg or ReprConfig()
    m = traj_to_motion(pos, dt, **motion_kw)
    f = encode(m, cfg)
    kw = {} if cfg.mode != "ah" else {"v0_ms": m.v_ms[0] if v0 is None else v0}
    rec = decode(f, cfg, p0=m.p0, dt=dt, h_anchor=h_anchor, **kw)
    pos = np.asarray(pos, float)[:len(rec)]
    e = np.linalg.norm(rec - pos, axis=1)
    return float(e.mean()), float(e[-1])


def describe(motion: Motion) -> str:
    m = motion
    return (f"v {m.v_kph.min():5.1f}~{m.v_kph.max():5.1f} km/h | "
            f"a {m.a_kphs.min():+6.1f}~{m.a_kphs.max():+6.1f} km/h/s | "
            f"h {m.h_deg.min():+7.1f}~{m.h_deg.max():+7.1f} deg | "
            f"정지 {m.stopped.mean()*100:4.1f}%")


if __name__ == "__main__":
    t = np.arange(50) * DT
    pos = np.stack([8.0 * t + 0.5 * 1.2 * t**2, 1.5 * np.sin(0.6 * t)], axis=1)
    print("합성 궤적(등가속 + 사행) 50 step 왕복 오차 [m]")
    for mode in ("vh", "ah", "vah"):
        for hd in ("sincos", "rad"):
            cfg = ReprConfig(mode=mode, heading=hd)
            ade, fde = roundtrip_error(pos, cfg)
            print(f"  mode={mode:3s} heading={hd:6s} C={cfg.channels()} "
                  f"{str(cfg.names()):42s} ADE={ade:.2e} FDE={fde:.2e}")
    m = traj_to_motion(pos)
    print("\n " + describe(m))
    print(f"  v0 = {m.v_kph[0]:.1f} km/h")
    ade0, fde0 = roundtrip_error(pos, ReprConfig(mode="ah"), v0=0.0)
    print(f"  mode=ah 에서 v0=0 으로 잘못 주면: ADE={ade0:.2f} m  FDE={fde0:.2f} m")
