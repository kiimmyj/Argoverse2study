"""지도 융합 레벨 L0~L5 가 파이프라인 어디에 들어가는지 그린다."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False

BG="#14151a"; BOX="#242730"; EDGE="#3a3f4b"; TXT="#e6e8ec"; DIM="#9aa0aa"
OK="#4ec98a"; NO="#e0554a"; PRE="#e8c65a"; FLOW="#7b8494"

fig, ax = plt.subplots(figsize=(17, 9.6), facecolor=BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off"); ax.set_facecolor(BG)

def rbox(x, y, w, h, fc, ec, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw, zorder=2))

def arrow(x1, y1, x2, y2, color, lw=2.0, ls="-", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                                 lw=lw, linestyle=ls, mutation_scale=15, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))

# ---------- 오프라인 전처리 ----------
rbox(3, 50.5, 94, 6.2, "#2a2416", PRE)
ax.text(50, 55.0, "오프라인 전처리", color=PRE, fontsize=12.5, ha="center", fontweight="bold")
ax.text(50, 52.4, "지도 폴리라인    ·    후보 차선 경로 K개    ·    경로별 곡률 테이블 kappa_lane(s)    ·    off-road 거리장(SDF)",
        color="#c9b06a", fontsize=9.2, ha="center")

# ---------- 레벨 (한 줄로 정렬, 설명 3줄을 박스 안에) ----------
LY, LH = 33.0, 11.0
levels=[(3.0,17.5,"L1","탈락","과거 50 step 에 차선 상대좌표를 덧붙임","L2 입력에서 계산 가능 → 정보 기여 0",False,0),
        (22.2,17.5,"L2","채택","지도를 토큰으로 만들어 cross-attention","전제조건이 없는 유일한 레벨 → 1순위",True,1),
        (41.4,17.5,"L3","채택","후보 차선 경로 K개가 예측 모드가 됨","모드 평균 세탁을 막는 유일한 장치",True,2),
        (60.6,19.0,"L0","채택","kappa = kappa_lane(s) + delta","L3 가 경로를 골라줘야 성립",True,3),
        (81.3,15.7,"L4","채택","복원된 위치에 off-road 페널티","WTA 가 버린 모드를 감독",True,4)]
for x,w,name,verdict,desc,why,ok,si in levels:
    col = OK if ok else NO
    rbox(x, LY, w, LH, "#18271f" if ok else "#291a1a", col)
    ax.text(x+w/2, LY+LH*0.78, f"{name}   {verdict}", color=col, fontsize=12.5,
            ha="center", va="center", fontweight="bold")
    ax.text(x+w/2, LY+LH*0.48, desc, color=TXT, fontsize=8.8, ha="center", va="center")
    ax.text(x+w/2, LY+LH*0.19, why, color=DIM, fontsize=8.2, ha="center", va="center")

# ---------- 5개 단계 ----------
X=[3.0, 22.2, 41.4, 60.6, 81.3]; W=[17.5, 17.5, 17.5, 19.0, 15.7]; SY, SH = 13.0, 8.6
stages=[("① 인코더 입력","과거 50 step (a, kappa)"),
        ("② 인코더","self / cross attention"),
        ("③ 디코더","모드 K개 → (a, delta) × 60"),
        ("④ 적분기 f","cumsum 2회 · 학습 파라미터 없음"),
        ("⑤ 손실","복원된 위치 p 에 건다")]
for x,w,(t,s) in zip(X,W,stages):
    rbox(x, SY, w, SH, BOX, EDGE)
    ax.text(x+w/2, SY+SH*0.62, t, color=TXT, fontsize=11.5, ha="center", va="center", fontweight="bold")
    ax.text(x+w/2, SY+SH*0.27, s, color=DIM, fontsize=8.5, ha="center", va="center")
for i in range(4):
    arrow(X[i]+W[i]+0.5, SY+SH/2, X[i+1]-0.9, SY+SH/2, FLOW, lw=2.4)

# 레벨 -> 단계
for (x,w,_,_,_,_,ok,si) in levels:
    col = OK if ok else NO
    arrow(x+w/2, LY-0.6, X[si]+W[si]/2, SY+SH+0.7, col, lw=2.2, ls="-" if ok else (0,(4,3)))
# 전처리 -> 레벨
for (x,w,_,_,_,_,ok,_) in levels:
    arrow(x+w/2, 50.3, x+w/2, LY+LH+0.7, PRE, lw=1.1, ls=(0,(2,3)), z=1)

# ---------- L5 (탈락) ----------
rbox(58.0, 1.5, 24.5, 6.4, "#291a1a", NO)
ax.text(70.2, 6.0, "L5   탈락", color=NO, fontsize=12, ha="center", fontweight="bold")
ax.text(70.2, 3.4, "rollout 중 예측 위치로 지도 재조회", color="#d9998f", fontsize=8.6, ha="center")
# 화살표는 오른쪽으로, 설명 글자는 왼쪽으로 갈라 겹치지 않게 한다
arrow(78.5, 8.1, X[3]+W[3]-3.0, SY-0.4, NO, lw=2.0, ls=(0,(4,3)))
ax.text(X[3]+4.5, 10.2, "위치→방향→곡률→조회 가 순환\n→ DAG 깨짐, 60 step 순차 루프",
        color=NO, fontsize=8.4, ha="left", va="center", linespacing=1.5)

ax.text(50, 58.7, "지도 융합 레벨 — 파이프라인 어디에 들어가는가",
        color=TXT, fontsize=15.5, ha="center", fontweight="bold")
ax.text(26, 6.0, "손실을 '복원된 위치'에 걸기 때문에\ngradient 가 적분기를 거슬러 액션까지 흘러간다",
        color=DIM, fontsize=9.0, ha="center", style="italic", linespacing=1.6)
fig.savefig("map_levels.png", dpi=115, facecolor=BG, bbox_inches="tight")
print("saved -> map_levels.png")
