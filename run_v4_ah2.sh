#!/bin/bash
# v4 (a, h) 2채널 입력 학습 — 사용자 결정(2026-09-14): L0·L4nw 의 (a, h) 판 2개만 돌린다.
#   입력  --input ah2       a = 속도 증분(첫 값 v0), h = 위치차분 진행방향(= k + θ). 둘 다 관측 구간만.
#   θ₀    --th0 guard       wrap(h0 − k(s0)). |값| > 90° 면 −k(s0) 로 되돌린다.
#                           가드가 없으면 θ 가 첫 스텝부터 ±90° 에 걸려 궤적이 옆으로 미끄러진다.
#   폴백  --fallback straight1   단계 3·4 와 같게.
#
# 비교 대상: runs/v4_l0b_s0.json, runs/v4_l4_nw_s0.json (5채널 입력, θ₀ current, route_point 이전 적분기).
#   입력만 다른 게 아니라 θ₀ 규칙과 적분기 기하도 달라서, 입력 효과만 따로 떼어 볼 수는 없다.
#   (5채널 재학습 기준선은 사용자 결정으로 돌리지 않는다.)
#
# 에폭 시간은 runs/*.json 의 history[].sec 에 남는다. 이 스크립트는 workers 20 · 동시 2판으로 고정한다 —
# 다른 실행과 벽시계를 비교하려면 이 조건을 맞춰야 한다.
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
# --smooth 0 — 2026-09-16 부터 L0 기본에 흔들림 벌점(1.0)이 들어가므로, 벌점 없이 학습한 이 판들을 재현하려면 끈다.
C="--level l0 --input ah2 --th0 guard --fallback straight1 --theta 0 --rules 1 --seed 0 \
   --limit 50000 --val-limit 2000 --epochs 15 --lr 5e-4 --batch 32 --workers 20 --outdir runs --smooth 0"
$PY src/train_v4.py $C                                  --tag v4_l0_ah2_s0   > runs/v4_l0_ah2_s0.log   2>&1 &
$PY src/train_v4.py $C --offlane 1.0 --off-nonwinner 1  --tag v4_l4nw_ah2_s0 > runs/v4_l4nw_ah2_s0.log 2>&1 &
wait
echo "=== DONE ==="; tail -n 1 runs/v4_l0_ah2_s0.log runs/v4_l4nw_ah2_s0.log
