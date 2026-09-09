#!/bin/bash
# 단계 3 재실행 + 단계 4. L0 에 Δθ 레이트 제한(8°/step)을 넣은 뒤 세 판을 함께 돌린다.
#   l0b     : 레이트 제한만 (단계 3 최종). 제한 없던 판은 궤적의 71.4% 가 실현 불가능이었다.
#   l4_all  : + 차로 이탈 hinge 를 모든 모드에 (v4 문서 원안)
#   l4_nw   : + 이탈 hinge 를 버려진 모드에만 (측정 근거: 기준 경로를 하나 고르면
#             정답의 14.05% step 이 벌점을 받는다. 승자는 거리 손실이 이미 감독한다)
# workers 16 x 3 = 48 (코어 56) — 세 판이 CPU 안에 들어가게 낮춤.
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
C="--level l0 --theta 0 --rules 1 --seed 0 --limit 50000 --val-limit 2000 \
   --epochs 15 --lr 5e-4 --batch 32 --workers 16 --outdir runs"
$PY src/train_v4.py $C                             --tag v4_l0b_s0    > runs/v4_l0b_s0.log    2>&1 &
$PY src/train_v4.py $C --offlane 1.0 --off-nonwinner 0 --tag v4_l4_all_s0 > runs/v4_l4_all_s0.log 2>&1 &
$PY src/train_v4.py $C --offlane 1.0 --off-nonwinner 1 --tag v4_l4_nw_s0  > runs/v4_l4_nw_s0.log  2>&1 &
wait
echo "=== DONE ==="; tail -n 1 runs/v4_l0b_s0.log runs/v4_l4_all_s0.log runs/v4_l4_nw_s0.log
