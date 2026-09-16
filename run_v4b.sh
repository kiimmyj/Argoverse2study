#!/bin/bash
# L3 재실험(경로 중복 제거 + 종방향 하위모드) 과 L0 을 동시에.
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
# --smooth 0 — 2026-09-16 부터 L0 기본에 흔들림 벌점(1.0)이 들어가므로, 벌점 없이 학습한 이 판들을 재현하려면 끈다.
C="--theta 0 --rules 1 --seed 0 --limit 50000 --val-limit 2000 --epochs 15 --lr 5e-4 \
   --batch 32 --workers 24 --outdir runs --smooth 0"
$PY src/train_v4.py --level l3 $C --tag v4_l3b_s0 > runs/v4_l3b_s0.log 2>&1 &
$PY src/train_v4.py --level l0 $C --tag v4_l0_s0  > runs/v4_l0_s0.log  2>&1 &
wait
echo "=== DONE ==="; tail -n 1 runs/v4_l3b_s0.log runs/v4_l0_s0.log
