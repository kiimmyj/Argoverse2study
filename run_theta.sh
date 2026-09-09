#!/bin/bash
# v4 방향 표현 A/B — 기존 5채널에 (sin theta, cos theta, valid) 3채널을 **덧붙인다**.
#   기준선 : runs/lane_rules_s0.json  minADE6 1.292 / minFDE6 2.883 (5채널, 규칙 포함)
#   th_av2   : + theta,  h = AV2 heading            <- theta 채널만의 효과
#   th_build : + theta,  h = 위치차분 + 저속 AV2 보조 <- h 출처의 효과
# 하이퍼파라미터는 기준선과 동일 (50k / 15 epoch / lr 5e-4 / batch 32 / seed 0).
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
mkdir -p runs
for SRC in av2 build; do
  $PY src/train_lane.py --rules 1 --theta 1 --h-src $SRC --tag lane_th_${SRC}_s0 \
      --seed 0 --limit 50000 --val-limit 2000 --epochs 15 --lr 5e-4 --batch 32 \
      --workers 24 --outdir runs > runs/lane_th_${SRC}_s0.log 2>&1 &
done
wait
echo "=== ALL DONE ==="
tail -1 runs/lane_th_av2_s0.log; tail -1 runs/lane_th_build_s0.log
