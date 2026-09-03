#!/bin/bash
# v3 + 차로 이동 규칙 피처 학습 (v3 원래 설정: 50k / 15 epoch / lr 5e-4 / batch 32 / seed 0)
#   lane_rules  : 차선 좌표 20 + 규칙 피처 10 = lane_encoder 입력 30
#   lane_norule : 같은 Dataset, 규칙 없음 = 입력 20  (엄밀한 대조군)
# 참고 기준선: dataset_map 기반 v3 = minADE6 1.318 / minFDE6 2.921
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
mkdir -p runs
$PY src/train_lane.py --rules 1 --seed 0 --limit 50000 --val-limit 2000 \
    --epochs 15 --lr 5e-4 --batch 32 --workers 24 --outdir runs > runs/lane_rules_s0.log 2>&1 &
$PY src/train_lane.py --rules 0 --seed 0 --limit 50000 --val-limit 2000 \
    --epochs 15 --lr 5e-4 --batch 32 --workers 24 --outdir runs > runs/lane_norule_s0.log 2>&1 &
wait
echo "=== ALL DONE ==="
tail -1 runs/lane_rules_s0.log; tail -1 runs/lane_norule_s0.log
