#!/bin/bash
# 표현 4종을 같은 조건으로 병렬 학습 (v3 원래 설정: 50k / 15 epoch / lr 5e-4 / batch 32)
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
mkdir -p runs
for R in raw5 vah3 ah0_2 vah4; do
  $PY src/train_motion.py --repr $R --limit 50000 --val-limit 2000 \
      --epochs 15 --lr 5e-4 --batch 32 --workers 12 --seed 0 --outdir runs \
      > runs/$R.log 2>&1 &
done
wait
echo "=== ALL DONE ==="
for R in raw5 vah3 ah0_2 vah4; do tail -1 runs/$R.log; done
