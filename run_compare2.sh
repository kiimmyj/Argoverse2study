#!/bin/bash
# 2차 표현 비교 — v3 원래 설정(50k / 15 epoch / lr 5e-4 / batch 32)
#   vahw5 s0 : 실측 권장안 (v, a, sin h, cos h, ω_field)  ← 핵심
#   aw0_2 s0 : (페달, 핸들) 둘 다 누적 — 누적합 가설 재검증
#   vah4  s1 : 시드 반복 (기존 s0 = 1.265)
#   raw5  s1 : 시드 반복 (기존 s0 = 1.318)
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
mkdir -p runs
run() { $PY src/train_motion.py --repr "$1" --seed "$2" --limit 50000 --val-limit 2000 \
        --epochs 15 --lr 5e-4 --batch 32 --workers 12 --outdir runs > "runs/$1_s$2.log" 2>&1 & }
run vahw5 0
run aw0_2 0
run vah4  1
run raw5  1
wait
echo "=== ALL DONE ==="
for f in runs/vahw5_s0.log runs/aw0_2_s0.log runs/vah4_s1.log runs/raw5_s1.log; do tail -1 "$f"; done
