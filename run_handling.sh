#!/bin/bash
# ① 안(v,a,h) vs 핸들링 관점 — 2x2 비교
#   추가형: heading 을 유지하고 조작 신호를 채널로 추가
#   대체형: heading 을 조작 신호로 갈아끼움
# 조작 신호는 전부 AV2 heading 필드에서 생성 (실측상 위치차분 대비 20배 깨끗)
# 기준선(seed 0, 이미 확보): raw5 1.318 / vah3 1.280 / vah4 1.265
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
mkdir -p runs
run() { $PY src/train_motion.py --repr "$1" --seed 0 --limit 50000 --val-limit 2000 \
        --epochs 15 --lr 5e-4 --batch 32 --workers 12 --outdir runs > "runs/$1_s0.log" 2>&1 & }
run vahw5    # 추가형 · 요레이트 ω   (v, a, sin h, cos h, ω)
run vahdf5   # 추가형 · 조향각 δ     (v, a, sin h, cos h, δ)
run vawf3    # 대체형 · 요레이트 ω   (v, a, ω)
run vadf3    # 대체형 · 조향각 δ     (v, a, δ)
wait
echo "=== ALL DONE ==="
for r in vahw5 vahdf5 vawf3 vadf3; do tail -1 "runs/${r}_s0.log"; done
