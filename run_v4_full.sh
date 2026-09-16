#!/bin/bash
# v4 (a, h) 전체 데이터 학습과 흔들림 벌점 (2026-09-16).
#   데이터  train 199,908 / val 24,988 전부. prepare_v4 로 캐시를 먼저 굽고 --cache 로 읽는다.
#           --limit 0 은 '전체'가 아니라 0개로 처리되므로 개수를 적는다.
#   설정    run_v4_ah2.sh 의 (a, h) 2판과 같다(θ₀ guard, straight1, 15에폭, lr 5e-4, batch 32, seed 0).
#   벌점    --smooth w — 액션 (a, dθ) 의 스텝간 변화 제곱(train_v4.jitter). 0 이면 기존 손실 그대로.
#
# 다섯 판을 **차례로** 돌린다. 에폭 시간은 단독 실행·workers 고정으로 재야 비교가 되기 때문이다.
# 에폭 시간은 runs/*.json 의 history[].sec (train + val 평가) 에 남는다. 한 판에 약 30분.
# 끝나면 python src/compare_v4_full.py 로 같은 val 에서 다시 채점한다 (runs/v4_full_compare.json).
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python

# 전처리 캐시 (약 17분, 32 workers — 뒤쪽 train 은 HDD 라 느려진다). 이미 있으면 재사용한다.
$PY -u src/prepare_v4.py --level l0 --input ah2 --th0 guard --fallback straight1 --theta 0 \
    --limit 199908 --val-limit 24988 --workers 32 | tee /data/argoverse2/cache/v4/prepare_l0_ah2_full.log
grep -q "모든 점검 통과" /data/argoverse2/cache/v4/prepare_l0_ah2_full.log || exit 1

C="--level l0 --input ah2 --th0 guard --fallback straight1 --theta 0 --rules 1 --seed 0 \
   --limit 199908 --val-limit 24988 --epochs 15 --lr 5e-4 --batch 32 --workers 8 --cache --outdir runs"
L4="--offlane 1.0 --off-nonwinner 1"
$PY -u src/train_v4.py $C                   --tag v4_l0_ah2_full_s0          > runs/v4_l0_ah2_full_s0.log          2>&1
$PY -u src/train_v4.py $C $L4               --tag v4_l4nw_ah2_full_s0        > runs/v4_l4nw_ah2_full_s0.log        2>&1
$PY -u src/train_v4.py $C     --smooth 1.0  --tag v4_l0_ah2_full_sm1_s0      > runs/v4_l0_ah2_full_sm1_s0.log      2>&1
$PY -u src/train_v4.py $C $L4 --smooth 1.0  --tag v4_l4nw_ah2_full_sm1_s0    > runs/v4_l4nw_ah2_full_sm1_s0.log    2>&1
$PY -u src/train_v4.py $C     --smooth 0.1  --tag v4_l0_ah2_full_sm0.1_s0    > runs/v4_l0_ah2_full_sm0.1_s0.log    2>&1
echo "=== DONE ==="; grep -h BEST runs/v4_*_full*_s0.log
