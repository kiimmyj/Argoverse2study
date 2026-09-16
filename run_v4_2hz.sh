#!/bin/bash
# v4 (a, h) 2 Hz 입력 학습 (2026-09-17) — 입력만 평활 + 10 Hz → 2 Hz 로 바꾼 L0·L4nw.
#   입력    --input ah2_2hz  관측 위치를 Savitzky–Golay(5점·2차) 로 평활한 뒤 0.5초 간격 10스텝 (a, h).
#           창 가장자리 램프(인덱스 0~3)는 평활에서 뺀다. 정답·후보 경로·적분기 h0 는 10 Hz 판과 같다.
#   벌점    --smooth 1.0 — L0 기본값(흔들림 벌점은 L0 의 일부). 기본값이 바뀌어도 같게 돌도록 적는다.
#   비교    run_v4_full.sh 의 v4_l0_ah2_full_sm1_s0 / v4_l4nw_ah2_full_sm1_s0 와 입력만 다르다.
#
# 두 판을 **차례로** 돌린다(에폭 시간은 단독 실행·workers 고정으로 재야 비교가 된다). 한 판에 약 30분.
# 끝나면 python src/compare_v4_full.py 로 같은 val 에서 다시 채점한다.
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python

# 전처리 캐시 (40 workers 약 8분 — 페이지 캐시가 차가우면 더 걸린다). 이미 있으면 재사용한다.
$PY -u src/prepare_v4.py --level l0 --input ah2_2hz --th0 guard --fallback straight1 --theta 0 \
    --limit 199908 --val-limit 24988 --workers 40 | tee /data/argoverse2/cache/v4/prepare_l0_ah2_2hz_full.log
grep -q "모든 점검 통과" /data/argoverse2/cache/v4/prepare_l0_ah2_2hz_full.log || exit 1

C="--level l0 --input ah2_2hz --th0 guard --fallback straight1 --theta 0 --rules 1 --seed 0 \
   --limit 199908 --val-limit 24988 --epochs 15 --lr 5e-4 --batch 32 --workers 8 --cache --outdir runs --smooth 1.0"
L4="--offlane 1.0 --off-nonwinner 1"
$PY -u src/train_v4.py $C      --tag v4_l0_ah2_2hz_full_sm1_s0     > runs/v4_l0_ah2_2hz_full_sm1_s0.log     2>&1
$PY -u src/train_v4.py $C $L4  --tag v4_l4nw_ah2_2hz_full_sm1_s0   > runs/v4_l4nw_ah2_2hz_full_sm1_s0.log   2>&1
echo "=== DONE ==="; grep -h BEST runs/v4_*_2hz_full_sm1_s0.log
