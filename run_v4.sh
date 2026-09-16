#!/bin/bash
# v4 레벨별 누적 학습. 입력은 기준선과 같은 5채널로 고정 — 지표 차이는 레벨 때문이다.
#   기준선 : runs/lane_rules_s0.json  minADE6 1.292 (model_lane.py, 익명 슬롯 K=6)
#   l2 : V4Net 의 L2 경로가 기준선을 재현하는지 (하네스 동등성 확인)
#   l3 : 모드 = 지도에서 열거된 후보 경로
#   l0 : l3 + 출력을 액션 (a, theta) 로, Frenet 적분기로 복원
#   l4 : l0 + 차로 이탈 단측 hinge
cd /home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
# --smooth 0 — 2026-09-16 부터 L0 기본에 흔들림 벌점(1.0)이 들어가므로, 벌점 없이 학습한 이 판들을 재현하려면 끈다.
COMMON="--theta 0 --rules 1 --seed 0 --limit 50000 --val-limit 2000 --epochs 15 \
        --lr 5e-4 --batch 32 --workers 24 --outdir runs --smooth 0"
for LV in "$@"; do
  case $LV in
    l2|l3|l0) $PY src/train_v4.py --level $LV $COMMON --tag v4_${LV}_s0 \
                  > runs/v4_${LV}_s0.log 2>&1 & ;;
    l4)       $PY src/train_v4.py --level l0 --offlane 1.0 $COMMON --tag v4_l4_s0 \
                  > runs/v4_l4_s0.log 2>&1 & ;;
  esac
done
wait
echo "=== DONE: $* ==="
for LV in "$@"; do tail -n 1 runs/v4_${LV}_s0.log; done
