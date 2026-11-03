#!/bin/bash
# R0-s43 — R0 와 모든 것이 같고 시드만 42 → 43. (사용자 승인 2026-11-03)
#
# 왜 도는가: R1 − R0 안에는 교란이 셋 섞여 있다.
#   ① 입력 표현 (14ch → 11ch)        ← 우리가 재려는 것
#   ② 초기화 추첨                      1층 원소 수 5,376 → 4,224 로 RNG 스트림이 밀린다
#   ③ 데이터 순서                      train.py:70 DataLoader(shuffle=True) 에 generator= 가 없어
#                                      RandomSampler 가 밀린 전역 RNG 에서 base_seed 를 뽑는다
# 이 런은 ①을 빼고 ②③만 바꾼다. 따라서 |R0s43 − R0s42| 가 바로 ②③의 크기이고,
# 그것이 |R1 − R0| 를 판정할 자다. 이 값이 없으면 R1 이 이겨도 초기화 운과 못 가른다.
#
# ⚠️ train.py 는 고치지 않는다. 이유는 "교란 구조가 R1 과 달라져서"가 아니다 —
#    ③을 고쳐도 시드가 42 와 43 이면 데이터 순서는 어차피 갈라지므로 ②③ 성분은 그대로다.
#    진짜 이유는 **R0-s42 와 R0-s43 이 같은 프로그램이어야 한다**는 것이다. 코드를 바꾸면
#    두 런의 차이에 코드 변경이 섞여 '시드 하나만 다른 쌍' 이 아니게 된다. (③ 수정은 R2 부터.)
set -u
cd /home/user/av2_simpl
PY=/home/user/miniforge3/envs/av2/bin/python
unset SIMPL_ACTOR_FEAT          # 원본 14채널 — ah2 가 새어들면 R0 가 아니다
echo "=== R0-s43 시작 $(date '+%F %T') · git $(git rev-parse --short HEAD) · feat=${SIMPL_ACTOR_FEAT:-orig} ==="
CUDA_VISIBLE_DEVICES=0 $PY -u train.py \
  --features_dir /data/argoverse2/simpl_features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --val_interval 2 \
  --train_epoches 50 \
  --seed 43 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --no_pbar \
  --adv_cfg_path config.simpl_av2_cfg
rc=$?   # ⚠️ 먼저 잡는다 — 같은 줄에서 $(date) 를 쓰면 $? 가 그걸로 덮인다
echo "=== R0-s43 끝 $(date '+%F %T') · 종료코드 $rc ==="
