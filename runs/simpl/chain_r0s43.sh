#!/bin/bash
# R1 이 50에폭을 정상 완주하면 이어서 R0-s43 을 띄운다 (사용자 승인 2026-11-03).
# GPU 가 1장이라 동시 실행이 안 되므로 줄을 세운다. 완주 실패면 띄우지 않는다 —
# R1 이 깨졌는데 다음 런을 밀어 넣으면 원인 조사만 어려워진다.
#
# ⚠️ 종료코드로는 판정할 수 없다. run_r1.sh 의 마지막 줄이
#     echo "=== R1 끝 $(date ...) · 종료코드 $? ==="
# 인데 **$? 가 같은 줄의 $(date) 종료코드로 덮인다.** 게다가 tee 파이프라서
# $? 는 애초에 tee 의 것이다. 즉 학습이 죽어도 로그에는 "종료코드 0" 이 찍힌다.
#   $ bash -c 'false; echo "rc=$(date +%F) -> $?"'   →  rc=2026-11-03 -> 0
# 그래서 세 가지 실물로 판정한다:
#   ① saved_models/<런ID>_Simpl_epoch50.tar — 50에폭을 다 돌고 저장까지 끝나야 생긴다
#   ② 검증 37회 (에폭 25까지 2에폭마다, 26부터 매 에폭)
#   ③ 로그에 Traceback 없음
set -u
D=/home/user/av2_simpl
NOTE=$D/chain_r0s43.status
note() { echo "[$(date '+%F %T')] $*" | tee -a "$NOTE"; }

RUNID=$(grep -o '[0-9]\{8\}-[0-9]\{6\}' "$D/r1.log" | head -1)
note "대기 시작 — R1(train.py) 종료를 기다린다 · 런ID ${RUNID:-불명}"
while pgrep -f "train.py.*simpl_av2_cfg" > /dev/null; do sleep 120; done
note "R1 프로세스 종료 확인"

CKPT="$D/saved_models/${RUNID}_Simpl_epoch50.tar"
# ⚠️ `grep -c` 는 0건일 때 "0" 을 찍고 종료코드 1 을 낸다. `|| echo 0` 을 붙이면
#    "0\n0" 이 되어 [ "$TB" -ne 0 ] 가 "integer expression expected" 로 깨진다. `|| true` 를 쓴다.
NV=$(grep -c '^\[Validation\]' "$D/r1.log" 2>/dev/null || true); NV=${NV:-0}
TB=$(grep -c 'Traceback' "$D/r1.log" 2>/dev/null || true); TB=${TB:-0}
note "체크포인트 $([ -f "$CKPT" ] && echo 있음 || echo 없음) · 검증 ${NV}회 · Traceback ${TB}건  (기준: 있음 · 37 · 0)"

if [ ! -f "$CKPT" ] || [ "$NV" -ne 37 ] || [ "$TB" -ne 0 ]; then
  note "⚠️ R1 이 완주하지 않았다 — R0-s43 을 띄우지 않는다. 사람이 확인할 것."
  exit 1
fi
if pgrep -f "train.py" > /dev/null; then
  note "⚠️ 아직 train.py 가 있다 — 중단"; exit 1
fi

note "R0-s43 시작"
tmux new-session -d -s r0s43 "bash $D/run_r0s43.sh 2>&1 | tee $D/r0s43.log"
sleep 20
if pgrep -f "train.py" > /dev/null; then
  note "✅ R0-s43 기동 확인 (tmux r0s43 · 로그 r0s43.log)"
else
  note "⚠️ R0-s43 기동 실패 — r0s43.log 확인할 것"; exit 1
fi
