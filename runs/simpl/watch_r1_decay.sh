#!/bin/bash
# R1 의 다음 두 관문을 세션과 무관하게 기록한다.
#   ① LR 감쇠 직후(에폭 37 = 25번째 검증) → r1_decay.txt   ※ 여기서부터 판정이 가능해진다
#   ② 50에폭 완주(에폭 49 = 37번째 검증) → r1_final.txt
# 검증은 에폭 25까지 2에폭마다, 26부터 매 에폭이다(R0 로그로 확인).
# 파이프 없이 폴링한다 — tail|awk 는 mawk 버퍼링 때문에 조용히 멈춘다([[mawk-pipe-buffering]]).
LOG=/home/user/av2_simpl/r1.log
PY=/home/user/miniforge3/envs/av2/bin/python
CMP=/home/user/av2_simpl/tools/cmp_r0_r1.py
mark() {  # $1 = 필요한 검증 횟수, $2 = 출력 파일
  [ -f "$2" ] && return 0
  [ "$(grep -c '^\[Validation\]' "$LOG")" -ge "$1" ] || return 1
  $PY "$CMP" > "$2" 2>&1
  echo "[$(date '+%F %T')] 기록됨 → $2"
}
while true; do
  mark 25 /home/user/av2_simpl/r1_decay.txt
  if mark 37 /home/user/av2_simpl/r1_final.txt; then exit 0; fi
  if ! pgrep -f "train.py.*simpl_av2_cfg" > /dev/null; then
    echo "[$(date '+%F %T')] 학습 프로세스가 사라졌다 — 감시 중단"
    $PY "$CMP" > /home/user/av2_simpl/r1_final.txt 2>&1
    exit 1
  fi
  sleep 300
done
