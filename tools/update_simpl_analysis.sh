#!/bin/bash
# SIMPL 런 분석을 다시 돌려 docs/r1_result_analysis.md 를 갱신하고 커밋·푸시한다.
#
# 왜 자동인가 — 갱신 시점이 사람이 깨어 있는 시각과 안 맞는다(R1 완주 11-04 새벽,
# R0-s43 완주 11-06). 로그에서 기계적으로 읽는 부분만 자동으로 올려 두고,
# **해석 문장은 사람이 나중에 더한다.** 그래서 노션에는 밀지 않는다 — 노션은 검토 후.
#
# 갱신되는 것은 문서의 <!-- NUMBERS --> 구간과 그림 둘뿐이다. 산문은 건드리지 않는다.
#
#   bash tools/update_simpl_analysis.sh "에폭 37"     # 라벨은 커밋 메시지에 들어간다
set -u
REPO=/home/user/Argoverse2study
PY=/home/user/miniforge3/envs/av2/bin/python
LABEL="${1:-자동 갱신}"
cd "$REPO" || exit 1

"$PY" src/analyze_simpl_runs.py || { echo "분석 실패"; exit 1; }

# 내 파일만 담는다 — 다른 세션의 작업 중인 변경을 끌어오면 안 된다(git add -A 금지).
git add docs/r1_result_analysis.md viz/r1_analysis/ 2>/dev/null
if git diff --cached --quiet; then
  echo "[$(date '+%F %T')] 바뀐 것 없음 — 커밋 생략"
  exit 0
fi

git -c user.name=kiimmyj commit -q -F - <<EOF
R1 분석 수치 갱신 — ${LABEL}

src/analyze_simpl_runs.py 재실행 결과다. 로그에서 읽은 수치와 그림만 바뀌었고
해석 산문은 그대로다 — 해석은 사람이 검토해서 더한다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
git push -q origin lane-rules && echo "[$(date '+%F %T')] 갱신·푸시 완료 — ${LABEL}"
