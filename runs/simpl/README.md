# SIMPL 실행 로그

`~/av2_simpl` 은 upstream 클론이라 우리 저장소가 아니다. 거기서 나온 로그만 여기 남긴다.
체크포인트(32 MB)는 git 에 넣지 않고 워크스테이션 `~/av2_simpl/saved_models/` 에 둔다.

| 로그 | 무엇 | 결과 |
| --- | --- | --- |
| `r0.log` | **R0** — 원본 SIMPL 14채널. 2026-10-29 10:05 ~ 10-31 08:01 (45.9 h, 50에폭) | val **minADE6 0.795** · minFDE6 1.488 · MR6 0.206 (에폭 49) |
| `r1.log` | **R1** — (a, h) 2채널 11입력. 2026-11-02 **10:28** 시작 (PID 11250 실측) | 에폭 5 기준 val minADE6 **1.028** (R0 동일 에폭 1.032) · 진행 중 |

R0 과 R1 은 **입력 말고 전부 같다** — 같은 커밋, 같은 시드(42), 같은 배치(16), 같은 스케줄,
같은 캐시(`/data/argoverse2/simpl_features/`). 차이는 `SIMPL_ACTOR_FEAT` 환경변수 하나다.
R1 의 코드 변경은 `patches/simpl_r1_ah2.patch` 에 보존돼 있다 — 새 모듈 `simpl/ah_feat.py`,
`av2_dataset.py` 의 분기, `config` 의 `in_actor`, 검증 도구 `tools/verify_ah2.py`,
보조 비율 측정 `tools/measure_fill_share.py`.

**입력이 바뀐 독립 증거** — `actor_net` 파라미터 1,246,144 → 1,245,760, 차이 384 = (14 − 11) × 128.

참고 — 배포 체크포인트(저자 재학습, 8 GPU) 0.777 · 논문 보고치 0.72.
SIMPL 이 재는 minADE 는 "최소 FDE 모드의 ADE" 라 일반 정의보다 같거나 크다.
