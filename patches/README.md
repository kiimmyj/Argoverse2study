# patches

외부 저장소를 이 머신에서 돌아가게 하려고 고친 것들. 원본에는 올리지 않는다.

## simpl_local_fixes.patch

`~/av2_simpl` (https://github.com/HKUST-Aerial-Robotics/SIMPL, MIT, 커밋 2a33314) 에 적용한 수정.
클론을 다시 받으면 이 패치를 적용해야 돌아간다.

```bash
git clone --depth 1 https://github.com/HKUST-Aerial-Robotics/SIMPL.git ~/av2_simpl
cd ~/av2_simpl && git apply /home/user/Argoverse2study/patches/simpl_local_fixes.patch
```

| 파일 | 왜 |
| --- | --- |
| `simpl/simpl.py` | `from fractions import gcd` 는 Python 3.9 에서 제거됨 → `math.gcd` |
| `utils/utils.py`, `utils/logger.py` | `import imp` 는 Python 3.12 에서 제거됨 → `importlib` |
| `utils/optimizer.py` | torch 2.13 의 `LRScheduler` 에 `verbose` 인자가 없음 |
| `utils/evaluator.py`, `utils/argo_eval_forecasting.py` | 지표 함수가 AV1 API 에 의존 → 원본 그대로 복사해 끊음 |
| `run_prep_and_measure.sh` | 전처리 + 1에폭 실측 러너 (추가 파일) |

환경은 기존 conda `av2` 를 그대로 쓴다. 추가 설치는 `shapely`, `tensorboard` 둘뿐이다.
