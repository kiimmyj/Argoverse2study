# focal 추종 탑뷰 애니메이션 (NGSIM / highD)

focal 차량 한 대를 정해서, 카메라가 그 차를 따라가는 탑뷰 애니메이션을 만든다.
도로(차선·램프)와 주변 차량이 함께 그려진다. Argoverse 2의 focal agent 개념을
고속도로 궤적 데이터셋에 옮겨온 것.

## 구성

| 파일 | 역할 |
|---|---|
| `scripts/ngsim_fetch.py` | NGSIM을 US DOT ITS DataHub API에서 필요한 부분만 받아 `data/`에 캐시 |
| `scripts/traffic_data.py` | NGSIM / highD를 공통 좌표계·공통 컬럼으로 변환 |
| `scripts/animate_focal.py` | 렌더링 + mp4 인코딩 |
| `output/` | 결과 mp4 (gitignore) |
| `data/` | 다운로드 캐시 (gitignore) |

## 사용

```bash
conda activate av2

python scripts/animate_focal.py                      # us-101, 가장 오래 머무는 차
python scripts/animate_focal.py --location i-80
python scripts/ngsim_fetch.py --location us-101 --list 20   # focal 후보 보기
python scripts/animate_focal.py --track-id 2113      # 차선변경 하는 차 지정
python scripts/animate_focal.py --start-seconds 40 --max-seconds 15 --speed 2
```

주요 옵션: `--ahead/--behind`(시야 m), `--speed`(배속), `--smooth`(위치 스무딩 창, 1=끔), `--gif`.

## 공통 좌표계

렌더러는 데이터셋을 구분하지 않는다. 어댑터가 아래 형태로 맞춰준다.

- `x` 주행방향 [m] (항상 + 방향 진행), `y` 횡방향 [m] (차량 왼쪽이 +, 즉 1차로가 화면 위)
- `x, y`는 차량 **중심**
- tracks 컬럼: `track_id frame t x y vx vy speed yaw length width lane_id cls`

## 데이터

### NGSIM (자동 다운로드)

`data.transportation.gov` Socrata API에서 받는다. 전체는 구간당 450만 행이라,
focal이 살아있는 시간창만 골라 받는다(약 20만 행 / 2분 분량).

차로 구성은 데이터에서 직접 유도한다 (lane별 `local_x` 평균 → 차선 경계, `local_y` 범위 → 램프 구간).

- us-101: 본선 5차로 + ON-RAMP(112~200m) → AUX(188~412m) → OFF-RAMP(399~489m), 전체 681m
- i-80: 본선 6차로 + ON-RAMP(38~213m), 전체 546m

간선도로 두 구간(lankershim, peachtree)은 교차로가 있어 구간마다 차로 구성이 달라진다.
궤적은 그대로 읽히지만 도로 그리기는 고속도로 두 구간만 제대로 지원한다.

주의할 점 세 가지 (코드에서 처리됨):

1. **`vehicle_id`는 전역 유일이 아니다.** us-101/i-80은 15분 녹화 3개를 합친 것이라 ID가 재사용된다.
   그래서 차량 인덱스를 `vehicle_id` + `total_frames`로 묶어 녹화별 트랙으로 갈라낸 뒤,
   "시간폭과 프레임 수가 일치하는" 연속 트랙만 focal 후보로 쓴다.
   (`vehicle_id`만으로 묶으면 재사용된 ID가 통째로 탈락하면서 그 안의 최장 트랙까지 같이 버려진다 —
   us-101 후보가 663개에서 5,075개로, i-80은 1,085개에서 5,670개로 늘었다.)
   시간창을 받은 뒤에는 시간 gap으로 한 번 더 쪼갠다.
2. **좌표는 차량 중심이 아니라 앞면 중앙이다.** 길이 절반만큼 뒤로 밀어 중심으로 바꾼다.
3. **차량 박스가 가끔 겹친다.** 원본 데이터의 같은 차로 범퍼 간격을 세어보면
   us-101은 0.24%, i-80은 2.59%가 음수다(최소 -7.2m). 변환 문제가 아니라 NGSIM 자체의
   판독 오차이므로 그대로 그린다. HUD의 `gap`도 음수로 나올 수 있다.
4. **위치 지터가 크다.** 특히 횡방향은 실제 움직임이 느린 데 비해 노이즈가 커서
   종방향의 두 배 창으로 스무딩한다. 안 하면 박스가 떨고 헤딩이 튄다.

### highD (직접 다운로드 필요)

levelXdata(RWTH Aachen) 라이선스가 필요해 자동으로 받을 수 없다. 받은 뒤:

```bash
python scripts/animate_focal.py --dataset highd --highd-prefix /path/to/highd/data/01
```

어댑터는 공개 포맷 명세(`*_tracks.csv` / `*_tracksMeta.csv` / `*_recordingMeta.csv`)에 맞춰
작성했고 합성 파일로 구조 검증까지 했으나, **실제 highD 파일로는 아직 검증하지 못했다.**
bbox 좌상단 → 중심 보정, 상·하행 좌표 뒤집기, `recordingMeta`의 차선 표시선 좌표 사용까지 구현되어 있다.

## 출력

matplotlib(Agg)으로 프레임을 그려 RGBA 버퍼를 꺼낸 뒤 인코딩한다. 1500x376, 10fps(NGSIM 실시간).

인코더는 두 갈래다.

- **ffmpeg가 있으면** 원시 프레임을 파이프로 밀어넣어 H.264(libx264, 기본 crf 20)로 뽑는다.
  `--crf`로 화질을 조절한다(낮을수록 고화질·큰 파일).
- **없으면** OpenCV `VideoWriter`(mp4v)로 떨어진다. 같은 화면인데 파일이 3배쯤 커진다
  (10초 기준 68 KB vs 235 KB). OpenCV 배포판에는 라이선스 문제로 H.264 인코더가 없어
  `avc1`은 열리지 않는다.

ffmpeg는 `conda install -c conda-forge ffmpeg`로 `av2` 환경에 설치돼 있다.
PATH에 없어도 실행 중인 파이썬 옆(`envs/av2/bin`)을 한 번 더 찾으므로, 환경을 활성화하지 않고
전체 경로로 실행해도 H.264로 나온다.
