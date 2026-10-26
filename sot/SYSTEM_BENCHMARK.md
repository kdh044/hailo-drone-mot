# 동일 조건 시스템 속도 측정

실행 코드: `benchmark_system.py`. ROI 입력은 사용하지 않는다.

현재 원격 에이전트 실행 환경에서는 `/dev/hailo0`가 보이지 않아 실제 Hailo
전체 속도를 측정하지 못했다. 아래 명령은 Pi 호스트 터미널에서 실행해야 한다.
기존 FPS와 새 FPS를 동일 실험인 것처럼 섞지 않는다.

## Pi에서 실행

```bash
cd ~/hailo-test
~/sot-venv/bin/python sot/benchmark_system.py \
  ~/datasets/visdrone/mot-val-hf/sequences/uav0000086_00000_v \
  --hef models/remdet_s_visdrone.hef \
  --mot-site-packages ~/mot-venv/lib/python3.13/site-packages \
  --frames 200 --trials 3 \
  --output sot/results/system_hailo.raw.json
```

OpenCV contrib 4.10을 먼저 로드하고 기존 MOT 환경의 추가 의존성을 사용한다.
여기서는 실제로 KCF/CSRT/ByteTrack/Holdfast 생성과 캐시 입력 실행을 확인했다.
Python 버전이 다르면 `--mot-site-packages` 경로를 그대로 혼합하지 말고 같은
환경에 필요한 패키지를 설치해야 한다.

기본 대상은 첫 검출의 최고 confidence 박스이며 이후 그 초기 선택을 유지한다.
특정 대상을 정하려면 `--selection-box x y width height`로 첫 선택 영역을 지정한다.
여러 번 실행할 때는 모든 방법에 동일한 선택 영역·입력 프레임을 사용한다.

## 포함하는 시간

- 모든 방법: 같은 원본 JPEG 읽기, full decode, 동일한 640 입력 전처리와 RemDet-S.
- KCF/CSRT: DETECT에서만 검출, TRACK에서는 SOT, 실패 시 지역 템플릿 복구,
  5회 복구 실패 후 다시 검출. 복구의 identity 정확도는 별도 검증이 필요하다.
- ByteTrack/Holdfast: 매 프레임 검출 및 기존 joint tracker 실행, 선택 ID를 평가.
  ID 누락 시 대기/복구 상태를 거쳐 다시 선택하며 재획득 identity를 보장하지 않는다.
- 동일한 직렬 처리, OpenCV 스레드 1개, OpenCL 미사용. Hailo 추론 5회 warmup.
- 입력 읽기 시작부터 예측 박스 준비까지의 p50/p95 지연, 전체 파일 입력 FPS,
  검출 호출 수, 복구 프레임 수, 출력 누락 수, 프레임별 box와 상태를 저장한다.
- 3회 반복하며 방법 실행 순서를 순환시킨다. 반복값을 모두 기록한다.

이것은 파일 입력 기반 처리량이며 카메라 capture/queue/display/transmission latency가 아니다.
기존 32.4/36.2 FPS 측정은 동시 파이프라인이라 이 직렬 기준선과 별도 비교해야 한다.
고속이더라도 잘못된 대상을 추적할 수 있으므로 box 결과를 GT로 별도 채점해야 한다.

## 하드웨어 없는 로직 검사

```bash
~/sot-venv/bin/python sot/benchmark_system.py \
  ~/datasets/visdrone/mot-val-hf/sequences/uav0000086_00000_v \
  --mot-site-packages ~/mot-venv/lib/python3.13/site-packages \
  --cached-detections mot/dets/s/uav0000086_00000_v.npy \
  --frames 10 --trials 1 --output /tmp/system_logic_smoke.json
```

캐시 모드는 `backend=cached_logic_only`, `hardware_e2e_fps=null`을 기록한다.
검출 시간이 빠진 캐시 측정값을 실제 시스템 FPS로 보고하지 않는다.

## 실패 판단의 의미

`tracker.update()`가 False이거나 박스가 유효하지 않은 경우는 코드에서 감지할 수 있다.
다른 사람/배경으로 drift했지만 True를 반환하는 경우는 이것만으로 감지할 수 없다.
정답 박스로 측정하는 실패(IoU 등)와 실제 운용 중 사용하는 실패 신호는 다르다.

실제 기존 공통 5개 대상 결과에서 CSRT-full은 채점한 979프레임 모두 박스를
반환했지만, 그중 145프레임은 GT와 IoU가 0.1 미만이었다. 이는 오프라인에서
정답으로 확인한 심한 위치 오류이며, 운용 중 GT로 실패 판정한다는 뜻은 아니다.

현재 코드의 실패 기준은 native 실패/박스 유효성의 첫 기준선이다.
지역 복구는 초기 템플릿 NCC>=.85라는 미검증 휴리스틱이다. 이것을 신뢰성 있는
drift detector 또는 ReID라고 주장하지 않는다. 향후 외형 일치, 이동/크기 변화,
필요 시 순차 검출 확인 등을 추가하고 별도 검증 데이터에서 오탐·미탐을 재야 한다.
