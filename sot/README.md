# 단일 객체 추적 기준선

- [같은 조건의 전체 시스템 속도 측정 코드와 실행법](SYSTEM_BENCHMARK.md)
- [E2E 후보 조사와 GitHub 인기도](../docs/e2e_tracker_candidates.md)

[20초 비교 영상: KCF / CSRT / 검출+ByteTrack / 검출+Holdfast](results/tracker_comparison.mp4)

원본 입력 추적 결과만 사용. 초록색=GT, 주황색=예측. 모든 칸은 같은 재생 속도이며
실제 처리 속도를 표현하지 않는다. 확대 창은 관찰용이며 추적 입력의 ROI crop이 아니다.
보행자 실패 사례와 차량 유지/실패 사례 3개를 선택해 총 20초로 구성했다.
선택된 예시 영상이며 전체 정량 결과를 대신하지 않는다.

```bash
~/sot-venv/bin/python sot/render_comparison.py
ffmpeg -i sot/results/tracker_comparison_temp.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart sot/results/tracker_comparison.mp4
```

## 동일 검출 박스 초기화 추가 실험 (2026-10-26)

[추가 실험 결과](results/shared_init_comparison.md),
[추가 실험 곡선](results/shared_init_comparison.png).

초기 선택 가능한 5/13개 대상에서 Holdfast의 첫 출력 박스를 KCF/CSRT에도 제공해
같은 프레임·같은 검출 박스로 시작했다. 새 SOT 실행이며 Holdfast는 저장 결과를 사용했다.
AUC: Holdfast 75.5, CSRT 원본 56.9 / ROI 55.5, KCF 원본 32.2 / ROI 31.3.
읽기 포함 SOT 속도: CSRT 12.5 / 16.3 FPS, KCF 39.2 / 43.1 FPS.
검출 가능한 대상에 한정한 작은 예비 실험이며 검출 실패한 8개 대상은 포함하지 않는다.

```bash
~/sot-venv/bin/python sot/shared_init.py
~/sot-venv/bin/python sot/report_shared.py sot/results/shared_init.raw.json --output sot/results/shared_init_comparison.md
```

`shared_init.py`는 아래 기존 비교의 로컬 원본 결과 `modes_13targets.json`을 입력으로 사용한다.
GitHub의 요약 파일만으로 표·곡선을 재생성하려면:

```bash
~/sot-venv/bin/python sot/report_shared.py sot/results/shared_init_comparison.summary.json --output /tmp/shared-init.md
```

## 원본/ROI vs MOT 비교 (2026-10-26)

[전체 결과 및 조건](results/modes_comparison.md),
[전체 곡선](results/modes_comparison.png),
[Holdfast 초기 선택 성공 대상의 곡선](results/modes_comparison_acquired.png).

평가기: GOT-10k toolkit 0.1.3의 UAV123 metric/curve 코드를 사용.
데이터는 VisDrone에서 선별한 13개 궤적이므로 공식 UAV123 점수는 아님.
첫 프레임 제외 및 GT 누락 프레임 제외는 이 진단 실험에서 정한 규칙.
MOT는 초기 선택 박스에서 5프레임 내 트랙을 선택하고 이후 ID를 고정.
SOT의 GT 초기화와 MOT의 검출 초기화 차이 때문에 전체/초기 선택 성공 조건부 결과를 모두 표시.

환경 준비 (저장소 루트):

```bash
python -m venv --system-site-packages ~/sot-venv
~/sot-venv/bin/python -m pip install -r sot/requirements-eval-lock.txt
```

시스템의 OpenCV contrib(KCF/CSRT 포함)과 NumPy를 사용한다.
기존 MOT 준비 과정의 `mot/prepare_gt.py`로 만든 `mot/gt.pkl`과
`mot/data/trackers/VD{person,vehicle}-val/s__{holdfast,rf-bytetrack}-joint/data/` 결과가 필요하다.
데이터·모델은 저장소에 포함하지 않는다. 경로는 기존 MOT 설정 및 `--images`를 따른다.

GitHub에는 GT/프레임별 박스가 없는 요약 JSON도 포함한다.
추적을 다시 실행하지 않고 표·그래프만 재생성할 수 있다:

```bash
~/sot-venv/bin/python sot/report_modes.py sot/results/modes_comparison.summary.json --output /tmp/sot-comparison.md
```

```bash
~/sot-venv/bin/python sot/compare_modes.py --output sot/results/modes_13targets.json
~/sot-venv/bin/python sot/report_modes.py sot/results/modes_13targets.json --output sot/results/modes_comparison.md
```

`--output`에 기존 결과가 있으면 완료된 항목을 재사용한다. 다시 측정하려면 다른 출력 경로를 쓴다.
평가 패키지는 `~/sot-venv`에 설치했고 버전은 `requirements-eval-lock.txt`에 기록.
SOT 원본/512 ROI는 실제 실행. MOT 정확도는 저장 결과를 같은 지표로 재채점.
Hailo 장치 접근 불가로 MOT 전체 FPS는 기존 측정을 별도로 인용.
검출→실패 복구→재감지를 포함하는 SOT 시스템 전체 비교는 아직 아님.

## 확장 비교 (2026-10-26)

val 7개 영상에서 사람 7개·차량 6개 궤적을 선택하여 대상당 최대 200프레임을
실행한 결과는 [전체 비교표](results/comparison.md)에 있음. 각 트래커 2,464개
GT 프레임 채점. 평균 IoU KCF .359 / CSRT .537, IoU>.5 비율 39.7% / 57.6%.
update 처리량 99.8 / 15.5 FPS, JPEG 읽기 포함 30.6 / 11.9 FPS.
선별한 13개 궤적에 대한 첫 기준선이며 전체 객체/공식 SOT 벤치마크 결과는 아님.
KCF 실패 호출이 빠르므로 처리량만으로 실용성을 판단하면 안 됨.

```bash
python sot/benchmark.py --frames 200 --sequences 7 --width 0 --output sot/results/visdrone_7seq_200.json
python sot/report.py sot/results/visdrone_7seq_200.json --output sot/results/comparison.md
```

## 현재 구현

- `benchmark.py`: KCF/CSRT, 첫 GT 박스로만 초기화, GT 재초기화 없음.
  기존 VisDrone MOT를 단일 궤적으로 뽑은 진단 실험이며 UAV123 공식 평가는 아님.
- `pipeline.py`: 전체 영상 감지 → 대상 선택 → SOT → 실패 시 지역 템플릿
  탐색 → 복구 실패 시 감지. 감지와 추적은 같은 프레임에 실행하지 않음.
  수동 선택이 기본이며 화면 없는 환경에서는 `--select highest-confidence` 사용.
  전체 재감지 후 자동 선택은 같은 객체임을 보장하지 않음.

## 실행

저장소 루트에서:

```bash
python sot/benchmark.py --frames 80 --sequences 2 --width 0 --output sot/results/baseline_original.json
python sot/pipeline.py ~/datasets/visdrone/mot-val-hf/sequences/uav0000086_00000_v --hef models/remdet_s_visdrone.hef --output /tmp/sot_trace.jsonl
```

## 2026-10-26 예비 측정

현재 Pi 실행 환경, OpenCV 4.10.0, OpenCV 스레드 1개, 원본 해상도.
각 영상의 가장 오래 보이는 사람/차량 궤적 중 하나, 첫 80프레임, 79회 update.
초기화 시간은 별도 기록. E2E에는 JPEG 읽기 및 초기화·평가 비용 포함.
실행 순서/캐시/다른 작업 영향을 통제한 최종 벤치마크가 아님.

| 영상 | 트래커 | update FPS | E2E FPS | 평균 IoU | IoU > .5 비율 |
|---|---|---:|---:|---:|---:|
| 0086 | KCF | 213.0 | 87.2 | .041 | 5.1% |
| 0086 | CSRT | 18.0 | 16.1 | .361 | 31.6% |
| 0117 | KCF | 125.0 | 31.3 | .000 | 0% |
| 0117 | CSRT | 14.9 | 11.0 | .007 | 0% |

KCF의 update FPS에는 실패를 빨리 반환하는 호출이 포함됨.
성공을 반환한 호출만의 FPS도 JSON에 따로 저장했으나 성공 플래그는
정확한 추적을 보장하지 않음. CSRT는 두 영상에서 계속 성공을 반환했지만
실제 GT와의 일치는 낮았음. 이 표로 일반적인 정확도 순위를 주장하면 안 됨.

## 검증 및 남은 작업

- 두 스크립트 문법 검사 통과, 실제 영상에서 KCF/CSRT 기준선 실행 완료.
- pipeline Hailo 실행은 이 실행 환경에서 `/dev/hailo0`를 열지 못해 검증 불가.
- 실시간 파이프라인은 원본 입력 기준선만 구현. 동적 외부 ROI 입력 및 원본 좌표
  역변환의 오프라인 비교는 `compare_modes.py`에서 실행 완료.
  UAV123 평가, 딥러닝 SOT, 실패 감지 신뢰도 보강, 실시간 ROI 통합은 아직 남음.
- KCF/CSRT 자체가 로컬 검색 영역을 사용하므로 외부 crop만으로 속도가
  크게 빨라진다고 가정하면 안 됨. 이동하는 crop을 적용할 때는 트래커 내부
  좌표계 보정 또는 재초기화 정책까지 명시해야 함.
- 복구의 template matching은 스케일/회전/유사 객체에 취약한 첫 기준선임.
  tracker.update() 성공 플래그만으로 drift를 감지하지 못함.
- 추적 객체 변경은 새 감지/선택으로 처리하는 UI 연결이 남음.
- 메모리 peak/온도 및 장시간 반복 속도 측정은 아직 하지 않음.

참고: [KCF 논문](https://arxiv.org/abs/1404.7584),
[CSR-DCF 논문](https://arxiv.org/abs/1611.08461).
