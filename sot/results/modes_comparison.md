# 원본/ROI SOT와 기존 MOT 비교

VisDrone MOT val에서 영상별 가장 긴 pedestrian/car 궤적 하나씩: 사람 7, 차량 6.
대상별 최대 200프레임. OpenCV 4.10.0, 스레드 1개.
MOT GT를 객체별로 분리한 자체 SOT 프로토콜이며 공식 UAV123 결과가 아님.
설치된 GOT-10k toolkit 0.1.3의 ExperimentUAV123._calc_metrics/_calc_curves로 계산.
21개 IoU 임계값 Success curve의 평균=AUC, Precision@20px. 모든 점수는 0–100.
각 대상 점수를 동일 가중 평균. 초기화 프레임 제외, GT 없는 프레임 제외.

## 전체 대상

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| KCF-full | 13 | 35.0 | 42.2 | 39.3 | 99.3 | 33.8 |
| KCF-roi | 13 | 33.7 | 39.1 | 38.9 | 130.1 | 36.7 |
| CSRT-full | 13 | 53.3 | 72.0 | 57.8 | 15.4 | 11.8 |
| CSRT-roi | 13 | 53.4 | 72.5 | 61.0 | 23.0 | 15.9 |
| Holdfast-S | 13 | 29.0 | 36.7 | 34.4 | historical below | historical below |
| ByteTrack-S | 13 | 29.6 | 37.6 | 35.2 | historical below | historical below |

## 초기화 조건 차이

SOT는 첫 GT 박스를 제공받는다. MOT는 저장된 RemDet-S 검출+추적 결과에서
첫 선택 GT 박스를 고정하고 최대 5프레임 동안 그 박스와 IoU>=0.5인 트랙 생성을 기다린 뒤 ID를 고정한다.
생성 전 프레임은 누락으로 채점. 5프레임 내 선택 실패는 전 구간 0점. 이후 GT로 ID를 갈아타지 않는다.
선택 대기 중에도 첫 박스만 사용하고 미래 GT 위치는 매칭에 쓰지 않는다.
따라서 전체 대상 표에는 검출/트랙 생성 실패의 영향도 포함된다.
MOT는 영상 시작부터 실행된 결과이며 나중에 등장한 대상에서는 이전 문맥을 가진다.
원 저자 SOT/MOT 알고리즘의 공정한 공식 순위로 해석하면 안 된다.

### Holdfast-S: 초기 선택 성공 5/13

해당 MOT와 초기 선택이 성공한 동일 대상들에서만 SOT 비교:

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| KCF-full | 5 | 35.4 | 42.8 | 42.7 | 91.6 | 43.1 |
| KCF-roi | 5 | 34.0 | 34.8 | 42.0 | 118.8 | 48.1 |
| CSRT-full | 5 | 53.4 | 71.5 | 55.0 | 15.8 | 13.2 |
| CSRT-roi | 5 | 57.3 | 74.2 | 69.8 | 21.0 | 16.7 |
| Holdfast-S | 5 | 75.5 | 95.5 | 89.5 | historical below | historical below |

### ByteTrack-S: 초기 선택 성공 5/13

해당 MOT와 초기 선택이 성공한 동일 대상들에서만 SOT 비교:

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| KCF-full | 5 | 35.4 | 42.8 | 42.7 | 91.6 | 43.1 |
| KCF-roi | 5 | 34.0 | 34.8 | 42.0 | 118.8 | 48.1 |
| CSRT-full | 5 | 53.4 | 71.5 | 55.0 | 15.8 | 13.2 |
| CSRT-roi | 5 | 57.3 | 74.2 | 69.8 | 21.0 | 16.7 |
| ByteTrack-S | 5 | 77.1 | 97.9 | 91.4 | historical below | historical below |

## 클래스별 정확도

### pedestrian

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| KCF-full | 7 | 16.0 | 25.2 | 19.8 | 152.9 | 39.4 |
| KCF-roi | 7 | 16.0 | 25.0 | 19.8 | 199.4 | 41.9 |
| CSRT-full | 7 | 43.9 | 69.7 | 50.1 | 15.6 | 12.1 |
| CSRT-roi | 7 | 42.1 | 68.9 | 47.2 | 22.8 | 15.9 |
| Holdfast-S | 7 | 18.8 | 27.3 | 22.8 | historical below | historical below |
| ByteTrack-S | 7 | 19.2 | 27.9 | 23.2 | historical below | historical below |

### car

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| KCF-full | 6 | 57.0 | 62.1 | 62.1 | 71.5 | 29.2 |
| KCF-roi | 6 | 54.4 | 55.4 | 61.1 | 93.9 | 32.3 |
| CSRT-full | 6 | 64.3 | 74.7 | 66.9 | 15.1 | 11.6 |
| CSRT-roi | 6 | 66.6 | 76.7 | 77.1 | 23.3 | 15.8 |
| Holdfast-S | 6 | 41.0 | 47.8 | 48.0 | historical below | historical below |
| ByteTrack-S | 6 | 41.8 | 49.0 | 49.1 | historical below | historical below |

## 속도 해석

SOT: 이번 실행의 실제 측정. 영상 decode + crop + 추적 + ROI 이동에 따른 재초기화 포함.
검출, 실패 복구, 화면 표시, 송출, 실시간 카메라 디코딩은 제외.
ROI도 원본 JPEG 전체를 decode하므로 crop만으로 읽기 비용은 줄지 않음.
MOT 정확도는 저장된 결과에서 계산하므로 이번 평가 파일 읽기 속도를 추적 FPS로 쓰지 않음.
다음은 기존 e2e_pipeline_all.json의 val 전체 2,846프레임 실측이며 이번 실험과 입력 전처리/시점이 다름:

| 기존 파이프라인 | 전체 FPS | 추적 단계 FPS | 프레임 수 |
|---|---:|---:|---:|
| RemDet-S + Holdfast | 32.4 | 32.7 | 2846 |
| RemDet-S + ByteTrack | 36.2 | 190.8 | 2846 |

## ROI 구현 조건

원본 영상에서 512x512 crop. resize 없음.
프레임 밖은 회색 padding. 첫 GT 박스 중심으로 시작하며 이후 직전 예측으로만 갱신.
예측 중심이 ROI 중심에서 폭의 20% 이상 벗어나면 ROI를 이동한다.
OpenCV 내부 좌표계는 직접 이동할 수 없으므로 이동 시 예측 박스로 재초기화한다.
재초기화 횟수와 비용을 기록한다. 이 결과는 단순 crop 효과와 필터 재학습 효과가 함께 포함된다.
추적 실패 시 GT 복구 없음. 별도 복구/재감지 파이프라인의 평가는 아직 포함하지 않음.

## 대상별 점수

| Sequence | Target | Method | AUC | Precision@20 | ROI resets |
|---|---|---|---:|---:|---:|
| uav0000086_00000_v | pedestrian:11 | KCF-full | 1.6 | 2.0 | 0 |
| uav0000086_00000_v | pedestrian:11 | KCF-roi | 1.6 | 2.0 | 0 |
| uav0000086_00000_v | pedestrian:11 | CSRT-full | 14.6 | 16.6 | 0 |
| uav0000086_00000_v | pedestrian:11 | CSRT-roi | 14.6 | 16.6 | 1 |
| uav0000117_02622_v | pedestrian:0 | KCF-full | 0.0 | 0.0 | 0 |
| uav0000117_02622_v | pedestrian:0 | KCF-roi | 0.0 | 0.0 | 0 |
| uav0000117_02622_v | pedestrian:0 | CSRT-full | 0.4 | 0.0 | 0 |
| uav0000117_02622_v | pedestrian:0 | CSRT-roi | 0.2 | 0.0 | 7 |
| uav0000117_02622_v | car:23 | KCF-full | 36.2 | 0.0 | 0 |
| uav0000117_02622_v | car:23 | KCF-roi | 27.7 | 0.0 | 6 |
| uav0000117_02622_v | car:23 | CSRT-full | 36.8 | 0.0 | 0 |
| uav0000117_02622_v | car:23 | CSRT-roi | 34.3 | 0.0 | 6 |
| uav0000137_00458_v | pedestrian:41 | KCF-full | 22.4 | 27.1 | 0 |
| uav0000137_00458_v | pedestrian:41 | KCF-roi | 22.0 | 26.1 | 1 |
| uav0000137_00458_v | pedestrian:41 | CSRT-full | 70.9 | 89.4 | 0 |
| uav0000137_00458_v | pedestrian:41 | CSRT-roi | 59.2 | 82.4 | 7 |
| uav0000137_00458_v | car:13 | KCF-full | 86.0 | 100.0 | 0 |
| uav0000137_00458_v | car:13 | KCF-roi | 77.6 | 59.8 | 5 |
| uav0000137_00458_v | car:13 | CSRT-full | 80.4 | 99.5 | 0 |
| uav0000137_00458_v | car:13 | CSRT-roi | 79.9 | 98.5 | 5 |
| uav0000182_00000_v | pedestrian:75 | KCF-full | 23.6 | 31.7 | 0 |
| uav0000182_00000_v | pedestrian:75 | KCF-roi | 23.6 | 31.7 | 0 |
| uav0000182_00000_v | pedestrian:75 | CSRT-full | 55.0 | 100.0 | 0 |
| uav0000182_00000_v | pedestrian:75 | CSRT-roi | 51.6 | 100.0 | 4 |
| uav0000182_00000_v | car:127 | KCF-full | 58.2 | 76.9 | 0 |
| uav0000182_00000_v | car:127 | KCF-roi | 58.2 | 76.9 | 0 |
| uav0000182_00000_v | car:127 | CSRT-full | 72.9 | 100.0 | 0 |
| uav0000182_00000_v | car:127 | CSRT-roi | 72.9 | 100.0 | 0 |
| uav0000268_05773_v | pedestrian:26 | KCF-full | 43.5 | 88.9 | 0 |
| uav0000268_05773_v | pedestrian:26 | KCF-roi | 43.5 | 88.9 | 0 |
| uav0000268_05773_v | pedestrian:26 | CSRT-full | 49.1 | 100.0 | 0 |
| uav0000268_05773_v | pedestrian:26 | CSRT-roi | 49.1 | 100.0 | 0 |
| uav0000268_05773_v | car:8 | KCF-full | 85.3 | 100.0 | 0 |
| uav0000268_05773_v | car:8 | KCF-roi | 85.3 | 100.0 | 0 |
| uav0000268_05773_v | car:8 | CSRT-full | 84.6 | 100.0 | 0 |
| uav0000268_05773_v | car:8 | CSRT-roi | 84.4 | 100.0 | 0 |
| uav0000305_00000_v | pedestrian:55 | KCF-full | 8.0 | 10.3 | 0 |
| uav0000305_00000_v | pedestrian:55 | KCF-roi | 8.0 | 10.3 | 0 |
| uav0000305_00000_v | pedestrian:55 | CSRT-full | 56.2 | 89.1 | 0 |
| uav0000305_00000_v | pedestrian:55 | CSRT-roi | 56.0 | 89.1 | 4 |
| uav0000305_00000_v | car:22 | KCF-full | 65.0 | 79.8 | 0 |
| uav0000305_00000_v | car:22 | KCF-roi | 66.5 | 79.8 | 0 |
| uav0000305_00000_v | car:22 | CSRT-full | 66.5 | 73.2 | 0 |
| uav0000305_00000_v | car:22 | CSRT-roi | 70.4 | 82.0 | 14 |
| uav0000339_00001_v | pedestrian:17 | KCF-full | 13.2 | 16.1 | 0 |
| uav0000339_00001_v | pedestrian:17 | KCF-roi | 13.2 | 16.1 | 0 |
| uav0000339_00001_v | pedestrian:17 | CSRT-full | 60.8 | 93.0 | 0 |
| uav0000339_00001_v | pedestrian:17 | CSRT-roi | 63.7 | 94.0 | 9 |
| uav0000339_00001_v | car:5 | KCF-full | 11.3 | 16.1 | 0 |
| uav0000339_00001_v | car:5 | KCF-roi | 11.3 | 16.1 | 0 |
| uav0000339_00001_v | car:5 | CSRT-full | 44.9 | 75.4 | 0 |
| uav0000339_00001_v | car:5 | CSRT-roi | 57.9 | 79.9 | 7 |
| uav0000086_00000_v | pedestrian:11 | Holdfast-S | 67.2 | 95.5 | - |
| uav0000086_00000_v | pedestrian:11 | ByteTrack-S | 70.8 | 100.0 | - |
| uav0000117_02622_v | pedestrian:0 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000117_02622_v | pedestrian:0 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000117_02622_v | car:23 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000117_02622_v | car:23 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000137_00458_v | pedestrian:41 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000137_00458_v | pedestrian:41 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000137_00458_v | car:13 | Holdfast-S | 91.4 | 100.0 | - |
| uav0000137_00458_v | car:13 | ByteTrack-S | 91.3 | 100.0 | - |
| uav0000182_00000_v | pedestrian:75 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000182_00000_v | pedestrian:75 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000182_00000_v | car:127 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000182_00000_v | car:127 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000268_05773_v | pedestrian:26 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000268_05773_v | pedestrian:26 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000268_05773_v | car:8 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000268_05773_v | car:8 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000305_00000_v | pedestrian:55 | Holdfast-S | 0.0 | 0.0 | - |
| uav0000305_00000_v | pedestrian:55 | ByteTrack-S | 0.0 | 0.0 | - |
| uav0000305_00000_v | car:22 | Holdfast-S | 71.8 | 91.3 | - |
| uav0000305_00000_v | car:22 | ByteTrack-S | 76.8 | 98.4 | - |
| uav0000339_00001_v | pedestrian:17 | Holdfast-S | 64.2 | 95.5 | - |
| uav0000339_00001_v | pedestrian:17 | ByteTrack-S | 63.8 | 95.5 | - |
| uav0000339_00001_v | car:5 | Holdfast-S | 82.7 | 95.5 | - |
| uav0000339_00001_v | car:5 | ByteTrack-S | 82.6 | 95.5 | - |
