# 같은 검출 박스로 시작한 SOT / MOT 비교

초기 선택에 성공한 5/13개 대상의 조건부 비교.
VisDrone val에서 고른 기존 13개 대상 중 Holdfast가 초기 선택에 성공한 5개를 사용.
각 방법은 같은 프레임의 같은 Holdfast 출력 박스로 시작. OpenCV는 좌표를 정수 반올림.
KCF/CSRT에는 GT 초기 박스를 제공하지 않음. 이후 GT는 점수 계산에만 사용.
Holdfast는 기존 RemDet-S 검출+MOT 전체 결과에서 선택한 ID를 계속 유지.
SOT는 매 프레임 영상으로 추적하며 검출·재감지·복구는 사용하지 않음.
초기화 프레임 제외. GOT-10k toolkit 0.1.3의 UAV123 metric/curve 코드 사용.
객체별 점수 동일 가중 평균. 공식 UAV123 결과가 아니며 전체 객체의 성능을 대표하지 않음.

## 정확도와 이번 SOT 실측 속도

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| Holdfast-S | 5 | 75.5 | 95.5 | 89.5 | historical below | historical below |
| KCF-full | 5 | 32.2 | 42.6 | 42.0 | 75.1 | 39.2 |
| KCF-roi | 5 | 31.3 | 34.4 | 41.5 | 93.2 | 43.1 |
| CSRT-full | 5 | 56.9 | 76.0 | 68.4 | 14.8 | 12.5 |
| CSRT-roi | 5 | 55.5 | 77.7 | 66.4 | 20.3 | 16.3 |

## 속도 비교의 범위

SOT FPS: 이번 실행의 원본 JPEG 읽기 + crop + 추적 + ROI 재초기화 비용 포함.
검출·복구·화면 표시·영상 송출 제외. OpenCV 스레드 1개.
실패를 빠르게 반환하는 update 호출도 FPS에 포함되므로 속도만으로 실용성을 판단하지 않음.
ROI는 512x512 원본 crop, resize 없음. 중심 이동 시 예측 박스로만 재초기화.
Holdfast FPS: 이번에는 Hailo 장치 접근 불가로 재측정하지 못함.
기존 val 전체 2,846프레임 측정: RemDet-S + Holdfast 32.4 FPS (검출 포함).
기존 MOT는 reduced decode 및 동시 파이프라인을 사용하므로 이번 SOT FPS와 직접적인 동일 조건 속도 순위는 아님.

## 해석 제한

검출 가능한 5개 대상으로 제한한 결과이므로 나머지 8개의 초기 선택 실패는 별도로 고려해야 함.
Holdfast는 모든 사람/차량을 함께 추적한 결과이고 SOT는 대상 하나씩 독립 실행.
ROI 변경 시 필터 재초기화 효과가 섞여 있어 crop만의 효과로 해석하면 안 됨.
작은 표본·최대 200프레임·단일 실행의 예비 비교. 실시간 카메라 및 실패 복구까지 검증한 결과는 아님.

## 대상별 결과

| Sequence | Target | Method | AUC | Precision@20 | ROI resets |
|---|---|---|---:|---:|---:|
| uav0000086_00000_v | pedestrian:11 | Holdfast-S | 67.2 | 95.5 | - |
| uav0000086_00000_v | pedestrian:11 | KCF-full | 1.3 | 1.5 | 0 |
| uav0000086_00000_v | pedestrian:11 | KCF-roi | 1.3 | 1.5 | 0 |
| uav0000086_00000_v | pedestrian:11 | CSRT-full | 14.7 | 23.6 | 0 |
| uav0000086_00000_v | pedestrian:11 | CSRT-roi | 14.6 | 23.1 | 1 |
| uav0000137_00458_v | car:13 | Holdfast-S | 91.4 | 100.0 | - |
| uav0000137_00458_v | car:13 | KCF-full | 81.1 | 99.5 | 0 |
| uav0000137_00458_v | car:13 | KCF-roi | 74.7 | 58.8 | 5 |
| uav0000137_00458_v | car:13 | CSRT-full | 84.4 | 99.5 | 0 |
| uav0000137_00458_v | car:13 | CSRT-roi | 81.1 | 99.0 | 5 |
| uav0000305_00000_v | car:22 | Holdfast-S | 71.8 | 91.3 | - |
| uav0000305_00000_v | car:22 | KCF-full | 53.8 | 79.8 | 0 |
| uav0000305_00000_v | car:22 | KCF-roi | 56.5 | 79.8 | 0 |
| uav0000305_00000_v | car:22 | CSRT-full | 60.6 | 72.7 | 0 |
| uav0000305_00000_v | car:22 | CSRT-roi | 64.6 | 77.6 | 15 |
| uav0000339_00001_v | pedestrian:17 | Holdfast-S | 64.2 | 95.5 | - |
| uav0000339_00001_v | pedestrian:17 | KCF-full | 13.3 | 16.1 | 0 |
| uav0000339_00001_v | pedestrian:17 | KCF-roi | 12.8 | 16.1 | 0 |
| uav0000339_00001_v | pedestrian:17 | CSRT-full | 56.8 | 88.4 | 0 |
| uav0000339_00001_v | pedestrian:17 | CSRT-roi | 57.5 | 93.5 | 9 |
| uav0000339_00001_v | car:5 | Holdfast-S | 82.7 | 95.5 | - |
| uav0000339_00001_v | car:5 | KCF-full | 11.5 | 16.1 | 0 |
| uav0000339_00001_v | car:5 | KCF-roi | 11.5 | 16.1 | 0 |
| uav0000339_00001_v | car:5 | CSRT-full | 68.3 | 96.0 | 0 |
| uav0000339_00001_v | car:5 | CSRT-roi | 59.8 | 95.5 | 7 |
