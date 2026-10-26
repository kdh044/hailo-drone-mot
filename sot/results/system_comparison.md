# Hailo 포함 시스템 실측

Five previously acquired VisDrone targets; original full JPEG input; same detector and serial pipeline; three rotated trials; detector-derived selection box; GT used offline only; no ROI input; initial frame excluded from scores
5/13 previously acquired targets; conditional diagnostic, not official UAV123 results

| 방법 | 전체 FPS | 실행별 FPS 범위 | AUC | P@20 | 박스 누락 | CPU (1코어=100%) |
|---|---:|---:|---:|---:|---:|---:|
| KCF | 30.0 | 17.0–43.6 | 50.1 | 65.8 | 17.9% | 93.9% |
| CSRT | 12.5 | 8.6–15.8 | 53.1 | 74.7 | 0.0% | 99.9% |
| bytetrack | 25.7 | 18.9–31.1 | 77.1 | 97.8 | 0.8% | 63.6% |
| holdfast | 15.1 | 11.5–18.6 | 75.5 | 95.5 | 2.5% | 78.8% |

JPEG 읽기·전처리·Hailo 검출·추적·복구 포함. 화면 표시/전송/카메라 대기 제외. 원본 입력, ROI 없음. 속도는 총 프레임/총 경과 시간, 정확도는 대상별 동일 가중 평균.
매 시행 방법 순서를 순환. 기존 병렬 MOT FPS와 직접 같은 조건이 아님. ByteTrack/Holdfast는 기존 joint MOT를 매 프레임 실행하고 선택 ID를 평가.
AUC/P@20은 GOT-10k UAV123 evaluator의 곡선 구현을 사용한 VisDrone 진단값. 공식 UAV123 결과가 아님. 실패/복구 중 박스 누락도 0 overlap으로 포함.

## 연구원 요청 구현 및 검증

- 감지→대상 선택→SOT. SOT의 TRACK/RECOVER 중 검출 호출이 없음을 실제 로그에서 검사.
- native 실패/유효하지 않은 박스→지역 NCC 복구→최대 5회 실패 시 DETECT. 대상 변경 API는 DETECT로 복귀.
- `check_recovery.py`: KCF/CSRT 복구 성공·복구 시간 초과·재감지·대상 변경 상태 전환 통과. 합성 검사이며 실환경 복구 정확도 증명이 아님.
- 다른 물체로 drift하면서 valid box를 반환하는 실패는 아직 검출하지 못함. 재감지 시 동일 identity 보장 없음.
- 실제 UAV 하드웨어 성능, 카메라 지연, NPU 점유율, 장시간 신뢰성은 미검증. CPU 수치는 프로세스 CPU 시간/벽시계 시간.

| 방법 | 복구 박스 출력 | 그중 GT IoU≥0.5 | 박스는 있지만 GT IoU<0.1 프레임 |
|---|---:|---:|---:|
| KCF | 0 | 0 | 453 |
| CSRT | 0 | 0 | 435 |
| bytetrack | 3 | 3 | 0 |
| holdfast | 0 | 0 | 0 |

위 횟수는 3회 반복 합계이며 독립 표본 수가 아님. GT는 사후 평가에만 사용. MOT의 RECOVER→TRACK은 같은 ID 재출현이며 SOT 템플릿 복구와 다름.
