# 연구원 요청 진행 상태 (2026-10-26)

이번 비교는 사용자 요청에 따라 원본 입력의 KCF/CSRT와 기존 검출+KF 계열
ByteTrack/Holdfast를 대상으로 한다. ROI 입력 비교는 이번 측정에 포함하지 않는다.

| 요청 | 현재 구현/증거 | 남은 일 |
|---|---|---|
| 감지→하나 선택→SOT | `Engine`의 DETECT/TRACK 모드, 실제 Hailo 검출로 시작 | 파일 입력 OpenCV 클릭 UI 연결 완료; 실제 카메라는 미검증 |
| SOT와 검출을 동시에 실행하지 않기 | SOT TRACK/RECOVER에 detector_called가 없음을 실제 로그에서 assert | UAV 제품 조건 검증 |
| 실패 시 복구, 불가하면 감지 복귀 | native 실패/invalid box→NCC 지역 복구 최대 5회→DETECT | drift 실패 감지 및 복구 identity 정확도 개선 |
| 대상 변경 시 다시 감지 | `Engine.select_target(box)`가 기존 추적 상태를 지우고 DETECT로 복귀 | OpenCV 클릭으로 변경 연결 완료; 카메라 미검증 |
| 전통적/기존 방법의 정확도·속도 | 같은 5개 대상, 각 3회, 실제 Hailo 전체 파이프라인, GT는 사후 채점 | 더 많은 독립 영상/대상 및 전체 초기 선택 실패 포함 |
| 자원 제약 확인 | CPU 시간, read-to-prediction 지연, 단계 시간, 온도; Pi spot check 스로틀링 없음 | 연속 RAM/NPU/전력 계측, 실제 UAV 보드 사양 확인 |
| 경량 딥러닝 SOT 검토 | OSTrack 실제 export/parser/quantization 진행 | 1024장/레벨4 양자화 통과; HEF 할당 오류 우회 시도 및 변형 모델 검증 남음 |

기존 ByteTrack/Holdfast 비교군은 매 프레임 검출하는 joint MOT에서 하나의 ID를
고른 방식이다. 이것을 검출을 끈 SOT와 같은 실행 구조라고 부르지 않는다.
이번 FPS는 파일 입력 직렬 처리이며 이전 동시 파이프라인 FPS와 구분한다.

결과: [시스템 정확도·속도](../sot/results/system_comparison.md).
검증: `sot/check_recovery.py`에서 KCF와 CSRT의 합성 실패/복구/복귀/대상 변경
상태 전환 통과. 실제 데이터의 전환 및 복구 출력의 정답 일치는 결과에서 따로 집계.
정상 반환하면서 다른 물체로 넘어가는 drift는 현재 실패 기준이 잡지 못한다.

연구원 요청 전체가 끝난 상태는 아니다. 우선 실행 흐름과 재현 가능한 속도/정확도
비교를 구성했고, 실환경 실패 판단 및 동일 대상 재획득 신뢰성이 남아 있다.

OpenCV 데모 사용법/성능 측정 조건: [LIVE_DEMO](../sot/LIVE_DEMO.md).
