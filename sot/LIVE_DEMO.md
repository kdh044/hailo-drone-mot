# OpenCV에서 대상 하나 선택하기

```bash
cd ~/hailo-test
QT_QPA_PLATFORM=xcb ~/sot-venv/bin/python -u sot/live_demo.py --tracker holdfast
```

실제 RemDet-S HEF로 검출한다. GT/정답 박스는 읽지 않는다.
영상에서 객체 클릭: 대상 선택/변경. 배경 클릭 또는 R: 전체 검출 박스 표시.
Space: 정지/재생. 정지 중 객체를 클릭하면 대상 박스만 선택하고,
다시 Space를 눌러 진행한다. KF 모드의 ID 확정은 재생 후 실제 후속 프레임에서 한다.

- 1 KCF / 2 CSRT / 3 ByteTrack / 4 Holdfast
- N/P: 다음/이전 시퀀스, S: 시퀀스 목록, B: 처음부터
- I: CPU/RAM/온도/단계 시간, H: 도움말, Tab: 버튼, Q/Esc: 종료

KCF/CSRT는 대상 선택 후 검출을 끈 SOT다. native 실패 후 지역 템플릿 복구,
최대 5회 실패 후 재검출을 시도한다. 복구는 NCC>=0.85 휴리스틱이며
심한 drift를 성공으로 반환하는 문제와 identity 보장이 남아 있다.
복구 검색 영역을 키우므로 일부 프레임이 크게 느려질 수 있다.

ByteTrack/Holdfast는 검출이 매 프레임 필요한 비교 모드다. 전체 검출 중
선택 대상 위치/클래스로 후보 하나를 골라 추적기에 입력한다.
전체 객체 MOT에서 ID만 강조하는 기존 비교와 입력 조건이 다르므로
그때의 정확도 수치를 이 UI의 성능으로 인용하면 안 된다.

## 파일 처리량과 카메라 성능

KF 모드는 순서가 보존되는 두 개의 작은 대기열로 읽기/전처리, 검출,
추적을 겹친다. 각 프레임은 반드시 자기 검출 결과와 묶여 전달된다.
KCF/CSRT 모드에는 검출 producer를 붙이지 않는다.

기본값은 입력/재생 FPS를 강제하지 않고 파일을 처리한다.
`procFPS`는 처리량이며 카메라 촬영 FPS 또는 입력부터 출력까지의 지연이 아니다.
`--source-fps N`은 사용자가 원할 때만 파일 도착 간격을 흉내 내는 옵션이다.
실시간 카메라 성능은 실제 입력 타임스탬프, 출력 지연, 누락 프레임을 별도로 측정해야 한다.
ROI crop과 축소 JPEG 디코딩은 아직 적용하지 않았다.

## 자원과 시간

상단 CPU는 시스템 전체 4코어 평균이다. I 화면의 앱 CPU는 1코어=100%,
4코어 최대=400%이며, RAM은 현재 프로세스 RSS와 시스템 available을 구분한다.
Hailo infer~%는 최근 추론 API 호출 시간/경과 시간의 추정치이며 DMA·대기를 포함한다.
순수 NPU 점유율, 남은 TOPS 또는 추가 모델을 넣을 수 있는 용량을 뜻하지 않는다.

`/tmp/hailo-live-demo.events.jsonl`에 프레임별 모드/검출 호출/입력 후보 수와
읽기 대기, 백그라운드 디코딩/전처리, 추론, 후처리, 추적 시간을 기록한다.
병렬 단계의 시간을 단순 합산해서 FPS를 계산하면 안 된다.

짧은 동일 첫 시퀀스/한 대상/120프레임 예비 측정(실제 Hailo, GUI/재생 제한 제외):
Holdfast 직렬 검출+추적 22.2 FPS → 병렬 37.2 FPS.
카메라 실측이나 정확도 개선 결과가 아니다.
`results/selected_pipeline_profile.summary.json`,
`results/selected_holdfast_parallel_profile.summary.json` 참조.

Holdfast가 여전히 ByteTrack보다 비싼 주된 이유는 배경 카메라 움직임 보정이다.
24프레임 예비 분리 측정에서 프레임 처리 포함 32.16ms, 영상 처리 제외 KF/매칭은
0.064ms였다. 객체 하나로 줄여도 보정 계산은 남는다. 보정을 끄면 알고리즘 조건이 달라진다.
