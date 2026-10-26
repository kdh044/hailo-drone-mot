# E2E 추적기 후보 조사 (2026-10-26)

여기서 인기도는 연결된 GitHub API의 당일 stargazers_count로 확인했다.
후보들을 확인한 값이며 GitHub 전체를 망라한 순위/인용수 순위가 아니다.
스타 수와 최신 benchmark SOTA는 다른 개념이다.

| 저장소 | 종류 | 확인한 스타 | 용도/주의 |
|---|---|---:|---|
| [PySOT](https://github.com/STVIR/pysot) | SOT 프레임워크 | 4,606 | 단일 E2E 모델이 아님 |
| [SiamMask](https://github.com/foolwood/SiamMask) | Siamese 추적/분할 | 3,552 | 널리 알려진 이전 세대, 최신 SOTA라는 뜻은 아님 |
| [STARK](https://github.com/researchmm/Stark) | Transformer SOT | 715 | ICCV 2021 비교 후보 |
| [OSTrack](https://github.com/botaoye/OSTrack) | One-stream Transformer SOT | 695 | ECCV 2022 대표 기준선 |
| [TransT](https://github.com/chenxin-dlut/TransT) | Transformer SOT | 636 | CVPR 2021 비교 후보 |
| [MixFormer](https://github.com/MCG-NJU/MixFormer) | E2E SOT | 543 | CVPR 2022 / TPAMI 2024 |
| [SUTrack](https://github.com/chenxin-dlut/SUTrack) | 통합 다중모달 SOT | 170 | AAAI 2025, 정확도 비교 후보 |
| [UETrack](https://github.com/kangben258/UETrack) | 경량 통합 SOT | 54 | CVPR 2026, CPU/AGX 속도 공개 |
| [MOTR](https://github.com/megvii-research/MOTR) | E2E MOT | 812 | 검출/다중 객체 추적 통합, SOT와 구분 |
| [TrackFormer](https://github.com/timmeinhardt/trackformer) | E2E MOT | 651 | 다중 객체용 Transformer |

## 현재 프로젝트에서 먼저 검토할 후보

- 기준선: OSTrack. 템플릿과 검색 영상을 함께 처리하는 대표 one-stream 구조.
- 최신 경량 후보: UETrack-T/S. 코드·checkpoint·raw 결과를 공식 공개했다.
- 정확도 확장 후보: SUTrack. 최신 경량/최고 정확도/인기도를 한 순위로 섞지 않는다.

UETrack 공식 README의 RGB 결과: B/S/T LaSOT AUC 69.2/66.9/63.4,
CPU FPS 56/68/83. 이 수치는 저자 환경의 값이며 Raspberry Pi/Hailo 실측이 아니다.
LaSOT AUC를 우리 VisDrone 5개 대상 AUC 77.1 등과 직접 비교하면 안 된다.
공식 README: https://github.com/kangben258/UETrack
논문: https://arxiv.org/abs/2603.01412

OSTrack의 실제 ONNX 내보내기 및 DFC 검증 결과는 [호환성 실험](ostrack_hailo_probe.md)에 기록한다.
UETrack/SUTrack의 학습·양자화·HEF 실행은 아직 수행하지 않았다.
Hailo-8에서 지원되는 연산·고정 입력·여러 입력의 구성, 메모리, 양자화 오차를
실제로 검증해야 한다. CPU 또는 다른 GPU에서 빠르다는 이유로 Hailo 호환을 가정하지 않는다.

## 속도 측정과 실패 판단

[동일 조건 시스템 벤치마크](../sot/SYSTEM_BENCHMARK.md)를 준비했다.
장치 접근 복구 후 네 방법의 실제 Hailo 포함 속도 측정을 수행한다.
[실측 결과](../sot/results/system_comparison.md)와 캐시 로직 검사를 구분한다.

현재 실패 기준은 native 성공/실패와 유효한 박스 여부뿐이다.
CSRT가 잘못된 다른 객체를 따라가면서 성공을 반환하는 drift는 이 기준으로 못 잡는다.
운용 시 GT는 사용할 수 없으므로 외형 일치·이동/크기 일관성·순차 검출 확인 등
독립 신호의 오탐/미탐을 검증해야 한다. 이를 이미 해결했다고 주장하지 않는다.
