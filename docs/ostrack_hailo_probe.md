# OSTrack → Hailo-8 호환성 실험

2026-10-26, 공식 OSTrack 저장소/README가 연결한 ModelScope checkpoint 사용.
체크포인트는 OSTrack-384 (template 192, search 384). pretrained state strict load 통과.
원본 가중치 SHA256: `8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894`.

| 단계 | 결과 |
|---|---|
| 원본 CE 유지 FP32 ONNX | checker 및 PyTorch↔ONNXRuntime 수치 일치 통과 |
| 원본 ONNX DFC 3.34.0 parser | 실패 |
| 원본과 수치 동등한 onnxsim | parser 실패, ReduceMean layout 경로 |
| CE 비활성화 변형 | ONNX 수치 검사 및 Hailo-8 parser 통과, HAR 생성 |
| 초기 낮은 설정 호환성 확인 | 양자화 통과, HEF 컴파일 conv2 reshape 할당 실패 |
| 사용자 지정 최종 조건 | 서로 다른 유효 이미지 ≥1024, optimization_level=4, compression_level=0, GPU 필수 |
| CE 비활성화 변형의 추적 정확도 / Pi 속도 | 미검증 |

CE 비활성화는 원본 모델 동작을 바꾸므로 원본 SOTA 정확도를 주장할 수 없다.
낮은 설정의 초기 결과는 최종 양자화 결과로 사용하지 않는다. 이후 스크립트는
1024장 미만 또는 GPU 미검출 시 오류로 중단하며 CPU 최적화로 낮추지 않는다.
로그에서 실제 최적화 레벨 및 AdaRound 실행 여부도 확인해야 한다.

기존 보정 JPEG 1024장 중 1022장만 confidence 0.25 이상 대상이 검출되었다.
원본 보정 폴더를 변경하지 않고 별도 폴더에 기존 이미지와 공개
`Voxel51/VisDrone2019-DET` train 행 0–15 이미지를 합쳐 다시 선택한다.
파일 내용 SHA256으로 중복을 제거하고 실제 사용 파일/박스를 manifest에 남긴다.
기존 보정 이미지의 train/val 원출처는 확인되지 않았으므로 이것을 최종 평가용
누수 검증을 마친 데이터로 간주하면 안 된다.

각 이미지의 RemDet 검출 박스로 factor 2/4 template/search를 만들며 OSTrack의
crop/padding 및 RGB mean/std를 적용한다. 같은 이미지의 두 crop으로 만든 보정쌍이며
실제 시간 변화·가림·추적 오차의 분포를 검증한 temporal 보정쌍은 아니다.
1024장/최고 레벨 조건 충족과 좋은 보정 데이터의 대표성은 별도 검증 사항이다.

코드: `dfc/probe_ostrack.py`, `dfc/probe_hailo_parse.py`, `dfc/probe_ostrack_quantize.py`.
원시 모델·보정 이미지·HAR·HEF는 공개 저장소에 업로드하지 않는다.

최적화 레벨에 대한 Hailo 문서:
https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/OPTIMIZATION.rst

HEF가 생성되더라도 DFC 서버 HailoRT 4.24와 Pi HailoRT 4.23의 호환성,
실제 Hailo 로드, 정확도 저하, template/search 전체 전후처리 포함 속도를 검증해야 한다.

실행 조건 보충: 데이터 1024쌍을 전달하는 것만으로는 SDK 내부 기본값 때문에
64쌍만 쓰는 단계가 있어, `model_optimization_config(calibration, batch_size=1,
calibset_size=1024)`를 명시했다. 실제 로그의 `Using dataset with 1024 entries for
calibration` 및 GPU 사용을 확인했다. AdaRound 완료나 HEF 생성의 의미는 아니다.

## 최종 레벨 4 결과와 컴파일 우회

1024개 서로 다른 이미지, optimization_level=4, compression_level=0으로
양자화가 통과했다. 소요 시간은 20,930초(약 5시간 49분). AdaRound 92/92 완료.
그 결과 HAR의 최대 성능 컴파일도 18초 후 conv2 reshape 할당 오류로 실패했다.
HEF는 아직 생성되지 않았다.

`dfc/probe_ostrack_compile.py`와 `dfc/ostrack_compile/*.alls`는 기존 양자화
가중치를 그대로 사용하는 컴파일 전용 시도다. 버퍼 지정만으로는 같은 오류가
발생했으며, search 패치의 공간 배치를 명시하는 시도를 진행한다. 이 설정들은
성공이 검증된 배포 설정이 아니며 HEF 생성·하드웨어 수치 검사 전까지 실험용이다.

공식 ViT ALLS 및 Hailo SDK 형식 변환 명령을 참조했다.

## ViT 전용 설정 재시도와 저장공간 조치

기존 HAR의 수동 공간 배치로 conv2 오류를 우회했으나, 이후 첫 Attention
`matmul1`의 Titan 분할 오류가 확인됐다. 기존 graph의 `zp_comp_weights` 방식 대신
공식 ViT 설정의 `zp_comp_block`, 덧셈 fusion 분리, negative exponent rank=0을
적용한 레벨 4 재양자화를 시작했다. 원본 CE-disabled ONNX/가중치는 유지한다.

첫 재시도는 AdaRound 캐시 최대 추정 181.19GB에 비해 디스크 여유가
106.57GiB여서 중단됐다. 데이터 수나 레벨은 줄이지 않았다. SDK가 지원하는
`cache_compression=enabled`는 `numpy.savez_compressed` 기반의 무손실 중간 캐시
압축이다. 모델 압축 `compression_level=0`과는 별개이며, 양자화 정밀도를 낮추지 않는다.

임시 optimizer work dir를 태스크 전용 tmpfs로 옮기고, 디스크·메모리 여유가
부족해지면 해당 child job만 중지하는 `dfc/run_storage_guard.py`를 붙였다.
`dfc/cleanup_optimizer_cache.py`는 작업 종료 후 전용 임시 캐시만 해제한다.
최종 HAR/HEF와 보고서는 영구 작업 폴더에 남긴다. 수정한 실행에서
1024개 데이터로 AdaRound 단계 진입을 확인했다. 최종 HEF 성공은 아직 미확인.

HEF 성공 후에는 Pi의 HailoRT 형식 검사와 `dfc/smoke_ostrack_hef.py`로
실제 장치 실행을 확인한다. 이 검사는 정적인 입력쌍의 finite 출력 확인이며
추적 정확도나 실시간 카메라 성능 검증이 아니다. 장치가 사용 중이면 기존
프로그램을 중지하지 않고 비는 시점을 기다린다.
