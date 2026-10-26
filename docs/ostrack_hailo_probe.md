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
