# Hailo DFC로 모델 경량화하기 (ONNX → HEF)

Hailo-8에서 모델을 돌리려면 **HEF**(Hailo Executable Format) 파일이 있어야 합니다.
HEF는 Hailo가 배포하는 **Dataflow Compiler(DFC)** 로 만듭니다. 이 과정에서 FP32 모델이 **INT8로 양자화**되고 Hailo-8 칩 구조에 맞게 배치됩니다. 이 프로젝트에서 말하는 "경량화"가 바로 이 과정입니다.

> 이 문서는 아래 [참고 자료](#참고-자료)에 있는 Hailo 공식 문서와 커뮤니티 글을 바탕으로 정리했습니다.
> DFC 버전마다 옵션 이름이 조금씩 다르니 실제로 쓰기 전에 `hailo <명령> --help`로 확인하세요.

---

## 전체 흐름

```
 [x86 PC, Ubuntu]                                                   [Raspberry Pi 5 + Hailo-8]
 PyTorch(.pt) ──export──▶ ONNX ──parse──▶ HAR ──optimize──▶ HAR(INT8) ──compile──▶ HEF ──▶ HailoRT 추론
                                    │              │                                       + CPU 후처리(NMS)
                                    │              └─ 캘리브레이션 이미지 (≥1024장 권장)          + 트래커
                                    └─ 끝 노드 지정 (NMS·decode는 잘라냄)
```

| 단계 | 하는 일 | 결과물 |
|---|---|---|
| Parse | ONNX를 Hailo 내부 표현으로 변환 | `*.har` |
| Optimize | 캘리브레이션 데이터로 **FP32 → INT8 양자화** | `*_optimized.har` |
| Compile | Hailo-8 자원 배치 및 마이크로코드 생성 | `*.hef` |
| Run | Pi에서 HailoRT로 HEF 실행 | 추론 결과 |

---

## 0. 준비물

- **x86_64 Linux PC** (Ubuntu 권장). DFC는 Raspberry Pi(ARM)에서 돌아가지 않습니다.
- **DFC 버전:** Hailo-8/8L은 **DFC v3.x**, Hailo-10H/15는 v5.x를 씁니다. ([Ultralytics 문서](https://docs.ultralytics.com/integrations/hailo))
- [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)에 가입한 뒤 DFC wheel이나 **AI Software Suite(Docker)** 를 받습니다.
- **GPU가 있으면 좋습니다.** GPU 없이 캘리브레이션 이미지도 적으면 최적화 레벨이 0으로 떨어지고 압축도 건너뜁니다. ([커뮤니티](https://community.hailo.ai/t/custom-model-for-raspberry-pi-5-fails-on-hailo-dataflow-compiler/2704))
- Pi 쪽에는 **HailoRT**(DFC와 버전을 맞춘 것)가 설치돼 있어야 합니다.

---

## 1. ONNX로 내보내기

- 입력 크기를 고정합니다. 이 프로젝트는 `1×3×640×640`입니다.
- **박스 decode와 NMS는 Hailo에서 돌리지 않고 잘라냅니다.** 동적 연산이라 NPU와 잘 맞지 않아서 CPU에서 처리합니다.

이 프로젝트에서 쓴 RemDet HEF의 출력 구조는 다음과 같습니다(`hailortcli parse-hef`로 확인).

```
Input  input_layer1  UINT8, NHWC(640x640x3)
Output conv36  (80x80x10)   ← stride 8  클래스 점수 (VisDrone 10클래스)
Output conv37  (80x80x64)   ← stride 8  박스 회귀 (DFL, 4×16)
Output conv47  (40x40x10)   ← stride 16
Output conv48  (40x40x64)
Output conv57  (20x20x10)   ← stride 32
Output conv58  (20x20x64)
```

즉, 헤드의 마지막 conv 6개를 **끝 노드(end nodes)** 로 잡았고, DFL decode, sigmoid, NMS는 Pi CPU에서 numpy로 처리합니다.

---

## 2. Parse (ONNX → HAR)

CLI 방식:

```bash
hailo parser onnx remdet_s.onnx --hw-arch hailo8 \
    --end-node-names <cls_conv_p3> <reg_conv_p3> <cls_conv_p4> <reg_conv_p4> <cls_conv_p5> <reg_conv_p5>
```

Python(`ClientRunner`) 방식 ([Macnica 예제](https://www.macnica.co.jp/en/business/semiconductor/articles/hailo/144843/) 참고):

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(
    "remdet_s.onnx", "remdet_s_visdrone",
    end_node_names=[...],                      # Netron으로 ONNX를 열어 헤드 마지막 conv 이름 확인
    net_input_shapes={"images": [1, 3, 640, 640]},
)
runner.save_har("remdet_s_parsed.har")
```

> 끝 노드 이름은 [Netron](https://netron.app)으로 ONNX를 열어 보면 가장 쉽게 찾을 수 있습니다.

---

## 3. Optimize

### 3-1. 캘리브레이션 데이터

- **실제 사용 환경과 비슷한 이미지**를 씁니다. 여기서는 VisDrone train 이미지입니다.
- **1024장 이상**을 권장합니다. ([Ultralytics](https://docs.ultralytics.com/integrations/hailo), [Model Zoo 커스텀 예제](https://doleron.substack.com/p/deploying-a-custom-yolov8-model-on))
- 전처리는 **추론할 때와 완전히 같게** 합니다(640 리사이즈/레터박스, RGB 순서). 이게 다르면 정확도가 크게 떨어집니다.
- 랜덤 데이터(`--use-random-calib-set`)는 컴파일이 되는지 확인하는 용도로만 쓰세요. 정확도를 보장하지 않습니다.

```python
import numpy as np, cv2, glob
files = sorted(glob.glob("visdrone/train/images/*.jpg"))[:1024]
calib = np.stack([cv2.resize(cv2.imread(f)[..., ::-1], (640, 640)) for f in files]).astype(np.float32)  # NHWC, 0~255
np.save("calib_640.npy", calib)
```

### 3-2. 모델 스크립트(.alls)

양자화 방식을 지정하는 설정 파일입니다.

```
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])   # /255 를 칩 안에서 처리
model_optimization_flavor(optimization_level=2, compression_level=0)
```

| 옵션 | 의미 |
|---|---|
| `normalization` | 입력 정규화를 HEF 안에 포함시킵니다. Pi에서는 UINT8 이미지를 그대로 넣으면 됩니다. |
| `optimization_level` | 0~4. 높을수록 양자화 오차를 더 보정하지만(GPU 필요) 시간이 오래 걸립니다. |
| `compression_level` | 0~5. 높을수록 일부 레이어를 4bit로 압축해 더 빨라지지만 정확도가 떨어질 수 있습니다. |

### 3-3. 실행

```bash
hailo optimize remdet_s_parsed.har --hw-arch hailo8 \
    --calib-set-path calib_640.npy --model-script remdet_s.alls
```

```python
runner.load_model_script(open("remdet_s.alls").read())
runner.optimize(calib)                      # calib: (N, 640, 640, 3)
runner.save_har("remdet_s_optimized.har")
```

> 컴파일하기 전에 **에뮬레이터(quantized mode)** 로 정확도를 먼저 확인하세요. 하드웨어 없이 INT8 결과를 미리 볼 수 있습니다. ([RidgeRun 위키](https://developer.ridgerun.com/wiki/index.php/Hailo/Hailo-8/AI_Software_and_Tools/Hailo_Dataflow_Compiler))

---

## 4. Compile (HAR → HEF)

```bash
hailo compiler remdet_s_optimized.har --hw-arch hailo8
```

```python
hef = runner.compile()
open("remdet_s_visdrone.hef", "wb").write(hef)
```

- 모델이 칩 하나에 다 들어가지 않으면 **multi-context**(여러 번 나눠 실행)로 컴파일됩니다. 큰 모델일수록 context가 많아지고 그만큼 느려집니다.
- `hailo profiler <har>`로 예상 FPS와 자원 사용량을 HTML 리포트로 볼 수 있습니다.

---

## 5. Raspberry Pi에서 확인

```bash
hailortcli parse-hef remdet_s_visdrone.hef     # 입출력 구조, context 수
hailortcli benchmark remdet_s_visdrone.hef     # 순수 NPU 속도
```

이 프로젝트의 결과(Raspberry Pi 5 + Hailo-8, 640×640 입력):

| 모델 | Context 수 | 추론 FPS | 실제 FPS (사진 읽기~후처리) |
|---|---|---|---|
| RemDet-Tiny | 2 | 103 | 75 |
| RemDet-S | 2 | 56 | 46 |
| RemDet-M | 4 | 26 | 23 |
| RemDet-L | 5 | 13 | 12.5 |

---

## 정확도가 떨어질 때 확인할 것

1. **전처리 불일치:** 캘리브레이션, 추론, 학습할 때의 리사이즈 방식과 RGB/BGR 순서가 같은지 확인합니다.
2. **캘리브레이션 데이터:** 장수가 충분한지, 실제 드론 영상과 비슷한 이미지인지 확인합니다.
3. **최적화 레벨:** GPU를 쓰는 환경에서 `optimization_level`을 2 이상으로 올립니다.
4. **압축:** `compression_level`을 0으로 내립니다.
5. **끝 노드 위치:** sigmoid나 decode 같은 연산이 HEF 안에 남아 있으면 양자화 오차가 커질 수 있습니다. 출력 직전에서 자르세요.
6. 커뮤니티에도 비슷한 사례가 많습니다. ([정확도 저하 사례](https://community.hailo.ai/t/the-accuracy-of-the-custom-trained-yolov8m-model-is-low/4859))

---

## 참고 자료

**공식**
- [Hailo Developer Zone: 문서 (DFC User Guide)](https://hailo.ai/developer-zone/documentation/), [소프트웨어 다운로드](https://hailo.ai/developer-zone/software-downloads/) (로그인 필요)
- [hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo): [GETTING_STARTED](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst) (`hailomz parse/optimize/compile/eval`)
- [hailo-ai/hailort](https://github.com/hailo-ai/hailort): Pi 쪽 런타임
- [hailo-ai/hailo-rpi5-examples](https://github.com/hailo-ai/hailo-rpi5-examples): Raspberry Pi 5 예제
- [Hailo Community](https://community.hailo.ai/)

**튜토리얼 및 정리 글**
- [Ultralytics: Hailo Integration](https://docs.ultralytics.com/integrations/hailo): 전체 파이프라인, DFC 버전, 캘리브레이션 1024장 권장
- [Macnica: DFC Python(ClientRunner) 예제](https://www.macnica.co.jp/en/business/semiconductor/articles/hailo/144843/)
- [RidgeRun: Hailo Dataflow Compiler](https://developer.ridgerun.com/wiki/index.php/Hailo/Hailo-8/AI_Software_and_Tools/Hailo_Dataflow_Compiler): 단계별 개념, 에뮬레이터
- [커스텀 YOLOv8을 Raspberry Pi 5 + Hailo에 배포하기](https://doleron.substack.com/p/deploying-a-custom-yolov8-model-on)
- [Docker로 커스텀 YOLOv8 HEF 컴파일하기 (커뮤니티)](https://community.hailo.ai/t/guide-compiling-custom-yolov8-hefs-on-non-ubuntu-linux-using-docker-no-shell-script/19095)
- [ONNX → HEF CLI 변환 (커뮤니티)](https://community.hailo.ai/t/convert-onxx-to-hef-using-the-cli/9106)
