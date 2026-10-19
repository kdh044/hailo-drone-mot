# hailo-drone-mot: Raspberry Pi 5 + Hailo-8 드론 영상 실시간 추적

드론에 실을 수 있는 작은 엣지 보드에서 **항공영상 속 사람과 차량을 실시간(30 FPS 이상)으로 검출하고 추적**하는 프로젝트입니다. 검출은 Hailo-8 NPU에서 RemDet으로, 추적은 Pi의 CPU에서 합니다.

> - 데이터셋: [VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset) val 7개 영상 (비상업적 연구용)
> - 검출기: [RemDet](https://github.com/HZAI-ZJNU/RemDet) Tiny / S / M / L, VisDrone 학습 후 INT8 HEF로 변환
> - Hailo 변환 방법은 [Hailo DFC 가이드](docs/hailo_dfc_guide.md)에 따로 정리했습니다.

![추적 데모](docs/assets/demo.webp)

<sub>RemDet-S (Hailo-8) + ByteTrack, VisDrone `uav0000137_00458_v` · [원본 영상](docs/assets/demo.mp4)</sub>

## 한눈에 보기

| 단계 | 어디서 | 하는 일 | 스레드 |
|---|---|---|---|
| 1. 사진 읽기 | CPU | 큰 사진은 작게 풀어 읽고 640×640으로 줄입니다. | 1 |
| 2. 검출 | Hailo-8 | RemDet(INT8)이 박스 후보를 냅니다. | 2 |
| 3. 후처리 | CPU | 박스를 풀고(DFL) 겹친 박스를 지웁니다(NMS). | 2 |
| 4. 추적 | CPU | 트래커가 프레임 사이 박스를 이어 ID를 붙입니다. 사람·차를 트래커 하나로 함께 추적합니다. | 3 |

세 스레드는 동시에 돌아갑니다. 한 프레임을 추적하는 동안 다음 프레임을 검출하고, 그다음 프레임을 읽습니다.

| 결과 | RemDet-S | RemDet-Tiny |
|---|---:|---:|
| 가장 좋은 트래커 | Holdfast | Holdfast |
| 통합 HOTA | **47.2** | **44.3** |
| E2E FPS (읽기~추적 전부) | **32.4** | **37.8** |

---

## 1. 검출기 속도

**이 표는?** RemDet만 Pi + Hailo-8에서 돌린 속도입니다. 추적은 빼고, 사진 읽기부터 후처리까지 포함합니다.

| 모델 | 추론 FPS | 사진 읽기~후처리 FPS |
|---|---:|---:|
| RemDet-Tiny | 103 | **75** |
| RemDet-S | 56 | **46** |
| RemDet-M | 26 | 23 |
| RemDet-L | 13 | 12.5 |

실시간 추적에는 S와 Tiny를 씁니다. M, L은 검출만으로도 30 FPS가 안 됩니다.

---

## 2. 트래커 비교

**이 표는?** 트래커 19종(범용 13 + 드론 전용 6)을 같은 조건으로 비교했습니다. HOTA는 사람+차량 통합 값이고, FPS는 Pi에서 영상 읽기부터 추적까지 전부 포함해 잰 값입니다. 아래는 HOTA 상위 6개만 추렸습니다.

**RemDet-S**

| 트래커 | 종류 | HOTA | 사람 | 차량 | FPS |
|---|---|---:|---:|---:|---:|
| **Holdfast** | 드론 전용 | **47.2** | 39.6 | 54.5 | **32.4** |
| SFTrack | 드론 전용 | 45.3 | 37.3 | 53.0 | 8.4 |
| BoT-SORT + CMC | 범용 | 44.9 | 35.5 | 53.6 | 19.7 |
| C-BIoU | 범용 | 44.6 | 35.5 | 53.1 | 37.0 |
| ByteTrack | 범용 | 44.5 | 35.7 | 52.9 | 36.2 |
| McByte | 범용 | 43.6 | 34.3 | 52.2 | 38.0 |

**RemDet-Tiny**

| 트래커 | 종류 | HOTA | 사람 | 차량 | FPS |
|---|---|---:|---:|---:|---:|
| **Holdfast** | 드론 전용 | **44.3** | 33.7 | 53.8 | **37.8** |
| SFTrack | 드론 전용 | 42.1 | 31.2 | 51.9 | 9.8 |
| ByteTrack | 범용 | 41.5 | 30.9 | 50.9 | 53.6 |
| ByteTrack (boxmot) | 범용 | 40.8 | 31.2 | 49.6 | 38.8 |
| C-BIoU | 범용 | 40.7 | 28.5 | 51.4 | 51.4 |
| BoT-SORT + CMC | 범용 | 40.7 | 28.0 | 51.6 | 25.9 |

**정리**

- **Holdfast가 S·Tiny 모두 HOTA 1위이면서 30 FPS를 넘습니다.**
- SFTrack, BoT-SORT + CMC는 정확도는 좋지만 트래커 자체가 30 FPS보다 느립니다.
- C-BIoU, ByteTrack은 트래커는 빠르지만, 사진 읽기·Hailo 단계에서 막혀 S에서는 37 FPS 안팎입니다.

> ByteTrack은 roboflow 구현입니다. 19종 전체 표와 IDF1은 [`mot/pipeline_tables.md`](mot/pipeline_tables.md)에 있습니다.
> 사람·차를 트래커 하나로 함께 돌린 HOTA는 따로 돌렸을 때와 모든 트래커에서 0.4 이내로 같았습니다.

---

## 3. 동시 실행과 자원 사용

**이 그림은?** RemDet-S + Holdfast에서 1초 동안 각 단계가 언제 일하는지 보여줍니다. 위는 단계를 차례대로 돌린 것이고, 아래는 세 스레드로 동시에 돌린 것입니다. 동시 실행에서는 트래커(초록)가 쉬지 않고 일합니다. 즉 트래커가 전체 속도를 정합니다.

![차례대로 vs 동시 실행](docs/assets/pipe_timeline_s.png)

| | 차례대로 | 동시 실행 |
|---|---:|---:|
| E2E FPS | 20.4 | **32.4** |
| CPU (4코어 = 100%) | 19% | 39% |
| Hailo-8 NPU (추론 중인 시간) | 33% | 62% |
| GPU | 0% | 0% |

> 동시 실행 CPU 39%는 사진 읽기 13% + Hailo 호출·후처리 4% + 트래커 19% + 기타 4%입니다. 메모리 3.0 GB, 최고 온도 70°C입니다. 막대 그림은 [`docs/assets/pipe_resources.png`](docs/assets/pipe_resources.png)에 있습니다.
> GPU(VideoCore)는 쓰지 않습니다. 검출은 Hailo-8, 나머지는 CPU가 합니다.

---

## 4. 직접 돌려보기

**필요한 것:** Raspberry Pi 5 (8GB), Hailo-8 (M.2), HailoRT, Python 3, [boxmot](https://github.com/mikel-brostrom/boxmot), [trackers](https://github.com/roboflow/trackers), [TrackEval](https://github.com/JonathonLuiten/TrackEval). VisDrone2019-MOT val을 받고 HEF를 `models/`에 둡니다.

```bash
python mot/dump_dets.py                                                           # 1) 검출 결과 저장 (Hailo)
python mot/run_trackers.py --dets s tiny --trackers holdfast-joint cbiou-joint    # 2) 트래커 실행
python mot/make_all_group.py && python mot/evaluate.py                            # 3) HOTA 채점
python mot/e2e_pipeline_all.py --model s --trackers holdfast cbiou                # 4) E2E FPS·자원 측정 (Hailo)
python mot/make_pipeline_table.py                                                 # 5) 결과 표 만들기
```

> 트래커 이름 끝의 `-joint`는 사람·차를 트래커 하나로 함께 추적한다는 뜻입니다.
> 학습 가중치, HEF, 데이터셋은 저장소에 포함하지 않습니다.

**파일 구조**

| 파일 | 하는 일 |
|---|---|
| `remdet_bench.py`, `remdet_eval.py` | RemDet 속도·mAP 측정 (Hailo) |
| `remdet_live.py`, `remdet_track.py` | 실시간 검출·추적 데모 |
| `docs/hailo_dfc_guide.md` | ONNX → HEF 변환 가이드 |
| `mot/common.py` | 경로, 영상 목록, 클래스 그룹(사람/차량) |
| `mot/dump_dets.py` | 검출 결과 저장 |
| `mot/run_trackers.py` | 트래커 일괄 실행 |
| `mot/evaluate.py` | HOTA / IDF1 채점 (TrackEval) |
| `mot/e2e_pipeline_all.py` | E2E FPS·자원 측정 (동시 실행) |
| `mot/ext_trackers.py`, `mot/ext/` | 드론 전용 트래커 어댑터와 구현 |
| `mot/pipeline_tables.md` | 트래커 19종 전체 결과 |

---

## 참고

- `mot/ext/` 아래 트래커는 각 원저자의 공개 구현을 어댑터용으로 고친 것이며, 원래 저장소의 라이선스를 따릅니다.
- 정성 비교 그림: [ByteTrack vs Holdfast](docs/assets/bytetrack_vs_holdfast.png), [RemDet-S 검출 예시](docs/assets/remdet_s_det.jpg)
