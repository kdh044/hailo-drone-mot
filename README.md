<div align="center">

# hailo-drone-mot

**Raspberry Pi 5 + Hailo-8 위에서 돌아가는 실시간 드론 항공영상 다중 객체 추적(MOT)**

![Raspberry Pi 5](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?logo=raspberrypi&logoColor=white)
![Hailo-8](https://img.shields.io/badge/NPU-Hailo--8-0A66C2)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-VisDrone2019--MOT-2E7D32)

<img src="docs/assets/demo.webp" width="100%" alt="RemDet-S + ByteTrack 추적 데모">

<sub>RemDet-S (Hailo-8) + ByteTrack, VisDrone `uav0000137_00458_v` · [원본 영상](docs/assets/demo.mp4)</sub>

</div>

---

## 개요

드론에 실을 수 있는 작은 엣지 보드에서 **항공영상 속 사람과 차량을 실시간(30 FPS 이상)으로 검출·추적**하는 것이 목표입니다.

| | |
|---|---|
| **검출기** | [RemDet](https://github.com/HZAI-ZJNU/RemDet) Tiny / S / M / L — VisDrone 학습, Hailo-8 INT8 HEF로 변환 ([DFC 가이드](docs/hailo_dfc_guide.md)) |
| **트래커** | 범용 13종 + 드론 전용 6종, 같은 조건에서 비교 |
| **평가** | VisDrone2019-MOT val 7개 영상 — HOTA / IDF1, Pi에서 잰 E2E FPS |

```
사진 읽기 ─▶ 640 리사이즈 ─▶ Hailo-8: RemDet ─▶ CPU: 후처리(NMS) ─▶ CPU: 트래커 ─▶ ID 박스
└──────── 스레드 1 ────────┘  └──────────── 스레드 2 ────────────┘  └─ 스레드 3 ─┘
```

---

## 결과

### 검출기 속도

| 모델 | 추론 FPS | 사진 읽기~후처리 FPS |
|---|---:|---:|
| RemDet-Tiny | 103 | **75** |
| RemDet-S | 56 | **46** |
| RemDet-M | 26 | 23 |
| RemDet-L | 13 | 12.5 |

### 트래커 비교

사람·차를 트래커 하나로 함께 추적하고, 세 단계를 스레드로 동시에 돌린 E2E FPS입니다 (영상 읽기~트래커 전부 포함).

<table>
<tr><th>RemDet-S</th><th>RemDet-Tiny</th></tr>
<tr><td>

| 트래커 | HOTA | 사람 | 차량 | FPS |
|---|---:|---:|---:|---:|
| **Holdfast** | **47.2** | 39.6 | 54.5 | **32.4** |
| SFTrack | 45.3 | 37.3 | 53.0 | 8.4 |
| BoT-SORT + CMC | 44.9 | 35.5 | 53.6 | 19.7 |
| C-BIoU | 44.6 | 35.5 | 53.1 | 37.0 |
| ByteTrack | 44.5 | 35.7 | 52.9 | 36.2 |
| McByte | 43.6 | 34.3 | 52.2 | 38.0 |

</td><td>

| 트래커 | HOTA | 사람 | 차량 | FPS |
|---|---:|---:|---:|---:|
| **Holdfast** | **44.3** | 33.7 | 53.8 | **37.8** |
| SFTrack | 42.1 | 31.2 | 51.9 | 9.8 |
| ByteTrack | 41.5 | 30.9 | 50.9 | 53.6 |
| ByteTrack (boxmot) | 40.8 | 31.2 | 49.6 | 38.8 |
| C-BIoU | 40.7 | 28.5 | 51.4 | 51.4 |
| BoT-SORT + CMC | 40.7 | 28.0 | 51.6 | 25.9 |

</td></tr>
</table>

<sub>HOTA = 사람+차량 통합. ByteTrack = roboflow 구현. 19개 전체와 IDF1·차례대로 실행 FPS: [`mot/pipeline_tables.md`](mot/pipeline_tables.md)</sub>

- **Holdfast가 S·Tiny 모두 HOTA 1위이면서 30 FPS를 넘습니다.**
- 단계를 차례대로 돌리면 Holdfast는 S 20.4 / Tiny 25.9 FPS입니다. 동시 실행으로 트래커 대기 시간이 사라져 30 FPS를 넘었습니다.
- SFTrack, BoT-SORT + CMC는 트래커 자체가 30 FPS보다 느립니다. C-BIoU, ByteTrack은 트래커는 빠르지만 사진 읽기·Hailo 단계에서 막힙니다.

### 동시 실행과 자원 사용 (RemDet-S + Holdfast)

<img src="docs/assets/pipe_timeline_s.png" width="100%" alt="차례대로 vs 동시 실행 타임라인">

| | 차례대로 (20.4 FPS) | 동시 실행 (32.4 FPS) |
|---|---:|---:|
| CPU (4코어 = 100%) | 19% | 39% |
| Hailo-8 NPU | 33% | 62% |
| GPU | 0% | 0% |

<sub>동시 실행 CPU 39% = 사진 읽기 13% + Hailo 호출·후처리 4% + 트래커 19% + 기타 4%. 메모리 3.0 GB, 최고 70°C. 막대 그림: [`docs/assets/pipe_resources.png`](docs/assets/pipe_resources.png)</sub>

<details>
<summary>정성 비교: ByteTrack(위) vs Holdfast(아래)</summary>

<img src="docs/assets/bytetrack_vs_holdfast.png" alt="ByteTrack vs Holdfast">

</details>

<details>
<summary>RemDet-S 검출 예시</summary>

<img src="docs/assets/remdet_s_det.jpg" alt="RemDet-S detection">

</details>

---

## 구조

```
.
├── remdet_bench.py, remdet_eval.py    RemDet 속도·mAP 측정 (Hailo)
├── remdet_live.py, remdet_track.py    실시간 검출·추적 데모
├── docs/hailo_dfc_guide.md            ONNX → HEF 변환 가이드
└── mot/
    ├── common.py                      경로, 영상, 클래스 그룹(사람/차량)
    ├── dump_dets.py                   검출 결과 캐시 (Hailo)
    ├── run_trackers.py                트래커 일괄 실행
    ├── evaluate.py                    HOTA/IDF1 평가 (TrackEval)
    ├── e2e_pipeline_all.py            E2E FPS·자원 측정 (동시 실행, Hailo)
    ├── ext_trackers.py, ext/          드론 전용 트래커 어댑터·구현
    └── pipeline_tables.md             전체 결과 표
```

## 실행

```bash
# 준비: HailoRT 설치, VisDrone2019-MOT val 다운로드, HEF를 models/ 에 배치
python mot/dump_dets.py                                                        # 1) 검출 결과 캐시
python mot/run_trackers.py --dets s tiny --trackers holdfast-joint cbiou-joint # 2) 트래커 실행 (사람·차 함께)
python mot/make_all_group.py && python mot/evaluate.py                         # 3) HOTA 평가
python mot/e2e_pipeline_all.py --model s --trackers holdfast cbiou             # 4) E2E FPS·자원 측정
python mot/make_pipeline_table.py                                              # 5) 표 만들기
```

**환경:** Raspberry Pi 5 (8GB), Hailo-8 (M.2), HailoRT, Python 3, [boxmot](https://github.com/mikel-brostrom/boxmot), [trackers](https://github.com/roboflow/trackers), [TrackEval](https://github.com/JonathonLuiten/TrackEval)

> 학습 가중치, HEF, 데이터셋은 저장소에 포함하지 않습니다.

---

## 참고

- 데이터셋: [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (비상업적 연구용)
- 검출기: [RemDet](https://github.com/HZAI-ZJNU/RemDet)
- `mot/ext/` 아래 트래커들은 각 원저자의 공개 구현을 어댑터용으로 수정한 것이며, 원래 저장소의 라이선스를 따릅니다.
