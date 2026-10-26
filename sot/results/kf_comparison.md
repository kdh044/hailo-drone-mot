# KCF / CSRT / 검출+KF(ByteTrack) 비교

초기 선택에 성공한 공통 5개 대상. 같은 Holdfast 검출 박스로 선택/초기화.
KCF/CSRT 원본 입력 실측 결과를 재사용하며, ByteTrack은 저장된 RemDet-S 검출+추적 결과에서 초기 박스로 ID를 선택 후 고정.
GOT-10k toolkit의 UAV123 metric 코드 사용. 자체 VisDrone 조건부 평가이며 공식 UAV123 결과는 아님.

| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |
|---|---:|---:|---:|---:|---:|---:|
| KCF-full | 5 | 32.2 | 42.6 | 42.0 | 75.1 | 39.2 |
| CSRT-full | 5 | 56.9 | 76.0 | 68.4 | 14.8 | 12.5 |
| ByteTrack-S | 5 | 77.1 | 97.9 | 91.4 | historical below | historical below |

ByteTrack은 칼만필터 예측 + 검출 박스 연결을 수행하는 KF 기반 추적기이며 KF만 단독 실행한 것은 아님.
KCF/CSRT FPS는 읽기+추적이며 매 프레임 검출 비용 제외. ByteTrack은 기존 검출 포함 36.2 FPS로, 이번 동일 조건 재측정은 아님.
표본 5개, 대상당 최대 200프레임. 검출 선택 실패 8개 제외. 이후 GT ID 재매칭 및 SOT GT 재초기화 없음.
