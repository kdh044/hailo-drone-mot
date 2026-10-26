"""Summarize measured Hailo system runs, including offline recovery diagnostics."""
import json
import hashlib
from pathlib import Path
import numpy as np
from benchmark import iou
ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'sot/results/system_suite.summary.json'
    doc=json.loads(path.read_text())
    reference=json.loads((ROOT/'sot/results/shared_init.raw.json').read_text())
    targets={(r['sequence'],tuple(r['target'])):r for r in reference['results'] if r['method']=='Holdfast-S'}
    latencies={}
    for r in doc['results']:
        seq=r['sequence'];label,tid=r['target']
        raw=json.loads((ROOT/f'sot/results/system_{seq}_{label}_{tid}.raw.json').read_text())
        events=next(x['events'] for x in raw['results'] if x['method']==r['method'] and x['trial']==r['trial'])
        latencies.setdefault(r['method'],[]).extend(e['read_to_prediction_ms'] for e in events)
        target=targets[(seq,tuple(r['target']))]
        overlaps=[iou(e['box'],g) if e['box'] is not None else 0 for e,g in zip(events,target['ground_truth'])]
        valid=target['valid']
        r['severe_error_with_output_frames']=sum(i>0 and v and e['box'] is not None and bool(ov<.1) for i,(e,ov,v) in enumerate(zip(events,overlaps,valid)))
        recovered=[(e,ov) for e,ov,v in zip(events,overlaps,valid) if v and e['mode']=='RECOVER' and e['next_mode']=='TRACK']
        r['recovery_outputs']=len(recovered)
        r['recovery_outputs_iou50']=sum(bool(ov>=.5) for e,ov in recovered)
        r['detection_mean_ms']=float(np.mean([e['detection_ms'] for e in events if e['detector_called']])) if r['detection_calls'] else 0
    aggregate=[]
    for method in ('KCF','CSRT','bytetrack','holdfast'):
        rows=[r for r in doc['results'] if r['method']==method]
        if not rows:continue
        fps=[r['hardware_e2e_fps'] for r in rows]
        aggregate.append(dict(method=method,targets=len({(r['sequence'],tuple(r['target'])) for r in rows}),trials_per_target=3,
            pooled_fps=sum(r['frames'] for r in rows)/sum(r['wall_seconds'] for r in rows),
            run_fps_min=min(fps),run_fps_max=max(fps),
            pooled_latency_p50_ms=float(np.percentile(latencies[method],50)),pooled_latency_p95_ms=float(np.percentile(latencies[method],95)),
            auc=float(np.mean([r['metrics']['auc'] for r in rows])),precision20=float(np.mean([r['metrics']['precision20'] for r in rows])),
            missing_fraction=sum(r['missing_outputs'] for r in rows)/sum(r['frames'] for r in rows),
            cpu_one_core_pct=float(np.mean([r['cpu_one_core_pct'] for r in rows])),
            recovery_outputs=sum(r['recovery_outputs'] for r in rows),recovery_outputs_iou50=sum(r['recovery_outputs_iou50'] for r in rows),
            severe_error_with_output_frames=sum(r['severe_error_with_output_frames'] for r in rows)))
    doc['aggregate']=aggregate
    doc['detector']=dict(name='RemDet-S',hef_sha256=hashlib.sha256((ROOT/'models/remdet_s_visdrone.hef').read_bytes()).hexdigest(),confidence=.05,input_size=[640,640],warmup_inferences=5)
    doc['environment']=json.loads((ROOT/'sot/results/system_environment.summary.json').read_text())
    doc['opencv']=raw['opencv']
    path.write_text(json.dumps(doc,indent=2))
    lines=['# Hailo 포함 시스템 실측','',doc['protocol'],doc['coverage'],'',
        '| 방법 | 전체 FPS | 실행별 FPS 범위 | AUC | P@20 | 박스 누락 | CPU (1코어=100%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['method']} | {r['pooled_fps']:.1f} | {r['run_fps_min']:.1f}–{r['run_fps_max']:.1f} | {100*r['auc']:.1f} | {100*r['precision20']:.1f} | {100*r['missing_fraction']:.1f}% | {r['cpu_one_core_pct']:.1f}% |")
    lines+=['','JPEG 읽기·전처리·Hailo 검출·추적·복구 포함. 화면 표시/전송/카메라 대기 제외. 원본 입력, ROI 없음. 속도는 총 프레임/총 경과 시간, 정확도는 대상별 동일 가중 평균.',
        '매 시행 방법 순서를 순환. 기존 병렬 MOT FPS와 직접 같은 조건이 아님. ByteTrack/Holdfast는 기존 joint MOT를 매 프레임 실행하고 선택 ID를 평가.',
        'AUC/P@20은 GOT-10k UAV123 evaluator의 곡선 구현을 사용한 VisDrone 진단값. 공식 UAV123 결과가 아님. 실패/복구 중 박스 누락도 0 overlap으로 포함.',
        '', '## 연구원 요청 구현 및 검증', '',
        '- 감지→대상 선택→SOT. SOT의 TRACK/RECOVER 중 검출 호출이 없음을 실제 로그에서 검사.',
        '- native 실패/유효하지 않은 박스→지역 NCC 복구→최대 5회 실패 시 DETECT. 대상 변경 API는 DETECT로 복귀.',
        '- `check_recovery.py`: KCF/CSRT 복구 성공·복구 시간 초과·재감지·대상 변경 상태 전환 통과. 합성 검사이며 실환경 복구 정확도 증명이 아님.',
        '- 다른 물체로 drift하면서 valid box를 반환하는 실패는 아직 검출하지 못함. 재감지 시 동일 identity 보장 없음.',
        '- 실제 UAV 하드웨어 성능, 카메라 지연, NPU 점유율, 장시간 신뢰성은 미검증. CPU 수치는 프로세스 CPU 시간/벽시계 시간.',
        '', '| 방법 | 복구 박스 출력 | 그중 GT IoU≥0.5 | 박스는 있지만 GT IoU<0.1 프레임 |', '|---|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['method']} | {r['recovery_outputs']} | {r['recovery_outputs_iou50']} | {r['severe_error_with_output_frames']} |")
    lines+=['','위 횟수는 3회 반복 합계이며 독립 표본 수가 아님. GT는 사후 평가에만 사용. MOT의 RECOVER→TRACK은 같은 ID 재출현이며 SOT 템플릿 복구와 다름.']
    (ROOT/'sot/results/system_comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]))
if __name__=='__main__':main()
