"""Report matched target SOT/locked-ID MOT; separate historical hardware speed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(rows):
    return {k:float(np.mean([r['metrics'][k] for r in rows]))
            for k in ('auc','precision20','mean_iou','success50')}


def table(rows):
    lines = ['| Method | Targets | AUC | Precision@20 | IoU>0.5 | Update FPS | Read+track FPS |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in dict.fromkeys(r['method'] for r in rows):
        group = [r for r in rows if r['method']==method]
        s = summarize(group)
        timing = [r['timing'] for r in group if 'wall_seconds' in r['timing']]
        if timing:
            fps = sum(t['updates'] for t in timing)/sum(t['update_seconds'] for t in timing)
            end = sum(t['frames'] for t in timing)/sum(t['wall_seconds'] for t in timing)
            speed = f'{fps:.1f} | {end:.1f}'
        else:
            speed = 'historical below | historical below'
        lines.append(f"| {method} | {len(group)} | {100*s['auc']:.1f} | {100*s['precision20']:.1f} | {100*s['success50']:.1f} | {speed} |")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    doc = json.loads(a.input.read_text())
    rows = doc['results']
    lines = ['# 원본/ROI SOT와 기존 MOT 비교', '',
        'VisDrone MOT val에서 영상별 가장 긴 pedestrian/car 궤적 하나씩: 사람 7, 차량 6.',
        f"대상별 최대 {doc['settings']['frames']}프레임. OpenCV {doc['opencv']}, 스레드 1개.",
        'MOT GT를 객체별로 분리한 자체 SOT 프로토콜이며 공식 UAV123 결과가 아님.',
        '설치된 GOT-10k toolkit 0.1.3의 ExperimentUAV123._calc_metrics/_calc_curves로 계산.',
        '21개 IoU 임계값 Success curve의 평균=AUC, Precision@20px. 모든 점수는 0–100.',
        '각 대상 점수를 동일 가중 평균. 초기화 프레임 제외, GT 없는 프레임 제외.', '',
        '## 전체 대상', '']+table(rows)
    lines += ['', '## 초기화 조건 차이', '',
        'SOT는 첫 GT 박스를 제공받는다. MOT는 저장된 RemDet-S 검출+추적 결과에서',
        '첫 선택 GT 박스를 고정하고 최대 5프레임 동안 그 박스와 IoU>=0.5인 트랙 생성을 기다린 뒤 ID를 고정한다.',
        '생성 전 프레임은 누락으로 채점. 5프레임 내 선택 실패는 전 구간 0점. 이후 GT로 ID를 갈아타지 않는다.',
        '선택 대기 중에도 첫 박스만 사용하고 미래 GT 위치는 매칭에 쓰지 않는다.',
        '따라서 전체 대상 표에는 검출/트랙 생성 실패의 영향도 포함된다.',
        'MOT는 영상 시작부터 실행된 결과이며 나중에 등장한 대상에서는 이전 문맥을 가진다.',
        '원 저자 SOT/MOT 알고리즘의 공정한 공식 순위로 해석하면 안 된다.', '']
    for method in ('Holdfast-S','ByteTrack-S'):
        chosen = [r for r in rows if r['method']==method]
        good = {(r['sequence'],tuple(r['target'])) for r in chosen if r['timing']['initial_detection_matched']}
        lines += [f"### {method}: 초기 선택 성공 {len(good)}/{len(chosen)}", '',
                  '해당 MOT와 초기 선택이 성공한 동일 대상들에서만 SOT 비교:', '']
        subset = [r for r in rows if (r['sequence'],tuple(r['target'])) in good and
                  (not r['method'].endswith('-S') or r['method']==method)]
        if subset:
            lines += table(subset)
        lines.append('')
    lines += ['## 클래스별 정확도', '']
    for label in ('pedestrian','car'):
        lines += [f'### {label}', '']+table([r for r in rows if r['target'][0]==label])+['']
    lines += ['## 속도 해석', '',
        'SOT: 이번 실행의 실제 측정. 영상 decode + crop + 추적 + ROI 이동에 따른 재초기화 포함.',
        '검출, 실패 복구, 화면 표시, 송출, 실시간 카메라 디코딩은 제외.',
        'ROI도 원본 JPEG 전체를 decode하므로 crop만으로 읽기 비용은 줄지 않음.',
        'MOT 정확도는 저장된 결과에서 계산하므로 이번 평가 파일 읽기 속도를 추적 FPS로 쓰지 않음.',
        '다음은 기존 e2e_pipeline_all.json의 val 전체 2,846프레임 실측이며 이번 실험과 입력 전처리/시점이 다름:', '',
        '| 기존 파이프라인 | 전체 FPS | 추적 단계 FPS | 프레임 수 |',
        '|---|---:|---:|---:|']
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/'mot/e2e_pipeline_all.json').read_text())
    for name,key in [('RemDet-S + Holdfast','s__holdfast__pipe'),('RemDet-S + ByteTrack','s__rf-bytetrack__pipe')]:
        if key in old:
            v = old[key]
            lines.append(f"| {name} | {v['e2e_fps']:.1f} | {v['stage_fps']['track']:.1f} | {v['frames']} |")
    lines += ['', '## ROI 구현 조건', '',
        f"원본 영상에서 {doc['settings']['roi']}x{doc['settings']['roi']} crop. resize 없음.",
        '프레임 밖은 회색 padding. 첫 GT 박스 중심으로 시작하며 이후 직전 예측으로만 갱신.',
        '예측 중심이 ROI 중심에서 폭의 20% 이상 벗어나면 ROI를 이동한다.',
        'OpenCV 내부 좌표계는 직접 이동할 수 없으므로 이동 시 예측 박스로 재초기화한다.',
        '재초기화 횟수와 비용을 기록한다. 이 결과는 단순 crop 효과와 필터 재학습 효과가 함께 포함된다.',
        '추적 실패 시 GT 복구 없음. 별도 복구/재감지 파이프라인의 평가는 아직 포함하지 않음.', '',
        '## 대상별 점수', '',
        '| Sequence | Target | Method | AUC | Precision@20 | ROI resets |',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['target'][0]}:{r['target'][1]} | {r['method']} | {r['metrics']['auc']*100:.1f} | {r['metrics']['precision20']*100:.1f} | {r['timing'].get('prediction_based_reinitializations','-')} |")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n')
    summary = {k:v for k,v in doc.items() if k not in ('results','manifest')}
    summary['results'] = [{k:v for k,v in r.items()
                           if k not in ('predictions','ground_truth','valid')}
                          for r in rows]
    for r in summary['results']:
        r['timing'] = {k:v for k,v in r['timing'].items() if k!='initial_box'}
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for method in dict.fromkeys(r['method'] for r in rows):
        group = [r for r in rows if r['method']==method]
        sc = np.mean([r['metrics']['success_curve'] for r in group],axis=0)
        pc = np.mean([r['metrics']['precision_curve'] for r in group],axis=0)
        axes[0].plot(np.linspace(0,1,21),sc,label=f'{method} [{sc.mean()*100:.1f}]')
        axes[1].plot(np.arange(51),pc,label=f'{method} [{pc[20]*100:.1f}]')
    for ax,title,xlabel in zip(axes,['Success (macro AUC)','Precision (P@20px)'],['IoU threshold','Center error threshold (original pixels)']):
        ax.set(title=title,xlabel=xlabel,ylabel='Fraction of scored frames',ylim=(0,1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('VisDrone-derived SOT diagnostic, 13 selected targets (not official UAV123)')
    fig.tight_layout()
    fig.savefig(a.output.with_suffix('.png'),dpi=160)
    plt.close(fig)
    common = {(r['sequence'],tuple(r['target'])) for r in rows
              if r['method']=='Holdfast-S' and r['timing']['initial_detection_matched']}
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for method in dict.fromkeys(r['method'] for r in rows):
        group = [r for r in rows if r['method']==method and (r['sequence'],tuple(r['target'])) in common]
        if not group:
            continue
        sc = np.mean([r['metrics']['success_curve'] for r in group],axis=0)
        pc = np.mean([r['metrics']['precision_curve'] for r in group],axis=0)
        axes[0].plot(np.linspace(0,1,21),sc,label=f'{method} [{sc.mean()*100:.1f}]')
        axes[1].plot(np.arange(51),pc,label=f'{method} [{pc[20]*100:.1f}]')
    for ax,title,xlabel in zip(axes,['Success (macro AUC)','Precision (P@20px)'],['IoU threshold','Center error threshold (original pixels)']):
        ax.set(title=title,xlabel=xlabel,ylabel='Fraction of scored frames',ylim=(0,1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'Conditional diagnostic: {len(common)} targets acquired by Holdfast (selection bias; not official benchmark)')
    fig.tight_layout()
    fig.savefig(a.output.with_name(a.output.stem+'_acquired.png'),dpi=160)
    plt.close(fig)
    print('\n'.join(lines[:19]))


if __name__=='__main__':
    main()
