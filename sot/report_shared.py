"""Summarize the detection-initialized SOT comparison without exporting GT."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_modes import table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    doc = json.loads(a.input.read_text())
    rows = doc['results']
    coverage = doc['acquisition_coverage']
    lines = ['# 같은 검출 박스로 시작한 SOT / MOT 비교', '',
        f"초기 선택에 성공한 {coverage['selected']}/{coverage['candidates']}개 대상의 조건부 비교.",
        'VisDrone val에서 고른 기존 13개 대상 중 Holdfast가 초기 선택에 성공한 5개를 사용.',
        '각 방법은 같은 프레임의 같은 Holdfast 출력 박스로 시작. OpenCV는 좌표를 정수 반올림.',
        'KCF/CSRT에는 GT 초기 박스를 제공하지 않음. 이후 GT는 점수 계산에만 사용.',
        'Holdfast는 기존 RemDet-S 검출+MOT 전체 결과에서 선택한 ID를 계속 유지.',
        'SOT는 매 프레임 영상으로 추적하며 검출·재감지·복구는 사용하지 않음.',
        '초기화 프레임 제외. GOT-10k toolkit 0.1.3의 UAV123 metric/curve 코드 사용.',
        '객체별 점수 동일 가중 평균. 공식 UAV123 결과가 아니며 전체 객체의 성능을 대표하지 않음.', '',
        '## 정확도와 이번 SOT 실측 속도', '']+table(rows)
    lines += ['', '## 속도 비교의 범위', '',
        'SOT FPS: 이번 실행의 원본 JPEG 읽기 + crop + 추적 + ROI 재초기화 비용 포함.',
        '검출·복구·화면 표시·영상 송출 제외. OpenCV 스레드 1개.',
        '실패를 빠르게 반환하는 update 호출도 FPS에 포함되므로 속도만으로 실용성을 판단하지 않음.',
        'ROI는 512x512 원본 crop, resize 없음. 중심 이동 시 예측 박스로만 재초기화.',
        'Holdfast FPS: 이번에는 Hailo 장치 접근 불가로 재측정하지 못함.',
        '기존 val 전체 2,846프레임 측정: RemDet-S + Holdfast 32.4 FPS (검출 포함).',
        '기존 MOT는 reduced decode 및 동시 파이프라인을 사용하므로 이번 SOT FPS와 직접적인 동일 조건 속도 순위는 아님.', '',
        '## 해석 제한', '',
        '검출 가능한 5개 대상으로 제한한 결과이므로 나머지 8개의 초기 선택 실패는 별도로 고려해야 함.',
        'Holdfast는 모든 사람/차량을 함께 추적한 결과이고 SOT는 대상 하나씩 독립 실행.',
        'ROI 변경 시 필터 재초기화 효과가 섞여 있어 crop만의 효과로 해석하면 안 됨.',
        '작은 표본·최대 200프레임·단일 실행의 예비 비교. 실시간 카메라 및 실패 복구까지 검증한 결과는 아님.', '',
        '## 대상별 결과', '',
        '| Sequence | Target | Method | AUC | Precision@20 | ROI resets |',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['target'][0]}:{r['target'][1]} | {r['method']} | {r['metrics']['auc']*100:.1f} | {r['metrics']['precision20']*100:.1f} | {r['timing'].get('prediction_based_reinitializations','-')} |")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n')
    summary = {k:v for k,v in doc.items() if k!='results'}
    summary['results'] = []
    for r in rows:
        clean = {k:v for k,v in r.items() if k not in ('predictions','ground_truth','valid')}
        clean['timing'] = {k:v for k,v in r['timing'].items() if k!='initial_box'}
        summary['results'].append(clean)
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for method in dict.fromkeys(r['method'] for r in rows):
        group = [r for r in rows if r['method']==method]
        sc = np.mean([r['metrics']['success_curve'] for r in group],axis=0)
        pc = np.mean([r['metrics']['precision_curve'] for r in group],axis=0)
        axes[0].plot(np.linspace(0,1,21),sc,label=f'{method} [{sc.mean()*100:.1f}]')
        axes[1].plot(np.arange(51),pc,label=f'{method} [{pc[20]*100:.1f}]')
    for ax,title,xlabel in zip(axes,['Success (macro AUC)','Precision (P@20px)'],['IoU threshold','Center error threshold (pixels)']):
        ax.set(title=title,xlabel=xlabel,ylabel='Fraction of scored frames',ylim=(0,1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Shared detection initialization: 5 acquired targets; conditional diagnostic')
    fig.tight_layout()
    fig.savefig(a.output.with_suffix('.png'),dpi=160)
    plt.close(fig)
    print('\n'.join(lines[:23]))


if __name__=='__main__':
    main()
