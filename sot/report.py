"""Summarize matched SOT diagnostic runs, weighting accuracy by scored frames."""
import argparse
import json
from pathlib import Path


def summarize(rows):
    n = sum(r['scored_updates'] for r in rows)
    updates = sum(r['frames']-1 for r in rows)
    frames = sum(r['frames'] for r in rows)
    return dict(
        cases=len(rows), scored=n,
        iou=sum(r['mean_iou']*r['scored_updates'] for r in rows)/n,
        success=sum(r['success_iou_50']*r['scored_updates'] for r in rows)/n,
        precision=sum(r['precision_20px_original']*r['scored_updates'] for r in rows)/n,
        update_fps=updates/sum((r['frames']-1)/r['update_fps'] for r in rows),
        e2e_fps=frames/sum(r['frames']/r['e2e_fps'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    doc = json.loads(a.input.read_text())
    rows = doc['results']
    lines = ['# KCF / CSRT: VisDrone 단일 대상 진단', '',
        f"OpenCV {doc['opencv']}, 스레드 {doc['threads']}개, 입력 폭 {doc['width']} (0=원본).", '',
        '각 영상·클래스에서 가장 긴 궤적 하나. 첫 GT 박스로 초기화 후 재초기화 없음.',
        '초기화 프레임은 정확도에서 제외. GT 없는 프레임은 채점에서 제외.',
        '공식 UAV123 평가가 아니며 객체 전체를 대표하는 표본도 아님.', '',
        '| 범위 | 트래커 | 대상 수 | 채점 프레임 | 평균 IoU | IoU>.5 | P@20px | update FPS | E2E FPS |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in [None,'pedestrian','car']:
        for name in ['KCF','CSRT']:
            group = [r for r in rows if r['tracker']==name and (label is None or r['target'][0]==label)]
            if not group:
                continue
            s = summarize(group)
            lines.append(f"| {label or '전체'} | {name} | {s['cases']} | {s['scored']} | {s['iou']:.3f} | {s['success']*100:.1f}% | {s['precision']*100:.1f}% | {s['update_fps']:.1f} | {s['e2e_fps']:.1f} |")
    lines += ['', '정확도는 채점 프레임 수로 가중. FPS는 총 프레임/총 시간으로 집계.',
        'update FPS에 실패 반환 포함. E2E에는 JPEG 읽기·초기화·평가 코드 포함, 검출기 미포함.',
        'P@20px는 원본 좌표의 중심 오차 20픽셀 이하 비율이며 해상도의 영향을 받음.', '',
        '| 영상 | 대상 | 트래커 | 평균 IoU | IoU>.5 | update FPS | E2E FPS |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['target'][0]}:{r['target'][1]} | {r['tracker']} | {r['mean_iou']:.3f} | {r['success_iou_50']*100:.1f}% | {r['update_fps']:.1f} | {r['e2e_fps']:.1f} |")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:19]))


if __name__ == '__main__':
    main()
