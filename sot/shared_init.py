"""SOT vs Holdfast with exactly the same detection-derived initial box.

Conditional on Holdfast acquiring the selected target; reports acquisition
coverage separately. MOT outputs are cached, SOT tracking is executed anew.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_modes import ROOT, sot, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'sot/results/modes_13targets.json')
    p.add_argument('--output',type=Path,default=ROOT/'sot/results/shared_init.raw.json')
    p.add_argument('--images',type=Path,default=Path.home()/'datasets/visdrone/mot-val-hf/sequences')
    a = p.parse_args()
    source = json.loads(a.source.read_text())
    selected = [r for r in source['results'] if r['method']=='Holdfast-S' and r['timing']['initial_detection_matched']]
    manifest = {(r['sequence'],tuple(r['target'])):r['frames'] for r in source['manifest']}
    settings = dict(source['settings'],initialization='shared-holdfast-detection')
    rows = []
    if a.output.exists():
        previous = json.loads(a.output.read_text())
        if previous['settings']!=settings:
            p.error('Existing output settings differ')
        rows = previous['results']
    done = {(r['sequence'],tuple(r['target']),r['method']) for r in rows}
    cv2.setNumThreads(1)
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        temp = a.output.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(settings=settings,opencv=cv2.__version__,
            toolkit=source['toolkit'],protocol='same Holdfast output initial box and frame for SOT/MOT; no GT reinitialization; initial frame excluded',
            acquisition_coverage=dict(selected=len(selected),candidates=sum(r['method']=='Holdfast-S' for r in source['results'])),results=rows),indent=2))
        temp.replace(a.output)
    for old in selected:
        seq,key = old['sequence'],tuple(old['target'])
        frame_ids = manifest[(seq,key)]
        start = frame_ids.index(old['timing']['acquired_frame'])
        ids = frame_ids[start:]
        paths = [a.images/seq/f'{f:07d}.jpg' for f in ids]
        truth = np.array(old['ground_truth'],dtype=float)[start:]
        valid = np.array(old['valid'],dtype=bool)[start:]
        valid[0] = False
        holdfast = np.array(old['predictions'],dtype=float)[start:]
        seed_truth = truth.copy()
        seed_truth[0] = holdfast[0]
        for method in ['Holdfast-S','KCF-full','KCF-roi','CSRT-full','CSRT-roi']:
            if (seq,key,method) in done:
                continue
            if method=='Holdfast-S':
                pred,timing = holdfast,old['timing']
            else:
                pred,timing = sot(paths,seed_truth,settings['roi'] if method.endswith('roi') else 0,method.split('-')[0])
            row = dict(sequence=seq,target=list(key),method=method,metrics=evaluate(pred,truth,valid),
                       timing=timing,predictions=pred.tolist(),ground_truth=truth.tolist(),valid=valid.tolist())
            rows.append(row)
            save()
            print(seq,key,method,'AUC',round(row['metrics']['auc']*100,1),'FPS',round(timing.get('e2e_fps',0),1),flush=True)
    save()


if __name__=='__main__':
    main()
