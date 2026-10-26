"""Compare original-input KCF/CSRT with cached detector + KF (ByteTrack)."""
import json
from pathlib import Path

import numpy as np

from compare_modes import ROOT, cached_mot, evaluate
from report_modes import table


def main():
    source = json.loads((ROOT/'sot/results/shared_init.raw.json').read_text())
    original = json.loads((ROOT/'sot/results/modes_13targets.json').read_text())
    manifest = {(r['sequence'],tuple(r['target'])):r['frames'] for r in original['manifest']}
    rows = [r for r in source['results'] if r['method'] in ('KCF-full','CSRT-full')]
    for h in [r for r in source['results'] if r['method']=='Holdfast-S']:
        seq,label = h['sequence'],h['target'][0]
        ids = manifest[(seq,tuple(h['target']))]
        ids = ids[ids.index(h['timing']['acquired_frame']):]
        paths = [Path(str(i)) for i in ids]
        truth = np.array(h['ground_truth'],float)
        seed = truth.copy()
        seed[0] = h['predictions'][0]
        pred,timing = cached_mot(seq,label,paths,seed,'rf-bytetrack')
        rows.append(dict(sequence=seq,target=h['target'],method='ByteTrack-S',
                         metrics=evaluate(pred,truth,np.array(h['valid'],bool)),timing=timing))
    summary = {k:v for k,v in source.items() if k!='results'}
    summary['results'] = []
    for r in rows:
        clean = {k:v for k,v in r.items() if k not in ('predictions','ground_truth','valid')}
        clean['timing'] = {k:v for k,v in r['timing'].items() if k!='initial_box'}
        summary['results'].append(clean)
    (ROOT/'sot/results/kf_comparison.summary.json').write_text(json.dumps(summary,indent=2))
    print('\n'.join(table(rows)))


if __name__=='__main__':
    main()
