"""Run hardware benchmark on the five previously acquired targets and score offline."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from compare_modes import evaluate
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=json.loads((ROOT/'sot/results/shared_init.raw.json').read_text())
    original=json.loads((ROOT/'sot/results/modes_13targets.json').read_text())
    manifests={(r['sequence'],tuple(r['target'])):r['frames'] for r in original['manifest']}
    results=[]
    for target in [r for r in source['results'] if r['method']=='Holdfast-S']:
        seq=target['sequence'];label,tid=target['target']
        ids=manifests[(seq,tuple(target['target']))]
        start=target['timing']['acquired_frame']
        ids=ids[ids.index(start):]
        path=ROOT/f'sot/results/system_{seq}_{label}_{tid}.raw.json'
        cmd=[sys.executable,str(ROOT/'sot/benchmark_system.py'),str(Path.home()/'datasets/visdrone/mot-val-hf/sequences'/seq),
            '--mot-site-packages',str(Path.home()/'mot-venv/lib/python3.13/site-packages'),
            '--start-frame',str(start),'--frames',str(len(ids)),'--trials','3',
            '--selection-box',*[str(v) for v in target['predictions'][0]],'--output',str(path)]
        subprocess.run(cmd,check=True,cwd=ROOT)
        truth=np.asarray(target['ground_truth'],float)
        valid=np.asarray(target['valid'],bool);valid[0]=False
        for r in json.loads(path.read_text())['results']:
            events=r['events']; assert [e['frame'] for e in events]==ids
            pred=np.asarray([e['box'] if e['box'] is not None else [0,0,0,0] for e in events],float)
            metrics=evaluate(pred,truth,valid)
            clean={k:v for k,v in r.items() if k!='events'}
            clean.update(sequence=seq,target=target['target'],metrics=metrics,
                transitions={f'{a}->{b}':sum(e['mode']==a and e['next_mode']==b for e in events) for a,b in [('DETECT','TRACK'),('TRACK','RECOVER'),('RECOVER','TRACK'),('RECOVER','DETECT')]})
            # SOT must never call detector while tracking or locally recovering.
            if r['method'] in ('KCF','CSRT'):
                assert all(not e['detector_called'] or e['mode']=='DETECT' for e in events)
            results.append(clean)
        report=dict(protocol='Five previously acquired VisDrone targets; original full JPEG input; same detector and serial pipeline; three rotated trials; detector-derived selection box; GT used offline only; no ROI input; initial frame excluded from scores',
            coverage='5/13 previously acquired targets; conditional diagnostic, not official UAV123 results',
            recovery='Native failure/invalid box then local NCC recovery; drift with valid boxes can be missed; recovered identity not guaranteed',results=results)
        (ROOT/'sot/results/system_suite.summary.json').write_text(json.dumps(report,indent=2))
        print('SCORED',seq,label,tid,flush=True)
if __name__=='__main__':main()
