"""Matched-target full/ROI SOT vs locked-ID cached MOT using got10k curves.

VisDrone-derived protocol, not an official UAV123 benchmark. First GT init only.
MOT identity is acquired within 5 frames against the fixed initial box, then locked.
ROI reinitializes from its own last prediction when shifting crop coordinates.
"""
import argparse
import collections
import json
from pathlib import Path
import pickle
import time

import cv2
import numpy as np
from got10k.experiments import ExperimentUAV123

ROOT = Path(__file__).resolve().parents[1]


def crop(frame, origin, side):
    x,y = origin
    h,w = frame.shape[:2]
    # Origin may be negative: pad to preserve a fixed local coordinate system.
    out = np.full((side,side,3),114,dtype=np.uint8)
    x1,y1,x2,y2 = max(0,x),max(0,y),min(w,x+side),min(h,y+side)
    if x2>x1 and y2>y1:
        out[y1-y:y2-y,x1-x:x2-x] = frame[y1:y2,x1:x2]
    return out


def origin_for(box, side):
    x,y,w,h = box
    return (round(x+w/2-side/2), round(y+h/2-side/2))


def local_box(box, origin):
    x,y,w,h = box
    return (round(x-origin[0]),round(y-origin[1]),max(2,round(w)),max(2,round(h)))


def evaluate(pred, truth, valid):
    # Call the installed upstream toolkit's implementation, not custom curves.
    exp = ExperimentUAV123.__new__(ExperimentUAV123)
    exp.nbins_iou,exp.nbins_ce = 21,51
    anno = np.asarray(truth,dtype=float).copy()
    anno[~valid] = np.nan
    ov,err = exp._calc_metrics(pred,anno)
    missing = (pred[valid,2:]<=0).any(1)
    ov[missing],err[missing] = 0,np.inf
    succ,prec = exp._calc_curves(ov,err)
    return dict(auc=float(succ.mean()),precision20=float(prec[20]),
                mean_iou=float(ov.mean()),success50=float((ov>.5).mean()),
                success_curve=succ.tolist(),precision_curve=prec.tolist(),scored=int(valid.sum()))


def sot(paths, truth, side, name):
    tracker = None
    origin = (0,0)
    box = truth[0].copy()
    preds,updates,decode,initializations = [],[],[],[]
    wall,cpu = time.perf_counter(),time.process_time()
    for idx,path in enumerate(paths):
        t = time.perf_counter()
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f'Cannot decode {path}')
        decode.append(time.perf_counter()-t)
        t = time.perf_counter()
        if idx == 0:
            if side:
                origin = origin_for(box,side)
            view = crop(img,origin,side) if side else img
            tracker = getattr(cv2,f'Tracker{name}_create')()
            tracker.init(view,local_box(box,origin))
            initializations.append(time.perf_counter()-t)
            preds.append(box.copy())
            continue
        view = crop(img,origin,side) if side else img
        ok,b = tracker.update(view)
        if ok and np.isfinite(b).all() and min(b[2:])>0:
            box = np.array(b,dtype=float)
            box[:2] += origin
            preds.append(box.copy())
            # Native OpenCV trackers cannot offset internal coordinates. Explicit
            # reset from prediction is part of this ROI variant, included in time.
            if side and max(abs(box[0]+box[2]/2-origin[0]-side/2),
                            abs(box[1]+box[3]/2-origin[1]-side/2))>side*.2:
                new_origin = origin_for(box,side)
                new_view = crop(img,new_origin,side)
                tracker = getattr(cv2,f'Tracker{name}_create')()
                ti = time.perf_counter()
                tracker.init(new_view,local_box(box,new_origin))
                initializations.append(time.perf_counter()-ti)
                origin = new_origin
        else:
            preds.append(np.zeros(4))
        updates.append(time.perf_counter()-t)
    elapsed = time.perf_counter()-wall
    return np.array(preds),dict(update_fps=len(updates)/sum(updates),
        e2e_fps=len(paths)/elapsed,decode_seconds=sum(decode),wall_seconds=elapsed,
        update_seconds=sum(updates),updates=len(updates),frames=len(paths),
        p95_ms=float(np.percentile(updates,95)*1000),
        cpu_one_core_pct=(time.process_time()-cpu)/elapsed*100,
        prediction_based_reinitializations=len(initializations)-1,
        initial_box=truth[0].tolist())


def cached_mot(seq,label,paths,truth,tracker):
    group = 'person' if label=='pedestrian' else 'vehicle'
    source = ROOT/f'mot/data/trackers/VD{group}-val/s__{tracker}-joint/data/{seq}.txt'
    rows = np.loadtxt(source,delimiter=',',ndmin=2) if source.stat().st_size else np.empty((0,9))
    first = int(paths[0].stem)
    pred = np.zeros((len(paths),4))
    selected = None
    acquired_frame = None
    # Some MOT implementations confirm tracks only on the second frame. Give
    # them 5 frames to attach to the SAME fixed initial selection rectangle.
    # No future GT boxes, global GT matching, or later ID switching is used.
    for frame in range(first,first+5):
        candidates = rows[rows[:,0]==frame]
        if not len(candidates):
            continue
        from got10k.utils.metrics import rect_iou
        ov = rect_iou(candidates[:,2:6],np.tile(truth[0],(len(candidates),1)))
        best = int(ov.argmax())
        if ov[best]>=.5:
            selected = int(candidates[best,1])
            acquired_frame = frame
            lookup = {int(r[0]):r[2:6] for r in rows if int(r[1])==selected and int(r[0])>=frame}
            pred = np.array([lookup.get(int(p.stem),np.zeros(4)) for p in paths])
            break
    return pred,dict(locked_id=selected,initial_detection_matched=selected is not None,
                     acquired_frame=acquired_frame,selection_window=5,
                     source=str(source.relative_to(ROOT)),speed_source='historical full-sequence pipeline benchmark')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int,default=200)
    p.add_argument('--sequences',type=int,default=7)
    p.add_argument('--roi',type=int,default=512)
    p.add_argument('--refresh-mot-only',action='store_true',help='Re-evaluate cached MOT only; retain all SOT runs')
    p.add_argument('--output',type=Path,default=ROOT/'sot/results/modes.json')
    p.add_argument('--images',type=Path,default=Path.home()/'datasets/visdrone/mot-val-hf/sequences')
    a = p.parse_args()
    if a.frames<2 or a.roi<16 or a.sequences<1:
        p.error('invalid frame, ROI, or sequence count')
    cv2.setNumThreads(1)
    gt = pickle.load(open(ROOT/'mot/gt.pkl','rb'))
    manifest,results = [],[]
    if a.output.exists():
        existing = json.loads(a.output.read_text())
        if existing['settings']!={'frames':a.frames,'sequences':a.sequences,'roi':a.roi}:
            p.error('output settings differ; choose a new output')
        results = existing['results']
        if a.refresh_mot_only:
            results = [r for r in results if not r['method'].endswith('-S')]
    done = {(r['sequence'],tuple(r['target']),r['method']) for r in results}
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(settings=dict(frames=a.frames,sequences=a.sequences,roi=a.roi),
            opencv=cv2.__version__,toolkit='got10k 0.1.3 ExperimentUAV123',
            protocol='VisDrone derived; initial box only; MOT fixed-box acquisition within 5 frames; no later GT identity rematching; initial frame excluded',
            manifest=manifest,results=results),indent=2))
        tmp.replace(a.output)
    for seq,obj in sorted(gt.items())[:a.sequences]:
        tracks = collections.defaultdict(dict)
        for fr,tid,x,y,xx,yy,label in obj['objs']:
            if label in ('pedestrian','car'):
                tracks[(label,tid)][int(fr)] = [x,y,xx-x,yy-y]
        for label in ('pedestrian','car'):
            keys = [k for k in tracks if k[0]==label]
            if not keys:
                continue
            key = max(keys,key=lambda k:len(tracks[k]))
            trajectory = tracks[key]
            paths = [p for p in sorted((a.images/seq).glob('*.jpg'))
                     if min(trajectory)<=int(p.stem)<=max(trajectory)][:a.frames]
            truth = np.array([trajectory.get(int(p.stem),[np.nan]*4) for p in paths])
            valid = np.isfinite(truth).all(1)
            valid[0] = False
            manifest.append(dict(sequence=seq,target=list(key),frames=[int(p.stem) for p in paths]))
            for method in ['Holdfast-S','ByteTrack-S','KCF-full','KCF-roi','CSRT-full','CSRT-roi']:
                if (seq,key,method) in done:
                    continue
                if a.refresh_mot_only and not method.endswith('-S'):
                    continue
                if method.endswith('-S'):
                    pred,timing = cached_mot(seq,label,paths,truth,'holdfast' if method.startswith('Holdfast') else 'rf-bytetrack')
                else:
                    pred,timing = sot(paths,truth,a.roi if method.endswith('roi') else 0,method.split('-')[0])
                result = dict(sequence=seq,target=list(key),method=method,
                    metrics=evaluate(pred,truth,valid),timing=timing,
                    predictions=pred.tolist(),ground_truth=np.nan_to_num(truth).tolist(),valid=valid.tolist())
                results.append(result)
                save()
                print(seq,key,method, {k:round(v,3) for k,v in result['metrics'].items() if isinstance(v,float)},
                      'FPS',round(timing.get('e2e_fps',0),1),flush=True)
    save()


if __name__=='__main__':
    main()
