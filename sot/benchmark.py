"""Open-loop, first-frame-only SOT baseline on existing VisDrone annotations.

This is a VisDrone-derived diagnostic, NOT an official UAV123 benchmark.
No GT-based resets. Failed updates count as zero overlap and are timed separately.
"""
import argparse
import collections
import json
from pathlib import Path
import pickle
import time

import cv2
import numpy as np


def iou(a, b):
    x, y = np.maximum(a[:2], b[:2])
    xx, yy = np.minimum(np.array(a[:2]) + a[2:], np.array(b[:2]) + b[2:])
    inter = max(0, xx-x) * max(0, yy-y)
    return inter / max(1e-9, a[2]*a[3] + b[2]*b[3] - inter)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=int, default=100)
    p.add_argument('--sequences', type=int, default=1)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--labels', nargs='+', default=['pedestrian', 'car'])
    p.add_argument('--width', type=int, default=1280,
                   help='Aspect-preserving full-frame resize; 0 means original')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--images', type=Path, default=Path.home()/'datasets/visdrone/mot-val-hf/sequences')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.frames, a.sequences, a.threads) < 1 or a.width < 0:
        p.error('frames, sequences, threads must be positive; width must be nonnegative')
    cv2.setNumThreads(a.threads)
    gt = pickle.load(open(a.root/'mot/gt.pkl', 'rb'))
    results = []
    for seq, obj in sorted(gt.items())[:a.sequences]:
        tracks = collections.defaultdict(dict)
        for fr, tid, x1, y1, x2, y2, label in obj['objs']:
            if label in ('car', 'pedestrian'):
                tracks[(label, tid)][int(fr)] = (x1, y1, x2-x1, y2-y1)
        for label in a.labels:
            candidates = [k for k in tracks if k[0] == label]
            if not candidates:
                continue
            run_target(a, seq, tracks, max(candidates, key=lambda k: len(tracks[k])), results)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(opencv=cv2.__version__, threads=a.threads,
        width=a.width, protocol='VisDrone-derived SOT; first GT init only; no resets; decode+resize included in E2E',
        results=results), indent=2))


def run_target(a, seq, tracks, key, results):
    # One longest trajectory per requested class and video, selected once.
    truth = tracks[key]
    first = min(truth)
    paths = sorted((a.images/seq).glob('*.jpg'))
    paths = [q for q in paths if first <= int(q.stem) <= max(truth)][:a.frames]
    if not paths:
        raise ValueError(f'No frames for {seq}')
    for name in ('KCF', 'CSRT'):
        tracker = getattr(cv2, f'Tracker{name}_create')()
        durations, success_times, overlaps, errors = [], [], [], []
        wall = time.perf_counter()
        cpu = time.process_time()
        init_ms = None
        for idx, path in enumerate(paths):
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f'Cannot read {path}')
            h, w = img.shape[:2]
            if a.width:
                img = cv2.resize(img, (a.width, round(h*a.width/w)))
            sx, sy = img.shape[1]/w, img.shape[0]/h
            fr = int(path.stem)
            if idx == 0:
                b = truth[first]
                box = tuple(round(v*s) for v,s in zip(b,(sx,sy,sx,sy)))
                x,y,bw,bh = box
                box = (max(0,x), max(0,y), min(bw,img.shape[1]-max(0,x)), min(bh,img.shape[0]-max(0,y)))
                if min(box[2:]) < 2:
                    raise ValueError('Initial target too small after resize')
                t = time.perf_counter()
                tracker.init(img, box)
                init_ms = (time.perf_counter()-t)*1000
                continue
            t = time.perf_counter()
            ok, box = tracker.update(img)
            dt = time.perf_counter()-t
            durations.append(dt)
            if ok:
                success_times.append(dt)
            if fr in truth:
                pred = np.array(box)/[sx,sy,sx,sy]
                overlaps.append(iou(pred, truth[fr]) if ok else 0.)
                target = np.array(truth[fr])
                errors.append(float(np.linalg.norm(pred[:2]+pred[2:]/2-target[:2]-target[2:]/2)) if ok else float('inf'))
        elapsed = time.perf_counter()-wall
        cpu_s = time.process_time()-cpu
        r = dict(sequence=seq, target=list(key), tracker=name,
                 frames=len(paths), scored_updates=len(overlaps), init_ms=init_ms,
                 update_fps=len(durations)/sum(durations) if durations else None,
                 successful_update_fps=len(success_times)/sum(success_times) if success_times else None,
                 p95_update_ms=float(np.percentile(durations,95)*1000) if durations else None,
                 e2e_fps=len(paths)/elapsed, cpu_one_core_pct=cpu_s/elapsed*100,
                 reported_success_fraction=len(success_times)/len(durations) if durations else None,
                 mean_iou=float(np.mean(overlaps)) if overlaps else None,
                 success_iou_50=float(np.mean(np.array(overlaps)>.5)) if overlaps else None,
                 precision_20px_original=float(np.mean(np.array(errors)<=20)) if errors else None)
        results.append(r)
        print(json.dumps(r), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(opencv=cv2.__version__, threads=a.threads,
        width=a.width, protocol='VisDrone-derived SOT; first GT init only; no resets; decode+resize included in E2E',
        results=results), indent=2))


if __name__ == '__main__':
    main()
