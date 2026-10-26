"""Sequential detection -> single-target tracking -> local recovery -> detection.

Full-frame baseline. No detector calls during TRACK or RECOVER. Auto-selection
chooses highest confidence; it does not guarantee identity after full redetection.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def frames(source):
    path = Path(source)
    if path.is_dir():
        for p in sorted(path.glob('*.jpg')):
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f'Cannot decode {p}')
            yield img
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise ValueError(f'Cannot open {source}')
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield img
        finally:
            cap.release()


def clip(box, img):
    x,y,w,h = map(float, box)
    x1,y1 = max(0,round(x)),max(0,round(y))
    x2,y2 = min(img.shape[1],round(x+w)),min(img.shape[0],round(y+h))
    return (x1,y1,x2-x1,y2-y1) if x2-x1>=2 and y2-y1>=2 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('--hef', type=Path, required=True)
    p.add_argument('--tracker', choices=['KCF','CSRT'], default='KCF')
    p.add_argument('--select', choices=['manual','highest-confidence'], default='manual')
    p.add_argument('--recovery-frames', type=int, default=5)
    p.add_argument('--recovery-threshold', type=float, default=.85)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.recovery_frames < 1 or not 0 <= a.recovery_threshold <= 1:
        p.error('invalid recovery settings')
    import remdet_bench as rb
    from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
        InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    state, tracker, box, template, attempts = 'DETECT', None, None, None, 0
    with ExitStack() as stack:
        out = stack.enter_context(a.output.open('w'))
        device = stack.enter_context(VDevice())
        hef = HEF(str(a.hef))
        ng = device.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        pipe = stack.enter_context(InferVStreams(ng,
            InputVStreamParams.make(ng, format_type=FormatType.UINT8),
            OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)))
        stack.enter_context(ng.activate(ng.create_params()))
        name = hef.get_input_vstream_infos()[0].name
        outputs = rb.get_outs(hef)

        def initialize(img, selected):
            t = getattr(cv2, f'Tracker{a.tracker}_create')()
            t.init(img, selected)
            return t

        for fr,img in enumerate(frames(a.source),1):
            start = time.perf_counter()
            mode, found, detector_called = state, None, False
            if state == 'DETECT':
                x,s = rb.preprocess(img)
                boxes,scores,classes = rb.postprocess(pipe.infer({name:x}),outputs,s)
                detector_called = True
                candidates = [i for i,c in enumerate(classes) if c in (0,1,3,4,5,8)]
                if candidates:
                    if a.select == 'manual':
                        preview = img.copy()
                        for i in candidates:
                            x1,y1,x2,y2 = map(int,boxes[i])
                            cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),1)
                        chosen = cv2.selectROI('Select target; Enter confirms', preview, False)
                        cv2.destroyWindow('Select target; Enter confirms')
                        found = clip(chosen,img)
                    else:
                        i = max(candidates,key=lambda i:scores[i])
                        x1,y1,x2,y2 = boxes[i]
                        found = clip((x1,y1,x2-x1,y2-y1),img)
            elif state == 'TRACK':
                ok, predicted = tracker.update(img)
                found = clip(predicted,img) if ok else None
                if found is None:
                    state, attempts = 'RECOVER',0
            else:
                # Local image-template recovery is a heuristic, not an identity model.
                attempts += 1
                x,y,w,h = box
                radius = max(64,2*max(w,h))*attempts
                x1,y1 = max(0,x-radius),max(0,y-radius)
                x2,y2 = min(img.shape[1],x+w+radius),min(img.shape[0],y+h+radius)
                search = cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
                if search.shape[0]>=template.shape[0] and search.shape[1]>=template.shape[1]:
                    response = cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
                    _,score,_,location = cv2.minMaxLoc(response)
                    if score>=a.recovery_threshold:
                        found = clip((x1+location[0],y1+location[1],template.shape[1],template.shape[0]),img)
                if found is None and attempts>=a.recovery_frames:
                    state = 'DETECT'
            if found is not None:
                if mode != 'TRACK':
                    tracker = initialize(img,found)
                    x,y,w,h = found
                    template = cv2.cvtColor(img[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
                    # Constant templates produce meaningless normalized matches.
                    if template.std()<1:
                        template = None
                box,state = found,'TRACK'
            if state == 'RECOVER' and template is None:
                state = 'DETECT'
            out.write(json.dumps(dict(frame=fr,mode=mode,next_mode=state,
                detector_called=detector_called,box=found,
                processing_ms=(time.perf_counter()-start)*1000))+'\n')
            out.flush()
            if a.max_frames and fr>=a.max_frames:
                break


if __name__ == '__main__':
    main()
