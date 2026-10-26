"""One serial file-input pipeline for SOT and detector+MOT, including recovery.

Real FPS requires /dev/hailo0. Cached detections are an explicit logic smoke test,
never a replacement for hardware E2E speed. No external ROI tracking input.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import clip


def xyxy(box):
    x,y,w,h=box
    return np.array([x,y,x+w,y+h],float)


def best_match(boxes,reference,minimum=.1):
    if len(boxes)==0:
        return None
    ref=xyxy(reference)
    inter=np.maximum(0,np.minimum(boxes[:,2:],ref[2:])-np.maximum(boxes[:,:2],ref[:2])).prod(1)
    area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    iou=inter/np.maximum(1e-9,area+reference[2]*reference[3]-inter)
    i=int(iou.argmax())
    return i if iou[i]>=minimum else None


class Engine:
    def __init__(self,method,recovery_frames=5,selection_box=None):
        self.method=method
        self.is_mot=method in ('bytetrack','holdfast')
        self.state='DETECT'
        self.box=self.template=self.track_id=self.tracker=None
        self.selected_class=None
        self.retries=0
        self.limit=recovery_frames
        self.selection=selection_box
        if self.is_mot:
            # OpenCV is imported first so all methods use the same contrib build.
            sys.path.insert(0,str(ROOT/'mot'))
            from run_trackers import JointTracker,step
            self.joint=JointTracker('rf-bytetrack' if method=='bytetrack' else 'holdfast')
            self.step=step

    def initialize(self,img,box):
        self.tracker=getattr(cv2,f'Tracker{self.method}_create')()
        self.tracker.init(img,box)
        x,y,w,h=box
        self.template=cv2.cvtColor(img[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
        if self.template.std()<1:
            self.template=None

    def update(self,img,dets):
        previous=self.state
        found=None
        reason=None
        if self.is_mot:
            self.step.frame=img
            out=self.joint.update(dets)
            active=np.concatenate(list(out.values()),axis=0)
            if self.track_id is not None:
                selected=active[active[:,4]==self.track_id]
                if len(selected):
                    found=clip((selected[0,0],selected[0,1],selected[0,2]-selected[0,0],selected[0,3]-selected[0,1]),img)
        if previous=='DETECT':
            candidates=dets
            if self.selected_class is not None:
                candidates=candidates[candidates[:,5]==self.selected_class]
            reference=self.box if self.box is not None else self.selection
            i=best_match(candidates[:,:4],reference) if reference is not None else (int(candidates[:,4].argmax()) if len(candidates) else None)
            if i is not None:
                d=candidates[i]
                candidate=clip((d[0],d[1],d[2]-d[0],d[3]-d[1]),img)
                if candidate is not None:
                    if self.selection is None and self.box is None:
                        self.selection=candidate
                    self.selected_class=int(d[5])
                    if self.is_mot:
                        j=best_match(active[:,:4],candidate,minimum=.5)
                        if j is not None:
                            self.track_id=int(active[j,4])
                            b=active[j,:4]
                            found=clip((b[0],b[1],b[2]-b[0],b[3]-b[1]),img)
                    else:
                        found=candidate
                        self.initialize(img,found)
        elif not self.is_mot and previous=='TRACK':
            ok,b=self.tracker.update(img)
            found=clip(b,img) if ok else None
            if found is None:
                reason='tracker_failed_or_invalid_box'
        elif not self.is_mot and previous=='RECOVER':
            self.retries+=1
            if self.template is not None:
                x,y,w,h=self.box
                radius=min(max(img.shape[:2]),max(64,2*max(w,h))*self.retries)
                x1,y1=max(0,x-radius),max(0,y-radius)
                x2,y2=min(img.shape[1],x+w+radius),min(img.shape[0],y+h+radius)
                region=cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
                if region.shape[0]>=self.template.shape[0] and region.shape[1]>=self.template.shape[1]:
                    match=cv2.matchTemplate(region,self.template,cv2.TM_CCOEFF_NORMED)
                    _,score,_,location=cv2.minMaxLoc(match)
                    if score>=.85:
                        found=clip((x1+location[0],y1+location[1],self.template.shape[1],self.template.shape[0]),img)
                        if found is not None:
                            self.initialize(img,found)
        if found is not None:
            self.box=found
            self.state='TRACK'
            self.retries=0
        elif previous!='DETECT':
            if self.is_mot:
                self.retries+=1
            elif previous=='TRACK':
                self.retries=0
            self.state='DETECT' if self.retries>=self.limit else 'RECOVER'
            if self.state=='DETECT':
                self.track_id=None
            reason=reason or ('selected_id_missing' if self.is_mot else 'local_recovery_unconfirmed')
        return dict(mode=previous,next_mode=self.state,box=found,reason=reason)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,help='Same original JPEG directory for all methods')
    p.add_argument('--hef',type=Path,default=ROOT/'models/remdet_s_visdrone.hef')
    p.add_argument('--methods',nargs='+',default=['KCF','CSRT','bytetrack','holdfast'],choices=['KCF','CSRT','bytetrack','holdfast'])
    p.add_argument('--frames',type=int,default=200)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--recovery-frames',type=int,default=5)
    p.add_argument('--selection-box',nargs=4,type=float,help='Optional initial target rectangle x y width height; used only for selection')
    p.add_argument('--mot-site-packages',type=Path,help='Optional existing MOT dependencies; OpenCV remains the already imported build')
    p.add_argument('--cached-detections',type=Path,help='LOGIC TEST ONLY: existing mot/dets/s/<seq>.npy')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.frames,a.trials,a.recovery_frames)<1:
        p.error('frames, trials and recovery-frames must be positive')
    if a.selection_box is not None and min(a.selection_box[2:])<=0:
        p.error('invalid selection box')
    if not a.cached_detections and not Path('/dev/hailo0').exists():
        p.exit(2,'Hailo device /dev/hailo0 is not visible. No hardware speed result was produced. Run this command in the Pi host terminal.\n')
    if a.mot_site_packages:
        sys.path.append(str(a.mot_site_packages))
    files=sorted(a.source.glob('*.jpg'))[:a.frames]
    if not files:
        p.error('no JPEG inputs found')
    if any(m in ('KCF','CSRT') and not hasattr(cv2,f'Tracker{m}_create') for m in a.methods):
        p.error('OpenCV contrib with TrackerKCF/CSRT is required')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    results=[]
    with ExitStack() as stack:
        if a.cached_detections:
            cached=np.load(a.cached_detections)
            def detect(img,frame):
                return cached[cached[:,0]==frame,1:7].copy()
        else:
            import remdet_bench as rb
            rb.CONF=.05
            from hailo_platform import HEF,VDevice,ConfigureParams,HailoStreamInterface,InputVStreamParams,OutputVStreamParams,FormatType,InferVStreams
            device=stack.enter_context(VDevice())
            hef=HEF(str(a.hef));outputs=rb.get_outs(hef)
            ng=device.configure(hef,ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe))[0]
            pipe=stack.enter_context(InferVStreams(ng,InputVStreamParams.make(ng,format_type=FormatType.UINT8),OutputVStreamParams.make(ng,format_type=FormatType.FLOAT32)))
            stack.enter_context(ng.activate(ng.create_params()))
            name=hef.get_input_vstream_infos()[0].name
            def detect(img,frame):
                x,s=rb.preprocess(img)
                b,score,cls=rb.postprocess(pipe.infer({name:x}),outputs,s)
                return np.c_[b,score,cls]
            sample=cv2.imread(str(files[0]))
            for _ in range(5):
                detect(sample,0)
        for trial in range(a.trials):
            order=a.methods[trial%len(a.methods):]+a.methods[:trial%len(a.methods)]
            for method in order:
                engine=Engine(method,a.recovery_frames,a.selection_box)
                events=[]
                wall=time.perf_counter()
                for path in files:
                    start=time.perf_counter()
                    img=cv2.imread(str(path))
                    if img is None:
                        raise ValueError(f'Cannot decode {path}')
                    read_end=time.perf_counter()
                    needed=engine.is_mot or engine.state=='DETECT'
                    dets=detect(img,int(path.stem)) if needed else np.empty((0,6))
                    dets=dets[np.isin(dets[:,5],[0,1,3,4,5,8])]
                    det_end=time.perf_counter()
                    event=engine.update(img,dets)
                    end=time.perf_counter()
                    event.update(frame=int(path.stem),detector_called=needed,read_ms=(read_end-start)*1000,
                        detection_ms=(det_end-read_end)*1000,tracking_recovery_ms=(end-det_end)*1000,
                        read_to_prediction_ms=(end-start)*1000)
                    events.append(event)
                elapsed=time.perf_counter()-wall
                latency=np.array([r['read_to_prediction_ms'] for r in events])
                result=dict(method=method,trial=trial+1,frames=len(events),wall_seconds=elapsed,
                    hardware_e2e_fps=len(events)/elapsed if not a.cached_detections else None,
                    read_to_prediction_p50_ms=float(np.percentile(latency,50)),read_to_prediction_p95_ms=float(np.percentile(latency,95)),
                    detection_calls=sum(e['detector_called'] for e in events),
                    recovery_frames=sum(e['mode']=='RECOVER' for e in events),missing_outputs=sum(e['box'] is None for e in events),events=events)
                results.append(result)
                report=dict(backend='cached_logic_only' if a.cached_detections else 'Hailo-8',opencv=cv2.__version__,
                    protocol='same serial full-JPEG input; no GUI/render/transmit; read-to-prediction latency is not camera capture latency; SOT detection only in DETECT; MOT detection every frame',
                    recovery_policy='SOT local template NCC>=0.85 up to configured attempts, then class+last-box gated redetection; MOT retains selected ID until timeout; identity after reacquisition is not guaranteed; native tracker success can still drift',
                    results=results)
                a.output.parent.mkdir(parents=True,exist_ok=True)
                a.output.write_text(json.dumps(report,indent=2))
                print(method,'trial',trial+1,'hardware FPS',result['hardware_e2e_fps'],
                      'p95',round(result['read_to_prediction_p95_ms'],1),'ms','missing',result['missing_outputs'],flush=True)


if __name__=='__main__':
    main()
