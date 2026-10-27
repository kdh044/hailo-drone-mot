"""Load a paired-input OSTrack HEF and check finite outputs on a real image.

This is an execution smoke check, not tracking accuracy or full-system FPS.
Input float values match the calibration pipeline; HailoRT quantizes them.
"""
import argparse
from contextlib import ExitStack
import hashlib,json,math,time
from pathlib import Path
import cv2
import numpy as np
from hailo_platform import (HEF,VDevice,ConfigureParams,HailoStreamInterface,
    InputVStreamParams,OutputVStreamParams,FormatType,InferVStreams)


def crop(image,box,factor,size):
    x,y,w,h=box
    side=math.ceil(math.sqrt(w*h)*factor)
    x1,y1=round(x+w/2-side/2),round(y+h/2-side/2)
    x2,y2=x1+side,y1+side
    lp,rp=max(0,-x1),max(x2-image.shape[1]+1,0)
    tp,bp=max(0,-y1),max(y2-image.shape[0]+1,0)
    patch=image[y1+tp:y2-bp,x1+lp:x2-rp]
    patch=cv2.copyMakeBorder(patch,tp,bp,lp,rp,cv2.BORDER_CONSTANT)
    patch=cv2.resize(patch,(size,size)).astype(np.float32)/255
    return ((patch-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32))[None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hef',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();report=dict(model=a.hef.name,stage='Hailo execution smoke only',status='running')
    try:
        image=cv2.imread(str(a.image))
        if image is None:raise ValueError('Cannot decode smoke input image')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h,w=image.shape[:2]
        box=(w*.45,h*.4,w*.1,h*.15)  # arbitrary image region; no ground truth used
        hef=HEF(str(a.hef))
        inputs={}
        for info in hef.get_input_vstream_infos():
            shape=tuple(info.shape)
            if shape not in ((192,192,3),(384,384,3)):
                raise ValueError(f'Unexpected paired input shape: {shape}')
            inputs[info.name]=crop(image,box,2 if shape[0]==192 else 4,shape[0])
        with ExitStack() as stack:
            device=stack.enter_context(VDevice())
            ng=device.configure(hef,ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe))[0]
            pipeline=stack.enter_context(InferVStreams(ng,
                InputVStreamParams.make(ng,quantized=False,format_type=FormatType.FLOAT32),
                OutputVStreamParams.make(ng,quantized=False,format_type=FormatType.FLOAT32)))
            stack.enter_context(ng.activate(ng.create_params()))
            times=[]
            for i in range(5):
                t=time.perf_counter();out=pipeline.infer(inputs);times.append((time.perf_counter()-t)*1000)
                if not all(np.isfinite(v).all() for v in out.values()):raise ValueError('Nonfinite Hailo outputs')
        report.update(status='passed',hef_sha256=hashlib.sha256(a.hef.read_bytes()).hexdigest(),
            inputs={k:list(v.shape) for k,v in inputs.items()},
            outputs={k:dict(shape=list(v.shape),minimum=float(v.min()),maximum=float(v.max())) for k,v in out.items()},
            inference_api_ms=times,note='Single static pair, no GT, no tracking accuracy claim; API time is not camera latency')
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='passed' else 2)


if __name__=='__main__':main()
