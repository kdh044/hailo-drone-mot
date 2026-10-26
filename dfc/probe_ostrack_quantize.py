"""Accuracy-priority quantization/compilation feasibility probe for the no-CE variant.

Uses existing calibration JPEGs and RemDet to select one target per image.
Pairs are STATIC crops from the same image, not a validated temporal calibration
set. Passing this probe establishes neither tracking accuracy nor Pi speed.
"""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort


def sample(img,box,factor,size):
    # Same crop/padding convention as OSTrack processing_utils.sample_target.
    x,y,w,h=box
    side=math.ceil(math.sqrt(w*h)*factor)
    x1,y1=round(x+w/2-side/2),round(y+h/2-side/2)
    x2,y2=x1+side,y1+side
    lp,rp=max(0,-x1),max(x2-img.shape[1]+1,0)
    tp,bp=max(0,-y1),max(y2-img.shape[0]+1,0)
    cut=img[y1+tp:y2-bp,x1+lp:x2-rp]
    cut=cv2.copyMakeBorder(cut,tp,bp,lp,rp,cv2.BORDER_CONSTANT)
    return cv2.resize(cut,(size,size))


def select_target(session,img):
    h,w=img.shape[:2]
    ratio=640/max(h,w)
    canvas=np.full((640,640,3),114,np.uint8)
    resized=cv2.resize(img,(int(w*ratio),int(h*ratio)))
    canvas[:resized.shape[0],:resized.shape[1]]=resized
    inputs=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)[None]/255
    outputs=session.run(None,{session.get_inputs()[0].name:inputs})
    best=None
    allowed=[0,1,3,4,5,8]
    for cls,reg in zip(outputs[::2],outputs[1::2]):
        H,W=cls.shape[2:]
        logits=cls[0].transpose(1,2,0).reshape(-1,10)
        scores=1/(1+np.exp(-np.clip(logits,-80,80)))
        chosen=scores[:,allowed].max(1)
        i=int(chosen.argmax());score=float(chosen[i])
        if score<.25 or (best and score<=best[0]):
            continue
        distribution=reg[0].transpose(1,2,0).reshape(-1,4,16)[i]
        distribution=np.exp(distribution-distribution.max(-1,keepdims=True))
        distribution/=distribution.sum(-1,keepdims=True)
        stride=640/H
        d=(distribution*np.arange(16)).sum(-1)*stride
        cx,cy=(i%W+.5)*stride,(i//W+.5)*stride
        x1,y1=max(0,(cx-d[0])/ratio),max(0,(cy-d[1])/ratio)
        x2,y2=min(w,(cx+d[2])/ratio),min(h,(cy+d[3])/ratio)
        if min(x2-x1,y2-y1)>=4:
            best=(score,[float(x1),float(y1),float(x2-x1),float(y2-y1)])
    return None if best is None else best[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--har',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--detector',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--compile',action='store_true')
    p.add_argument('--reuse-prepared',action='store_true',help='Reuse this output directory NPZ and unique-image manifest')
    a=p.parse_args()
    if a.samples < 1024:
        p.error('At least 1024 distinct calibration images are required')
    import tensorflow as tf
    if not tf.config.list_physical_devices('GPU'):
        p.error('Optimization level 4 requires a visible GPU; refusing CPU downgrade')
    from hailo_sdk_client import ClientRunner,__version__
    cv2.setNumThreads(1)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(model='OSTrack-384 CE-disabled compatibility variant',dfc=__version__,target='hailo8',
        calibration_source='existing RemDet calibration JPEGs; original split provenance not verified',
        static_same_image_pairs=True,template_factor=2,search_factor=4,template_size=192,search_size=384,
        calibration='running',quantization='not_run',compilation='not_run',accuracy_evaluation='not_run',
        optimization_level=4,compression_level=0,
        performance_claim='none; compatibility variant requires separate accuracy validation')
    path=a.output_dir/'quant_compile_probe.json'
    def save():path.write_text(json.dumps(report,indent=2))
    save()
    try:
        if a.reuse_prepared:
            manifest=json.loads((a.output_dir/'calibration_manifest.json').read_text())
            if len(manifest)!=a.samples or len({r['sha256'] for r in manifest})!=a.samples:
                raise ValueError('Prepared manifest must contain requested number of distinct images')
            with np.load(a.output_dir/'calibration_pairs.npz') as stored:
                data={k:stored[k] for k in stored.files}
            expected={'ostrack384/input_layer1':(a.samples,192,192,3),'ostrack384/input_layer2':(a.samples,384,384,3)}
            if set(data)!=set(expected) or any(data[k].shape!=shape for k,shape in expected.items()):
                raise ValueError('Prepared calibration input shapes differ')
            templates=manifest
        else:
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            detector=ort.InferenceSession(str(a.detector),sess_options=options,providers=['CPUExecutionProvider'])
            files=sorted(a.images.glob('*.jpg'))
            np.random.default_rng(17).shuffle(files)
            templates,searches=[],[]
            selected_files=[]
            import hashlib
            unique_hashes=set()
            mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
            for file in files:
                img=cv2.imread(str(file))
                if img is None:
                    continue
                digest=hashlib.sha256(file.read_bytes()).hexdigest()
                if digest in unique_hashes:
                    continue
                box=select_target(detector,img)
                if box is None:
                    continue
                unique_hashes.add(digest)
                selected_files.append(dict(file=file.name,sha256=digest,box=box))
                rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                templates.append((sample(rgb,box,2,192).astype(np.float32)/255-mean)/std)
                searches.append((sample(rgb,box,4,384).astype(np.float32)/255-mean)/std)
                if len(templates)==a.samples:
                    break
            if len(templates)<a.samples:
                raise ValueError(f'Only {len(templates)} images produced valid detections')
            (a.output_dir/'calibration_manifest.json').write_text(json.dumps(selected_files,indent=2))
            data={'ostrack384/input_layer1':np.stack(templates),'ostrack384/input_layer2':np.stack(searches)}
            np.savez(a.output_dir/'calibration_pairs.npz',**data)
        report.update(calibration='prepared',samples=len(templates),input_shapes={k:list(v.shape) for k,v in data.items()},quantization='running')
        save();print('CALIBRATION PREPARED',a.samples,flush=True)
        runner=ClientRunner(har=str(a.har))
        runner.load_model_script(f'model_optimization_flavor(optimization_level=4, compression_level=0)\nmodel_optimization_config(calibration, batch_size=1, calibset_size={a.samples})\n')
        start=time.perf_counter()
        (a.output_dir/'optimize').mkdir(parents=True,exist_ok=True)
        runner.optimize(data,work_dir=str(a.output_dir/'optimize'))
        quant=a.output_dir/'ostrack384_noce_quant.har'
        runner.save_har(str(quant))
        report.update(quantization='passed',quantization_seconds=time.perf_counter()-start,optimized_har=quant.name)
        save();print('QUANTIZATION PASSED',flush=True)
        if a.compile:
            report['compilation']='running';save();start=time.perf_counter()
            hef=runner.compile()
            (a.output_dir/'ostrack384_noce.hef').write_bytes(hef)
            report.update(compilation='passed',compilation_seconds=time.perf_counter()-start,hef_bytes=len(hef))
            save();print('COMPILATION PASSED',flush=True)
    except Exception as e:
        for stage in ('calibration','quantization','compilation'):
            if report[stage]=='running':report[stage]='failed'
        report.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        save();print(json.dumps(report,indent=2),flush=True);raise SystemExit(2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
