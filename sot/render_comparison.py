"""Render cached predictions only; display insets do not affect tracker inputs."""
import json
from pathlib import Path

import cv2
import numpy as np

from compare_modes import ROOT, cached_mot


def overlap(a,b):
    if min(a[2:])<=0:
        return 0.
    xy=np.maximum(a[:2],b[:2]);end=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    inter=np.maximum(0,end-xy).prod()
    return float(inter/max(1e-9,np.prod(a[2:])+np.prod(b[2:])-inter))


def annotate(img,box,color,scale=1.,offset=(0,0),thickness=2):
    if min(box[2:])<=0:
        return
    x,y,w,h=box
    p1=(round((x-offset[0])*scale),round((y-offset[1])*scale))
    p2=(round((x+w-offset[0])*scale),round((y+h-offset[1])*scale))
    cv2.rectangle(img,p1,p2,color,thickness)


def main():
    doc=json.loads((ROOT/'sot/results/shared_init.raw.json').read_text())
    original=json.loads((ROOT/'sot/results/modes_13targets.json').read_text())
    manifest={(r['sequence'],tuple(r['target'])):r['frames'] for r in original['manifest']}
    examples=[('uav0000086_00000_v',('pedestrian',11)),
              ('uav0000137_00458_v',('car',13)),
              ('uav0000339_00001_v',('car',5))]
    methods=['KCF-full','CSRT-full','ByteTrack-S','Holdfast-S']
    output=ROOT/'sot/results/tracker_comparison.mp4'
    # Intermediate mp4v is transcoded to browser-compatible H.264 by ffmpeg.
    temporary=ROOT/'sot/results/tracker_comparison_temp.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),15,(960,624))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video encoder')
    count=0
    for seq,key in examples:
        rows={r['method']:r for r in doc['results'] if r['sequence']==seq and tuple(r['target'])==key}
        hold=rows['Holdfast-S']
        ids=manifest[(seq,key)]
        ids=ids[ids.index(hold['timing']['acquired_frame']):]
        truth=np.array(hold['ground_truth'],float)
        seed=truth.copy();seed[0]=hold['predictions'][0]
        bt,_=cached_mot(seq,key[0],[Path(str(f)) for f in ids],seed,'rf-bytetrack')
        predictions={m:np.array(rows[m]['predictions'],float) for m in methods if m in rows}
        predictions['ByteTrack-S']=bt
        for i in range(0,len(ids),2):
            frame=cv2.imread(str(Path.home()/f'datasets/visdrone/mot-val-hf/sequences/{seq}/{ids[i]:07d}.jpg'))
            if frame is None:
                raise ValueError('Missing input frame')
            canvas=np.full((624,960,3),22,np.uint8)
            cv2.putText(canvas,'GT: GREEN   Prediction: ORANGE   |   same playback speed, NOT processing speed',
                        (12,23),cv2.FONT_HERSHEY_SIMPLEX,.47,(230,230,230),1,cv2.LINE_AA)
            cv2.putText(canvas,f'{seq}  {key[0]}:{key[1]}  frame {ids[i]}',
                        (12,47),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1,cv2.LINE_AA)
            for j,method in enumerate(methods):
                x0,y0=(j%2)*480,60+(j//2)*282
                panel=np.full((282,480,3),30,np.uint8)
                scale=min(480/frame.shape[1],250/frame.shape[0])
                resized=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
                panel[30:30+resized.shape[0],:resized.shape[1]]=resized
                pred=predictions[method][i]
                gt=truth[i]
                annotate(panel[30:],gt,(70,240,60),scale)
                annotate(panel[30:],pred,(0,150,255),scale)
                # Zoom inset is for viewing only and uses GT to keep the target visible.
                cx,cy=gt[:2]+gt[2:]/2
                side=max(100,int(max(gt[2:])*2.5))
                left=max(0,min(frame.shape[1]-side,round(cx-side/2)))
                top=max(0,min(frame.shape[0]-side,round(cy-side/2)))
                cut=frame[top:min(frame.shape[0],top+side),left:min(frame.shape[1],left+side)]
                thumb=cv2.resize(cut,(150,150))
                sx,sy=150/cut.shape[1],150/cut.shape[0]
                for b,col in [(gt,(70,240,60)),(pred,(0,150,255))]:
                    bb=b.copy();bb[:2]-=[left,top];bb*=np.array([sx,sy,sx,sy])
                    annotate(thumb,bb,col)
                panel[116:266,320:470]=thumb
                cv2.rectangle(panel,(319,115),(471,267),(230,230,230),1)
                cv2.putText(panel,'DISPLAY ZOOM',(324,111),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1)
                label=method.replace('-full','').replace('-S',' + DET')
                status=f'IoU {overlap(pred,gt):.2f}' if min(pred[2:])>0 else 'NO OUTPUT'
                cv2.putText(panel,f'{label}   {status}',(10,21),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
                canvas[y0:y0+282,x0:x0+480]=panel
            writer.write(canvas);count+=1
    writer.release()
    print(f'{temporary}: {count} frames, {count/15:.1f}s; transcode to {output}')


if __name__=='__main__':
    main()
