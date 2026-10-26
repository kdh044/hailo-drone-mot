"""Continuous local OpenCV viewer with target, tracker and sequence switching.

The processing thread owns Hailo and trackers; the GUI stays responsive. KCF/CSRT
only detect during selection/redetection, whereas KF comparison modes detect each
frame. File playback is paced up to --fps, slowing to actual processing speed.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from contextlib import ExitStack
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np
from benchmark_system import ROOT
from selected_tracker import SelectedTargetEngine as Engine
from live_support import PrefetchFrames, ResourceSampler, InferenceFrames
from pipeline import frames

TITLE = 'Drone Tracking - live target selection'
METHODS = ['KCF', 'CSRT', 'bytetrack', 'holdfast']
LABELS = dict(KCF='KCF', CSRT='CSRT', bytetrack='ByteTrack', holdfast='Holdfast')
WIDTH, VIEW_HEIGHT, HEADER = 1200, 650, 236


def sources_for(source):
    path = Path(source)
    if not path.is_dir():
        return [source]
    siblings = sorted(str(p) for p in path.parent.iterdir()
                      if (p.is_dir() and next(p.glob('*.jpg'), None))
                      or (p.is_file() and p.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov')))
    return siblings if str(path) in siblings else [source]


def pick_box(detections, point, shape):
    """Point is normalized to the displayed frame, independent of window scaling."""
    x, y = point[0]*shape[1], point[1]*shape[0]
    inside = [d for d in detections if d[4] >= .25 and d[0] <= x <= d[2] and d[1] <= y <= d[3]]
    if not inside:
        return None
    d = min(inside, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
    return (float(d[0]), float(d[1]), float(d[2]-d[0]), float(d[3]-d[1]))


@dataclass
class Packet:
    image: object = None
    detections: object = None
    box: object = None
    mode: str = 'LOADING'
    method: str = 'CSRT'
    sequence: int = 0
    generation: int = 0
    frame: int = 0
    fps: float = 0
    cpu: str = '--'
    hailo: str = '--'
    detector: bool = False
    notice: str = 'Loading real Hailo detector...'
    paused: bool = False
    error: str = ''
    resources: object = None
    timings: object = None
    selected_inputs: int = 0
    hailo_call_busy_pct: float = 0
    show_detections: bool = False


class Player:
    def __init__(self, args):
        self.args = args
        self.sources = sources_for(args.source)
        self.commands = Queue()
        self.stop = Event()
        self.lock = Lock()
        self.packet = Packet(method=args.tracker, sequence=self.sources.index(args.source))
        self.thread = Thread(target=self.run, daemon=True)

    def publish(self, **values):
        with self.lock:
            old = self.packet.__dict__.copy()
            old.update(values)
            self.packet = Packet(**old)

    def snapshot(self):
        with self.lock:
            return self.packet

    def start(self):
        self.thread.start()

    def run(self):
        stream = None
        try:
            sys.path.insert(0, str(ROOT))
            sys.path.append(str(self.args.mot_site_packages))
            import remdet_bench as rb
            from hailo_platform import (HEF, VDevice, Device, ConfigureParams, HailoStreamInterface,
                                        InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
            rb.CONF = .05
            # Import all four adapters once so method changes do not import libraries mid-playback.
            for method in METHODS:
                Engine(method)
            method = self.args.tracker
            engine = Engine(method)
            sequence = self.sources.index(self.args.source)
            generation = 0
            stream = None
            armed = paused = False
            show_all = False
            frame_no = 0
            fps = 0.
            cpu = hailo = '--'
            last_temp = 0.
            inference_window = deque()
            notice = 'Video keeps playing. Click any target, or press R to see detections.'
            with ExitStack() as stack:
                device = stack.enter_context(VDevice())
                hef = HEF(str(self.args.hef))
                outputs = rb.get_outs(hef)
                group = device.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
                infer = stack.enter_context(InferVStreams(group,
                    InputVStreamParams.make(group, format_type=FormatType.UINT8),
                    OutputVStreamParams.make(group, format_type=FormatType.FLOAT32)))
                stack.enter_context(group.activate(group.create_params()))
                physical = stack.enter_context(Device())
                name = hef.get_input_vstream_infos()[0].name
                sampler = ResourceSampler()
                sampler.start()
                stack.callback(sampler.close)
                log = stack.enter_context(open(self.args.event_log, 'a', buffering=1))
                stage = {}
                infer_lock = Lock()
                def detect_metrics(image, prepared=None):
                    with infer_lock:
                        t0 = time.perf_counter()
                        x, scale = prepared if prepared is not None else rb.preprocess(image)
                        t1 = time.perf_counter()
                        raw = infer.infer({name:x})
                        t2 = time.perf_counter()
                        boxes, scores, classes = rb.postprocess(raw, outputs, scale)
                        t3 = time.perf_counter()
                    result = np.c_[boxes,scores,classes]
                    return result[np.isin(result[:,5],[0,1,3,4,5,8])], dict(
                        preprocess_sync_ms=(t1-t0)*1000,infer_ms=(t2-t1)*1000,post_ms=(t3-t2)*1000)
                def detect(image, prepared=None):
                    result,timing = detect_metrics(image,prepared)
                    for key,value in timing.items(): stage[key] = stage.get(key,0)+value
                    return result
                def make_stream(start_frame=1):
                    raw = PrefetchFrames(self.sources[sequence],rb.preprocess,start_frame=start_frame,
                                         source_fps=getattr(self.args,'source_fps',0))
                    return InferenceFrames(raw,detect_metrics) if engine.is_mot else raw
                def read_next():
                    item = next(stream)
                    return item if len(item)==6 else (*item,None,{})
                stream = make_stream()
                pipeline_mode = engine.is_mot
                # Close producer threads before closing the Hailo streams they use.
                stack.callback(lambda: stream.close() if stream is not None else None)
                while not self.stop.is_set():
                    started = time.perf_counter()
                    stage = {}
                    click = None
                    reset_stream = False
                    reset_state = False
                    while True:
                        try: command, value = self.commands.get_nowait()
                        except Empty: break
                        if command == 'pause':
                            paused = not paused
                        elif command == 'method':
                            show_all = False
                            method = value
                            reset_state = True
                            notice = f'{LABELS[method]} selected. Click a target; video keeps playing.'
                        elif command == 'reselect':
                            show_all = True
                            reset_state = True
                            notice = 'Detection mode: click another target; video keeps playing.'
                        elif command == 'sequence':
                            show_all = False
                            sequence = int(value) % len(self.sources)
                            reset_stream = reset_state = True
                            click = None
                            notice = 'Sequence changed. Click a target.'
                        elif command == 'replay':
                            show_all = False
                            reset_stream = reset_state = True
                            click = None
                            notice = 'Replaying from frame 1. Click a target.'
                        elif command == 'click' and value['generation'] == generation:
                            click = value
                    if reset_state:
                        inference_window.clear()
                        engine = Engine(method)
                        armed = False
                        fps = 0.
                    if reset_stream or pipeline_mode != engine.is_mot:
                        stream.close()
                        if reset_stream:
                            paused = False
                            frame_no = 0
                            generation += 1
                        stream = make_stream(frame_no+1)
                        pipeline_mode = engine.is_mot
                    if paused:
                        inference_window.clear()
                        self.publish(hailo_call_busy_pct=0)
                        frozen = self.snapshot()
                        if (click is not None or reset_state) and frozen.image is not None:
                            engine.select_target()
                            snapshot = click['image'] if click is not None else frozen.image
                            click_dets = click['detections'] if click is not None else frozen.detections
                            if click_dets is None:
                                click_dets = detect(snapshot)
                            target = pick_box(click_dets, click['point'], snapshot.shape) if click is not None else None
                            engine.select_target(target)
                            armed = target is not None
                            if click is not None:
                                show_all = not armed
                            if armed and not engine.is_mot:
                                engine.update(snapshot, click_dets)
                            # KF confirmation needs future frames. Preview selection now;
                            # do not consume video or fake repeated KF measurements while paused.
                            self.publish(image=snapshot,box=target,detections=click_dets,
                                mode='TARGET SELECTED' if armed else 'DETECT / SELECT',method=method,
                                paused=True,show_detections=show_all,notice='Selected. Space to resume.' if armed else 'Click a green box.',
                                resources=sampler.latest.copy(),timings=stage.copy())
                            print('PAUSED_SELECT', method, frozen.frame, target, flush=True)
                        else:
                            self.publish(paused=True,resources=sampler.latest.copy())
                        self.stop.wait(.02)
                        continue
                    try:
                        read_started = time.perf_counter()
                        image, prepared, decode_ms, prefetch_pre_ms, prefetched_dets, prefetched_timing = read_next()
                        frame_no += 1
                    except StopIteration:
                        stream.close()
                        stream = make_stream()
                        generation += 1
                        frame_no = 1
                        image, prepared, decode_ms, prefetch_pre_ms, prefetched_dets, prefetched_timing = read_next()
                        engine = Engine(method)
                        armed = False
                        click = None
                        show_all = False
                        notice = 'Video looped to beginning. Click a target.'
                    stage.update(wait_for_frame_ms=(time.perf_counter()-read_started)*1000,decode_ms=decode_ms,prefetch_preprocess_ms=prefetch_pre_ms)
                    if frame_no == 1:
                        source_shape = image.shape[:2]
                    elif image.shape[:2] != source_shape:
                        image = cv2.resize(image, source_shape[::-1])
                        prepared = None
                    initialized_on_snapshot = False
                    if click is not None:
                        # Detect on the exact frame the user clicked, then track forward.
                        # SOT update is not executed during this detection operation.
                        engine.select_target()
                        snapshot = click['image']
                        click_dets = click['detections']
                        if click_dets is None:
                            click_dets = detect(snapshot)
                        target = pick_box(click_dets, click['point'], snapshot.shape)
                        engine.select_target(target)
                        armed = target is not None
                        show_all = not armed
                        if armed:
                            if not engine.is_mot:
                                result = engine.update(snapshot, click_dets)
                                initialized_on_snapshot = True
                            notice = 'Target changed while video keeps playing.'
                            print('SELECT', method, frame_no, target, flush=True)
                        else:
                            notice = 'No confident detection at click. Green boxes are selectable.'
                    detector_called = engine.is_mot or not armed or engine.state == 'DETECT'
                    if detector_called and prefetched_dets is not None:
                        detections = prefetched_dets
                        for key,value in prefetched_timing.items(): stage[key] = stage.get(key,0)+value
                    else:
                        detections = detect(image,prepared) if detector_called else np.empty((0,6))
                    # Pending ByteTrack acquisition and recovery remain tied to selected reference.
                    box = None
                    mode = 'DETECT / SELECT'
                    previous = engine.state
                    track_started = time.perf_counter()
                    if armed:
                        result = engine.update(image, detections)
                        box = result['box']
                        mode = engine.state
                        if mode == 'RECOVER':
                            notice = f'Recovering target ({engine.retries}/{engine.limit}); video keeps playing.'
                        elif mode == 'DETECT':
                            notice = 'Redetecting near last target. Click anywhere to change target.'
                        elif previous != 'TRACK':
                            notice = 'Target acquired. Click another object anytime, or press R.'
                    stage['track_ms'] = (time.perf_counter()-track_started)*1000
                    elapsed = time.perf_counter()-started
                    fps = (1/elapsed) if not fps else .9*fps+.1/elapsed
                    now = time.monotonic()
                    if now-last_temp >= 3:
                        last_temp = now
                        try: cpu = f"{float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000:.1f}C"
                        except (OSError, ValueError): cpu = '--'
                        try:
                            t = physical.control.get_chip_temperature()
                            hailo = f'{max(t.ts0_temperature,t.ts1_temperature):.1f}C'
                        except Exception: hailo = '--'
                    inference_now = time.perf_counter()
                    inference_window.append((started, stage.get('infer_ms',0)))
                    while len(inference_window)>1 and inference_now-inference_window[0][0]>2:
                        inference_window.popleft()
                    hailo_call_busy_pct = min(100, sum(v for _,v in inference_window)/10/max(.000001,inference_now-inference_window[0][0]))
                    self.publish(image=image, detections=detections if detector_called else None,
                        box=box, mode=mode, method=method, sequence=sequence, generation=generation,
                        frame=frame_no, fps=fps, cpu=cpu, hailo=hailo, detector=detector_called,
                        notice=notice, paused=paused, show_detections=show_all, resources=sampler.latest.copy(), timings=stage.copy(),selected_inputs=engine.gated_detections if armed and engine.is_mot else 0,hailo_call_busy_pct=hailo_call_busy_pct)
                    log.write(json.dumps(dict(frame=frame_no,sequence=Path(self.sources[sequence]).name,
                        generation=generation,method=method,mode=mode,processing_mode=previous,detector_called=detector_called,
                        selected=armed,box=box,clicked=click is not None,
                        snapshot_detection=click is not None and click['detections'] is None,
                        initialized_on_snapshot=initialized_on_snapshot,
                        selected_tracker_inputs=engine.gated_detections if armed and engine.is_mot else 0,
                        timings_ms=stage,resources=sampler.latest,processing_ms=elapsed*1000,
                        cpu_temperature_c=cpu,hailo_temperature_c=hailo))+'\n')
                    if self.args.max_frames and frame_no >= self.args.max_frames:
                        self.stop.set()
                    if not paused and self.args.fps > 0:
                        self.stop.wait(max(0,1/self.args.fps-(time.perf_counter()-started)))
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.publish(error=str(error), mode='ERROR', notice='Processing error. See console log.')
        finally:
            if stream is not None: stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', default=str(Path.home()/'datasets/visdrone/mot-val-hf/sequences/uav0000086_00000_v'))
    parser.add_argument('--tracker', choices=METHODS, default='CSRT')
    parser.add_argument('--hef', type=Path, default=ROOT/'models/remdet_s_visdrone.hef')
    parser.add_argument('--mot-site-packages', type=Path, default=Path.home()/'mot-venv/lib/python3.13/site-packages')
    parser.add_argument('--layout', choices=['classic','controls'], default='classic')
    parser.add_argument('--source-fps',type=float,default=0,help='File frame arrival rate; 0 for throughput benchmark only')
    parser.add_argument('--fps', type=float, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--event-log', default='/tmp/hailo-live-demo.events.jsonl')
    args = parser.parse_args()
    global HEADER
    HEADER = 68 if args.layout == 'classic' else 108
    if args.fps < 0 or args.source_fps < 0: parser.error('fps must be nonnegative')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    player = Player(args)
    cv2.namedWindow(TITLE, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    ui = {'buttons':[], 'packet':None, 'mapping':None, 'menu':False, 'stats':False, 'help':False}
    def send(command,value=None): player.commands.put((command,value))
    def mouse(event,x,y,flags,param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        for (x1,y1,x2,y2),command,value in ui['buttons']:
            if x1 <= x < x2 and y1 <= y < y2:
                if command == 'quit': player.stop.set()
                elif command == 'menu': ui['menu'] = not ui['menu']
                else:
                    send(command,value)
                    ui['menu'] = False
                return
        if ui['menu']: return
        packet = ui['packet']
        if packet is None or packet.image is None: return
        ox,oy,vw,vh = ui['mapping']
        if ox <= x < ox+vw and oy <= y < oy+vh:
            send('click',dict(point=((x-ox)/vw,(y-oy)/vh),generation=packet.generation,
                              image=packet.image,detections=packet.detections))
    cv2.setMouseCallback(TITLE,mouse)
    player.start()
    try:
        while not player.stop.is_set():
            packet = player.snapshot()
            ui['packet'] = packet
            HEADER = 68 + (40 if args.layout == 'controls' else 0) + (58 if ui['stats'] else 0)
            canvas = np.full((HEADER+VIEW_HEIGHT,WIDTH,3),24,np.uint8)
            buttons=[]
            def button(x,w,label,command,value=None,active=False,y=8):
                rect=(x,y,x+w,y+32)
                cv2.rectangle(canvas,(x,y),(x+w,y+32),(65,110,65) if active else (65,65,65),-1)
                cv2.putText(canvas,label,(x+8,y+22),cv2.FONT_HERSHEY_SIMPLEX,.47,(245,245,245),1,cv2.LINE_AA)
                buttons.append((rect,command,value))
            if args.layout == 'controls':
                for i,method in enumerate(METHODS):
                    button(12+112*i,104,LABELS[method],'method',method,method==packet.method)
                button(464,86,'Resume' if packet.paused else 'Pause','pause')
                button(558,94,'Reselect','reselect')
                button(660,80,'Replay','replay')
                button(748,68,'Prev','sequence',packet.sequence-1)
                button(824,68,'Next','sequence',packet.sequence+1)
                button(900,142,'Sequence list','menu')
                button(1050,70,'Quit','quit')
            mode = ('PAUSED / SELECTED' if packet.box is not None else 'PAUSED / SELECT') if packet.paused else packet.mode
            resource = packet.resources or {}
            lines=[f"{LABELS[packet.method]} | {mode} | {packet.fps:.1f} procFPS | CPU {resource.get('system_cpu_total_pct',0):.0f}% | Hailo infer~{packet.hailo_call_busy_pct:.0f}% | Video {packet.sequence+1}/{len(player.sources)}",
                   'Click: target | Space: pause | 1 KCF  2 CSRT  3 Byte  4 Hold | N/P: video | B: replay | I: stats | H: help']
            if packet.error:
                lines[0] = 'ERROR: ' + packet.error[:120]
            if ui['stats']:
                resource = packet.resources or {}
                timing = packet.timings or {}
                lines += [f"CPU app {resource.get('process_cpu_one_core_pct',0):.0f}% / {resource.get('cpu_cores',4)*100}% | system idle {resource.get('system_idle_pct',0):.0f}% | RAM app {resource.get('process_rss_mb',0):.0f}MB / available {resource.get('system_available_ram_mb',0)/1024:.2f}GB",
                          f"CPU {packet.cpu} | Hailo {packet.hailo} | ms: wait {timing.get('wait_for_frame_ms',0):.1f}, infer {timing.get('infer_ms',0):.1f}, post {timing.get('post_ms',0):.1f}, track {timing.get('track_ms',0):.1f}"]
            offset = 40 if args.layout == 'controls' else 0
            for i,line in enumerate(lines):
                cv2.putText(canvas,line[:145],(14,offset+24+29*i),cv2.FONT_HERSHEY_SIMPLEX,.48,
                            (90,160,255) if packet.error else (230,230,230),1,cv2.LINE_AA)
            if packet.image is not None:
                h,w=packet.image.shape[:2]
                scale=min(WIDTH/w,VIEW_HEIGHT/h)
                vw,vh=round(w*scale),round(h*scale)
                ox,oy=(WIDTH-vw)//2,HEADER+(VIEW_HEIGHT-vh)//2
                ui['mapping']=(ox,oy,vw,vh)
                view=cv2.resize(packet.image,(vw,vh))
                if packet.show_detections and packet.detections is not None:
                    for d in packet.detections:
                        if d[4] < .25: continue
                        x1,y1,x2,y2=[round(v*scale) for v in d[:4]]
                        cv2.rectangle(view,(x1,y1),(x2,y2),(80,215,80),1)
                if packet.box is not None:
                    x,y,bw,bh=packet.box
                    cv2.rectangle(view,(round(x*scale),round(y*scale)),(round((x+bw)*scale),round((y+bh)*scale)),(0,170,255),3)
                    cv2.putText(view,'TARGET',(round(x*scale),max(15,round(y*scale)-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,170,255),1)
                canvas[oy:oy+vh,ox:ox+vw]=view
            if ui['menu']:
                for i,source in enumerate(player.sources):
                    button(20,690,f'{i+1}. {Path(source).name}','sequence',i,packet.sequence==i,y=HEADER+8+36*i)
            if ui['help']:
                help_lines=['Click another object to change target (also while paused).',
                            'Space pause/resume | R show detections | B replay',
                            '1 KCF | 2 CSRT | 3 ByteTrack | 4 Holdfast',
                            'N/P next/previous video | S sequence list',
                            'I resource stats | Tab buttons | H close help | Q quit']
                cv2.rectangle(canvas,(12,HEADER+8),(850,HEADER+180),(30,30,30),-1)
                for i,line in enumerate(help_lines):
                    cv2.putText(canvas,line,(24,HEADER+34+30*i),cv2.FONT_HERSHEY_SIMPLEX,.53,(240,240,240),1,cv2.LINE_AA)
            ui['buttons']=buttons
            cv2.imshow(TITLE,canvas)
            key=cv2.waitKey(15)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(TITLE,cv2.WND_PROP_VISIBLE)<1:break
            if key==9:
                args.layout='controls' if args.layout=='classic' else 'classic'
            elif key==ord('i'):ui['stats']=not ui['stats']
            elif key==ord('h'):ui['help']=not ui['help']
            elif key==ord('s'):ui['menu']=not ui['menu']
            elif key==32:send('pause')
            elif key==ord('r'):send('reselect')
            elif key==ord('b'):send('replay')
            elif key==ord('n'):send('sequence',packet.sequence+1)
            elif key==ord('p'):send('sequence',packet.sequence-1)
            elif key in map(ord,'1234'):send('method',METHODS[key-ord('1')])
    finally:
        player.stop.set()
        player.thread.join(timeout=10)
        cv2.destroyAllWindows()

if __name__=='__main__':main()
