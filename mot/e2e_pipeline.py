import sys, os, glob, time, json, threading, queue, argparse, cv2, numpy as np, psutil
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb
from common import ROOT, SEQ_ROOT, SEQS, GROUPS
from run_trackers import step, JointTracker
import ext_trackers

# usage: ~/mot-venv/bin/python e2e_pipeline.py --model s [--cmc-width 640] [--seqs ...] [--max-frames N]
# 단계를 차례로 돌리지 않고 동시에 돌리는 파이프라인:
#   스레드1 사진 열기(작게 풀기)+640 줄이기 -> 스레드2 Hailo 추론+후처리 -> 스레드3(메인) Holdfast(사람·차 함께)
# Hailo 호출과 Holdfast(C++, ctypes)는 파이썬 GIL을 놓기 때문에 실제로 겹쳐서 돈다.
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cmc-width", type=int, default=640)
ap.add_argument("--seqs", nargs="+", default=SEQS)
ap.add_argument("--max-frames", type=int, default=0)
a = ap.parse_args()
rb.CONF = 0.05
REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}
ALL = sum((GROUPS[g]["det"] for g in GROUPS), [])

def factor(w):
    f = 1
    while f < 8 and w / (f * 2) >= rb.SIZE: f *= 2
    return f

hef = HEF(os.path.expanduser(f"~/hailo-test/models/remdet_{a.model}_visdrone.hef")); outs = rb.get_outs(hef)
T = {"read": [], "infer": [], "track": []}
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        frames, wall = 0, 0.0
        psutil.cpu_percent(); temps = []
        for seq in a.seqs:
            files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
            if a.max_frames: files = files[:a.max_frames]
            fac = factor(cv2.imread(files[0]).shape[1])
            q1, q2 = queue.Queue(maxsize=4), queue.Queue(maxsize=4)

            def reader():
                hw = None
                for f in files:
                    t0 = time.perf_counter()
                    img = cv2.imread(f, REDUCED[fac]) if fac > 1 else cv2.imread(f)
                    if hw is None: hw = img.shape[:2]
                    elif img.shape[:2] != hw: img = cv2.resize(img, hw[::-1])
                    x, s = rb.preprocess(img); T["read"].append(time.perf_counter() - t0)
                    q1.put((x, s, img))
                q1.put(None)

            def inferer():
                while (it := q1.get()) is not None:
                    x, s, img = it; t0 = time.perf_counter()
                    b, sc, c = rb.postprocess(pipe.infer({in_name: x}), outs, s)
                    T["infer"].append(time.perf_counter() - t0)
                    q2.put((np.c_[b, sc, c], img))
                q2.put(None)

            jt = JointTracker.__new__(JointTracker)
            jt.t = ext_trackers.HoldfastAdapter(work_width=a.cmc_width); jt.vote = {}
            th = [threading.Thread(target=reader, daemon=True), threading.Thread(target=inferer, daemon=True)]
            t_seq = time.perf_counter()
            for t in th: t.start()
            n = 0
            while (it := q2.get()) is not None:
                d, img = it; step.frame = img; t0 = time.perf_counter()
                jt.update(d[np.isin(d[:, 5], ALL)]); T["track"].append(time.perf_counter() - t0); n += 1
            wall += time.perf_counter() - t_seq; frames += n
            temps.append(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000)
        cpu = psutil.cpu_percent()
fps = lambda k: len(T[k]) / sum(T[k])
print(f"{a.model:5s} 파이프라인 holdfast-joint cmc폭 {a.cmc_width:4d} | E2E {frames / wall:5.1f} FPS | 단계별 FPS: 열기 {fps('read'):.0f} "
      f"추론+후처리 {fps('infer'):.0f} 트래커 {fps('track'):.0f} | CPU {cpu:.0f}% 최고온도 {max(temps):.0f}C | {frames}프레임", flush=True)
