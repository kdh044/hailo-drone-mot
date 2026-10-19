import sys, os, glob, time, json, threading, queue, argparse, cv2, numpy as np, psutil
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb
from common import ROOT, SEQ_ROOT, SEQS, GROUPS
from run_trackers import make, step, needs_frame, JointTracker

# usage: ~/mot-venv/bin/python e2e_bench.py --model s --trackers bytetrack ocsort [--seqs ...]
# Real pipeline: reader thread (jpg decode + resize) -> Hailo infer -> postprocess -> tracker.
# Reports per-stage ms, wait-for-frame stall (file loading bottleneck), E2E FPS (wall clock),
# Hailo busy ratio (infer time / wall), CPU per core, RAM, SoC temperature.
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--trackers", nargs="+", default=["none"])
ap.add_argument("--seqs", nargs="+", default=SEQS)
ap.add_argument("--max-frames", type=int, default=0, help="영상마다 앞에서부터 N프레임만 (0 = 전부)")
ap.add_argument("--decode", choices=["full", "reduced"], default="full",
                help="full: decode original jpg; reduced: libjpeg DCT-scaled decode (1/2, 1/4, 1/8) keeping the long side >= 640")
a = ap.parse_args()
REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def pick_factor(w):
    f = 1
    while f < 8 and w / (f * 2) >= rb.SIZE: f *= 2
    return f
rb.CONF = 0.05


def reader(q, files, need_full, factor):
    # factor > 1: decode at 1/factor size; boxes then come out in that reduced frame and are scaled back by factor
    for f in files:
        t0 = time.perf_counter()
        img = cv2.imread(f) if factor == 1 else cv2.imread(f, REDUCED[factor])
        t1 = time.perf_counter()
        x, s = rb.preprocess(img); t2 = time.perf_counter()
        q.put((x, s, img if need_full else None, t1 - t0, t2 - t1))
    q.put(None)


def temp():
    try: return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception: return float("nan")


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.cpu, self.mem, self.tmp = [], [], []
        self.proc = psutil.Process()
    def run(self):
        psutil.cpu_percent(percpu=True)
        while not self.stop:
            time.sleep(0.5)
            self.cpu.append(psutil.cpu_percent(percpu=True))
            self.mem.append(psutil.virtual_memory().used / 2**20)
            self.tmp.append(temp())


hef = HEF(os.path.expanduser(f"~/hailo-test/models/remdet_{a.model}_visdrone.hef")); outs = rb.get_outs(hef)
results = {}
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for trk in a.trackers:
            T = {k: [] for k in ("decode", "resize", "wait", "infer", "post", "track")}
            per_seq, frames, wall = {}, 0, 0.0
            smp = Sampler(); smp.start()
            for seq in a.seqs:
                joint = trk.endswith("-joint")
                trackers = None if trk == "none" else ({"all": JointTracker(trk[:-6])} if joint else {g: make(trk) for g in GROUPS})
                need_full = trk != "none" and any(needs_frame(t.t if joint else t) for t in trackers.values())
                files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
                if a.max_frames: files = files[:a.max_frames]
                factor = pick_factor(cv2.imread(files[0]).shape[1]) if a.decode == "reduced" else 1
                q = queue.Queue(maxsize=8)
                threading.Thread(target=reader, args=(q, files, need_full, factor), daemon=True).start()
                t_seq = time.perf_counter(); n = 0
                while True:
                    tw = time.perf_counter(); item = q.get(); tw = time.perf_counter() - tw
                    if item is None: break
                    x, s, full, td, tr = item; n += 1
                    t0 = time.perf_counter(); res = pipe.infer({in_name: x})
                    t1 = time.perf_counter(); b, sc, c = rb.postprocess(res, outs, s)
                    t2 = time.perf_counter()
                    if trackers:
                        step.frame = full
                        d = np.c_[b, sc, c]
                        if joint:
                            trackers["all"].update(d[np.isin(c, sum((GROUPS[g]["det"] for g in GROUPS), []))])
                        else:
                            for g, spec in GROUPS.items():
                                step(trackers[g], d[np.isin(c, spec["det"])])
                    t3 = time.perf_counter()
                    for k, v in zip(T, (td, tr, tw, t1 - t0, t2 - t1, t3 - t2)): T[k].append(v * 1000)
                dt = time.perf_counter() - t_seq
                per_seq[seq] = {"frames": n, "fps": n / dt}
                frames += n; wall += dt
            smp.stop = True; smp.join()
            cpu = np.array(smp.cpu)
            r = {"e2e_fps": frames / wall, "frames": frames,
                 **{f"{k}_ms": float(np.mean(v)) for k, v in T.items()},
                 **{f"{k}_p95_ms": float(np.percentile(v, 95)) for k, v in T.items()},
                 "hailo_busy_pct": float(np.sum(T["infer"]) / 1000 / wall * 100),
                 "cpu_total_pct": float(cpu.mean()), "cpu_per_core_pct": cpu.mean(0).round(1).tolist(),
                 "ram_used_mb": float(np.max(smp.mem)), "ram_total_mb": psutil.virtual_memory().total / 2**20,
                 "temp_max_c": float(np.nanmax(smp.tmp)), "per_seq": per_seq}
            r["decode_mode"] = a.decode
            results[f"{a.model}__{trk}__{a.decode}"] = r
            fps = lambda k: 1000 / r[f"{k}_ms"] if r[f"{k}_ms"] > 0 else float("inf")
            print(f"{a.model:5s} {trk:20s} [{a.decode:7s}] E2E {r['e2e_fps']:5.1f} FPS | stage FPS: decode {fps('decode'):.0f} "
                  f"resize {fps('resize'):.0f} infer {fps('infer'):.0f} post {fps('post'):.0f} track {fps('track'):.0f} | "
                  f"frame-wait {r['wait_ms'] / (1000 / r['e2e_fps']) * 100:.0f}% of time | "
                  f"hailo {r['hailo_busy_pct']:.0f}% cpu {r['cpu_total_pct']:.0f}% ram {r['ram_used_mb']:.0f}MB temp {r['temp_max_c']:.0f}C", flush=True)

f = f"{ROOT}/e2e.json"
allr = json.load(open(f)) if os.path.exists(f) else {}
allr.update(results); json.dump(allr, open(f, "w"), indent=1)
