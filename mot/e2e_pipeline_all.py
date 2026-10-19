import sys, os, glob, time, json, threading, queue, argparse, cv2, numpy as np, psutil
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb
from common import ROOT, SEQ_ROOT, SEQS, GROUPS
from run_trackers import step, JointTracker

# usage: ~/mot-venv/bin/python e2e_pipeline_all.py --model s --trackers holdfast rf-bytetrack ... [--max-frames N] [--timeline]
# e2e_pipeline.py 를 모든 트래커로 넓힌 버전. 트래커는 전부 사람·차 함께(트래커 1개), 단계는 동시 실행.
#   스레드1 사진 열기(작게 풀기)+640 줄이기 -> 스레드2 Hailo 추론+후처리 -> 스레드3(메인) 트래커
# 자원: 단계(스레드)별 CPU 사용량, 코어별 CPU, Hailo 사용 시간 비율, 메모리, 온도를 함께 잰다.
# 결과는 e2e_pipeline_all.json 에 트래커별로 쌓는다. --timeline 이면 첫 영상의 프레임별 단계 시각도 남긴다.
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--trackers", nargs="+", required=True)
ap.add_argument("--seqs", nargs="+", default=SEQS)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--mode", choices=["pipe", "seq"], default="pipe")
ap.add_argument("--timeline", action="store_true")
a = ap.parse_args()
rb.CONF = 0.05
REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}
ALL = sum((GROUPS[g]["det"] for g in GROUPS), [])
OUT = f"{ROOT}/e2e_pipeline_all.json"

def factor(w):
    f = 1
    while f < 8 and w / (f * 2) >= rb.SIZE: f *= 2
    return f

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.cpu, self.mem, self.tmp = [], [], []
    def run(self):
        psutil.cpu_percent(percpu=True)
        while not self.stop:
            time.sleep(0.25)
            self.cpu.append(psutil.cpu_percent(percpu=True)); self.mem.append(psutil.virtual_memory().used / 2**20)
            self.tmp.append(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000)

proc = psutil.Process()
def thread_cpu():
    return {t.id: t.user_time + t.system_time for t in proc.threads()}

hef = HEF(os.path.expanduser(f"~/hailo-test/models/remdet_{a.model}_visdrone.hef")); outs = rb.get_outs(hef)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for trk in a.trackers:
            proc_cpu = 0.0
            T = {"read": [], "hailo": [], "post": [], "track": []}
            stage_cpu = {"read": 0.0, "infer": 0.0, "track": 0.0}; other_cpu = 0.0
            timeline = []; frames, wall = 0, 0.0
            smp = Sampler(); smp.start()
            for si, seq in enumerate(a.seqs):
                files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
                if a.max_frames: files = files[:a.max_frames]
                fac = factor(cv2.imread(files[0]).shape[1])
                tids = {}; rec = a.timeline and si == 0
                jt = JointTracker(trk)

                def read(f, hw):
                    t0 = time.perf_counter()
                    img = cv2.imread(f, REDUCED[fac]) if fac > 1 else cv2.imread(f)
                    if hw is not None and img.shape[:2] != hw: img = cv2.resize(img, hw[::-1])
                    x, s = rb.preprocess(img); t1 = time.perf_counter(); T["read"].append(t1 - t0)
                    return x, s, img, (t0, t1)

                def infer(x, s):
                    t0 = time.perf_counter(); r = pipe.infer({in_name: x}); t1 = time.perf_counter()
                    b, sc, c = rb.postprocess(r, outs, s); t2 = time.perf_counter()
                    T["hailo"].append(t1 - t0); T["post"].append(t2 - t1)
                    return np.c_[b, sc, c], (t0, t1, t2)

                def track(d, img):
                    step.frame = img; t0 = time.perf_counter()
                    jt.update(d[np.isin(d[:, 5], ALL)]); t1 = time.perf_counter(); T["track"].append(t1 - t0)
                    return (t0, t1)

                c0 = thread_cpu(); p0 = sum(proc.cpu_times()[:2]); t_seq = time.perf_counter(); n = 0
                if a.mode == "seq":
                    tids["read"] = tids["infer"] = tids["track"] = threading.get_native_id(); hw = None
                    for f in files:
                        x, s, img, tr = read(f, hw); hw = hw or img.shape[:2]
                        d, ti = infer(x, s); tt = track(d, img); n += 1
                        if rec: timeline.append((tr, ti, tt))
                else:
                    q1, q2 = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
                    def reader():
                        tids["read"] = threading.get_native_id(); hw = None; c = time.thread_time()
                        for f in files:
                            x, s, img, tr = read(f, hw); hw = hw or img.shape[:2]; q1.put((x, s, img, tr))
                        stage_cpu["read"] += time.thread_time() - c; q1.put(None)
                    def inferer():
                        tids["infer"] = threading.get_native_id(); c = time.thread_time()
                        while (it := q1.get()) is not None:
                            x, s, img, tr = it; d, ti = infer(x, s); q2.put((d, img, tr, ti))
                        stage_cpu["infer"] += time.thread_time() - c; q2.put(None)
                    tids["track"] = threading.get_native_id(); ct = time.thread_time()
                    th = [threading.Thread(target=reader, daemon=True), threading.Thread(target=inferer, daemon=True)]
                    for t in th: t.start()
                    while (it := q2.get()) is not None:
                        d, img, tr, ti = it; tt = track(d, img); n += 1
                        if rec: timeline.append((tr, ti, tt))
                    stage_cpu["track"] += time.thread_time() - ct
                    for t in th: t.join()
                    c1 = thread_cpu()
                if a.mode == "seq": c1 = thread_cpu()
                wall += time.perf_counter() - t_seq; frames += n; proc_cpu += sum(proc.cpu_times()[:2]) - p0
                d_cpu = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
                if a.mode == "seq": stage_cpu["track"] += d_cpu.get(tids["track"], 0)
                # 기타 = 프로세스 전체 CPU 에서 세 단계를 뺀 것 (HailoRT 내부 스레드, 샘플러 등)
                if rec:
                    base = timeline[0][0][0]
                    timeline = [[[round(v - base, 5) for v in grp] for grp in fr] for fr in timeline]
            smp.stop = True; smp.join()
            other_cpu = proc_cpu - sum(stage_cpu.values())
            cpu = np.array(smp.cpu); ncore = cpu.shape[1]
            fps = lambda k: len(T[k]) / sum(T[k])
            r = {"e2e_fps": frames / wall, "frames": frames, "mode": a.mode,
                 "stage_fps": {"read": fps("read"), "hailo": fps("hailo"), "post": fps("post"),
                               "infer_post": len(T["hailo"]) / (sum(T["hailo"]) + sum(T["post"])), "track": fps("track")},
                 # 단계별 CPU: 그 스레드가 쓴 CPU 시간 / 걸린 시간 (코어 1개 = 100%)
                 "stage_cpu_pct": {k: v / wall * 100 for k, v in stage_cpu.items()} | {"other": other_cpu / wall * 100},
                 "hailo_busy_pct": sum(T["hailo"]) / wall * 100,
                 "cpu_total_pct": float(cpu.mean()), "cpu_per_core_pct": cpu.mean(0).round(1).tolist(), "ncore": ncore,
                 "ram_used_mb": float(np.max(smp.mem)), "ram_total_mb": psutil.virtual_memory().total / 2**20,
                 "temp_max_c": float(np.max(smp.tmp))}
            if a.timeline: r["timeline"] = timeline
            key = f"{a.model}__{trk}__{a.mode}" + ("__tl" if a.timeline else "")
            res = json.load(open(OUT)) if os.path.exists(OUT) else {}
            res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
            sf = r["stage_fps"]; sc = r["stage_cpu_pct"]
            print(f"{a.model:5s} {trk:22s} [{a.mode}] E2E {r['e2e_fps']:5.1f} FPS | 열기 {sf['read']:.0f} 추론+후처리 {sf['infer_post']:.0f} "
                  f"트래커 {sf['track']:.0f} | CPU 열기 {sc['read']:.0f}% 추론 {sc['infer']:.0f}% 트래커 {sc['track']:.0f}% 기타 {sc['other']:.0f}% "
                  f"(전체 {r['cpu_total_pct']:.0f}%) Hailo {r['hailo_busy_pct']:.0f}% | {frames}프레임", flush=True)
