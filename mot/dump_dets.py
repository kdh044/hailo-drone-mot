import sys, os, glob, time, threading, queue, cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb

# usage: python3 dump_dets.py <hef> <sequences dir> <out dir>
# saves <out>/<seq>.npy rows: frame(1-based), x1, y1, x2, y2, score, class(VisDrone 0..9)
hef_path, seq_root, out = sys.argv[1:4]
rb.CONF = 0.05  # low threshold: trackers apply their own score thresholds
os.makedirs(out, exist_ok=True)

def reader(q, files):
    for f in files: q.put(rb.preprocess(cv2.imread(f)))
    q.put(None)

hef = rb.HEF(hef_path); outs = rb.get_outs(hef)
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for seq in sorted(glob.glob(f"{seq_root}/*/")):
            name = os.path.basename(seq.rstrip("/"))
            files = sorted(glob.glob(f"{seq}*.jpg"))
            q = queue.Queue(maxsize=8)
            threading.Thread(target=reader, args=(q, files), daemon=True).start()
            rows, t_inf, t_post, fi = [], 0.0, 0.0, 0
            while (item := q.get()) is not None:
                fi += 1; x, s = item
                t0 = time.perf_counter(); res = pipe.infer({in_name: x})
                t1 = time.perf_counter(); b, sc, c = rb.postprocess(res, outs, s)
                t2 = time.perf_counter(); t_inf += t1 - t0; t_post += t2 - t1
                rows.append(np.c_[np.full(len(b), fi), b, sc, c])
            np.save(f"{out}/{name}.npy", np.concatenate(rows).astype(np.float32))
            print(f"{name} frames={fi} infer={t_inf/fi*1000:.1f}ms post={t_post/fi*1000:.1f}ms", flush=True)
