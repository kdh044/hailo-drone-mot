import sys, os, glob, time, threading, queue, cv2, numpy as np, psutil
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb

# usage: python3 det_speed.py <hef> <image dir> <full|reduced>
# 실제 검출 파이프라인 속도: 읽기 스레드(사진 열기+640 줄이기) -> Hailo 추론 -> 후처리. 화면 표시 없음.
hef_path, img_dir, mode = sys.argv[1:4]
rb.CONF = 0.05
REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}
images = sorted(glob.glob(f"{img_dir}/*.jpg"))
widths = {p: cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[1] * 8 for p in images}

def factor(w):
    f = 1
    while f < 4 and w / (f * 2) >= rb.SIZE: f *= 2
    return f

def reader(q):
    for p in images:
        if mode == "full": img, f = cv2.imread(p), 1
        else:
            f = factor(widths[p]); img = cv2.imread(p, REDUCED[f]) if f > 1 else cv2.imread(p)
        x, s = rb.preprocess(img); q.put((x, s / f))
    q.put(None)

hef = rb.HEF(hef_path); outs = rb.get_outs(hef)
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for _ in range(10): pipe.infer({in_name: np.zeros((1, 640, 640, 3), np.uint8)})  # 예열
        q = queue.Queue(maxsize=8); threading.Thread(target=reader, args=(q,), daemon=True).start()
        psutil.cpu_percent(); t_inf = t_post = t_wait = 0.0; n = 0; t_start = time.perf_counter()
        while True:
            t0 = time.perf_counter(); item = q.get(); t_wait += time.perf_counter() - t0
            if item is None: break
            x, s = item
            t1 = time.perf_counter(); res = pipe.infer({in_name: x}); t2 = time.perf_counter()
            rb.postprocess(res, outs, s); t3 = time.perf_counter()
            t_inf += t2 - t1; t_post += t3 - t2; n += 1
        wall = time.perf_counter() - t_start; cpu = psutil.cpu_percent()
print(f"{os.path.basename(hef_path):26s} {mode:7s} 실제 {n / wall:5.1f} FPS | 추론 {n / t_inf:5.1f} FPS 후처리 {n / t_post:5.1f} FPS | "
      f"사진 기다린 시간 {t_wait / wall * 100:4.1f}% | Hailo 사용 {t_inf / wall * 100:4.1f}% | CPU {cpu:.0f}%")
