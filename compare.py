# CPU(ONNX Runtime) vs Hailo-8 같은 YOLOv8s 나란히 비교. q=종료
# 사용법: compare.py [cpu|hailo]  (생략하면 둘 다)
import sys, time, threading, cv2, numpy as np, onnxruntime as ort
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
from detect_labels import COCO

VIDEO, CONF, PW, PH = "example.mp4", 0.35, 800, 450
COLORS = np.random.default_rng(3).integers(60, 255, (80, 3)).tolist()
latest = {"CPU": None, "Hailo-8": None}
stop = threading.Event()

def letterbox(frame):
    h, w = frame.shape[:2]; s = 640 / max(h, w)
    c = np.full((640, 640, 3), 114, np.uint8)
    c[:int(h*s), :int(w*s)] = cv2.resize(frame, (int(w*s), int(h*s)))
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB), s

def draw(frame, dets, s, label, fps, ms, color):
    for cls, sc, x1, y1, x2, y2 in dets:
        p1, p2 = (int(x1/s), int(y1/s)), (int(x2/s), int(y2/s))
        cv2.rectangle(frame, p1, p2, COLORS[cls], 2)
        cv2.putText(frame, f"{COCO[cls]} {sc:.2f}", (p1[0], max(p1[1]-6, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
    frame = cv2.resize(frame, (PW, PH))
    cv2.rectangle(frame, (0, 0), (PW, 60), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS {fps:.1f}   infer {ms:.0f} ms", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def loop(key, label, color, infer):
    cap = cv2.VideoCapture(VIDEO); t_last, n, fps = time.time(), 0, 0.0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        x, s = letterbox(frame)
        t0 = time.time(); dets = infer(x); ms = (time.time() - t0) * 1000
        n += 1
        if time.time() - t_last > 1 or fps == 0:
            fps, n, t_last = n / (time.time() - t_last), 0, time.time()
        latest[key] = draw(frame, dets, s, label, fps, ms, color)

def cpu_worker():
    sess = ort.InferenceSession("yolov8s_onnx/yolov8s.onnx", providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    def infer(x):
        out = sess.run(None, {name: (x.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)})[0][0].T
        sc = out[:, 4:]; cls = sc.argmax(1); conf = sc.max(1); m = conf > CONF
        b, cls, conf = out[m, :4], cls[m], conf[m]
        xywh = np.c_[b[:, 0] - b[:, 2]/2, b[:, 1] - b[:, 3]/2, b[:, 2], b[:, 3]]
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), CONF, 0.7)
        return [(int(cls[i]), float(conf[i]), *xywh[i, :2], *(xywh[i, :2] + xywh[i, 2:])) for i in np.array(keep).flatten()]
    loop("CPU", "CPU only (Raspberry Pi 5, ONNX Runtime)", (0, 140, 255), infer)

def hailo_worker():
    hef = HEF("/usr/share/hailo-models/yolov8s_h8.hef")
    with VDevice() as t:
        ng = t.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        name = hef.get_input_vstream_infos()[0].name
        with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
            def infer(x):
                res = next(iter(pipe.infer({name: x[None]}).values()))[0]
                return [(c, float(sc), x1*640, y1*640, x2*640, y2*640)
                        for c, boxes in enumerate(res) for y1, x1, y2, x2, sc in boxes if sc >= CONF]
            loop("Hailo-8", "Hailo-8 accelerated (HailoRT)", (0, 255, 0), infer)

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
KEYS = {"cpu": ["CPU"], "hailo": ["Hailo-8"]}.get(MODE, ["CPU", "Hailo-8"])
workers = {"CPU": cpu_worker, "Hailo-8": hailo_worker}
threads = [threading.Thread(target=workers[k], daemon=True) for k in KEYS]
for th in threads: th.start()
WIN = "YOLOv8s: " + " vs ".join(KEYS)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, PW*len(KEYS), PH)
blank = np.zeros((PH, PW, 3), np.uint8)
while True:
    panels = [latest[k] if latest[k] is not None else blank for k in KEYS]
    cv2.imshow(WIN, np.hstack(panels))
    if cv2.waitKey(30) & 0xFF in (ord("q"), 27): break
stop.set(); time.sleep(0.5); cv2.destroyAllWindows()
