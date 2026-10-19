# Hailo-8 실시간 데모: 1=YOLO26s  2=YOLO26n  3=YOLOv8s  q=종료
import time, threading, queue, cv2, numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
from detect_labels import COCO

MODELS = {ord("1"): ("YOLO26s", "yolo26s.hef"),
          ord("2"): ("YOLO26n", "yolo26n.hef"),
          ord("3"): ("YOLOv8s", "/usr/share/hailo-models/yolov8s_h8.hef")}
VIDEO, CONF, WIN = "example.mp4", 0.35, "Hailo-8 live"
COLORS = np.random.default_rng(3).integers(60, 255, (80, 3)).tolist()

def decode_yolo26(res):
    # NMS-free 헤드: 스케일별 (박스 ltrb 거리, 클래스 로짓) 출력
    outs = sorted(res.values(), key=lambda a: -a.shape[1])
    boxes4 = [o[0] for o in outs if o.shape[-1] == 4]
    logits = [o[0] for o in outs if o.shape[-1] == 80]
    dets = []
    for b, c in zip(boxes4, logits):
        g = b.shape[0]; stride = 640 / g
        score = 1 / (1 + np.exp(-c.max(-1)))
        ys, xs = np.nonzero(score > CONF)
        for y, x in zip(ys, xs):
            l, t, r, bt = b[y, x]
            cx, cy = x + 0.5, y + 0.5
            dets.append((int(c[y, x].argmax()), float(score[y, x]),
                         (cx - l) * stride, (cy - t) * stride, (cx + r) * stride, (cy + bt) * stride))
    return dets

def decode_v8(res):
    dets = []
    for cls, boxes in enumerate(next(iter(res.values()))[0]):
        for y1, x1, y2, x2, sc in boxes:
            if sc >= CONF: dets.append((cls, float(sc), x1*640, y1*640, x2*640, y2*640))
    return dets

def cpu_temp():
    return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000


def reader(cap, q_in, stop):
    # 스레드 1: 영상 읽기 + 전처리
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        h, w = frame.shape[:2]; s = 640 / max(h, w)
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[:int(h*s), :int(w*s)] = cv2.resize(frame, (int(w*s), int(h*s)))
        q_in.put((frame, s, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]))

def worker(pipe, in_name, decode, q_in, q_out, stop):
    # 스레드 2: Hailo 추론 + 후처리
    while not stop.is_set():
        try: frame, s, x = q_in.get(timeout=0.5)
        except queue.Empty: continue
        t0 = time.time(); res = pipe.infer({in_name: x}); inf_ms = (time.time() - t0) * 1000
        q_out.put((frame, s, decode(res), inf_ms))

cap = cv2.VideoCapture(VIDEO)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1280, 720)
key = ord("1")
with VDevice() as target:
    phys = target.get_physical_devices()[0]
    while key in MODELS:
        name, path = MODELS[key]; key = None
        hef = HEF(path)
        ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        in_name = hef.get_input_vstream_infos()[0].name
        decode = decode_v8 if "v8" in name else decode_yolo26
        fps, n, last, ht, ct = 0.0, 0, time.time(), 0.0, 0.0
        with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
            stop = threading.Event(); q_in, q_out = queue.Queue(3), queue.Queue(3)
            threads = [threading.Thread(target=reader, args=(cap, q_in, stop), daemon=True),
                       threading.Thread(target=worker, args=(pipe, in_name, decode, q_in, q_out, stop), daemon=True)]
            for t in threads: t.start()
            # 메인 스레드: 그리기 + 화면 출력
            while True:
                try: frame, s, dets, inf_ms = q_out.get(timeout=2)
                except queue.Empty: continue
                for cls, sc, x1, y1, x2, y2 in dets:
                    p1, p2 = (int(x1/s), int(y1/s)), (int(x2/s), int(y2/s)); col = COLORS[cls]
                    cv2.rectangle(frame, p1, p2, col, 2)
                    cv2.putText(frame, f"{COCO[cls]} {sc:.2f}", (p1[0], max(p1[1]-6, 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                n += 1
                if time.time() - last > 1:
                    fps, n, last = n / (time.time() - last), 0, time.time()
                    try: ht = phys.control.get_chip_temperature().ts0_temperature
                    except Exception: pass
                    ct = cpu_temp()
                info = (f"{name} | FPS {fps:.0f} | infer {inf_ms:.1f}ms | objs {len(dets)} | "
                        f"Hailo {ht:.0f}C CPU {ct:.0f}C | 1:26s 2:26n 3:v8s q:quit")
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imshow(WIN, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27): key = None; break
                if k in MODELS and MODELS[k][0] != name: key = k; break
            stop.set()
            for q in (q_in, q_out):
                while not q.empty(): q.get_nowait()
            for t in threads: t.join(timeout=2)
cv2.destroyAllWindows()
