import glob, time, cv2, numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
from detect_labels import COCO

HEF_PATH = "/usr/share/hailo-models/yolov8s_h8.hef"
IMGS = [cv2.imread(p) for p in sorted(glob.glob("/usr/share/rpd-wallpaper/*.jpg"))]
IMGS = [i for i in IMGS if i is not None]
W, H = 960, 540
COLORS = np.random.default_rng(1).integers(60, 255, (80, 3)).tolist()

def frames():
    # 가상 카메라: 각 사진 위를 줌인하며 좌우로 이동
    while True:
        for img in IMGS:
            ih, iw = img.shape[:2]
            for k in range(150):
                t = k / 149
                zoom = 1.0 + 0.8 * t
                cw, ch = int(iw / zoom), int(ih / zoom)
                cx = int((iw - cw) * (0.5 + 0.5 * np.sin(t * np.pi * 2)))
                cy = int((ih - ch) * 0.5)
                yield cv2.resize(img[cy:cy+ch, cx:cx+cw], (W, H))

def temp(dev):
    try: return dev.control.get_chip_temperature().ts0_temperature
    except Exception: return float("nan")

def cpu_temp():
    return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000

hef = HEF(HEF_PATH)
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    out_name = hef.get_output_vstream_infos()[0].name
    phys = target.get_physical_devices()[0]
    cv2.namedWindow("Hailo-8 YOLOv8s", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hailo-8 YOLOv8s", W, H)
    fps, last, ht, ct, n = 0.0, time.time(), 0, 0, 0
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for frame in frames():
            s = 640 / W
            canvas = np.full((640, 640, 3), 114, np.uint8)
            canvas[:int(H*s), :640] = cv2.resize(frame, (640, int(H*s)))
            t0 = time.time()
            dets = pipe.infer({in_name: cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]})[out_name][0]
            inf_ms = (time.time() - t0) * 1000
            for cls, boxes in enumerate(dets):
                for y1, x1, y2, x2, sc in boxes:
                    if sc < 0.35: continue
                    p1, p2 = (int(x1*640/s), int(y1*640/s)), (int(x2*640/s), int(y2*640/s))
                    c = COLORS[cls]
                    cv2.rectangle(frame, p1, p2, c, 2)
                    cv2.putText(frame, f"{COCO[cls]} {sc:.2f}", (p1[0], max(p1[1]-6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            n += 1
            if time.time() - last > 1:
                fps, last, n = n / (time.time() - last), time.time(), 0
                ht, ct = temp(phys), cpu_temp()
            info = f"FPS {fps:.0f} | Hailo infer {inf_ms:.1f} ms | Hailo {ht:.0f}C  CPU {ct:.0f}C | q: quit"
            cv2.rectangle(frame, (0, 0), (W, 30), (0, 0, 0), -1)
            cv2.putText(frame, info, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Hailo-8 YOLOv8s", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
cv2.destroyAllWindows()
