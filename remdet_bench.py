import sys, time, glob, cv2, numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)

# usage: python3 remdet_bench.py <hef> <image_dir> [N]
VISDRONE = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
SIZE, CONF, IOU, REG = 640, 0.25, 0.6, 16

def preprocess(img):
    h, w = img.shape[:2]
    s = SIZE / max(h, w); nh, nw = int(h*s), int(w*s)
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None], s

def postprocess(res, outs, s):
    boxes, scores, classes = [], [], []
    for cls_name, reg_name in outs:
        cls = res[cls_name][0]; reg = res[reg_name][0]
        H, W = cls.shape[:2]; stride = SIZE // H
        cls = cls.reshape(-1, cls.shape[-1])
        if cls.min() < 0 or cls.max() > 1:  # logits -> prob
            cls = 1 / (1 + np.exp(-cls))
        sc = cls.max(1); keep = sc > CONF
        if not keep.any(): continue
        c = cls.argmax(1)[keep]; sc = sc[keep]
        r = reg.reshape(-1, 4, REG)[keep]
        r = np.exp(r - r.max(-1, keepdims=True)); r /= r.sum(-1, keepdims=True)
        d = (r * np.arange(REG)).sum(-1) * stride  # l, t, r, b
        idx = np.nonzero(keep)[0]
        cx = (idx % W + 0.5) * stride; cy = (idx // W + 0.5) * stride
        boxes.append(np.stack([cx-d[:,0], cy-d[:,1], cx+d[:,2], cy+d[:,3]], 1) / s)
        scores.append(sc); classes.append(c)
    if not boxes: return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b = np.concatenate(boxes); sc = np.concatenate(scores); c = np.concatenate(classes)
    xywh = np.c_[b[:, :2] + c[:, None] * 4096, b[:, 2:] - b[:, :2]]  # class-aware offset
    k = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), CONF, IOU)
    k = np.array(k).reshape(-1)
    return b[k], sc[k], c[k]

def get_outs(hef):
    shapes = {o.name: o.shape for o in hef.get_output_vstream_infos()}
    cls_outs = sorted([n for n in shapes if shapes[n][-1] == len(VISDRONE)], key=lambda n: -shapes[n][0])
    reg_outs = sorted([n for n in shapes if shapes[n][-1] == 4*REG], key=lambda n: -shapes[n][0])
    return list(zip(cls_outs, reg_outs))

if __name__ == "__main__":
    hef_path, img_dir = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    images = sorted(glob.glob(f"{img_dir}/*.jpg"))[:N]
    hef = HEF(hef_path)
    outs = get_outs(hef)

    T = {"read": [], "pre": [], "infer": [], "post": [], "total": []}; ndet = []
    with VDevice() as target:
        cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cp)[0]
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        in_name = hef.get_input_vstream_infos()[0].name
        with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
            for i, path in enumerate(images):
                t0 = time.perf_counter(); img = cv2.imread(path)
                t1 = time.perf_counter(); x, s = preprocess(img)
                t2 = time.perf_counter(); res = pipe.infer({in_name: x})
                t3 = time.perf_counter(); b, sc, c = postprocess(res, outs, s)
                t4 = time.perf_counter()
                if i < 5: continue  # warmup
                for k, v in zip(T, [t1-t0, t2-t1, t3-t2, t4-t3, t4-t0]): T[k].append(v*1000)
                ndet.append(len(b))
            img = cv2.imread(images[0]); x, s = preprocess(img)
            b, sc, c = postprocess(pipe.infer({in_name: x}), outs, s)
            for (x1, y1, x2, y2), p, k in zip(b.astype(int), sc, c):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{VISDRONE[k]} {p:.2f}", (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(hef_path.rsplit("/", 1)[-1].replace(".hef", "_vis.jpg"), img)

    print(f"{hef_path.rsplit('/',1)[-1]}  images={len(T['total'])}  avg dets={np.mean(ndet):.1f} (max {max(ndet)})")
    for k, v in T.items():
        print(f"  {k:6s} mean {np.mean(v):6.2f} ms   p95 {np.percentile(v,95):6.2f} ms")
    print(f"  end-to-end FPS (sync, no read): {1000/np.mean(np.array(T['total'])-np.array(T['read'])):.1f}")
    print(f"  end-to-end FPS (incl. jpg read): {1000/np.mean(T['total']):.1f}")
