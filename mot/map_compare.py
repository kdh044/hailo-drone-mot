import sys, os, json, glob, time, cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb

# usage: python3 map_compare.py <hef> <image dir> <out prefix>
# 같은 이미지로 두 가지 열기 방식(원본 열기 / 작게 열기) 예측을 만들고 열기 속도도 잰다
hef_path, img_dir, out = sys.argv[1:4]
rb.CONF = 0.001
REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}
images = sorted(glob.glob(f"{img_dir}/*.jpg"))
hef = rb.HEF(hef_path); outs = rb.get_outs(hef)

def factor(w):
    f = 1
    while f < 4 and w / (f * 2) >= rb.SIZE: f *= 2
    return f

with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for mode in ("full", "reduced"):
            preds, t_open = [], 0.0
            for path in images:
                w = int(cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[1] * 8) if mode == "reduced" else 0
                t0 = time.perf_counter()
                if mode == "full":
                    img = cv2.imread(path); f = 1
                else:
                    f = factor(w); img = cv2.imread(path, REDUCED[f]) if f > 1 else cv2.imread(path)
                x, s = rb.preprocess(img)
                t_open += time.perf_counter() - t0
                b, sc, c = rb.postprocess(pipe.infer({in_name: x}), outs, s / f)
                o = np.argsort(-sc)[:300]; iid = os.path.basename(path)[:-4]
                for (x1, y1, x2, y2), p, k in zip(b[o], sc[o], c[o]):
                    preds.append({"image_id": iid, "category_id": int(k) + 1,
                                  "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(p)})
            json.dump(preds, open(f"{out}_{mode}.json", "w"))
            print(f"{mode}: 열기+640 줄이기 {len(images) / t_open:.1f} FPS", flush=True)
