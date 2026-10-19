import sys, json, glob, os, cv2, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "0"] + sys.argv[3:]
import importlib.util
# reuse preprocess/postprocess from remdet_bench without running its benchmark
src = open(os.path.join(os.path.dirname(__file__), "remdet_bench.py")).read().split("T = {")[0]
g = {}; exec(src, g)
g["CONF"] = 0.001
from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
hef, img_dir = g["hef"], sys.argv[2]
images = sorted(glob.glob(f"{img_dir}/*.jpg")); preds = []
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for path in images:
            img = cv2.imread(path); x, s = g["preprocess"](img)
            b, sc, c = g["postprocess"](pipe.infer({in_name: x}), g["outs"], s)
            o = np.argsort(-sc)[:300]
            iid = os.path.basename(path)[:-4]
            for (x1, y1, x2, y2), p, k in zip(b[o], sc[o], c[o]):
                preds.append({"image_id": iid, "category_id": int(k)+1, "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(p)})
json.dump(preds, open(sys.argv[3], "w"))
print(len(images), "images", len(preds), "dets")
