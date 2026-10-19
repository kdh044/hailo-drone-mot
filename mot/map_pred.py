import sys, os, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb

# usage: python3 map_pred.py <hef> <image dir> <out.json>   (then ~/hailo-test/coco_eval.py for mAP)
hef_path, img_dir, out = sys.argv[1:4]
rb.CONF = 0.001
hef = rb.HEF(hef_path); outs = rb.get_outs(hef); preds = []
images = sorted(glob.glob(f"{img_dir}/*.jpg"))
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for path in images:
            x, s = rb.preprocess(cv2.imread(path))
            b, sc, c = rb.postprocess(pipe.infer({in_name: x}), outs, s)
            o = np.argsort(-sc)[:300]; iid = os.path.basename(path)[:-4]
            for (x1, y1, x2, y2), p, k in zip(b[o], sc[o], c[o]):
                preds.append({"image_id": iid, "category_id": int(k) + 1,
                              "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(p)})
json.dump(preds, open(out, "w"))
print(len(images), "images", len(preds), "dets")
