import sys, time, cv2, numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

hef_path = "/usr/share/hailo-models/yolov8s_h8.hef"
images = sys.argv[1:]
hef = HEF(hef_path)
with VDevice() as target:
    cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = target.configure(hef, cp)[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    out_name = hef.get_output_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        for path in images:
            img = cv2.imread(path); h, w = img.shape[:2]
            s = 640 / max(h, w); nh, nw = int(h*s), int(w*s)
            canvas = np.full((640, 640, 3), 114, np.uint8)
            canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
            x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
            t = time.time(); res = pipe.infer({in_name: x}); dt = (time.time()-t)*1000
            dets = res[out_name][0]
            found = []
            for cls, boxes in enumerate(dets):
                for y1, x1, y2, x2, sc in boxes:
                    if sc < 0.4: continue
                    p1 = (int(x1*640/s), int(y1*640/s)); p2 = (int(x2*640/s), int(y2*640/s))
                    cv2.rectangle(img, p1, p2, (0,255,0), 3)
                    cv2.putText(img, f"{COCO[cls]} {sc:.2f}", (p1[0], max(p1[1]-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    found.append(f"{COCO[cls]}({sc:.2f})")
            out = path.rsplit("/",1)[-1].replace(".jpg", "_det.jpg")
            cv2.imwrite(out, img)
            print(f"{path.rsplit('/',1)[-1]}: {dt:.1f}ms  {', '.join(found) or '탐지 없음'}")
