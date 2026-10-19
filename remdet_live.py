import sys, glob, time, threading, queue, cv2, numpy as np
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
from remdet_bench import preprocess, postprocess, get_outs, VISDRONE

# usage: python3 remdet_live.py <hef> <video file | image dir> [ms per frame, 0 = max speed]
#   q: quit, space: pause/resume
hef_path, source = sys.argv[1], sys.argv[2]
DELAY = int(sys.argv[3]) if len(sys.argv) > 3 else 500

def read_frames(q):
    # decode in a background thread so it overlaps with Hailo inference
    if source.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(source)
        while (ok := cap.read())[0]: q.put(ok[1])
    else:
        for p in sorted(glob.glob(f"{source}/*.jpg")):
            q.put(cv2.imread(p))
    q.put(None)

q = queue.Queue(maxsize=8)
threading.Thread(target=read_frames, args=(q,), daemon=True).start()
colors = [tuple(int(c) for c in np.random.RandomState(i).randint(60, 255, 3)) for i in range(10)]
cv2.namedWindow("RemDet", cv2.WINDOW_NORMAL); cv2.resizeWindow("RemDet", 1280, 720)
hef = HEF(hef_path); outs = get_outs(hef)
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        fps, paused = 0, False
        while (img := q.get()) is not None:
            t = time.perf_counter()
            x, s = preprocess(img)
            b, sc, c = postprocess(pipe.infer({in_name: x}), outs, s)
            ms = (time.perf_counter() - t) * 1000
            fps = 0.9*fps + 100/ms if fps else 1000/ms
            for (x1, y1, x2, y2), p, k in zip(b.astype(int), sc, c):
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[k], 1)
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(img, f"{hef_path.rsplit('/',1)[-1][:-4]}  {fps:.1f} FPS  pre+infer+post {ms:.1f}ms  dets {len(b)}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("RemDet", img)
            # FPS shown is processing speed only; DELAY just slows playback for viewing
            key = cv2.waitKey(0 if paused else max(1, DELAY - int((time.perf_counter() - t) * 1000))) & 0xFF
            if key == ord("q"): break
            if key == ord(" "): paused = not paused
cv2.destroyAllWindows()
