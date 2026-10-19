import sys, glob, os, time, threading, queue, cv2, numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb

# usage: ~/track-venv/bin/python remdet_track.py <hef> <sequences dir | one sequence dir> [ms per frame, 0 = max speed]
#   q: quit, space: pause/resume, n: next sequence
hef_path, source = sys.argv[1], sys.argv[2]
DELAY = int(sys.argv[3]) if len(sys.argv) > 3 else 33
rb.CONF = 0.1  # keep low-score boxes for ByteTrack's second association
# VisDrone: 0 pedestrian, 1 people -> person / 3 car, 4 van, 5 truck, 8 bus -> vehicle
GROUPS = {"person": [0, 1], "vehicle": [3, 4, 5, 8]}
COLOR = {"person": (0, 200, 255), "vehicle": (0, 255, 0)}

seqs = sorted(d for d in glob.glob(f"{source}/*") if os.path.isdir(d)) or [source]

def read_frames(q, seq):
    for p in sorted(glob.glob(f"{seq}/*.jpg")): q.put(cv2.imread(p))
    q.put(None)

def id_color(tid):
    return tuple(int(c) for c in np.random.RandomState(tid).randint(60, 255, 3))

cv2.namedWindow("RemDet + ByteTrack", cv2.WINDOW_NORMAL); cv2.resizeWindow("RemDet + ByteTrack", 1344, 756)
hef = rb.HEF(hef_path); outs = rb.get_outs(hef)
with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        quit_all = False
        for seq in seqs:
            trackers = {g: ByteTrackTracker(frame_rate=30, track_activation_threshold=0.4,
                                            high_conf_det_threshold=0.4, minimum_consecutive_frames=2)
                        for g in GROUPS}
            q = queue.Queue(maxsize=8)
            threading.Thread(target=read_frames, args=(q, seq), daemon=True).start()
            fps, paused, fi = 0, False, 0
            while (img := q.get()) is not None:
                fi += 1
                t0 = time.perf_counter()
                x, s = rb.preprocess(img)
                res = pipe.infer({in_name: x})
                t1 = time.perf_counter()
                b, sc, c = rb.postprocess(res, outs, s)
                t2 = time.perf_counter()
                n_tracks = 0
                for g, ids in GROUPS.items():
                    m = np.isin(c, ids)
                    det = sv.Detections(xyxy=b[m].astype(np.float32), confidence=sc[m].astype(np.float32), class_id=c[m])
                    tr = trackers[g].update(det)
                    for (x1, y1, x2, y2), tid in zip(tr.xyxy.astype(int), tr.tracker_id):
                        if tid < 0: continue
                        n_tracks += 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), id_color(tid), 2)
                        cv2.putText(img, f"{g[0].upper()}{tid}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[g], 1)
                t3 = time.perf_counter()
                ms = (t3 - t0) * 1000; fps = 0.9*fps + 100/ms if fps else 1000/ms
                cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(img, f"{os.path.basename(seq)}  frame {fi}  tracks {n_tracks}  P=person V=vehicle",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(img, f"{fps:.1f} FPS | infer {(t1-t0)*1000:.1f}  post {(t2-t1)*1000:.1f}  track {(t3-t2)*1000:.1f} ms",
                            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow("RemDet + ByteTrack", img)
                key = cv2.waitKey(0 if paused else max(1, DELAY - int(ms))) & 0xFF
                if key == ord("q"): quit_all = True; break
                if key == ord("n"): break
                if key == ord(" "): paused = not paused
            if quit_all: break
cv2.destroyAllWindows()
