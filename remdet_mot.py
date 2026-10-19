import sys, glob, os, json, time, collections, cv2, numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from trackers.eval import evaluate_mot_sequences
import remdet_bench as rb
import trackers.eval.evaluate as _ev
_load = _ev.load_mot_file
_ev.load_mot_file = lambda p: _load(p) if os.path.getsize(p) > 1 else {}  # empty result = no predictions

# usage: ~/track-venv/bin/python remdet_mot.py <hef> [<hef> ...]
#   1) Hailo inference on VisDrone MOT val -> cached detections (mot_out/<model>/dets.npz)
#   2) ByteTrack per class group -> MOTChallenge txt (mot_out/<model>/<group>/<seq>.txt)
#   3) CLEAR / HOTA / Identity vs GT, person and vehicle evaluated separately
DATA = os.path.expanduser("~/datasets/visdrone/mot-val-hf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mot_out")
rb.CONF = 0.1  # keep low-score boxes for ByteTrack's second association
GROUPS = {"person": [0, 1], "vehicle": [3, 4, 5, 8]}
GT_LABELS = {"person": {"pedestrian", "people"}, "vehicle": {"car", "van", "truck", "bus"}}

def build_gt():
    """GT -> MOTChallenge txt: target group = class 1, ignored_region = class 8 (distractor)."""
    samples = json.load(open(f"{DATA}/samples.json"))["samples"]
    seqs = collections.defaultdict(list)
    for s in samples: seqs[s["scene_id"]].append(s)
    for g, labels in GT_LABELS.items():
        os.makedirs(f"{DATA}/gt_{g}", exist_ok=True)
        for seq, frames in seqs.items():
            lines = []
            for s in frames:
                fn = s["frame_number"]
                if not os.path.exists(f"{DATA}/sequences/{seq}/{fn:07d}.jpg"): continue  # missing image
                W, H = s["metadata"]["width"], s["metadata"]["height"]
                dets = s["detections"]; dets = dets.get("detections", []) if isinstance(dets, dict) else (dets or [])
                for o in dets:
                    cls = 1 if o["label"] in labels else 8 if o["label"] == "ignored_region" else None
                    if cls is None: continue
                    x, y, w, h = o["bounding_box"]
                    lines.append(f"{fn},{o['index']},{x*W:.2f},{y*H:.2f},{w*W:.2f},{h*H:.2f},1,{cls},1")
            open(f"{DATA}/gt_{g}/{seq}.txt", "w").write("\n".join(lines) + "\n")
    return sorted(seqs)

def detect(hef_path, seqs, cache):
    from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
        InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
    hef = rb.HEF(hef_path); outs = rb.get_outs(hef); out, t_inf = {}, []
    with VDevice() as target:
        ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        in_name = hef.get_input_vstream_infos()[0].name
        with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
            for seq in seqs:
                for p in sorted(glob.glob(f"{DATA}/sequences/{seq}/*.jpg")):
                    img = cv2.imread(p); t0 = time.perf_counter()
                    x, s = rb.preprocess(img); b, sc, c = rb.postprocess(pipe.infer({in_name: x}), outs, s)
                    t_inf.append(time.perf_counter() - t0)
                    fn = int(os.path.basename(p)[:-4])
                    out[f"{seq}/{fn}"] = np.c_[b, sc, c].astype(np.float32)
                print(f"  det {seq} done", flush=True)
    np.savez_compressed(cache, **out)
    print(f"  infer+post mean {np.mean(t_inf)*1000:.1f} ms/frame")

def track(dets, seqs, odir):
    t_trk = []
    for g, ids in GROUPS.items():
        os.makedirs(f"{odir}/{g}", exist_ok=True)
        for seq in seqs:
            tr = ByteTrackTracker(frame_rate=30, track_activation_threshold=0.4,
                                  high_conf_det_threshold=0.4, minimum_consecutive_frames=2)
            frames = sorted(int(k.split("/")[1]) for k in dets if k.startswith(seq + "/"))
            lines = []
            for fn in frames:
                d = dets[f"{seq}/{fn}"]; d = d[np.isin(d[:, 5].astype(int), ids)]
                t0 = time.perf_counter()
                r = tr.update(sv.Detections(xyxy=d[:, :4], confidence=d[:, 4], class_id=d[:, 5].astype(int)))
                t_trk.append(time.perf_counter() - t0)
                for (x1, y1, x2, y2), tid, cf in zip(r.xyxy, r.tracker_id, r.confidence):
                    if tid < 0: continue
                    lines.append(f"{fn},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{cf:.3f},-1,-1,-1")
            open(f"{odir}/{g}/{seq}.txt", "w").write("\n".join(lines) + "\n")
    print(f"  ByteTrack mean {np.mean(t_trk)*1000:.1f} ms/update (2 groups per frame)")

if __name__ == "__main__":
    seqs = build_gt(); summary = []
    for hef_path in sys.argv[1:]:
        name = os.path.basename(hef_path).replace(".hef", ""); odir = f"{OUT}/{name}"
        os.makedirs(odir, exist_ok=True); cache = f"{odir}/dets.npz"
        print(f"== {name}", flush=True)
        if not os.path.exists(cache): detect(hef_path, seqs, cache)
        track(dict(np.load(cache)), seqs, odir)
        for g in GROUPS:
            res = evaluate_mot_sequences(f"{DATA}/gt_{g}", f"{odir}/{g}", metrics=["CLEAR", "HOTA", "Identity"])
            print(f"-- {name} / {g}"); print(res.table(columns=["HOTA", "MOTA", "IDF1", "IDSW"]))
            c = res.aggregate
            summary.append((name, g, c.HOTA.HOTA, c.CLEAR.MOTA, c.Identity.IDF1, c.CLEAR.IDSW))
    print("\n== SUMMARY (COMBINED, 7 val seqs)")
    print(f"{'model':24s}{'group':9s}{'HOTA':>7s}{'MOTA':>8s}{'IDF1':>7s}{'IDSW':>6s}")
    for n, g, h, m, i, s in summary:
        print(f"{n:24s}{g:9s}{h*100:7.1f}{m*100:8.1f}{i*100:7.1f}{s:6d}")
