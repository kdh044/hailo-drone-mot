import os, json, glob, numpy as np

ROOT = os.path.expanduser("~/hailo-test/mot")
SEQ_ROOT = os.path.expanduser("~/datasets/visdrone/mot-val-hf/sequences")
SAMPLES = os.path.expanduser("~/datasets/visdrone/mot-val-hf/samples.json")
# eval super-classes; detector classes are VisDrone DET ids (rb.VISDRONE order)
GROUPS = {"person": {"gt": ["pedestrian", "people"], "det": [0, 1]},
          "vehicle": {"gt": ["car", "van", "truck", "bus"], "det": [3, 4, 5, 8]}}
SEQS = sorted(os.path.basename(d) for d in glob.glob(f"{SEQ_ROOT}/*"))


def load_gt():
    """-> {seq: {"size": (w, h), "n": frames, "objs": [frame, id, x1, y1, x2, y2, label], "ign": [frame, x1, y1, x2, y2]}}"""
    out = {}
    for s in json.load(open(SAMPLES))["samples"]:
        md, seq = s["metadata"], s["scene_id"]
        w, h = md["width"], md["height"]
        o = out.setdefault(seq, {"size": (w, h), "n": 0, "objs": [], "ign": []})
        f = s["frame_number"]; o["n"] = max(o["n"], f)
        for d in s["detections"] or []:
            x, y, bw, bh = d["bounding_box"]
            box = [x * w, y * h, (x + bw) * w, (y + bh) * h]
            if d["label"] in ("ignored_region", "others"):
                o["ign"].append([f, *box])
            else:
                o["objs"].append([f, d["index"], *box, d["label"]])
    return out


def ignored_mask(boxes, ign_boxes, thr=0.5):
    """True for boxes whose area is > thr inside the union of ignored regions (approximated per region)."""
    if len(boxes) == 0 or len(ign_boxes) == 0:
        return np.zeros(len(boxes), bool)
    b, g = np.asarray(boxes, float)[:, None], np.asarray(ign_boxes, float)[None]
    iw = (np.minimum(b[..., 2], g[..., 2]) - np.maximum(b[..., 0], g[..., 0])).clip(0)
    ih = (np.minimum(b[..., 3], g[..., 3]) - np.maximum(b[..., 1], g[..., 1])).clip(0)
    area = ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])).clip(1e-6)
    return ((iw * ih) / area).max(1) > thr


def write_mot(path, rows):
    """rows: frame, id, x1, y1, x2, y2, score -> MOTChallenge txt"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, x1, y1, x2, y2, sc in rows:
            f.write(f"{int(fr)},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{sc:.4f},1,1\n")
