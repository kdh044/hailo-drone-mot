import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, glob, time, pickle, argparse, json, cv2, numpy as np
cv2.setNumThreads(1)  # one thread per worker; parallelism comes from the process pool
from multiprocessing import Pool
from common import ROOT, SEQ_ROOT, SEQS, GROUPS, ignored_mask, write_mot

# usage: ~/mot-venv/bin/python run_trackers.py --dets tiny s oracle --trackers bytetrack ocsort ... [--jobs 4]
# one tracker instance per (sequence, class group); output goes to data/trackers/VD<group>-val/<dets>__<tracker>/data/
TRACKERS = {
    # motion-only (no appearance model)
    "sort":        ("rf", "SORTTracker", {}),
    "rf-bytetrack": ("rf", "ByteTrackTracker", {}),  # roboflow impl used by the live viewer
    "cbiou":       ("rf", "CBIoUTracker", {}),
    "mcbyte":      ("rf", "McByteTracker", {}),
    "bytetrack":   ("bm", "ByteTrack", {}),
    "ocsort":      ("bm", "OcSort", {}),
    "ocsort-byte": ("bm", "OcSort", {"use_byte": True}),
    "sfsort":      ("bm", "SFSORT", {}),
    "botsort":     ("bm", "BotSort", {"use_embeddings": False, "use_cmc": False}),
    "botsort-cmc": ("bm", "BotSort", {"use_embeddings": False, "use_cmc": True, "cmc_method": "ecc"}),
    "deepocsort-cmc": ("bm", "DeepOcSort", {"use_embeddings": False}),
    "hybridsort":  ("bm", "HybridSort", {"use_embeddings": False}),
    "boosttrack":  ("bm", "BoostTrack", {"use_embeddings": False}),
    # drone-specific (ext_trackers.py adapters, ReID CNN removed)
    "sftrack":     ("ext", "SFTrackAdapter", {}),
    "detecttrack-uav": ("ext", "DetectTrackUAVAdapter", {}),
    "detecttrack-uav-cmc": ("ext", "DetectTrackUAVAdapter", {"cmc_method": "sparseOptFlow"}),
    "detecttrack-uav-noar": ("ext", "DetectTrackUAVAdapter", {"aspect_ratio_thresh": None}),  # repo drops w/h>1.6 boxes
    "detecttrack-uav-cmc-noar": ("ext", "DetectTrackUAVAdapter", {"cmc_method": "sparseOptFlow", "aspect_ratio_thresh": None}),
    "smot-ocsort": ("ext", "SMOTOCSortAdapter", {}),
    "holdfast":    ("ext", "HoldfastAdapter", {}),
    "mosaic":      ("ext", "MosaicAdapter", {}),
    "csf-tst":     ("ext", "CSFTSTAdapter", {}),
    # with ReID (osnet_x0_25, CPU)
    "botsort-reid":    ("bm", "BotSort", {"use_embeddings": True, "use_cmc": True, "cmc_method": "ecc"}),
    "deepocsort-reid": ("bm", "DeepOcSort", {"use_embeddings": True}),
    "strongsort-reid": ("bm", "StrongSort", {}),
    "boosttrack-reid": ("bm", "BoostTrack", {"use_embeddings": True}),
}
REID_W = os.path.expanduser("~/hailo-test/mot/weights/osnet_x0_25_msmt17.pt")


def make(name):
    lib, cls, kw = TRACKERS[name]
    if lib == "ext":
        import ext_trackers
        return getattr(ext_trackers, cls)(**kw)
    if lib == "rf":
        import trackers
        return getattr(trackers, cls)(**kw)
    import boxmot
    kw = dict(kw)
    if name.endswith("-reid"):
        kw.update(reid_weights=REID_W, device="cpu", half=False)
    return getattr(boxmot, cls)(**kw)


def needs_frame(t):
    if type(t).__module__ == "ext_trackers":
        return True  # CMC / colour-histogram cues read the image
    r = getattr(t, "requirements", None)
    return bool(r and (r.frame or getattr(r, "embeddings", False)))


def step(t, dets):
    """dets: N x 6 [x1,y1,x2,y2,score,cls] -> M x 6 [x1,y1,x2,y2,id,score]"""
    if type(t).__module__.startswith("trackers"):
        import supervision as sv
        d = sv.Detections(xyxy=dets[:, :4].astype(np.float32), confidence=dets[:, 4].astype(np.float32),
                          class_id=dets[:, 5].astype(int))
        r = t.update(d)
        m = r.tracker_id >= 0
        return np.c_[r.xyxy[m], r.tracker_id[m], r.confidence[m]]
    r = t.update(dets.astype(np.float32), step.frame)
    return np.zeros((0, 6)) if len(r) == 0 else np.c_[r[:, :5], r[:, 5]]


class JointTracker:
    """사람·차를 트래커 하나로 함께 추적(이름 끝 '-joint'). 트래커는 클래스를 보지 않으므로,
    트랙마다 매칭된 검출의 클래스로 사람/차 다수결을 매겨 결과를 그룹별로 나눈다."""
    def __init__(self, base):
        self.t = make(base); self.vote = {}
    def update(self, d):
        """d: N x 6 [x1,y1,x2,y2,score,cls] (사람+차) -> {group: M x 6 [x1,y1,x2,y2,id,score]}"""
        r = step(self.t, d)
        out = {g: [] for g in GROUPS}
        if not len(r): return {g: np.zeros((0, 6)) for g in GROUPS}
        if len(d):
            a, b = r[:, :4], d[:, :4]
            x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
            x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
            inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
            M = inter / (((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None] - inter + 1e-9)
        for i, row in enumerate(r):
            tid = int(row[4]); v = self.vote.setdefault(tid, {g: 0 for g in GROUPS})
            if len(d) and M[i].max() > 0.3:
                c = d[M[i].argmax(), 5]
                for g, spec in GROUPS.items():
                    if c in spec["det"]: v[g] += 1
            out[max(v, key=v.get)].append(row)
        return {g: np.array(v_) if v_ else np.zeros((0, 6)) for g, v_ in out.items()}


def run(job):
    det_name, trk = job
    gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))
    times, nframes = 0.0, 0
    for seq in SEQS:
        D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy")
        joint = trk.endswith("-joint")
        trackers = {"all": JointTracker(trk[:-6])} if joint else {g: make(trk) for g in GROUPS}
        frame_needed = any(needs_frame(t.t if joint else t) for t in trackers.values())
        files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
        ign = {}
        for r in gt[seq]["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
        out = {g: [] for g in GROUPS}
        for fi in range(1, gt[seq]["n"] + 1):
            step.frame = cv2.imread(files[fi - 1]) if frame_needed else None
            if step.frame is not None:
                # the HF mirror has 2 odd-sized frames (117/0000349, 182/0000363); keep CMC inputs one size per sequence
                if fi == 1: seq_hw = step.frame.shape[:2]
                elif step.frame.shape[:2] != seq_hw: step.frame = cv2.resize(step.frame, seq_hw[::-1])
            fd = D[D[:, 0] == fi]
            if joint:
                allc = sum((GROUPS[g]["det"] for g in GROUPS), [])
                t0 = time.perf_counter()
                rr = trackers["all"].update(fd[np.isin(fd[:, 6], allc)][:, 1:7])
                times += time.perf_counter() - t0
                for g, r in rr.items():
                    if len(r):
                        r = r[~ignored_mask(r[:, :4], ign.get(fi, []))]
                        out[g] += [[fi, tid, x1, y1, x2, y2, sc] for x1, y1, x2, y2, tid, sc in r]
                nframes += 1
                continue
            for g, spec in GROUPS.items():
                d = fd[np.isin(fd[:, 6], spec["det"])][:, 1:7]
                t0 = time.perf_counter()
                r = step(trackers[g], d)
                times += time.perf_counter() - t0
                if len(r):
                    r = r[~ignored_mask(r[:, :4], ign.get(fi, []))]
                    out[g] += [[fi, tid, x1, y1, x2, y2, sc] for x1, y1, x2, y2, tid, sc in r]
            nframes += 1
        for g in GROUPS:
            write_mot(f"{ROOT}/data/trackers/VD{g}-val/{det_name}__{trk}/data/{seq}.txt", out[g])
    ms = times / nframes * 1000
    print(f"{det_name:8s} {trk:18s} track {ms:6.2f} ms/frame (both groups)", flush=True)
    return f"{det_name}__{trk}", ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", nargs="+", required=True)
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(d, t) for d in a.dets for t in a.trackers]
    with Pool(a.jobs) as p:
        res = p.map(run, jobs, chunksize=1)
    tf = f"{ROOT}/track_times.json"
    tt = json.load(open(tf)) if os.path.exists(tf) else {}
    tt.update(dict(res)); json.dump(tt, open(tf, "w"), indent=1)
