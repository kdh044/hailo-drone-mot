import os, sys, glob, pickle, shutil, contextlib, io, numpy as np, cv2
import trackeval
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, GROUPS, ignored_mask, write_mot
from run_trackers import make, step, needs_frame

# usage: ~/mot-venv/bin/python tile_eval.py <seq> <dets name> [<dets name> ...]
# 한 영상에서 검출 방식(조각 수)별로: 찾은 비율·정밀도(점수 0.3, IoU 0.5), 크기별 찾은 비율, ByteTrack/Holdfast HOTA
seq = sys.argv[1]; names = sys.argv[2:]
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))[seq]; W, H = gt["size"]
ign = {}
for r in gt["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

# 이 영상만 담은 작은 TrackEval 벤치마크
for g in GROUPS:
    d = f"{ROOT}/data/gt/VD{g}1-val/{seq}"; os.makedirs(d, exist_ok=True)
    for fn in ("seqinfo.ini",): shutil.copy(f"{ROOT}/data/gt/VD{g}-val/{seq}/{fn}", d)
    os.makedirs(f"{d}/gt", exist_ok=True); shutil.copy(f"{ROOT}/data/gt/VD{g}-val/{seq}/gt/gt.txt", f"{d}/gt/gt.txt")
    open(f"{ROOT}/data/seqmaps/VD{g}1-val.txt", "w").write(f"name\n{seq}\n")

def hota(g, tname):
    ecfg = trackeval.Evaluator.get_default_eval_config(); ecfg.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False)
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=f"{ROOT}/data/gt", TRACKERS_FOLDER=f"{ROOT}/data/trackers", SEQMAP_FOLDER=f"{ROOT}/data/seqmaps", BENCHMARK=f"VD{g}1", SPLIT_TO_EVAL="val", TRACKERS_TO_EVAL=[tname], DO_PREPROC=False, PRINT_CONFIG=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r, _ = trackeval.Evaluator(ecfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)], [trackeval.metrics.HOTA(), trackeval.metrics.Identity({"PRINT_CONFIG": False})])
    p = r["MotChallenge2DBox"][tname][seq]["pedestrian"]
    return p["HOTA"]["HOTA"].mean() * 100, p["Identity"]["IDF1"] * 100

files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
BINS = [0, 8, 16, 24, 1e9]; LAB = ["<8px", "8-16px", "16-24px", ">=24px"]
for name in names:
    D = np.load(f"{ROOT}/dets/{name}/{seq}.npy")
    line = [f"{name:12s}"]
    for g, spec in GROUPS.items():
        objs = np.array([r[:6] for r in gt["objs"] if r[6] in spec["gt"]], float).reshape(-1, 6)
        Dg = D[np.isin(D[:, 6], spec["det"]) & (D[:, 5] >= 0.3)]
        tp = fp = tot = 0; hit = np.zeros(4); cnt = np.zeros(4)
        for f in range(1, gt["n"] + 1):
            G = objs[objs[:, 0] == f][:, 2:6]
            if len(G): G = G[~ignored_mask(G, ign.get(f, []))]
            P = Dg[Dg[:, 0] == f][:, 1:5]
            if len(P): P = P[~ignored_mask(P, ign.get(f, []))]
            M = iou(G, P); M[M < 0.5] = 0; m = set()
            if M.size:
                r, c = linear_sum_assignment(-M); m = {i for i, j in zip(r, c) if M[i, j] >= 0.5}
            tp += len(m); fp += len(P) - len(m); tot += len(G)
            if len(G):
                b = np.digitize((G[:, 3] - G[:, 1]) * 640 / max(W, H), BINS) - 1
                np.add.at(cnt, b, 1); np.add.at(hit, b, [i in m for i in range(len(G))])
        line.append(f"{g}: 찾음 {tp / max(tot, 1) * 100:5.1f}% 정밀도 {tp / max(tp + fp, 1) * 100:5.1f}% (크기별 " + " ".join(f"{l} {h / c * 100:4.0f}%" if c else f"{l} -" for l, h, c in zip(LAB, hit, cnt)) + ")")
    # 추적: ByteTrack(roboflow)와 Holdfast
    for trk in ("rf-bytetrack", "holdfast"):
        res = []
        for g, spec in GROUPS.items():
            t = make(trk); out = []; nf = needs_frame(t)
            for fi in range(1, gt["n"] + 1):
                step.frame = cv2.imread(files[fi - 1], cv2.IMREAD_REDUCED_COLOR_4) if nf else None
                d = D[(D[:, 0] == fi) & np.isin(D[:, 6], spec["det"])][:, 1:7].copy()
                if nf: d[:, :4] /= 4   # 프레임을 1/4로 열었으니 좌표도 맞춘다
                r = step(t, d)
                if len(r):
                    if nf: r[:, :4] *= 4
                    r = r[~ignored_mask(r[:, :4], ign.get(fi, []))]
                    out += [[fi, tid, x1, y1, x2, y2, sc] for x1, y1, x2, y2, tid, sc in r]
            tn = f"{name}__{trk}"
            write_mot(f"{ROOT}/data/trackers/VD{g}1-val/{tn}/data/{seq}.txt", out)
            res.append((g, *hota(g, tn)))
        line.append(f"{trk}: " + ", ".join(f"{g} HOTA {h:.1f} IDF1 {i:.1f}" for g, h, i in res))
    print("\n  ".join(line), flush=True)
