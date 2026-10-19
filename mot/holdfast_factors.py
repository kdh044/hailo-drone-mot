import os, json, glob, pickle, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, SEQS, GROUPS, SAMPLES, ignored_mask

# S 검출 + Holdfast: 가림 / 카메라 움직임 / 객체 움직임 / 크기 별로
#   놓침(검출기 못 봄, 트래커 버림), ID 바뀜(바로 다음 프레임), 끊겼다 새 ID 를 비율로 센다.
DET, TRK = "s", "holdfast"
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))
occ = {}
for s in json.load(open(SAMPLES))["samples"]:
    for d in s["detections"] or []:
        if "index" in d: occ[(s["scene_id"], s["frame_number"], d["label"], d["index"])] = min(d.get("occlusion", 0), 2)

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

def cam_motion(seq):
    files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg")); prev = None; out = [0.0]
    for f in files[1:] if False else files:
        g = cv2.imread(f, cv2.IMREAD_REDUCED_GRAYSCALE_8).astype(np.float32)
        if prev is not None and g.shape == prev.shape:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g); out.append(float(np.hypot(dx, dy) * 8))
        elif prev is not None: out.append(out[-1])
        prev = g
    return np.array(out[: len(files)])

def load(path):
    if not os.path.getsize(path): return np.zeros((0, 9))
    T = np.loadtxt(path, delimiter=",", ndmin=2); T[:, 4] += T[:, 2]; T[:, 5] += T[:, 3]; return T

FACTORS = {"가림": (["없음", "일부"], None),
           "카메라 이동(px/프레임, 원본)": (["<2", "2-5", "5-10", "10-20", "≥20"], [0, 2, 5, 10, 20, 1e9]),
           "객체 이동(px/프레임, 640 기준, 카메라 이동 뺌)": (["<0.5", "0.5-1", "1-2", "2-4", "≥4"], [0, 0.5, 1, 2, 4, 1e9]),
           "크기(높이 px, 640 기준)": (["<8", "8-16", "16-24", "24-32", "≥32"], [0, 8, 16, 24, 32, 1e9]),
           "가림 × 크기": (["없음·16px 미만", "일부·16px 미만", "없음·16~32px", "일부·16~32px", "없음·32px 이상", "일부·32px 이상"], None)}
res = {g: {fac: {l: dict(n=0, fn_det=0, fn_trk=0, tp=0, sw=0, reborn=0) for l in labs} for fac, (labs, _) in FACTORS.items()} for g in GROUPS}
for seq in SEQS:
    o = gt[seq]; W, H = o["size"]; k640 = 640 / max(W, H); cm = cam_motion(seq)
    ign = {}
    for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
    for g, spec in GROUPS.items():
        rows = [r for r in o["objs"] if r[6] in spec["gt"]]
        if not rows: continue
        objs = np.array([r[:6] for r in rows], float); labs_ = [r[6] for r in rows]
        center = {(int(r[0]), r[6], int(r[1])): ((r[2] + r[4]) / 2, (r[3] + r[5]) / 2) for r in rows}
        T = load(f"{ROOT}/data/trackers/VD{g}-val/{DET}__{TRK}/data/{seq}.txt")
        D = np.load(f"{ROOT}/dets/{DET}/{seq}.npy"); D = D[np.isin(D[:, 6], spec["det"]) & (D[:, 5] >= 0.05)]
        last, lastf = {}, {}
        for f in range(1, o["n"] + 1):
            sel = np.nonzero(objs[:, 0] == f)[0]
            if not len(sel): continue
            G = objs[sel]; L = [labs_[i] for i in sel]
            keep = ~ignored_mask(G[:, 2:6], ign.get(f, [])); G = G[keep]; L = [l for l, kk in zip(L, keep) if kk]
            if not len(G): continue
            P = T[T[:, 0] == f]; Dd = D[D[:, 0] == f]
            M = iou(G[:, 2:6], P[:, 2:6]); M[M < 0.5] = 0
            r, c = linear_sum_assignment(-M) if M.size else ([], [])
            m = {i: int(P[j, 1]) for i, j in zip(r, c) if M[i, j] >= 0.5}
            seen = (iou(G[:, 2:6], Dd[:, 1:5]).max(1) >= 0.5) if len(Dd) else np.zeros(len(G), bool)
            cam = cm[f - 1] if f - 1 < len(cm) else 0.0
            for i in range(len(G)):
                gid = int(G[i, 1]); key = (L[i], gid)
                prev = center.get((f - 1, L[i], gid)); cur = center[(f, L[i], gid)]
                mv = (abs(np.hypot(cur[0] - prev[0], cur[1] - prev[1]) - cam) * k640) if prev else None
                bins = {"가림": ["없음", "일부"][min(occ.get((seq, f, L[i], gid), 0), 1)],
                        "카메라 이동(px/프레임, 원본)": FACTORS["카메라 이동(px/프레임, 원본)"][0][np.digitize(cam, FACTORS["카메라 이동(px/프레임, 원본)"][1]) - 1],
                        "크기(높이 px, 640 기준)": FACTORS["크기(높이 px, 640 기준)"][0][np.digitize((G[i, 5] - G[i, 3]) * k640, FACTORS["크기(높이 px, 640 기준)"][1]) - 1]}
                hh = (G[i, 5] - G[i, 3]) * k640; sz = "16px 미만" if hh < 16 else ("16~32px" if hh < 32 else "32px 이상")
                bins["가림 × 크기"] = f"{bins['가림']}·{sz}"
                if mv is not None:
                    fl = FACTORS["객체 이동(px/프레임, 640 기준, 카메라 이동 뺌)"]; bins[list(FACTORS)[2]] = fl[0][np.digitize(mv, fl[1]) - 1]
                ev = "tp" if i in m else ("fn_trk" if seen[i] else "fn_det")
                extra = None
                if i in m:
                    if key in last and last[key] != m[i]: extra = "sw" if lastf[key] == f - 1 else "reborn"
                    last[key] = m[i]; lastf[key] = f
                for fac, lab in bins.items():
                    cell = res[g][fac][lab]; cell["n"] += 1; cell[ev] += 1
                    if extra: cell[extra] += 1
json.dump(res, open(f"{ROOT}/holdfast_factors.json", "w"), indent=1, ensure_ascii=False)
for g in GROUPS:
    print(f"\n===== {g} =====")
    for fac, (labs, _) in FACTORS.items():
        print(f"[{fac}]")
        for l in labs:
            c = res[g][fac][l]
            if not c["n"]: continue
            print(f"  {l:>6s}  정답 {c['n']:6d}  못봄 {c['fn_det'] / c['n'] * 100:5.1f}%  버림 {c['fn_trk'] / c['n'] * 100:5.1f}%  "
                  f"ID바뀜 {c['sw'] / max(c['tp'], 1) * 1000:5.1f}/1000  새ID {c['reborn'] / max(c['tp'], 1) * 1000:5.1f}/1000")
