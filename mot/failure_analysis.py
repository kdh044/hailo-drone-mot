import os, sys, json, glob, pickle, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, SEQS, GROUPS, SAMPLES, ignored_mask

# usage: ~/mot-venv/bin/python failure_analysis.py <dets name> <tracker>
# 프레임마다 GT와 트래커 결과를 IoU>=0.5로 짝지어
#  - 놓친 객체(FN): 검출기가 아예 못 봄 / 검출기는 봤는데(점수>=0.05) 트래커가 버림
#  - ID 바뀜: 그 순간의 카메라 움직임 크기, 가림 정도, 박스 크기별로 센다
det_name, trk = sys.argv[1:3]
cv2.setNumThreads(1)
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))

# 가림 정보(occlusion: 0 없음, 1 일부, 2 심함)는 원본 samples.json에만 있다
occ = {}
for s in json.load(open(SAMPLES))["samples"]:
    for d in s["detections"] or []:
        if "index" in d: occ[(s["scene_id"], s["frame_number"], d["label"], d["index"])] = d.get("occlusion", 0)


def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


def camera_motion(seq):
    """프레임 간 전역 이동량(원본 픽셀/프레임). 1/8 크기 흑백에서 위상 상관으로 추정."""
    files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg")); prev = None; out = [0.0]
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_REDUCED_GRAYSCALE_8).astype(np.float32)
        if prev is not None:
            if g.shape != prev.shape: out.append(out[-1]); prev = g; continue
            (dx, dy), _ = cv2.phaseCorrelate(prev, g); out.append(float(np.hypot(dx, dy) * 8))
        prev = g
    return np.array(out[1:] if len(out) > len(files) else out)


def load_trk(path):
    T = np.loadtxt(path, delimiter=",", ndmin=2) if os.path.getsize(path) else np.zeros((0, 9))
    T[:, 4] += T[:, 2]; T[:, 5] += T[:, 3]   # x,y,w,h -> x1,y1,x2,y2
    return T

MOT_BINS = [0, 2, 5, 10, 20, 1e9]; MOT_LAB = ["<2px", "2-5px", "5-10px", "10-20px", ">=20px"]
SIZE_BINS = [0, 8, 16, 24, 1e9]; SIZE_LAB = ["<8px", "8-16px", "16-24px", ">=24px"]
report = {}
for g, spec in GROUPS.items():
    fn_det = fn_trk = tp = 0; fn_size = np.zeros((2, 4)); idsw_motion = np.zeros(5); frames_motion = np.zeros(5)
    idsw_occ = np.zeros(3); match_occ = np.zeros(3); idsw_size = np.zeros(4); match_size = np.zeros(4)
    per_seq = {}; worst = []
    for seq in SEQS:
        o = gt[seq]; w, h = o["size"]; scale = 640 / max(w, h)
        objs = np.array([r[:6] for r in o["objs"] if r[6] in spec["gt"]], float).reshape(-1, 6)
        labs = [r[6] for r in o["objs"] if r[6] in spec["gt"]]
        if not len(objs): continue
        T = load_trk(f"{ROOT}/data/trackers/VD{g}-val/{det_name}__{trk}/data/{seq}.txt")
        D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy"); D = D[np.isin(D[:, 6], spec["det"])]
        mot = camera_motion(seq)
        ign = {}
        for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
        last = {}; s_sw = s_fn = s_tp = 0; sw_frames = {}
        for f in range(1, o["n"] + 1):
            sel = objs[:, 0] == f
            G = objs[sel]; L = [labs[i] for i in np.nonzero(sel)[0]]
            keep = ~ignored_mask(G[:, 2:6], ign.get(f, [])) if len(G) else np.zeros(0, bool)
            G = G[keep]; L = [l for l, k in zip(L, keep) if k]
            if not len(G): continue
            P = T[T[:, 0] == f]; Dd = D[D[:, 0] == f]
            M = iou(G[:, 2:6], P[:, 2:6]); M[M < 0.5] = 0
            r, c = linear_sum_assignment(-M) if M.size else (np.zeros(0, int), np.zeros(0, int))
            ok = M[r, c] >= 0.5 if len(r) else np.zeros(0, bool)
            matched = {int(r[i]): int(P[c[i], 1]) for i in range(len(r)) if ok[i]}
            seen_by_det = (iou(G[:, 2:6], Dd[:, 1:5]).max(1) >= 0.5) if len(Dd) else np.zeros(len(G), bool)
            hpx = (G[:, 5] - G[:, 3]) * scale; sb = np.digitize(hpx, SIZE_BINS) - 1
            mb = int(np.digitize(mot[f - 1] if f - 1 < len(mot) else 0, MOT_BINS) - 1); frames_motion[mb] += 1
            for i in range(len(G)):
                gid = (L[i], int(G[i, 1])); oc = min(occ.get((seq, f, L[i], int(G[i, 1])), 0), 2)
                if i in matched:
                    tp += 1; s_tp += 1; match_occ[oc] += 1; match_size[sb[i]] += 1
                    if gid in last and last[gid] != matched[i]:
                        idsw_motion[mb] += 1; idsw_occ[oc] += 1; idsw_size[sb[i]] += 1; s_sw += 1
                        sw_frames[f] = sw_frames.get(f, 0) + 1
                    last[gid] = matched[i]
                else:
                    s_fn += 1
                    if seen_by_det[i]: fn_trk += 1; fn_size[1, sb[i]] += 1
                    else: fn_det += 1; fn_size[0, sb[i]] += 1
        per_seq[seq] = {"tp": s_tp, "fn": s_fn, "idsw": s_sw, "motion_mean": float(mot.mean()), "motion_p95": float(np.percentile(mot, 95))}
        worst += [(n, seq, f, float(mot[f - 1])) for f, n in sw_frames.items()]
    worst.sort(reverse=True)
    report[g] = {"fn_detector_missed": fn_det, "fn_tracker_dropped": fn_trk, "tp": tp,
                 "fn_by_size": {"detector_missed": dict(zip(SIZE_LAB, fn_size[0].tolist())), "tracker_dropped": dict(zip(SIZE_LAB, fn_size[1].tolist()))},
                 "idsw_per_1000_matches_by_motion": {l: (float(i / max(fr, 1)), int(i), int(fr)) for l, i, fr in zip(MOT_LAB, idsw_motion, frames_motion)},
                 "idsw_rate_by_occlusion": {k: float(i / max(m, 1) * 1000) for k, i, m in zip(["none", "partial", "heavy"], idsw_occ, match_occ)},
                 "idsw_rate_by_size": {k: float(i / max(m, 1) * 1000) for k, i, m in zip(SIZE_LAB, idsw_size, match_size)},
                 "per_seq": per_seq, "worst_frames": worst[:8]}
    tot_fn = fn_det + fn_trk
    print(f"\n[{g}] 놓친 객체 {tot_fn}개 중 검출기가 못 본 것 {fn_det / tot_fn * 100:.1f}% / 검출기는 봤는데 트래커가 버린 것 {fn_trk / tot_fn * 100:.1f}%")
    print("  크기별 놓친 객체 (못 봄 / 버림):", {l: (int(a), int(b)) for l, a, b in zip(SIZE_LAB, fn_size[0], fn_size[1])})
    print("  카메라 움직임별 ID 바뀜 (프레임당):", {l: f"{i / max(fr, 1):.2f} ({int(fr)}프레임)" for l, i, fr in zip(MOT_LAB, idsw_motion, frames_motion)})
    print("  가림 정도별 ID 바뀜 (짝지어진 1000개당):", {k: f"{v:.1f}" for k, v in report[g]["idsw_rate_by_occlusion"].items()})
    print("  크기별 ID 바뀜 (짝지어진 1000개당):", {k: f"{v:.1f}" for k, v in report[g]["idsw_rate_by_size"].items()})
    print("  영상별:", {s[:10]: (v["idsw"], round(v["motion_mean"], 1)) for s, v in per_seq.items()})
    print("  ID가 가장 많이 바뀐 프레임:", worst[:5])
json.dump(report, open(f"{ROOT}/failure_{det_name}__{trk}.json", "w"), indent=1)
