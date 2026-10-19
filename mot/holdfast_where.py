import os, json, glob, pickle, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, SEQS, GROUPS, ignored_mask

# S 검출 + Holdfast: 영상별로 어디서 깨지는지
#  - 영상별: HOTA·DetA·AssA, 놓친 정답(검출기 못 봄 / 트래커가 버림), ID 바뀜, 새 ID
#  - 영상마다 오류가 가장 많은 프레임 1장: 초록 = 맞게 추적된 정답, 빨강 = 검출기가 못 본 정답,
#    주황 = 검출은 됐는데 트래커가 버린 정답, 노란 원 = ID 바뀜, 색 박스+숫자 = Holdfast 트랙
DET, TRK = "s", "holdfast"
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb")); R = json.load(open(f"{ROOT}/results.json"))[f"{DET}__{TRK}"]

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

def load(path):
    if not os.path.getsize(path): return np.zeros((0, 9))
    T = np.loadtxt(path, delimiter=",", ndmin=2); T[:, 4] += T[:, 2]; T[:, 5] += T[:, 3]; return T

summary = {}
for seq in SEQS:
    o = gt[seq]; ign = {}
    for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
    per_frame = {}; stat = {}
    for g, spec in GROUPS.items():
        objs = np.array([r[:6] for r in o["objs"] if r[6] in spec["gt"]], float).reshape(-1, 6)
        T = load(f"{ROOT}/data/trackers/VD{g}-val/{DET}__{TRK}/data/{seq}.txt")
        D = np.load(f"{ROOT}/dets/{DET}/{seq}.npy"); D = D[np.isin(D[:, 6], spec["det"]) & (D[:, 5] >= 0.05)]
        last, lastf = {}, {}; s = dict(gt=0, tp=0, fn_det=0, fn_trk=0, sw=0, reborn=0)
        for f in range(1, o["n"] + 1):
            G = objs[objs[:, 0] == f]
            if len(G): G = G[~ignored_mask(G[:, 2:6], ign.get(f, []))]
            if not len(G): continue
            P = T[T[:, 0] == f]; Dd = D[D[:, 0] == f]
            M = iou(G[:, 2:6], P[:, 2:6]); M[M < 0.5] = 0
            r, c = linear_sum_assignment(-M) if M.size else ([], [])
            m = {i: int(P[j, 1]) for i, j in zip(r, c) if M[i, j] >= 0.5}
            seen = (iou(G[:, 2:6], Dd[:, 1:5]).max(1) >= 0.5) if len(Dd) else np.zeros(len(G), bool)
            pf = per_frame.setdefault(f, {"ok": [], "fn_det": [], "fn_trk": [], "sw": [], "score": 0})
            for i in range(len(G)):
                s["gt"] += 1; gid = int(G[i, 1]); box = G[i, 2:6].tolist()
                if i in m:
                    s["tp"] += 1; pf["ok"].append(box)
                    if gid in last and last[gid] != m[i]:
                        if lastf[gid] == f - 1: s["sw"] += 1
                        else: s["reborn"] += 1
                        pf["sw"].append(box); pf["score"] += 5
                    last[gid] = m[i]; lastf[gid] = f
                elif seen[i]: s["fn_trk"] += 1; pf["fn_trk"].append(box); pf["score"] += 1
                else: s["fn_det"] += 1; pf["fn_det"].append(box); pf["score"] += 1
        stat[g] = s
    summary[seq] = stat
    # 오류가 가장 많은 프레임
    f = max(per_frame, key=lambda k: per_frame[k]["score"]); pf = per_frame[f]
    img = cv2.imread(sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))[f - 1]); H, W = img.shape[:2]; k = 1344 / W
    img = cv2.resize(img, (1344, int(H * k)))
    for g in GROUPS:
        for t in load(f"{ROOT}/data/trackers/VD{g}-val/{DET}__{TRK}/data/{seq}.txt"):
            if t[0] != f: continue
            col = tuple(int(v) for v in np.random.RandomState(int(t[1]) + (0 if g == "person" else 7777)).randint(90, 255, 3))
            cv2.rectangle(img, (int(t[2] * k), int(t[3] * k)), (int(t[4] * k), int(t[5] * k)), col, 1)
            cv2.putText(img, str(int(t[1])), (int(t[2] * k), int(t[3] * k) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    for key, col in (("ok", (0, 200, 0)), ("fn_det", (0, 0, 255)), ("fn_trk", (0, 140, 255))):
        for b in pf[key]: cv2.rectangle(img, (int(b[0] * k) - 2, int(b[1] * k) - 2), (int(b[2] * k) + 2, int(b[3] * k) + 2), col, 2)
    for b in pf["sw"]: cv2.circle(img, (int((b[0] + b[2]) / 2 * k), int((b[1] + b[3]) / 2 * k)), 16, (0, 255, 255), 2)
    cv2.imwrite(f"{ROOT}/figs/where_{seq[:10]}.png", img)
    summary[seq]["frame"] = f; summary[seq]["frame_counts"] = {kk: len(v) for kk, v in pf.items() if kk != "score"}
json.dump(summary, open(f"{ROOT}/where_holdfast.json", "w"), indent=1)

print(f"{'영상':11s} {'해상도':>9s} | 사람 HOTA DetA AssA 정답 못봄% 버림% 바뀜 새ID | 차 HOTA 못봄% 버림% 바뀜 새ID | 대표프레임")
for seq in SEQS:
    w, h = gt[seq]["size"]; out = [f"{seq[:10]:11s} {w}x{h:>4d} |"]
    for g in GROUPS:
        r = R[g].get(seq, {}); s = summary[seq][g]; n = max(s["gt"], 1)
        if g == "person": out.append(f" {r.get('HOTA', 0):5.1f} {r.get('DetA', 0):5.1f} {r.get('AssA', 0):5.1f} {s['gt']:5d} {s['fn_det'] / n * 100:5.1f} {s['fn_trk'] / n * 100:5.1f} {s['sw']:4d} {s['reborn']:4d} |")
        else: out.append(f" {r.get('HOTA', 0):5.1f} {s['fn_det'] / n * 100:5.1f} {s['fn_trk'] / n * 100:5.1f} {s['sw']:4d} {s['reborn']:4d} |")
    out.append(f" f{summary[seq]['frame']} {summary[seq]['frame_counts']}")
    print("".join(out))
