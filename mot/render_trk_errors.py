import sys, glob, pickle, random, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, SEQS, GROUPS, ignored_mask

# usage: python3 render_trk_errors.py <dets> <tracker> <group> <out.png>
# 추적 오류 예시: (1) 검출은 됐는데 트래커가 버림 (2) ID 바뀜(직전 프레임과 나란히) (3) 다시 나타났을 때 새 ID(끊김)
det_name, trk, g, out = sys.argv[1:5]; spec = GROUPS[g]; random.seed(1)
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))
CELL, CROP = 260, 120

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

dropped, switched, reborn = [], [], []
data = {}
for seq in SEQS:
    o = gt[seq]
    objs = np.array([r[:6] for r in o["objs"] if r[6] in spec["gt"]], float).reshape(-1, 6)
    if not len(objs): continue
    pth = f"{ROOT}/data/trackers/VD{g}-val/{det_name}__{trk}/data/{seq}.txt"
    T = np.loadtxt(pth, delimiter=",", ndmin=2) if __import__("os").path.getsize(pth) else np.zeros((0, 9))
    T[:, 4] += T[:, 2]; T[:, 5] += T[:, 3]
    D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy"); D = D[np.isin(D[:, 6], spec["det"])]
    data[seq] = (objs, T, D)
    ign = {}
    for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
    last, last_f = {}, {}
    for f in range(1, o["n"] + 1):
        G = objs[objs[:, 0] == f]
        if len(G): G = G[~ignored_mask(G[:, 2:6], ign.get(f, []))]
        if not len(G): continue
        P = T[T[:, 0] == f]; Dd = D[(D[:, 0] == f) & (D[:, 5] >= 0.05)]
        M = iou(G[:, 2:6], P[:, 2:6]); M[M < 0.5] = 0
        r, c = linear_sum_assignment(-M) if M.size else ([], [])
        m = {i: int(P[j, 1]) for i, j in zip(r, c) if M[i, j] >= 0.5}
        Md = iou(G[:, 2:6], Dd[:, 1:5])
        for i in range(len(G)):
            gid = int(G[i, 1])
            if i in m:
                if gid in last and last[gid] != m[i]:
                    (switched if last_f[gid] == f - 1 else reborn).append((seq, f, last_f[gid], gid))
                last[gid] = m[i]; last_f[gid] = f
            elif Md.shape[1] and Md[i].max() >= 0.5:
                dropped.append((seq, f, gid, float(Dd[Md[i].argmax(), 5])))

def cell(seq, f, gid, note):
    objs, T, D = data[seq]
    img = cv2.imread(sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))[f - 1]); H, W = img.shape[:2]
    gb = objs[(objs[:, 0] == f) & (objs[:, 1] == gid)]
    if not len(gb): return np.zeros((CELL, CELL, 3), np.uint8)
    x1, y1, x2, y2 = gb[0, 2:6]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; half = max(CROP / 2, x2 - x1, y2 - y1)
    a, b, c, d = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))
    s = CELL / max(c - a, d - b); crop = cv2.resize(img[b:d, a:c], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    cv = np.zeros((CELL, CELL, 3), np.uint8); cv[:crop.shape[0], :crop.shape[1]] = crop
    P = lambda v, o: int((v - o) * s)
    for dd in D[(D[:, 0] == f) & (D[:, 5] >= 0.05)]:
        cv2.rectangle(cv, (P(dd[1], a), P(dd[2], b)), (P(dd[3], a), P(dd[4], b)), (0, 220, 0), 1)
    for t in T[T[:, 0] == f]:
        col = tuple(int(v) for v in np.random.RandomState(int(t[1])).randint(80, 255, 3))
        cv2.rectangle(cv, (P(t[2], a), P(t[3], b)), (P(t[4], a), P(t[5], b)), col, 2)
        cv2.putText(cv, str(int(t[1])), (P(t[2], a), P(t[3], b) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)  # 트랙 ID 숫자는 남김
    cv2.rectangle(cv, (P(x1, a) - 4, P(y1, b) - 4), (P(x2, a) + 4, P(y2, b) + 4), (0, 0, 255), 2)
    return cv

def row(title, cells):
    sep = np.full((CELL, 3, 3), 60, np.uint8); r = cells[0]
    for c in cells[1:]: r = np.hstack([r, sep, c])
    bar = np.full((6, r.shape[1], 3), 60, np.uint8)
    return np.vstack([bar, r])

def pick(lst, n, key=lambda e: e[0]):
    random.shuffle(lst); out, seen = [], set()
    for e in lst:
        if key(e) not in seen: out.append(e); seen.add(key(e))
        if len(out) == n: break
    return out + [e for e in lst if e not in out][: n - len(out)]

rows = []
d = pick(dropped, 4)
rows.append(row("Detected but dropped by tracker (green = detection, no track on red GT)", [cell(s, f, gid, f"score {sc:.2f}") for s, f, gid, sc in d]))
sw = pick(switched, 2)
rows.append(row("", sum([[cell(s, pf, gid, "before"), cell(s, f, gid, "after")] for s, f, pf, gid in sw], [])))
rb_ = pick(reborn, 2)
rows.append(row("", sum([[cell(s, pf, gid, "last seen"), cell(s, f, gid, "new ID")] for s, f, pf, gid in rb_], [])))
cv2.imwrite(out, np.vstack(rows)[6:])
print(out, "버림", len(dropped), "ID 바뀜(연속)", len(switched), "끊겼다 새 ID", len(reborn))
