import os, sys, glob, pickle, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, GROUPS

# usage: python3 capture_failures.py <dets> <tracker> <group> <seq> <frame> <out.png> [title]
# frame-1 과 frame 을 나란히 그린다. 흰 박스 = 정답, 색 박스 = 트래커(ID 표시), 빨간 원 = 이 프레임에서 ID가 바뀐 객체
det_name, trk, group, seq, f, out = sys.argv[1:7]; f = int(f)
title = sys.argv[7] if len(sys.argv) > 7 else ""
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))[seq]; spec = GROUPS[group]
objs = np.array([r[:6] for r in gt["objs"] if r[6] in spec["gt"]], float)
T = np.loadtxt(f"{ROOT}/data/trackers/VD{group}-val/{det_name}__{trk}/data/{seq}.txt", delimiter=",", ndmin=2)
T[:, 4] += T[:, 2]; T[:, 5] += T[:, 3]
files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    return inter / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None].__add__(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None]).__sub__(inter).clip(1e-9)

def match(fr):
    G = objs[objs[:, 0] == fr]; P = T[T[:, 0] == fr]
    M = iou(G[:, 2:6], P[:, 2:6]); M[M < 0.5] = 0
    r, c = linear_sum_assignment(-M) if M.size else ([], [])
    return G, P, {int(G[i, 1]): int(P[j, 1]) for i, j in zip(r, c) if M[i, j] >= 0.5}

Gp, Pp, mp = match(f - 1); Gc, Pc, mc = match(f)
switched = [g for g in mc if g in mp and mp[g] != mc[g]]
panels = []
for fr, G, P, m in ((f - 1, Gp, Pp, mp), (f, Gc, Pc, mc)):
    img = cv2.imread(files[fr - 1]); H, W = img.shape[:2]; s = 1344 / W; img = cv2.resize(img, (1344, int(H * s)))
    for g in G: cv2.rectangle(img, (int(g[2] * s), int(g[3] * s)), (int(g[4] * s), int(g[5] * s)), (255, 255, 255), 1)
    for p in P:
        c = tuple(int(v) for v in np.random.RandomState(int(p[1])).randint(60, 255, 3))
        cv2.rectangle(img, (int(p[2] * s), int(p[3] * s)), (int(p[4] * s), int(p[5] * s)), c, 2)
        cv2.putText(img, str(int(p[1])), (int(p[2] * s), int(p[3] * s) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    for g in G:
        if int(g[1]) in switched:
            cx, cy = int((g[2] + g[4]) / 2 * s), int((g[3] + g[5]) / 2 * s)
            cv2.circle(img, (cx, cy), 28, (0, 0, 255), 2)
    panels.append(img)
canvas = np.vstack([panels[0], np.full((6, panels[0].shape[1], 3), 60, np.uint8), panels[1]])
if False:
    bar = np.zeros((40, canvas.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    canvas = np.vstack([bar, canvas])
cv2.imwrite(out, canvas); print(out, "ID 바뀜", len(switched), "개")
