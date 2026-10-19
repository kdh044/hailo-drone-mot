import os, json, glob, pickle, random, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQ_ROOT, SEQS, GROUPS, SAMPLES, ignored_mask

# S 검출 + Holdfast, 상황별로 정리:
#   움직일 때(카메라 이동 / 객체 이동), 가려질 때(가림 라벨 / 다른 정답과 겹침), 작을 때, 점수 낮을 때
# 수치(비율)와 상황별 예시 사진을 함께 만든다. 사람·차를 합쳐서 센다.
DET, TRK = "s", "holdfast"; random.seed(3)
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))
occ = {}
for s in json.load(open(SAMPLES))["samples"]:
    for d in s["detections"] or []:
        if "index" in d: occ[(s["scene_id"], s["frame_number"], d["label"], d["index"])] = min(d.get("occlusion", 0), 1)

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

def cam_motion(seq):
    files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg")); prev = None; out = []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_REDUCED_GRAYSCALE_8).astype(np.float32)
        if prev is None: out.append(0.0)
        elif g.shape != prev.shape: out.append(out[-1])
        else: (dx, dy), _ = cv2.phaseCorrelate(prev, g); out.append(float(np.hypot(dx, dy) * 8))
        prev = g
    return np.array(out)

def load(path):
    if not os.path.getsize(path): return np.zeros((0, 9))
    T = np.loadtxt(path, delimiter=",", ndmin=2); T[:, 4] += T[:, 2]; T[:, 5] += T[:, 3]; return T

SIT = {"카메라 이동": [("2px 미만", lambda a: a["cam"] < 2), ("2~10px", lambda a: 2 <= a["cam"] < 10), ("10px 이상", lambda a: a["cam"] >= 10)],
       "객체 이동": [("1px 미만", lambda a: a["mv"] is not None and a["mv"] < 1), ("1~4px", lambda a: a["mv"] is not None and 1 <= a["mv"] < 4), ("4px 이상", lambda a: a["mv"] is not None and a["mv"] >= 4)],
       "다른 정답과 겹침": [("안 겹침", lambda a: a["ov"] < 0.05), ("조금 겹침 (IoU 0.05~0.3)", lambda a: 0.05 <= a["ov"] < 0.3), ("많이 겹침 (IoU 0.3 이상)", lambda a: a["ov"] >= 0.3)],
       "가림 라벨": [("없음", lambda a: a["occ"] == 0), ("일부", lambda a: a["occ"] == 1)],
       "크기 (640 기준 높이)": [("16px 미만", lambda a: a["h"] < 16), ("16~32px", lambda a: 16 <= a["h"] < 32), ("32px 이상", lambda a: a["h"] >= 32)]}
stat = {k: {l: dict(n=0, fn_det=0, fn_trk=0, tp=0, sw=0, reborn=0) for l, _ in v} for k, v in SIT.items()}
EX = {"cam": [], "fast": [], "overlap": [], "small": [], "low": []}
for seq in SEQS:
    o = gt[seq]; W, H = o["size"]; k640 = 640 / max(W, H); cm = cam_motion(seq)
    ign = {}
    for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
    for g, spec in GROUPS.items():
        rows = [r for r in o["objs"] if r[6] in spec["gt"]]
        if not rows: continue
        objs = np.array([r[:6] for r in rows], float); labs = [r[6] for r in rows]
        center = {(int(r[0]), r[6], int(r[1])): ((r[2] + r[4]) / 2, (r[3] + r[5]) / 2) for r in rows}
        T = load(f"{ROOT}/data/trackers/VD{g}-val/{DET}__{TRK}/data/{seq}.txt")
        D = np.load(f"{ROOT}/dets/{DET}/{seq}.npy"); D = D[np.isin(D[:, 6], spec["det"]) & (D[:, 5] >= 0.05)]
        last, lastf = {}, {}
        for f in range(1, o["n"] + 1):
            sel = np.nonzero(objs[:, 0] == f)[0]
            if not len(sel): continue
            G = objs[sel]; L = [labs[i] for i in sel]
            keep = ~ignored_mask(G[:, 2:6], ign.get(f, [])); G = G[keep]; L = [l for l, kk in zip(L, keep) if kk]
            if not len(G): continue
            GG = iou(G[:, 2:6], G[:, 2:6]); np.fill_diagonal(GG, 0); ov = GG.max(1) if len(G) > 1 else np.zeros(len(G))
            P = T[T[:, 0] == f]; Dd = D[D[:, 0] == f]
            M = iou(G[:, 2:6], P[:, 2:6]); M[M < 0.5] = 0
            r, c = linear_sum_assignment(-M) if M.size else ([], [])
            m = {i: int(P[j, 1]) for i, j in zip(r, c) if M[i, j] >= 0.5}
            Md = iou(G[:, 2:6], Dd[:, 1:5]); seen = (Md.max(1) >= 0.5) if Md.shape[1] else np.zeros(len(G), bool)
            cam = cm[f - 1]
            for i in range(len(G)):
                gid = int(G[i, 1]); key = (L[i], gid); prev = center.get((f - 1, L[i], gid)); cur = center[(f, L[i], gid)]
                a = dict(cam=cam, mv=(abs(np.hypot(cur[0] - prev[0], cur[1] - prev[1]) - cam) * k640) if prev else None,
                         ov=float(ov[i]), occ=occ.get((seq, f, L[i], gid), 0), h=(G[i, 5] - G[i, 3]) * k640)
                ev = "tp" if i in m else ("fn_trk" if seen[i] else "fn_det"); extra = None
                if i in m:
                    if key in last and last[key] != m[i]: extra = "sw" if lastf[key] == f - 1 else "reborn"
                    if extra: info = (seq, g, f, lastf[key], gid, L[i])
                    last[key] = m[i]; lastf[key] = f
                for k, bins in SIT.items():
                    for lab, fn in bins:
                        if fn(a):
                            cell = stat[k][lab]; cell["n"] += 1; cell[ev] += 1
                            if extra: cell[extra] += 1
                if extra:
                    if cam >= 10: EX["cam"].append(info)
                    if a["mv"] is not None and a["mv"] >= 4: EX["fast"].append(info)
                    if a["ov"] >= 0.3: EX["overlap"].append(info)
                elif ev == "fn_det" and a["h"] < 16: EX["small"].append((seq, g, f, f, gid, L[i]))
                elif ev == "fn_trk" and a["h"] >= 16: EX["low"].append((seq, g, f, f, gid, L[i], float(Dd[Md[i].argmax(), 5])))
json.dump(stat, open(f"{ROOT}/holdfast_situations.json", "w"), indent=1, ensure_ascii=False)
for k, v in stat.items():
    print(f"[{k}]")
    for lab, c in v.items():
        if c["n"]: print(f"  {lab:24s} 정답 {c['n']:6d}  못봄 {c['fn_det'] / c['n'] * 100:5.1f}%  버림 {c['fn_trk'] / c['n'] * 100:5.1f}%  ID바뀜 {c['sw'] / max(c['tp'], 1) * 1000:5.1f}  새ID {c['reborn'] / max(c['tp'], 1) * 1000:5.1f}")
print({k: len(v) for k, v in EX.items()})

# ---- 상황별 예시 사진 (글씨 없음). 빨강 = 해당 정답, 초록 = 검출, 색 박스+숫자 = Holdfast 트랙
CELL, CROP = 300, 110
cache = {}
def tracks(seq, g):
    if (seq, g) not in cache: cache[(seq, g)] = load(f"{ROOT}/data/trackers/VD{g}-val/{DET}__{TRK}/data/{seq}.txt")
    return cache[(seq, g)]
def cell(seq, g, f, gid, lab):
    img = cv2.imread(sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))[f - 1]); Hh, Ww = img.shape[:2]
    gb = [r for r in gt[seq]["objs"] if r[0] == f and r[1] == gid and r[6] == lab]
    if not gb: return np.zeros((CELL, CELL, 3), np.uint8)
    x1, y1, x2, y2 = gb[0][2:6]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; half = max(CROP / 2, 1.6 * (y2 - y1), 1.6 * (x2 - x1))
    a, b, c, d = int(max(0, cx - half)), int(max(0, cy - half)), int(min(Ww, cx + half)), int(min(Hh, cy + half))
    s = CELL / max(c - a, d - b); cr = cv2.resize(img[b:d, a:c], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    cv = np.zeros((CELL, CELL, 3), np.uint8); cv[:cr.shape[0], :cr.shape[1]] = cr
    Pz = lambda v, o: int((v - o) * s)
    D = np.load(f"{ROOT}/dets/{DET}/{seq}.npy"); D = D[(D[:, 0] == f) & (D[:, 5] >= 0.05) & np.isin(D[:, 6], GROUPS[g]["det"])]
    for dd in D: cv2.rectangle(cv, (Pz(dd[1], a), Pz(dd[2], b)), (Pz(dd[3], a), Pz(dd[4], b)), (0, 220, 0), 1)
    for t in tracks(seq, g)[tracks(seq, g)[:, 0] == f]:
        col = tuple(int(v) for v in np.random.RandomState(int(t[1])).randint(80, 255, 3))
        cv2.rectangle(cv, (Pz(t[2], a), Pz(t[3], b)), (Pz(t[4], a), Pz(t[5], b)), col, 2)
        cv2.putText(cv, str(int(t[1])), (Pz(t[2], a), Pz(t[3], b) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.rectangle(cv, (Pz(x1, a) - 4, Pz(y1, b) - 4), (Pz(x2, a) + 4, Pz(y2, b) + 4), (0, 0, 255), 2)
    return cv
def pick(lst, n):
    random.shuffle(lst); out, seen = [], set()
    for e in lst:
        if (e[0], e[1]) not in seen: out.append(e); seen.add((e[0], e[1]))
        if len(out) == n: break
    return out + [e for e in lst if e not in out][: n - len(out)]
def save(cells, name):
    sep = np.full((CELL, 4, 3), 60, np.uint8); row = cells[0]
    for c_ in cells[1:]: row = np.hstack([row, sep, c_])
    cv2.imwrite(f"{ROOT}/figs/{name}.png", row)
for key in ("cam", "fast", "overlap"):   # 앞뒤 프레임 쌍 2개 = 4칸
    cells = []
    for seq, g, f, pf, gid, lab in pick(EX[key], 2): cells += [cell(seq, g, pf, gid, lab), cell(seq, g, f, gid, lab)]
    save(cells, f"sit_{key}")
for key in ("small", "low"):
    save([cell(e[0], e[1], e[2], e[4], e[5]) for e in pick(EX[key], 4)], f"sit_{key}")
print("사진 완료")
