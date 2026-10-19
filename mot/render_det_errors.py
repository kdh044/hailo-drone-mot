import sys, json, glob, pickle, cv2, numpy as np
from common import ROOT, SEQ_ROOT, GROUPS

# usage: python3 render_det_errors.py <dets> <group> <out.png>
# 검출 오류 종류별 예시 4개씩 확대 크롭. 흰색 = 정답, 초록 = 검출(점수 0.3 이상), 빨강 = 이 오류의 대상
det_name, g, out = sys.argv[1:4]
EX = json.load(open(f"{ROOT}/det_error_examples_{det_name}.json"))
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb")); spec = GROUPS[g]
ROWS = [("fp_background", "False positive: background (no object)"),
        ("fp_localization", "False positive: box misplaced (IoU 0.1-0.5)"),
        ("fp_duplicate", "False positive: duplicate box"),
        ("fp_unlabeled_class", "False positive: unscored class (bike/motor rider)"),
        ("fn", "Missed object (red = GT not found)")]
CROP, CELL, N = 120, 260, 4
files = {}
rows = []
for key, title in ROWS:
    ex = [e for e in EX.get(f"{g}/{key}", [])]
    # 고르게 보이도록 서로 다른 영상에서 먼저 고른다
    picked, seen = [], set()
    for e in ex:
        if e[0] not in seen: picked.append(e); seen.add(e[0])
        if len(picked) == N: break
    for e in ex:
        if len(picked) == N: break
        if e not in picked: picked.append(e)
    cells = []
    for seq, f, x1, y1, x2, y2 in picked:
        if seq not in files: files[seq] = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
        img = cv2.imread(files[seq][int(f) - 1]); H, W = img.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; half = max(CROP / 2, (x2 - x1), (y2 - y1))
        a, b = int(max(0, cx - half)), int(max(0, cy - half)); c, d = int(min(W, cx + half)), int(min(H, cy + half))
        s = CELL / max(c - a, d - b)
        crop = cv2.resize(img[b:d, a:c], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((CELL, CELL, 3), np.uint8); canvas[:crop.shape[0], :crop.shape[1]] = crop
        P = lambda v, o: int((v - o) * s)
        for r in gt[seq]["objs"]:
            if r[0] == f and r[6] in spec["gt"]: cv2.rectangle(canvas, (P(r[2], a), P(r[3], b)), (P(r[4], a), P(r[5], b)), (255, 255, 255), 1)
        D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy"); D = D[(D[:, 0] == f) & (D[:, 5] >= 0.3) & np.isin(D[:, 6], spec["det"])]
        for dd in D: cv2.rectangle(canvas, (P(dd[1], a), P(dd[2], b)), (P(dd[3], a), P(dd[4], b)), (0, 220, 0), 1)
        cv2.rectangle(canvas, (P(x1, a) - 3, P(y1, b) - 3), (P(x2, a) + 3, P(y2, b) + 3), (0, 0, 255), 2)
        hpx = (y2 - y1) * 640 / max(W, H)
        cells.append(canvas)
    while len(cells) < N: cells.append(np.zeros((CELL, CELL, 3), np.uint8))
    bar = np.full((6, CELL * N + 3 * (N - 1), 3), 60, np.uint8)  # 줄 사이 구분선만
    sep = np.full((CELL, 3, 3), 60, np.uint8)
    row = cells[0]
    for cimg in cells[1:]: row = np.hstack([row, sep, cimg])
    rows.append(np.vstack([bar, row]))
cv2.imwrite(out, np.vstack(rows)[6:]); print(out)
