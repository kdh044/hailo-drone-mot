import json, glob, cv2, numpy as np
from common import ROOT, SEQ_ROOT

# 놓친 사람(검출기가 못 본 것) 예시를 원본 화질과 검출기가 실제로 보는 640 화질로 나란히 보여준다.
# 왼쪽 = 원본 픽셀, 오른쪽 = 640으로 줄였다가 같은 크기로 다시 키운 것(검출기 입력과 같은 정보량)
EX = json.load(open(f"{ROOT}/det_error_examples_s.json"))["person/fn"]
BIG = ("uav0000268", "uav0000117", "uav0000137", "uav0000305", "uav0000339")
ex = [e for e in EX if e[0].startswith(BIG)]
seen, picked = set(), []
for e in ex:
    if e[0] not in seen: picked.append(e); seen.add(e[0])
for e in ex:
    if len(picked) >= 6: break
    if e not in picked: picked.append(e)
CELL = 240; rows = []
for seq, f, x1, y1, x2, y2 in picked[:6]:
    img = cv2.imread(sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))[int(f) - 1]); H, W = img.shape[:2]
    k = 640 / max(W, H)
    small = cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)   # 검출기 입력과 같은 축소
    seen640 = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; half = max(60, 2.5 * (y2 - y1))
    a, b, c, d = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))
    pair = []
    for src in (img, seen640):
        crop = cv2.resize(src[b:d, a:c], (CELL, CELL), interpolation=cv2.INTER_NEAREST)
        s = CELL / (c - a)
        cv2.rectangle(crop, (int((x1 - a) * s) - 3, int((y1 - b) * s) - 3), (int((x2 - a) * s) + 3, int((y2 - b) * s) + 3), (0, 0, 255), 2)
        pair.append(crop)
    rows.append(np.hstack([pair[0], np.full((CELL, 3, 3), 60, np.uint8), pair[1]]))
grid = [np.hstack([rows[i], np.full((CELL, 12, 3), 30, np.uint8), rows[i + 1]]) for i in range(0, len(rows) - 1, 2)]
out = np.vstack([np.vstack([g, np.full((6, g.shape[1], 3), 60, np.uint8)]) for g in grid])[:-6]
cv2.imwrite(f"{ROOT}/figs/compare_orig_vs_640.png", out)
print([(e[0][:10], int(e[1]), round((e[5] - e[3]), 1)) for e in picked[:6]])
