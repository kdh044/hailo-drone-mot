import sys, glob, pickle, cv2, numpy as np
from common import ROOT, SEQ_ROOT, GROUPS

# usage: python3 render_640_vs_4k.py <seq> <frame> <out.png> [dets_a=s_tile1x1] [dets_b=s_tile6x4]
# 같은 프레임, 같은 정답(흰색). 왼쪽 = 640으로 줄여서 검출, 오른쪽 = 4K 화질(조각)로 검출. 초록 = 사람, 주황 = 차 검출(점수 0.3 이상)
seq, f, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
A = sys.argv[4] if len(sys.argv) > 4 else "s_tile1x1"; B = sys.argv[5] if len(sys.argv) > 5 else "s_tile6x4"
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))[seq]
img = cv2.imread(sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))[f - 1]); H, W = img.shape[:2]
G = [r for r in gt["objs"] if r[0] == f and (r[6] in GROUPS["person"]["gt"] or r[6] in GROUPS["vehicle"]["gt"])]
P = [r for r in G if r[6] in GROUPS["person"]["gt"]]
# 사람이 모인 곳을 확대 영역으로
if P: cx, cy = np.median([(r[2] + r[4]) / 2 for r in P]), np.median([(r[3] + r[5]) / 2 for r in P])
else: cx, cy = W / 2, H / 2
cw, ch = W // 3, H // 3
x0, y0 = int(np.clip(cx - cw / 2, 0, W - cw)), int(np.clip(cy - ch / 2, 0, H - ch))
panels = []
for name in (A, B):
    im = img.copy(); D = np.load(f"{ROOT}/dets/{name}/{seq}.npy"); D = D[(D[:, 0] == f) & (D[:, 5] >= 0.3)]
    lw = max(2, W // 900)
    for r in G: cv2.rectangle(im, (int(r[2]), int(r[3])), (int(r[4]), int(r[5])), (255, 255, 255), lw)
    for d in D:
        col = (0, 230, 0) if d[6] in GROUPS["person"]["det"] else (0, 160, 255) if d[6] in GROUPS["vehicle"]["det"] else None
        if col: cv2.rectangle(im, (int(d[1]), int(d[2])), (int(d[3]), int(d[4])), col, lw + 1)
    full = cv2.resize(im, (960, int(960 * H / W)))
    s = 960 / W; cv2.rectangle(full, (int(x0 * s), int(y0 * s)), (int((x0 + cw) * s), int((y0 + ch) * s)), (0, 255, 255), 2)
    zoom = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (960, int(960 * ch / cw)), interpolation=cv2.INTER_NEAREST)
    panels.append(np.vstack([full, np.full((6, 960, 3), 60, np.uint8), zoom]))
cv2.imwrite(out, np.hstack([panels[0], np.full((panels[0].shape[0], 8, 3), 60, np.uint8), panels[1]]))
for name in (A, B):
    D = np.load(f"{ROOT}/dets/{name}/{seq}.npy"); D = D[(D[:, 0] == f) & (D[:, 5] >= 0.3)]
    print(name, "사람 검출", int(np.isin(D[:, 6], GROUPS["person"]["det"]).sum()), "차 검출", int(np.isin(D[:, 6], GROUPS["vehicle"]["det"]).sum()), "| 정답 사람", len(P), "차", len(G) - len(P))
