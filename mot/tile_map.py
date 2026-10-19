import sys, pickle, contextlib, io, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from common import ROOT, ignored_mask

# usage: ~/mot-venv/bin/python tile_map.py <seq> <dets name> [...]
# 한 영상(978프레임)의 MOT 정답으로 COCO mAP 계산. 채점 클래스 = 사람(pedestrian, people) + 차(car, van, truck, bus)
seq = sys.argv[1]; names = sys.argv[2:]
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))[seq]
LAB = {"pedestrian": 1, "people": 2, "car": 4, "van": 5, "truck": 6, "bus": 9}   # VisDrone DET 번호(1부터)
ign = {}
for r in gt["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
W, H = gt["size"]
imgs = [{"id": f, "width": W, "height": H} for f in range(1, gt["n"] + 1)]
anns = []
for k, r in enumerate(gt["objs"]):
    if r[6] not in LAB: continue
    x1, y1, x2, y2 = r[2:6]
    crowd = int(ignored_mask([[x1, y1, x2, y2]], ign.get(int(r[0]), []))[0])
    anns.append({"id": k + 1, "image_id": int(r[0]), "category_id": LAB[r[6]], "bbox": [x1, y1, x2 - x1, y2 - y1],
                 "area": (x2 - x1) * (y2 - y1), "iscrowd": crowd})
G = COCO(); G.dataset = {"images": imgs, "annotations": anns, "categories": [{"id": i} for i in sorted(set(LAB.values()))]}
with contextlib.redirect_stdout(io.StringIO()): G.createIndex()
print(f"{'방식':12s} {'mAP':>6s} {'AP50':>6s} {'AP75':>6s} {'AP작음':>6s} {'AP중간':>6s} | 사람 AP50  차 AP50")
for name in names:
    D = np.load(f"{ROOT}/dets/{name}/{seq}.npy")
    preds = [{"image_id": int(d[0]), "category_id": int(d[6]) + 1, "bbox": [float(d[1]), float(d[2]), float(d[3] - d[1]), float(d[4] - d[2])], "score": float(d[5])}
             for d in D if int(d[6]) + 1 in LAB.values()]
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(G, G.loadRes(preds), "bbox"); E.params.maxDets = [100, 300, 1000]; E.evaluate(); E.accumulate(); E.summarize()
        per = {}
        for grp, cats in (("사람", [1, 2]), ("차", [4, 5, 6, 9])):
            e = COCOeval(G, G.loadRes(preds), "bbox"); e.params.catIds = cats; e.params.maxDets = [100, 300, 1000]
            e.evaluate(); e.accumulate(); e.summarize(); per[grp] = e.stats[1]
    s = E.stats
    print(f"{name:12s} {s[0]*100:6.1f} {s[1]*100:6.1f} {s[2]*100:6.1f} {s[3]*100:6.1f} {s[4]*100:6.1f} | {per['사람']*100:7.1f}  {per['차']*100:7.1f}", flush=True)
