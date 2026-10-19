import sys, json, glob, os, cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# VisDrone txt -> COCO GT (drop 0=ignored region, 11=others)
root, pred = sys.argv[1], sys.argv[2]
imgs, anns, aid = [], [], 1
for f in sorted(glob.glob(f"{root}/annotations/*.txt")):
    iid = os.path.basename(f)[:-4]
    h, w = cv2.imread(f"{root}/images/{iid}.jpg").shape[:2]
    imgs.append({"id": iid, "width": w, "height": h})
    for line in open(f):
        v = line.strip().rstrip(",").split(",")
        if len(v) < 6: continue
        x, y, bw, bh, cat = int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[5])
        if cat in (0, 11) or bw <= 0 or bh <= 0: continue
        anns.append({"id": aid, "image_id": iid, "category_id": cat, "bbox": [x, y, bw, bh], "area": bw*bh, "iscrowd": 0}); aid += 1
gt = COCO(); gt.dataset = {"images": imgs, "annotations": anns, "categories": [{"id": i} for i in range(1, 11)]}; gt.createIndex()
E = COCOeval(gt, gt.loadRes(pred), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
