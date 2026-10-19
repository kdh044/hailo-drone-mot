import sys, json, pickle, numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT, SEQS, GROUPS, SAMPLES, ignored_mask

# usage: ~/mot-venv/bin/python det_error_analysis.py <dets name> [score_thr=0.3]
# 검출 오류 분류 (그룹별, IoU 0.5 기준)
#  헛검출(FP): 다른 그룹과 혼동 / 채점 안 하는 클래스(자전거·오토바이 등) / 중복 / 위치 부정확(IoU 0.1~0.5) / 배경
#  놓침(FN): 가림 정도별, 크기별
det_name = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))
occ = {}
for s in json.load(open(SAMPLES))["samples"]:
    for d in s["detections"] or []:
        if "index" in d: occ[(s["scene_id"], s["frame_number"], d["label"], d["index"])] = min(d.get("occlusion", 0), 2)
OTHER = {"bicycle", "motor", "tricycle", "awning-tricycle"}

def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

import random
random.seed(0)
EX = {}
out = {}
for g, spec in GROUPS.items():
    other_g = [x for x in GROUPS if x != g][0]
    fp = dict(confused=0, unlabeled_class=0, duplicate=0, localization=0, background=0); tp = 0
    fn_occ = np.zeros(3); gt_occ = np.zeros(3)
    for seq in SEQS:
        o = gt[seq]; D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy"); D = D[D[:, 5] >= THR]
        ign = {}
        for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
        byf = {}
        for r in o["objs"]: byf.setdefault(int(r[0]), []).append(r)
        for f in range(1, o["n"] + 1):
            rows = byf.get(f, [])
            G = np.array([r[2:6] for r in rows if r[6] in spec["gt"]], float).reshape(-1, 4)
            Gid = [(r[6], int(r[1])) for r in rows if r[6] in spec["gt"]]
            Go = np.array([r[2:6] for r in rows if r[6] in GROUPS[other_g]["gt"]], float).reshape(-1, 4)
            Gu = np.array([r[2:6] for r in rows if r[6] in OTHER], float).reshape(-1, 4)
            keep = ~ignored_mask(G, ign.get(f, [])) if len(G) else np.zeros(0, bool)
            G = G[keep]; Gid = [x for x, k in zip(Gid, keep) if k]
            P = D[(D[:, 0] == f) & np.isin(D[:, 6], spec["det"])]
            P = P[np.argsort(-P[:, 5])][:, 1:5] if len(P) else np.zeros((0, 4))
            P = P[~ignored_mask(P, ign.get(f, []))] if len(P) else P
            M = iou(G, P); matched_g = set(); matched_p = set()
            if M.size:
                Mm = M.copy(); Mm[Mm < 0.5] = 0
                r, c = linear_sum_assignment(-Mm)
                for i, j in zip(r, c):
                    if Mm[i, j] >= 0.5: matched_g.add(i); matched_p.add(j)
            tp += len(matched_p)
            for i, gid in enumerate(Gid):
                oc = occ.get((seq, f, gid[0], gid[1]), 0); gt_occ[oc] += 1
                if i not in matched_g:
                    fn_occ[oc] += 1
                    EX.setdefault(f"{g}/fn", []).append([seq, f, *map(float, G[i])])
            for j in range(len(P)):
                if j in matched_p: continue
                best_same = M[:, j].max() if len(G) else 0
                if best_same >= 0.5: k = "duplicate"
                elif len(Go) and iou(P[j:j + 1], Go).max() >= 0.5: k = "confused"
                elif len(Gu) and iou(P[j:j + 1], Gu).max() >= 0.5: k = "unlabeled_class"
                elif best_same >= 0.1: k = "localization"
                else: k = "background"
                fp[k] += 1
                EX.setdefault(f"{g}/fp_{k}", []).append([seq, f, *map(float, P[j])])
    nfp = sum(fp.values())
    out[g] = {"tp": tp, "fp": fp, "fn_rate_by_occlusion": {k: float(fn_occ[i] / max(gt_occ[i], 1) * 100) for i, k in enumerate(["none", "partial", "heavy"])},
              "gt_by_occlusion": gt_occ.tolist()}
    print(f"\n[{g}] 점수 {THR} 이상 검출: 맞춤 {tp}, 헛검출 {nfp} (정밀도 {tp / (tp + nfp) * 100:.1f}%)")
    for k, v in fp.items(): print(f"  헛검출 - {k}: {v} ({v / nfp * 100:.1f}%)")
    for i, k in enumerate(["가림 없음", "일부 가림", "심한 가림"]):
        print(f"  {k}: 정답 {int(gt_occ[i])}개 중 놓침 {fn_occ[i] / max(gt_occ[i], 1) * 100:.1f}%")
json.dump(out, open(f"{ROOT}/det_errors_{det_name}.json", "w"), indent=1)
json.dump({k: random.sample(v, min(60, len(v))) for k, v in EX.items()}, open(f"{ROOT}/det_error_examples_{det_name}.json", "w"))
