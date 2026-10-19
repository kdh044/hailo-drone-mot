import pickle, numpy as np, json
from common import ROOT, SEQS, GROUPS, ignored_mask

# 정답 박스가 검출기 입력(640)에서 몇 픽셀이 되는지, 크기별로 검출기가 몇 %를 찾는지(IoU>=0.5, 점수>=0.3)
gt = pickle.load(open(f"{ROOT}/gt.pkl", "rb"))
BINS = [0, 4, 6, 8, 12, 16, 24, 32, 1e9]
LAB = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", ">=32"]

def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)

out = {}
for model in ["tiny", "s"]:
    for g, spec in GROUPS.items():
        hit = np.zeros(len(LAB)); tot = np.zeros(len(LAB)); per_seq = {}
        for seq in SEQS:
            o = gt[seq]; w, h = o["size"]; scale = 640 / max(w, h)
            D = np.load(f"{ROOT}/dets/{model}/{seq}.npy")
            D = D[(D[:, 5] >= 0.3) & np.isin(D[:, 6], spec["det"])]
            objs = np.array([r[:6] for r in o["objs"] if r[6] in spec["gt"]], float).reshape(-1, 6)
            if not len(objs): continue
            ign = {}
            for r in o["ign"]: ign.setdefault(int(r[0]), []).append(r[1:])
            sh = st = 0; sizes = []
            for f in np.unique(objs[:, 0]).astype(int):
                G = objs[objs[:, 0] == f][:, 2:6]
                G = G[~ignored_mask(G, ign.get(f, []))]
                if not len(G): continue
                P = D[D[:, 0] == f][:, 1:5]
                m = (iou(G, P).max(1) >= 0.5) if len(P) else np.zeros(len(G), bool)
                hgt = (G[:, 3] - G[:, 1]) * scale  # 검출기 입력에서의 박스 높이(px)
                b = np.digitize(hgt, BINS) - 1
                np.add.at(tot, b, 1); np.add.at(hit, b, m)
                sh += m.sum(); st += len(G); sizes += list(hgt)
            per_seq[seq] = {"res": f"{w}x{h}", "recall": sh / max(st, 1) * 100, "median_h_px": float(np.median(sizes)), "n": st}
        out[f"{model}/{g}"] = {"bins": dict(zip(LAB, [(int(t), float(hh / t * 100) if t else None) for t, hh in zip(tot, hit)])), "per_seq": per_seq}
        print(f"\n[{model} / {g}] 크기별(검출기 입력 기준 박스 높이) 찾은 비율")
        for l, t, hh in zip(LAB, tot, hit):
            if t: print(f"  {l:>6s}px  정답 {int(t):6d}개 ({t / tot.sum() * 100:4.1f}%)  찾음 {hh / t * 100:5.1f}%")
        for seq, v in per_seq.items():
            print(f"  {seq} {v['res']:>9s}  박스 높이 중앙값 {v['median_h_px']:5.1f}px  찾은 비율 {v['recall']:5.1f}%")
json.dump(out, open(f"{ROOT}/size_analysis.json", "w"), indent=1)
