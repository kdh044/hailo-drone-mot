import os, pickle, numpy as np
from common import *

# writes TrackEval MOTChallenge layout: gt/VD<group>-val/<seq>/{gt/gt.txt, seqinfo.ini}, seqmaps/VD<group>-val.txt
gt = load_gt()
pickle.dump(gt, open(f"{ROOT}/gt.pkl", "wb"))
for g, spec in GROUPS.items():
    bench = f"VD{g}-val"
    for seq in SEQS:
        o = gt[seq]; ign = [r[1:] for r in o["ign"]]
        ign_by_f = {}
        for r in o["ign"]: ign_by_f.setdefault(r[0], []).append(r[1:])
        rows = []
        for f, tid, x1, y1, x2, y2, lab in o["objs"]:
            if lab not in spec["gt"]: continue
            if ignored_mask([[x1, y1, x2, y2]], ign_by_f.get(f, []))[0]: continue
            # ids are per-class in VisDrone; offset by label so merged classes don't collide
            rows.append([f, tid + 100000 * spec["gt"].index(lab), x1, y1, x2, y2, 1])
        d = f"{ROOT}/data/gt/{bench}/{seq}"
        write_mot(f"{d}/gt/gt.txt", rows)
        w, h = o["size"]
        open(f"{d}/seqinfo.ini", "w").write(
            f"[Sequence]\nname={seq}\nframeRate=30\nseqLength={o['n']}\nimWidth={w}\nimHeight={h}\n")
    os.makedirs(f"{ROOT}/data/seqmaps", exist_ok=True)
    open(f"{ROOT}/data/seqmaps/{bench}.txt", "w").write("name\n" + "\n".join(SEQS) + "\n")
    print(g, sum(1 for s in SEQS for _ in open(f"{ROOT}/data/gt/{bench}/{s}/gt/gt.txt")), "gt boxes")
