import os, shutil
from common import ROOT, SEQS

# 사람+차를 합친 "통합" 벤치마크(VDall-val). 사람·차 따로 결과는 그대로 둔다.
# 차 ID에는 1,000,000을 더해 사람 ID와 겹치지 않게 한다.
OFF = 1_000_000
def merge(files_offsets, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as w:
        for path, off in files_offsets:
            if not os.path.exists(path): continue
            for line in open(path):
                p = line.split(","); p[1] = str(int(float(p[1])) + off); w.write(",".join(p))
for seq in SEQS:
    d = f"{ROOT}/data/gt/VDall-val/{seq}"; os.makedirs(f"{d}/gt", exist_ok=True)
    shutil.copy(f"{ROOT}/data/gt/VDperson-val/{seq}/seqinfo.ini", f"{d}/seqinfo.ini")
    merge([(f"{ROOT}/data/gt/VDperson-val/{seq}/gt/gt.txt", 0), (f"{ROOT}/data/gt/VDvehicle-val/{seq}/gt/gt.txt", OFF)], f"{d}/gt/gt.txt")
shutil.copy(f"{ROOT}/data/seqmaps/VDperson-val.txt", f"{ROOT}/data/seqmaps/VDall-val.txt")
done = []
for t in sorted(os.listdir(f"{ROOT}/data/trackers/VDperson-val")):
    if t.split("__")[0] not in ("tiny", "s"): continue
    pv = [f"{ROOT}/data/trackers/VD{g}-val/{t}/data/{s}.txt" for g in ("person", "vehicle") for s in SEQS]
    if not all(os.path.exists(p) for p in pv): continue
    for s in SEQS:
        merge([(f"{ROOT}/data/trackers/VDperson-val/{t}/data/{s}.txt", 0), (f"{ROOT}/data/trackers/VDvehicle-val/{t}/data/{s}.txt", OFF)],
              f"{ROOT}/data/trackers/VDall-val/{t}/data/{s}.txt")
    done.append(t)
print(len(done), "개 트래커 통합 완료")
