import sys, glob, os, time, threading, queue, cv2, numpy as np
from common import ROOT, SEQ_ROOT, SEQS, GROUPS
from run_trackers import make, step

# usage: ~/mot-venv/bin/python view_replay.py <dets name> <tracker name>
# replays cached detections (dets/<name>/) through a tracker at video speed; space pause, n next seq, q quit
det_name, trk = sys.argv[1], sys.argv[2]
COLOR = {"person": (0, 200, 255), "vehicle": (0, 255, 0)}
WIN = f"{det_name} + {trk}"

def reader(q, files):
    for f in files:
        img = cv2.imread(f)
        s = 1344 / img.shape[1]
        q.put((cv2.resize(img, None, fx=s, fy=s) if s < 1 else img, img if s < 1 else None, s))
    q.put(None)

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1344, 756)
quit_all = False
for seq in SEQS:
    D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy")
    trackers = {g: make(trk) for g in GROUPS}
    q = queue.Queue(maxsize=16)
    threading.Thread(target=reader, args=(q, sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))), daemon=True).start()
    fi, paused, tms = 0, False, 0.0
    while (item := q.get()) is not None:
        t_start = time.perf_counter()
        img, full, s = item; fi += 1
        step.frame = full if full is not None else img
        fd = D[D[:, 0] == fi]; n = 0; tt = 0.0
        for g, spec in GROUPS.items():
            t0 = time.perf_counter()
            r = step(trackers[g], fd[np.isin(fd[:, 6], spec["det"])][:, 1:7])
            tt += time.perf_counter() - t0
            for x1, y1, x2, y2, tid, _ in r:
                n += 1
                p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
                c = tuple(int(v) for v in np.random.RandomState(int(tid)).randint(60, 255, 3))
                cv2.rectangle(img, p1, p2, c, 2)
                cv2.putText(img, f"{g[0].upper()}{int(tid)}", (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR[g], 1)
        tms = 0.9 * tms + 0.1 * tt * 1000 if tms else tt * 1000
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(img, f"{seq}  frame {fi}  tracks {n}   [{det_name} dets + {trk}]", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"tracker {tms:.1f} ms/frame   P=person V=vehicle", (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow(WIN, img)
        wait = max(1, 33 - int((time.perf_counter() - t_start) * 1000))
        key = cv2.waitKey(0 if paused else wait) & 0xFF
        if key == ord("q"): quit_all = True; break
        if key == ord("n"): break
        if key == ord(" "): paused = not paused
    if quit_all: break
cv2.destroyAllWindows()
