import sys, glob, subprocess, cv2, numpy as np
from common import ROOT, SEQ_ROOT, GROUPS
from run_trackers import make, step

# usage: ~/mot-venv/bin/python render_video.py <dets name> <tracker> <seq> <out.mp4> [caption]
# draws tracks with short trails; H.264 via ffmpeg so the mp4 plays in browsers
det_name, trk, seq, out = sys.argv[1:5]
caption = sys.argv[5] if len(sys.argv) > 5 else ""
W = 1344
COLOR = {"person": (0, 200, 255), "vehicle": (0, 255, 0)}
files = sorted(glob.glob(f"{SEQ_ROOT}/{seq}/*.jpg"))
h0, w0 = cv2.imread(files[0]).shape[:2]; s = W / w0; H = int(round(h0 * s / 2) * 2)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                       "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
D = np.load(f"{ROOT}/dets/{det_name}/{seq}.npy")
trackers = {g: make(trk) for g in GROUPS}
trail = {}
for fi, f in enumerate(files, 1):
    full = cv2.imread(f); img = cv2.resize(full, (W, H))
    step.frame = full
    fd = D[D[:, 0] == fi]; n = {g: 0 for g in GROUPS}
    for g, spec in GROUPS.items():
        r = step(trackers[g], fd[np.isin(fd[:, 6], spec["det"])][:, 1:7])
        for x1, y1, x2, y2, tid, _ in r:
            n[g] += 1; key = (g, int(tid))
            c = tuple(int(v) for v in np.random.RandomState(int(tid) + (0 if g == "person" else 7777)).randint(60, 255, 3))
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            trail.setdefault(key, []).append((fi, ((p1[0] + p2[0]) // 2, p2[1])))
            pts = [p for t, p in trail[key] if fi - t < 30]
            for a, b in zip(pts, pts[1:]): cv2.line(img, a, b, c, 1)
            cv2.rectangle(img, p1, p2, c, 2)
            cv2.putText(img, f"{g[0].upper()}{int(tid)}", (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR[g], 1)
    cv2.rectangle(img, (0, 0), (W, 46), (0, 0, 0), -1)
    cv2.putText(img, f"RemDet-{det_name.upper()} (Hailo-8) + {trk}   {seq}  frame {fi}/{len(files)}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, f"persons {n['person']}  vehicles {n['vehicle']}   {caption}", (8, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    ff.stdin.write(img.tobytes())
ff.stdin.close(); ff.wait()
print("wrote", out)
