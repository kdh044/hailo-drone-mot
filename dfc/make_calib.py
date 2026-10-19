# 3단계 준비: 캘리브레이션 데이터 만들기. 전처리는 Pi 추론 코드(remdet_bench.preprocess)와 똑같이 한다.
# usage: python make_calib.py <image_dir> <out.npy> [N=1024]
import sys, glob, random, cv2, numpy as np

img_dir, out = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
SIZE = 640


def letterbox(img):
    """긴 변을 640에 맞추고 왼쪽 위에 붙인 뒤 나머지는 회색(114). BGR -> RGB. 값은 0~255 그대로."""
    h, w = img.shape[:2]
    s = SIZE / max(h, w); nh, nw = int(h * s), int(w * s)
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


files = sorted(glob.glob(f"{img_dir}/*.jpg"))
random.seed(0); files = random.sample(files, min(n, len(files)))
calib = np.stack([letterbox(cv2.imread(f)) for f in files])  # (N, 640, 640, 3) uint8
np.save(out, calib)
print(out, calib.shape, calib.dtype)
