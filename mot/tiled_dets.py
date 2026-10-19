import sys, os, glob, time, cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/hailo-test"))
from hailo_platform import (VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
import remdet_bench as rb

# usage: python3 tiled_dets.py <hef> <seq dir> <out dir> <nx> <ny>
# 프레임을 nx×ny 조각(10% 겹침)으로 잘라 조각마다 640 입력으로 검출하고, 결과를 합쳐 NMS.
# nx=ny=1 이면 지금 방식(프레임 전체를 640으로 축소)과 같다. 출력 형식은 dump_dets.py와 같다.
hef_path, seq_dir, out, nx, ny = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rb.CONF = 0.05; OV = 0.10
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(f"{seq_dir}/*.jpg"))
hef = rb.HEF(hef_path); outs = rb.get_outs(hef)

def tiles(W, H):
    tw, th = W / (nx - (nx - 1) * OV), H / (ny - (ny - 1) * OV)
    for j in range(ny):
        for i in range(nx):
            x0 = int(i * tw * (1 - OV)); y0 = int(j * th * (1 - OV))
            yield x0, y0, min(W, int(x0 + tw)), min(H, int(y0 + th))

with VDevice() as target:
    ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
        rows, t_inf, n_inf = [], 0.0, 0
        H0, W0 = cv2.imread(files[0]).shape[:2]
        for fi, f in enumerate(files, 1):
            img = cv2.imread(f)
            if img.shape[:2] != (H0, W0): img = cv2.resize(img, (W0, H0))
            B, S, C = [], [], []
            for x0, y0, x1, y1 in tiles(W0, H0):
                x, s = rb.preprocess(img[y0:y1, x0:x1])
                t0 = time.perf_counter(); res = pipe.infer({in_name: x}); t_inf += time.perf_counter() - t0; n_inf += 1
                b, sc, c = rb.postprocess(res, outs, s)
                if len(b): B.append(b + [x0, y0, x0, y0]); S.append(sc); C.append(c)
            if B:
                b = np.concatenate(B); sc = np.concatenate(S); c = np.concatenate(C)
                if nx * ny > 1:  # 조각 경계에서 겹친 박스 정리 (클래스별 NMS)
                    xywh = np.c_[b[:, :2] + c[:, None] * 8192, b[:, 2:] - b[:, :2]]
                    k = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), rb.CONF, rb.IOU)).reshape(-1)
                    b, sc, c = b[k], sc[k], c[k]
                rows.append(np.c_[np.full(len(b), fi), b, sc, c])
        name = os.path.basename(seq_dir.rstrip("/"))
        np.save(f"{out}/{name}.npy", np.concatenate(rows).astype(np.float32))
        print(f"{name} {nx}x{ny} 조각 {nx*ny}개/프레임, Hailo 추론만 {n_inf / t_inf / (nx*ny):.1f} 프레임/초", flush=True)
