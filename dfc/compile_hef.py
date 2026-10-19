# 2~4단계를 한 번에: ONNX -> (parse) HAR -> (optimize) INT8 HAR -> (compile) HEF
# usage: python compile_hef.py <model.onnx> <calib.npy> <model_name> [model_script.alls]
#   예: python compile_hef.py remdet_s.onnx calib_640.npy remdet_s_visdrone remdet.alls
import sys, numpy as np
from hailo_sdk_client import ClientRunner

onnx_path, calib_path, name = sys.argv[1:4]
alls = sys.argv[4] if len(sys.argv) > 4 else "remdet.alls"

# 2단계 Parse
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(onnx_path, name, net_input_shapes={"images": [1, 3, 640, 640]})
runner.save_har(f"{name}_parsed.har")
print("[2/4] parse done ->", f"{name}_parsed.har")

# 3단계 Optimize (INT8 양자화)
calib = np.load(calib_path).astype(np.float32)  # (N, 640, 640, 3), 0~255
runner.load_model_script(open(alls).read())
runner.optimize(calib)
runner.save_har(f"{name}_optimized.har")
print("[3/4] optimize done ->", f"{name}_optimized.har")

# 4단계 Compile
hef = runner.compile()
open(f"{name}.hef", "wb").write(hef)
runner.save_har(f"{name}_compiled.har")
print("[4/4] compile done ->", f"{name}.hef")
