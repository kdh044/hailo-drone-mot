#!/bin/sh
# Builds libholdfast_c.so from ~/ext/holdfast (tracker core + LK/RANSAC CMC only; no ONNX, no apps).
# Eigen 3.4.0 headers from ~/ext/eigen-3.4.0 (header-only; libeigen3-dev is not installed); system OpenCV 4.10.
set -e
HF=${HF:-$HOME/ext/holdfast}
EIGEN=${EIGEN:-$HOME/ext/eigen-3.4.0}
OUT=$(dirname "$0")
g++ -std=c++20 -O3 -DNDEBUG -fPIC -shared -Wall -Wextra \
  -I"$HF/include" -I"$EIGEN" -I/usr/include/opencv4 \
  "$HF/src/bbox.cpp" "$HF/src/kalman.cpp" "$HF/src/hungarian.cpp" "$HF/src/tracker.cpp" \
  "$HF/src/app/camera_motion.cpp" "$OUT/hf_capi.cpp" \
  -lopencv_video -lopencv_calib3d -lopencv_imgproc -lopencv_core \
  -o "$OUT/libholdfast_c.so"
echo built "$OUT/libholdfast_c.so"
