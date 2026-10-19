#!/bin/bash
# 4K 영상 조각 검출: 24조각(4K 화질) 먼저, 그다음 9조각, 4조각. 도는 동안 트래커 실행은 잠시 멈춘다.
cd ~/hailo-test/mot
P=$(pgrep -f "^/home/danny/mot-venv/bin/python run_trackers"); [ -n "$P" ] && kill -STOP $P
SEQ=~/datasets/visdrone/mot-val-hf/sequences/uav0000268_05773_v
for g in "6 4" "3 3" "2 2"; do
  set -- $g
  python3 tiled_dets.py ../models/remdet_s_visdrone.hef $SEQ dets/s_tile$1x$2 $1 $2 2>&1 | grep -v HailoRT
done
[ -n "$P" ] && kill -CONT $P
echo "조각 검출 끝"
