#!/bin/bash
# 남은 트래커 실행이 끝나면: DetectTrack-UAV(CMC+필터 끔) 추가 실행 -> 통합 데이터 갱신 -> 채점
cd ~/hailo-test/mot
while pgrep -f "^/home/danny/mot-venv/bin/python run_trackers" >/dev/null; do sleep 30; done
~/mot-venv/bin/python run_trackers.py --dets tiny s --trackers detecttrack-uav-cmc-noar --jobs 2 2>&1 | grep -E "track |Error"
python3 make_all_group.py
names=$(ls data/trackers/VDall-val | tr '\n' ' ')
~/mot-venv/bin/python evaluate.py $names 2>&1 | grep -v BURST | tail -3
echo "남은 트래커 채점 끝"
