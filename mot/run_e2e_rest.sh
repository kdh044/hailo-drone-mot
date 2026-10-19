#!/bin/bash
# 채점이 끝나면 남은 E2E 측정 -> 끝나면 멈춰 둔 트래커 실행 재개
cd ~/hailo-test/mot
while pgrep -f "^/home/danny/mot-venv/bin/python evaluate.py" >/dev/null; do sleep 10; done
SEQ="uav0000086_00000_v uav0000305_00000_v uav0000117_02622_v uav0000268_05773_v"
FAST="none rf-bytetrack cbiou mcbyte sfsort sort bytetrack botsort ocsort ocsort-byte detecttrack-uav detecttrack-uav-noar smot-ocsort holdfast"
SLOW="botsort-cmc detecttrack-uav-cmc boosttrack sftrack deepocsort-cmc hybridsort"
run() { ~/mot-venv/bin/python e2e_bench.py --model $1 --trackers $2 --seqs $SEQ --max-frames $3 --decode ${4:-reduced} 2>&1 | grep -E "E2E|Error|Trace"; }
run tiny hybridsort 50
run s none 150 full
for t in $FAST; do run s $t 150; done
for t in $SLOW; do run s $t 50; done
run tiny hybridsort 50
kill -CONT $(pgrep -f "^/home/danny/mot-venv/bin/python run_trackers") 2>/dev/null
echo "E2E 측정 끝, 트래커 실행 재개"
