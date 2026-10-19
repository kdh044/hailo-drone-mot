#!/bin/bash
# 다른 작업을 잠깐 멈추고 E2E 측정 -> 끝나면 다시 이어서 실행
cd ~/hailo-test/mot
PIDS=$(pgrep -f "^/home/danny/mot-venv/bin/python (run_trackers|evaluate)")
[ -n "$PIDS" ] && kill -STOP $PIDS
SEQ="uav0000086_00000_v uav0000305_00000_v uav0000117_02622_v uav0000268_05773_v"
FAST="none rf-bytetrack cbiou mcbyte sfsort sort bytetrack botsort ocsort ocsort-byte detecttrack-uav detecttrack-uav-noar smot-ocsort holdfast"
SLOW="botsort-cmc detecttrack-uav-cmc boosttrack sftrack deepocsort-cmc hybridsort"
for m in tiny s; do
  ~/mot-venv/bin/python e2e_bench.py --model $m --trackers none --seqs $SEQ --max-frames 150 --decode full 2>&1 | grep -E "E2E|Error|Trace"
  for t in $FAST $SLOW; do
    ~/mot-venv/bin/python e2e_bench.py --model $m --trackers $t --seqs $SEQ --max-frames 150 --decode reduced 2>&1 | grep -E "E2E|Error|Trace"
  done
done
[ -n "$PIDS" ] && kill -CONT $PIDS
echo "E2E 측정 끝, 다른 작업 다시 실행"
