#!/bin/bash
# 모든 트래커를 사람·차 함께(트래커 1개)로 돌려 HOTA 채점
cd ~/hailo-test/mot; PY=~/mot-venv/bin/python
FAST="rf-bytetrack cbiou mcbyte sfsort sort bytetrack botsort ocsort ocsort-byte detecttrack-uav detecttrack-uav-noar smot-ocsort botsort-cmc detecttrack-uav-cmc boosttrack"
SLOW="sftrack deepocsort-cmc hybridsort"
J() { for t in $*; do echo -n "$t-joint "; done; }
$PY run_trackers.py --dets s tiny --trackers $(J $FAST) --jobs 4
$PY make_all_group.py && $PY evaluate.py $(for d in s tiny; do for t in $FAST; do echo -n "${d}__$t-joint "; done; done)
echo "빠른 트래커 채점 끝"
$PY run_trackers.py --dets s tiny --trackers $(J $SLOW) --jobs 4
$PY make_all_group.py && $PY evaluate.py $(for d in s tiny; do for t in $SLOW; do echo -n "${d}__$t-joint "; done; done)
echo 끝
