#!/bin/bash
# 모든 트래커를 동시 실행 파이프라인(사람·차 함께)으로 S, Tiny 측정
cd ~/hailo-test/mot; PY=~/mot-venv/bin/python
FAST="holdfast rf-bytetrack cbiou mcbyte sfsort sort bytetrack botsort ocsort ocsort-byte detecttrack-uav detecttrack-uav-noar smot-ocsort botsort-cmc detecttrack-uav-cmc boosttrack"
SLOW="sftrack deepocsort-cmc hybridsort"
for m in s tiny; do
  $PY e2e_pipeline_all.py --model $m --trackers holdfast --mode seq --timeline --max-frames 200
  $PY e2e_pipeline_all.py --model $m --trackers holdfast --timeline --max-frames 200
  $PY e2e_pipeline_all.py --model $m --trackers $FAST
  $PY e2e_pipeline_all.py --model $m --trackers $SLOW --max-frames 100
done 2>&1 | grep -v "^\[HailoRT\]"
echo 끝
