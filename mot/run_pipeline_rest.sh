#!/bin/bash
# 멈췄던 나머지: S 느린 3개 + Tiny 전체 (동시 실행, 사람·차 함께)
cd ~/hailo-test/mot; PY=~/mot-venv/bin/python
FAST="holdfast rf-bytetrack cbiou mcbyte sfsort sort bytetrack botsort ocsort ocsort-byte detecttrack-uav detecttrack-uav-noar smot-ocsort botsort-cmc detecttrack-uav-cmc boosttrack"
SLOW="sftrack deepocsort-cmc hybridsort"
{
$PY e2e_pipeline_all.py --model s --trackers $SLOW --max-frames 100
$PY e2e_pipeline_all.py --model tiny --trackers $FAST
$PY e2e_pipeline_all.py --model tiny --trackers $SLOW --max-frames 100
$PY e2e_pipeline_all.py --model tiny --trackers holdfast --mode seq --timeline --max-frames 200
$PY e2e_pipeline_all.py --model tiny --trackers holdfast --timeline --max-frames 200
} 2>&1 | grep --line-buffered -v "^\[HailoRT\]"
echo 끝
