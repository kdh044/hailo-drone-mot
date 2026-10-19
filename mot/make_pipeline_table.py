import json, re
from common import ROOT

# 동시 실행 E2E FPS(e2e_pipeline_all.json) + 사람·차 함께 HOTA(results.json, *-joint)를 모델별 표로 만든다.
# 출력: pipeline_tables.md (README·보고서에 붙여 넣음)
R = json.load(open(f"{ROOT}/e2e_pipeline_all.json")); H = json.load(open(f"{ROOT}/results.json"))
seq = {}
for l in open(f"{ROOT}/e2e_log.txt"):
    m = re.match(r"(\w+)\s+(\S+)\s+\[reduced\] E2E\s+([\d.]+)", l)
    if m: seq[(m[1], m[2])] = float(m[3])
seq[("s", "holdfast")], seq[("tiny", "holdfast")] = 20.4, 25.9  # 사람·차 함께, 차례대로 (e2e_holdfast_clean.txt)
NAME = {"holdfast": "Holdfast (베이스라인)", "rf-bytetrack": "ByteTrack (roboflow)", "cbiou": "C-BIoU", "mcbyte": "McByte",
        "sfsort": "SFSORT", "sort": "SORT", "bytetrack": "ByteTrack (boxmot)", "botsort": "BoT-SORT", "ocsort": "OC-SORT",
        "ocsort-byte": "OC-SORT+BYTE", "detecttrack-uav": "DetectTrack-UAV", "detecttrack-uav-noar": "DetectTrack-UAV (가로 박스 필터 끔)",
        "smot-ocsort": "SMOT-OCSORT", "botsort-cmc": "BoT-SORT + CMC", "detecttrack-uav-cmc": "DetectTrack-UAV + CMC",
        "boosttrack": "BoostTrack", "sftrack": "SFTrack", "deepocsort-cmc": "Deep OC-SORT (CMC)", "hybridsort": "Hybrid-SORT"}
out = []
for m in ("s", "tiny"):
    rows = []
    for t in NAME:
        r = R.get(f"{m}__{t}__pipe"); h = H.get(f"{m}__{t}-joint")
        if not r or not h or "COMBINED_SEQ" not in h.get("all", {}): continue
        g = lambda k: h[k]["COMBINED_SEQ"]["HOTA"]
        rows.append((g("all"), g("person"), g("vehicle"), h["all"]["COMBINED_SEQ"]["IDF1"], seq.get((m, t)), r["e2e_fps"], r["stage_fps"]["track"], t, r["frames"]))
    rows.sort(key=lambda x: -x[0])
    out += [f"**RemDet-{'S' if m == 's' else 'Tiny'}**", "",
            "| 트래커 | 통합 HOTA | 사람 HOTA | 차 HOTA | 통합 IDF1 | E2E FPS (차례대로) | E2E FPS (동시 실행) | 트래커 단독 FPS | 30 FPS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for a, p, v, i, s, e, tk, t, fr in rows:
        b = (lambda x: f"**{x}**") if t == "holdfast" else (lambda x: x)
        nm = NAME[t] + (" *" if fr < 1000 else "")
        out.append(f"| {b(nm)} | {b(f'{a:.1f}')} | {p:.1f} | {v:.1f} | {i:.1f} | {s if s else '-'} | {b(f'{e:.1f}')} | {tk:.0f} | {'O' if e >= 30 else 'X'} |")
    out.append("")
open(f"{ROOT}/pipeline_tables.md", "w").write("\n".join(out)); print("\n".join(out))
