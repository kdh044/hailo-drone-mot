import json, re
R = json.load(open('results.json'))
e2e = {}
for l in open('e2e_log.txt'):
    m = re.match(r'(\w+)\s+(\S+)\s+\[reduced\] E2E\s+([\d.]+)', l)
    if m: e2e[(m[1], m[2])] = float(m[3])
names = {'holdfast':('Holdfast','드론'),'sftrack':('SFTrack','드론'),'detecttrack-uav-cmc':('DetectTrack-UAV + CMC','드론'),'detecttrack-uav-cmc-noar':('DetectTrack-UAV + CMC, 필터 끔','드론'),'detecttrack-uav-noar':('DetectTrack-UAV (필터 끔)','드론'),'detecttrack-uav':('DetectTrack-UAV (원래 설정)','드론'),'smot-ocsort':('SMOT-OCSORT','드론'),'csf-tst':('CSF-TST','드론'),'mosaic':('MOSAIC-Tracker','드론'),
 'hybridsort':('Hybrid-SORT','일반'),'bytetrack':('ByteTrack (boxmot)','일반'),'rf-bytetrack':('ByteTrack (roboflow)','일반'),'botsort-cmc':('BoT-SORT + CMC','일반'),'cbiou':('C-BIoU','일반'),'ocsort':('OC-SORT','일반'),'mcbyte':('McByte (마스크 없음)','일반'),'ocsort-byte':('OC-SORT + BYTE','일반'),'boosttrack':('BoostTrack','일반'),'botsort':('BoT-SORT','일반'),'deepocsort-cmc':('Deep OC-SORT (CMC)','일반'),'sort':('SORT','일반'),'sfsort':('SFSORT','일반')}
out = ["# 트래커 전체 지표\n\n검출 모델별 표 2개(S, Tiny)이고, **통합 HOTA 순서로 정렬**했다(통합 = 사람+차를 한 번에 채점). 실험이 끝나는 대로 이 표를 계속 갱신한다.\n\n- E2E FPS: 사진 열기(작게 풀기)부터 검출·후처리·트래커까지 전부 포함한 실제 속도. 4개 영상(1344~4K)에서 측정, 다른 작업을 멈춘 상태.\n- 지표: HOTA = √(DetA × AssA), DetA = 검출 정확도, AssA = 연결 정확도, IDF1 = 같은 객체에 같은 ID가 붙은 비율, IDSW = ID 바뀐 횟수.\n"]
pending = []
for m, mk in [('S', 's'), ('Tiny', 'tiny')]:
    rows = []
    for k, (n, t) in names.items():
        r = R.get(f'{mk}__{k}')
        if r and 'all' in r and 'person' in r and 'vehicle' in r: rows.append((k, n, t, r))
        else: pending.append(f"{m} {n}")
    rows.sort(key=lambda x: -x[3]['all']['COMBINED_SEQ']['HOTA'])
    out.append(f"## {m} 검출\n\n| 순위 | 트래커 | 종류 | 통합 HOTA | 통합 DetA | 통합 AssA | 통합 IDF1 | 통합 IDSW | 사람 HOTA | 차 HOTA | E2E FPS |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, (k, n, t, r) in enumerate(rows, 1):
        a = r['all']['COMBINED_SEQ']; fps = e2e.get((mk, k)); fps = '측정 중' if fps is None else f'{fps:.1f}'
        out.append(f"| {i} | {n} | {t} | {a['HOTA']:.1f} | {a['DetA']:.1f} | {a['AssA']:.1f} | {a['IDF1']:.1f} | {int(a['IDSW'])} | {r['person']['COMBINED_SEQ']['HOTA']:.1f} | {r['vehicle']['COMBINED_SEQ']['HOTA']:.1f} | {fps} |")
    det = e2e.get((mk, 'none'))
    if det: out.append(f"\n트래커 없이 검출만 했을 때 E2E: {det:.1f} FPS")
    out.append("")
out.append("아직 실행·채점 중: " + ", ".join(pending) + ". ReID 트래커 4종(StrongSORT 등)은 다음 주.")
md = '\n'.join(out); open('all_metrics2.md', 'w').write(md); print(md)
