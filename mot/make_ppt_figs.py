import json, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as fm
FONT = [f.fname for f in fm.fontManager.ttflist if "NanumSquare" in f.name]
for f in FONT: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "NanumSquare"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# 1) 정확도-속도 산점도 (S 검출, 통합 HOTA vs E2E FPS)
R = json.load(open("results.json")); E = {}
for l in open("e2e_log.txt"):
    m = re.match(r"(\w+)\s+(\S+)\s+\[reduced\] E2E\s+([\d.]+)", l)
    if m: E[(m[1], m[2])] = float(m[3])
names = {'holdfast':('Holdfast',1),'sftrack':('SFTrack',1),'detecttrack-uav-cmc':('DetectTrack-UAV+CMC',1),'detecttrack-uav-noar':('DetectTrack-UAV',1),'smot-ocsort':('SMOT',1),
 'hybridsort':('Hybrid-SORT',0),'bytetrack':('ByteTrack(boxmot)',0),'rf-bytetrack':('ByteTrack',0),'botsort-cmc':('BoT-SORT+CMC',0),'cbiou':('C-BIoU',0),'ocsort':('OC-SORT',0),
 'mcbyte':('McByte',0),'ocsort-byte':('OC-SORT+BYTE',0),'boosttrack':('BoostTrack',0),'botsort':('BoT-SORT',0),'deepocsort-cmc':('Deep OC-SORT',0),'sort':('SORT',0),'sfsort':('SFSORT',0)}
LABEL = {'holdfast','sftrack','rf-bytetrack','botsort-cmc','cbiou','hybridsort','sort','detecttrack-uav-cmc'}
fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
for k, (n, drone) in names.items():
    r = R.get(f"s__{k}"); fps = E.get(("s", k))
    if not r or "all" not in r or fps is None: continue
    h = r["all"]["COMBINED_SEQ"]["HOTA"]
    ax.scatter(fps, h, s=70, marker="o" if drone else "s", color=BLUE if drone else ORANGE, edgecolor=SURF, linewidth=2, zorder=3)
    off = {"cbiou": (-46, 8), "rf-bytetrack": (6, -14)}.get(k, (7, 4))
    if k in LABEL: ax.annotate(n, (fps, h), xytext=off, textcoords="offset points", fontsize=9, color=INK)
ax.axvline(30, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); ax.text(30.5, 34.2, "실시간 목표 30 FPS", color=MUTED, fontsize=9)
ax.set_xlabel("E2E 속도 (FPS, 사진 열기~트래커, 라즈베리파이 5 + Hailo-8)", color=INK); ax.set_ylabel("통합 HOTA (사람+차)", color=INK)
ax.set_title("S 검출 기준 트래커 정확도와 속도", color=INK, fontsize=13, loc="left", pad=12)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors=MUTED)
ax.scatter([], [], marker="o", color=BLUE, s=60, label="드론 전용"); ax.scatter([], [], marker="s", color=ORANGE, s=60, label="일반")
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.set_xlim(0, 46); ax.set_ylim(33, 48.5)
fig.tight_layout(); fig.savefig("figs/ppt_hota_vs_fps.png", facecolor=SURF)

# 2) 베이스라인(Holdfast) 파이프라인 그림
fig, ax = plt.subplots(figsize=(13, 6.2), dpi=200); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF); ax.axis("off")
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)
def box(x, y, w, h, title, sub, col, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", lw=1.6, ec=col, fc=fc))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=11.5, color=INK, weight="bold")
    ax.text(x + w / 2, y + h - 0.78, sub, ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.45)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
LB, LO = "#e6f0fb", "#fdebe3"
ax.text(0.1, 6.0, "베이스라인 추적 파이프라인: RemDet(Hailo-8) + Holdfast", fontsize=14, color=INK, weight="bold", va="top")
box(0.1, 3.4, 1.9, 1.7, "카메라 프레임", "드론 영상\n(jpg면 작게 풀기)", MUTED, SURF)
box(2.4, 3.4, 2.0, 1.7, "RemDet 검출", "640×640 입력\nHailo-8에서 추론\n박스 + 점수 + 클래스", ORANGE, LO)
box(4.8, 3.4, 2.0, 1.7, "① 카메라 움직임 보정", "LK 광학 흐름 + RANSAC\n→ 화면 전체 이동(어파인)\n→ 모든 트랙 위치 보정", BLUE, LB)
box(7.2, 3.4, 2.0, 1.7, "② 칼만 예측", "상태 = 위치·크기·속도\n실제 시간 간격(dt)으로\n다음 위치 예측", BLUE, LB)
box(9.6, 3.2, 3.3, 1.9, "③ 단계별 연결", "1) 높은 점수(≥0.5) ↔ 트랙: IoU\n2) 낮은 점수(0.1~0.5) ↔ 남은 트랙: IoU\n3) 잃은 트랙 복구: 마할라노비스 거리\n4) 임시 트랙 연결 → 5) 새 트랙(≥0.6)", BLUE, LB)
for x1, x2 in ((2.0, 2.4), (4.4, 4.8), (6.8, 7.2), (9.2, 9.6)): arrow(x1, 4.25, x2, 4.25)
box(9.6, 0.5, 3.3, 2.2, "④ 트랙 상태 관리", "임시 --3번 연속--> 확정\n확정 --놓침--> 잃음 --복구--> 확정(같은 ID)\n잃음 1.5초 넘으면 삭제\n다시 찾으면 OC-SORT식 경로 보정", BLUE, LB)
arrow(11.25, 3.2, 11.25, 2.7)
box(5.6, 0.5, 3.4, 2.2, "출력", "사람(P)·차(V) 트랙 ID와 박스\n사람용·차용 트래커는 따로 동작", MUTED, SURF)
arrow(9.6, 1.6, 9.0, 1.6)
ax.text(0.1, 2.55, "핵심 아이디어", fontsize=11, color=INK, weight="bold")
ax.text(0.1, 2.15, "· 드론이 움직여도 화면 전체 이동을 빼고 매칭\n· 점수 낮은 박스도 2차로 사용(ByteTrack)\n· 잠깐 사라진 객체는 불확실성 범위로 다시 찾음\n· GPU 없이 C++로 CPU에서 동작", fontsize=9.2, color=MUTED, va="top", linespacing=1.6)
fig.savefig("figs/ppt_baseline_holdfast.png", facecolor=SURF, bbox_inches="tight")
print("ok")
