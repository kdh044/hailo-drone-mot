import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from common import ROOT

# 동시 실행 파이프라인 그림: 1) 차례대로 vs 동시 실행 타임라인  2) 자원 사용(얇은 막대)  3) 트래커별 속도·정확도
for f in [f.fname for f in fm.fontManager.ttflist if "NanumSquare" in f.name]: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "NanumSquare"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#222222", "#888888", "#e6e6e6"
COL = {"read": "#4e79a7", "hailo": "#e15759", "post": "#f28e2b", "track": "#59a14f", "other": "#b07aa1", "idle": "#e6e6e6"}
NAME = {"read": "사진 열기+640 줄이기", "hailo": "Hailo 추론", "post": "후처리(NMS)", "track": "트래커(Holdfast)"}
R = json.load(open(f"{ROOT}/e2e_pipeline_all.json"))
HOTA = {k: v["all"]["COMBINED_SEQ"]["HOTA"] for k, v in json.load(open(f"{ROOT}/results.json")).items() if "COMBINED_SEQ" in v.get("all", {})}

def tl(model, mode):
    for k in (f"{model}__holdfast__{mode}__tl", f"{model}__holdfast__{mode}"):
        if k in R and "timeline" in R[k]: return R[k]
# ---- 1) 타임라인: 같은 1초 구간에서 단계가 언제 도는지
for model in ("s", "tiny"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    for ax, mode, title in zip(axes, ("seq", "pipe"), ("차례대로 (한 단계가 끝나야 다음 단계)", "동시 실행 (스레드 3개)")):
        r = tl(model, mode)
        if r is None: continue
        T = r["timeline"]; t0 = T[60][2][0]  # 시작 직후는 빼고, 60번째 프레임 트래커 시작부터 1초
        rows = [("read", lambda fr: fr[0]), ("hailo", lambda fr: (fr[1][0], fr[1][1])), ("post", lambda fr: (fr[1][1], fr[1][2])), ("track", lambda fr: fr[2])]
        n = 0
        n = sum(1 for fr in T if 0 <= fr[2][1] - t0 <= 1.0)
        for i, fr in enumerate(T):
            for y, (k, get) in enumerate(rows):
                a, b = get(fr); a -= t0; b -= t0
                if b < 0 or a > 1.0: continue
                a, b = max(a, 0), min(b, 1.0)
                ax.barh(3 - y, b - a, left=a, height=0.62, color=COL[k], alpha=1.0 if i % 2 == 0 else 0.6, lw=0)
        ax.set_yticks(range(4)); ax.set_yticklabels([NAME[k] for k, _ in rows][::-1], fontsize=9.5, color=INK)
        ax.set_xlim(0, 1.0); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.set_title(f"{title} — 1초에 {n}프레임 완료", loc="left", fontsize=11.5, color=INK)
    axes[1].set_xlabel("시간 (초)", color=MUTED)
    fig.suptitle(f"RemDet-{model.upper()} + Holdfast: 1초 동안 각 단계가 일하는 구간 (진하고 연한 색 = 서로 다른 프레임)", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/pipe_timeline_{model}.png", dpi=160); plt.close(fig)

# ---- 2) 자원 사용: 얇은 막대. CPU 는 Pi 전체(4코어)=100%, Hailo 는 추론 중인 시간 비율, GPU 는 안 씀
rows = []
for model in ("s", "tiny"):
    for mode, lab in (("seq", "차례대로"), ("pipe", "동시 실행")):
        k = f"{model}__holdfast__{mode}" if mode == "pipe" else (f"{model}__holdfast__seq__tl" if f"{model}__holdfast__seq__tl" in R else f"{model}__holdfast__seq")
        if k in R: rows.append((f"{model.upper()} · {lab}", R[k]))
fig, ax = plt.subplots(figsize=(11, 0.9 + 1.25 * len(rows)))
y = 0; yt, yl = [], []
for name, r in rows[::-1]:
    nc = r["ncore"]; sc = r["stage_cpu_pct"]
    if r["mode"] == "seq": parts = [("track", sum(sc[k] for k in ("read", "infer", "track")) / nc)]  # 한 스레드가 전부 함
    else: parts = [("read", sc["read"] / nc), ("hailo", sc["infer"] / nc), ("track", sc["track"] / nc)]
    parts.append(("other", max(sc["other"], 0) / nc))
    left = 0
    for k, v in parts:
        ax.barh(y + 0.5, v, left=left, height=0.22, color=COL[k] if not (r["mode"] == "seq" and k == "track") else "#9c9c9c", lw=0); left += v
    ax.barh(y + 0.5, 100 - left, left=left, height=0.22, color=COL["idle"], lw=0)
    ax.text(101, y + 0.5, f"{left:.0f}%", va="center", fontsize=9, color=INK)
    ax.barh(y + 0.2, r["hailo_busy_pct"], height=0.22, color="#e15759", lw=0)
    ax.barh(y + 0.2, 100 - r["hailo_busy_pct"], left=r["hailo_busy_pct"], height=0.22, color=COL["idle"], lw=0)
    ax.text(101, y + 0.2, f"{r['hailo_busy_pct']:.0f}%", va="center", fontsize=9, color=INK)
    ax.barh(y - 0.1, 0.4, height=0.22, color="#999999", lw=0); ax.text(101, y - 0.1, "0%", va="center", fontsize=9, color=INK)
    yt += [y + 0.5, y + 0.2, y - 0.1]; yl += [f"{name}  ({r['e2e_fps']:.1f} FPS)   CPU", "Hailo-8 NPU", "GPU"]
    y += 1.25
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9.5, color=INK); ax.set_xlim(0, 108); ax.set_xticks(range(0, 101, 20))
ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)], color=MUTED); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COL["read"], label="열기+줄이기"), Patch(color=COL["hailo"], label="Hailo 호출+후처리 / NPU 사용"),
                   Patch(color=COL["track"], label="트래커"), Patch(color="#9c9c9c", label="차례대로(한 스레드)"),
                   Patch(color=COL["other"], label="기타(HailoRT 내부 등)"), Patch(color=COL["idle"], label="쉼")],
          ncol=6, fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
ax.set_title("RemDet + Holdfast 자원 사용 (CPU = 4코어 전체 100%, GPU 는 안 씀)", loc="left", fontsize=12.5, color=INK, pad=28)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/pipe_resources.png", dpi=160); plt.close(fig)

# ---- 3) 트래커별: 동시 실행 E2E FPS vs 통합 HOTA
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, model in zip(axes, ("s", "tiny")):
    pts = [(k.split("__")[1], v["e2e_fps"], HOTA.get(f"{model}__{k.split('__')[1]}")) for k, v in R.items()
           if k.startswith(model + "__") and k.endswith("__pipe")]
    pts = [p for p in pts if p[2] is not None]
    for n, fps, h in pts:
        c = "#e15759" if n == "holdfast" else "#4e79a7"
        ax.scatter(fps, h, s=38, color=c, zorder=3)
        ax.annotate(n, (fps, h), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK)
    ax.axvline(30, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("E2E FPS (동시 실행, 사람·차 함께)", color=MUTED); ax.set_ylabel("통합 HOTA", color=MUTED)
    ax.set_title(f"RemDet-{model.upper()}", loc="left", fontsize=12, color=INK); ax.grid(color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/pipe_trackers.png", dpi=160); plt.close(fig)
print("그림 완료")
