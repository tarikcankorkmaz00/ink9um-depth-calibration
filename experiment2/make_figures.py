# -*- coding: utf-8 -*-
"""Redraw experiment 2's two figures from the committed data.

    python experiment2/make_figures.py

Reads experiment2/data/*.json, writes experiment2/figures/*.png.
numpy + matplotlib only, no GPU, no downloads.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

PH = json.load(open(os.path.join(DATA, "phase.json")))
BL = json.load(open(os.path.join(DATA, "blended.json")))
TI = json.load(open(os.path.join(DATA, "paired_tiles.json")))

# Okabe-Ito, same palette as experiment 1
BLACK, ORANGE, SKY = "#000000", "#E69F00", "#56B4E9"
GREEN, BLUE, VERM = "#009E73", "#0072B2", "#D55E00"
PURPLE, GREY = "#CC79A7", "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "white", "figure.facecolor": "white",
})
SEG = ["w035", "w039", "w040", "w041", "w044"]


def derive(curve):
    c = np.asarray(curve, float)
    k = int(np.argmax(c))
    out = {"v0": float(c[4]), "peak": float(c[k])}
    for j in (1, 2):
        v = [c[k] - c[k + s * j] for s in (-1, 1) if 0 <= k + s * j <= 8]
        out["drop%d" % j] = float(np.mean(v))
    return out


def seg_mean(run, metric="peak", frame=("N_own", "A_own"), agg=np.mean):
    T = TI["runs"][run]["tiles"]
    seg = np.array([t["seg"] for t in T])
    d = np.array([derive(t[frame[0]])[metric] - derive(t[frame[1]])[metric] for t in T])
    return float(np.mean([agg(d[seg == s]) for s in SEG])), d, seg, T


# ============================================================ FIGURE 1: PHASE
fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.5))

# (a) one-axis shift scan
a = ax[0]
for fam, col in (("native9", BLUE), ("aligned", VERM)):
    b = PH["shift_1d"][fam]
    xs = sorted(int(k) for k in b["shift_px"])
    ys = [b["shift_px"][str(x)]["mean_delta"] for x in xs]
    a.plot(xs, ys, "-o", color=col, ms=3.5, lw=1.2, label="%s (n=%d)" % (fam, b["n_tiles"]))
for x in (0, 8, 16, 24, 32):
    a.axvline(x, color=GREY, lw=0.6, ls=":", zorder=0)
a.axhline(0, color=BLACK, lw=0.8)
a.set_xlabel("128x128 window origin moved by (px)\ndotted lines are multiples of 8")
a.set_ylabel("change in mean ROC AUC")
a.set_title("(a) same pixels, same labels,\nonly the window moves")
a.legend(fontsize=8, loc="lower left", framealpha=1.0, edgecolor="white")

# (b) blended inference, real sliding window
b = ax[1]
A = {k: {int(s): v for s, v in r.items()} for k, r in BL["auc_by_stride"].items()}
strides = sorted(set().union(*[set(r) for r in A.values()]))
for k in sorted(A):
    ys = [A[k][s] - A[k][64] for s in strides]
    b.plot(strides, ys, "-", color=GREY, lw=0.8, alpha=0.8, zorder=1)
on = [s for s in strides if s % 8 == 0]
off = [s for s in strides if s % 8]
for ss, col, mk, lab in ((on, GREEN, "o", "stride is a multiple of 8"),
                         (off, VERM, "s", "stride is not")):
    b.scatter([s for s in ss for _ in A], [A[k][s] - A[k][64] for s in ss for k in sorted(A)],
              color=col, s=16, marker=mk, zorder=3, label=lab)
b.axhline(0, color=BLACK, lw=0.8)
b.set_xlabel("sliding-window stride (px)")
b.set_ylabel("change in ROC AUC vs stride 64")
b.set_title("(b) real blended inference,\n5 measurements")
b.legend(frameon=False, fontsize=8, loc="lower left")
d90 = np.mean([A[k][90] - A[k][64] for k in A])
b.annotate("--overlap 0.3\nis stride 90:\n%+.3f AUC" % d90, xy=(90, d90),
           xytext=(41, d90 - 0.006), fontsize=7.5, color=VERM,
           arrowprops=dict(arrowstyle="->", color=VERM, lw=0.8,
                           connectionstyle="arc3,rad=-0.2"))

# (c) which --overlap values land off the grid
c = ax[2]
tab = PH["overlap_to_stride"]["table"]
xs = [r["overlap"] for r in tab]
ys = [r["stride"] for r in tab]
cols = [GREEN if r["stride_mod8"] == 0 else VERM for r in tab]
c.bar(range(len(xs)), ys, color=cols, width=0.72)
c.set_xticks(range(len(xs)))
c.set_xticklabels([("%g" % x) for x in xs], fontsize=7.5, rotation=90)
c.set_xlabel("--overlap")
c.set_ylabel("resulting stride")
c.set_title("(c) what infer.py's --overlap\nturns into")
for i, r in enumerate(tab):
    if r["overlap"] == 0.5:
        c.annotate("default", xy=(i, r["stride"]), xytext=(i - 1.4, r["stride"] + 26),
                   fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
c.legend(handles=[Line2D([], [], color=GREEN, lw=6, label="on the grid"),
                  Line2D([], [], color=VERM, lw=6, label="off the grid")],
         frameon=False, fontsize=8, loc="upper right")

fig.suptitle("The published ink_9um checkpoint is sensitive to where the patch starts",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGS, "fig1_patch_phase.png"), bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig1_patch_phase.png")

# ================================================ FIGURE 2: THE FAMILY RESULT
fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.7))

# (a) the melt ladder
a = ax[0]
rows = [("no phase lock (artefact)", seg_mean("unlocked")[0], GREY),
        ("phase mod 8, aligned anchor", seg_mean("mod8_aligned")[0], SKY),
        ("phase mod 8, native anchor", seg_mean("mod8_native")[0], SKY),
        ("phase mod 32, aligned anchor", seg_mean("mod32_aligned")[0], BLUE),
        ("phase mod 32, native anchor", seg_mean("mod32_native")[0], BLUE)]
m32, d32, seg32, T32 = seg_mean("mod32_aligned")
npx32 = np.array([t["n_px_A"] for t in T32], float)
rows.append(("mod 32 aligned, pixel weighted",
             float(np.mean([np.average(d32[seg32 == s], weights=npx32[seg32 == s]) for s in SEG])),
             GREEN))
rows.append(("mod 32 aligned, tile median",
             seg_mean("mod32_aligned", agg=np.median)[0], GREEN))
y = np.arange(len(rows))[::-1]
a.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
a.set_yticks(y)
a.set_yticklabels([r[0] for r in rows], fontsize=8)
a.axvline(0, color=BLACK, lw=0.8)
a.set_xlabel("peak ROC AUC, native9 minus aligned")
a.set_title("(a) the difference melts as\ncontrols go in")
for yy, r in zip(y, rows):
    a.text(r[1] + (0.0012 if r[1] > 0 else -0.0012), yy, "%+.4f" % r[1],
           va="center", ha="left" if r[1] > 0 else "right", fontsize=7.5)
a.set_xlim(-0.037, 0.016)

# (b) the difference lives in the tiles with least evidence
b = ax[1]
b.scatter(npx32, d32, s=7, color=BLUE, alpha=0.35, edgecolors="none")
b.axhline(0, color=BLACK, lw=0.8)
q = np.percentile(npx32, [25, 75])
for lo, hi, lab in ((npx32.min(), q[0], "smallest quarter"), (q[1], npx32.max(), "largest quarter")):
    m = (npx32 >= lo) & (npx32 <= hi)
    b.plot([lo, hi], [d32[m].mean()] * 2, color=VERM, lw=2.2, zorder=4)
    b.text((lo + hi) / 2, d32[m].mean(), "\n%s\nmean %+.4f" % (lab, d32[m].mean()),
           ha="center", va="top", fontsize=7.5, color=VERM)
b.set_xlabel("pixels actually scored in the tile")
b.set_ylabel("peak AUC, native9 minus aligned")
b.set_title("(b) where the mean comes from:\nthe noisiest tiles")

# (c) specification curve
c = ax[2]
spec = []
FR = {"own": ("N_own", "A_own"), "frame_A": ("N_res", "A_own"), "frame_N": ("N_own", "A_res")}
for run in TI["runs"]:
    if run == "unlocked":
        continue
    for fr in FR.values():
        for met in ("peak", "v0", "drop1", "drop2"):
            for agg in (np.mean, np.median):
                spec.append(seg_mean(run, met, fr, agg)[0])
spec = np.sort(np.array(spec))
c.plot(np.arange(len(spec)), spec, "-", color=BLUE, lw=1.4)
c.fill_between(np.arange(len(spec)), 0, spec, color=BLUE, alpha=0.13)
c.axhline(0, color=BLACK, lw=0.8)
rep = seg_mean("mod32_aligned")[0]
c.axhline(rep, color=VERM, lw=1.0, ls="--")
c.text(len(spec) * 0.02, rep, " the number a first draft reported (%+.4f)" % rep,
       color=VERM, fontsize=7.5, va="bottom")
c.set_xlabel("%d defensible analysis choices, sorted" % len(spec))
c.set_ylabel("native9 minus aligned")
c.set_title("(c) the answer moves further\nthan the effect")
c.text(0.97, 0.05, "%d of %d positive" % (int((spec > 0).sum()), len(spec)),
       transform=c.transAxes, ha="right", fontsize=8, color=GREY)

fig.suptitle("Same papyrus, two imaging conditions: no measurable difference in model output",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGS, "fig2_family_difference.png"), bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig2_family_difference.png")
