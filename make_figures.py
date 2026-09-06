# -*- coding: utf-8 -*-
"""Redraw the three figures from the committed data. numpy + matplotlib only.

    python make_figures.py

Reads data/verification.json and data/tile_scan_531.json, writes figures/*.png.
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

V = json.load(open(os.path.join(DATA, "verification.json"), encoding="utf-8"))

# Okabe-Ito, safe under deuteranopia and protanopia
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def blocks_of(tiles):
    """Same 8-neighbour connected components the verification scripts use."""
    seg = [e["seg"] for e in tiles]
    yx = [(e["y"], e["x"]) for e in tiles]
    lab = [-1] * len(tiles)
    nb = 0
    for s in sorted(set(seg)):
        left = {i for i in range(len(tiles)) if seg[i] == s}
        while left:
            root = left.pop(); stack = [root]; lab[root] = nb
            while stack:
                i = stack.pop()
                for j in list(left):
                    if abs(yx[i][0] - yx[j][0]) <= 1 and abs(yx[i][1] - yx[j][1]) <= 1:
                        left.discard(j); lab[j] = nb; stack.append(j)
            nb += 1
    return np.array(lab), nb


# =====================================================================
# FIGURE 1 - the gain depends entirely on where the offset is allowed to come from
# =====================================================================
def figure1():
    C = V["controls"]
    pol = C["policy_summary"]
    mbase = C["policy_matched_baseline"]
    ntil = C["policy_n_tiles"]
    base = C["policy_baseline"]

    def get(sub):
        for k in pol:
            if sub in k:
                return pol[k], mbase[k], ntil[k]
        raise KeyError(sub)

    rows = [
        # label, key fragment, group, colour
        ("The tile's own data\n(per-tile oracle, not a method)", "per-tile oracle", 0, GREY),
        ("One common offset, picked on\nthe scored data (oracle)", "ORACLE", 0, GREY),
        ("Touching neighbour tiles\n(same spatial block)", "SAME BLOCK", 0, ORANGE),
        ("Same segment, neighbours\nincluded  <- the original claim", "original claim", 0, VERM),
        ("Same segment, but a spatially\nSEPARATE block", "OTHER BLOCK", 1, BLUE),
        ("Other segments\n(leave-one-segment-out)", "OTHER SEGMENTS", 1, SKY),
        ("One common offset,\ncross-validated", "leave-one-tile-out CV", 1, GREEN),
    ]
    rows = rows[::-1]
    vals = [get(r[1]) for r in rows]
    gains = [v[0] - v[1] for v in vals]
    ypos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    bars = ax.barh(ypos, gains, height=0.62,
                   color=[r[3] for r in rows], edgecolor="white", linewidth=0.8)
    for b, (auc, bl, nt), g in zip(bars, vals, gains):
        x = b.get_width()
        txt = "%+.4f   (AUC %.3f vs %.3f)" % (g, auc, bl)
        if nt != len(V["heldout_scan"]["tiles"]):
            txt += " on %d tiles" % nt
        # all labels sit to the right of zero, so nothing collides with the row names
        ax.text(max(x, 0) + 0.004, b.get_y() + b.get_height() / 2, txt,
                va="center", ha="left", fontsize=8.2)

    ax.axvline(0, color=BLACK, lw=1.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_xlabel("Change in mean ROC AUC against the canonical window, measured on the same "
                  "tiles\n(canonical = %.5f over all 42 held-out tiles; the "
                  "spatially-separate row exists only for the 31 tiles\nwhose segment has "
                  "more than one patch, so its baseline is 0.802)" % base, fontsize=8.4)
    ax.set_xlim(-0.055, 0.205)

    # divider between the two groups
    n_apart = sum(1 for r in rows if r[2] == 1)
    ax.axhline(n_apart - 0.5, color="#cccccc", lw=1, ls="--")
    ax.text(0.170, n_apart - 0.40, "offset comes from data that TOUCHES or CONTAINS the tile",
            ha="right", va="bottom", fontsize=8.2, style="italic", color="#444444")
    ax.text(0.170, n_apart - 0.60, "offset comes from SPATIALLY SEPARATE data",
            ha="right", va="top", fontsize=8.2, style="italic", color="#444444")

    ax.set_title("A per-segment depth offset only helps when it is allowed to be read off "
                 "neighbouring tiles\n"
                 "Same 42 held-out tiles, same weights, same 17-slice window - only the "
                 "source of the offset changes.", loc="left", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig1_offset_source.png")
    fig.savefig(p); plt.close(fig)
    print("written", p)


# =====================================================================
# FIGURE 2 - two patches of one segment want opposite offsets
# =====================================================================
def figure2():
    tiles = V["heldout_scan"]["tiles"]
    off = np.array(tiles[0]["um"], float)
    lab, nb = blocks_of(tiles)
    seg = np.array([e["seg"] for e in tiles])
    yy = np.array([e["y"] for e in tiles])
    xx = np.array([e["x"] for e in tiles])
    M = np.stack([np.array(e["auc"], float) for e in tiles])
    i0 = int(np.argmin(np.abs(off)))

    targets = ["pherc0139-w016", "pherc1667-w029"]
    pair_cols = [VERM, BLUE]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6),
                             gridspec_kw=dict(height_ratios=[0.78, 1.0]))
    for col, sg in enumerate(targets):
        m = seg == sg
        bl = sorted(set(lab[m]))

        # ---- top: where the tiles physically are
        ax = axes[0][col]
        for k, b in enumerate(bl):
            s = m & (lab == b)
            ax.scatter(xx[s], yy[s], s=64, marker="s", color=pair_cols[k],
                       edgecolor="white", linewidth=0.6,
                       label="patch %d  (n=%d)" % (k + 1, s.sum()))
        ax.set_xlabel("tile column (1 tile = 128 px)")
        ax.set_ylabel("tile row")
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title("%s - the held-out tiles sit in two separate patches" % sg,
                     loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.margins(0.25)

        # ---- bottom: each patch's pooled AUC against depth
        ax = axes[1][col]
        ax.axvspan(-19.192, 19.192, color="#eeeeee", zorder=0)
        peaks = []
        for k, b in enumerate(bl):
            s = m & (lab == b)
            c = M[s].mean(0)
            j = int(np.argmax(c))
            peaks.append(off[j])
            ax.plot(off, c, "-o", ms=3.6, lw=1.7, color=pair_cols[k],
                    label="patch %d, best depth %+.1f um" % (k + 1, off[j]))
            ax.plot([off[j]], [c[j]], marker="v", ms=10, color=pair_cols[k],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.axvline(0, color=BLACK, lw=1.0, ls=":")
        ax.set_xlabel("depth shift of the 17-slice window (um)")
        ax.set_ylabel("mean ROC AUC over the patch")
        ax.set_title("optima %.1f um apart, opposite signs" % abs(peaks[0] - peaks[1]),
                     loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="lower center")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.04 * (hi - lo))
        ax.text(off.min(), ax.get_ylim()[1], "grey band: +-19.2 um z-jitter seen in training",
                fontsize=7.4, color="#666666", ha="left", va="top")

    fig.suptitle("One segment does not have one best depth\n"
                 "Two spatially separate patches of the same segment put their optimum on "
                 "opposite sides of the canonical window.",
                 fontsize=10.5, x=0.008, ha="left", y=0.998, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(FIGS, "fig2_two_patches_one_segment.png")
    fig.savefig(p); plt.close(fig)
    print("written", p)


# =====================================================================
# FIGURE 3 - 433 tiles: which grouping level owns the offset, and does any policy work
# =====================================================================
def figure3():
    A = V["all_tiles_analysis"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7),
                             gridspec_kw=dict(width_ratios=[1.0, 1.35]))

    # ---- (a) variance components
    ax = axes[0]
    names = ["SCROLL\n(4 groups)", "SEGMENT\n(24 groups)", "BLOCK\n(273 groups)"]
    keys = ["icc_SCROLL", "icc_SEGMENT", "icc_BLOCK"]
    icc = [A[k]["ICC"] for k in keys]
    sdb = [A[k]["sd_between"] for k in keys]
    pv = [A[k]["p"] for k in keys]
    bars = ax.bar(np.arange(3), icc, width=0.6, color=[SKY, ORANGE, BLUE],
                  edgecolor="white", linewidth=0.8)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                "ICC %.3f\n%.1f um between groups\np = %.2f" % (icc[i], sdb[i], pv[i]),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylabel("share of peak-depth variance sitting BETWEEN groups (ICC)")
    ax.set_ylim(0, 0.245)
    ax.set_xlim(-0.72, 2.62)
    ax.set_title("(a) The best depth is a local property\n"
                 "433 tiles, 24 aligned segments, 4 scrolls",
                 loc="left", fontsize=9.5)

    # ---- (b) oracle vs honest cross-validation
    ax = axes[1]
    k0 = A["oracle"]["k0"]
    lv = ["one common\noffset", "per SCROLL", "per SEGMENT", "per BLOCK",
          "per SEGMENT,\noffset from a\nSEPARATE block"]
    ok = [A["oracle"]["common"], A["oracle"]["scroll"],
          A["oracle"]["segment"], A["oracle"]["block"], np.nan]
    cv = [A["cv"]["common"], A["cv"]["scroll"], A["cv"]["segment"], A["cv"]["block"],
          A["out_of_block_cv"]["auc"]]
    xs = np.arange(5)
    w = 0.38
    b1 = ax.bar(xs - w / 2, [v - k0 for v in ok], width=w, color=GREY,
                edgecolor="white", linewidth=0.8,
                label="offset picked on the group's OWN data (oracle)")
    b2 = ax.bar(xs + w / 2, [v - k0 for v in cv], width=w, color=VERM,
                edgecolor="white", linewidth=0.8,
                label="offset cross-validated (honest)")
    for i in range(len(xs)):
        pair = [(b1[i], ok[i] - k0), (b2[i], cv[i] - k0)]
        if not np.isnan(pair[0][1]) and abs(pair[0][1] - pair[1][1]) < 1e-9:
            pair = [(None, pair[0][1])]          # identical values: label the pair once
        for b, h in pair:
            if np.isnan(h):
                continue
            cx = xs[i] if b is None else b.get_x() + b.get_width() / 2
            ax.text(cx, h + (0.0005 if h >= 0 else -0.0005), "%+.4f" % h,
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=7.4)
    ax.axhline(0, color=BLACK, lw=1.4)
    ax.text(4.02, 0.0012, "no oracle\nequivalent", ha="center", va="bottom",
            fontsize=7.2, color="#777777", style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels(lv, fontsize=8.2)
    ax.set_ylabel("change in mean ROC AUC against the canonical window")
    ax.set_ylim(-0.0105, 0.0245)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(b) Every honest per-group offset is WORSE than doing nothing\n"
                 "canonical AUC = %.5f; grey looks good only because each\n"
                 "group picks its offset on its own data" % k0,
                 loc="left", fontsize=9.5)

    fig.tight_layout()
    p = os.path.join(FIGS, "fig3_scale_and_honest_cv.png")
    fig.savefig(p); plt.close(fig)
    print("written", p)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
