# -*- coding: utf-8 -*-
"""Step 4 - the confounders, the noise floor, and the corpus/deployment table.

Three separate things live here, and they are the reason the family result is
reported as a negative rather than as a small positive:

  A. how big a "difference" this measurement produces when there is no
     difference at all - the noise floor and two placebos,
  B. how far above the model's held-out performance this whole experiment sits,
  C. what the training corpus actually contains, computed from the sampler's
     own hierarchy rather than assumed.

Reads ../data/tile_scan_531.json (experiment 1's committed sweep, re-gridded
here), experiment2/data/context.json and experiment2/data/paired_tiles.json.
No GPU, no downloads.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
SCAN = json.load(open(os.path.join(REPO, "data", "tile_scan_531.json")))
CTX = json.load(open(os.path.join(EXP, "data", "context.json")))
TIL = json.load(open(os.path.join(EXP, "data", "paired_tiles.json")))
head = lambda t: (print("=" * 74), print(t), print("=" * 74))

# --------------------------------------------------------------- common z grid
# aligned tiles were swept at 17 points of 4.798 um, native9 at 11 points of
# 9.362 um. Take every second aligned point so both families sit on a ~9.5 um
# grid of nine points, k = -4 .. +4, all in phase.
rows = []
for e in SCAN:
    a = np.asarray(e["auc"], float)
    if e["family"] == "aligned":
        if len(a) != 17:
            continue
        v = a[[8 + 2 * k for k in range(-4, 5)]]
    else:
        if len(a) != 11:
            continue
        v = a[[5 + k for k in range(-4, 5)]]
    k = int(np.argmax(v))
    n_lab = e["n_ink"] + e["n_bg"]
    rows.append(dict(seg=e["seg"], scroll=e["scroll"], family=e["family"],
                     n_lab=n_lab, ink=e["n_ink"] / n_lab,
                     auc0=float(v[4]), peak=float(v[k]),
                     drop2=float(v[k] - 0.5 * (v[k - 2] + v[k + 2])) if 2 <= k <= 6 else None))

head("0) THE CORPUS SWEEP, RE-GRIDDED")
print("Rebuilt from experiment 1's committed data/tile_scan_531.json.")
print("     tiles on the common 9-point in-phase grid: %d of %d" % (len(rows), len(SCAN)))
for fam in ("aligned", "native9"):
    r = [x for x in rows if x["family"] == fam]
    print("     %-9s %3d tiles, %2d segments, %d scroll(s)"
          % (fam, len(r), len(set(x["seg"] for x in r)), len(set(x["scroll"] for x in r))))
print("\n     Note the asymmetry that makes the unpaired comparison useless: every")
print("     native9 tile is from one scroll. 'family' and 'scroll' are confounded")
print("     at corpus level, which is why the rest of this repo uses paired segments.")

PAIRS = [(p["aligned_label_name"], p["native9_label_name"]) for p in CTX["pairing"]]
byseg = {}
for r in rows:
    byseg.setdefault(r["seg"], []).append(r)
MET = ("peak", "auc0", "drop2")
val = lambda rr, m: np.array([x[m] for x in rr if x[m] is not None], float)

head("A1) NOISE FLOOR - how much a segment mean moves just from which tiles were drawn")
print("Analytic, using the real tile counts of the five paired segments.\n")
print("     %-8s %-14s %-16s %-16s" % ("metric", "sampling sd", "+-1.96 sd band", "raw family gap"))
raw = {}
for m in MET:
    se2, gaps = [], []
    for a, n in PAIRS:
        va, vn = val(byseg[a], m), val(byseg[n], m)
        se2.append(va.var(ddof=1) / len(va) + vn.var(ddof=1) / len(vn))
        gaps.append(vn.mean() - va.mean())
    sd = np.sqrt(np.sum(se2)) / len(PAIRS)
    raw[m] = np.mean(gaps)
    inside = "inside" if abs(raw[m]) <= 1.96 * sd else "%.2f sd out" % (abs(raw[m]) / sd)
    print("     %-8s %-14.5f [%+.5f,%+.5f] %+.5f   %s"
          % (m, sd, -1.96 * sd, 1.96 * sd, raw[m], inside))
print("\n     The unpaired family gaps sit inside the band that tile sampling alone")
print("     produces. That is before any confounder is considered.")

head("A2) PLACEBO 1 - split one family's own tiles at random, twice")
rng = np.random.default_rng(11)
print("4000 random half/half splits, aligned side of the five pairs, no family change.\n")
print("     %-8s %-12s %-24s" % ("metric", "sd", "95% of the 'differences'"))
for m in ("peak", "auc0"):
    out = []
    for _ in range(4000):
        d = []
        for a, _n in PAIRS:
            r = byseg[a]
            idx = rng.permutation(len(r))
            h = len(r) // 2
            d.append(val([r[i] for i in idx[:h]], m).mean() - val([r[i] for i in idx[h:2 * h]], m).mean())
        out.append(np.mean(d))
    out = np.array(out)
    print("     %-8s %-12.5f [%+.5f, %+.5f]"
          % (m, out.std(), np.percentile(out, 2.5), np.percentile(out, 97.5)))
print("\n     (a half/half split gives each side half the tiles, so this sd runs about")
print("      1.5x larger than the real comparison's. A1 is the accurate figure.)")

head("A3) PLACEBO 2 - split one family's own tiles by labelled-pixel count")
print("Still no family change. Upper half minus lower half, within a family.\n")
print("     %-8s %-22s %-14s %s" % ("metric", "family", "difference", "same direction"))
plac = {}
for m in ("peak", "auc0"):
    for nm, side in (("aligned", 0), ("native9", 1)):
        d = []
        for pr in PAIRS:
            r = sorted(byseg[pr[side]], key=lambda z: z["n_lab"])
            h = len(r) // 2
            d.append(val(r[-h:], m).mean() - val(r[:h], m).mean())
        plac[(m, nm)] = np.mean(d)
        print("     %-8s %-22s %-+14.5f %d/5"
              % (m, nm, np.mean(d), max(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))))
print("\n     Density imbalance on its own produces %.1fx the raw peak-AUC family gap"
      % (plac[("peak", "aligned")] / raw["peak"]))
print("     and %.1fx the raw offset-0 gap. The two families' tiles were imbalanced"
      % (plac[("auc0", "aligned")] / raw["auc0"]))
ia = np.mean([x["ink"] for a, _ in PAIRS for x in byseg[a]])
inn = np.mean([x["ink"] for _, n in PAIRS for x in byseg[n]])
la = np.mean([x["n_lab"] for a, _ in PAIRS for x in byseg[a]])
ln = np.mean([x["n_lab"] for _, n in PAIRS for x in byseg[n]])
print("     in exactly that variable: labelled pixels %.0f vs %.0f, ink fraction %.3f vs %.3f."
      % (la, ln, ia, inn))
print("\n     This is why the paired run in step 3 matches tiles on density first.")

head("B) HOW HIGH ABOVE THE MODEL'S REAL PERFORMANCE THIS SITS")
ho = CTX["held_out_reference"]
T = TIL["runs"]["mod32_aligned"]["tiles"]
A = np.array([max(t["A_own"]) for t in T])
N = np.array([max(t["N_own"]) for t in T])
print("     same checkpoint, held-out region (experiment 1):        %.5f" % ho["mean_roc_auc"])
print("     this experiment, aligned side, mean peak AUC:           %.5f" % A.mean())
print("     memorisation pedestal:                                  %.3f" % (A.mean() - ho["mean_roc_auc"]))
print("     tiles above 0.99: %.0f%% (aligned) / %.0f%%  (native9)"
      % (100 * (A > 0.99).mean(), 100 * (N > 0.99).mean()))
print("     tiles above 0.999: %.0f%% / %.0f%%" % (100 * (A > 0.999).mean(), 100 * (N > 0.999).mean()))
print("\n     Every measurement in this experiment is inside the training region.")
print("     It is looking for a ripple of 0.002 on top of a plateau of 0.20.")
print("     Nothing here supports a claim about generalisation.")

head("B2) COVERAGE - how much of each segment was actually looked at")
CH = CTX["supervision_chunk_counts"]
print("     %-8s %-16s %-20s %s" % ("segment", "pixels scored", "supervision px (max)", "coverage"))
for s in ["w035", "w039", "w040", "w041", "w044"]:
    px = sum(t["n_px_A"] for t in T if t["seg"] == s)
    up = CH[s] * 128 * 128
    print("     %-8s %-16d %-20d <= %.1f%%" % (s, px, up, 100.0 * px / up))
print("\n     %s" % CH["note"])
print("     Tiles were also not drawn at random: they are the positions that survive")
print("     the mod-32 snap and pass an ink/background filter in BOTH families.")

head("C) WHAT THE TRAINING CORPUS ACTUALLY CONTAINS")
tr = CTX["training"]
COLS = CTX["corpus"]["columns"]
rowsC = [dict(zip(COLS, r)) for r in CTX["corpus"]["rows"]]
phys_by_scroll, reps_by_phys, role_of = {}, {}, {}
for r in rowsC:
    phys_by_scroll.setdefault(r["scroll"], set()).add(r["sampler_physical_key"])
    reps_by_phys.setdefault((r["scroll"], r["sampler_physical_key"]), set()).add(r["representation"])
    role_of[r["representation"]] = r["role"]
share = {}
for scroll, q in tr["scroll_quota"].items():
    P = len(phys_by_scroll[scroll])
    for phys in phys_by_scroll[scroll]:
        R = len(reps_by_phys[(scroll, phys)])
        for rep in reps_by_phys[(scroll, phys)]:
            share[rep] = q / P / R
B = tr["batch_size"]
nat = sum(v for k, v in share.items() if role_of[k] == "native_9p362_level0")
pub = sum(v for k, v in share.items() if role_of[k] == "public_2p4_level2_zmean4")
print("Sampler: scroll quota -> uniform over physical segments -> uniform over")
print("representations. Recomputed here from the hierarchy in data/context.json.\n")
print("     batch size %d, quotas %s" % (B, tr["scroll_quota"]))
print("     total share checks out: %.4f\n" % sum(share.values()))
print("     %-46s %-10s %s" % ("imaging condition", "per batch", "share"))
print("     %-46s %-10.2f %.2f%%" % ("short propagation (0.22 m / 78 keV), pooled", pub, 100 * pub / B))
print("     %-46s %-10.2f %.2f%%" % ("long propagation (1.2 m / 113 keV), native", nat, 100 * nat / B))
print("     %-46s %-10.2f %.2f%%" % ("anything at 8.640 um", 0.0, 0.0))
print("\n     per-representation share ranges %.2f%% to %.2f%%, a factor of %.2f"
      % (100 * min(share.values()) / B, 100 * max(share.values()) / B,
         max(share.values()) / min(share.values())))
print("\n     The 13 scrolls eligible for the prize are all long-propagation; nine are")
print("     at 9.362 um and four at 8.640 um. So the model spent %.0f%% of its training"
      % (100 * pub / B))
print("     on a condition it will not meet at deployment, %.0f%% on the one it will," % (100 * nat / B))
print("     and nothing at all on the voxel size of 4 of the 13 target scrolls.")
print("     That is an exposure statement, not a loss statement. It is not converted")
print("     to AUC anywhere in this repo, because the exposure/AUC relation measured")
print("     with the scroll held fixed came out null.")

head("C2) A NAMING INCONSISTENCY WORTH FIXING IN THE CONFIG")
nb = CTX["sampler_naming_bug"]
print(nb["what"] + "\n")
print(nb["effect"] + "\n")
w044 = sum(v for k, v in share.items() if k in ("pherc0139-w028", "w044"))
w041 = sum(v for k, v in share.items() if k in ("pherc0139-w041", "w041"))
print("     physical w044 (as two 'segments'): %.2f per batch = %.2f%%" % (w044, 100 * w044 / B))
print("     physical w041 (correctly one):     %.2f per batch = %.2f%%" % (w041, 100 * w041 / B))
print("     so one piece of papyrus is seen %.1fx as often as its neighbours." % (w044 / w041))
for p in CTX["pairing"]:
    if p["physical_segment"] == "w044":
        print("\n     Both volumes live in the same S3 segment folder:")
        print("       %s" % p["shared_segment_folder"])
        print("       aligned label name: %s" % p["aligned_label_name"])
        print("       native label name:  %s" % p["native9_label_name"])

head("C3) ONE THING THIS EXPERIMENT CANNOT VERIFY ABOUT ITSELF")
vc = CTX["validation_cases"]
print("     reserved_validation_cases: %s" % ", ".join(vc["reserved_validation_cases"]))
print("     online_validation_cases:   %s" % ", ".join(vc["online_validation_cases"]))
print("     masks actually published:  %s" % ", ".join(vc["published_masks"]))
print("\n     %s" % vc["warning"])
