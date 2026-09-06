# -*- coding: utf-8 -*-
"""Step 3 - the paired family comparison, and why its headline number is not real.

Five physical PHerc0139 segments are published twice: once as a derived 9.6 um
representation pooled from a 2.4 um scan ("aligned"), once as a real 9.362 um
scan ("native9"). Same papyrus, same ink, same annotation. If the representation
family mattered, this is where it would show.

This step reports, in this order:
  the pre-registered primary metric (it is null),
  the secondary metric that a first draft put in the headline,
  and six ways of aggregating that secondary metric, which between them show
  the headline number was an averaging artefact.

Reads data/paired_tiles.json only. No GPU, no downloads.
"""
import json, os, itertools
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
D = json.load(open(os.path.join(EXP, "data", "paired_tiles.json")))
SEG = ["w035", "w039", "w040", "w041", "w044"]
FRAMES = {"own": ("N_own", "A_own"), "frame_A": ("N_res", "A_own"), "frame_N": ("N_own", "A_res")}
head = lambda t: (print("=" * 74), print(t), print("=" * 74))


def derive(curve):
    """The four metrics, exactly as pre-registered: peak, offset 0, and the
    peak-aligned drop at one and two steps."""
    c = np.asarray(curve, float)
    k = int(np.argmax(c))
    out = {"v0": float(c[4]), "peak": float(c[k])}
    for j in (1, 2):
        v = [c[k] - c[k + s * j] for s in (-1, 1) if 0 <= k + s * j <= 8]
        out["drop%d" % j] = float(np.mean(v))
    return out


def run_table(name):
    r = D["runs"][name]
    T = r["tiles"]
    seg = np.array([t["seg"] for t in T])
    met = {f: [derive(t[f]) for t in T] for f in ("A_own", "N_own", "A_res", "N_res")}
    return r, T, seg, met


def per_segment(seg, dd, agg=np.mean):
    return np.array([agg(dd[seg == s]) for s in SEG], float)


def sign_perm_p(v):
    """Exact two-sided sign-flip permutation over 2^n. The smallest value it
    can return with n = 5 is 0.0625."""
    v = np.asarray(v, float)
    obs = abs(v.mean())
    hit = tot = 0
    for s in itertools.product([1, -1], repeat=len(v)):
        tot += 1
        if abs((v * np.array(s)).mean()) >= obs - 1e-15:
            hit += 1
    return hit / tot


def cluster_ci(seg, dd, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(SEG, size=len(SEG), replace=True)
        vals = []
        for s in pick:
            d = dd[seg == s]
            vals.append(d[rng.integers(0, len(d), len(d))].mean())
        out.append(np.mean(vals))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


head("0) WHAT THE PAIRS ARE")
r = D["runs"]["mod32_aligned"]
print(D["note"])
print("\n     window depth %d, checkpoint %s" % (r["window_depth"], r["checkpoint"]))
print("     z grid, aligned (um): %s" % r["um_aligned"])
print("     z grid, native  (um): %s" % r["um_native"])
print("\n     %-16s %-8s %-14s %s" % ("run", "phase", "tiles", "what it is"))
for k, v in D["runs"].items():
    print("     %-16s %-8s %-14d %s"
          % (k, v["phase_lock"] or "none", v["n_tiles"], v["description"]))

head("1) THE FIRST RUN WAS AN ARTEFACT - and this is how you can see it")
r, T, seg, met = run_table("unlocked")
ali_in = sum(1 for t in T if t["y0"] % 8 == 0 and t["x0"] % 8 == 0)
nat_in = sum(1 for t in T if t["ny0"] % 8 == 0 and t["nx0"] % 8 == 0)
print("With no phase lock, of %d tiles:" % len(T))
print("     aligned window on a multiple of 8:  %d  (%.1f%%)" % (ali_in, 100 * ali_in / len(T)))
print("     native  window on a multiple of 8:  %d  (%.1f%%)" % (nat_in, 100 * nat_in / len(T)))
dd = np.array([derive(t["N_own"])["peak"] - derive(t["A_own"])["peak"] for t in T])
f = per_segment(seg, dd)
print("\n     that run said: peak AUC, native9 minus aligned = %+.5f (%d/5 segments)"
      % (f.mean(), max((f > 0).sum(), (f < 0).sum())))
print("     which is step 1's phase effect wearing a family costume. Discarded.")

for RUN, LABEL in (("mod32_aligned", "PRIMARY RUN A - phase locked to mod 32, aligned anchor"),
                   ("mod32_native", "PRIMARY RUN B - phase locked to mod 32, native anchor")):
    r, T, seg, met = run_table(RUN)
    n_own, a_own = "N_own", "A_own"
    dd = {}
    for m in ("peak", "v0", "drop1", "drop2"):
        dd[m] = np.array([met[n_own][i][m] - met[a_own][i][m] for i in range(len(T))])

    head("2) %s   (n = %d tiles, %d segments)" % (LABEL, len(T), len(SEG)))

    print("PRE-REGISTERED PRIMARY METRIC: peak-aligned drop at +-2 steps")
    print("(how sharply model quality falls off as the window moves in z)\n")
    m = "drop2"
    f = per_segment(seg, dd[m])
    fm = per_segment(seg, dd[m], np.median)
    print("     per segment  %s" % "  ".join("%s %+.5f" % (s, x) for s, x in zip(SEG, f)))
    print("     mean         %+.5f      same direction in %d/5"
          % (f.mean(), max((f > 0).sum(), (f < 0).sum())))
    print("     median       %+.5f" % fm.mean())
    print("     sign-flip p  %.4f       (the floor at n=5 is 0.0625)" % sign_perm_p(f))
    print("     95%% CI       [%+.5f, %+.5f]" % cluster_ci(seg, dd[m]))
    print("     -> null. The metric the experiment was registered on found nothing.")

    print("\nSECONDARY METRIC: peak AUC. A first draft put this in the headline.")
    m = "peak"
    f = per_segment(seg, dd[m])
    print("     per segment  %s" % "  ".join("%s %+.5f" % (s, x) for s, x in zip(SEG, f)))
    print("     mean         %+.5f      same direction in %d/5   sign-flip p %.4f"
          % (f.mean(), max((f > 0).sum(), (f < 0).sum()), sign_perm_p(f)))
    print("     95%% CI       [%+.5f, %+.5f]  - contains zero" % cluster_ci(seg, dd[m]))
    lo = [np.mean(np.delete(f, i)) for i in range(5)]
    print("     leave one segment out: %s" % " ".join("%+.5f" % x for x in lo))
    biggest = int(np.argmax(np.abs(f)))
    print("     drop the largest single segment (%s) and the mean goes %+.5f -> %+.5f"
          % (SEG[biggest], f.mean(), np.mean(np.delete(f, biggest))))

    print("\nSIX WAYS TO AGGREGATE THE SAME PER-TILE NUMBERS")
    d = dd["peak"]
    npx = np.array([t["n_px_A"] for t in T], float)
    A = np.array([max(t["A_own"]) for t in T])
    N = np.array([max(t["N_own"]) for t in T])
    dice = np.array([t["dice_label"] for t in T])
    rows = [("mean over tiles (reported)", per_segment(seg, d).mean()),
            ("median over tiles", per_segment(seg, d, np.median).mean())]
    tr = lambda x: np.mean(np.sort(x)[int(0.2 * len(x)):len(x) - int(0.2 * len(x))])
    rows.append(("20% trimmed mean", per_segment(seg, d, tr).mean()))
    rows.append(("pixel weighted",
                 np.mean([np.average(d[seg == s], weights=npx[seg == s]) for s in SEG])))
    big = npx >= np.percentile(npx, 75)
    rows.append(("largest quarter of tiles",
                 np.mean([d[big & (seg == s)].mean() for s in SEG])))
    ceil = (A > 0.99) & (N > 0.99)
    rows.append(("both families at ceiling (%d%% of tiles)" % round(100 * ceil.mean()),
                 np.mean([d[ceil & (seg == s)].mean() for s in SEG])))
    clean = ceil & (dice >= 0.95)
    rows.append(("at ceiling AND label Dice >= 0.95", np.mean([d[clean & (seg == s)].mean() for s in SEG])))
    for nm, v in rows:
        print("     %-42s %+.5f" % (nm, v))

    print("\nWHERE THE MEAN COMES FROM")
    o = np.argsort(-np.abs(d))
    k5 = max(1, int(len(d) * 0.05))
    tot = d.sum()
    print("     most extreme  1%% of tiles (n=%3d) carry %5.0f%% of the total difference"
          % (max(1, len(d) // 100), 100 * d[o[:max(1, len(d) // 100)]].sum() / tot))
    print("     most extreme  5%% of tiles (n=%3d) carry %5.0f%% of it" % (k5, 100 * d[o[:k5]].sum() / tot))
    ex, rest = o[:k5], o[k5:]
    print("     those extreme tiles have %5.0f evaluated pixels; the rest have %5.0f"
          % (npx[ex].mean(), npx[rest].mean()))
    print("     their aligned peak AUC is %.3f; the rest are at %.3f" % (A[ex].mean(), A[rest].mean()))
    print("     their label Dice is %.3f; the rest are at %.3f  (no difference)"
          % (dice[ex].mean(), dice[rest].mean()))
    q = np.percentile(npx, [25, 75])
    for nm, msk in (("smallest quarter", npx <= q[0]), ("largest quarter", npx >= q[1])):
        print("     %-18s n=%3d  mean diff %+.5f   mean |diff| %.4f"
              % (nm, msk.sum(), d[msk].mean(), np.abs(d[msk]).mean()))
    print("     -> the difference lives entirely in the tiles whose AUC is noisiest.")
    print("        Weight the tiles by how much evidence they carry and it is zero.")

head("3) THE MELT LADDER")
print("Peak AUC, native9 minus aligned, segment as the unit, n = 5.\n")
lad = []
r, T, seg, met = run_table("unlocked")
d = np.array([met["N_own"][i]["peak"] - met["A_own"][i]["peak"] for i in range(len(T))])
lad.append(("no phase lock (artefact)", per_segment(seg, d).mean()))
for RUN, nm in (("mod8_aligned", "phase mod 8, aligned anchor"),
                ("mod8_native", "phase mod 8, native anchor"),
                ("mod32_aligned", "phase mod 32, aligned anchor"),
                ("mod32_native", "phase mod 32, native anchor")):
    r, T, seg, met = run_table(RUN)
    d = np.array([met["N_own"][i]["peak"] - met["A_own"][i]["peak"] for i in range(len(T))])
    lad.append((nm, per_segment(seg, d).mean()))
    if RUN.startswith("mod32"):
        npx = np.array([t["n_px_A"] for t in T], float)
        lad.append(("   same, pixel weighted",
                    np.mean([np.average(d[seg == s], weights=npx[seg == s]) for s in SEG])))
        lad.append(("   same, tile median", per_segment(seg, d, np.median).mean()))
for nm, v in lad:
    print("     %-40s %+.5f" % (nm, v))
print("\n     Note what the mod-8 rows do: the sign depends on which family's grid")
print("     drove the tile selection. That instability disappears at mod 32, which")
print("     is the training stride. Both mod-32 rows are reported; neither is")
print("     more correct than the other.")

head("4) SPECIFICATION CURVE - does the answer survive the analysis choices")
print("anchor (2) x phase lock (2) x scoring frame (3) x aggregation (2) x metric (4)\n")
spec = []
for RUN in D["runs"]:
    if RUN == "unlocked":
        continue
    r, T, seg, met = run_table(RUN)
    for fname, (fn, fa) in FRAMES.items():
        for m in ("peak", "v0", "drop1", "drop2"):
            d = np.array([met[fn][i][m] - met[fa][i][m] for i in range(len(T))])
            for agg, fun in (("mean", np.mean), ("median", np.median)):
                spec.append((RUN, fname, m, agg, per_segment(seg, d, fun).mean()))
print("     %d analysis choices in total\n" % len(spec))
print("     %-10s %-9s %-11s %-11s %s" % ("metric", "mean", "min", "max", "positive"))
for m in ("peak", "v0", "drop1", "drop2"):
    v = [x[4] for x in spec if x[2] == m]
    print("     %-10s %-+11.5f %-+11.5f %-+11.5f %d/%d"
          % (m, np.mean(v), min(v), max(v), sum(1 for x in v if x > 0), len(v)))
pk = [x[4] for x in spec if x[2] == "peak"]
print("\n     width of the peak-AUC choice space: %.5f" % (max(pk) - min(pk)))
print("     the number a first draft reported:  %.5f"
      % per_segment(np.array([t["seg"] for t in D["runs"]["mod32_aligned"]["tiles"]]),
                    np.array([derive(t["N_own"])["peak"] - derive(t["A_own"])["peak"]
                              for t in D["runs"]["mod32_aligned"]["tiles"]])).mean())
print("     -> the choice space is several times the claimed effect, and the sign")
print("        flips inside it. This is a coin toss, not a measurement.")

head("5) WHAT THIS DESIGN COULD HAVE DETECTED")
for RUN in ("mod32_aligned", "mod32_native"):
    r, T, seg, met = run_table(RUN)
    d = np.array([met["N_own"][i]["peak"] - met["A_own"][i]["peak"] for i in range(len(T))])
    f = per_segment(seg, d)
    sd = f.std(ddof=1)
    # smallest effect this design would find 80% of the time, two-sided paired t
    mde = (stats.t.ppf(0.975, 4) + stats.t.ppf(0.80, 4)) * sd / np.sqrt(5)
    print("     %-16s between-segment sd %.5f   smallest detectable effect ~%.5f"
          % (RUN, sd, mde))
    print("     %-16s observed %+.5f  =  %.0f%% of that" % ("", f.mean(), 100 * abs(f.mean()) / mde))
print("\n     So this is NOT an equivalence proof. An effect smaller than roughly")
print("     0.005-0.013 AUC cannot be detected here at all. The honest sentence is")
print("     'no difference measurable', never 'no difference exists'.")

head("6) WHAT PEAK SELECTION IS WORTH, AND WHAT AN HONEST CONTROL COSTS")
print("The peak of a nine-point curve is chosen with the labels in hand. An earlier")
print("draft controlled that by choosing the peak on a random half of a TILE's pixels")
print("and reporting it on the other half - but neighbouring pixels of the same ink")
print("stroke land in both halves, so the two halves are not independent and that")
print("removes almost nothing. A real outer fold is: choose the z offset on the OTHER")
print("segments, then report what this segment gets at that offset.\n")
print("     %-16s %-11s %-11s %-11s %s" % ("run", "oracle A", "LOSO A", "oracle N", "LOSO N"))
for RUN in ("mod32_aligned", "mod32_native"):
    r, T, seg, met = run_table(RUN)
    cA = np.array([t["A_own"] for t in T], float)
    cN = np.array([t["N_own"] for t in T], float)
    oA, oN = cA.max(1), cN.max(1)
    lA, lN = np.zeros(len(T)), np.zeros(len(T))
    for s in SEG:
        m = seg == s
        lA[m] = cA[m, int(np.argmax(cA[~m].mean(0)))]
        lN[m] = cN[m, int(np.argmax(cN[~m].mean(0)))]
    print("     %-16s %-11.5f %-11.5f %-11.5f %.5f"
          % (RUN, oA.mean(), lA.mean(), oN.mean(), lN.mean()))
    print("     %-16s cost of honest selection: aligned %.5f   native %.5f"
          % ("", oA.mean() - lA.mean(), oN.mean() - lN.mean()))
    d = lN - lA
    f = per_segment(seg, d)
    print("     %-16s family difference under LOSO: %+.5f (%d/5), tile median %+.5f"
          % ("", f.mean(), max((f > 0).sum(), (f < 0).sum()),
             per_segment(seg, d, np.median).mean()))
print("\n     Honest selection is worth about 0.012 AUC; the within-tile split removed")
print("     a tiny fraction of that, which is why it is not called a control here.")
print("     The family answer itself does not change under LOSO - selection leakage")
print("     inflates both families equally. What the bad control breaks is the claim")
print("     that leakage was controlled, not the negative result.")

head("7) SANITY: WERE THE TWO SIDES ACTUALLY MATCHED")
for RUN in ("mod32_aligned", "mod32_native"):
    r, T, seg, met = run_table(RUN)
    ia = np.array([t["n_ink_A"] / t["n_px_A"] for t in T])
    inn = np.array([t["n_ink_N"] / t["n_px_N"] for t in T])
    pa = np.array([t["n_px_A"] for t in T], float)
    pn = np.array([t["n_px_N"] for t in T], float)
    dice = np.array([t["dice_label"] for t in T])
    print("     %-16s ink fraction  A %.4f  N %.4f" % (RUN, ia.mean(), inn.mean()))
    print("     %-16s pixels        A %.0f  N %.0f" % ("", pa.mean(), pn.mean()))
    print("     %-16s label Dice    mean %.4f  median %.4f  5th pct %.4f  min %.4f"
          % ("", dice.mean(), np.median(dice), np.percentile(dice, 5), dice.min()))
    print("     %-16s tiles with Dice < 0.90: %.1f%%" % ("", 100 * (dice < 0.90).mean()))
print("\n     Ink density is matched, which matters (see step 4). The labels are NOT")
print("     the same pixels: they are two rasterisations of one annotation, agreeing")
print("     at Dice ~0.97. A ~3% label mismatch is large next to a 0.002 claim.")
