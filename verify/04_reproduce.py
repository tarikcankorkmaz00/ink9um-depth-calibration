# -*- coding: utf-8 -*-
"""Step 4 - rebuild the cross-validation statistics from scratch.

Takes the raw z-sweep of step 3 and computes, with its own code:
  - leave-one-tile-out (LOTO) offset policies,
  - the conservative comparison against a single common offset chosen on all the data,
  - a cluster bootstrap with the segment as the resampling unit,
  - leave-one-SEGMENT-out (LOSO).

Optionally compares tile by tile against an earlier run, if you point
VZ_ROOT at a directory holding prior_val_sweep.json.
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
ROOT = os.environ.get("VZ_ROOT", os.path.join(REPO, "data"))
os.makedirs(WORK, exist_ok=True)


def load(name, section):
    """Prefer a fresh file in WORK; fall back to the committed snapshot."""
    p = os.path.join(WORK, name)
    if os.path.exists(p):
        return json.load(open(p))
    return json.load(open(os.path.join(REPO, "data", "verification.json")))[section]


SCAN = load("heldout_scan.json", "heldout_scan")
R = {}

# ------------------------------------------------ 1) TILE BY TILE VS AN EARLIER RUN
prior_path = os.path.join(ROOT, "prior_val_sweep.json")
print("=" * 74)
print("1) TILE BY TILE COMPARISON AGAINST AN EARLIER RUN")
print("=" * 74)
if not os.path.exists(prior_path):
    print("  skipped: %s not present (needs the earlier run's own output)" % prior_path)
else:
    PRIOR = json.load(open(prior_path))
    o = {(e["seg"], e["y"], e["x"]): e for e in PRIOR}
    diffs, peak_diffs, matched = [], [], 0
    for e in SCAN["tiles"]:
        k = (e["seg"], e["y"], e["x"])
        if k not in o:
            continue
        matched += 1
        a = np.array(e["auc"], float)
        b = np.array([np.nan if v is None else v for v in o[k]["auc"]], float)
        assert np.allclose(e["um"], o[k]["um"]), (e["um"], o[k]["um"])
        diffs.append(np.abs(a - b))
        peak_diffs.append(e["um"][int(np.argmax(a))] - o[k]["um"][int(np.nanargmax(b))])
        if e["n_ink"] != o[k]["n_ink"] or e["n_bg"] != o[k]["n_bg"]:
            print("  PIXEL COUNTS DIFFER", k, (e["n_ink"], e["n_bg"]),
                  (o[k]["n_ink"], o[k]["n_bg"]))
    F = np.concatenate(diffs)
    print("  matched tiles: %d/%d (earlier list has %d)"
          % (matched, len(SCAN["tiles"]), len(PRIOR)))
    print("  |AUC difference|: mean=%.2e  median=%.2e  max=%.2e"
          % (F.mean(), np.median(F), F.max()))
    print("  peak position difference: all zero = %s  max|diff|=%.2f um"
          % (all(abs(t) < 1e-9 for t in peak_diffs), max(abs(t) for t in peak_diffs)))
    R["prior_run_comparison"] = dict(
        matched=matched, n_here=len(SCAN["tiles"]), n_prior=len(PRIOR),
        auc_diff_mean=float(F.mean()), auc_diff_median=float(np.median(F)),
        auc_diff_max=float(F.max()),
        peak_diff_max_um=float(max(abs(t) for t in peak_diffs)))

# ------------------------------------------------ 2) MY OWN POLICY / CV CODE
V = SCAN["tiles"]
off = np.array(V[0]["um"], float)
M = np.stack([np.array(e["auc"], float) for e in V])          # [42, 17]
seg = np.array([e["seg"] for e in V])
i0 = int(np.argmin(np.abs(off)))


def loto(M, seg, off, i0):
    """A tile's offset is chosen from the OTHER tiles of the same segment."""
    n = len(M)
    a_k0 = np.empty(n); a_seg = np.empty(n); a_glob = np.empty(n); a_orac = np.empty(n)
    for i in range(n):
        oth = np.arange(n) != i
        same = oth & (seg == seg[i])
        a_k0[i] = M[i, i0]
        a_glob[i] = M[i, int(np.argmax(M[oth].mean(0)))]
        a_seg[i] = M[i, int(np.argmax(M[same].mean(0)))]
        a_orac[i] = M[i].max()
    return a_k0, a_glob, a_seg, a_orac


a_k0, a_glob, a_seg, a_orac = loto(M, seg, off, i0)
print("\n" + "=" * 74)
print("2) LEAVE-ONE-TILE-OUT CROSS-VALIDATION (n=%d tiles, 3 segments)" % len(V))
print("=" * 74)
print("  canonical k=0                        : %.5f" % a_k0.mean())
print("  single COMMON offset (CV over tiles) : %.5f  (%+.5f)"
      % (a_glob.mean(), a_glob.mean() - a_k0.mean()))
print("  PER-SEGMENT offset (CV over tiles)   : %.5f  (%+.5f)"
      % (a_seg.mean(), a_seg.mean() - a_k0.mean()))
print("  per-tile oracle (upper bound)        : %.5f  (%+.5f)"
      % (a_orac.mean(), a_orac.mean() - a_k0.mean()))

for nm, x, y in (("per-segment vs k=0", a_seg, a_k0),
                 ("per-segment vs common", a_seg, a_glob),
                 ("common vs k=0", a_glob, a_k0)):
    d = x - y
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    t = stats.ttest_rel(x, y)
    sg = stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5)
    print("  %-24s mean=%+.5f med=%+.5f better=%d/%d  Wilcoxon p=%.4g  t p=%.3g  sign p=%.3g"
          % (nm, d.mean(), np.median(d), int((d > 0).sum()), len(d),
             w.pvalue, t.pvalue, sg.pvalue))
    R["cv_" + nm.replace(" ", "_")] = dict(
        diff=float(d.mean()), median=float(np.median(d)),
        n_better=int((d > 0).sum()), n=len(d), wilcoxon_p=float(w.pvalue),
        ttest_p=float(t.pvalue), sign_p=float(sg.pvalue))
R["cv_summary"] = dict(k0=float(a_k0.mean()), common=float(a_glob.mean()),
                       per_segment=float(a_seg.mean()), tile_oracle=float(a_orac.mean()),
                       n=len(V))

# ------------------------------------------------ 3) CONSERVATIVE TEST
print("\n" + "=" * 74)
print("3) CONSERVATIVE TEST: the common offset is picked on ALL the data (oracle),")
print("   the per-segment offset is still cross-validated")
print("=" * 74)
gi = int(np.argmax(M.mean(0)))
a_fix = M[:, gi]
d = a_seg - a_fix
w = stats.wilcoxon(d)
print("  best FIXED common offset (oracle) = %+.2f um -> AUC %.5f" % (off[gi], a_fix.mean()))
print("  per-segment (cross-validated)     -> AUC %.5f" % a_seg.mean())
print("  difference = %+.5f  better on %d/%d  Wilcoxon p=%.4g"
      % (d.mean(), int((d > 0).sum()), len(d), w.pvalue))
R["conservative_test"] = dict(fixed_offset_um=float(off[gi]), fixed_auc=float(a_fix.mean()),
    segment_cv_auc=float(a_seg.mean()), diff=float(d.mean()),
    n_better=int((d > 0).sum()), n=len(d), wilcoxon_p=float(w.pvalue))

# ------------------------------------------------ 4) CLUSTER BOOTSTRAP (SEGMENT AS UNIT)
rng = np.random.default_rng(20260906)
segs_u = sorted(set(seg))


def boot(x, y, n=20000):
    bs = np.empty(n)
    for b in range(n):
        pick = rng.choice(len(segs_u), len(segs_u), replace=True)
        idx = np.concatenate([np.where(seg == segs_u[p])[0] for p in pick])
        bs[b] = (x[idx] - y[idx]).mean()
    return bs


for nm, x, y in (("per-segment vs k=0", a_seg, a_k0),
                 ("per-segment vs common", a_seg, a_glob),
                 ("per-segment vs fixed oracle common", a_seg, a_fix)):
    bs = boot(x, y)
    print("  cluster bootstrap %-35s: mean=%+.5f 95%%[%+.5f,%+.5f] P(>0)=%.4f"
          % (nm, bs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()))
    R["bootstrap_" + nm.replace(" ", "_")] = dict(
        mean=float(bs.mean()), lo=float(np.percentile(bs, 2.5)),
        hi=float(np.percentile(bs, 97.5)), P_positive=float((bs > 0).mean()))

# ------------------------------------------------ 5) PER-SEGMENT POOLED CURVES
print("\n" + "=" * 74)
print("5) POOLED AUC-VS-DEPTH CURVE PER SEGMENT")
print("=" * 74)
R["segment_curves"] = {}
for sg in segs_u:
    c = M[seg == sg].mean(0)
    k = int(np.argmax(c))
    print("  %-16s n=%3d  AUC@0=%.5f  peak=%.5f @ %+6.1f um  gain=%+.5f"
          % (sg, int((seg == sg).sum()), c[i0], c.max(), off[k], c.max() - c[i0]))
    R["segment_curves"][sg] = dict(n=int((seg == sg).sum()), auc_k0=float(c[i0]),
        auc_peak=float(c.max()), peak_um=float(off[k]), gain=float(c.max() - c[i0]),
        curve=[float(v) for v in c])
R["offsets_um"] = [float(v) for v in off]

# ------------------------------------------------ 6) LEAVE ONE SEGMENT OUT
print("\n" + "=" * 74)
print("6) HARDER TEST: hold a whole segment out (LOSO), take its offset from the other two")
print("=" * 74)
a_loso = np.empty(len(V))
for i in range(len(V)):
    oth = seg != seg[i]
    a_loso[i] = M[i, int(np.argmax(M[oth].mean(0)))]
d = a_seg - a_loso
print("  offset-from-other-segments AUC = %.5f   within-segment CV = %.5f"
      % (a_loso.mean(), a_seg.mean()))
print("  difference = %+.5f  better on %d/%d  Wilcoxon p=%.4g"
      % (d.mean(), int((d > 0).sum()), len(d), stats.wilcoxon(d).pvalue))
R["loso"] = dict(loso_auc=float(a_loso.mean()), segment_cv_auc=float(a_seg.mean()),
    diff=float(d.mean()), n_better=int((d > 0).sum()), n=len(d),
    wilcoxon_p=float(stats.wilcoxon(d).pvalue))

json.dump(R, open(os.path.join(WORK, "analysis.json"), "w"), indent=1)
print("\nwritten ->", os.path.join(WORK, "analysis.json"))
