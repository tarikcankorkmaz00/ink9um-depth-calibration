# -*- coding: utf-8 -*-
"""Step 5 - the tests the original analysis did not run.

A) SPATIAL BLOCK CV : neighbouring tiles can leak into each other. Split the tiles into
                      spatially connected components and force the offset to come from a
                      DIFFERENT block of the same segment.
B) PERMUTATION NULL : shuffle the segment labels and repeat the same CV 10000 times.
C) DISTRIBUTION     : is the gain carried by a handful of extreme tiles?
D) TRAINING JITTER  : the model was trained with +-2 pooled slices of z-jitter. Do the
                      measured optima fall INSIDE or OUTSIDE that range?
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
os.makedirs(WORK, exist_ok=True)


def load(name, section):
    p = os.path.join(WORK, name)
    if os.path.exists(p):
        return json.load(open(p))
    return json.load(open(os.path.join(REPO, "data", "verification.json")))[section]


SCAN = load("heldout_scan.json", "heldout_scan")
V = SCAN["tiles"]
off = np.array(V[0]["um"], float)
M = np.stack([np.array(e["auc"], float) for e in V])
seg = np.array([e["seg"] for e in V])
yx = [(e["y"], e["x"]) for e in V]
i0 = int(np.argmin(np.abs(off)))
n = len(M)
R = {}


def blocks(seg, yx):
    """Tiles of one segment that touch in 8-neighbourhood form one block."""
    lab = np.full(len(yx), -1)
    nb = 0
    for s in sorted(set(seg)):
        idxs = [i for i in range(len(yx)) if seg[i] == s]
        left = set(idxs)
        while left:
            root = left.pop()
            stack = [root]
            lab[root] = nb
            while stack:
                i = stack.pop()
                for j in list(left):
                    if abs(yx[i][0] - yx[j][0]) <= 1 and abs(yx[i][1] - yx[j][1]) <= 1:
                        left.discard(j)
                        lab[j] = nb
                        stack.append(j)
            nb += 1
    return lab, nb


blk, nb = blocks(seg, yx)
print("=" * 74)
print("A) SPATIAL BLOCKS (clusters of touching tiles inside one segment)")
print("=" * 74)
R["blocks"] = []
for b in range(nb):
    m = np.where(blk == b)[0]
    ys = [yx[i][0] for i in m]
    xs = [yx[i][1] for i in m]
    print("  block %d: %-16s n=%3d  y=%d-%d  x=%d-%d"
          % (b, seg[m[0]], len(m), min(ys), max(ys), min(xs), max(xs)))
    R["blocks"].append(dict(block=int(b), seg=str(seg[m[0]]), n=int(len(m)),
                            y=[int(min(ys)), int(max(ys))], x=[int(min(xs)), int(max(xs))]))

a_k0 = M[:, i0]

# --- offset chosen inside the same segment but from ANOTHER block
a_blk = np.full(n, np.nan)
ok = np.zeros(n, bool)
for i in range(n):
    src = (seg == seg[i]) & (blk != blk[i])
    if src.sum() == 0:
        continue
    a_blk[i] = M[i, int(np.argmax(M[src].mean(0)))]
    ok[i] = True
print("\n  tiles where out-of-block CV is possible: %d/%d  (single-block segment: %s)"
      % (ok.sum(), n, sorted(set(map(str, seg[~ok]))) or "none"))

if ok.sum() >= 6:
    d = a_blk[ok] - a_k0[ok]
    w = stats.wilcoxon(d)
    print("  SAME-SEGMENT-OTHER-BLOCK offset : AUC %.5f  vs k=0 %.5f"
          % (a_blk[ok].mean(), a_k0[ok].mean()))
    print("  difference = %+.5f  median=%+.5f  better on %d/%d  Wilcoxon p=%.4g"
          % (d.mean(), np.median(d), int((d > 0).sum()), len(d), w.pvalue))
    idxs = np.where(ok)[0]
    a_loto2 = np.empty(len(idxs))
    for t, i in enumerate(idxs):
        src = (seg == seg[i]) & (np.arange(n) != i)
        a_loto2[t] = M[i, int(np.argmax(M[src].mean(0)))]
    print("  neighbour-inclusive LOTO on the SAME tiles = %.5f (%+.5f)"
          "  -> contribution of neighbourhood leakage %+.5f"
          % (a_loto2.mean(), a_loto2.mean() - a_k0[ok].mean(),
             a_loto2.mean() - a_blk[ok].mean()))
    R["block_cv"] = dict(n=int(ok.sum()), auc_out_of_block=float(a_blk[ok].mean()),
                         auc_k0=float(a_k0[ok].mean()), diff=float(d.mean()),
                         median=float(np.median(d)), n_better=int((d > 0).sum()),
                         wilcoxon_p=float(w.pvalue),
                         segment_not_testable=sorted(set(map(str, seg[~ok]))),
                         loto_same_tiles=float(a_loto2.mean()),
                         leakage_contribution=float(a_loto2.mean() - a_blk[ok].mean()))

# ---------------------------------------------------------------- B) PERMUTATION
print("\n" + "=" * 74)
print("B) PERMUTATION NULL (segment labels shuffled, group sizes preserved)")
print("=" * 74)


def loto_gain(group):
    a = np.empty(n)
    for i in range(n):
        k = (group == group[i]) & (np.arange(n) != i)
        a[i] = M[i, i0] if not k.any() else M[i, int(np.argmax(M[k].mean(0)))]
    return a.mean() - M[:, i0].mean()


real = loto_gain(seg)
rng = np.random.default_rng(20260906)
NP = 10000
null = np.array([loto_gain(rng.permutation(seg)) for _ in range(NP)])
p = float(((null >= real).sum() + 1) / (NP + 1))
print("  gain of the real segment grouping = %+.5f" % real)
print("  null gain: mean=%+.5f sd=%.5f  95th=%+.5f  max=%+.5f"
      % (null.mean(), null.std(), np.percentile(null, 95), null.max()))
print("  permutation p = %.4g  (%d/%d permutations >= real)"
      % (p, int((null >= real).sum()), NP))
R["permutation"] = dict(real_gain=float(real), null_mean=float(null.mean()),
                        null_sd=float(null.std()), null_p95=float(np.percentile(null, 95)),
                        null_max=float(null.max()), p=p, n_perm=NP)

real_b = loto_gain(blk)
nullb = np.array([loto_gain(rng.permutation(blk)) for _ in range(NP)])
pb = float(((nullb >= real_b).sum() + 1) / (NP + 1))
print("  [BLOCK grouping] real=%+.5f  null mean=%+.5f  p=%.4g" % (real_b, nullb.mean(), pb))
R["permutation_block"] = dict(real_gain=float(real_b), null_mean=float(nullb.mean()), p=pb)

# ---------------------------------------------------------------- C) DISTRIBUTION
print("\n" + "=" * 74)
print("C) DISTRIBUTION OF THE GAIN - is it carried by a few extreme tiles?")
print("=" * 74)
a_seg = np.empty(n)
for i in range(n):
    k = (seg == seg[i]) & (np.arange(n) != i)
    a_seg[i] = M[i, int(np.argmax(M[k].mean(0)))]
d = a_seg - a_k0
srt = np.sort(d)[::-1]
print("  mean=%+.5f  median=%+.5f  sd=%.5f" % (d.mean(), np.median(d), d.std(ddof=1)))
print("  five largest gains: %s" % ["%+.4f" % v for v in srt[:5]])
print("  five largest losses: %s" % ["%+.4f" % v for v in srt[-5:]])
for k in (1, 3, 5):
    cut = srt[k:]
    print("  dropping the best %d tiles -> mean gain = %+.5f  (Wilcoxon p=%.4g)"
          % (k, cut.mean(), stats.wilcoxon(cut).pvalue))
    R["trimmed_%d" % k] = dict(mean=float(cut.mean()), p=float(stats.wilcoxon(cut).pvalue))
tr = float(stats.trim_mean(d, 0.1))
print("  10%% trimmed mean = %+.5f   (median %+.5f)" % (tr, np.median(d)))
R["distribution"] = dict(mean=float(d.mean()), median=float(np.median(d)),
                         sd=float(d.std(ddof=1)), trimmed_mean_10pct=tr,
                         largest5=[float(v) for v in srt[:5]],
                         smallest5=[float(v) for v in srt[-5:]])

# ---------------------------------------------------------------- D) JITTER
print("\n" + "=" * 74)
print("D) THE TRAINING z-JITTER RANGE VERSUS THE MEASURED OPTIMA")
print("=" * 74)
JIT_UM = 2 * 9.596
print("  z-jitter applied during training: +-2 pooled slices = +-%.2f um (probability 1.0)"
      % JIT_UM)
best = {}
for sg in sorted(set(seg)):
    c = M[seg == sg].mean(0)
    k = int(np.argmax(c))
    best[str(sg)] = float(off[k])
    print("  %-16s best offset %+6.1f um -> %s the jitter range"
          % (sg, off[k], "INSIDE" if abs(off[k]) <= JIT_UM else "OUTSIDE"))
inside = np.abs(off) <= JIT_UM
a_seg_in = np.empty(n)
for i in range(n):
    k = (seg == seg[i]) & (np.arange(n) != i)
    mean_curve = M[k].mean(0).copy()
    mean_curve[~inside] = -np.inf
    a_seg_in[i] = M[i, int(np.argmax(mean_curve))]
d2 = a_seg_in - a_k0
print("  if the offset may only be picked inside +-%.1f um the gain = %+.5f (Wilcoxon p=%.4g)"
      "  [full grid: %+.5f]" % (JIT_UM, d2.mean(), stats.wilcoxon(d2).pvalue, d.mean()))
R["jitter"] = dict(jitter_um=float(JIT_UM), gain_inside_jitter=float(d2.mean()),
                   p=float(stats.wilcoxon(d2).pvalue), gain_full_grid=float(d.mean()),
                   segment_best=best)

json.dump(R, open(os.path.join(WORK, "hard_tests.json"), "w"), indent=1)
print("\nwritten -> " + os.path.join(WORK, "hard_tests.json"))
