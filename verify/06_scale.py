# -*- coding: utf-8 -*-
"""Step 6 - at what spatial scale does the optimal z-offset actually live?

Step 5 turned the gain negative once the offset had to come from a spatially separate
block. This step decomposes the scale: is the optimum shared by the segment, by the
block, or only by immediate neighbours? It also repeats the key tests with AUC-PR to
check whether the sign of the result depends on the metric.
"""
import json, os, itertools
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
AP = np.stack([np.array(e["ap"], float) for e in V])
seg = np.array([e["seg"] for e in V])
yx = np.array([(e["y"], e["x"]) for e in V])
i0 = int(np.argmin(np.abs(off)))
n = len(M)
R = {}


def blocks(seg, yx):
    lab = np.full(len(yx), -1)
    nb = 0
    for s in sorted(set(seg)):
        left = {i for i in range(len(yx)) if seg[i] == s}
        while left:
            root = left.pop(); stack = [root]; lab[root] = nb
            while stack:
                i = stack.pop()
                for j in list(left):
                    if abs(yx[i][0] - yx[j][0]) <= 1 and abs(yx[i][1] - yx[j][1]) <= 1:
                        left.discard(j); lab[j] = nb; stack.append(j)
            nb += 1
    return lab, nb


blk, nb = blocks(seg, yx)
a_k0 = M[:, i0]

# ---------------------------------------------------------------- 1) BLOCK CURVES
print("=" * 76)
print("1) POOLED AUC CURVE PER BLOCK - do two blocks of one segment want the same offset?")
print("=" * 76)
R["block_curves"] = {}
for b in range(nb):
    m = blk == b
    c = M[m].mean(0)
    k = int(np.argmax(c))
    print("  block %d %-16s n=%2d  y%d-%d x%d-%d   AUC@0=%.5f  peak=%.5f @ %+6.1f um  gain=%+.5f"
          % (b, seg[m][0], m.sum(), yx[m][:, 0].min(), yx[m][:, 0].max(),
             yx[m][:, 1].min(), yx[m][:, 1].max(), c[i0], c.max(), off[k], c.max() - c[i0]))
    R["block_curves"][int(b)] = dict(seg=str(seg[m][0]), n=int(m.sum()), auc_k0=float(c[i0]),
                                     peak_um=float(off[k]), gain=float(c.max() - c[i0]),
                                     curve=[float(v) for v in c])
print("\n  --> best offsets of the two blocks of the same segment:")
for sg in sorted(set(seg)):
    bs = sorted(set(blk[seg == sg]))
    ts = [float(off[int(np.argmax(M[blk == b].mean(0)))]) for b in bs]
    if len(bs) > 1:
        print("      %-16s blocks %s -> offsets %s   DIFFERENCE = %+.1f um"
              % (sg, [int(b) for b in bs], ["%+.1f" % t for t in ts], ts[0] - ts[1]))
    else:
        print("      %-16s single block -> %+.1f um (nothing to compare)" % (sg, ts[0]))
R["block_peaks"] = {str(sg): [float(off[int(np.argmax(M[blk == b].mean(0)))])
                              for b in sorted(set(blk[seg == sg]))]
                    for sg in sorted(set(seg))}

# ---------------------------------------------------------------- 2) OUT-OF-BLOCK CV PER SEGMENT
print("\n" + "=" * 76)
print("2) OUT-OF-BLOCK CV PER SEGMENT (offset from ANOTHER block of the SAME segment)")
print("=" * 76)
R["block_cv_per_segment"] = {}
for sg in sorted(set(seg)):
    idx = np.where(seg == sg)[0]
    if len(set(blk[idx])) < 2:
        print("  %-16s single block -> not testable" % sg)
        R["block_cv_per_segment"][str(sg)] = "single_block"
        continue
    ab, ak = [], []
    for i in idx:
        k = (seg == seg[i]) & (blk != blk[i])
        ab.append(M[i, int(np.argmax(M[k].mean(0)))])
        ak.append(M[i, i0])
    ab, ak = np.array(ab), np.array(ak)
    d = ab - ak
    print("  %-16s n=%2d  out-of-block AUC=%.5f  k=0 AUC=%.5f  diff=%+.5f  better %d/%d  p=%.3g"
          % (sg, len(d), ab.mean(), ak.mean(), d.mean(), int((d > 0).sum()), len(d),
             stats.wilcoxon(d).pvalue))
    R["block_cv_per_segment"][str(sg)] = dict(n=int(len(d)), out_of_block=float(ab.mean()),
                                              k0=float(ak.mean()), diff=float(d.mean()),
                                              n_better=int((d > 0).sum()),
                                              p=float(stats.wilcoxon(d).pvalue))

# ---------------------------------------------------------------- 3) WITHIN-BLOCK LOTO
print("\n" + "=" * 76)
print("3) WITHIN-BLOCK LOTO (offset from the OTHER tiles of the SAME block)")
print("=" * 76)
a_bi = np.empty(n)
for i in range(n):
    k = (blk == blk[i]) & (np.arange(n) != i)
    a_bi[i] = M[i, i0] if not k.any() else M[i, int(np.argmax(M[k].mean(0)))]
d = a_bi - a_k0
print("  within-block LOTO AUC = %.5f  vs k=0 %.5f   diff=%+.5f  median=%+.5f  better %d/%d  p=%.4g"
      % (a_bi.mean(), a_k0.mean(), d.mean(), np.median(d), int((d > 0).sum()), n,
         stats.wilcoxon(d).pvalue))
R["within_block_loto"] = dict(auc=float(a_bi.mean()), k0=float(a_k0.mean()),
                              diff=float(d.mean()), median=float(np.median(d)),
                              n_better=int((d > 0).sum()),
                              p=float(stats.wilcoxon(d).pvalue))

# ---------------------------------------------------------------- 4) VARIANCE COMPONENTS
print("\n" + "=" * 76)
print("4) VARIANCE COMPONENTS OF THE PER-TILE PEAK OFFSET (segment / block / tile)")
print("=" * 76)


def parabolic_peak(c):
    k = int(np.argmax(c))
    if k == 0 or k == len(c) - 1:
        return float(off[k]), True
    den = c[k - 1] - 2 * c[k] + c[k + 1]
    dd = 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (c[k - 1] - c[k + 1]) / den, -1, 1))
    return float(off[k] + dd * (off[1] - off[0])), False


peak = np.array([parabolic_peak(M[i])[0] for i in range(n)])
edge = np.array([parabolic_peak(M[i])[1] for i in range(n)])
print("  tiles whose peak sits on the grid edge: %d/%d (unreliable estimate)" % (edge.sum(), n))
print("  all tiles: mean=%+.2f um  sd=%.2f um" % (peak.mean(), peak.std(ddof=1)))
for sg in sorted(set(seg)):
    t = peak[seg == sg]
    print("    %-16s n=%2d  mean=%+7.2f  sd=%6.2f  min=%+7.2f  max=%+7.2f"
          % (sg, len(t), t.mean(), t.std(ddof=1), t.min(), t.max()))


def variance_components(group, x):
    ks = sorted(set(group))
    N = len(x); a = len(ks)
    grand = x.mean()
    SSB = sum((group == k).sum() * (x[group == k].mean() - grand) ** 2 for k in ks)
    SSW = sum(((x[group == k] - x[group == k].mean()) ** 2).sum() for k in ks)
    dfB, dfW = a - 1, N - a
    MSB, MSW = SSB / dfB, SSW / dfW
    ni = [int((group == k).sum()) for k in ks]
    n0 = (N - sum(v * v for v in ni) / N) / (a - 1)
    vb = max(0.0, (MSB - MSW) / n0)
    F = MSB / MSW
    return dict(a=a, N=N, var_between=vb, var_within=MSW, sd_between=float(np.sqrt(vb)),
                sd_within=float(np.sqrt(MSW)), ICC=vb / (vb + MSW) if vb + MSW > 0 else 0.0,
                F=float(F), p=float(1 - stats.f.cdf(F, dfB, dfW)))


vs = variance_components(seg, peak)
vb2 = variance_components(blk, peak)
print("\n  SEGMENT level : sd_between=%.2f um  sd_within=%.2f um  ICC=%.4f  F=%.2f p=%.4f"
      % (vs["sd_between"], vs["sd_within"], vs["ICC"], vs["F"], vs["p"]))
print("  BLOCK   level : sd_between=%.2f um  sd_within=%.2f um  ICC=%.4f  F=%.2f p=%.4f"
      % (vb2["sd_between"], vb2["sd_within"], vb2["ICC"], vb2["F"], vb2["p"]))
R["peak_variance"] = dict(segment=vs, block=vb2, on_edge=int(edge.sum()),
                          peak_sd_overall=float(peak.std(ddof=1)))

# ---------------------------------------------------------------- 5) DISTANCE VS SIMILARITY
print("\n" + "=" * 76)
print("5) SPATIAL DISTANCE VS CURVE SIMILARITY (tile pairs inside one segment)")
print("=" * 76)
Mc = M - M.mean(1, keepdims=True)
Mc /= np.linalg.norm(Mc, axis=1, keepdims=True)
dist, corr, peakdiff, same_block = [], [], [], []
for i, j in itertools.combinations(range(n), 2):
    if seg[i] != seg[j]:
        continue
    dist.append(float(np.hypot(*(yx[i] - yx[j])) * 128))     # pixels (1 tile = 128 px)
    corr.append(float(Mc[i] @ Mc[j]))
    peakdiff.append(abs(peak[i] - peak[j]))
    same_block.append(blk[i] == blk[j])
dist, corr, peakdiff = map(np.array, (dist, corr, peakdiff))
same_block = np.array(same_block)
r_s = stats.spearmanr(dist, corr)
print("  pairs=%d   Spearman(distance, curve correlation) rho=%+.3f  p=%.3g"
      % (len(dist), r_s.statistic, r_s.pvalue))
for lo, hi in ((0, 200), (200, 400), (400, 800), (800, 2000), (2000, 1e9)):
    m = (dist >= lo) & (dist < hi)
    if m.sum() < 3:
        continue
    print("    distance %5.0f-%-6.0f px  n=%4d  mean curve correlation=%+.3f  mean|peak diff|=%6.2f um"
          % (lo, hi, m.sum(), corr[m].mean(), peakdiff[m].mean()))
print("  same block:  n=%d  correlation=%+.3f  |peak diff|=%.2f um"
      % (same_block.sum(), corr[same_block].mean(), peakdiff[same_block].mean()))
print("  different block (same segment): n=%d  correlation=%+.3f  |peak diff|=%.2f um"
      % ((~same_block).sum(), corr[~same_block].mean(), peakdiff[~same_block].mean()))
R["distance"] = dict(spearman_rho=float(r_s.statistic), spearman_p=float(r_s.pvalue),
                     same_block_corr=float(corr[same_block].mean()),
                     other_block_corr=float(corr[~same_block].mean()),
                     same_block_peak_diff=float(peakdiff[same_block].mean()),
                     other_block_peak_diff=float(peakdiff[~same_block].mean()),
                     n_pairs=int(len(dist)))

# ---------------------------------------------------------------- 6) SAME TESTS WITH AUC-PR
print("\n" + "=" * 76)
print("6) THE SAME TESTS WITH AUC-PR (average precision) - is the sign metric dependent?")
print("=" * 76)
for nm, X in (("AUC-ROC", M), ("AUC-PR", AP)):
    ak = X[:, i0]
    asg = np.empty(n); abl = np.full(n, np.nan); okb = np.zeros(n, bool)
    for i in range(n):
        k = (seg == seg[i]) & (np.arange(n) != i)
        asg[i] = X[i, int(np.argmax(X[k].mean(0)))]
        kb = (seg == seg[i]) & (blk != blk[i])
        if kb.any():
            abl[i] = X[i, int(np.argmax(X[kb].mean(0)))]; okb[i] = True
    d1 = asg - ak
    d2 = abl[okb] - ak[okb]
    print("  %-8s  neighbour-inclusive LOTO %+.5f (p=%.3g)   |   out-of-block CV %+.5f (p=%.3g, n=%d)"
          % (nm, d1.mean(), stats.wilcoxon(d1).pvalue, d2.mean(),
             stats.wilcoxon(d2).pvalue, okb.sum()))
    R["metric_" + nm] = dict(loto=float(d1.mean()), loto_p=float(stats.wilcoxon(d1).pvalue),
                             out_of_block=float(d2.mean()),
                             out_of_block_p=float(stats.wilcoxon(d2).pvalue))

json.dump(R, open(os.path.join(WORK, "scale.json"), "w"), indent=1)
print("\nwritten -> " + os.path.join(WORK, "scale.json"))
