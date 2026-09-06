# -*- coding: utf-8 -*-
"""Step 9 - how stable is the supervision -> validation transfer gain of step 8?

The supervision curve is saturated (AUC around 0.99), so its peak may be chosen by noise.
This step measures the dynamic range of the supervision curves, splits the supervision
tiles in half to see whether both halves pick the same peak, and puts a bootstrap
confidence interval on the transfer gain by resampling the supervision tiles.
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


SUP = load("supervision_scan.json", "supervision_scan")
VAL = load("heldout_scan.json", "heldout_scan")["tiles"]
off = np.array(VAL[0]["um"], float)
Mv = np.stack([np.array(e["auc"], float) for e in VAL])
segv = np.array([e["seg"] for e in VAL])
Ms = np.stack([np.array(e["auc"], float) for e in SUP])
segs_ = np.array([e["seg"] for e in SUP])
i0 = int(np.argmin(np.abs(off)))
rng = np.random.default_rng(20260906)
R = {}

print("=" * 76)
print("1) DYNAMIC RANGE OF THE SUPERVISION CURVES - is picking a peak meaningful there?")
print("=" * 76)
for sg in sorted(set(segs_)):
    c = Ms[segs_ == sg].mean(0)
    print("  %-16s n=%2d  AUC range %.5f - %.5f  (width %.5f)  peak %+6.1f um"
          % (sg, int((segs_ == sg).sum()), c.min(), c.max(), c.max() - c.min(),
             off[int(np.argmax(c))]))
    cv = Mv[segv == sg].mean(0)
    print("      validation curve  AUC range %.5f - %.5f  (width %.5f)  peak %+6.1f um"
          % (cv.min(), cv.max(), cv.max() - cv.min(), off[int(np.argmax(cv))]))
R["dynamic_range"] = {str(sg): dict(
    sup_width=float(Ms[segs_ == sg].mean(0).max() - Ms[segs_ == sg].mean(0).min()),
    val_width=float(Mv[segv == sg].mean(0).max() - Mv[segv == sg].mean(0).min()))
    for sg in sorted(set(segs_))}

print("\n" + "=" * 76)
print("2) STABILITY OF THE SUPERVISION PEAK (split the supervision tiles in half)")
print("=" * 76)
R["sup_half_split"] = {}
for sg in sorted(set(segs_)):
    idx = np.where(segs_ == sg)[0]
    f = []
    for _ in range(3000):
        p = rng.permutation(idx); h = len(idx) // 2
        f.append(abs(off[int(np.argmax(Ms[p[:h]].mean(0)))] -
                     off[int(np.argmax(Ms[p[h:2 * h]].mean(0)))]))
    f = np.array(f)
    print("  %-16s n=%2d  half-vs-half peak difference: median=%5.1f um  mean=%5.1f um  "
          "P(identical)=%.2f" % (sg, len(idx), np.median(f), f.mean(), (f == 0).mean()))
    R["sup_half_split"][str(sg)] = dict(n=int(len(idx)), median_um=float(np.median(f)),
                                        mean_um=float(f.mean()),
                                        frac_identical=float((f == 0).mean()))

print("\n" + "=" * 76)
print("3) BOOTSTRAP CONFIDENCE INTERVAL FOR THE TRANSFER GAIN")
print("   (supervision tiles resampled, the offset re-estimated each time)")
print("=" * 76)
NB = 5000
gains = np.empty(NB)
seg_gains = {sg: np.empty(NB) for sg in sorted(set(segv))}
peaks = {sg: [] for sg in sorted(set(segv))}
for b in range(NB):
    a = np.empty(len(Mv))
    for sg in sorted(set(segv)):
        idx = np.where(segs_ == sg)[0]
        pick = rng.choice(idx, len(idx), replace=True)
        j = int(np.argmax(Ms[pick].mean(0)))
        peaks[sg].append(off[j])
        mv = segv == sg
        a[mv] = Mv[mv, j]
        seg_gains[sg][b] = Mv[mv, j].mean() - Mv[mv, i0].mean()
    gains[b] = a.mean() - Mv[:, i0].mean()
a_point = np.empty(len(Mv))
for sg in sorted(set(segv)):
    a_point[segv == sg] = Mv[segv == sg, int(np.argmax(Ms[segs_ == sg].mean(0)))]
print("  OVERALL transfer gain, point estimate = %+.5f" % (a_point.mean() - Mv[:, i0].mean()))
print("    bootstrap mean=%+.5f  95%% [%+.5f, %+.5f]  P(gain>0)=%.3f  P(gain=0)=%.3f"
      % (gains.mean(), np.percentile(gains, 2.5), np.percentile(gains, 97.5),
         (gains > 0).mean(), (gains == 0).mean()))
R["bootstrap_overall"] = dict(mean=float(gains.mean()), lo=float(np.percentile(gains, 2.5)),
                              hi=float(np.percentile(gains, 97.5)),
                              P_positive=float((gains > 0).mean()),
                              P_zero=float((gains == 0).mean()))
R["bootstrap_per_segment"] = {}
for sg in sorted(set(segv)):
    g = seg_gains[sg]; t = np.array(peaks[sg])
    vals, cnts = np.unique(t, return_counts=True)
    top = vals[np.argsort(-cnts)][:3]
    frac = np.sort(cnts)[::-1][:3] / NB
    print("  %-16s gain mean=%+.5f  95%%[%+.5f,%+.5f]  P(>0)=%.3f | most frequent peaks: %s"
          % (sg, g.mean(), np.percentile(g, 2.5), np.percentile(g, 97.5), (g > 0).mean(),
             ", ".join("%+.1fum(%.0f%%)" % (v, 100 * c) for v, c in zip(top, frac))))
    R["bootstrap_per_segment"][str(sg)] = dict(
        mean=float(g.mean()), lo=float(np.percentile(g, 2.5)), hi=float(np.percentile(g, 97.5)),
        P_positive=float((g > 0).mean()),
        most_frequent_peaks=[[float(v), float(c)] for v, c in zip(top, frac)])

print("\n" + "=" * 76)
print("4) THE SEGMENT AS THE UNIT OF ANALYSIS: n=3")
print("=" * 76)
gz = []
for sg in sorted(set(segv)):
    c = Ms[segs_ == sg].mean(0); j = int(np.argmax(c)); mv = segv == sg
    gz.append(Mv[mv, j].mean() - Mv[mv, i0].mean())
    print("  %-16s %+.5f" % (sg, gz[-1]))
gz = np.array(gz)
print("  over n=3 segments: mean=%+.5f  sd=%.5f" % (gz.mean(), gz.std(ddof=1)))
print("  one-sample t-test (n=3): t=%.3f p=%.3f   -> %s"
      % (stats.ttest_1samp(gz, 0).statistic, stats.ttest_1samp(gz, 0).pvalue,
         "significant" if stats.ttest_1samp(gz, 0).pvalue < 0.05 else "NOT SIGNIFICANT"))
print("  sign test (n=3, 2 positive 1 zero): no exact p, n is far too small")
R["segment_level_n3"] = dict(gains=[float(v) for v in gz], mean=float(gz.mean()),
                             sd=float(gz.std(ddof=1)),
                             t=float(stats.ttest_1samp(gz, 0).statistic),
                             p=float(stats.ttest_1samp(gz, 0).pvalue))

json.dump(R, open(os.path.join(WORK, "stability.json"), "w"), indent=1)
print("\nwritten -> " + os.path.join(WORK, "stability.json"))
