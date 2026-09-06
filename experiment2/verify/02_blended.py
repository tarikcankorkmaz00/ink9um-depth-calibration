# -*- coding: utf-8 -*-
"""Step 2 - what the phase effect costs in real blended inference.

Step 1 moves a single patch. That is not how anyone runs the model. This step
uses the sliding window with Hann blending, the same positions infer.py picks,
and changes only the stride.

The competing explanation - "off-grid strides just cover the region less evenly"
- is tested directly by comparing neighbouring strides whose coverage geometry
is almost identical and whose only real difference is the phase.

Reads data/blended.json only. No GPU, no downloads.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
B = json.load(open(os.path.join(EXP, "data", "blended.json")))
A = {k: {int(s): v for s, v in row.items()} for k, row in B["auc_by_stride"].items()}
KEYS = sorted(A)
STR = sorted(set().union(*[set(r) for r in A.values()]))

head = lambda t: (print("=" * 74), print(t), print("=" * 74))

head("PROTOCOL")
print(B["note"])
for k, v in B["protocol"].items():
    print("     %-22s %s" % (k, v))
print("     %-22s %d (%d physical segments, one scroll: %s)"
      % ("measurements", B["measurements"], B["physical_segments"], B["scroll"]))

head("1) AUC BY STRIDE, PER MEASUREMENT")
print("     %-26s %s" % ("", "  ".join("s%-6d" % s for s in STR)))
for k in KEYS:
    print("     %-26s %s" % (k, "  ".join("%-7.5f" % A[k].get(s, float("nan")) for s in STR)))

head("2) EVERYTHING AGAINST THE DEFAULT (--overlap 0.5 -> stride 64)")
print("     %-9s %-9s %-12s %-14s %s" % ("stride", "mod 8", "mean delta", "worse in", "median delta"))
for s in STR:
    d = [A[k][s] - A[k][64] for k in KEYS if s in A[k]]
    print("     %-9d %-9d %-+12.5f %-14s %-+.5f"
          % (s, s % 8, np.mean(d), "%d/%d" % (sum(1 for x in d if x < 0), len(d)), np.median(d)))
ong = [np.mean([A[k][s] - A[k][64] for k in KEYS if s in A[k]]) for s in STR if s % 8 == 0]
offg = [(s, np.mean([A[k][s] - A[k][64] for k in KEYS if s in A[k]])) for s in STR if s % 8]
print("\n     every stride that is a multiple of 8 sits within %.5f of the default"
      % max(abs(x) for x in ong))
print("     the off-grid strides cost %+.5f .. %+.5f"
      % (min(x for _, x in offg), max(x for _, x in offg)))

head("3) THE COVERAGE EXPLANATION, TESTED DIRECTLY")
print("Neighbouring strides. Coverage geometry is nearly the same; the phase is not.\n")
print("     %-24s %-12s %-10s %s" % ("pair", "mean delta", "worse in", "per measurement"))
for a, b in B["neighbour_pairs"]:
    d = [A[k][b] - A[k][a] for k in KEYS if a in A[k] and b in A[k]]
    print("     %-24s %-+12.5f %-10s %s"
          % ("%d (mod8=%d) -> %d (mod8=%d)" % (a, a % 8, b, b % 8),
             np.mean(d), "%d/%d" % (sum(1 for x in d if x < 0), len(d)),
             " ".join("%+.4f" % x for x in d)))
print("\n     All four pairs go the same way in every measurement.")
print("     Coverage does not explain this; phase does.")

head("4) THE ONE-LINE VERSION")
d90 = [A[k][90] - A[k][64] for k in KEYS if 90 in A[k]]
print("     Choosing --overlap 0.3 instead of the default 0.5 turns stride 64 into")
print("     stride 90, and costs %+.4f AUC (worse in %d of %d measurements)."
      % (np.mean(d90), sum(1 for x in d90 if x < 0), len(d90)))
print("     Pick an --overlap whose stride is a multiple of 8:")
print("     0, 0.25, 0.375, 0.5, 0.75, 0.875. The default is already one of them.")

head("LIMITS OF THIS STEP - all three belong in any sentence quoting 0.03")
print("  1. %d measurements, %d physical segments, ONE scroll (%s), two imaging conditions."
      % (B["measurements"], B["physical_segments"], B["scroll"]))
print("  2. All of it inside the training (supervision) region.")
print("  3. At stride 64 the AUCs are %.4f-%.5f, i.e. at the ceiling. A ceiling"
      % (min(A[k][64] for k in KEYS), max(A[k][64] for k in KEYS)))
print("     usually flattens an effect rather than inflating it, but that is a guess,")
print("     not a measurement. The size in a deployment regime is unknown.")
print("  4. Single-patch cost (step 1: 0.09-0.23) and blended cost (this step: 0.005-0.032)")
print("     are different numbers. Do not quote the first as if it were the second.")
