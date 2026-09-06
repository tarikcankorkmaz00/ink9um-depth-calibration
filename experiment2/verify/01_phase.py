# -*- coding: utf-8 -*-
"""Step 1 - the patch-phase effect in the published ink_9um checkpoint.

Every AUC below is computed on the SAME pixels with the SAME labels. The only
thing that changes between rows is where the 128x128 window starts.

Reads data/phase.json only. No GPU, no downloads.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
P = json.load(open(os.path.join(EXP, "data", "phase.json")))

line = lambda: print("-" * 74)
head = lambda t: (print("=" * 74), print(t), print("=" * 74))

head("1) ONE-AXIS SHIFT SCAN - single 128x128 patch per tile")
print("Window origin moved in y only. Delta is against shift 0.")
print("'.' marks a shift that is a multiple of 8.\n")
for fam in ("native9", "aligned"):
    b = P["shift_1d"][fam]
    print("  %s, n = %d tiles" % (fam, b["n_tiles"]))
    print("     %-7s %-11s %-11s %-11s" % ("shift", "mean AUC", "mean delta", "median delta"))
    for k in sorted(b["shift_px"], key=int):
        v = b["shift_px"][k]
        print("     %-4s%s  %-11.5f %-+11.5f %-+11.5f"
              % (k, "." if int(k) % 8 == 0 else " ",
                 v["mean_auc"], v["mean_delta"], v["median_delta"]))
    on = [v["mean_delta"] for k, v in b["shift_px"].items() if int(k) and int(k) % 8 == 0]
    off = [v["mean_delta"] for k, v in b["shift_px"].items() if int(k) % 8]
    print("     worst nonzero multiple of 8: %+.5f   worst non-multiple: %+.5f\n"
          % (min(on), min(off)))

print("  The mean moves far more than the median: a few tiles collapse, most")
print("  degrade moderately. Quoting the mean alone overstates the typical tile.")

head("2) THE SAME THING IN TWO AXES")
b = P["phase_2d"]
print("n = %d tiles. Phase is (shift mod 8) in y and x.\n" % b["n_tiles"])
print("     %-9s %-9s %-11s %-11s" % ("shift y,x", "phase", "mean delta", "median delta"))
for k in b["shift_yx"]:
    v = b["shift_yx"][k]
    print("     %-9s %-9s %-+11.5f %-+11.5f"
          % (k, "%d,%d" % tuple(v["phase_mod8"]), v["mean_delta"], v["median_delta"]))
inph = [v["mean_delta"] for v in b["shift_yx"].values() if v["phase_mod8"] == [0, 0]]
outph = [v["mean_delta"] for v in b["shift_yx"].values() if v["phase_mod8"] != [0, 0]]
print("\n     phase (0,0):     worst %+.5f over %d combinations" % (min(inph), len(inph)))
print("     any other phase: worst %+.5f over %d combinations" % (min(outph), len(outph)))

head("3) IS IT NUMERICAL? fp16 vs fp32")
b = P["precision"]
print("n = %d tiles.\n" % b["n_tiles"])
print("     %-9s %-12s %-12s %-10s" % ("shift y,x", "fp16 delta", "fp32 delta", "gap"))
for k, v in b["shift_yx"].items():
    print("     %-9s %-+12.5f %-+12.5f %.6f"
          % (k, v["fp16_delta"], v["fp32_delta"], abs(v["fp16_delta"] - v["fp32_delta"])))
gap = max(abs(v["fp16_delta"] - v["fp32_delta"]) for v in b["shift_yx"].values())
print("\n     largest fp16/fp32 gap: %.6f  -> not a precision artefact." % gap)

head("4) IS IT THE INPUT NORMALISATION?")
b = P["normalisation_control"]
print("%s\n%s, n = %d blocks.\n" % (b["note"], b["segment"], b["n_blocks"]))
print("     %-7s %-12s %-12s" % ("shift", "single", "per window"))
for k in sorted(b["shift_px"], key=int):
    v = b["shift_px"][k]
    print("     %-7s %-12.5f %-12.5f" % (k, v["single"], v["per_window"]))
s = b["shift_px"]
c4 = (s["4"]["single"] - s["0"]["single"], s["4"]["per_window"] - s["0"]["per_window"])
print("\n     cost of shift 4:  single %+.5f   per window %+.5f   difference %.5f"
      % (c4[0], c4[1], abs(c4[0] - c4[1])))
print("     -> normalisation is not where this comes from.")

head("5) INDEPENDENT REPRODUCTION - second code path, written from scratch")
b = P["independent_reproduction"]
print("%s\n%s, n = %d blocks.\n" % (b["note"], b["segment"], b["n_blocks"]))
print("     %-7s %-10s %-12s %-10s" % ("shift", "AUC", "delta", "logit r"))
for k in sorted(b["shift_px"], key=int):
    v = b["shift_px"][k]
    mk = "." if int(k) % 8 == 0 else " "
    print("     %-4s%s  %-10.4f %-+12.5f %-10.3f" % (k, mk, v["auc"], v["delta_auc"], v["logit_r"]))
on = {int(k): v for k, v in b["shift_px"].items() if int(k) % 8 == 0}
off = {int(k): v for k, v in b["shift_px"].items() if int(k) % 8}
print("\n     multiples of 8:     delta %+.4f .. %+.4f" %
      (min(v["delta_auc"] for v in on.values()), max(v["delta_auc"] for v in on.values())))
print("     everything else:    delta %+.4f .. %+.4f" %
      (min(v["delta_auc"] for v in off.values()), max(v["delta_auc"] for v in off.values())))
print("     but even at a multiple of 8 the logits are not identical: r = %.2f .. %.2f"
      % (min(v["logit_r"] for k, v in on.items() if k),
         max(v["logit_r"] for k, v in on.items() if k)))
print("     -> 'a multiple of 8 is safe' is about AUC, not about equal output.")

head("6) NO LABELS AT ALL - two logit maps on the same global pixels")
b = P["label_free"]
print("%s\n" % b["note"])
print("     %-16s %-9s %-14s %-14s" % ("family|norm", "shifts", "mean |d logit|", "mean logit r"))
for key in sorted(b["data"]):
    d = b["data"][key]
    on = [v for k, v in d.items() if int(k) and int(k) % 8 == 0]
    off = [v for k, v in d.items() if int(k) % 8]
    if not on or not off:
        continue
    for nm, grp in (("mult. of 8", on), ("all others", off)):
        print("     %-16s %-9s %-14.3f %-14.3f"
              % (key if nm.startswith("mult") else "", nm,
                 np.mean([x[0] for x in grp]), np.mean([x[1] for x in grp])))
print("\n  Same period without a single label, so the effect is in the model output,")
print("  not in how the tiles were scored. The separation is wide on native9 and")
print("  narrow on aligned - stated as measured, not smoothed over.")

head("7) WHERE THIS IS REACHABLE FROM THE OFFICIAL INFERENCE PATH")
b = P["overlap_to_stride"]
print("%s\npatch_size = %d, default --overlap = %s\n"
      % (b["formula"], b["patch_size"], b["default_overlap"]))
print("     %-10s %-9s %-11s %s" % ("--overlap", "stride", "stride mod 8", "verdict"))
for r in b["table"]:
    print("     %-10s %-9d %-11d %s"
          % (r["overlap"], r["stride"], r["stride_mod8"],
             "on the grid" if r["stride_mod8"] == 0 else "OFF the grid"))
offg = [r for r in b["table"] if r["stride_mod8"]]
oneplace = [r for r in b["table"] if round(r["overlap"], 1) == r["overlap"]]
badone = [r for r in oneplace if r["stride_mod8"]]
print("\n     of the %d one-decimal overlap values, %d land off the grid"
      % (len(oneplace), len(badone)))
print("     the default (0.5 -> stride 64) is safe")
print("\n     %s" % b["last_row_note"])

head("8) THE MODEL'S OWN CONFIG DOES NOT EXPLAIN A PERIOD OF 8")
m = P["model_config"]
for k in ("conv_op", "norm_op", "pool_op_kernel_sizes", "num_pool_per_axis",
          "must_be_divisible_by"):
    print("     %-22s %s" % (k, m[k]))
print("\n     %s" % m["comment"])
print("     source: %s" % m["source"])
line()
print("Measured: the behaviour. Not measured: the reason.")
