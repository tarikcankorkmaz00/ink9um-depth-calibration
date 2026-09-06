# -*- coding: utf-8 -*-
"""Step 7 - controls for the negative out-of-block result.

Out-of-block CV came out negative in step 5. Two explanations compete:
  (a) it is real - the offset does not transfer across space, or
  (b) it is an artefact - the out-of-block estimate simply uses fewer tiles and is noisier.
A SIZE-MATCHED control separates the two.

Also: do the per-tile peaks follow a smooth spatial gradient (a warped surface)?
And how stable is a block's peak when the block is split in half?
Finally, a summary table of every offset-source policy on the held-out tiles.
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
yx = np.array([(e["y"], e["x"]) for e in V])
i0 = int(np.argmin(np.abs(off)))
n = len(M)
R = {}


def blocks(seg, yx):
    lab = np.full(len(yx), -1); nb = 0
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
rng = np.random.default_rng(20260906)

# ---------------------------------------------------------------- 1) SIZE-MATCHED CONTROL
print("=" * 76)
print("1) SIZE-MATCHED CONTROL")
print("   out-of-block CV: offset from ANOTHER block of the same segment (k tiles, spatially apart)")
print("   random-k       : offset from k RANDOM tiles of the same segment (neighbours allowed)")
print("=" * 76)
test = [i for i in range(n) if ((seg == seg[i]) & (blk != blk[i])).any()]
NB = 2000
a_blk = np.empty(len(test))
a_rand = np.zeros((NB, len(test)))
for t, i in enumerate(test):
    out = np.where((seg == seg[i]) & (blk != blk[i]))[0]
    a_blk[t] = M[i, int(np.argmax(M[out].mean(0)))]
    pool = np.where((seg == seg[i]) & (np.arange(n) != i))[0]
    k = len(out)
    for b in range(NB):
        pick = rng.choice(pool, k, replace=False)
        a_rand[b, t] = M[i, int(np.argmax(M[pick].mean(0)))]
ak = a_k0[test]
d_blk = a_blk - ak
d_rand = a_rand - ak[None, :]
rm = d_rand.mean(1)
print("  tiles tested: %d  (the random arm uses the same number of source tiles)" % len(test))
print("  out-of-block (spatially APART) gain = %+.5f" % d_blk.mean())
print("  random-k (neighbours allowed)       = %+.5f   95%%[%+.5f, %+.5f]"
      % (rm.mean(), np.percentile(rm, 2.5), np.percentile(rm, 97.5)))
p_in = float(((rm <= d_blk.mean()).sum() + 1) / (NB + 1))
print("  the out-of-block gain sits at percentile %.1f of the size-matched random distribution  (p=%.4g)"
      % (100 * (rm <= d_blk.mean()).mean(), p_in))
print("  --> the difference comes from SPATIAL NEIGHBOURHOOD, not from sample size"
      if p_in < 0.05 else "  --> sample size can explain the difference; not conclusive")
R["size_matched"] = dict(n_test=len(test), out_of_block=float(d_blk.mean()),
                         random_mean=float(rm.mean()),
                         random_lo=float(np.percentile(rm, 2.5)),
                         random_hi=float(np.percentile(rm, 97.5)),
                         p=p_in, n_boot=NB)

# ---------------------------------------------------------------- 2) SPATIAL GRADIENT
print("\n" + "=" * 76)
print("2) IS THERE A SMOOTH SPATIAL GRADIENT IN THE PEAK OFFSETS? (warped-surface hypothesis)")
print("=" * 76)


def ppeak(c):
    k = int(np.argmax(c))
    if k in (0, len(c) - 1):
        return float(off[k])
    den = c[k - 1] - 2 * c[k] + c[k + 1]
    dd = 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (c[k - 1] - c[k + 1]) / den, -1, 1))
    return float(off[k] + dd * (off[1] - off[0]))


peak = np.array([ppeak(M[i]) for i in range(n)])
R["gradient"] = {}
for sg in sorted(set(seg)):
    m = seg == sg
    Y = yx[m, 0].astype(float) * 128; X = yx[m, 1].astype(float) * 128
    A = np.column_stack([np.ones(m.sum()), Y - Y.mean(), X - X.mean()])
    b, *_ = np.linalg.lstsq(A, peak[m], rcond=None)
    pred = A @ b
    ss_t = ((peak[m] - peak[m].mean()) ** 2).sum()
    r2 = 1 - ((peak[m] - pred) ** 2).sum() / ss_t if ss_t > 0 else 0.0
    dfn, dfd = 2, m.sum() - 3
    F = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 and dfd > 0 else np.inf
    p = float(1 - stats.f.cdf(F, dfn, dfd))
    print("  %-16s n=%2d  plane fit R2=%.3f  p=%.3f   slope: %+.2f um/1000px(y)  %+.2f um/1000px(x)"
          % (sg, m.sum(), r2, p, b[1] * 1000, b[2] * 1000))
    R["gradient"][str(sg)] = dict(n=int(m.sum()), R2=float(r2), p=p,
                                  slope_y_um_per_1000px=float(b[1] * 1000),
                                  slope_x_um_per_1000px=float(b[2] * 1000))
print("  (low R2 + high p => the peaks are not explained by a smooth surface tilt, they are noisy)")

# ---------------------------------------------------------------- 3) STABILITY OF A BLOCK PEAK
print("\n" + "=" * 76)
print("3) HOW STABLE IS A BLOCK PEAK? (split the block in half, do both halves agree?)")
print("=" * 76)
R["half_split"] = {}
for b in sorted(set(blk)):
    idx = np.where(blk == b)[0]
    if len(idx) < 6:
        print("  block %d (%s) n=%d -> too small, skipped" % (b, seg[idx[0]], len(idx)))
        continue
    diffs = []
    for _ in range(2000):
        p = rng.permutation(idx)
        h = len(idx) // 2
        t1 = off[int(np.argmax(M[p[:h]].mean(0)))]
        t2 = off[int(np.argmax(M[p[h:2 * h]].mean(0)))]
        diffs.append(abs(t1 - t2))
    diffs = np.array(diffs)
    print("  block %d %-16s n=%2d  half-vs-half peak difference: median=%5.1f um  mean=%5.1f um  "
          "P(>19.2um)=%.2f" % (b, seg[idx[0]], len(idx), np.median(diffs),
                               diffs.mean(), (diffs > 19.2).mean()))
    R["half_split"][int(b)] = dict(seg=str(seg[idx[0]]), n=int(len(idx)),
                                   median_um=float(np.median(diffs)),
                                   mean_um=float(diffs.mean()),
                                   frac_above_19um=float((diffs > 19.2).mean()))

# ---------------------------------------------------------------- 4) POLICY SUMMARY
print("\n" + "=" * 76)
print("4) HELD-OUT AUC OF EVERY OFFSET-SOURCE POLICY")
print("   Each policy is compared against the canonical window ON THE SAME TILES, because")
print("   one of them can only be computed on a subset.")
print("=" * 76)
pol = {}          # label -> (policy AUC, canonical AUC on the same tiles, n tiles)
allt = np.ones(n, bool)


def add(label, a, mask=allt):
    pol[label] = (float(np.mean(a[mask])), float(a_k0[mask].mean()), int(mask.sum()))


add("canonical k=0 (do nothing)", a_k0)
gi = int(np.argmax(M.mean(0)))
add("single COMMON offset, ORACLE (%+.1f um, chosen on all the data)" % off[gi], M[:, gi])
a = np.empty(n)
for i in range(n):
    k = np.arange(n) != i
    a[i] = M[i, int(np.argmax(M[k].mean(0)))]
add("single COMMON offset, leave-one-tile-out CV", a)
a = np.empty(n)
for i in range(n):
    k = seg != seg[i]
    a[i] = M[i, int(np.argmax(M[k].mean(0)))]
add("offset from OTHER SEGMENTS (LOSO)", a)
a = np.full(n, np.nan)
for i in range(n):
    k = (seg == seg[i]) & (blk != blk[i])
    if k.any():
        a[i] = M[i, int(np.argmax(M[k].mean(0)))]
sub = ~np.isnan(a)
add("offset from SAME SEGMENT / OTHER BLOCK (spatially apart)", a, sub)
a = np.empty(n)
for i in range(n):
    k = (blk == blk[i]) & (np.arange(n) != i)
    a[i] = M[i, i0] if not k.any() else M[i, int(np.argmax(M[k].mean(0)))]
add("offset from the SAME BLOCK (neighbouring tiles)", a)
a = np.empty(n)
for i in range(n):
    k = (seg == seg[i]) & (np.arange(n) != i)
    a[i] = M[i, int(np.argmax(M[k].mean(0)))]
add("offset from the SAME SEGMENT (original claim, neighbours included)", a)
add("per-tile oracle (upper bound, not a method)", M.max(1))
for k, (v, b, nt) in pol.items():
    print("  %-58s AUC %.5f  vs canonical %.5f on n=%2d  ->  %+.5f"
          % (k, v, b, nt, v - b))
print("  note: the spatially-apart policy exists only for the %d tiles whose segment has"
      % int(sub.sum()))
print("        more than one block, so its canonical baseline is %.5f, not %.5f."
      % (a_k0[sub].mean(), a_k0.mean()))
R["policy_summary"] = {k: v[0] for k, v in pol.items()}
R["policy_matched_baseline"] = {k: v[1] for k, v in pol.items()}
R["policy_n_tiles"] = {k: v[2] for k, v in pol.items()}
R["policy_baseline"] = float(a_k0.mean())

json.dump(R, open(os.path.join(WORK, "controls.json"), "w"), indent=1)
print("\nwritten -> " + os.path.join(WORK, "controls.json"))
