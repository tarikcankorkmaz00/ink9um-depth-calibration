# -*- coding: utf-8 -*-
"""Step 10 - the same question at scale: 531 tiles, 24 aligned segments, 4 scrolls.

The earlier claim was "the offset belongs to the SEGMENT, not to the scroll:
per-segment adds +0.0070 AUC, per-scroll only +0.0004, a factor of 16.8".
This step re-measures that with its own pipeline AND adds one level the earlier
analysis never looked at: the spatially connected BLOCK inside a segment.

It also runs every grouping level through an honest leave-one-tile-out
cross-validation, instead of picking each group's offset on its own data.

The measurement half needs a GPU, the villa package and the cached chunks. The
analysis half only needs numpy and scipy, and runs directly on the committed
tile scan in data/tile_scan_531.json.
"""
import json, os, re, sys, time, collections
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
ROOT = os.environ.get("VZ_ROOT", os.path.join(REPO, "data"))
os.makedirs(WORK, exist_ok=True)

SCAN_WORK = os.path.join(WORK, "tile_scan.json")
SCAN_SHIPPED = os.path.join(REPO, "data", "tile_scan_531.json")

if os.path.exists(SCAN_WORK):
    out = json.load(open(SCAN_WORK))
    print("using the scan in %s: %d tiles" % (SCAN_WORK, len(out)))
elif os.path.exists(SCAN_SHIPPED):
    out = json.load(open(SCAN_SHIPPED))
    print("using the committed scan in data/tile_scan_531.json: %d tiles" % len(out))
else:
    # ---------------------------------------------------------------- MEASUREMENT (needs GPU)
    import torch, numcodecs
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, os.path.join(ROOT, "villa", "vesuvius", "src"))
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.inference.inference_runtime import TargetModel
    from vesuvius.image_proc.intensity.normalization import normalize_robust

    TREE = os.path.join(WORK, "tree_full.json")
    CACHE = os.path.join(ROOT, "cache", "labels")
    VOLD = os.path.join(ROOT, "cache", "volumes")
    SEGS = {s["key"]: s for s in json.load(
        open(os.path.join(ROOT, "segments.json"), encoding="utf-8"))["segments"]}
    _BL = numcodecs.Blosc(); _c = {}

    PAT = re.compile(r"^ink_9um/labels/([^/]+)/([^/]+)/([^/]+)\.zarr/0/(\d+)\.(\d+)\.(\d+)$")
    KIND = {"_inklabels": "ink", "_supervision_mask": "sup", "_validation_mask": "val"}
    IDXP = os.path.join(WORK, "full_index.json")
    if os.path.exists(IDXP):
        IDX = json.load(open(IDXP))
    else:
        IDX = collections.defaultdict(dict)
        for e in json.load(open(TREE)):
            m = PAT.match(e["path"])
            if not m:
                continue
            fam, seg_, arr, z, y, x = m.groups()
            k = next((v for s_, v in KIND.items() if arr.endswith(s_)), None)
            if k is None:
                continue
            IDX[f"{fam}/{seg_}|{k}"][f"{y},{x}"] = e["xetHash"]
        IDX = dict(IDX)
        json.dump(IDX, open(IDXP, "w"))
    print("index ready:", len(IDX), "segment-kind entries")

    def blob(h, nz):
        if h in _c:
            return _c[h]
        fp = os.path.join(CACHE, h[:2], h)
        a = (np.zeros((nz, 128, 128), np.uint8)
             if not (os.path.exists(fp) and os.path.getsize(fp))
             else np.frombuffer(_BL.decode(open(fp, "rb").read()),
                                np.uint8).reshape(nz, 128, 128).copy())
        if len(_c) < 4000:
            _c[h] = a
        return a

    def label(segkey, kind, y, x, nz):
        h = IDX.get(f"{segkey}|{kind}", {}).get(f"{y},{x}")
        return np.zeros((nz, 128, 128), np.uint8) if h is None else blob(h, nz)

    ck = torch.load(os.path.join(ROOT, "model", "ink_9um", "hybrid_3d2d-seed42",
                                 "step-075000.pth"), map_location="cpu", weights_only=False)
    WIN = int(ck["config"]["patch_size"][0])
    cfg = InkConfig.from_mapping(ck["config"])
    mm = make_model(cfg)
    st = ck["model"]
    if st and all(str(k).startswith("module.") for k in st):
        st = {k[len("module."):]: v for k, v in st.items()}
    mm.load_state_dict(st, strict=False)
    model = TargetModel(mm, input_pad_depth_to=cfg.model.input_pad_depth_to).eval().cuda()

    def make_input(raw, s, d, WIN=WIN):
        if s["family"].startswith("aligned"):
            z0, z1 = s["source_z_slice"]
            nlab = s["label_shape_zyx"][0]
            pool = (z1 - z0) // nlab
            start = z0 + pool * (s["annotation_center_channel"] - WIN // 2) + d
            if start < 0 or start + pool * WIN > raw.shape[0]:
                return None
            b = raw[start:start + pool * WIN].astype(np.float32)
            return np.rint(b.reshape(WIN, pool, 128, 128).mean(axis=1))
        start = s["annotation_center_channel"] - WIN // 2 + d
        if start < 0 or start + WIN > raw.shape[0]:
            return None
        return raw[start:start + WIN].astype(np.float32)

    files = sorted(f for f in os.listdir(VOLD) if f.endswith(".npy"))
    out, t0 = [], time.time()
    for i, fn in enumerate(files):
        stem = fn[:-4]; segkey, yxs = stem.rsplit("__", 1)
        segkey = segkey.replace("__", "/")
        y, x = (int(v) for v in yxs.split("_"))
        s = SEGS.get(segkey)
        if s is None:
            continue
        ali = s["family"].startswith("aligned")
        nz = s["label_chunk_zyx"][0]; zc = s["annotation_center_channel"]
        sup = label(segkey, "sup", y, x, nz)[zc] > 0
        if sup.sum() < 500:
            continue
        ink = label(segkey, "ink", y, x, nz)[zc] > 0
        yl = ink[sup]
        if yl.sum() < 60 or (~yl).sum() < 60:
            continue
        raw = np.load(os.path.join(VOLD, fn))
        D = list(range(-16, 17, 2)) if ali else list(range(-5, 6))
        batch, keep = [], []
        for d in D:
            g = make_input(raw, s, d)
            if g is None:
                continue
            batch.append(normalize_robust(g)); keep.append(d)
        if not batch:
            continue
        xb = torch.from_numpy(np.stack(batch)).unsqueeze(1).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg = model(xb)
        lg = lg.float().squeeze(1).cpu().numpy()
        um = (float(s["volume_base_voxel_um"][0]) if ali
              else float(s["effective_voxel_um"][0]))
        out.append(dict(segkey=segkey, scroll=s["scroll"].split(" ")[0], seg=s["segment"],
                        family="aligned" if ali else "native9", y=y, x=x,
                        n_ink=int(yl.sum()), n_bg=int((~yl).sum()),
                        um=[d * um for d in keep],
                        auc=[round(float(roc_auc_score(yl.astype(np.uint8), lg[k][sup])), 6)
                             for k in range(len(keep))]))
        if (i + 1) % 100 == 0:
            print("  %d/%d  %.0fs" % (i + 1, len(files), time.time() - t0), flush=True)
    json.dump(out, open(SCAN_WORK, "w"))
    print("%d tiles measured, %.0fs" % (len(out), time.time() - t0))

# ---------------------------------------------------------------- ANALYSIS
A = [e for e in out if e["family"] == "aligned"]
off = np.array(A[0]["um"], float)
M = np.stack([np.array(e["auc"], float) for e in A])
scroll = np.array([e["scroll"] for e in A])
seg = np.array([e["seg"] for e in A])
yx = np.array([(e["y"], e["x"]) for e in A])
i0 = int(np.argmin(np.abs(off)))
n = len(M)
R = {"n_aligned_tiles": n, "n_segments": len(set(seg)), "n_scrolls": len(set(scroll))}
print("\naligned tiles=%d  segments=%d  scrolls=%d" % (n, len(set(seg)), len(set(scroll))))


def blocks(grp, yx):
    lab = np.full(len(yx), -1); nb = 0
    for s in sorted(set(grp)):
        left = {i for i in range(len(yx)) if grp[i] == s}
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
print("spatial blocks=%d  (mean %.1f tiles per block)" % (nb, n / nb))
R["n_blocks"] = int(nb)


def policy_oracle(group):
    """in-sample: each group's best offset, chosen on that group's own data."""
    a = np.empty(n)
    for g in set(group):
        m = group == g
        a[m] = M[m, int(np.argmax(M[m].mean(0)))]
    return a.mean()


def policy_cv(group):
    """honest: a tile's offset comes from the OTHER tiles of the same group."""
    a = np.empty(n)
    for i in range(n):
        k = (group == group[i]) & (np.arange(n) != i)
        a[i] = M[i, i0] if not k.any() else M[i, int(np.argmax(M[k].mean(0)))]
    return a.mean()


base = M[:, i0].mean()
glob = M[:, int(np.argmax(M.mean(0)))].mean()
print("\n" + "=" * 76)
print("ORACLE (in-sample) POLICIES - this is where the earlier analysis stopped")
print("=" * 76)
print("  canonical k=0              : %.5f" % base)
print("  single COMMON offset       : %.5f  (%+.5f)" % (glob, glob - base))
p_sc = policy_oracle(scroll); p_sg = policy_oracle(seg); p_bl = policy_oracle(blk)
print("  per SCROLL                 : %.5f  (%+.5f)   ADDED over common  = %+.5f"
      % (p_sc, p_sc - base, p_sc - glob))
print("  per SEGMENT                : %.5f  (%+.5f)   ADDED over scroll  = %+.5f"
      % (p_sg, p_sg - base, p_sg - p_sc))
print("  per BLOCK (spatial)        : %.5f  (%+.5f)   ADDED over segment = %+.5f"
      % (p_bl, p_bl - base, p_bl - p_sg))
print("  per-tile oracle            : %.5f  (%+.5f)"
      % (M.max(1).mean(), M.max(1).mean() - base))
add_scroll = p_sc - glob; add_seg = p_sg - p_sc
print("\n  --> earlier claim : segment ADDED=+0.0070, scroll ADDED=+0.0004, ratio 16.8x")
print("  --> this run      : segment ADDED=%+.5f, scroll ADDED=%+.5f, ratio %.1fx"
      % (add_seg, add_scroll, (add_seg / add_scroll) if add_scroll > 1e-9 else float("inf")))
R["oracle"] = dict(k0=float(base), common=float(glob), scroll=float(p_sc), segment=float(p_sg),
                   block=float(p_bl), tile=float(M.max(1).mean()),
                   added_scroll=float(add_scroll), added_segment=float(add_seg),
                   added_block=float(p_bl - p_sg),
                   ratio_segment_over_scroll=float(add_seg / add_scroll)
                   if add_scroll > 1e-9 else None)

print("\n" + "=" * 76)
print("THE SAME POLICIES UNDER HONEST CROSS-VALIDATION (not oracle)")
print("=" * 76)
c_sc = policy_cv(scroll); c_sg = policy_cv(seg); c_bl = policy_cv(blk)
c_gl = np.mean([M[i, int(np.argmax(M[np.arange(n) != i].mean(0)))] for i in range(n)])
print("  k=0=%.5f  common(CV)=%.5f(%+.5f)  scroll(CV)=%.5f(%+.5f)  segment(CV)=%.5f(%+.5f)  block(CV)=%.5f(%+.5f)"
      % (base, c_gl, c_gl - base, c_sc, c_sc - base, c_sg, c_sg - base, c_bl, c_bl - base))
R["cv"] = dict(k0=float(base), common=float(c_gl), scroll=float(c_sc),
               segment=float(c_sg), block=float(c_bl))

# out-of-block CV (spatially apart) - the real test of the per-segment claim
a = np.full(n, np.nan)
for i in range(n):
    k = (seg == seg[i]) & (blk != blk[i])
    if k.any():
        a[i] = M[i, int(np.argmax(M[k].mean(0)))]
ok = ~np.isnan(a)
print("\n  OUT-OF-BLOCK CV (offset from ANOTHER block of the same segment, n=%d tiles):" % ok.sum())
if ok.sum() > 10:
    d = a[ok] - M[ok, i0]
    print("     AUC=%.5f  k=0=%.5f  diff=%+.5f  median=%+.5f  better %d/%d  Wilcoxon p=%.4g"
          % (a[ok].mean(), M[ok, i0].mean(), d.mean(), np.median(d),
             int((d > 0).sum()), int(ok.sum()), stats.wilcoxon(d).pvalue))
    ai = np.array([M[i, int(np.argmax(M[(blk == blk[i]) & (np.arange(n) != i)].mean(0)))]
                   if ((blk == blk[i]) & (np.arange(n) != i)).any() else M[i, i0]
                   for i in np.where(ok)[0]])
    print("     WITHIN-BLOCK CV on the same tiles = %.5f (%+.5f)  -> neighbourhood contributes %+.5f"
          % (ai.mean(), ai.mean() - M[ok, i0].mean(), ai.mean() - a[ok].mean()))
    R["out_of_block_cv"] = dict(n=int(ok.sum()), auc=float(a[ok].mean()),
                                k0=float(M[ok, i0].mean()), diff=float(d.mean()),
                                median=float(np.median(d)), n_better=int((d > 0).sum()),
                                p=float(stats.wilcoxon(d).pvalue),
                                within_block=float(ai.mean()),
                                neighbourhood_contribution=float(ai.mean() - a[ok].mean()))


# variance components of the peak offsets: scroll / segment / block
def ppeak(c):
    k = int(np.argmax(c))
    if k in (0, len(c) - 1):
        return float(off[k])
    den = c[k - 1] - 2 * c[k] + c[k + 1]
    dd = 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (c[k - 1] - c[k + 1]) / den, -1, 1))
    return float(off[k] + dd * (off[1] - off[0]))


peak = np.array([ppeak(M[i]) for i in range(n)])


def icc(group, x):
    ks = sorted(set(group)); a = len(ks); N = len(x); gr = x.mean()
    SSB = sum((group == k).sum() * (x[group == k].mean() - gr) ** 2 for k in ks)
    SSW = sum(((x[group == k] - x[group == k].mean()) ** 2).sum() for k in ks)
    MSB, MSW = SSB / (a - 1), SSW / (N - a)
    ni = [int((group == k).sum()) for k in ks]
    n0 = (N - sum(v * v for v in ni) / N) / (a - 1)
    vb = max(0.0, (MSB - MSW) / n0)
    F = MSB / MSW
    return dict(a=a, sd_between=float(np.sqrt(vb)), sd_within=float(np.sqrt(MSW)),
                ICC=float(vb / (vb + MSW)) if vb + MSW > 0 else 0.0, F=float(F),
                p=float(1 - stats.f.cdf(F, a - 1, N - a)))


print("\n" + "=" * 76)
print("VARIANCE COMPONENTS OF THE PER-TILE PEAK OFFSET (n=%d tiles)" % n)
print("=" * 76)
for nm, g in (("SCROLL", scroll), ("SEGMENT", seg), ("BLOCK", blk)):
    v = icc(g, peak)
    print("  %-8s groups=%3d  sd_between=%6.2f um  sd_within=%6.2f um  ICC=%.4f  F=%.2f  p=%.4g"
          % (nm, v["a"], v["sd_between"], v["sd_within"], v["ICC"], v["F"], v["p"]))
    R["icc_" + nm] = v

json.dump(R, open(os.path.join(WORK, "all_tiles.json"), "w"), indent=1)
print("\nwritten -> " + os.path.join(WORK, "all_tiles.json"))
