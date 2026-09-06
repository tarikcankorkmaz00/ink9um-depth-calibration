# -*- coding: utf-8 -*-
"""Step 8 - the deployment test.

In practice what you hold is the labelled SUPERVISION region of a segment; the region
you want to predict is a different one. Step 2 verified that the two do not share a
single pixel. So the honest question is not "does a per-segment offset help when it is
fitted on tiles next to the ones it is scored on", it is:

    estimate the offset on the supervision region, apply it to the held-out region.

This step sweeps the supervision tiles (with validation pixels excluded from the
scoring mask), takes each segment's supervision peak, and applies it to that segment's
held-out tiles.

Needs a CUDA GPU, the villa package, the checkpoint and the cached chunks.
"""
import json, os, sys, time
import numpy as np, torch, numcodecs
from sklearn.metrics import roc_auc_score
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
ROOT = os.environ.get("VZ_ROOT", os.path.join(REPO, "data"))

sys.path.insert(0, os.path.join(ROOT, "villa", "vesuvius", "src"))
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from vesuvius.image_proc.intensity.normalization import normalize_robust

CACHE = os.path.join(ROOT, "cache", "labels")
VOLD = os.path.join(ROOT, "cache", "volumes")
IDX = json.load(open(os.path.join(WORK, "label_index.json")))
SEGS = {s["key"]: s for s in json.load(
    open(os.path.join(ROOT, "segments.json"), encoding="utf-8"))["segments"]}
_BL = numcodecs.Blosc(); _c = {}


def blob(h, nz):
    if h in _c:
        return _c[h]
    fp = os.path.join(CACHE, h[:2], h)
    a = (np.zeros((nz, 128, 128), np.uint8) if not (os.path.exists(fp) and os.path.getsize(fp))
         else np.frombuffer(_BL.decode(open(fp, "rb").read()),
                            np.uint8).reshape(nz, 128, 128).copy())
    if len(_c) < 3000:
        _c[h] = a
    return a


def label(segkey, kind, y, x, nz):
    h = IDX["index"].get(f"{segkey}|{kind}", {}).get(f"{y},{x}")
    return np.zeros((nz, 128, 128), np.uint8) if h is None else blob(h, nz)


ck = torch.load(os.path.join(ROOT, "model", "ink_9um", "hybrid_3d2d-seed42",
                             "step-075000.pth"), map_location="cpu", weights_only=False)
WIN = int(ck["config"]["patch_size"][0])
cfg = InkConfig.from_mapping(ck["config"])
m = make_model(cfg)
st = ck["model"]
if st and all(str(k).startswith("module.") for k in st):
    st = {k[len("module."):]: v for k, v in st.items()}
m.load_state_dict(st, strict=False)
model = TargetModel(m, input_pad_depth_to=cfg.model.input_pad_depth_to).eval().cuda()


def make_input(raw, s, d):
    z0, z1 = s["source_z_slice"]
    nlab = s["label_shape_zyx"][0]
    pool = (z1 - z0) // nlab
    start = z0 + pool * (s["annotation_center_channel"] - WIN // 2) + d
    if start < 0 or start + pool * WIN > raw.shape[0]:
        return None
    b = raw[start:start + pool * WIN].astype(np.float32)
    return np.rint(b.reshape(WIN, pool, 128, 128).mean(axis=1))


D = list(range(-16, 17, 2))
TARGET = ["aligned-scrollprizeorg-21slices/pherc0139-w016",
          "aligned-scrollprizeorg-21slices/pherc0814-46527",
          "aligned-scrollprizeorg-21slices/pherc1667-w029"]

sup_out = []
t0 = time.time()
print("measuring the SUPERVISION tiles (validation pixels are excluded)...")
for segkey in TARGET:
    s = SEGS[segkey]
    nz = s["label_chunk_zyx"][0]; zc = s["annotation_center_channel"]
    um = float(s["volume_base_voxel_um"][0])
    count = 0
    for yx in IDX["index"][f"{segkey}|sup"]:
        y, x = (int(v) for v in yx.split(","))
        fp = os.path.join(VOLD, segkey.replace("/", "__") + f"__{y}_{x}.npy")
        if not os.path.exists(fp):
            continue
        sup = label(segkey, "sup", y, x, nz)[zc] > 0
        vm = label(segkey, "val", y, x, nz)[zc] > 0
        mask = sup & (~vm)                      # validation pixels strictly excluded
        if mask.sum() < 500:
            continue
        ink = label(segkey, "ink", y, x, nz)[zc] > 0
        yl = ink[mask]
        if yl.sum() < 60 or (~yl).sum() < 60:
            continue
        raw = np.load(fp)
        batch, keep = [], []
        for d in D:
            g = make_input(raw, s, d)
            if g is None:
                continue
            batch.append(normalize_robust(g)); keep.append(d)
        xb = torch.from_numpy(np.stack(batch)).unsqueeze(1).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg = model(xb)
        lg = lg.float().squeeze(1).cpu().numpy()
        aucs = [float(roc_auc_score(yl.astype(np.uint8), lg[i][mask])) for i in range(len(keep))]
        sup_out.append(dict(seg=s["segment"], y=y, x=x, n_ink=int(yl.sum()),
                            n_bg=int((~yl).sum()), um=[d * um for d in keep],
                            auc=[round(v, 6) for v in aucs]))
        count += 1
    print(f"  {s['segment']:16s} supervision tiles measured = {count}")
print(f"{len(sup_out)} supervision tiles in total, {time.time()-t0:.0f}s")
json.dump(sup_out, open(os.path.join(WORK, "supervision_scan.json"), "w"), indent=1)

# ---------------------------------------------------------------- TRANSFER TEST
VAL = json.load(open(os.path.join(WORK, "heldout_scan.json")))["tiles"]
off = np.array(VAL[0]["um"], float)
Mv = np.stack([np.array(e["auc"], float) for e in VAL])
segv = np.array([e["seg"] for e in VAL])
i0 = int(np.argmin(np.abs(off)))
Ms = np.stack([np.array(e["auc"], float) for e in sup_out])
segs_ = np.array([e["seg"] for e in sup_out])

print("\n" + "=" * 76)
print("OFFSET ESTIMATED ON SUPERVISION -> APPLIED TO THE HELD-OUT REGION")
print("=" * 76)
R = {"n_sup_tiles": len(sup_out), "per_segment": {}}
a_tr = np.empty(len(Mv))
for sg in sorted(set(segv)):
    c = Ms[segs_ == sg].mean(0)
    j = int(np.argmax(c))
    mv = segv == sg
    a_tr[mv] = Mv[mv, j]
    print("  %-16s sup n=%2d  peak of the supervision curve = %+6.1f um  "
          "(sup AUC@0=%.5f peak=%.5f)"
          % (sg, int((segs_ == sg).sum()), off[j], c[i0], c[j]))
    print("      -> applied to the held-out region: AUC %.5f   (k=0 gives %.5f)   diff %+.5f"
          % (Mv[mv, j].mean(), Mv[mv, i0].mean(), Mv[mv, j].mean() - Mv[mv, i0].mean()))
    R["per_segment"][str(sg)] = dict(n_sup=int((segs_ == sg).sum()), sup_peak_um=float(off[j]),
                                     sup_auc_k0=float(c[i0]), sup_auc_peak=float(c[j]),
                                     val_auc_transfer=float(Mv[mv, j].mean()),
                                     val_auc_k0=float(Mv[mv, i0].mean()),
                                     diff=float(Mv[mv, j].mean() - Mv[mv, i0].mean()))
d = a_tr - Mv[:, i0]
w = stats.wilcoxon(d)
print("\n  OVERALL: transfer AUC=%.5f  k=0 AUC=%.5f  diff=%+.5f  median=%+.5f  "
      "better on %d/%d  Wilcoxon p=%.4g"
      % (a_tr.mean(), Mv[:, i0].mean(), d.mean(), np.median(d),
         int((d > 0).sum()), len(d), w.pvalue))
R["overall"] = dict(transfer_auc=float(a_tr.mean()), k0_auc=float(Mv[:, i0].mean()),
                    diff=float(d.mean()), median=float(np.median(d)),
                    n_better=int((d > 0).sum()), n=len(d), wilcoxon_p=float(w.pvalue))

# how far apart are the supervision peak and the held-out peak?
print("\n  SUPERVISION peak vs HELD-OUT peak (same segment):")
for sg in sorted(set(segv)):
    ts = off[int(np.argmax(Ms[segs_ == sg].mean(0)))]
    tv = off[int(np.argmax(Mv[segv == sg].mean(0)))]
    print("    %-16s sup=%+6.1f um   held-out=%+6.1f um   DIFFERENCE=%+6.1f um" % (sg, ts, tv, ts - tv))
    R["per_segment"][str(sg)]["val_peak_um"] = float(tv)
    R["per_segment"][str(sg)]["sup_minus_val_peak_um"] = float(ts - tv)

json.dump(R, open(os.path.join(WORK, "transfer.json"), "w"), indent=1)
print("\nwritten -> " + os.path.join(WORK, "transfer.json"))
