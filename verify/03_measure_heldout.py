# -*- coding: utf-8 -*-
"""Step 3 - the measurement: sweep the depth of the 17-slice window on the held-out tiles.

Two things are deliberately not hard-coded:
  - the window position is DERIVED from the label metadata (source_z_slice, the label
    depth and annotation_center_channel), not written in as a magic number;
  - the input depth comes from the checkpoint config (patch_size[0]), not from memory.

AUC is sklearn.metrics.roc_auc_score, cross-checked against scipy's Mann-Whitney U
divided by n1*n0. Average precision is recorded too, so later steps can ask whether
the conclusion depends on the metric.

Needs a CUDA GPU, the villa package, the checkpoint and the cached chunks.
"""
import json, os, sys, time
import numpy as np, torch, numcodecs
from sklearn.metrics import roc_auc_score, average_precision_score
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
CKPT = os.path.join(ROOT, "model", "ink_9um", "hybrid_3d2d-seed42", "step-075000.pth")
IDX = json.load(open(os.path.join(WORK, "label_index.json")))
SEL = json.load(open(os.path.join(WORK, "tile_selection.json")))
SEGS = {s["key"]: s for s in json.load(
    open(os.path.join(ROOT, "segments.json"), encoding="utf-8"))["segments"]}
_BL = numcodecs.Blosc()
_c = {}


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


# ---------------------------------------------------------------- MODEL
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_map = ck["config"]
WIN = int(cfg_map["patch_size"][0])
NORM = cfg_map["image_normalization"]["mode"]
assert NORM == "robust_mad", NORM
print(f"checkpoint: patch_size={cfg_map['patch_size']}  normalization={NORM}")
print(f"z-jitter used in training: {cfg_map.get('flat_z_window_jitter')}")
print(f"online_validation_cases: {cfg_map['data_provenance'].get('online_validation_cases')}")
cfg = InkConfig.from_mapping(cfg_map)
m = make_model(cfg)
st = ck["model"]
if st and all(str(k).startswith("module.") for k in st):
    st = {k[len("module."):]: v for k, v in st.items()}
inc = m.load_state_dict(st, strict=False)
print(f"state_dict loaded: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")
assert len(inc.missing_keys) == 0, inc.missing_keys[:5]
model = TargetModel(m, input_pad_depth_to=cfg.model.input_pad_depth_to).eval().cuda()


# ---------------------------------------------------------------- BUILDING THE INPUT
def window_start(s, d):
    """Source-slice start of the canonical window, derived from metadata."""
    z0, z1 = s["source_z_slice"]                    # label .zattrs, e.g. [13, 97]
    nlab = s["label_shape_zyx"][0]                  # 21
    pool = (z1 - z0) // nlab                        # 4
    assert pool * nlab == (z1 - z0), (z0, z1, nlab)
    zc = s["annotation_center_channel"]             # 10
    assert (nlab - WIN) % 2 == 0
    return z0 + pool * (zc - WIN // 2) + d, pool


def make_input(raw, s, d):
    start, pool = window_start(s, d)
    if start < 0 or start + pool * WIN > raw.shape[0]:
        return None
    b = raw[start:start + pool * WIN].astype(np.float32)
    return np.rint(b.reshape(WIN, pool, 128, 128).mean(axis=1))


def auc_sklearn(score, y):
    return float(roc_auc_score(y.astype(np.uint8), score))


def auc_mannwhitney(score, y):
    a = score[y]; b = score[~y]
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


# ---------------------------------------------------------------- SWEEP
D = list(range(-16, 17, 2))            # 17 points, step 2 source slices = 4.798 um
out, t0 = [], time.time()
for segkey, cands in SEL.items():
    s = SEGS[segkey]
    nz = s["label_chunk_zyx"][0]; zc = s["annotation_center_channel"]
    um_src = float(s["volume_base_voxel_um"][0])
    for a in cands:
        if not a["eligible"]:
            continue
        y, x = a["y"], a["x"]
        fp = os.path.join(VOLD, segkey.replace("/", "__") + f"__{y}_{x}.npy")
        if not os.path.exists(fp):
            print("VOLUME MISSING", segkey, y, x)
            continue
        raw = np.load(fp)
        assert raw.shape == (s["volume_level_chunk_zyx"][0], 128, 128), raw.shape
        ink = label(segkey, "ink", y, x, nz)[zc] > 0
        vm = label(segkey, "val", y, x, nz)[zc] > 0
        yl = ink[vm]
        if yl.sum() < 60 or (~yl).sum() < 60:
            continue
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
        aucs = [auc_sklearn(lg[i][vm], yl) for i in range(len(keep))]
        mws = [auc_mannwhitney(lg[i][vm], yl) for i in range(len(keep))]
        aps = [float(average_precision_score(yl.astype(np.uint8), lg[i][vm]))
               for i in range(len(keep))]
        out.append(dict(segkey=segkey, scroll=s["scroll"].split(" ")[0], seg=s["segment"],
                        y=y, x=x, n_ink=int(yl.sum()), n_bg=int((~yl).sum()),
                        d=keep, um=[d * um_src for d in keep],
                        auc=[round(v, 6) for v in aucs],
                        auc_mw=[round(v, 6) for v in mws],
                        ap=[round(v, 6) for v in aps]))
        print(f"  {s['segment']:16s} y={y:3d} x={x:3d}  n_ink={int(yl.sum()):6d} "
              f"n_bg={int((~yl).sum()):6d}  AUC@0={aucs[keep.index(0)]:.5f} "
              f"max={max(aucs):.5f}@{keep[int(np.argmax(aucs))]*um_src:+.1f}um "
              f"|sklearn-MW|max={max(abs(p - q) for p, q in zip(aucs, mws)):.2e}", flush=True)

json.dump(dict(win=WIN, D=D, n_tiles=len(out), tiles=out),
          open(os.path.join(WORK, "heldout_scan.json"), "w"), indent=1)
print(f"\n{len(out)} tiles measured, {time.time()-t0:.0f}s -> "
      f"{os.path.join(WORK, 'heldout_scan.json')}")
