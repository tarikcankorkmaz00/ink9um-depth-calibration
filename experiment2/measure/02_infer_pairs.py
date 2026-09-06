# -*- coding: utf-8 -*-
"""Step A2 - run the checkpoint over the locked tile list at nine z offsets.

Each tile is scored twice in each family: once against that family's own
label rasterisation and once against the other family's, resampled. Nine z
offsets on a common in-phase grid. Writes the per-tile curves that
experiment2/data/paired_tiles.json is built from.

usage: python 02_infer_pairs.py [aligned|native] [8|32]

This is a MEASUREMENT step: it needs a CUDA GPU, the villa package, the
ink_9um checkpoint and the label and volume chunks. See experiment2/README.md
for the layout of VZ_ROOT. Results go to VZ_WORK (default: <repo>/work).
The committed data in experiment2/data/ is what this produced; nothing in
experiment2/verify/ needs any of it.
"""
# A few identifiers below are in the author's own language, left as they ran:
#   tuval = canvas, yonga = tile, kayma = shift, faz = phase, secim = selection,
#   aday = candidate, kok = origin, cerceve = frame, cift = pair, olcum = measurement,
#   ortak = shared, kosu/kosum = run, sonuc = result, ham = raw, etiket = label.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_os.path.dirname(_HERE))
ROOT = _os.environ.get("VZ_ROOT", _os.path.join(_REPO, "cache"))
OUT = _os.environ.get("VZ_WORK", _os.path.join(_REPO, "work"))
_os.makedirs(OUT, exist_ok=True)
LABELS = _os.path.join(ROOT, "cache", "labels")
VOLUMES = _os.path.join(ROOT, "cache", "volumes")
LABEL_INDEX = _os.path.join(ROOT, "cache", "label_index.json")
SEGMENTS = _os.path.join(ROOT, "segments.json")
CHECKPOINT = _os.path.join(ROOT, "model", "ink_9um",
                           "hybrid_3d2d-seed42", "step-075000.pth")
VILLA_SRC = _os.path.join(ROOT, "villa", "vesuvius", "src")
TRANSFORM = _os.path.join(_os.path.dirname(_HERE), "data", "context.json")


def _manifest():
    """segments.json, tolerating either top-level key name."""
    m = json.load(open(SEGMENTS, encoding="utf-8"))
    return m.get("segments", m.get("segmentler"))

import json, os, sys, time
import numpy as np, torch, numcodecs
from scipy.ndimage import map_coordinates
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, VILLA_SRC)
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from vesuvius.image_proc.intensity.normalization import normalize_robust

CACHE = LABELS
VOL = VOLUMES
IDX = json.load(open(LABEL_INDEX))
SEGS = {s["key"]: s for s in _manifest()}
ANCHOR = sys.argv[1] if len(sys.argv) > 1 else "aligned"
ETIKET = sys.argv[2] if len(sys.argv) > 2 else ""
P = json.load(open("%s/yonga_listesi_%s%s.json" % (OUT, ANCHOR, ETIKET)))
_BL = numcodecs.Blosc()

D_A = [-16, -12, -8, -4, 0, 4, 8, 12, 16]
D_N = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
UM_A = [d * 2.399 for d in D_A]
UM_N = [d * 9.362 for d in D_N]

ck = torch.load(CHECKPOINT,
                map_location="cpu", weights_only=False)
WIN = int(ck["config"]["patch_size"][0])
cfg = InkConfig.from_mapping(ck["config"])
mm = make_model(cfg)
st = ck["model"]
if st and all(str(k).startswith("module.") for k in st):
    st = {k[len("module."):]: v for k, v in st.items()}
mm.load_state_dict(st, strict=False)
model = TargetModel(mm, input_pad_depth_to=cfg.model.input_pad_depth_to).eval().cuda()

def tuval(segkey, kind):
    s = SEGS[segkey]
    nz, H, W = s["label_shape_zyx"]
    zc = s["annotation_center_channel"]
    out = np.zeros((H, W), np.uint8)
    for yx, h in IDX.get(segkey + "|" + kind, {}).items():
        fp = os.path.join(CACHE, h[:2], h)
        if not (os.path.exists(fp) and os.path.getsize(fp)):
            continue
        a = np.frombuffer(_BL.decode(open(fp, "rb").read()),
                          np.uint8).reshape(nz, 128, 128)
        y, x = (int(v) for v in yx.split(","))
        y0, x0 = y * 128, x * 128
        sl = (a[zc] > 0).astype(np.uint8)
        h_, w_ = min(128, H - y0), min(128, W - x0)
        out[y0:y0 + h_, x0:x0 + w_] = sl[:h_, :w_]
    return out

def chunk(d, iy, ix, nz):
    fp = "%s/c_0_%d_%d" % (d, iy, ix)
    if not (os.path.exists(fp) and os.path.getsize(fp) == nz * 128 * 128):
        return None
    return np.fromfile(fp, np.uint8).reshape(nz, 128, 128)

def pencere(d, y0, x0, nz, H, W):
    out = np.zeros((nz, 128, 128), np.uint8)
    for iy in range(y0 // 128, (y0 + 127) // 128 + 1):
        for ix in range(x0 // 128, (x0 + 127) // 128 + 1):
            if iy < 0 or ix < 0 or iy * 128 >= H or ix * 128 >= W:
                return None
            c = chunk(d, iy, ix, nz)
            if c is None:
                return None
            gy0, gx0 = iy * 128, ix * 128
            ay0, ay1 = max(y0, gy0), min(y0 + 128, gy0 + 128)
            ax0, ax1 = max(x0, gx0), min(x0 + 128, gx0 + 128)
            if ay1 <= ay0 or ax1 <= ax0:
                continue
            out[:, ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = \
                c[:, ay0 - gy0:ay1 - gy0, ax0 - gx0:ax1 - gx0]
    return out

def girdi_aligned(raw, s, d):
    z0, z1 = s["source_z_slice"]
    pool = (z1 - z0) // s["label_shape_zyx"][0]
    start = z0 + pool * (s["annotation_center_channel"] - WIN // 2) + d
    if start < 0 or start + pool * WIN > raw.shape[0]:
        return None
    b = raw[start:start + pool * WIN].astype(np.float32)
    return np.rint(b.reshape(WIN, pool, 128, 128).mean(axis=1))

def girdi_native(raw, s, d):
    start = s["annotation_center_channel"] - WIN // 2 + d
    if start < 0 or start + WIN > raw.shape[0]:
        return None
    return raw[start:start + WIN].astype(np.float32)

def kos(batch):
    xb = torch.from_numpy(np.stack(batch)).unsqueeze(1).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        lg = model(xb)
    return lg.float().squeeze(1).cpu().numpy()

def m2(y, s):
    return (round(float(roc_auc_score(y, s)), 6),
            round(float(average_precision_score(y, s)), 6))

kayitlar = []
t0 = time.time()
for phys, d in P["segmentler"].items():
    ak, nk = d["aligned_key"], d["native_key"]
    sa, sn = SEGS[ak], SEGS[nk]
    s_, dy, dx = d["olcek"], d["dy"], d["dx"]
    A_ink, A_sup = tuval(ak, "ink"), tuval(ak, "sup")
    N_ink, N_sup = tuval(nk, "ink"), tuval(nk, "sup")
    Ha, Wa = A_ink.shape
    Hn, Wn = N_ink.shape
    da, dn = VOL + "/ali_" + phys, VOL + "/" + phys
    nok = 0
    for c in d["yongalar"]:
        y0, x0, ny0, nx0 = c["y0"], c["x0"], c["ny0"], c["nx0"]
        ay0, ay1, ax0, ax1 = c["roi"]
        rawA = pencere(da, y0, x0, 109, Ha, Wa)
        rawN = pencere(dn, ny0, nx0, 28, Hn, Wn)
        if rawA is None or rawN is None:
            nok += 1
            continue
        bA = [girdi_aligned(rawA, sa, k) for k in D_A]
        bN = [girdi_native(rawN, sn, k) for k in D_N]
        if any(v is None for v in bA + bN):
            nok += 1
            continue
        lgA = kos([normalize_robust(v) for v in bA])
        lgN = kos([normalize_robust(v) for v in bN])

        # --- frame A: aligned grid + aligned label, on the shared ROI
        supA = A_sup[ay0:ay1, ax0:ax1] > 0
        gy, gx = np.mgrid[ay0:ay1, ax0:ax1]
        cyN = (gy - dy) / s_ - ny0
        cxN = (gx - dx) / s_ - nx0
        okA = supA & (cyN >= 0) & (cyN <= 127) & (cxN >= 0) & (cxN <= 127)
        yA = (A_ink[ay0:ay1, ax0:ax1] > 0)[okA].astype(np.uint8)

        # --- frame N: native grid + native label, on the SAME physical ROI
        ny, nx = np.mgrid[ny0:ny0 + 128, nx0:nx0 + 128]
        cyA = ny * s_ + dy - y0
        cxA = nx * s_ + dx - x0
        okN = ((N_sup[ny0:ny0 + 128, nx0:nx0 + 128] > 0) &
               (cyA >= ay0 - y0) & (cyA <= ay1 - 1 - y0) &
               (cxA >= ax0 - x0) & (cxA <= ax1 - 1 - x0))
        yN = (N_ink[ny0:ny0 + 128, nx0:nx0 + 128] > 0)[okN].astype(np.uint8)

        if (yA.sum() < 60 or (1 - yA).sum() < 60 or
                yN.sum() < 60 or (1 - yN).sum() < 60):
            nok += 1
            continue

        crd_N = np.vstack([cyN[okA], cxN[okA]])
        crd_A = np.vstack([cyA[okN], cxA[okN]])
        rec = dict(phys=phys, y0=y0, x0=x0, ny0=ny0, nx0=nx0,
                   roi=[ay0, ay1, ax0, ax1],
                   n_pix_A=int(okA.sum()), n_ink_A=int(yA.sum()),
                   n_pix_N=int(okN.sum()), n_ink_N=int(yN.sum()))
        lblN_onA = map_coordinates(
            (N_ink[ny0:ny0 + 128, nx0:nx0 + 128] > 0).astype(np.float32),
            crd_N, order=1, mode="nearest") > 0.5
        u = yA.astype(bool)
        rec["dice_etiket"] = round(float(2 * (u & lblN_onA).sum() /
                                         max(1, u.sum() + lblN_onA.sum())), 4)
        rgA = np.random.default_rng((hash((phys, y0, x0)) ^ 0xA1) % (2 ** 32))
        rgN = np.random.default_rng((hash((phys, y0, x0)) ^ 0xB2) % (2 ** 32))
        hA = rgA.random(int(okA.sum())) < 0.5
        hN = rgN.random(int(okN.sum())) < 0.5
        yarim = {"A": (hA, ~hA), "N": (hN, ~hN)}
        for ad in ("A_own", "N_res", "N_own", "A_res"):
            cer = "A" if ad in ("A_own", "N_res") else "N"
            roc, pr, r1, r2 = [], [], [], []
            for k in range(9):
                if ad == "A_own":
                    sc = lgA[k][ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0][okA]
                    y = yA
                elif ad == "N_res":
                    sc = map_coordinates(lgN[k], crd_N, order=1, mode="nearest")
                    y = yA
                elif ad == "N_own":
                    sc = lgN[k][okN]
                    y = yN
                else:
                    sc = map_coordinates(lgA[k], crd_A, order=1, mode="nearest")
                    y = yN
                a, b = m2(y, sc)
                roc.append(a)
                pr.append(b)
                for h, dst in zip(yarim[cer], (r1, r2)):
                    yy = y[h]
                    dst.append(round(float(roc_auc_score(yy, sc[h])), 6)
                               if yy.sum() >= 10 and (1 - yy).sum() >= 10 else None)
            rec["roc_" + ad] = roc
            rec["pr_" + ad] = pr
            rec["roc_h1_" + ad] = r1
            rec["roc_h2_" + ad] = r2
        kayitlar.append(rec)
    print("%s: %d pairs, %d skipped  (%.0fs)"
          % (phys, len([r for r in kayitlar if r["phys"] == phys]), nok,
             time.time() - t0), flush=True)

json.dump(dict(protokol="ortak 9 nokta faz-ici z + patch faz kilidi",
               anchor=ANCHOR, etiket=ETIKET, faz=P.get("faz"),
               D_A=D_A, D_N=D_N, um_A=UM_A, um_N=UM_N, WIN=WIN,
               checkpoint="ink_9um/hybrid_3d2d-seed42/step-075000.pth",
               n=len(kayitlar), kayitlar=kayitlar),
          open("%s/ham_yonga_metrikleri_%s%s.json" % (OUT, ANCHOR, ETIKET), "w"))
print("\n%d pairs -> %s/ham_yonga_metrikleri_%s%s.json  (%.0fs)"
      % (len(kayitlar), OUT, ANCHOR, ETIKET, time.time() - t0))
