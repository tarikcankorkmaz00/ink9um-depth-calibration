# -*- coding: utf-8 -*-
"""Step B1 - the patch-phase scan.

Hold the pixels and the label fixed, move the 128x128 window origin, and
score the same region again. This is the measurement that produced
experiment2/data/phase.json's shift_1d section. It also prints the
checkpoint's own autoconfigure block, which is where the
must_be_divisible_by = [32, 32] in that file comes from.

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

import json, os, sys
import numpy as np, torch, numcodecs
from sklearn.metrics import roc_auc_score

sys.path.insert(0, VILLA_SRC)
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from vesuvius.image_proc.intensity.normalization import normalize_robust

CACHE = LABELS
VOL = VOLUMES
IDX = json.load(open(LABEL_INDEX))
SEGS = {s["key"]: s for s in _manifest()}
_BL = numcodecs.Blosc()

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

SHIFTS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]
MRG = 32   # half-width of the shared scoring region, at [32,96) inside the patch
# the window moves in +y only. The scored region is fixed in GLOBAL coordinates
# at [y0+32, y0+96), so the same physical pixels and the same label are used at
# every shift. Shifts stay in [0,32] so the region stays inside the patch;
# 40, 48, 64, 96 and 128 are reported separately, where it nears the patch edge.

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

PH = ("w035", "w039", "w040", "w041", "w044")
tablo = {}
for fam in ("native9", "aligned"):
    if fam == "native9":
        keys = {k: "native9-scrollprizeorg-21slices/" + k for k in PH}
        vold = {k: VOL + "/" + k for k in PH}
        nzv = 28
    else:
        keys = {k: "aligned-scrollprizeorg-21slices/" + v for k, v in
                (("w035", "pherc0139-w035"), ("w039", "pherc0139-w039"),
                 ("w040", "pherc0139-w040"), ("w041", "pherc0139-w041"),
                 ("w044", "pherc0139-w028"))}
        vold = {k: VOL + "/ali_" + k for k in PH}
        nzv = 109
    per = {s: [] for s in SHIFTS}
    for ph in PH:
        key = keys[ph]
        s = SEGS[key]
        H, W = s["label_shape_zyx"][1:]
        ink, sup = tuval(key, "ink"), tuval(key, "sup")
        aday = []
        for yx in IDX.get(key + "|sup", {}):
            iy, ix = (int(v) for v in yx.split(","))
            y0, x0 = iy * 128, ix * 128
            if y0 + 128 + max(SHIFTS) > H or x0 + 128 > W:
                continue
            m = sup[y0 + MRG:y0 + 96, x0 + MRG:x0 + 96] > 0
            if m.sum() < 400:
                continue
            i = (ink[y0 + MRG:y0 + 96, x0 + MRG:x0 + 96] > 0)[m]
            if i.sum() < 50 or (~i).sum() < 50:
                continue
            aday.append((y0, x0))
        aday = sorted(aday)
        if len(aday) > 20:
            k = np.linspace(0, len(aday) - 1, 20).round().astype(int)
            aday = [aday[i] for i in sorted(set(k.tolist()))]
        for y0, x0 in aday:
            m = sup[y0 + MRG:y0 + 96, x0 + MRG:x0 + 96] > 0
            yl = (ink[y0 + MRG:y0 + 96, x0 + MRG:x0 + 96] > 0)[m].astype(np.uint8)
            tmp, ok = {}, True
            for a in SHIFTS:
                raw = pencere(vold[ph], y0 + a, x0, nzv, H, W)
                if raw is None:
                    ok = False
                    break
                if fam == "native9":
                    st_ = s["annotation_center_channel"] - WIN // 2
                    g = raw[st_:st_ + WIN].astype(np.float32)
                else:
                    z0, z1 = s["source_z_slice"]
                    pool = (z1 - z0) // s["label_shape_zyx"][0]
                    st_ = z0 + pool * (s["annotation_center_channel"] - WIN // 2)
                    g = np.rint(raw[st_:st_ + pool * WIN].astype(np.float32)
                                .reshape(WIN, pool, 128, 128).mean(axis=1))
                xb = torch.from_numpy(normalize_robust(g)[None]).unsqueeze(1).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    lg = model(xb)
                lg = lg.float().squeeze().cpu().numpy()
                sy = MRG - a
                if sy < 0 or sy + 64 > 128:
                    tmp[a] = None
                    continue
                tmp[a] = float(roc_auc_score(yl, lg[sy:sy + 64, MRG:MRG + 64][m]))
            if ok:
                for k_, v in tmp.items():
                    per[k_].append(v)
    n = len(per[0])
    print("\n%s  n=%d tiles, shared 64x64 region, patch moves in +y only" % (fam, n))
    print("   %5s %9s %9s %9s %9s   %s" %
          ("kayma", "ort AUC", "medyan", "fark(ort)", "fark(med)", "egitim izgarasi (32'nin kati)"))
    base = np.array(per[0])
    tablo[fam] = {"n": n, "kayma": {}}
    for a in SHIFTS:
        v = per[a]
        if any(x is None for x in v):
            print("   %+5d  (shared region fell outside the patch - skipped)" % a)
            continue
        v = np.array(v)
        print("   %+5d %9.5f %9.5f %+9.5f %+9.5f   %s"
              % (a, v.mean(), np.median(v), (v - base).mean(),
                 np.median(v - base), "EVET" if a % 32 == 0 else "hayir"))
        tablo[fam]["kayma"][a] = dict(auc=float(v.mean()), medyan=float(np.median(v)),
                                      fark=float((v - base).mean()),
                                      fark_medyan=float(np.median(v - base)),
                                      izgarada=bool(a % 32 == 0))
json.dump(tablo, open(OUT + "/kontrol_kayma_taramasi.json", "w"), indent=1)
print("\n-> " + OUT + "/kontrol_kayma_taramasi.json")
