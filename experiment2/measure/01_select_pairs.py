# -*- coding: utf-8 -*-
"""Step A1 - build the spatially matched, phase-locked tile list.

For every one of the five physical PHerc0139 segments that is published in
both representations, walk the label canvases and keep the 128x128 positions
where BOTH families have enough supervision, enough ink and enough
background, and where both window origins land on the requested phase grid
(8 or 32). The mapping between the two families comes from a transform that
was fitted to the label geometry and frozen before any AUC was computed.

usage: python 01_select_pairs.py [aligned|native] [8|32]
The tile list this writes is then locked and never edited again.

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
import numpy as np
import numcodecs

CACHE = LABELS
IDX = json.load(open(LABEL_INDEX))
SEGS = {s["key"]: s for s in _manifest()}
T = {p["physical_segment"]: {"olcek": p["frozen_transform"]["scale"],
                             "dy_px": p["frozen_transform"]["dy_px"],
                             "dx_px": p["frozen_transform"]["dx_px"],
                             "toplam_olcek": p["frozen_transform"]["total_scale"]}
     for p in json.load(open(TRANSFORM))["pairing"]}
_BL = numcodecs.Blosc()

PAIR = {"w035": ("pherc0139-w035", "w035"),
        "w039": ("pherc0139-w039", "w039"),
        "w040": ("pherc0139-w040", "w040"),
        "w041": ("pherc0139-w041", "w041"),
        "w044": ("pherc0139-w028", "w044")}
AF = "aligned-scrollprizeorg-21slices/"
NF = "native9-scrollprizeorg-21slices/"
MIN_SUP, MIN_INK, MIN_BG = 500, 60, 60   # same filter as experiment 1, unchanged
FAZ = int(sys.argv[2]) if len(sys.argv) > 2 else 8   # measured phase period (8)
# FAZ=32: both families land on the training patch grid (stride_xy = 32)
ANCHOR = sys.argv[1] if len(sys.argv) > 1 else "aligned"
ETIKET = "" if (int(sys.argv[2]) if len(sys.argv) > 2 else 8) == 8 else "_faz%s" % sys.argv[2]
assert ANCHOR in ("aligned", "native")

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

def snap(v):
    return int(round(v / FAZ) * FAZ)

plan = {}
for phys, (ali, nat) in PAIR.items():
    ak, nk = AF + ali, NF + nat
    t = T[phys]
    s_, dy, dx = t["toplam_olcek"], t["dy_px"], t["dx_px"]
    A_ink, A_sup = tuval(ak, "ink"), tuval(ak, "sup")
    N_ink, N_sup = tuval(nk, "ink"), tuval(nk, "sup")
    Ha, Wa = A_ink.shape
    Hn, Wn = N_ink.shape
    aday, red = [], {"disi": 0, "ali_sup": 0, "ali_ink": 0,
                     "nat_sup": 0, "nat_ink": 0, "kirpik": 0, "roi_kucuk": 0}

    if ANCHOR == "aligned":
        kok = [(iy * 128, ix * 128) for iy in range(Ha // 128)
               for ix in range(Wa // 128)]
    else:
        kok = []
        for yx in IDX.get(nk + "|sup", {}):
            iy, ix = (int(v) for v in yx.split(","))
            if (iy + 1) * 128 <= Hn and (ix + 1) * 128 <= Wn:
                kok.append((iy * 128, ix * 128))
        kok = sorted(kok)

    for a, b in kok:
        if ANCHOR == "aligned":
            y0, x0 = a, b
            if y0 + 128 > Ha or x0 + 128 > Wa:
                red["kirpik"] += 1
                continue
            ny0 = snap((y0 + 64.0 - dy) / s_ - 64.0)
            nx0 = snap((x0 + 64.0 - dx) / s_ - 64.0)
        else:
            ny0, nx0 = a, b
            y0 = snap((ny0 + 64.0) * s_ + dy - 64.0)
            x0 = snap((nx0 + 64.0) * s_ + dx - 64.0)
        if (ny0 < 0 or nx0 < 0 or ny0 + 128 > Hn or nx0 + 128 > Wn or
                y0 < 0 or x0 < 0 or y0 + 128 > Ha or x0 + 128 > Wa):
            red["disi"] += 1
            continue
        assert y0 % FAZ == 0 and x0 % FAZ == 0 and ny0 % FAZ == 0 and nx0 % FAZ == 0
        # shared physical ROI, in the aligned frame: intersection of the two windows
        ay0 = max(y0, int(np.ceil(ny0 * s_ + dy)))
        ay1 = min(y0 + 128, int(np.floor((ny0 + 128) * s_ + dy)))
        ax0 = max(x0, int(np.ceil(nx0 * s_ + dx)))
        ax1 = min(x0 + 128, int(np.floor((nx0 + 128) * s_ + dx)))
        if ay1 - ay0 < 96 or ax1 - ax0 < 96:
            red["roi_kucuk"] += 1
            continue
        sup = A_sup[ay0:ay1, ax0:ax1] > 0
        if sup.sum() < MIN_SUP:
            red["ali_sup"] += 1
            continue
        ink = (A_ink[ay0:ay1, ax0:ax1] > 0)[sup]
        if ink.sum() < MIN_INK or (~ink).sum() < MIN_BG:
            red["ali_ink"] += 1
            continue
        # the same physical ROI, expressed in native coordinates
        gy, gx = np.mgrid[ny0:ny0 + 128, nx0:nx0 + 128]
        cy = gy * s_ + dy
        cx = gx * s_ + dx
        ic = (cy >= ay0) & (cy <= ay1 - 1) & (cx >= ax0) & (cx <= ax1 - 1)
        nsup = (N_sup[ny0:ny0 + 128, nx0:nx0 + 128] > 0) & ic
        if nsup.sum() < MIN_SUP:
            red["nat_sup"] += 1
            continue
        nink = (N_ink[ny0:ny0 + 128, nx0:nx0 + 128] > 0)[nsup]
        if nink.sum() < MIN_INK or (~nink).sum() < MIN_BG:
            red["nat_ink"] += 1
            continue
        aday.append(dict(y0=y0, x0=x0, ny0=ny0, nx0=nx0, roi=[ay0, ay1, ax0, ax1],
                         a_sup=int(sup.sum()), a_ink=int(ink.sum()),
                         n_sup=int(nsup.sum()), n_ink=int(nink.sum())))
    ach, nch = set(), set()
    for c in aday:
        for yy in (c["y0"], c["y0"] + 127):
            for xx in (c["x0"], c["x0"] + 127):
                ach.add((yy // 128, xx // 128))
        for yy in (c["ny0"], c["ny0"] + 127):
            for xx in (c["nx0"], c["nx0"] + 127):
                nch.add((yy // 128, xx // 128))
    plan[phys] = dict(aligned_key=ak, native_key=nk, olcek=s_, dy=dy, dx=dx,
                      aligned_shape=[Ha, Wa], native_shape=[Hn, Wn],
                      n_secilen=len(aday), red=red, yongalar=aday,
                      aligned_chunks=sorted(ach), native_chunks=sorted(nch))
    print("%s: kept=%4d  aligned_chunk=%4d  native_chunk=%4d  rejected=%s"
          % (phys, len(aday), len(ach), len(nch), red))

json.dump(dict(protokol="TASARIM.md 4.2 + faz kisiti (mod 8), yalniz geometri",
               anchor=ANCHOR, faz=FAZ,
               filtre=dict(min_sup=MIN_SUP, min_ink=MIN_INK, min_bg=MIN_BG,
                           min_roi=96),
               segmentler=plan),
          open("%s/yonga_listesi_%s%s.json" % (OUT, ANCHOR, ETIKET), "w"), indent=1)
ta = sum(len(v["aligned_chunks"]) for v in plan.values())
tn = sum(len(v["native_chunks"]) for v in plan.values())
tc = sum(v["n_secilen"] for v in plan.values())
print("\nTOTAL %d pairs; aligned chunks %d (~%.0f MB), native chunks %d (~%.0f MB)"
      % (tc, ta, ta * 1.785, tn, tn * 0.459))
print("-> %s/yonga_listesi_%s%s.json" % (OUT, ANCHOR, ETIKET))
