# -*- coding: utf-8 -*-
"""Step 2 - pick the held-out tiles from scratch.

No pre-made tile list is trusted. A tile is eligible when, inside the validation
mask of its segment, it carries at least 60 ink pixels and at least 60 non-ink pixels.

This step also checks the thing the whole exercise rests on: that the validation
region really was excluded from supervision. It sums the overlap between
supervision_mask and validation_mask; that sum must be 0.

If VZ_ROOT/prior_val_tiles.json exists, the selection is also compared against it.
"""
import json, os
import numpy as np
import numcodecs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
ROOT = os.environ.get("VZ_ROOT", os.path.join(REPO, "data"))

CACHE = os.path.join(ROOT, "cache", "labels")
VOLD = os.path.join(ROOT, "cache", "volumes")
IDX = json.load(open(os.path.join(WORK, "label_index.json")))
SEGS = {s["key"]: s for s in json.load(
    open(os.path.join(ROOT, "segments.json"), encoding="utf-8"))["segments"]}
_BL = numcodecs.Blosc()
_c = {}


def blob(h, nz):
    if h in _c:
        return _c[h]
    fp = os.path.join(CACHE, h[:2], h)
    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
        a = np.zeros((nz, 128, 128), np.uint8)
    else:
        a = np.frombuffer(_BL.decode(open(fp, "rb").read()),
                          np.uint8).reshape(nz, 128, 128).copy()
    if len(_c) < 3000:
        _c[h] = a
    return a


def tile(segkey, kind, y, x, nz):
    h = IDX["index"].get(f"{segkey}|{kind}", {}).get(f"{y},{x}")
    if h is None:
        return np.zeros((nz, 128, 128), np.uint8)
    return blob(h, nz)


TARGET = ["aligned-scrollprizeorg-21slices/pherc0139-w016",
          "aligned-scrollprizeorg-21slices/pherc0814-46527",
          "aligned-scrollprizeorg-21slices/pherc1667-w029"]

prior_path = os.path.join(ROOT, "prior_val_tiles.json")
prior = json.load(open(prior_path)) if os.path.exists(prior_path) else None

found = {}
for segkey in TARGET:
    s = SEGS[segkey]
    nz = s["label_chunk_zyx"][0]
    zc = s["annotation_center_channel"]
    vmap = IDX["index"][f"{segkey}|val"]
    cand = []
    for yx in vmap:
        y, x = (int(v) for v in yx.split(","))
        vm = tile(segkey, "val", y, x, nz)[zc] > 0
        if not vm.any():
            continue
        ink = tile(segkey, "ink", y, x, nz)[zc] > 0
        n1 = int((ink & vm).sum()); n0 = int(((~ink) & vm).sum())
        sup = tile(segkey, "sup", y, x, nz)[zc] > 0
        cand.append(dict(y=y, x=x, n_val=int(vm.sum()), n_ink=n1, n_bg=n0,
                         n_sup_overlap=int((sup & vm).sum()),
                         eligible=bool(n1 >= 60 and n0 >= 60),
                         volume_cached=os.path.exists(
                             os.path.join(VOLD, segkey.replace("/", "__") + f"__{y}_{x}.npy"))))
    cand.sort(key=lambda r: (r["y"], r["x"]))
    found[segkey] = cand
    elig = [a for a in cand if a["eligible"]]
    line = ("%-18s val tiles (non-empty)=%3d  passing the criterion=%3d"
            % (segkey.split("/")[1], len(cand), len(elig)))
    if prior is not None:
        old = {(t[0], t[1]) for t in prior[segkey]}
        mine = {(a["y"], a["x"]) for a in elig}
        line += ("  prior list=%3d  identical=%s  only-mine=%s  only-prior=%s"
                 % (len(old), mine == old, sorted(mine - old), sorted(old - mine)))
    print(line)
    print("    supervision-validation pixel overlap = %d  "
          "(must be 0 for the held-out set to be honest)"
          % sum(a["n_sup_overlap"] for a in elig))
    print("    eligible tiles whose volume is not cached = %d"
          % sum(1 for a in elig if not a["volume_cached"]))

json.dump(found, open(os.path.join(WORK, "tile_selection.json"), "w"), indent=1)
print("\nwritten ->", os.path.join(WORK, "tile_selection.json"))
