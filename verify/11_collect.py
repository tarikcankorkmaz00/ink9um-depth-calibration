# -*- coding: utf-8 -*-
"""Step 11 - collect every number into one file.

Reads whatever steps 1-10 left in WORK and writes a single verification.json.
Sections that this run cannot regenerate (because the corresponding step needs a GPU
and the cached chunks) are carried over unchanged from the committed
data/verification.json, so the file is never left half-empty.

Write it somewhere else with VZ_OUT=/some/dir.
"""
import json, os, hashlib, platform, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
ROOT = os.environ.get("VZ_ROOT", os.path.join(REPO, "data"))
OUT = os.environ.get("VZ_OUT", WORK)
os.makedirs(OUT, exist_ok=True)

COMMITTED = os.path.join(REPO, "data", "verification.json")
prev = json.load(open(COMMITTED, encoding="utf-8")) if os.path.exists(COMMITTED) else {}

CKPT = os.path.join(ROOT, "model", "ink_9um", "hybrid_3d2d-seed42", "step-075000.pth")
if os.path.exists(CKPT):
    sha = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
else:
    sha = prev.get("meta", {}).get("checkpoint_sha256")

D = {}
D["meta"] = dict(
    date=datetime.date.today().isoformat(),
    purpose="Independent re-measurement of the per-segment z-offset claim on the "
            "ink_9um held-out segments, and a test of the spatial scale the offset "
            "actually lives at.",
    method="Own pipeline: own label index, window position derived from metadata, "
           "sklearn.metrics.roc_auc_score, own cross-validation code.",
    checkpoint="ink_9um/hybrid_3d2d-seed42/step-075000.pth",
    checkpoint_sha256=sha,
    python=sys.version.split()[0], platform=platform.platform(),
)

SECTIONS = [
    ("label_index", "label_index.json"),
    ("tile_selection", "tile_selection.json"),
    ("heldout_scan", "heldout_scan.json"),
    ("analysis", "analysis.json"),
    ("hard_tests", "hard_tests.json"),
    ("scale", "scale.json"),
    ("controls", "controls.json"),
    ("supervision_scan", "supervision_scan.json"),
    ("transfer", "transfer.json"),
    ("stability", "stability.json"),
    ("all_tiles_analysis", "all_tiles.json"),
]

fresh, carried = [], []
for name, fn in SECTIONS:
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        d = json.load(open(p))
        if name == "label_index":
            # the full index is large and is an input, not a result
            d = dict(missing_summary=d["missing_summary"], target_report=d["target_report"])
        D[name] = d
        fresh.append(name)
    elif name in prev:
        D[name] = prev[name]
        carried.append(name)

path = os.path.join(OUT, "verification.json")
json.dump(D, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("regenerated in this run :", ", ".join(fresh) or "(none)")
print("carried over unchanged  :", ", ".join(carried) or "(none)")
print("written -> %s  (%.2f MB)" % (path, os.path.getsize(path) / 1e6))
