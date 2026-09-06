# -*- coding: utf-8 -*-
"""Step 1 - build the label chunk index from scratch.

Reads the HuggingFace file listing of the ink_9um label tree and maps every
(segment, mask kind, tile) to the content hash of its chunk. Nothing is taken from
an earlier run's index.

It also reports, explicitly, which chunk blobs are MISSING from the local cache.
That matters: code that silently turns a missing chunk into an all-zero array will
quietly change the labels it is scoring against.

Needs:
  WORK/tree_full.json      the file listing (path + xetHash per entry)
  VZ_ROOT/cache/labels/    the downloaded label chunks, sharded by the first two
                           hex characters of the hash
"""
import json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("VZ_WORK", os.path.join(REPO, "work"))
ROOT = os.environ.get("VZ_ROOT", os.path.join(REPO, "data"))
os.makedirs(WORK, exist_ok=True)

TREE = os.path.join(WORK, "tree_full.json")
CACHE = os.path.join(ROOT, "cache", "labels")
OUT = os.path.join(WORK, "label_index.json")

# ink_9um/labels/<family>/<segment>/<array>.zarr/0/<z>.<y>.<x>
PAT = re.compile(r"^ink_9um/labels/([^/]+)/([^/]+)/([^/]+)\.zarr/0/(\d+)\.(\d+)\.(\d+)$")
KIND = {"_inklabels": "ink", "_supervision_mask": "sup", "_validation_mask": "val"}

TARGET = {
    "aligned-scrollprizeorg-21slices/pherc0139-w016",
    "aligned-scrollprizeorg-21slices/pherc0814-46527",
    "aligned-scrollprizeorg-21slices/pherc1667-w029",
}


def main():
    t = json.load(open(TREE))
    idx = collections.defaultdict(dict)
    hash_count = collections.Counter()
    n_lab = 0
    for e in t:
        p = e["path"]
        if not p.startswith("ink_9um/labels/"):
            continue
        m = PAT.match(p)
        if not m:
            continue
        fam, seg, arr, z, y, x = m.groups()
        kind = None
        for suf, k in KIND.items():
            if arr.endswith(suf):
                kind = k
                break
        if kind is None:
            continue
        n_lab += 1
        hash_count[e["xetHash"]] += 1
        idx[f"{fam}/{seg}|{kind}"][f"{y},{x}"] = e["xetHash"]

    print("label chunk records:", n_lab)
    print("unique blob hashes:", len(hash_count))
    print("three most repeated hashes:", hash_count.most_common(3))

    # is the blob actually on disk? counted for every blob, not just the target segments
    present, missing = 0, collections.Counter()
    check = {}
    for h in hash_count:
        fp = os.path.join(CACHE, h[:2], h)
        e = os.path.exists(fp) and os.path.getsize(fp) > 0
        check[h] = e
        if e:
            present += 1
        else:
            missing[h] = hash_count[h]
    print(f"blobs PRESENT on disk: {present}/{len(hash_count)}   MISSING: {len(missing)}")
    print("chunks covered by missing blobs:", sum(missing.values()))
    print("three most-referenced missing blobs:", missing.most_common(3))

    # anything missing for the three target segments?
    report = {}
    for segkey in sorted(TARGET):
        for kind in ("ink", "sup", "val"):
            k = f"{segkey}|{kind}"
            m = idx.get(k, {})
            miss = [yx for yx, h in m.items() if not check[h]]
            report[k] = dict(n_chunks=len(m), n_missing=len(miss), missing=sorted(miss)[:20])
            print(f"  {k:70s} chunks={len(m):5d} missing={len(miss)}")

    json.dump(dict(index={k: v for k, v in idx.items() if k.split('|')[0] in TARGET},
                   on_disk={h: check[h] for h in check},
                   missing_summary=dict(n_missing_blobs=len(missing),
                                        n_missing_chunks=sum(missing.values())),
                   target_report=report),
              open(OUT, "w"))
    print("written ->", OUT)


if __name__ == "__main__":
    main()
