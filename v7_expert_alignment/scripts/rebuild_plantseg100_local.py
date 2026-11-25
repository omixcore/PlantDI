#!/usr/bin/env python3
"""Rebuild the FROZEN PlantSeg-100 evaluation set locally from public sources.

Sources (all read-only, verified reachable 2026-11-24):
  * image IDs / crop / disease / expert GT fractions:
      v6_three_branch/results/predictions_v6_plantseg100.csv  (immutable baseline)
  * images: HF mirror of `Voxel51/PlantSeg-Test`
      https://hf-mirror.com/datasets/Voxel51/PlantSeg-Test/resolve/main/data/<file>
  * lesion polygons: official PlantSeg GitHub COCO annotation_test.json
      https://api.github.com/repos/tqwei05/PlantSeg/contents/coco_annotations/annotation_test.json

IMPORTANT: masks reconstructed from COCO polygons are NOT the original Zenodo
PNG masks (mean ratio difference is a few 0.001). Correlation metrics use the
FROZEN `published` values, exactly as v5/v6 did. Pixel metrics from the
reconstructed masks must be disclosed as such. Nothing here changes the frozen
benchmarks or the policy manifest.
"""
import collections
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, ImageDraw

V7 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(V7)
DATA = os.path.join(V7, "data", "plantseg100")
IMGS = os.path.join(DATA, "imgs")
MASKS = os.path.join(DATA, "masks")
COCO = os.path.join(V7, "data", "_download", "annotation_test.json")
FROZEN = os.path.join(PROJECT, "v6_three_branch", "results", "predictions_v6_plantseg100.csv")
HF = "https://hf-mirror.com/datasets/Voxel51/PlantSeg-Test/resolve/main/data/{}?download=true"
PROV = os.path.join(DATA, "rebuild_provenance.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: asfh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_rows():
    rows = [r for r in csv.DictReader(open(FROZEN, encoding="utf-8"))
            if r.get("sample_id") not in ("sample_id", "", None)]
    seen, uniq = set(), []
    for r in rows:
        if r["sample_id"] not in seen:
            seen.add(r["sample_id"])
            uniq.append(r)
    return uniq


def download_one(name, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return name, True, "cached"
    for attempt in range(3):
        try:
            r = requests.get(HF.format(name), timeout=90)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dest, "wb") as fh:
                    fh.write(r.content)
                return name, True, "downloaded"
        except Exception as exc:
            err = repr(exc)
        time.sleep(1 + attempt)
    return name, False, err[:120] if 'err' in locals() else "http_error"


def main():
    os.makedirs(IMGS, exist_ok=True)
    os.makedirs(MASKS, exist_ok=True)
    rows = frozen_rows()
    print("frozen PlantSeg-100 rows:", len(rows), flush=True)

    # 1. download images
    targets = [(r["sample_id"].replace("plantseg_", "", 1) + ".jpg",
                os.path.join(IMGS, r["sample_id"] + ".jpg")) for r in rows]
    failed = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(download_one, n, p) for n, p in targets]
        for i, f in enumerate(as_completed(futs), 1):
            name, ok, msg = f.result()
            if not ok:
                failed.append((name, msg))
            if i % 25 == 0:
                print("  images %d/%d" % (i, len(targets)), flush=True)
    if failed:
        print("FAILED DOWNLOADS:", failed[:10], flush=True)
        raise SystemExit(2)

    # 2. rasterize official COCO polygons into expert masks
    if not os.path.exists(COCO):
        raise SystemExit("missing " + COCO)
    coco = json.load(open(COCO, encoding="utf-8"))
    by_name = {x["file_name"]: x for x in coco["images"]}
    by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_id[str(a["image_id"])].append(a)

    diffs = []
    meta_out = []
    for r in rows:
        sid = r["sample_id"]
        name = sid.replace("plantseg_", "", 1) + ".jpg"
        imeta = by_name.get(name)
        if imeta is None:
            raise SystemExit("no COCO entry for " + name)
        ip = os.path.join(IMGS, sid + ".jpg")
        im = Image.open(ip)
        if im.size != (imeta["width"], imeta["height"]):
            print("SIZE MISMATCH", sid, im.size, (imeta["width"], imeta["height"]), flush=True)
        mask = Image.new("1", (imeta["width"], imeta["height"]), 0)
        dr = ImageDraw.Draw(mask)
        for a in by_id[str(imeta["id"])]:
            for seg in a["segmentation"]:
                pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                if len(pts) >= 3:
                    dr.polygon(pts, fill=1)
        mask.save(os.path.join(MASKS, sid + ".png"))
        ratio = sum(mask.getdata()) / float(meta := imeta["width"] * imeta["height"])
        pub = float(r["published"])
        diffs.append(abs(ratio - pub))
        meta_out.append({
            "sample_id": sid, "crop": r.get("crop", ""), "disease": r.get("disease", ""),
            "in_v5_48": r.get("in_v5_48", ""), "published": r["published"],
            "reconstructed_mask_ratio": round(ratio, 6),
        })

    with open(os.path.join(DATA, "meta100.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "crop", "disease", "in_v5_48",
                                           "published", "reconstructed_mask_ratio"])
        w.writeheader()
        w.writerows(meta_out)

    prov = {
        "rebuilt_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_images": len(rows),
        "image_source": HF.replace("{}", "<file>"),
        "annotation_source": "https://api.github.com/repos/tqwei05/PlantSeg/contents/coco_annotations/annotation_test.json?ref=main",
        "id_gt_source": os.path.relpath(FROZEN, PROJECT),
        "coco_sha256": sha(COCO) if os.path.exists(COCO) else "",
        "mask_method": "PIL ImageDraw.polygon over official COCO `segmentation` polygons",
        "gt_ratio_abs_diff": {
            "mean": sum(diffs) / len(diffs),
            "max": max(diffs),
            "n_below_0.002": sum(d < 0.002 for d in diffs),
        },
        "caveat": "Reconstructed masks are NOT the original Zenodo PNG masks. "
                  "Use frozen `published` for correlation metrics; disclose reconstruction for pixel metrics.",
        "image_sha256_sample": {r["sample_id"]: sha(os.path.join(IMGS, r["sample_id"] + ".jpg"))
                                for r in rows[:3]},
    }
    json.dump(prov, open(PROV, "w", encoding="utf-8"), indent=2)
    print("images:", len(rows), "masks:", len(rows))
    print("GT ratio abs diff mean=%.5f max=%.5f (n<0.002: %d/%d)" %
          (prov["gt_ratio_abs_diff"]["mean"], prov["gt_ratio_abs_diff"]["max"],
           prov["gt_ratio_abs_diff"]["n_below_0.002"], len(diffs)))
    print("wrote", os.path.join(DATA, "meta100.csv"))
    print("wrote", PROV)


if __name__ == "__main__":
    main()
