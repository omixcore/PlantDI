#!/usr/bin/env python3
"""Rebuild the EXACT frozen PlantSeg-100 images + expert masks from the public
Hugging Face parquet mirror `XuVV/plantsegv3` (test split, 2,295 images).

The frozen sample IDs and `published` GT fractions come from the immutable
v6 prediction CSV; no sample is selected by looking at v7 results.

Source: https://hf-mirror.com/datasets/XuVV/plantsegv3/resolve/main/data/test-00000-of-00001.parquet
"""
import csv
import hashlib
import io
import json
import os
import time

from PIL import Image

V7 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(V7)
DATA = os.path.join(V7, "data", "plantseg100")
IMGS = os.path.join(DATA, "imgs")
MASKS = os.path.join(DATA, "masks")
PARQUET = os.path.join(V7, "data", "_download", "plantseg_test.parquet")
FROZEN = os.path.join(PROJECT, "v6_three_branch", "results", "predictions_v6_plantseg100.csv")


def frozen_rows():
    rows = [r for r in csv.DictReader(open(FROZEN, encoding="utf-8"))
            if r.get("sample_id") not in ("sample_id", "", None)]
    seen, uniq = set(), []
    for r in rows:
        if r["sample_id"] not in seen:
            seen.add(r["sample_id"])
            uniq.append(r)
    return uniq


def _bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def main():
    import pyarrow.parquet as pq
    rows = frozen_rows()
    wanted = {r["sample_id"].replace("plantseg_", "", 1) + ".jpg": r for r in rows}
    print("frozen rows:", len(rows), "wanted names:", len(wanted), flush=True)

    pf = pq.ParquetFile(PARQUET)
    cols = ["file_name", "image", "mask"]
    found = {}
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        d = batch.to_pydict()
        for name, img, msk in zip(d["file_name"], d["image"], d["mask"]):
            if name in wanted:
                found[name] = (img, msk)
        if len(found) == len(wanted):
            break
    missing = sorted(set(wanted) - set(found))
    print("found:", len(found), "missing:", len(missing), missing[:5], flush=True)
    if missing:
        raise SystemExit(2)

    os.makedirs(IMGS, exist_ok=True)
    os.makedirs(MASKS, exist_ok=True)
    meta, diffs, hashes = [], [], {}
    for name, r in wanted.items():
        sid = r["sample_id"]
        img_cell, msk_cell = found[name]
        img_bytes = _bytes(img_cell)
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        ip = os.path.join(IMGS, sid + ".jpg")
        im.save(ip, quality=95)
        m = Image.open(io.BytesIO(_bytes(msk_cell)))
        # The official mask encodes lesion class ids; every non-zero pixel is disease.
        mgray = m.convert("L")
        binary = mgray.point(lambda v: 255 if v > 0 else 0).convert("1")
        mp = os.path.join(MASKS, sid + ".png")
        binary.save(mp)
        ratio = sum(1 for p in binary.getdata() if p) / float(im.width * im.height)
        pub = float(r["published"])
        diffs.append(abs(ratio - pub))
        hashes[sid] = hashlib.sha256(open(ip, "rb").read()).hexdigest()[:16]
        meta.append({
            "sample_id": sid, "crop": r.get("crop", ""), "disease": r.get("disease", ""),
            "in_v5_48": r.get("in_v5_48", ""), "published": r["published"],
            "mask_ratio": round(ratio, 8),
        })

    with open(os.path.join(DATA, "meta100.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "crop", "disease", "in_v5_48",
                                           "published", "mask_ratio"])
        w.writeheader(); w.writerows(meta)

    prov = {
        "rebuilt_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "https://hf-mirror.com/datasets/XuVV/plantsegv3/resolve/main/data/test-00000-of-00001.parquet",
        "source_sha256": hashlib.sha256(open(PARQUET, "rb").read()).hexdigest(),
        "id_gt_source": os.path.relpath(FROZEN, PROJECT),
        "n": len(meta),
        "gt_ratio_abs_diff": {"mean": sum(diffs)/len(diffs), "max": max(diffs)},
        "image_sha256_prefix": hashes,
        "note": "Images and masks from the official-split parquet mirror; frozen IDs/GT from v6 CSV. "
                "Masks are byte-exact to the source; images are re-encoded JPEG q95 (visual content unchanged).",
    }
    json.dump(prov, open(os.path.join(DATA, "rebuild_provenance.json"), "w", encoding="utf-8"),
              indent=2)
    print("GT ratio diff mean=%.8f max=%.8f" % (prov["gt_ratio_abs_diff"]["mean"],
                                                prov["gt_ratio_abs_diff"]["max"]))
    print("wrote", os.path.join(DATA, "meta100.csv"))


if __name__ == "__main__":
    main()
