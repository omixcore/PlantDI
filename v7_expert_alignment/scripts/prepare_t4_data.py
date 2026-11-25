#!/usr/bin/env python3
"""T4 data preparation - extract official PlantSeg TRAIN split only.

Reads the 3 official `train` parquet shards already downloaded locally.
Writes:
  data/plantseg_train/imgs/<sid>.jpg
  data/plantseg_train/masks/<sid>.png
  data/plantseg_train/metadata_train.csv   (columns: sample_id, split, width, height)
  data/plantseg_train/vlm_prompts_train.json  (sid -> {boxes:[[x1,y1,x2,y2]], points:[]})
  data/plantseg_train/t4_data_audit.json

STRICT ISOLATION (re-verified here, not just trusted from split_audit.json):
  * Every extracted sid is rejected if it appears in the official Test split
    file list OR in PlantSeg-100 (meta100.csv).
  * bboxes come from the dataset's own annotation column [x, y, w, h] -> [x1,y1,x2,y2].
    These are training prompts ONLY; no GT mask is used as a prompt.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Dict, Set

import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DL = os.path.join(ROOT, "v7_expert_alignment", "data", "_download")
OUT = os.path.join(ROOT, "v7_expert_alignment", "data", "plantseg_train")
IMGD = os.path.join(OUT, "imgs")
MASKD = os.path.join(OUT, "masks")

TRAIN_SHARDS = [os.path.join(DL, "plantseg_train_%d.parquet" % i) for i in range(3)]
TEST_PARQUET = os.path.join(DL, "plantseg_test.parquet")
VAL_PARQUET = os.path.join(DL, "plantseg_val.parquet")
META100 = os.path.join(ROOT, "v7_expert_alignment", "data", "plantseg100", "meta100.csv")


def sid_of(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def load_exclusion_ids() -> Dict[str, Set[str]]:
    test_ids = set(pq.read_table(TEST_PARQUET, columns=["file_name"]).column("file_name").to_pylist())
    val_ids = set(pq.read_table(VAL_PARQUET, columns=["file_name"]).column("file_name").to_pylist())
    ps100 = set()
    with open(META100, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ps100.add(r["sample_id"].replace("plantseg_", "", 1) + ".jpg")
    return {
        "official_test": test_ids,
        "official_val": val_ids,
        "plantseg100": ps100,
    }


def main() -> None:
    excl = load_exclusion_ids()
    os.makedirs(IMGD, exist_ok=True)
    os.makedirs(MASKD, exist_ok=True)

    prompt_cache: Dict[str, dict] = {}
    meta_rows = []
    n_seen = 0
    n_skip_overlap = 0
    n_skip_nobox = 0
    n_skip_badbytes = 0
    sha = hashlib.sha256()

    for shard in TRAIN_SHARDS:
        if not os.path.exists(shard):
            raise FileNotFoundError(shard)
        pf = pq.ParquetFile(shard)
        cols = ["file_name", "width", "height", "image", "mask", "bboxes"]
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["file_name"])):
                n_seen += 1
                fname = d["file_name"][i]
                sid = sid_of(fname)
                # HARD isolation gates
                if fname in excl["official_test"] or fname in excl["plantseg100"]:
                    n_skip_overlap += 1
                    continue
                boxes = d["bboxes"][i] or []
                norm = []
                for b in boxes:
                    if b and len(b) >= 4:
                        x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        if w > 0 and h > 0:
                            norm.append([x, y, x + w, y + h])
                if not norm:
                    n_skip_nobox += 1
                    continue
                img_obj = d["image"][i]
                mask_obj = d["mask"][i]
                ib = img_obj.get("bytes") if isinstance(img_obj, dict) else None
                mb = mask_obj.get("bytes") if isinstance(mask_obj, dict) else None
                if not ib or not mb:
                    n_skip_badbytes += 1
                    continue
                ip = os.path.join(IMGD, sid + ".jpg")
                mp = os.path.join(MASKD, sid + ".png")
                with open(ip, "wb") as fh:
                    fh.write(ib)
                with open(mp, "wb") as fh:
                    fh.write(mb)
                prompt_cache[sid] = {"boxes": norm, "points": []}
                meta_rows.append({
                    "sample_id": sid,
                    "split": "train",
                    "width": int(d["width"][i]),
                    "height": int(d["height"][i]),
                })
                sha.update(sid.encode() + ib[:64])

    meta_path = os.path.join(OUT, "metadata_train.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "split", "width", "height"])
        w.writeheader()
        w.writerows(meta_rows)

    prompt_path = os.path.join(OUT, "vlm_prompts_train.json")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        json.dump(prompt_cache, fh, separators=(",", ":"))

    audit = {
        "source": "official PlantSeg train parquet shards (XuVV/plantsegv3 mirror)",
        "n_train_rows_seen": n_seen,
        "n_extracted": len(meta_rows),
        "n_skipped_overlap_test_or_ps100": n_skip_overlap,
        "n_skipped_no_bbox": n_skip_nobox,
        "n_skipped_bad_bytes": n_skip_badbytes,
        "exclusion_sizes": {k: len(v) for k, v in excl.items()},
        "overlap_extracted_with_test": len(set(r["sample_id"] + ".jpg" for r in meta_rows) & excl["official_test"]),
        "overlap_extracted_with_ps100": len(set(r["sample_id"] + ".jpg" for r in meta_rows) & excl["plantseg100"]),
        "prompt_source": "dataset_annotation_bboxes_xywh_to_xyxy (NOT VLM, NOT GT mask)",
        "content_sha256_prefix": sha.hexdigest()[:16],
    }
    audit_path = os.path.join(OUT, "t4_data_audit.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    assert audit["overlap_extracted_with_test"] == 0
    assert audit["overlap_extracted_with_ps100"] == 0
    assert audit["n_extracted"] > 1000
    print(json.dumps(audit, indent=2))
    print("T4 DATA PREP OK")


if __name__ == "__main__":
    main()
