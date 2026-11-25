#!/usr/bin/env python3
"""T4b - fetch ONE non-oracle VLM localization result per PlantSeg-100 image.

Protocol (user-authorized on 2026-11-25):
  * exactly the frozen v7 image-only prompts (POLYGON_PRIMARY_V7 / RETRY);
  * original image bytes only -- no filename, GT mask, disease label or metadata;
  * one effective localization per image: primary, then forced-recall retry
    only if primary produced no polygon/point; NO TTA, NO second view;
  * temperature=0.0, effort=high, max_tokens from frozen v7 config;
  * max_workers=4, per-image checkpoint so an interrupted run can resume.

The API key is read only from os.environ["DEEPSEEK_API_KEY"] (injected by
scripts/launch_with_key.py in-process). It is never written to this cache,
logs, or any result file. Only its 8-hex fingerprint is stored in metadata.

Usage:
  py -3.12 scripts/launch_with_key.py scripts/fetch_t4b_vlm_prompts.py \
      --limit 3 --out results/t4b_vlm_prompts_cache.json     # probe
  py -3.12 scripts/launch_with_key.py scripts/fetch_t4b_vlm_prompts.py \
      --out results/t4b_vlm_prompts_cache.json               # full/resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from llm_client_deepseek import chat_json, key_fingerprint  # noqa: E402
from prompts_v7 import (FROZEN_CONFIG_V7, POLYGON_PRIMARY_V7,
                        POLYGON_RETRY_V7, POLYGON_USER_V7)  # noqa: E402

WRITE_LOCK = threading.Lock()


def _target(a) -> bool:
    return bool(a and (a.get("lesions") or a.get("pos_points")))


def _dedup_boxes(boxes):
    seen, out = set(), []
    for bb in boxes:
        try:
            vals = [float(v) for v in bb[:4]]
        except Exception:
            continue
        if not (vals[2] > vals[0] and vals[3] > vals[1]):
            continue
        k = tuple(round(v, 1) for v in vals)
        if k in seen:
            continue
        seen.add(k)
        out.append(vals)
    return out


def boxes_from_annotation(ann, polygons):
    """Derive full-resolution boxes ONLY from VLM output (polygons / box fields)."""
    boxes = []
    for les in (ann or {}).get("lesions") or []:
        if not isinstance(les, dict):
            continue
        bb = les.get("box") or les.get("bbox")
        if bb is not None:
            try:
                boxes.append([float(v) for v in bb[:4]])
            except Exception:
                pass
    for poly in polygons or []:
        try:
            pts = np.array([[float(p[0]), float(p[1])] for p in poly],
                           dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) < 3:
                continue
            x, y, w, h = cv2.boundingRect(pts)
            if w > 0 and h > 0:
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
        except Exception:
            continue
    return _dedup_boxes(boxes)


def fetch_one(sid, image_path, key, model, reasoning):
    """Return one cache record. No ground truth or sample identity is sent."""
    t_start = time.perf_counter()
    r1 = chat_json(POLYGON_PRIMARY_V7, POLYGON_USER_V7, image_path,
                   model=model, max_output_tokens=FROZEN_CONFIG_V7["t1_max_output_tokens"],
                   reasoning_effort=reasoning, temperature=0.0, key=key)
    ann = r1.parsed
    source = "primary"
    t_in, t_out = r1.stats.input_tokens, r1.stats.output_tokens

    if not _target(ann):
        r2 = chat_json(POLYGON_RETRY_V7, POLYGON_USER_V7, image_path,
                       model=model, max_output_tokens=FROZEN_CONFIG_V7["t1_max_output_tokens"],
                       reasoning_effort=reasoning, temperature=0.0, key=key)
        t_in += r2.stats.input_tokens
        t_out += r2.stats.output_tokens
        if _target(r2.parsed):
            ann, source = r2.parsed, "retry"

    if not _target(ann):
        ann, source = {}, "empty"

    polygons, points = [], []
    for les in (ann.get("lesions") or []):
        if not isinstance(les, dict):
            continue
        poly = les.get("polygon") or les.get("poly")
        if poly and len(poly) >= 3:
            polygons.append(poly)
    for p in (ann.get("pos_points") or []):
        try:
            points.append([float(p[0]), float(p[1])])
        except Exception:
            continue

    boxes = boxes_from_annotation(ann, polygons)
    return {
        "sample_id": sid,
        "prompt_source": source,
        "boxes": boxes,
        "polygons": polygons,
        "pos_points": points,
        "leaf_bbox": ann.get("leaf_bbox"),
        "tokens_in": int(t_in),
        "tokens_out": int(t_out),
        "seconds": round(time.perf_counter() - t_start, 3),
        "model": model,
        "temperature": 0.0,
        "reasoning_effort": reasoning,
        "max_output_tokens": FROZEN_CONFIG_V7["t1_max_output_tokens"],
        "tta": False,
        "gt_used_as_prompt": False,
        "key_fingerprint": key_fingerprint(),
    }


def _valid(rec):
    return (isinstance(rec, dict) and rec.get("prompt_source") in
            ("primary", "retry", "empty") and "boxes" in rec and
            "polygons" in rec and "pos_points" in rec)


def save_cache(cache, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", default=os.path.join(ROOT, "data", "plantseg100", "imgs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "t4b_vlm_prompts_cache.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--reasoning", default=FROZEN_CONFIG_V7["t1_reasoning_effort"])
    args = ap.parse_args()

    from llm_client_deepseek import get_api_key
    key = get_api_key()  # env only, injected by launch_with_key.py

    ids = sorted(f[:-4] for f in os.listdir(args.imgdir) if f.lower().endswith(".jpg"))
    if args.limit:
        ids = ids[:args.limit]

    cache = {"schema": "t4b_vlm_prompts_v1", "records": {}, "meta": {
        "protocol": "frozen_v7_image_only_prompts / primary->retry / single_view_no_TTA",
        "temperature": 0.0,
        "gt_ever_sent": False,
        "sample_id_ever_sent": False,
        "key_fingerprint": key_fingerprint(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resumed_from_existing_cache": os.path.exists(args.out),
    }}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            old = json.load(fh)
        for sid, rec in (old.get("records") or {}).items():
            if _valid(rec):
                cache["records"][sid] = rec

    todo = [sid for sid in ids if sid not in cache["records"]]
    print("images=%d cached=%d todo=%d workers=%d key_fp=%s" %
          (len(ids), len(ids) - len(todo), len(todo), args.workers, key_fingerprint()),
          flush=True)
    if not todo:
        print("nothing to fetch; cache is complete", flush=True)
        return

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, sid, os.path.join(args.imgdir, sid + ".jpg"),
                        key, args.model, args.reasoning): sid for sid in todo
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                print("FAILED %s: %s" % (sid, repr(exc)[:240]), flush=True)
                continue
            with WRITE_LOCK:
                cache["records"][sid] = rec
                cache["meta"]["n_records"] = len(cache["records"])
                save_cache(cache, args.out)
            done += 1
            print("[%d/%d] %-48s src=%-7s boxes=%d polys=%d pts=%d tok=%d+%d %.1fs" %
                  (done, len(todo), sid, rec["prompt_source"], len(rec["boxes"]),
                   len(rec["polygons"]), len(rec["pos_points"]),
                   rec["tokens_in"], rec["tokens_out"], rec["seconds"]), flush=True)

    cache["meta"]["n_records"] = len(cache["records"])
    cache["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cache["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    with WRITE_LOCK:
        save_cache(cache, args.out)
    print("saved %s records=%d elapsed=%.1fs" %
          (args.out, len(cache["records"]), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
