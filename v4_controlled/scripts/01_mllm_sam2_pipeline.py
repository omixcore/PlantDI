#!/usr/bin/env python3
"""MLLM-guided SAM 2 pipeline for plant disease index quantification.

Usage:
    Set DEEPSEEK_API_KEY in your environment, then run: python 01_mllm_sam2_pipeline.py --limit 3
    Set DEEPSEEK_API_KEY in your environment, then run: python 01_mllm_sam2_pipeline.py

Resume-safe: samples already present in predictions.csv are skipped.
"""
import argparse
import base64
import csv
import json
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = os.environ.get("PLANTDI_ROOT", ".")
LEAF_DIR = os.path.join(ROOT, "data", "processed_leaves")
GT_PATH = os.path.join(LEAF_DIR, "ground_truth.csv")
PRED_PATH = os.path.join(ROOT, "results", "predictions.csv")
OVERLAY_DIR = os.path.join(ROOT, "results", "overlays")
CKPT = os.path.join(ROOT, "models", "sam2_hiera_tiny.pt")
CONFIG = "sam2_hiera_t.yaml"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

SYSTEM_PROMPT = """You are a meticulous plant pathology image analyst. Output ONLY strict JSON, no prose, no markdown.
For the image, identify:
- leaf_bbox: a TIGHT bounding box around ONE single most prominent leaf blade (usually the central/foreground leaf). Do NOT include neighbouring leaves, branches, soil or background. [x1,y1,x2,y2] pixels.
- lesion_bboxes: bounding boxes of EVERY diseased area on that leaf, including tiny orange/brown rust pustules, necrotic flecks, mould and chlorotic halos. Be exhaustive: even 2-5 pixel spots count. Do NOT confuse bright sunlight reflections (white/silvery shiny patches) with lesions.
- pos_points: one center point strictly inside each lesion, [[x,y], ...]. Same count/order as lesion_bboxes.
- neg_points: points on healthy green tissue, normal veins, and especially bright specular reflections, [[x,y], ...].
Coordinates must be integers inside the image. Ignore text labels, scale bars and panel letters.
Return exactly: {"leaf_bbox":[x1,y1,x2,y2],"lesion_bboxes":[[x1,y1,x2,y2],...],"pos_points":[[x,y],...],"neg_points":[[x,y],...]}"""

USER_PROMPT = """Analyse this leaf image. Return the JSON object defined by the system prompt."""


def salvage_json(text):
    """Recover complete arrays from truncated or thinking-only model output."""
    ann = {}
    for key in ("leaf_bbox", "lesion_bboxes", "pos_points", "neg_points"):
        m = re.search(r'"' + key + r'"\s*:\s*\[', text)
        if not m:
            continue
        start = m.end() - 1
        depth = 0
        end = None
        for j in range(start, len(text)):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end:
            try:
                ann[key] = json.loads(text[start:end])
            except Exception:
                pass
    if "leaf_bbox" in ann:
        for k in ("lesion_bboxes", "pos_points", "neg_points"):
            ann.setdefault(k, [])
        return ann
    return None


def query_mllm(path, max_retry=3):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]},
        ],
        "max_tokens": 4000,
    }
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    last = None
    for attempt in range(max_retry):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if r.status_code != 200:
                last = "HTTP %d %s" % (r.status_code, r.text[:200])
                time.sleep(2 + attempt * 3)
                continue
            data = r.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            combined = (content + "\n" + reasoning).strip()
            if not combined:
                last = "Empty response"
                continue
            ann = None
            try:
                m = re.search(r"\{.*\}", combined, re.S)
                if m:
                    ann = json.loads(m.group(0))
            except Exception:
                ann = None
            if ann is None:
                ann = salvage_json(combined)
            if ann is None or "leaf_bbox" not in ann:
                last = "No JSON in response: " + combined[:200]
                time.sleep(1)
                continue
            for k in ("lesion_bboxes", "pos_points", "neg_points"):
                ann.setdefault(k, [])
            return ann, data.get("usage", {})
        except Exception as exc:
            last = repr(exc)
            time.sleep(2 + attempt * 3)
    raise RuntimeError("MLLM failed: " + str(last))


def build_predictor():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(CONFIG, CKPT, device="cpu")
    return SAM2ImagePredictor(model)


def clip_box(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def segment(predictor, rgb, ann):
    h, w = rgb.shape[:2]
    predictor.set_image(rgb)
    leaf_box = clip_box(ann["leaf_bbox"], w, h)
    if leaf_box is None:
        leaf_box = [0, 0, w, h]
    masks, scores, _ = predictor.predict(box=np.array(leaf_box), multimask_output=False)
    leaf = np.asarray(masks[0], dtype=bool)
    # Safeguard: if the prompt box produced a near-full-frame mask, replace it
    # with the largest green-vegetation component (dilated to cover lesions).
    if leaf.sum() > 0.55 * h * w:
        r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
        prior = ((g > r + 8) & (g > b + 8) & (g > 40)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(prior, 8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            comp = (lab == idx).astype(np.uint8)
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
            leaf = comp > 0
    lesion = np.zeros((h, w), dtype=bool)
    neg_pts, neg_labels = [], []
    for p in ann.get("neg_points", []):
        if 0 <= p[0] < w and 0 <= p[1] < h:
            neg_pts.append([int(p[0]), int(p[1])]); neg_labels.append(0)
    boxes = [clip_box(b, w, h) for b in ann.get("lesion_bboxes", [])]
    boxes = [b for b in boxes if b]
    pos = [[int(p[0]), int(p[1])] for p in ann.get("pos_points", [])
           if 0 <= p[0] < w and 0 <= p[1] < h]
    if boxes or pos:
        prompts = []
        for i, b in enumerate(boxes):
            p = [pos[i]] if i < len(pos) else (
                [[(b[0] + b[2]) // 2, (b[1] + b[3]) // 2]] if b else [])
            prompts.append((b, p))
        if not boxes and pos:
            prompts.append((None, [[p[0], p[1]] for p in pos]))
        for b, one_pos in prompts:
            pts, labels = [], []
            for p in one_pos:
                pts.append(p); labels.append(1)
            pts += neg_pts; labels += neg_labels
            kw = dict(multimask_output=False)
            if b is not None:
                kw["box"] = np.array(b)
            if pts:
                kw["point_coords"] = np.array(pts)
                kw["point_labels"] = np.array(labels)
            try:
                ms, sc, _ = predictor.predict(**kw)
                if sc[0] > 0.20:
                    lesion |= np.asarray(ms[0], dtype=bool)
            except Exception:
                pass
    lesion &= leaf
    return leaf, lesion, float(scores[0])


def make_overlay(path, leaf, lesion):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    over = img.copy().astype(np.float32)
    contours, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, contours, -1, (0, 200, 0), 1)
    red = np.array([230, 40, 40], dtype=np.float32)
    alpha = 0.42 * lesion[..., None]
    over = over * (1 - alpha) + red * alpha
    return np.clip(over, 0, 255).astype(np.uint8)


def di_from_severity(crop, severity):
    """Configurable severity (%) -> standard DI mapping."""
    if crop == "rocole":
        if severity < 10: return 1.0
        if severity < 25: return 2.0
        if severity < 50: return 3.0
        return 4.0
    return round(severity, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process first N samples (small-batch test)")
    ap.add_argument("--gt", default=GT_PATH)
    args = ap.parse_args()

    os.makedirs(OVERLAY_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    gt = pd_read_csv(args.gt)
    done = set()
    if os.path.exists(PRED_PATH):
        with open(PRED_PATH, encoding="utf-8-sig") as fh:
            done = {r["sample_id"] for r in csv.DictReader(fh)}
    todo = [r for r in gt if r["sample_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("todo:", len(todo), "already done:", len(done), flush=True)
    if not todo:
        return

    predictor = build_predictor()
    fields = ["sample_id", "crop", "group_name", "published_di_or_area",
              "leaf_pixels", "lesion_pixels", "severity_percent", "di_pred",
              "leaf_score", "tokens", "elapsed_sec", "value_source", "timestamp"]
    new_file = not os.path.exists(PRED_PATH)
    with open(PRED_PATH, "a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(todo, 1):
            t0 = time.time()
            sid = row["sample_id"]
            img_path = os.path.join(LEAF_DIR, sid + ".png")
            if not os.path.exists(img_path):
                print("MISSING", sid, flush=True)
                continue
            try:
                ann, usage = query_mllm(img_path)
                rgb = np.array(Image.open(img_path).convert("RGB"))
                leaf, lesion, score = segment(predictor, rgb, ann)
                lp, sp = int(leaf.sum()), int(lesion.sum())
                sev = 100.0 * sp / max(lp, 1)
                di = di_from_severity(row["crop"], sev)
                out = os.path.join(OVERLAY_DIR, sid + "_overlay.png")
                Image.fromarray(make_overlay(img_path, leaf, lesion)).save(out)
                rec = dict(sample_id=sid, crop=row["crop"], group_name=row["group_name"],
                           published_di_or_area=row["published_di_or_area"],
                           leaf_pixels=lp, lesion_pixels=sp,
                           severity_percent=round(sev, 4), di_pred=di,
                           leaf_score=round(score, 4),
                           tokens=usage.get("total_tokens", 0),
                           elapsed_sec=round(time.time() - t0, 2),
                           value_source=row.get("value_source", ""),
                           timestamp=datetime.now().isoformat(timespec="seconds"))
                writer.writerow(rec)
                fh.flush()
                print("[%d/%d] %s sev=%.2f%% DI=%.2f pub=%s %.1fs" %
                      (i, len(todo), sid, sev, di, row["published_di_or_area"], time.time() - t0), flush=True)
            except Exception as exc:
                print("ERROR", sid, repr(exc), flush=True)


def pd_read_csv(path):
    import pandas as pd
    return pd.read_csv(path, encoding="utf-8-sig").to_dict("records")


if __name__ == "__main__":
    main()
