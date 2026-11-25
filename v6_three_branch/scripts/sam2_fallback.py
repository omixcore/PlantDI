#!/usr/bin/env python3
"""V6 SAM2 empty-prompt two-level fallback.

Problem (v5): deepseek-flash returned empty `lesions` on 30/48 PlantSeg images,
so SAM2 had no box prompt -> only 18/48 valid masks.

Fix:
  Level 1 (forced recall retry): a dedicated prompt that explicitly says the
     image CONTAINS disease and forbids an empty array; asks for 1-3 boxes or
     center pos_points of the most suspicious abnormal-color regions.
  Level 2 (CPU-friendly fallback): if retry still empty/unparseable, generate
     point prompts from a lightweight saliency heuristic (deviation from the
     dominant healthy-green / background color) + image center. NO full-image
     Automatic Mask Generator (too slow on CPU).

Every sample records prompt_source in {primary, retry, fallback} for ablation.
Coordinates returned by the VLM are on the downscaled image; they are mapped
back to the original resolution before SAM2 inference.
"""
import os, base64, json, time
import cv2, numpy as np, requests

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

PRIMARY_SYSTEM = (
    "Output ONLY JSON: {\"lesions\":[{\"box\":[x1,y1,x2,y2]}],\"leaf_bbox\":[x1,y1,x2,y2]}. "
    "List every distinct diseased / necrotic / abnormally-colored spot as a box. "
    "Coordinates are pixel integers of THIS image. Nothing else."
)
RETRY_SYSTEM = (
    "This image DEFINITELY contains a plant disease or abnormal leaf symptom. "
    "You MUST NOT return an empty lesions array. "
    "Output ONLY JSON: {\"lesions\":[{\"box\":[x1,y1,x2,y2]}, ...],\"pos_points\":[[x,y], ...]}. "
    "Identify the 1-3 MOST suspicious diseased / discolored regions. "
    "For each give a tight box [x1,y1,x2,y2] of pixel integers on THIS image. "
    "If a region is too diffuse for a box, give its center as a point in pos_points. "
    "It is MANDATORY to output at least one box or one point. Nothing else."
)


def parse(t):
    if not t:
        return None
    i = t.find("{")
    if i < 0:
        return None
    dep, in_str, esc = 0, False, False
    for j in range(i, len(t)):
        ch = t[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            dep += 1
        elif ch == "}":
            dep -= 1
            if dep == 0:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    return None
    return None


def _encode_small(path, max_side=900, quality=88):
    img = cv2.imread(path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode(), scale, small.shape[:2]


def _call(b64, system, key, max_retry=3, max_tokens=1200):
    payload = {
        "model": MODEL, "effort": "low", "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": "JSON only:"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]},
        ],
    }
    hdr = {"Authorization": "Bearer " + key}
    tok = 0
    for a in range(max_retry):
        try:
            r = requests.post(API, headers=hdr, json=payload, timeout=90)
            d = r.json()
            tok += d.get("usage", {}).get("total_tokens", 0)
            if r.status_code != 200:
                time.sleep(1 + a)
                continue
            m = d["choices"][0]["message"]
            ann = parse(m.get("content") or "") or parse(m.get("reasoning_content") or "")
            if ann is not None:
                return ann, tok
            time.sleep(1)
        except Exception:
            time.sleep(1 + a)
    return None, tok


def _has_lesion(ann):
    return bool(ann and (ann.get("lesions") or ann.get("pos_points")))


def saliency_points(bgr_small, max_points=3):
    """CPU-light saliency: find pixels deviating from the dominant healthy color,
    then take the strongest connected components' centers plus the image center.

    Handles green-toned lesions (mosaic / mildew) because the prior is
    'deviation from dominant hue', not 'brownness'.
    """
    h, w = bgr_small.shape[:2]
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    # healthy vegetation = green hue 25..95 (OpenCV 0..179)
    green = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) &
             (hsv[..., 1] > 40) & (hsv[..., 2] > 40))
    # very bright / very dark = background or shadow, not lesion
    mid_val = (hsv[..., 2] > 45) & (hsv[..., 2] < 250)
    sat_ok = hsv[..., 1] > 25
    # lesion candidates: non-green, mid value, adequate saturation
    cand = (~green) & mid_val & sat_ok
    cand = cand.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        cx, cy = cents[i]
        comps.append((area, float(cx), float(cy)))
    comps.sort(reverse=True)
    pts = [[int(round(x)), int(round(y))] for _, x, y in comps[:max_points]]
    # always include image center as a neutral anchor (guarantees a SAM2 mask)
    anchor = [int(w / 2), int(h / 2)]
    if anchor not in pts:
        pts.append(anchor)
    return pts[:max_points + 1]


def locate_with_fallback(path, key):
    """Return (lesion_boxes_fullres, pos_points_fullres, prompt_source, tokens,
    scale, ann_raw).

    Boxes/points are mapped from the downscaled image back to original scale.
    """
    b64, scale, small_shape = _encode_small(path)
    if b64 is None:
        return [], [], "fallback", 0, 1.0, None
    img = cv2.imread(path)
    H, W = img.shape[:2]

    ann, tok1 = _call(b64, PRIMARY_SYSTEM, key)
    src = "primary"
    if not _has_lesion(ann):
        ann2, tok2 = _call(b64, RETRY_SYSTEM, key)
        ann = ann2 if _has_lesion(ann2) else None
        src = "retry" if _has_lesion(ann2) else "fallback"
        tok1 += tok2

    boxes, points = [], []
    if ann and ann.get("lesions"):
        for L in ann["lesions"]:
            try:
                bb = L.get("box") or L.get("bbox")
                x1, y1, x2, y2 = [float(v) for v in bb[:4]]
                if x2 > x1 and y2 > y1:
                    boxes.append([x1 / scale, y1 / scale, x2 / scale, y2 / scale])
            except Exception:
                continue
    if ann and ann.get("pos_points"):
        for p in ann["pos_points"]:
            try:
                points.append([float(p[0]) / scale, float(p[1]) / scale])
            except Exception:
                continue

    if not boxes and not points:
        # Level 2: CPU-friendly saliency point hints (on downscaled image)
        small_img = cv2.resize(img, small_shape[::-1]) if scale != 1.0 else img
        sp = saliency_points(small_img)
        points = [[x / scale, y / scale] for x, y in sp]
        src = "fallback"
    return boxes, points, src, tok1, scale, ann


def mask_from_boxes_points(sam_predictor, bgr, boxes, points):
    """Run SAM2 with box (preferred) or point prompts. Returns best mask (bool)
    chosen by SAM2's own predicted IoU score (no ground-truth leakage)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(rgb)
    masks = []
    scores = []
    for box in boxes[:3]:
        try:
            bx = np.asarray(box, dtype=np.float32)
            ms, sc, _ = sam_predictor.predict(box=bx, multimask_output=True)
            for i in range(len(sc)):
                masks.append(np.asarray(ms[i], dtype=bool))
                scores.append(float(sc[i]))
        except Exception:
            continue
    if not masks:
        for p in points[:3]:
            try:
                pt = np.asarray(p, dtype=np.float32)
                lab = np.array([1], dtype=np.int32)
                ms, sc, _ = sam_predictor.predict(point_coords=pt[None, :],
                                                  point_labels=lab,
                                                  multimask_output=True)
                for i in range(len(sc)):
                    masks.append(np.asarray(ms[i], dtype=bool))
                    scores.append(float(sc[i]))
            except Exception:
                continue
    if not masks:
        return np.zeros(bgr.shape[:2], dtype=bool)
    best = masks[int(np.argmax(scores))]
    return best
