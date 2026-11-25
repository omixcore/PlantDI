#!/usr/bin/env python3
"""T3 - leaf-mask clipping, polygon localization, TTA and area calibration.

Zero-SAM2 implementation. Pipeline for a single image:

  1. VLM returns `leaf_bbox` + lesion `polygon`s at ORIGINAL resolution
     (primary -> forced-recall retry -> saliency fallback).
  2. Original + horizontal-flip TTA; the two masks are combined (union).
  3. `leaf_mask` from HSV non-background filtering (+ GrabCut when available),
     restricted to `leaf_bbox`.
  4. `final_lesion_mask = predicted_lesion_mask & leaf_mask`.
  5. Optional symptom-wise quantile calibration, fitted STRICTLY on training
     images (never on official Test or PlantSeg-100).

No ground-truth mask is ever used to build a prediction.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from llm_client_deepseek import CallStats, chat_json, get_api_key  # noqa: E402
from prompts_v7 import (FROZEN_CONFIG_V7, POLYGON_PRIMARY_V7, POLYGON_RETRY_V7,
                        POLYGON_USER_V7)  # noqa: E402


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for polygon_and_clip") from exc


# ---------------------------------------------------------------------------
# 1. Leaf foreground mask
# ---------------------------------------------------------------------------
def leaf_mask(bgr: np.ndarray, leaf_bbox: Optional[Sequence[float]] = None,
              method: str = "hsv_or_grabcut") -> np.ndarray:
    """Return a bool mask of the leaf foreground (lesions included)."""
    cv2 = _cv2()
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = [hsv[..., i] for i in range(3)]
    green = (H >= 25) & (H <= 95) & (S > 35) & (V > 40)
    nonwhite = ~((S < 45) & (V > 205))
    nonblack = V > 48
    foreground = (green | ((S > 40) & nonblack & (V < 250))) & nonwhite & nonblack

    if leaf_bbox is not None:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in leaf_bbox[:4]]
        except Exception:
            x1, y1, x2, y2 = 0, 0, w, h
        roi = np.zeros((h, w), dtype=bool)
        roi[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        foreground &= roi

    fg = foreground.astype(np.uint8)
    if method in ("hsv_or_grabcut", "grabcut") and leaf_bbox is not None:
        try:
            mask = np.zeros((h, w), np.uint8)
            mask[:] = cv2.GC_BGD
            ix1, iy1 = max(0, x1), max(0, y1)
            ix2, iy2 = min(w, x2), min(h, y2)
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = cv2.GC_PR_FGD
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(bgr.copy(), mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
                grab = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
                if grab.sum() >= 0.02 * foreground.size:
                    fg = (fg & grab.astype(np.uint8)) | (fg & (mask == cv2.GC_FGD).astype(np.uint8))
        except Exception:
            pass  # HSV-only fallback

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx((fg > 0).astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest
    return fg.astype(bool)


# ---------------------------------------------------------------------------
# 2. Polygon rasterization
# ---------------------------------------------------------------------------
def polygons_to_mask(shape: Tuple[int, int], polygons: Iterable[Sequence[Sequence[float]]],
                     points: Optional[Iterable[Sequence[float]]] = None,
                     scale: float = 1.0, point_radius: int = 6) -> np.ndarray:
    cv2 = _cv2()
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for poly in polygons or []:
        try:
            pts = np.array([[float(p[0]) / scale, float(p[1]) / scale] for p in poly],
                           dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 1)
        except Exception:
            continue
    for p in points or []:
        try:
            x = int(round(float(p[0]) / scale)); y = int(round(float(p[1]) / scale))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(m, (x, y), point_radius, 1, -1)
        except Exception:
            continue
    return m.astype(bool)


# ---------------------------------------------------------------------------
# 3. Localization: primary -> retry -> saliency fallback, with TTA
# ---------------------------------------------------------------------------
def _saliency_points(bgr: np.ndarray, max_points: int = 6) -> List[List[int]]:
    """CPU-light 'deviation from dominant healthy hue' point prompts."""
    cv2 = _cv2()
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) &
             (hsv[..., 1] > 40) & (hsv[..., 2] > 40))
    cand = ((~green) & (hsv[..., 2] > 45) & (hsv[..., 2] < 250) &
            (hsv[..., 1] > 25)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    n, _, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 20:
            comps.append((area, float(cents[i][0]), float(cents[i][1])))
    comps.sort(reverse=True)
    pts = [[int(round(x)), int(round(y))] for _, x, y in comps[:max_points]]
    anchor = [w // 2, h // 2]
    if anchor not in pts:
        pts.append(anchor)
    return pts[:max_points + 1]


def _one_view(image_path: str, *, model: str, reasoning: str) -> Dict[str, Any]:
    key = get_api_key()
    r1 = chat_json(POLYGON_PRIMARY_V7, POLYGON_USER_V7, image_path, model=model,
                   max_output_tokens=FROZEN_CONFIG_V7["t1_max_output_tokens"],
                   reasoning_effort=reasoning, key=key)
    ann = r1.parsed
    source = "primary"
    tokens_in = r1.stats.input_tokens
    tokens_out = r1.stats.output_tokens
    seconds = r1.stats.latency_s

    def has_target(a):
        return bool(a and (a.get("lesions") or a.get("pos_points")))

    if not has_target(ann):
        r2 = chat_json(POLYGON_RETRY_V7, POLYGON_USER_V7, image_path, model=model,
                       max_output_tokens=FROZEN_CONFIG_V7["t1_max_output_tokens"],
                       reasoning_effort=reasoning, key=key)
        tokens_in += r2.stats.input_tokens
        tokens_out += r2.stats.output_tokens
        seconds += r2.stats.latency_s
        if has_target(r2.parsed):
            ann = r2.parsed
            source = "retry"
    if not has_target(ann):
        ann = {"leaf_bbox": None, "lesions": [], "pos_points": []}
        source = "fallback"

    cv2 = _cv2()
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError("cannot read image: " + image_path)
    h, w = bgr.shape[:2]
    polys, symptoms, points = [], [], []
    for les in (ann.get("lesions") or [])[:12]:
        if not isinstance(les, dict):
            continue
        poly = les.get("polygon") or les.get("poly")
        if poly and len(poly) >= 3:
            polys.append(poly)
            symptoms.append(str(les.get("symptom", "other")))
    for p in (ann.get("pos_points") or [])[:8]:
        try:
            points.append([float(p[0]), float(p[1])])
        except Exception:
            continue
    if source == "fallback":
        points = [[float(x), float(y)] for x, y in _saliency_points(bgr)]

    leaf_bbox = ann.get("leaf_bbox")
    mask = polygons_to_mask((h, w), polys, points)
    return {
        "mask": mask, "polys": polys, "symptoms": symptoms, "points": points,
        "leaf_bbox": leaf_bbox, "prompt_source": source,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "seconds": seconds,
    }


def locate_and_clip(image_path: str, *, model: str = "deepseek-flash",
                    reasoning: str = "high", tta: bool = True,
                    tta_combine: str = "union") -> Dict[str, Any]:
    """Full T3 prediction for one image. Returns DI fraction, mask and audit info."""
    cv2 = _cv2()
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError("cannot read image: " + image_path)
    h, w = bgr.shape[:2]

    view1 = _one_view(image_path, model=model, reasoning=reasoning)
    tokens_in = view1["tokens_in"]
    tokens_out = view1["tokens_out"]
    seconds = view1["seconds"]
    source = view1["prompt_source"]

    pred = view1["mask"].copy()
    if tta:
        flipped = cv2.flip(bgr, 1)
        with tempfile.TemporaryDirectory(prefix="v7_tta_") as td:
            fp = os.path.join(td, "flip.png")
            cv2.imwrite(fp, flipped)
            view2 = _one_view(fp, model=model, reasoning=reasoning)
        tokens_in += view2["tokens_in"]
        tokens_out += view2["tokens_out"]
        seconds += view2["seconds"]
        if view2["prompt_source"] != "primary":
            source = source if source != "primary" else "tta_" + view2["prompt_source"]
        m2 = cv2.flip(view2["mask"], 1)
        if tta_combine == "intersection":
            pred = pred & m2
        elif tta_combine == "average":
            pred = (pred.astype(np.float32) + m2.astype(np.float32)) >= 1.0
        else:
            pred = pred | m2

    lm = leaf_mask(bgr, view1["leaf_bbox"])
    final = pred & lm
    di = float(final.sum()) / float(h * w)
    return {
        "di": di, "mask": final, "leaf_mask": lm,
        "leaf_bbox": view1["leaf_bbox"],
        "symptoms": view1["symptoms"],
        "prompt_source": source,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "seconds": round(seconds, 3),
        "n_polygons": len(view1["polys"]), "n_points": len(view1["points"]),
    }


# ---------------------------------------------------------------------------
# 4. Symptom-wise quantile calibration (TRAIN-ONLY, leakage-guarded)
# ---------------------------------------------------------------------------
def fit_calibration(train_rows: Sequence[Dict[str, Any]],
                    evaluation_ids: Iterable[str],
                    q: float = 0.75) -> Dict[str, Any]:
    """Fit per-symptom multiplicative coefficients on training images ONLY.

    `train_rows` items need: sample_id, symptom, pred_fraction, gt_fraction.
    Hard guard: any train sample_id appearing in `evaluation_ids` aborts the fit.
    """
    eval_ids = {str(x) for x in evaluation_ids}
    overlapping = [str(r.get("sample_id")) for r in train_rows
                   if str(r.get("sample_id")) in eval_ids]
    if overlapping:
        raise ValueError("CALIBRATION LEAKAGE: training rows overlap evaluation set: "
                         + ", ".join(overlapping[:8]))
    by_symptom: Dict[str, List[Tuple[float, float]]] = {}
    for r in train_rows:
        sym = str(r.get("symptom", "other"))
        try:
            p = float(r["pred_fraction"]); g = float(r["gt_fraction"])
        except Exception:
            continue
        if np.isfinite(p) and np.isfinite(g) and p > 0 and g >= 0:
            by_symptom.setdefault(sym, []).append((p, g))

    coeffs: Dict[str, float] = {}
    for sym, vals in by_symptom.items():
        if len(vals) < 8:
            continue  # too few training samples: keep identity
        pvals = np.array([v[0] for v in vals]); gvals = np.array([v[1] for v in vals])
        c = float(np.quantile(gvals, q) / max(np.quantile(pvals, q), 1e-6))
        coeffs[sym] = float(np.clip(c, 0.3, 3.0))
    return {
        "status": "fitted" if coeffs else "identity_insufficient_train_data",
        "quantile": q,
        "coefficients": coeffs,
        "n_train_rows": len(train_rows),
        "evaluation_ids_excluded": True,
    }


def apply_calibration(di: float, symptoms: Sequence[str], calibration: Optional[Dict[str, Any]]) -> float:
    if not calibration or calibration.get("status") != "fitted":
        return float(di)
    coeffs = calibration.get("coefficients", {})
    if not coeffs:
        return float(di)
    vals = [coeffs[s] for s in symptoms if s in coeffs]
    c = float(np.mean(vals)) if vals else 1.0
    return float(np.clip(di * c, 0.0, 1.0))


def selfcheck() -> Dict[str, Any]:
    """Network-free logic check with synthetic polygons (no cv2 required for JSON parts)."""
    cal = fit_calibration(
        [{"sample_id": "tr%d" % i, "symptom": "mosaic", "pred_fraction": 0.2,
          "gt_fraction": 0.1} for i in range(10)],
        ["eval_1"],
    )
    assert cal["status"] == "fitted"
    assert abs(apply_calibration(0.2, ["mosaic"], cal) - 0.1) < 1e-6
    try:
        fit_calibration([{"sample_id": "eval_1", "symptom": "mosaic",
                          "pred_fraction": 0.2, "gt_fraction": 0.1}], ["eval_1"])
        raise AssertionError("leakage guard failed")
    except ValueError:
        pass
    return {"calibration": "ok", "leakage_guard": "ok"}
