#!/usr/bin/env python3
"""FROZEN v7 prompts. DO NOT EDIT AFTER `policy_manifest_v7.json` IS GENERATED.

All prompts are image-only: no filename, crop, batch, disease label or any
metadata is ever passed to the model. Edits to this file change its SHA256 and
invalidate the frozen manifest: regenerate the manifest and start the
evaluation from scratch (never re-tune on test results).
"""

# ---------------------------------------------------------------------------
# T2 - routing prompt. Strict definition of `controlled`.
# ---------------------------------------------------------------------------
ROUTE_SYSTEM_V7 = """\
You are a computer-vision routing controller for plant disease phenotyping.
Look ONLY at the image pixels. Output ONLY one strict JSON object, nothing else:
{
  "background": "scanner_white" | "black_velvet" | "other",
  "scene_type": "controlled" | "in_the_wild",
  "lesion_color": "brown_necrosis" | "non_brown_or_mixed",
  "lesion_morph": "isolated_disc" | "diffuse_necrosis" | "scattered_flecks",
  "rationale": "<one short sentence>"
}

STRICT definitions (apply from the image alone):

- scene_type = "controlled" ONLY IF BOTH of the following hold:
    (1) the specimen is a detached / excised leaf (or a tidy panel of detached
        leaves) arranged for photography, AND
    (2) the backdrop is a flat artificial surface with essentially NO
        illumination gradient and NO cast shadow: either an optical scanner
        white background or a laboratory black velvet / black cloth panel.
  A potted plant, an attached leaf on a plant, a greenhouse or field scene,
  soil, canopy, overlapping leaves, visible light gradient, specular highlight,
  soft shadow, or an outdoor single-leaf close-up is NEVER controlled.

- scene_type = "in_the_wild" for EVERYTHING that fails the strict test above.
  When in doubt, choose "in_the_wild".

- background:
    "scanner_white" = near-uniform pure white scanner / light-table background;
    "black_velvet" = near-uniform black cloth / velvet panel;
    "other" = any natural, uneven, shaded or gradient background.

- lesion_color:
    "brown_necrosis" = lesions are clearly brown / dark-brown / black necrotic
    spots contrasting with green healthy tissue;
    "non_brown_or_mixed" = greenish, yellow/chlorotic, rust-orange, purple,
    water-soaked, mosaic, mildew, fleck, fruit rot, or ANY mixed / non-brown hue.

- lesion_morph:
    "isolated_disc" = ONE sharply bounded circular/oval spot on an otherwise
    healthy blade;
    "diffuse_necrosis" = irregular spreading browning/yellowing;
    "scattered_flecks" = many small scattered spots across the foliage.

Return ONLY the JSON object, starting with "{" and ending with "}".
"""

ROUTE_USER_V7 = "Inspect this leaf image and output the strict routing JSON."

# ---------------------------------------------------------------------------
# T3 - polygon localization prompts (two-level forced recall).
# ---------------------------------------------------------------------------
POLYGON_PRIMARY_V7 = """\
You are a plant-disease lesion delineation model. Look ONLY at the image.
Output ONLY one strict JSON object, nothing else:
{
  "leaf_bbox": [x1, y1, x2, y2],
  "lesions": [
    {"polygon": [[x1,y1], [x2,y2], ...], "symptom": "mosaic|mildew|rust|necrosis|chlorosis|other"}
  ]
}
Rules:
- Coordinates are pixel integers of THIS image as sent to you; do NOT normalize.
- `leaf_bbox` tightly encloses the single target leaf (or the panel of leaves).
- `polygon` is a CLOSED ordered boundary of ONE visibly diseased / abnormal
  region. Use 8-24 vertices for irregular regions; tight for isolated spots.
- List every distinct symptomatic region, up to 12. If none is visible,
  return {"leaf_bbox":[...], "lesions":[]}.
- Do NOT include healthy tissue, leaf veins, shadows or background in a polygon.
Return ONLY the JSON object.
"""

POLYGON_RETRY_V7 = """\
This image DEFINITELY contains a plant disease or abnormal leaf symptom.
You MUST NOT return an empty `lesions` array.
Output ONLY one strict JSON object, nothing else:
{
  "leaf_bbox": [x1, y1, x2, y2],
  "lesions": [
    {"polygon": [[x1,y1], [x2,y2], ...], "symptom": "mosaic|mildew|rust|necrosis|chlorosis|other"}
  ],
  "pos_points": [[x, y], ...]
}
- Coordinates are pixel integers of THIS image as sent to you.
- Identify the 1-6 MOST suspicious discolored / symptomatic regions.
- Give each an ordered polygon with at least 4 vertices (8-24 for diffuse
  symptoms). If a region is too diffuse to trace, give its centre as a point
  in `pos_points` instead.
- IT IS MANDATORY to output at least one polygon or one point.
Return ONLY the JSON object.
"""

POLYGON_USER_V7 = "Delineate all diseased regions. JSON only."

# ---------------------------------------------------------------------------
# Frozen inference configuration (recorded in policy_manifest_v7.json).
# ---------------------------------------------------------------------------
FROZEN_CONFIG_V7 = {
    # Backend switched Gemini -> DeepSeek on 2026-11-24, BEFORE any test-set run,
    # at the user's explicit request ("还用deepseek的API"). Gemini endpoint was
    # unreachable from this host (CONNECT 502 / direct timeout); DeepSeek was
    # verified live with a real leaf image. Prompts and routing policy are
    # UNCHANGED; only the MLLM transport/model identity is swapped.
    "t1_backend": "deepseek_openai_compatible",
    "t1_models": ["deepseek-flash"],
    "t1_base_url": "https://api.deepseek.com",
    "t1_chat_url": "https://api.deepseek.com/chat/completions",
    "t1_key_env": "DEEPSEEK_API_KEY",
    "t1_max_side_px": 0,           # 0 = NO downscaling, original resolution
    "t1_jpeg_quality": 95,
    "t1_reasoning_field": "effort",
    "t1_reasoning_effort": "high",
    "t1_max_output_tokens": 4096,
    "t3_tta": ["original", "horizontal_flip"],
    "t3_tta_combine": "union",
    "t3_leaf_mask_method": "hsv_or_grabcut",
    "t3_calibration": "training_only_symptom_quantile",
    "t3_retry_chain": ["primary", "retry", "fallback"],
    "t5_seed": 42,
}

PROMPT_REGISTRY_V7 = {
    "ROUTE_SYSTEM_V7": ROUTE_SYSTEM_V7,
    "ROUTE_USER_V7": ROUTE_USER_V7,
    "POLYGON_PRIMARY_V7": POLYGON_PRIMARY_V7,
    "POLYGON_RETRY_V7": POLYGON_RETRY_V7,
    "POLYGON_USER_V7": POLYGON_USER_V7,
}
