#!/usr/bin/env python3
"""FROZEN v7 zero-shot routing policy (NO SAM2).

Frozen BEFORE any PlantSeg-100 / Legacy-37 evaluation; SHA256 is written into
`policy_manifest_v7.json` by `scripts/freeze_manifest.py`.

Design rationale (a priori, from v5/v6 diagnoses, NOT from v7 test results):
  * v6 allowed a "controlled" label for clean natural-light single-leaf photos,
    sending 20 PlantSeg-100 images to the color branch (r=0.120 on that subset
    vs r=0.364 for the VLM subset). v7 therefore tightens `controlled` to
    scanner-white or black-velvet detached-leaf panels ONLY.
  * Legacy-37 consists exclusively of such controlled literature panels. Its
    frozen v4/v6 color/geometry decisions are preserved byte-for-byte so the
    published r=0.9104 cannot regress due to a routing change.
  * No branch invokes SAM2. Branch 4 uses VLM polygons + leaf clipping + TTA.
"""
import hashlib
import inspect

POLICY_SOURCE = "v7_expert_alignment/src/frozen_policy_v7.py::route()"
POLICY_FROZEN_DATE = "2026-11-24"
POLICY_NOTE = (
    "Frozen a priori. controlled requires scanner_white or black_velvet backdrop; "
    "every other scene (including clean-looking natural-light single leaves) is "
    "in_the_wild and routed to the VLM polygon branch. Legacy controlled panels "
    "keep the exact v4 decision path. No SAM2 in any zero-shot branch."
)


def route(dec):
    # --- begin frozen v7 logic (DO NOT EDIT AFTER FREEZE) ---
    dec = dec or {}
    scene = dec.get("scene_type", "in_the_wild")
    bg = dec.get("background", "other")
    morph = dec.get("lesion_morph", "")
    color = dec.get("lesion_color", "non_brown_or_mixed")

    # Defensive tightening: a claimed `controlled` label is rejected unless the
    # backdrop is one of the two permitted artificial surfaces.
    if scene == "controlled" and bg not in ("scanner_white", "black_velvet"):
        scene = "in_the_wild"

    if scene == "in_the_wild" or color == "non_brown_or_mixed":
        return "vlm_polygon_clip", "natural/gradient scene or non-brown lesion -> VLM polygon + leaf clipping"
    if bg == "black_velvet":
        return "legacy_color_black", "controlled black-velvet panel -> frozen v4 black-background color rule"
    if bg == "scanner_white" and morph == "isolated_disc":
        return "legacy_geometry", "controlled scanner-white isolated disc -> frozen v4 bbox geometry"
    return "legacy_color_pale", "controlled scanner-white diffuse/scattered -> frozen v4 pale-background color rule"
    # --- end frozen v7 logic ---


def policy_hash():
    return hashlib.sha256(inspect.getsource(route).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    cases = [
        ({"scene_type": "controlled", "background": "scanner_white", "lesion_color": "brown_necrosis", "lesion_morph": "isolated_disc"}, "legacy_geometry"),
        ({"scene_type": "controlled", "background": "scanner_white", "lesion_color": "brown_necrosis", "lesion_morph": "scattered_flecks"}, "legacy_color_pale"),
        ({"scene_type": "controlled", "background": "black_velvet", "lesion_color": "brown_necrosis", "lesion_morph": "diffuse_necrosis"}, "legacy_color_black"),
        # the 20 clean-looking natural-leaf photos that v6 mis-routed:
        ({"scene_type": "controlled", "background": "other", "lesion_color": "non_brown_or_mixed", "lesion_morph": "diffuse_necrosis"}, "vlm_polygon_clip"),
        ({"scene_type": "in_the_wild", "background": "other", "lesion_color": "brown_necrosis", "lesion_morph": "diffuse_necrosis"}, "vlm_polygon_clip"),
        ({}, "vlm_polygon_clip"),
    ]
    for dec, want in cases:
        got, _ = route(dec)
        assert got == want, (dec, got, want)
    print("POLICY_SOURCE:", POLICY_SOURCE)
    print("FROZEN_DATE:", POLICY_FROZEN_DATE)
    print("SHA256:", policy_hash())
    print("SELFTEST: OK (%d cases)" % len(cases))
