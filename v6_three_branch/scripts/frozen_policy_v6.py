#!/usr/bin/env python3
"""FROZEN v6 three-branch routing policy.

Frozen BEFORE any PlantSeg100 evaluation. Input is MLLM zero-shot image JSON
{scene_type, lesion_color, background, lesion_morph}. No filename / crop / batch
/ disease hints are ever used by route().

Priority (a priori, dictated by v5 diagnosis of 55% green-hue GT lesion pixels):
  Branch 3 (sam2):     scene_type == "in_the_wild" OR lesion_color == "non_brown_or_mixed"
  Branch 2 (geometry): scene_type == "controlled" AND background == "pale"
                       AND lesion_morph == "isolated_disc"
  Branch 1 (color):    all remaining controlled samples (brown on uniform bg)
"""
import hashlib
import inspect

POLICY_SOURCE = "v6_three_branch/scripts/frozen_policy_v6.py::route()"
POLICY_FROZEN_AT = "2026-11-22"
POLICY_NOTE = ("Frozen a priori from v5 diagnosis: 55% of expert GT lesion pixels in PlantSeg "
               "are green-hued; brown-on-green color threshold is invalid in-the-wild. "
               "No result-driven threshold tuning after freeze.")

def route(dec):
    # --- begin frozen v6 logic (DO NOT EDIT AFTER FREEZE) ---
    scene = dec.get("scene_type", "controlled")
    color = dec.get("lesion_color", "brown_necrosis")
    bg = dec.get("background", "natural")
    morph = dec.get("lesion_morph", "")
    if scene == "in_the_wild" or color == "non_brown_or_mixed":
        return "sam2", "in_the_wild / non-brown lesion -> foundation segmentation (SAM2)"
    if scene == "controlled" and bg == "pale" and morph == "isolated_disc":
        return "geometry", "controlled pale-bg isolated disc -> VLM bbox geometry"
    return "color", "controlled diffuse/scattered brown lesion -> per-pixel color area"
    # --- end frozen v6 logic ---

def policy_hash():
    return hashlib.sha256(inspect.getsource(route).encode()).hexdigest()

if __name__ == "__main__":
    print("POLICY_SOURCE:", POLICY_SOURCE)
    print("FROZEN_AT:", POLICY_FROZEN_AT)
    print("NOTE:", POLICY_NOTE)
    print("SHA256:", policy_hash())
    tests = [
        ({"scene_type":"controlled","lesion_color":"brown_necrosis","background":"pale","lesion_morph":"isolated_disc"}, "geometry"),
        ({"scene_type":"controlled","lesion_color":"brown_necrosis","background":"dark","lesion_morph":"scattered_flecks"}, "color"),
        ({"scene_type":"controlled","lesion_color":"brown_necrosis","background":"pale","lesion_morph":"diffuse_necrosis"}, "color"),
        ({"scene_type":"in_the_wild","lesion_color":"brown_necrosis","background":"natural","lesion_morph":"diffuse_necrosis"}, "sam2"),
        ({"scene_type":"controlled","lesion_color":"non_brown_or_mixed","background":"pale","lesion_morph":"isolated_disc"}, "sam2"),
        # legacy v4 compatibility: absent new fields defaults to controlled/brown -> v4 decisions preserved
        ({"background":"pale","lesion_morph":"isolated_disc"}, "geometry"),
        ({"background":"dark","lesion_morph":"scattered_flecks"}, "color"),
    ]
    for dec, want in tests:
        got, _ = route(dec)
        assert got == want, (dec, got, want)
    print("SELFTEST: OK (%d cases)" % len(tests))
