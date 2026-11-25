#!/usr/bin/env python3
"""FROZEN v4 routing policy — byte-for-byte logic from scripts/19_v4_agent_final.py.
This module MUST NOT be modified during v5 external validation.
Input: MLLM zero-shot image JSON {background, lesion_morph}. No filename/batch/crop.
"""
import hashlib

POLICY_SOURCE = "v4/scripts/19_v4_agent_final.py::route()"
POLICY_FROZEN_AT = "2026-11-22"

def route(dec):
    # --- begin verbatim frozen logic ---
    bg = dec.get("background", "natural")
    morph = dec.get("lesion_morph", "")
    if bg == "dark":
        return "color", "dark-bg detached-leaf panel -> per-pixel color area"
    if bg == "pale" and morph == "isolated_disc":
        return "geometry", "pale-bg single isolated disc -> semantic bbox geometry"
    return "color", "diffuse/scattered or natural bg -> per-pixel color area"
    # --- end verbatim frozen logic ---

def policy_hash():
    """SHA256 of the routing function source for auditability."""
    import inspect
    return hashlib.sha256(inspect.getsource(route).encode()).hexdigest()

if __name__ == "__main__":
    print("POLICY_SOURCE:", POLICY_SOURCE)
    print("FROZEN_AT:", POLICY_FROZEN_AT)
    print("SHA256:", policy_hash())
    # sanity check reproduces v4 decisions
    tests = [
        ({"background": "pale", "lesion_morph": "isolated_disc"}, "geometry"),
        ({"background": "dark", "lesion_morph": "isolated_disc"}, "color"),
        ({"background": "pale", "lesion_morph": "diffuse_necrosis"}, "color"),
        ({"background": "natural", "lesion_morph": "scattered_flecks"}, "color"),
    ]
    for dec, want in tests:
        got, _ = route(dec)
        assert got == want, (dec, got, want)
    print("SELFTEST: OK (%d cases)" % len(tests))
