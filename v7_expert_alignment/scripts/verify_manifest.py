#!/usr/bin/env python3
"""Verify that no frozen v7 policy/prompt/config/source or historical baseline changed.

Run this at the START and END of every evaluation. A non-zero exit code means
the run is not auditable and must be discarded.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V7 = os.path.dirname(HERE)
PROJECT = os.path.dirname(V7)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    mf_path = os.path.join(V7, "policy_manifest_v7.json")
    if not os.path.exists(mf_path):
        raise SystemExit("FAIL: policy_manifest_v7.json missing; freeze before evaluating")
    mf = json.load(open(mf_path, encoding="utf-8"))
    problems = []

    for rel, meta in mf["source_files"].items():
        p = os.path.join(V7, rel)
        if not os.path.exists(p) or sha(p) != meta["sha256"]:
            problems.append("source changed/missing: " + rel)

    for rel, meta in mf["immutable_baseline_sha256"].items():
        p = os.path.join(PROJECT, rel)
        if not os.path.exists(p) or sha(p) != meta["sha256"]:
            problems.append("HISTORICAL BASELINE CHANGED: " + rel)

    sys.path.insert(0, os.path.join(V7, "src"))
    from frozen_policy_v7 import policy_hash  # noqa: E402
    from prompts_v7 import PROMPT_REGISTRY_V7  # noqa: E402
    if policy_hash() != mf["policy"]["route_sha256"]:
        problems.append("frozen route() source changed")

    for name, text in PROMPT_REGISTRY_V7.items():
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h != mf["prompts"].get(name):
            problems.append("frozen prompt changed: " + name)

    if problems:
        print("MANIFEST VERIFICATION: FAIL")
        for p in problems:
            print(" -", p)
        return 1
    print("MANIFEST VERIFICATION: OK")
    print("route sha256:", mf["policy"]["route_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
