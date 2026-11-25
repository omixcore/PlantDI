#!/usr/bin/env python3
"""Generate `policy_manifest_v7.json` BEFORE any v7 evaluation run.

Usage:  python v7_expert_alignment/scripts/freeze_manifest.py
The generated manifest pins SHA256 of:
  * the frozen routing function source,
  * every frozen prompt string,
  * every v7 source file,
  * immutable historical baseline CSVs (integrity guard).
If any pinned file changes afterwards, `verify_manifest.py` fails and the run
must be repeated from scratch. Never regenerate this manifest to "improve" a
test result -- that would be target leakage.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
V7 = os.path.dirname(HERE)
SRC = os.path.join(V7, "src")
PROJECT = os.path.dirname(V7)
sys.path.insert(0, SRC)

from frozen_policy_v7 import POLICY_SOURCE, POLICY_FROZEN_DATE, POLICY_NOTE, policy_hash  # noqa: E402
from prompts_v7 import FROZEN_CONFIG_V7, PROMPT_REGISTRY_V7  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def main():
    source_files = {}
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".py"):
            p = os.path.join(SRC, fn)
            source_files["src/" + fn] = {
                "sha256": sha256_file(p),
                "bytes": os.path.getsize(p),
            }

    prompt_hashes = {
        name: sha256_bytes(text.encode("utf-8"))
        for name, text in PROMPT_REGISTRY_V7.items()
    }

    baseline_files = {
        "v6_three_branch/results/stats_summary_v6.csv": os.path.join(PROJECT, "v6_three_branch/results/stats_summary_v6.csv"),
        "v6_three_branch/ablation_samfree/stats_summary_samfree.csv": os.path.join(PROJECT, "v6_three_branch/ablation_samfree/stats_summary_samfree.csv"),
        "v5_formal/results/stats_summary_v5.csv": os.path.join(PROJECT, "v5_formal/results/stats_summary_v5.csv"),
    }
    baseline_hashes = {
        k: {"sha256": sha256_file(v), "bytes": os.path.getsize(v)}
        for k, v in baseline_files.items()
        if os.path.exists(v)
    }

    manifest = {
        "manifest_version": "v7.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_evaluation": True,
        "frozen_date": POLICY_FROZEN_DATE,
        "note": POLICY_NOTE,
        "policy": {
            "source": POLICY_SOURCE,
            "route_sha256": policy_hash(),
        },
        "prompts": prompt_hashes,
        "frozen_config": FROZEN_CONFIG_V7,
        "source_files": source_files,
        "immutable_baseline_sha256": baseline_hashes,
        "integrity_rule": (
            "Any change to policy/prompts/config/source after this manifest was "
            "generated invalidates all v7 results. Re-freeze and re-run from zero; "
            "never tune on the test split."
        ),
    }
    out = os.path.join(V7, "policy_manifest_v7.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("wrote", out)
    print("route sha256:", manifest["policy"]["route_sha256"])
    print("prompts:", len(prompt_hashes), "source files:", len(source_files))


if __name__ == "__main__":
    main()
