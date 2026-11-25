#!/usr/bin/env python3
"""Parallel ranged downloader for the large CUDA torch wheel.

Uses 16 concurrent HTTP range requests with retries, then concatenates parts.
Verifies total size against the expected content-length. No dependency on aria2.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl"
EXPECTED = 2510680879
OUT = sys.argv[1] if len(sys.argv) > 1 else "torch-2.5.1+cu124-cp312-cp312-win_amd64.whl"
NPARTS = 16
PARTDIR = OUT + ".parts"


def fetch_part(idx: int, start: int, end: int, retries: int = 4) -> tuple:
    part = os.path.join(PARTDIR, "part_%02d" % idx)
    want = end - start + 1
    for attempt in range(retries):
        if os.path.exists(part) and os.path.getsize(part) == want:
            return idx, want, True
        try:
            req = urllib.request.Request(URL, headers={"Range": "bytes=%d-%d" % (start, end),
                                                       "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == want:
                with open(part, "wb") as fh:
                    fh.write(data)
                return idx, want, True
            # some servers ignore Range and return full body -> abort this attempt
        except Exception as e:
            pass
        time.sleep(1 + attempt)
    return idx, os.path.getsize(part) if os.path.exists(part) else 0, False


def main() -> None:
    os.makedirs(PARTDIR, exist_ok=True)
    chunk = (EXPECTED + NPARTS - 1) // NPARTS
    ranges = []
    for i in range(NPARTS):
        s = i * chunk
        e = min(EXPECTED - 1, s + chunk - 1)
        if s <= e:
            ranges.append((i, s, e))
    print("downloading %d parts, chunk=%d bytes" % (len(ranges), chunk), flush=True)
    t0 = time.time()
    ok = True
    with ThreadPoolExecutor(max_workers=NPARTS) as pool:
        futs = [pool.submit(fetch_part, i, s, e) for i, s, e in ranges]
        done_n = 0
        for f in as_completed(futs):
            idx, size, success = f.result()
            done_n += 1
            if not success:
                ok = False
                print("PART %d FAILED (size=%d)" % (idx, size), flush=True)
            if done_n % 4 == 0 or done_n == len(ranges):
                print("progress %d/%d parts, %.1fs" % (done_n, len(ranges), time.time() - t0), flush=True)
    if not ok:
        raise RuntimeError("some parts failed; rerun to resume")
    with open(OUT, "wb") as out:
        for i, _, _ in ranges:
            with open(os.path.join(PARTDIR, "part_%02d" % i), "rb") as fh:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
    actual = os.path.getsize(OUT)
    print("assembled %s size=%d expected=%d elapsed=%.1fs" % (OUT, actual, EXPECTED, time.time() - t0), flush=True)
    assert actual == EXPECTED, "size mismatch"
    print("DOWNLOAD OK")


if __name__ == "__main__":
    main()
