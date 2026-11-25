#!/usr/bin/env python3
"""Frozen metric definitions for v7. Pure numpy (no scipy) so they are testable offline.

Pearson r, Spearman rho (tie-corrected), Lin's CCC, MAE, mIoU, Dice.
All functions return float('nan') for degenerate inputs instead of raising.
"""
from __future__ import annotations

import math

import numpy as np


def _clean(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def pearson(x, y) -> float:
    x, y = _clean(x, y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    a = np.asarray(a, dtype=float).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x, y = _clean(x, y)
    if len(x) < 2:
        return float("nan")
    return pearson(_rankdata(x), _rankdata(y))


def pvalue_r(r: float, n: int) -> float:
    """Two-sided t-test p-value for a Pearson correlation (no scipy needed)."""
    if not np.isfinite(r) or n < 3 or abs(r) >= 1:
        return float("nan")
    df = n - 2
    # Two-sided t-test: p = I_{df/(df+t^2)}(df/2, 1/2)
    t = abs(r) * np.sqrt(df / max(1 - r * r, 1e-300))
    x = df / (df + t * t) if t != 0 else 1.0
    return float(max(0.0, min(1.0, _betainc(df / 2.0, 0.5, x))))


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 500, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def ccc(x, y) -> float:
    x, y = _clean(x, y)
    if len(x) < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    den = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    if den <= 0:
        return float("nan")
    return float(2 * np.cov(x, y, ddof=1)[0, 1] / den)


def mae(x, y) -> float:
    x, y = _clean(x, y)
    return float(np.mean(np.abs(x - y))) if len(x) else float("nan")


def mask_metrics(pred, gt) -> tuple[float, float]:
    p = np.asarray(pred, dtype=bool).ravel()
    g = np.asarray(gt, dtype=bool).ravel()
    inter = float(np.count_nonzero(p & g))
    union = float(np.count_nonzero(p | g))
    total = float(p.sum() + g.sum())
    return (inter / union if union else 0.0,
            2 * inter / total if total else 0.0)


def summarize(y_true, y_pred, ious=None, dices=None) -> Dict[str, float]:
    r = pearson(y_true, y_pred)
    n = len(_clean(y_true, y_pred)[0])
    return {
        "n": n,
        "r": r,
        "rho": spearman(y_true, y_pred),
        "ccc": ccc(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "p": pvalue_r(r, n),
        "mean_iou": float(np.nanmean(ious)) if ious is not None and len(ious) else float("nan"),
        "mean_dice": float(np.nanmean(dices)) if dices is not None and len(dices) else float("nan"),
    }


def selfcheck() -> Dict[str, float]:
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = x * 2 + 0.1
    s = summarize(x, y)
    assert abs(s["r"] - 1.0) < 1e-9, s
    assert abs(s["rho"] - 1.0) < 1e-9, s
    assert abs(s["mae"] - 0.45) < 1e-9, s
    assert abs(pvalue_r(0.9, 30)) < 1e-6
    m = mask_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert abs(m[0] - 0.5) < 1e-9 and abs(m[1] - 2 / 3) < 1e-9, m
    return s
