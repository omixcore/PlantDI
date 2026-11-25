#!/usr/bin/env python3
"""T4 publication figures: benchmark bar/point comparison, oracle upper bound style.

Style follows the frozen v7 spec: 7pt Arial, 0.5pt pure-black lines, no colored fill,
SVG + PDF. Uses matplotlib (R/ggplot2 unavailable locally).
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7,
    "axes.linewidth": 0.5, "lines.linewidth": 0.5,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "svg.fonttype": "none",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
CSV = os.path.join(ROOT, "results", "benchmark_v7_summary.csv")

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["domain"] == "plantseg100_raw" and r["status"] == "COMPUTED"]
# select one representative per method, prefer Post-hoc/ZS computed
sel = []
seen = set()
for r in rows:
    key = r["method"]
    if key in seen:
        continue
    seen.add(key)
    sel.append(r)
order = ["imagej", "pure_sam2", "agent_v6", "samfree_ablation",
         "pure_vlm", "v7_zero_shot", "sam2_base_oracle", "supervised_ft_sam2"]
sel = [next((r for r in sel if r["method"] == m), None) for m in order]
sel = [r for r in sel if r]

metrics = [("r", "Pearson r"), ("ccc", "Lin's CCC"),
           ("mae", "MAE"), ("mean_iou", "mIoU"), ("mean_dice", "Dice")]
fig, axes = plt.subplots(1, 5, figsize=(13.0, 3.0))
labels = [r["method"] for r in sel]
for ax, (col, ylab) in zip(axes, metrics):
    vals = [float(r[col]) if r[col] else float("nan") for r in sel]
    x = range(len(vals))
    ax.plot(x, vals, marker="o", markersize=3.5, markerfacecolor="none",
            markeredgecolor="black", markeredgewidth=0.5, color="black",
            linestyle="None")
    for i, v in enumerate(vals):
        ax.plot([i, i], [0, v], color="black", linewidth=0.5, alpha=0.4)
    ax.set_ylabel(ylab)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # shade supervised upper-bound region (last two)
    ax.axvspan(len(vals) - 2 - 0.5, len(vals) - 0.5, color="black", alpha=0.04)
fig.suptitle("PlantSeg-100 (n=100): zero-shot vs supervised upper bound (oracle box)",
             fontsize=7)
fig.text(0.5, 0.01, "Last two bars = oracle GT-box prompt (optimistic upper bound, not deployment-comparable)",
         ha="center", fontsize=6)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
for ext in ("svg", "pdf"):
    fig.savefig(os.path.join(FIG, "fig_v7_t4_benchmark." + ext), dpi=300)
print("saved fig_v7_t4_benchmark.{svg,pdf}")
