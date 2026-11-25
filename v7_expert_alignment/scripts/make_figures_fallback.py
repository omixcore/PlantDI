#!/usr/bin/env python3
"""Publication-grade vector figures when R/ggplot2 is unavailable locally.

Plots ONLY rows with real numbers from benchmark_v7_summary.csv. NOT_RUN rows
are listed as text, never plotted as data. The canonical R script is
`src/make_figures.R`; this produces equivalent SVG/PDF with the frozen style:
portrait, Arial 7 pt, 0.5 pt pure-black lines, no coloured fills.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
V7 = os.path.dirname(HERE)
CSV = os.path.join(V7, "results/benchmark_v7_summary.csv")
OUT = os.path.join(V7, "figures")


def f(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    notrun = ["%s / %s" % (r["domain"], r["method"]) for r in rows if r["status"] == "NOT_RUN"]
    # De-duplicate by (method, paradigm), prefer the post-hoc samfree table for
    # samfree_ablation so mIoU/Dice are present; keep v7_zero_shot computed row.
    best = {}
    for r in rows:
        if r["domain"] != "plantseg100_raw" or r["status"] == "NOT_RUN":
            continue
        key = r["method"]
        if key not in best or (f(r["mean_iou"]) == f(r["mean_iou"]) and
                               (not (f(best[key]["mean_iou"]) == f(best[key]["mean_iou"])))):
            best[key] = r
    uniq = list(best.values())
    uniq.sort(key=lambda r: f(r["r"]), reverse=True)
    labels = []
    for r in uniq:
        labels.append(r["method"] + ("*" if r["method"] == "v7_zero_shot" else ""))

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.5, "lines.linewidth": 0.5,
        "axes.edgecolor": "#000000", "text.color": "#000000",
        "xtick.color": "#000000", "ytick.color": "#000000",
        "legend.fontsize": 7, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(4, 1, figsize=(5.2, 8.2))
    specs = [("r", "Pearson r"), ("ccc", "Lin's CCC"),
             ("mae", "MAE (fraction)"), ("mean_iou", "mIoU")]
    xs = list(range(len(uniq)))
    for ax, (col, ylab) in zip(axes, specs):
        for x, r in zip(xs, uniq):
            mk = "D" if r["method"] == "v7_zero_shot" else "o"
            ax.plot([x], [f(r[col])], marker=mk, markersize=3.5,
                    markerfacecolor="none", markeredgecolor="#000000",
                    markeredgewidth=0.5, color="#000000", linestyle="None")
        ax.set_ylabel(ylab)
        ax.grid(False)
        ax.axhline(0, color="#000000", linewidth=0.5)
    axes[0].set_title("PlantSeg-100 (raw) - frozen baselines vs v7 Zero-Shot (*)", fontsize=7)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    fig.text(0.08, 0.012, "NOT_RUN (no model/data yet): " + "; ".join(notrun),
             fontsize=6, color="#000000")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.11, hspace=0.5)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "pdf"):
        p = os.path.join(OUT, "fig_v7_benchmark." + ext)
        fig.savefig(p, format=ext, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
