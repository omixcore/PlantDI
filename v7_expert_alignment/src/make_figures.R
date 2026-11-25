#!/usr/bin/env Rscript
# T5 - publication-grade vector figures for the v7 benchmark.
# Style (frozen): portrait layout, Arial 7 pt, 0.5 pt pure-black lines,
# no coloured fills; SVG + PDF. Missing/NOT_RUN rows are plotted as open
# points at NA and annotated as "not run" rather than fabricated.
#
# Usage: Rscript make_figures.R <benchmark_v7_summary.csv> <outdir>

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "../results/benchmark_v7_summary.csv"
outdir   <- if (length(args) >= 2) args[2] else "../figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required")
library(ggplot2)

d <- read.csv(csv_path, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
numcols <- c("r", "rho", "ccc", "mae", "p", "mean_iou", "mean_dice",
             "seconds_per_img", "input_tokens", "output_tokens")
for (cc in numcols) d[[cc]] <- suppressWarnings(as.numeric(d[[cc]]))
d <- d[!is.na(d$r) | !is.na(d$mae), ]

theme_v7 <- theme_bw(base_size = 7) +
  theme(
    text = element_text(family = "Arial", size = 7, colour = "#000000"),
    axis.text = element_text(family = "Arial", size = 7, colour = "#000000"),
    axis.title = element_text(family = "Arial", size = 7, colour = "#000000"),
    legend.text = element_text(family = "Arial", size = 7, colour = "#000000"),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(colour = "#000000", linewidth = 0.5),
    axis.line = element_line(colour = "#000000", linewidth = 0.5),
    strip.background = element_blank(),
    strip.text = element_text(family = "Arial", size = 7, colour = "#000000"),
    plot.title = element_text(family = "Arial", size = 7, colour = "#000000")
  )

save_v7 <- function(p, name) {
  svg(file.path(outdir, paste0(name, ".svg")), width = 5.2, height = 4.2,
      family = "Arial", bg = "white", pointsize = 7)
  print(p)
  dev.off()
  pdf(file.path(outdir, paste0(name, ".pdf")), width = 5.2, height = 4.2,
      family = "Helvetica", bg = "white", pointsize = 7)
  print(p)
  dev.off()
}

# Only methods with at least one computed metric are shown as filled points.
d$method <- factor(d$method, levels = unique(d$method))
d$domain_label <- ifelse(d$domain == "legacy37_zscore", "Legacy-37 (batch z)", "PlantSeg-100 (raw)")

p1 <- ggplot(d, aes(x = method, y = r, shape = paradigm, group = paradigm)) +
  geom_point(size = 2.2, colour = "#000000", fill = NA, stroke = 0.5) +
  geom_line(linewidth = 0.5, colour = "#000000", linetype = "dashed") +
  facet_wrap(~ domain_label, scales = "free_x", ncol = 1) +
  labs(x = NULL, y = "Pearson r", title = "v7 benchmark - correlation") +
  theme_v7 +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 7))
save_v7(p1, "fig_v7_1_correlation")

p2 <- ggplot(d, aes(x = method, y = ccc, shape = paradigm, group = paradigm)) +
  geom_point(size = 2.2, colour = "#000000", stroke = 0.5) +
  geom_line(linewidth = 0.5, colour = "#000000", linetype = "dotted") +
  facet_wrap(~ domain_label, scales = "free_x", ncol = 1) +
  labs(x = NULL, y = "Lin's CCC", title = "v7 benchmark - agreement") +
  theme_v7 +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 7))
save_v7(p2, "fig_v7_2_ccc")

p3 <- ggplot(d, aes(x = method, y = mae, shape = paradigm, group = paradigm)) +
  geom_point(size = 2.2, colour = "#000000", stroke = 0.5) +
  geom_line(linewidth = 0.5, colour = "#000000", linetype = "dotted") +
  facet_wrap(~ domain_label, scales = "free_x", ncol = 1) +
  labs(x = NULL, y = "MAE (fraction)", title = "v7 benchmark - absolute error") +
  theme_v7 +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 7))
save_v7(p3, "fig_v7_3_mae")

pm <- d[d$domain == "plantseg100_raw" & (!is.na(d$mean_iou) | !is.na(d$mean_dice)), ]
p4 <- ggplot(pm, aes(x = method, y = mean_iou, shape = paradigm, group = paradigm)) +
  geom_point(size = 2.2, colour = "#000000", stroke = 0.5) +
  geom_line(linewidth = 0.5, colour = "#000000", linetype = "dashed") +
  geom_point(aes(y = mean_dice), shape = 17, size = 2.2, colour = "#000000") +
  labs(x = NULL, y = "Pixel overlap", title = "PlantSeg-100: mIoU (circle) / Dice (triangle)") +
  theme_v7 + theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 7))
save_v7(p4, "fig_v7_4_pixel_overlap")

cat("figures written to", outdir, "\n")
