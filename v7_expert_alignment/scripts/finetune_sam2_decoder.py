#!/usr/bin/env python3
"""T4 - Supervised Domain-Adapted Upper Bound (GPU, SAM2 tiny mask decoder).

Freezes the image encoder and the prompt encoder; trains ONLY sam_mask_decoder.
Prompts are the official train-split annotation boxes (converted xywh -> xyxy),
NEVER GT masks and NEVER test images. All isolation was verified in
prepare_t4_data.py (audit: 0 overlap with official Test and PlantSeg-100).

Works with the official SAM2 API (facebookresearch/sam2, v1.0):
  predictor.set_image() is @torch.no_grad -> encoder frozen + no grad graph.
  box prompt -> prompt encoder(points=(coords,labels), boxes=None) with labels [2,3].
  decoder.forward(image_embeddings, image_pe, sparse, dense, multimask_output,
                   repeat_image, high_res_features).

Loss is computed on the model's low-res mask logits against a GTDownsample
using F.interpolate, so no full-resolution mask postprocessing is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    inter = (prob * target).sum()
    dice = 1.0 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def load_items(meta_csv: str, imgdir: str, maskdir: str, prompt_json: str,
               limit: int = 0) -> list:
    import csv
    cache = json.load(open(prompt_json, encoding="utf-8"))
    rows = list(csv.DictReader(open(meta_csv, encoding="utf-8")))
    if limit > 0:
        rows = rows[:limit]
    items = []
    for r in rows:
        sid = r["sample_id"]
        ip = os.path.join(imgdir, sid + ".jpg")
        mp = os.path.join(maskdir, sid + ".png")
        if sid in cache and os.path.exists(ip) and os.path.exists(mp):
            items.append((sid, ip, mp, cache[sid]["boxes"][0]))
    return items


def train_one_epoch(model, predictor, decoder, opt, items, device, log_every) -> float:
    model.train()
    decoder.train()
    total = 0.0
    n = 0
    for sid, ip, mp, box in items:
        image = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gt_full = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if image is None or gt_full is None:
            continue
        # set_image is @torch.no_grad -> image encoder produces embeddings only.
        predictor.set_image(image)
        feats = predictor._features
        if feats is None:
            continue
        img_embed = feats["image_embed"][0:1]
        high_feats = [f[0:1] for f in feats["high_res_feats"]]

        # box prompt in the ORIGINAL image pixel coordinate system
        x1, y1, x2, y2 = [float(v) for v in box]
        # SAM2 expects box coords shaped (B, 2, 2) with labels (B, 2) = [2, 3]
        coords = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
        labels = torch.tensor([[2, 3]], dtype=torch.int64, device=device)
        with torch.no_grad():
            sparse, dense = model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            image_pe = model.sam_prompt_encoder.get_dense_pe()

        # prompt encoder + image encoder stay frozen, decoder is trainable
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            low_logits, _, _, _ = decoder(
                image_embeddings=img_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_feats,
            )
        low_logits = low_logits.float()

        # low-res target: downsample GT to match decoder output spatial size
        h, w = original_hw = predictor._orig_hw[0]
        gh, gw = low_logits.shape[-2:]
        gt_t = torch.as_tensor(gt_full > 0, dtype=torch.float32, device=device)[None, None]
        gt_low = F.interpolate(gt_t, size=(gh, gw), mode="bilinear", align_corners=False)
        gt_low = (gt_low > 0.5).float()

        loss = dice_bce(low_logits, gt_low)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in decoder.parameters() if p.requires_grad], 1.0)
        opt.step()
        total += float(loss.detach())
        n += 1
        if log_every > 0 and n % log_every == 0:
            print("  batch %d/%d loss=%.5f (running %.5f)" %
                  (n, len(items), float(loss.detach()), total / n), flush=True)
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2-root", required=True, help="path to sam2 source root (for config lookup)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--maskdir", required=True)
    ap.add_argument("--prompt-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-cfg", default="configs/sam2/sam2_hiera_t.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=250)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # `import sam2` auto-initializes hydra's config module search path (see sam2/__init__.py).
    # Config names are relative to the sam2 package root, e.g. "configs/sam2/sam2_hiera_t.yaml".
    cfg_name = args.model_cfg
    if not cfg_name.startswith("configs/"):
        cfg_name = "configs/sam2/" + os.path.basename(cfg_name)
    verify = os.path.join(args.sam2_root, "sam2", cfg_name)
    if not os.path.exists(verify):
        raise FileNotFoundError("model config not found: " + verify)

    items = load_items(args.meta, args.imgdir, args.maskdir, args.prompt_cache, args.limit)
    print("T4 items:", len(items), "device:", device, flush=True)
    assert len(items) > 0

    print("loading SAM2 from", args.checkpoint, flush=True)
    model = build_sam2(config_file=cfg_name, ckpt_path=args.checkpoint,
                       device=device, mode="eval")
    predictor = SAM2ImagePredictor(model)

    # Freeze everything; unfreeze ONLY mask decoder.
    for p in model.parameters():
        p.requires_grad_(False)
    decoder = model.sam_mask_decoder
    for p in decoder.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("trainable decoder params:", n_trainable, flush=True)

    opt = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad], lr=args.lr)

    t0 = time.time()
    for ep in range(args.epochs):
        random.shuffle(items)
        mean_loss = train_one_epoch(model, predictor, decoder, opt, items,
                                    device, args.log_every)
        print("epoch %d/%d mean_loss=%.5f elapsed=%.1fs" %
              (ep + 1, args.epochs, mean_loss, time.time() - t0), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"sam_mask_decoder": decoder.state_dict(),
                "config": args.model_cfg,
                "epochs": args.epochs,
                "lr": args.lr,
                "n_items": len(items),
                "seed": args.seed}, args.out)
    audit = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "out": os.path.abspath(args.out),
        "out_sha256": sha256(args.out),
        "n_items": len(items),
        "trainable_decoder_params": n_trainable,
        "frozen": "image_encoder + prompt_encoder",
        "trained": "sam_mask_decoder only",
        "prompt_source": "official train annotation bboxes (xywh->xyxy), NOT GT mask, NOT VLM",
        "device": device,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
        "isolation": "verified in t4_data_audit.json: 0 overlap with official Test + PlantSeg-100",
    }
    apath = os.path.splitext(args.out)[0] + "_audit.json"
    with open(apath, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    print("saved:", args.out, flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
