"""
codecoy_v2.py — CoDecoy v2: BRS + Target-Clone Decoy (combo attack)

Key insight from v1 failure: a color ghost alone cannot hijack SAM2.
We need TWO simultaneous attacks:
  1. WEAKEN the real target (BRS on boundary — proven effective)
  2. ATTRACT to decoy (low-frequency clone of target shape in background)

The decoy is NOT a color blob. It is the target's silhouette (mask crop),
blurred and alpha-blended into a background location, creating a
"ghost object" that has similar shape/texture statistics to the real target.

Usage:
  python codecoy_v2.py --videos bear,blackswan,dog --device cuda --tag v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor, run_tracking, codec_round_trip, frame_quality,
    apply_old_boundary_suppression,
)


def find_decoy_location(mask: np.ndarray, min_dist: int = 60) -> tuple[int, int] | None:
    """Find background location far from object, with enough space for a clone."""
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    obj_cy, obj_cx = int(ys.mean()), int(xs.mean())
    obj_h = ys.max() - ys.min()
    obj_w = xs.max() - xs.min()
    pad_h, pad_w = obj_h // 2 + 10, obj_w // 2 + 10

    Y, X = np.mgrid[0:H, 0:W]
    dist = np.sqrt((Y - obj_cy)**2 + (X - obj_cx)**2)

    valid = (mask == 0) & (dist > min_dist)
    valid[:pad_h, :] = False
    valid[-pad_h:, :] = False
    valid[:, :pad_w] = False
    valid[:, -pad_w:] = False

    if not valid.any():
        return None
    dist[~valid] = 0
    idx = np.unravel_index(dist.argmax(), dist.shape)
    return int(idx[0]), int(idx[1])


def generate_clone_decoy(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    decoy_cy: int,
    decoy_cx: int,
    clone_alpha: float = 0.5,
    blur_sigma: float = 5.0,
    scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a low-frequency CLONE of the target at (decoy_cy, decoy_cx).

    Unlike v1's color ghost, this copies the actual target content (blurred)
    to the decoy location, preserving shape and texture statistics.
    """
    H, W = frame_rgb.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return frame_rgb.copy(), np.zeros_like(mask)

    # Crop the target region
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_h, crop_w = y1 - y0, x1 - x0

    target_crop = frame_rgb[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1].copy()

    # Scale down
    new_h = int(crop_h * scale)
    new_w = int(crop_w * scale)
    if new_h < 10 or new_w < 10:
        return frame_rgb.copy(), np.zeros_like(mask)

    target_resized = cv2.resize(target_crop, (new_w, new_h))
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Low-pass filter (codec-safe)
    target_blurred = cv2.GaussianBlur(target_resized.astype(np.float32), (0, 0), blur_sigma)

    # Soft mask for blending (feathered edges)
    mask_soft = cv2.GaussianBlur(mask_resized.astype(np.float32), (0, 0),
                                  max(blur_sigma / 2, 2.0))
    mask_soft = np.clip(mask_soft * clone_alpha, 0, clone_alpha)

    # Place at decoy location
    dy0 = decoy_cy - new_h // 2
    dx0 = decoy_cx - new_w // 2
    dy1 = dy0 + new_h
    dx1 = dx0 + new_w

    # Clip to frame bounds
    src_y0 = max(0, -dy0)
    src_x0 = max(0, -dx0)
    dst_y0 = max(0, dy0)
    dst_x0 = max(0, dx0)
    dst_y1 = min(H, dy1)
    dst_x1 = min(W, dx1)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        return frame_rgb.copy(), np.zeros_like(mask)

    edited = frame_rgb.astype(np.float32).copy()
    patch = target_blurred[src_y0:src_y1, src_x0:src_x1]
    w = mask_soft[src_y0:src_y1, src_x0:src_x1, None]

    edited[dst_y0:dst_y1, dst_x0:dst_x1] = (
        edited[dst_y0:dst_y1, dst_x0:dst_x1] * (1.0 - w) + patch * w
    )
    edited = np.clip(edited, 0, 255).astype(np.uint8)

    # Decoy mask
    decoy_mask = np.zeros((H, W), dtype=np.uint8)
    dm = (mask_soft[src_y0:src_y1, src_x0:src_x1] > 0.05).astype(np.uint8)
    decoy_mask[dst_y0:dst_y1, dst_x0:dst_x1] = dm

    return edited, decoy_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--clone_alpha", type=float, default=0.6)
    p.add_argument("--brs_alpha", type=float, default=0.80)
    p.add_argument("--brs_ring_width", type=int, default=24)
    p.add_argument("--blur_sigma", type=float, default=4.0)
    p.add_argument("--clone_scale", type=float, default=0.7)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="v2")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "codecoy" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CoDecoy v2: BRS + Target-Clone Decoy ===")
    print(f"BRS: rw={args.brs_ring_width}, α={args.brs_alpha}")
    print(f"Clone: α={args.clone_alpha}, blur_σ={args.blur_sigma}, scale={args.clone_scale}")

    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    all_results = []

    for vid in videos:
        print(f"\n{'='*60}\nVideo: {vid}\n{'='*60}")
        try:
            frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        if not frames:
            continue
        frames = frames[:args.max_frames]
        masks = masks[:args.max_frames]

        # Find decoy location
        decoy_loc = find_decoy_location(masks[0], min_dist=80)
        if decoy_loc is None:
            print(f"  [skip] no valid decoy location")
            continue
        print(f"  Decoy location: {decoy_loc}")

        t0 = time.time()

        # Generate 3 variants:
        # A. BRS only (baseline)
        # B. Clone decoy only (no BRS)
        # C. BRS + Clone decoy (combo)

        variants = {}
        for variant_name, do_brs, do_clone in [
            ("brs_only", True, False),
            ("clone_only", False, True),
            ("combo", True, True),
        ]:
            edited = []
            decoy_masks_v = []
            for f, m in zip(frames, masks):
                ef = f.copy()
                dm = np.zeros_like(m)

                if do_brs:
                    ef = apply_old_boundary_suppression(
                        ef, m, ring_width=args.brs_ring_width, blend_alpha=args.brs_alpha)

                if do_clone:
                    ef, dm = generate_clone_decoy(
                        ef, m, decoy_loc[0], decoy_loc[1],
                        clone_alpha=args.clone_alpha,
                        blur_sigma=args.blur_sigma,
                        scale=args.clone_scale)

                edited.append(ef)
                decoy_masks_v.append(dm)

            # Quality
            ssims = [frame_quality(o, e)[0] for o, e in zip(frames[:10], edited[:10])]

            # Codec + SAM2
            codec_ed = codec_round_trip(edited, FFMPEG_PATH, args.crf)
            if codec_ed is None:
                print(f"  [{variant_name}] codec failed")
                continue

            pred_masks, jf_ed, _, _ = run_tracking(codec_ed, masks, predictor, device, args.prompt)

            variants[variant_name] = {
                "jf": float(jf_ed),
                "ssim": float(np.mean(ssims)),
                "pred_masks": pred_masks,
                "decoy_masks": decoy_masks_v,
            }

        # Clean baseline
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        if codec_clean is None:
            continue
        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)

        elapsed = time.time() - t0

        print(f"\n  === {vid} RESULTS ===")
        print(f"  Clean:      JF={jf_clean:.4f}")
        result = {"video": vid, "jf_clean": float(jf_clean), "elapsed_s": elapsed}

        for vname, vdata in variants.items():
            djf = (jf_clean - vdata["jf"]) * 100
            print(f"  {vname:12s}: JF={vdata['jf']:.4f}  ΔJF={djf:+.1f}pp  SSIM={vdata['ssim']:.4f}")
            result[f"{vname}_jf"] = vdata["jf"]
            result[f"{vname}_djf"] = float(jf_clean - vdata["jf"])
            result[f"{vname}_ssim"] = vdata["ssim"]

        # Combo vs BRS-only gain
        if "combo" in variants and "brs_only" in variants:
            combo_gain = (variants["brs_only"]["jf"] - variants["combo"]["jf"]) * 100
            print(f"  Combo over BRS: {combo_gain:+.1f}pp")
            result["combo_over_brs_pp"] = float(combo_gain)

        all_results.append(result)
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Aggregate
    if all_results:
        print(f"\n{'='*60}\nAGGREGATE (n={len(all_results)})")
        for vname in ["brs_only", "clone_only", "combo"]:
            djfs = [r[f"{vname}_djf"] * 100 for r in all_results if f"{vname}_djf" in r]
            if djfs:
                print(f"  {vname:12s}: mean ΔJF={np.mean(djfs):+.1f}pp")
        gains = [r["combo_over_brs_pp"] for r in all_results if "combo_over_brs_pp" in r]
        if gains:
            print(f"  Combo over BRS: mean {np.mean(gains):+.1f}pp")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)


if __name__ == "__main__":
    main()
