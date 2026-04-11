"""
codecoy_poc.py — CoDecoy Proof of Concept

Minimal validation: can a low-frequency feature decoy in the background
misdirect SAM2 tracking after H.264 compression?

Protocol:
  1. Extract DINOv2 target prototype from GT mask region
  2. Find background candidate region most dissimilar to target
  3. Generate a simple low-frequency decoy: blend target's low-freq
     color/texture template into the background candidate
  4. H.264 encode/decode
  5. Run SAM2 → measure if tracking shifts toward decoy

Success criterion: SAM2 IoU with GT drops AND IoU with decoy region rises.

Usage:
  python codecoy_poc.py --videos bear,blackswan,dog --device cuda --tag poc_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import build_predictor, run_tracking, codec_round_trip, frame_quality
from src.metrics import jf_score


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

_dino_model = None

def get_dino(device="cuda"):
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                      verbose=False).to(device).eval()
        for p in _dino_model.parameters():
            p.requires_grad_(False)
    return _dino_model

def extract_dino_features(frame_rgb: np.ndarray, device="cuda") -> torch.Tensor:
    """Extract DINOv2 patch tokens. Returns (1, N_patches, D) tensor."""
    dino = get_dino(device)
    # Preprocess: resize to 518x518 (14*37), normalize
    img = cv2.resize(frame_rgb, (518, 518))
    img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_t = (img_t - mean) / std
    with torch.no_grad():
        features = dino.forward_features(img_t)
        patch_tokens = features["x_norm_patchtokens"]  # (1, N, D)
    return patch_tokens  # (1, 37*37, 384)


def get_patch_grid_size(frame_shape, patch_size=14, resize=518):
    """Returns (H_patches, W_patches)."""
    return resize // patch_size, resize // patch_size  # 37, 37


# ─────────────────────────────────────────────────────────────────────────────
# Decoy Generation (Simple PoC version)
# ─────────────────────────────────────────────────────────────────────────────

def find_decoy_location(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    min_dist_px: int = 80,
    decoy_size: int = 64,
) -> tuple[int, int] | None:
    """
    Find a background location for the decoy.
    Picks the location farthest from the mask centroid that has enough space.
    Returns (cy, cx) or None.
    """
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    obj_cy, obj_cx = int(ys.mean()), int(xs.mean())

    # Distance from object centroid
    Y, X = np.mgrid[0:H, 0:W]
    dist = np.sqrt((Y - obj_cy)**2 + (X - obj_cx)**2)

    # Must be in background, far enough, and have room for decoy
    pad = decoy_size // 2
    valid = (mask == 0) & (dist > min_dist_px)
    valid[:pad, :] = False
    valid[-pad:, :] = False
    valid[:, :pad] = False
    valid[:, -pad:] = False

    if not valid.any():
        return None

    # Pick farthest valid point
    dist[~valid] = 0
    best_idx = np.unravel_index(dist.argmax(), dist.shape)
    return int(best_idx[0]), int(best_idx[1])


def generate_decoy_frame(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    decoy_cy: int,
    decoy_cx: int,
    decoy_size: int = 64,
    blend_alpha: float = 0.35,
    blur_sigma: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a low-frequency decoy at (decoy_cy, decoy_cx).

    The decoy is a blurred version of the target region's mean color/shape,
    blended softly into the background. This creates a low-frequency
    "ghost" that survives H.264 but is visually subtle.

    Returns:
        edited_frame: frame with decoy
        decoy_mask: binary mask of the decoy region
    """
    H, W = frame_rgb.shape[:2]
    half = decoy_size // 2

    # Extract target region stats
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return frame_rgb.copy(), np.zeros_like(mask)

    # Target mean color
    target_color = frame_rgb[mask > 0].mean(axis=0)  # (3,)

    # Create a soft elliptical stamp at decoy location
    Y, X = np.mgrid[0:H, 0:W]
    dy = (Y - decoy_cy).astype(np.float32) / max(half, 1)
    dx = (X - decoy_cx).astype(np.float32) / max(half, 1)
    dist_sq = dy**2 + dx**2

    # Soft elliptical mask (raised cosine falloff)
    stamp = np.clip(1.0 - dist_sq, 0, 1)
    stamp = stamp ** 2  # sharpen edges slightly

    # Low-pass the stamp
    stamp = cv2.GaussianBlur(stamp, (0, 0), blur_sigma)
    stamp = np.clip(stamp * blend_alpha, 0, blend_alpha)

    # Create decoy content: target color blended with local background
    # (low frequency, codec-safe)
    local_bg = cv2.GaussianBlur(frame_rgb.astype(np.float32), (0, 0), blur_sigma * 2)

    # Decoy = weighted average of target color and blurred background
    decoy_content = (
        0.6 * np.broadcast_to(target_color[None, None, :], frame_rgb.shape).astype(np.float32)
        + 0.4 * local_bg
    )

    # Blend
    w = stamp[:, :, None]
    edited = frame_rgb.astype(np.float32) * (1.0 - w) + decoy_content * w
    edited = np.clip(edited, 0, 255).astype(np.uint8)

    # Decoy mask (for IoU measurement)
    decoy_mask = (stamp > 0.05).astype(np.uint8)

    return edited, decoy_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--decoy_size", type=int, default=80)
    p.add_argument("--blend_alpha", type=float, default=0.35)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="poc_v1")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "codecoy" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CoDecoy Proof of Concept ===")
    print(f"Videos: {videos}, decoy_size={args.decoy_size}, alpha={args.blend_alpha}")

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

        t0 = time.time()

        # 1. Find decoy location (from first frame)
        decoy_loc = find_decoy_location(frames[0], masks[0],
                                         min_dist_px=100, decoy_size=args.decoy_size)
        if decoy_loc is None:
            print(f"  [skip] no valid decoy location")
            continue
        decoy_cy, decoy_cx = decoy_loc
        print(f"  Decoy location: ({decoy_cy}, {decoy_cx})")

        # 2. Generate decoy on all frames
        edited_frames = []
        decoy_masks = []
        for f, m in zip(frames, masks):
            ef, dm = generate_decoy_frame(f, m, decoy_cy, decoy_cx,
                                           args.decoy_size, args.blend_alpha)
            edited_frames.append(ef)
            decoy_masks.append(dm)

        # 3. Quality check
        ssims = [frame_quality(o, e)[0] for o, e in zip(frames[:10], edited_frames[:10])]
        mean_ssim = float(np.mean(ssims))
        print(f"  SSIM: {mean_ssim:.4f}")

        # 4. Codec round-trip
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        codec_edited = codec_round_trip(edited_frames, FFMPEG_PATH, args.crf)
        if codec_clean is None or codec_edited is None:
            print(f"  [skip] codec failed")
            continue

        # 5. SAM2 tracking
        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)
        _, jf_edited, _, _ = run_tracking(codec_edited, masks, predictor, device, args.prompt)

        # Also track with decoy masks to see if SAM2 was misdirected
        pred_clean_masks = run_tracking(codec_clean, masks, predictor, device, args.prompt)[0]
        pred_edited_masks = run_tracking(codec_edited, masks, predictor, device, args.prompt)[0]

        # Measure: how much does prediction overlap with decoy vs GT?
        gt_ious = []
        decoy_ious = []
        for pred_m, gt_m, dc_m in zip(pred_edited_masks, masks, decoy_masks):
            if pred_m is None:
                continue
            pred_bool = pred_m.astype(bool) if isinstance(pred_m, np.ndarray) else pred_m
            gt_bool = gt_m.astype(bool)
            dc_bool = dc_m.astype(bool)

            # IoU with GT
            inter_gt = (pred_bool & gt_bool).sum()
            union_gt = (pred_bool | gt_bool).sum()
            gt_ious.append(float(inter_gt / max(union_gt, 1)))

            # IoU with decoy
            inter_dc = (pred_bool & dc_bool).sum()
            union_dc = (pred_bool | dc_bool).sum()
            decoy_ious.append(float(inter_dc / max(union_dc, 1)))

        delta_jf = jf_clean - jf_edited
        mean_gt_iou = float(np.mean(gt_ious)) if gt_ious else 0
        mean_decoy_iou = float(np.mean(decoy_ious)) if decoy_ious else 0
        elapsed = time.time() - t0

        print(f"  JF clean: {jf_clean:.4f}")
        print(f"  JF edited: {jf_edited:.4f}")
        print(f"  ΔJF: {delta_jf*100:+.2f}pp")
        print(f"  Pred→GT IoU: {mean_gt_iou:.4f}")
        print(f"  Pred→Decoy IoU: {mean_decoy_iou:.4f}")
        hijacked = mean_decoy_iou > 0.05 and delta_jf > 0.05
        print(f"  Hijacked: {'YES' if hijacked else 'NO'}")

        result = {
            "video": vid,
            "decoy_location": [decoy_cy, decoy_cx],
            "ssim": mean_ssim,
            "jf_clean": float(jf_clean),
            "jf_edited": float(jf_edited),
            "delta_jf": float(delta_jf),
            "pred_gt_iou": mean_gt_iou,
            "pred_decoy_iou": mean_decoy_iou,
            "hijacked": hijacked,
            "elapsed_s": elapsed,
        }
        all_results.append(result)

        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Aggregate
    if all_results:
        djfs = [r["delta_jf"] * 100 for r in all_results]
        hijacks = sum(1 for r in all_results if r["hijacked"])
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(all_results)})")
        print(f"  Mean ΔJF: {np.mean(djfs):+.2f}pp")
        print(f"  Hijack rate: {hijacks}/{len(all_results)}")
        print(f"  Mean SSIM: {np.mean([r['ssim'] for r in all_results]):.4f}")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
