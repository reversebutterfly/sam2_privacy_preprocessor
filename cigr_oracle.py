"""
cigr_oracle.py — CIGR Oracle Search: Exhaustive 2^8 Operator Assignment Evaluation

For each video, evaluate ALL 256 binary operator assignments (BRS vs pixelation
per sector) with real H.264+SAM2. This data is used to:
  1. Find the oracle-best assignment
  2. Train the CIGR energy model (unary + pairwise potentials)

Usage:
  python cigr_oracle.py --videos bear,blackswan,dog --max_frames 30 \
      --device cuda --tag cigr_oracle_v1
"""

from __future__ import annotations

import argparse
import itertools
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
from eval_baselines import baseline_pixelation
from oracle_mask_search import _build_sector_geometry


def apply_mixed_operator(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    assignment: np.ndarray,   # (8,) binary: 0=BRS, 1=pixelation
    ring_width: int = 24,
    brs_alpha: float = 0.80,
    pixel_bs: int = 16,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    """
    Apply per-sector operator: BRS or pixelation based on binary assignment.
    Uses sector-weighted blending (same framework as oracle_mask_search).
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    n_a = len(assignment)
    geom = _build_sector_geometry(mask, n_a, ring_width, smooth_sigma)
    if geom is None:
        return frame_rgb.copy()

    ring_smooth = geom["ring_smooth"]
    sector_masks = geom["sector_masks"]
    H, W = mask.shape

    # Generate both operator outputs for the full frame
    brs_frame = apply_old_boundary_suppression(
        frame_rgb, mask, ring_width=ring_width, blend_alpha=brs_alpha)
    pix_frame = baseline_pixelation(frame_rgb, mask, block_size=pixel_bs)

    # Compose: each sector uses its assigned operator
    result = frame_rgb.astype(np.float32).copy()

    for k in range(n_a):
        w_k = (ring_smooth * sector_masks[k])[:, :, None]

        if assignment[k] == 0:  # BRS
            src = brs_frame.astype(np.float32)
        else:  # Pixelation
            src = pix_frame.astype(np.float32)

        # Blend: replace ring region of this sector with the chosen operator's output
        result = result * (1.0 - w_k) + src * w_k

    return np.clip(result, 0, 255).astype(np.uint8)


def extract_sector_features(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    n_angular: int = 8,
    ring_width: int = 24,
) -> dict:
    """Extract per-sector features for CIGR model training."""
    H, W = mask.shape
    geom = _build_sector_geometry(mask, n_angular, ring_width)
    if geom is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    cy, cx = float(ys.mean()), float(xs.mean())
    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1

    # Global features
    global_feats = np.array([
        float(mask.sum()) / (H * W),
        float(bbox_w) / max(bbox_h, 1),
        float(4.0 * 3.14159 * mask.sum() / max(cv2.arcLength(
            cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0][0], True) ** 2, 1.0)),
        float(cv2.arcLength(
            cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0][0], True)) / max(2 * (bbox_h + bbox_w), 1.0),
    ], dtype=np.float32)

    # Per-sector features
    Y, X = np.mgrid[0:H, 0:W]
    angle = np.arctan2(Y - cy, X - cx)
    angle_norm = (angle + np.pi) / (2.0 * np.pi)
    a_idx = np.clip((angle_norm * n_angular).astype(int), 0, n_angular - 1)

    kernel = np.ones((ring_width * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    ring = ((dilated > 0) & (eroded == 0))

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_dir = np.arctan2(gy, gx + 1e-8)

    ring_total = max(ring.sum(), 1)
    sector_feats = []
    for k in range(n_angular):
        sec = (a_idx == k) & ring
        cnt = sec.sum()
        area = float(cnt) / ring_total
        rgb = frame_rgb[sec].mean(axis=0) / 255.0 if cnt > 0 else np.zeros(3)
        grad = float(grad_mag[sec].mean()) / 255.0 if cnt > 0 else 0.0
        curv = float(grad_dir[sec].std()) if cnt > 0 else 0.0
        sector_feats.append(np.concatenate([[area], rgb, [grad, curv]]).astype(np.float32))

    # Pairwise boundary features (between adjacent sectors)
    pair_feats = []
    for k in range(n_angular):
        k_next = (k + 1) % n_angular
        sec_k = (a_idx == k) & ring
        sec_next = (a_idx == k_next) & ring

        # Boundary between sectors: pixels near both
        boundary = sec_k.astype(np.float32) * cv2.GaussianBlur(
            sec_next.astype(np.float32), (0, 0), 3.0)
        boundary_pixels = boundary > 0.1

        n_boundary = max(boundary_pixels.sum(), 1)
        # Texture contrast at boundary
        contrast = float(grad_mag[boundary_pixels].std()) if boundary_pixels.sum() > 5 else 0.0
        # Number of 8x8 blocks spanning boundary
        if boundary_pixels.sum() > 0:
            by, bx = np.where(boundary_pixels)
            block_ids = set((y // 8, x // 8) for y, x in zip(by, bx))
            n_blocks = len(block_ids)
        else:
            n_blocks = 0

        pair_feats.append(np.array([
            float(n_boundary) / ring_total,
            contrast / 255.0,
            float(n_blocks) / max(ring_total / 64, 1),
        ], dtype=np.float32))

    return {
        "global": global_feats,          # (4,)
        "sector": sector_feats,           # list of 8 × (6,)
        "pairwise": pair_feats,           # list of 8 × (3,)
        "ring_areas": geom["ring_areas"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--n_angular", type=int, default=8)
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--brs_alpha", type=float, default=0.80)
    p.add_argument("--pixel_bs", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="cigr_oracle_v1")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    save_dir = Path(ROOT) / "results_v100" / "cigr" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    n_a = args.n_angular
    n_assignments = 2 ** n_a  # 256

    print(f"=== CIGR Oracle Search ===")
    print(f"Videos: {videos}, {n_assignments} assignments per video")

    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    all_results = []

    for vid in videos:
        print(f"\n{'='*60}\n{vid}\n{'='*60}")
        try:
            frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] {e}"); continue
        if not frames: continue
        frames, masks = frames[:args.max_frames], masks[:args.max_frames]

        # Extract features once
        feats = extract_sector_features(frames[0], masks[0], n_a, args.ring_width)
        if feats is None:
            print(f"  [skip] no features"); continue

        t0 = time.time()

        # Clean codec baseline
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        if codec_clean is None:
            print(f"  [skip] codec failed"); continue
        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)
        print(f"  Clean JF = {jf_clean:.4f}")

        # Evaluate all 256 assignments
        evaluations = []
        best_djf = 0.0
        best_assignment = None

        for idx in range(n_assignments):
            assignment = np.array([(idx >> k) & 1 for k in range(n_a)], dtype=np.int32)

            # Apply mixed operator
            edited = [apply_mixed_operator(f, m, assignment, args.ring_width,
                                            args.brs_alpha, args.pixel_bs)
                      for f, m in zip(frames, masks)]

            # Quality
            ssim_vals = [frame_quality(o, e)[0] for o, e in zip(frames[:5], edited[:5])]
            mean_ssim = float(np.mean(ssim_vals))

            # Codec + SAM2
            codec_ed = codec_round_trip(edited, FFMPEG_PATH, args.crf)
            if codec_ed is None:
                evaluations.append({"idx": idx, "assignment": assignment.tolist(),
                                     "djf": 0.0, "ssim": 0.0, "valid": False})
                continue

            _, jf_ed, _, _ = run_tracking(codec_ed, masks, predictor, device, args.prompt)
            djf = jf_clean - jf_ed
            valid = mean_ssim >= 0.88  # slightly relaxed for search

            evaluations.append({
                "idx": idx,
                "assignment": assignment.tolist(),
                "djf": float(djf),
                "ssim": float(mean_ssim),
                "valid": valid,
            })

            if valid and djf > best_djf:
                best_djf = djf
                best_assignment = assignment.copy()

            if (idx + 1) % 32 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (n_assignments - idx - 1) / rate
                print(f"    [{idx+1:3d}/{n_assignments}] best ΔJF={best_djf*100:+.1f}pp  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        elapsed = time.time() - t0

        # Also get uniform baselines
        uni_brs = [apply_old_boundary_suppression(f, m, ring_width=args.ring_width,
                    blend_alpha=args.brs_alpha) for f, m in zip(frames, masks)]
        codec_brs = codec_round_trip(uni_brs, FFMPEG_PATH, args.crf)
        _, jf_brs, _, _ = run_tracking(codec_brs, masks, predictor, device, args.prompt) if codec_brs else (None, jf_clean, None, None)

        uni_pix = [baseline_pixelation(f, m, block_size=args.pixel_bs)
                   for f, m in zip(frames, masks)]
        codec_pix = codec_round_trip(uni_pix, FFMPEG_PATH, args.crf)
        _, jf_pix, _, _ = run_tracking(codec_pix, masks, predictor, device, args.prompt) if codec_pix else (None, jf_clean, None, None)

        djf_brs = (jf_clean - jf_brs) * 100
        djf_pix = (jf_clean - jf_pix) * 100
        best_djf_pp = best_djf * 100
        oracle_gain = best_djf_pp - max(djf_brs, djf_pix)

        print(f"\n  === {vid} RESULTS ===")
        print(f"  Uniform BRS: {djf_brs:+.1f}pp")
        print(f"  Uniform PIX: {djf_pix:+.1f}pp")
        print(f"  CIGR Oracle: {best_djf_pp:+.1f}pp")
        print(f"  Oracle gain over best uniform: {oracle_gain:+.1f}pp")
        if best_assignment is not None:
            labels = ['B' if a == 0 else 'P' for a in best_assignment]
            print(f"  Best assignment: {labels}")
        print(f"  Elapsed: {elapsed:.0f}s ({n_assignments} evals)")

        vid_result = {
            "video": vid,
            "jf_clean": float(jf_clean),
            "djf_brs": float(djf_brs),
            "djf_pix": float(djf_pix),
            "djf_oracle": float(best_djf_pp),
            "oracle_gain": float(oracle_gain),
            "best_assignment": best_assignment.tolist() if best_assignment is not None else None,
            "features": {
                "global": feats["global"].tolist(),
                "sector": [s.tolist() for s in feats["sector"]],
                "pairwise": [p.tolist() for p in feats["pairwise"]],
            },
            "all_evaluations": evaluations,
            "elapsed_s": elapsed,
        }
        all_results.append(vid_result)

        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Aggregate
    if all_results:
        gains = [r["oracle_gain"] for r in all_results]
        wins = sum(1 for g in gains if g > 2)
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(all_results)})")
        print(f"  Mean uniform BRS: {np.mean([r['djf_brs'] for r in all_results]):+.1f}pp")
        print(f"  Mean uniform PIX: {np.mean([r['djf_pix'] for r in all_results]):+.1f}pp")
        print(f"  Mean CIGR Oracle: {np.mean([r['djf_oracle'] for r in all_results]):+.1f}pp")
        print(f"  Mean gain: {np.mean(gains):+.1f}pp")
        print(f"  Wins: {wins}/{len(all_results)}")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)


if __name__ == "__main__":
    main()
