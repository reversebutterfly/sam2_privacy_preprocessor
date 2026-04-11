"""
btrr_probe.py — Block-Token Resonance Routing Hypothesis Validation

BTRR exploits the structural alignment between H.264's 8x8 DCT blocks
and SAM2's 8x8 memory tokens. The hypothesis: sectors where the object
boundary crosses more H.264 block boundaries are more vulnerable to
codec-amplified editing.

For each sector k, compute:
  1. block_crossing: # of 8x8 block boundaries the object contour crosses
  2. dct_energy: mean DCT energy near the boundary (texture complexity)
  3. orientation: alignment between boundary normal and block axes (0/90 deg)

Then correlate these features with marginal post-codec ΔJF (from cmt_probe v2 data
or computed fresh).

No SAM2 inference needed for the features — only for evaluation.

Usage:
  python btrr_probe.py --videos bear,blackswan,dog,dance-twirl,elephant \
      --device cuda --tag btrr_v1
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
from pilot_mask_guided import build_predictor, run_tracking, codec_round_trip
from oracle_mask_search import _build_sector_geometry, apply_sector_suppression


# ─────────────────────────────────────────────────────────────────────────────
# BTRR Feature Extraction (no SAM2 needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_btrr_features(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    n_angular: int = 8,
    ring_width: int = 24,
    block_size: int = 8,
) -> dict:
    """
    Compute per-sector BTRR vulnerability features.

    Returns dict with:
      block_crossings: (n_angular,) — contour crossings of 8x8 block boundaries per sector
      dct_energy: (n_angular,) — mean DCT coefficient energy in boundary ring per sector
      orientation_score: (n_angular,) — alignment of boundary normal with block axes
      combined_score: (n_angular,) — multiplicative combination
    """
    H, W = mask.shape

    # 1. Find object contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Largest contour
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.squeeze()  # (N, 2) in (x, y)
    if contour_pts.ndim < 2 or len(contour_pts) < 10:
        return None

    # Mask centroid for sector assignment
    ys, xs = np.where(mask > 0)
    cy, cx = float(ys.mean()), float(xs.mean())

    # Assign contour points to sectors
    angles = np.arctan2(contour_pts[:, 1] - cy, contour_pts[:, 0] - cx)
    angle_norm = (angles + np.pi) / (2 * np.pi)
    sector_idx = np.clip((angle_norm * n_angular).astype(int), 0, n_angular - 1)

    # 2. Block boundary crossings per sector
    # A contour point "crosses" a block boundary if consecutive points
    # straddle different 8x8 blocks
    block_crossings = np.zeros(n_angular, dtype=np.float64)
    for i in range(len(contour_pts) - 1):
        x0, y0 = contour_pts[i]
        x1, y1 = contour_pts[i + 1]
        # Check if they're in different 8x8 blocks (either x or y axis)
        if (x0 // block_size != x1 // block_size) or (y0 // block_size != y1 // block_size):
            s = sector_idx[i]
            block_crossings[s] += 1

    # Normalize by sector contour length
    sector_lengths = np.zeros(n_angular, dtype=np.float64)
    for i in range(len(contour_pts)):
        sector_lengths[sector_idx[i]] += 1
    block_crossings = block_crossings / np.maximum(sector_lengths, 1.0)

    # 3. DCT energy near boundary per sector
    # Convert to grayscale, compute local DCT energy via Laplacian variance
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    lap_sq = laplacian ** 2

    # Boundary ring
    kernel = np.ones((ring_width * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    ring = ((dilated > 0) & (eroded == 0))

    # Sector assignment for ring pixels
    Y, X = np.mgrid[0:H, 0:W]
    ring_angles = np.arctan2(Y - cy, X - cx)
    ring_angle_norm = (ring_angles + np.pi) / (2 * np.pi)
    ring_sector = np.clip((ring_angle_norm * n_angular).astype(int), 0, n_angular - 1)

    dct_energy = np.zeros(n_angular, dtype=np.float64)
    for k in range(n_angular):
        sec_ring = ring & (ring_sector == k)
        if sec_ring.sum() > 0:
            dct_energy[k] = float(lap_sq[sec_ring].mean())
    # Normalize
    dct_max = max(dct_energy.max(), 1e-8)
    dct_energy = dct_energy / dct_max

    # 4. Orientation score: how well does boundary normal align with block axes (0°/90°)
    # Compute boundary gradient direction
    gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    orientation_score = np.zeros(n_angular, dtype=np.float64)
    for k in range(n_angular):
        sec_ring = ring & (ring_sector == k)
        if sec_ring.sum() > 0:
            gx_sec = gx[sec_ring]
            gy_sec = gy[sec_ring]
            # Angle of boundary normal
            theta = np.arctan2(gy_sec, gx_sec + 1e-8)
            # Alignment with 0° or 90° (block axes) — cos(2θ)² peaks at 0°/90°
            alignment = np.cos(2 * theta) ** 2
            orientation_score[k] = float(alignment.mean())

    # 5. Combined score (multiplicative)
    combined = block_crossings * (0.5 + dct_energy) * (0.5 + orientation_score)
    # Normalize to [0, 1]
    cmax = max(combined.max(), 1e-8)
    combined = combined / cmax

    return {
        "block_crossings": block_crossings,
        "dct_energy": dct_energy,
        "orientation_score": orientation_score,
        "combined_score": combined,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--n_angular", type=int, default=8)
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--probe_alpha", type=float, default=0.80)
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="btrr_v1")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "btrr_probe" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== BTRR Hypothesis Validation ===")
    print(f"Testing: do block-crossing / DCT energy / orientation predict marginal ΔJF?")

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

        # 1. Compute BTRR features (NO SAM2 needed — pure image/mask features)
        feats = compute_btrr_features(frames[0], masks[0], args.n_angular, args.ring_width)
        if feats is None:
            print(f"  [skip] no contour")
            continue

        print(f"  BTRR features computed:")
        print(f"    block_crossings: {feats['block_crossings'].round(3)}")
        print(f"    dct_energy:      {feats['dct_energy'].round(3)}")
        print(f"    orientation:     {feats['orientation_score'].round(3)}")
        print(f"    combined:        {feats['combined_score'].round(3)}")

        # 2. Compute marginal ΔJF for each sector (same protocol as CMT v2)
        print(f"  Computing marginal ΔJF per sector (codec + SAM2)...")
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        if codec_clean is None:
            print(f"  [skip] codec failed")
            continue
        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)

        marginal_djfs = np.zeros(args.n_angular, dtype=np.float64)
        for k in range(args.n_angular):
            alphas = np.zeros(args.n_angular)
            alphas[k] = args.probe_alpha
            edited = [apply_sector_suppression(f, m, alphas, args.ring_width)
                      for f, m in zip(frames, masks)]
            codec_edited = codec_round_trip(edited, FFMPEG_PATH, args.crf)
            if codec_edited is None:
                continue
            _, jf_ed, _, _ = run_tracking(codec_edited, masks, predictor, device, args.prompt)
            marginal_djfs[k] = jf_clean - jf_ed
            print(f"    sector {k}: marginal_ΔJF={marginal_djfs[k]*100:+.2f}pp  "
                  f"combined={feats['combined_score'][k]:.3f}")

        elapsed = time.time() - t0

        # 3. Correlations
        result = {
            "video": vid,
            "block_crossings": feats["block_crossings"].tolist(),
            "dct_energy": feats["dct_energy"].tolist(),
            "orientation_score": feats["orientation_score"].tolist(),
            "combined_score": feats["combined_score"].tolist(),
            "marginal_djfs": marginal_djfs.tolist(),
            "jf_clean": float(jf_clean),
            "elapsed_s": elapsed,
        }

        active = marginal_djfs != 0
        if active.sum() >= 4:
            from scipy.stats import pearsonr, spearmanr
            m = marginal_djfs[active]
            for feat_name in ["block_crossings", "dct_energy", "orientation_score", "combined_score"]:
                f_arr = feats[feat_name][active]
                rp, pp = pearsonr(f_arr, m)
                rs, ps = spearmanr(f_arr, m)
                result[f"{feat_name}_pearson"] = float(rp)
                result[f"{feat_name}_spearman"] = float(rs)
                print(f"    {feat_name:20s}: Pearson={rp:+.3f} (p={pp:.3f})  Spearman={rs:+.3f}")

        all_results.append(result)
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Aggregate
    if all_results:
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(all_results)} videos)")
        for feat_name in ["block_crossings", "dct_energy", "orientation_score", "combined_score"]:
            rs = [r[f"{feat_name}_pearson"] for r in all_results if f"{feat_name}_pearson" in r]
            if rs:
                mr = np.mean(rs)
                print(f"  {feat_name:20s}: mean Pearson r = {mr:+.3f}  {'VIABLE' if mr > 0.4 else 'WEAK' if mr > 0.2 else 'NOT VIABLE'}")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
