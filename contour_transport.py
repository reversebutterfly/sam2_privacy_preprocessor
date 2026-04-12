"""
contour_transport.py — Contour-Transport Hallucination PoC

Core idea: instead of weakening the boundary (BRS) or creating a decoy
(CoDecoy), SHIFT the boundary to an incorrect position. SAM2's shape
memory will track the wrong contour.

For each sector k, apply a signed displacement d_k along the boundary
normal. Positive = expand outward, negative = shrink inward.
The resulting shell (M XOR M_shifted) is filled with codec-surviving content.

Expansion shell: filled with pixelated/flat object interior (looks like object)
Shrinkage shell: filled with pixelated/flat background (looks like background)

Usage:
  python contour_transport.py --videos bear,blackswan,dog,dance-twirl,elephant \
      --n_search 12 --max_frames 30 --device cuda --tag ct_v1
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
    apply_old_boundary_suppression, _apply_old_brs_proxy,
)
from src.brs_utils import mask_distance_fields


def apply_contour_transport(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    sector_displacements: np.ndarray,   # (n_angular,) signed px: +outward, -inward
    fill_mode: str = "pixelate",        # "pixelate" or "flat"
    pixelate_bs: int = 8,
    blur_sigma: float = 3.0,
) -> np.ndarray:
    """
    Shift the object boundary per-sector and fill the resulting shell.

    sector_displacements[k]: how many pixels to shift sector k's boundary
      positive = expand (make object look bigger)
      negative = shrink (make object look smaller)
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    H, W = mask.shape
    n_a = len(sector_displacements)
    mask_u8 = mask.astype(np.uint8)

    # SDF: positive outside, negative inside
    dist_out = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5).astype(np.float32)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
    sdf = dist_out - dist_in  # positive = outside mask

    # Sector assignment
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return frame_rgb.copy()
    cy, cx = float(ys.mean()), float(xs.mean())
    Y, X = np.mgrid[0:H, 0:W]
    angle = np.arctan2(Y - cy, X - cx)
    angle_norm = (angle + np.pi) / (2.0 * np.pi)
    a_idx = np.clip((angle_norm * n_a).astype(int), 0, n_a - 1)

    # Per-pixel displacement (smoothed sector assignment)
    disp_map = np.zeros((H, W), dtype=np.float32)
    for k in range(n_a):
        sector_mask = (a_idx == k).astype(np.float32)
        disp_map += float(sector_displacements[k]) * sector_mask

    # Smooth to avoid hard sector boundaries
    if blur_sigma > 0:
        disp_map = cv2.GaussianBlur(disp_map, (0, 0), blur_sigma)

    # Shifted SDF: shift boundary by displacement
    sdf_shifted = sdf - disp_map  # subtracting positive disp = expanding
    new_mask = (sdf_shifted < 0).astype(np.uint8)

    # Shell: where old and new mask differ
    expand_shell = (new_mask > 0) & (mask_u8 == 0)  # was background, now "object"
    shrink_shell = (new_mask == 0) & (mask_u8 > 0)  # was object, now "background"

    if expand_shell.sum() == 0 and shrink_shell.sum() == 0:
        return frame_rgb.copy()

    # Generate fill content
    result = frame_rgb.astype(np.float32).copy()

    if fill_mode == "pixelate":
        # Pixelated version of the frame
        small = cv2.resize(frame_rgb, (W // pixelate_bs, H // pixelate_bs),
                           interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        fill_content = pixelated
    else:  # flat
        # Object mean for expansion, background mean for shrinkage
        obj_mean = frame_rgb[mask > 0].mean(axis=0) if mask.sum() > 0 else frame_rgb.mean(axis=(0, 1))
        bg_mask = cv2.dilate(mask_u8, np.ones((49, 49), np.uint8))
        bg_region = (bg_mask > 0) & (mask_u8 == 0)
        bg_mean = frame_rgb[bg_region].mean(axis=0) if bg_region.sum() > 0 else frame_rgb.mean(axis=(0, 1))
        fill_content = np.zeros_like(result)
        fill_content[expand_shell] = obj_mean
        fill_content[shrink_shell] = bg_mean
        fill_content = cv2.GaussianBlur(fill_content, (0, 0), max(pixelate_bs / 2, 3))

    # Soft blending at shell edges (feathered)
    expand_f = expand_shell.astype(np.float32)
    shrink_f = shrink_shell.astype(np.float32)
    shell_f = np.clip(expand_f + shrink_f, 0, 1)
    shell_smooth = cv2.GaussianBlur(shell_f, (0, 0), max(blur_sigma, 2.0))

    w = shell_smooth[:, :, None]
    if fill_mode == "pixelate":
        result = result * (1.0 - w) + fill_content * w
    else:
        result = result * (1.0 - w) + fill_content * w

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--n_angular", type=int, default=8)
    p.add_argument("--n_search", type=int, default=12)
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--fill_mode", default="pixelate", choices=["pixelate", "flat"])
    p.add_argument("--pixelate_bs", type=int, default=8)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="ct_v1")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    save_dir = Path(ROOT) / "results_v100" / "contour_transport" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Contour-Transport Hallucination PoC ===")
    print(f"fill={args.fill_mode}, pixelate_bs={args.pixelate_bs}, n_search={args.n_search}")

    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    rng = np.random.default_rng(42)
    results = []

    for vid in videos:
        print(f"\n{'='*60}\n{vid}\n{'='*60}")
        try:
            frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] {e}"); continue
        if not frames: continue
        frames, masks = frames[:args.max_frames], masks[:args.max_frames]

        t0 = time.time()

        # Clean codec baseline
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        if codec_clean is None:
            print(f"  [skip] codec failed"); continue
        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)

        # BRS baseline (for comparison)
        brs_frames = [apply_old_boundary_suppression(f, m, ring_width=24, blend_alpha=0.80)
                      for f, m in zip(frames, masks)]
        codec_brs = codec_round_trip(brs_frames, FFMPEG_PATH, args.crf)
        if codec_brs is None: continue
        _, jf_brs, _, _ = run_tracking(codec_brs, masks, predictor, device, args.prompt)
        djf_brs = (jf_clean - jf_brs) * 100

        # Search over displacement patterns
        print(f"  Clean JF={jf_clean:.4f}, BRS ΔJF={djf_brs:+.1f}pp")
        print(f"  Searching {args.n_search} contour-transport configs...")

        best_djf = 0.0
        best_disps = None
        best_ssim = 1.0

        for i in range(args.n_search):
            # Generate displacement pattern
            if i == 0:
                disps = np.zeros(args.n_angular)  # no shift (sanity)
            elif i == 1:
                disps = np.full(args.n_angular, 5.0)  # uniform expand +5px
            elif i == 2:
                disps = np.full(args.n_angular, 10.0)  # uniform expand +10px
            elif i == 3:
                disps = np.full(args.n_angular, -5.0)  # uniform shrink -5px
            elif i == 4:
                disps = np.full(args.n_angular, 15.0)  # strong expand
            elif i == 5:
                # Asymmetric: expand one side, shrink other
                disps = np.array([12, 8, -5, -8, -5, 8, 12, 8], dtype=np.float64)[:args.n_angular]
            else:
                # Random signed displacements
                disps = rng.uniform(-10, 15, size=args.n_angular)

            # Apply to all frames
            edited = [apply_contour_transport(f, m, disps, args.fill_mode, args.pixelate_bs)
                      for f, m in zip(frames, masks)]

            # Quality check
            ssim_vals = [frame_quality(o, e)[0] for o, e in zip(frames[:5], edited[:5])]
            mean_ssim = float(np.mean(ssim_vals))

            # Codec + SAM2
            codec_ed = codec_round_trip(edited, FFMPEG_PATH, args.crf)
            if codec_ed is None:
                print(f"    [{i+1:2d}] codec failed"); continue
            _, jf_ed, _, _ = run_tracking(codec_ed, masks, predictor, device, args.prompt)
            djf = (jf_clean - jf_ed) * 100

            disp_str = ','.join(f'{d:+.0f}' for d in disps[:4])
            print(f"    [{i+1:2d}/{args.n_search}] ΔJF={djf:+.1f}pp  SSIM={mean_ssim:.3f}  "
                  f"disps=[{disp_str}...]")

            if djf > best_djf and mean_ssim >= 0.88:
                best_djf = djf
                best_disps = disps.copy()
                best_ssim = mean_ssim

        elapsed = time.time() - t0
        ct_gain = best_djf - djf_brs

        print(f"\n  === {vid} RESULTS ===")
        print(f"  BRS:              ΔJF={djf_brs:+.1f}pp")
        print(f"  Contour Transport: ΔJF={best_djf:+.1f}pp  SSIM={best_ssim:.3f}")
        print(f"  CT over BRS:      {ct_gain:+.1f}pp")
        if best_disps is not None:
            print(f"  Best displacements: {best_disps.round(1).tolist()}")
        print(f"  Elapsed: {elapsed:.0f}s")

        results.append({
            "video": vid, "jf_clean": float(jf_clean),
            "djf_brs": float(djf_brs), "djf_ct": float(best_djf),
            "ct_gain": float(ct_gain), "ssim": float(best_ssim),
            "best_displacements": best_disps.tolist() if best_disps is not None else None,
            "elapsed": elapsed,
        })
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)

    # Aggregate
    if results:
        gains = [r["ct_gain"] for r in results]
        wins = sum(1 for g in gains if g > 2)
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(results)})")
        print(f"  Mean BRS: {np.mean([r['djf_brs'] for r in results]):+.1f}pp")
        print(f"  Mean CT:  {np.mean([r['djf_ct'] for r in results]):+.1f}pp")
        print(f"  Mean gain over BRS: {np.mean(gains):+.1f}pp")
        print(f"  Wins: {wins}/{len(results)}")
        if np.mean(gains) > 5:
            print(f"  VERDICT: CONTOUR TRANSPORT IS VIABLE")
        elif np.mean(gains) > 0:
            print(f"  VERDICT: MARGINAL — needs stronger configs")
        else:
            print(f"  VERDICT: NOT VIABLE")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
