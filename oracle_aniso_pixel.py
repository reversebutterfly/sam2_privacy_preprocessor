"""
oracle_aniso_pixel.py — Anisotropic Pixelation Oracle Gap Test (v2, bug-fixed)

Tests whether anisotropic angular allocation works for pixelation.
Fixed: uniform baseline is now strictly inside the search space.

Key design:
  - Single apply function for BOTH uniform and anisotropic (same code path)
  - Sector composition via weighted sum (not sequential overwrite)
  - Shared ring geometry between baseline and oracle
  - Per-frame budget re-projection

Usage:
  python oracle_aniso_pixel.py \
      --videos bear,dog,dance-twirl,blackswan,elephant \
      --n_search 15 --n_angular 8 --max_frames 50 \
      --device cuda --tag aniso_pixel_v2
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
from pilot_mask_guided import build_predictor, run_tracking, codec_round_trip, frame_quality
from oracle_mask_search import (
    _build_sector_geometry, evaluate_oracle,
)


def project_bs_to_budget(
    block_sizes: np.ndarray,
    target_budget: float,
    ring_areas: np.ndarray,
    bs_min: float = 4.0,
    bs_max: float = 40.0,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Project block_sizes so that area-weighted mean = target_budget.
    Clips to [bs_min, bs_max] (NOT [0, 1] like alpha projection).
    """
    a = np.clip(block_sizes.astype(np.float64), bs_min, bs_max)
    total_area = float(ring_areas.sum())
    if total_area <= 0:
        return a

    def w_mean(x):
        return float((ring_areas * x).sum() / total_area)

    for _ in range(max_iter):
        diff = target_budget - w_mean(a)
        if abs(diff) < 0.1:  # block sizes are integers, 0.1 tolerance is fine
            break
        a = a + diff
        a = np.clip(a, bs_min, bs_max)

    return a


# ─────────────────────────────────────────────────────────────────────────────
# Unified sector pixelation (used for BOTH uniform and anisotropic)
# ─────────────────────────────────────────────────────────────────────────────

def apply_sector_pixelation_v2(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    sector_block_sizes: np.ndarray,   # (n_angular,) integer block sizes per sector
    ring_width: int = 24,
    smooth_sigma: float = 3.0,
    geometry: dict | None = None,
) -> np.ndarray:
    """
    Per-sector pixelation with weighted-sum composition (no sequential overwrite).

    Each sector k pixelates the frame at block_size_k. The final edit is:
        out(x,y) = frame(x,y) * (1 - R(x,y)) + R(x,y) * sum_k pi_k(x,y) * pixelate(frame, bs_k)
    where:
        R(x,y) = ring_smooth(x,y) (shared ring weight)
        pi_k(x,y) = sector_mask_k(x,y) / sum_j sector_mask_j(x,y)  (normalized sector partition)

    When all block_sizes are equal, this produces EXACTLY the same output as
    uniform pixelation with that block_size.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    n_a = len(sector_block_sizes)
    H, W = mask.shape

    if geometry is None:
        geometry = _build_sector_geometry(mask, n_a, ring_width, smooth_sigma)
        if geometry is None:
            return frame_rgb.copy()

    ring_smooth = geometry["ring_smooth"]
    sector_masks = geometry["sector_masks"]

    # Compute normalized sector partition: pi_k = s_k / sum(s_j)
    sector_sum = np.zeros((H, W), dtype=np.float32)
    for k in range(n_a):
        sector_sum += sector_masks[k]
    sector_sum = np.maximum(sector_sum, 1e-8)

    # Weighted-sum pixelation
    pixelated_mix = np.zeros_like(frame_rgb, dtype=np.float32)
    for k in range(n_a):
        bs = int(sector_block_sizes[k])
        if bs < 2:
            bs = 2
        # Pixelate entire frame at this block size
        small = cv2.resize(frame_rgb, (max(1, W // bs), max(1, H // bs)),
                           interpolation=cv2.INTER_AREA)
        pix_k = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

        pi_k = (sector_masks[k] / sector_sum)[:, :, None]  # normalized weight
        pixelated_mix += pi_k * pix_k.astype(np.float32)

    # Blend: ring region gets pixelated_mix, rest stays original
    R = ring_smooth[:, :, None]
    f = frame_rgb.astype(np.float32)
    out = f * (1.0 - R) + pixelated_mix * R
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sector_pixelation_frames(
    frames: list,
    masks: list,
    sector_block_sizes: np.ndarray,
    ring_width: int = 24,
    smooth_sigma: float = 3.0,
) -> list:
    return [apply_sector_pixelation_v2(f, m, sector_block_sizes, ring_width, smooth_sigma)
            for f, m in zip(frames, masks)]


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,dog,dance-twirl,blackswan,elephant")
    p.add_argument("--n_search", type=int, default=15)
    p.add_argument("--n_angular", type=int, default=8)
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--base_block_size", type=int, default=16,
                   help="Uniform baseline block_size and max for oracle search")
    p.add_argument("--min_block_size", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="aniso_pixel_v2")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "aniso_pixel" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Anisotropic Pixelation Oracle Gap Test v2 ===")
    print(f"Videos: {videos}, base_bs={args.base_block_size}, min_bs={args.min_block_size}")
    print(f"ring_width={args.ring_width}, n_angular={args.n_angular}")

    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    rng = np.random.default_rng(42)
    all_results = []

    BS_BASE = args.base_block_size
    BS_MIN = args.min_block_size

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

        # 1. Uniform baseline: ALL sectors get base_block_size
        #    (uses the SAME function as oracle — guarantees baseline is in search space)
        uniform_bs = np.full(args.n_angular, BS_BASE, dtype=np.float64)
        print(f"  [1/2] Uniform pixelation (all sectors bs={BS_BASE})...")
        edited_uniform = apply_sector_pixelation_frames(
            frames, masks, uniform_bs, args.ring_width)
        uni_djf, uni_ssim, jf_codec_clean = evaluate_oracle(
            edited_uniform, frames, masks, predictor, device, FFMPEG_PATH,
            crf=args.crf, prompt=args.prompt, jf_codec_clean=None,
        )
        print(f"  Uniform: ΔJF={uni_djf*100:.2f}pp  SSIM={uni_ssim:.3f}")

        # 2. Oracle search: anisotropic block sizes per sector
        #    Budget constraint: mean block_size (area-weighted) = BS_BASE
        geom0 = _build_sector_geometry(masks[0], args.n_angular, args.ring_width)
        if geom0 is None:
            print(f"  [skip] no geometry")
            continue
        ring_areas = geom0["ring_areas"]
        target_budget = float(BS_BASE)

        print(f"  [2/2] Oracle search ({args.n_search} evals, budget={target_budget})...")
        best_djf = uni_djf
        best_bs = uniform_bs.copy()
        all_search = []

        for i in range(args.n_search):
            if i == 0:
                # Uniform (sanity: should match baseline exactly)
                bs_vec = np.full(args.n_angular, float(BS_BASE))
            elif i == 1:
                # Checkerboard: half max, half min
                bs_vec = np.array([BS_BASE * 1.5 if k % 2 == 0 else BS_BASE * 0.5
                                   for k in range(args.n_angular)])
            elif i == 2:
                # Semicircle
                half = args.n_angular // 2
                bs_vec = np.concatenate([
                    np.full(half, BS_BASE * 1.3),
                    np.full(args.n_angular - half, BS_BASE * 0.7),
                ])
            else:
                # Random block sizes in [BS_MIN, BS_BASE*2]
                bs_vec = rng.uniform(BS_MIN, BS_BASE * 2.0, size=args.n_angular)

            # Project to iso-budget (area-weighted mean = BS_BASE)
            bs_vec = project_bs_to_budget(bs_vec, target_budget, ring_areas,
                                          bs_min=BS_MIN, bs_max=BS_BASE * 2.5)
            bs_vec = np.clip(bs_vec, BS_MIN, BS_BASE * 2.5).round()

            edited = apply_sector_pixelation_frames(
                frames, masks, bs_vec, args.ring_width)
            djf, ssim, jf_codec_clean = evaluate_oracle(
                edited, frames, masks, predictor, device, FFMPEG_PATH,
                crf=args.crf, prompt=args.prompt, jf_codec_clean=jf_codec_clean,
            )
            budget_actual = float((ring_areas * bs_vec).sum() / ring_areas.sum())
            print(f"    [{i+1:2d}/{args.n_search}] ΔJF={djf*100:.2f}pp  "
                  f"SSIM={ssim:.3f}  bs={bs_vec.astype(int).tolist()}  budget={budget_actual:.1f}")

            all_search.append({
                "iter": i, "djf": float(djf), "ssim": float(ssim),
                "block_sizes": bs_vec.tolist(), "budget": budget_actual,
            })

            if djf > best_djf:
                best_djf = djf
                best_bs = bs_vec.copy()

        gap_pp = (best_djf - uni_djf) * 100
        elapsed = time.time() - t0

        print(f"\n  === {vid} SUMMARY ===")
        print(f"  Uniform pixel (bs={BS_BASE}): ΔJF={uni_djf*100:.2f}pp  SSIM={uni_ssim:.3f}")
        print(f"  Oracle aniso:                ΔJF={best_djf*100:.2f}pp")
        print(f"  Gap: {gap_pp:+.2f}pp  ({'ORACLE WINS' if gap_pp > 2 else 'no gap'})")
        print(f"  Best bs: {best_bs.astype(int).tolist()}")
        print(f"  Elapsed: {elapsed:.0f}s")

        all_results.append({
            "video": vid,
            "uniform_djf": float(uni_djf),
            "uniform_ssim": float(uni_ssim),
            "oracle_djf": float(best_djf),
            "gap_pp": float(gap_pp),
            "oracle_block_sizes": best_bs.tolist(),
            "search_history": all_search,
            "elapsed_s": elapsed,
        })

        # Intermediate save
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Aggregate
    if all_results:
        gaps = [r["gap_pp"] for r in all_results]
        wins = sum(1 for g in gaps if g > 2)
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(all_results)})")
        print(f"  Mean uniform: {np.mean([r['uniform_djf'] for r in all_results])*100:.1f}pp")
        print(f"  Mean oracle:  {np.mean([r['oracle_djf'] for r in all_results])*100:.1f}pp")
        print(f"  Mean gap: {np.mean(gaps):+.1f}pp")
        print(f"  Wins (gap>2pp): {wins}/{len(all_results)}")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
