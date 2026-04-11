"""
Baseline comparison: BRS vs simple publisher-side alternatives at matched SSIM.

For each baseline method, sweeps parameters to find the setting achieving
SSIM ≈ 0.93 (matching BRS operating point), then evaluates post-codec ΔJF.

Usage:
  python eval_baselines.py \
      --videos bear,breakdance,car-shadow \
      --max_frames 50 --tag baselines_v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    apply_old_boundary_suppression,  # flat-mean proxy (the effective version)
    apply_boundary_blur,
    build_predictor,
    run_tracking,
    codec_round_trip,
    frame_quality,
)

TARGET_SSIM = 0.93

# LPIPS (lazy-loaded)
_lpips_fn = None

def compute_lpips(orig_frames, edited_frames, device="cuda", n_max=20):
    """Mean LPIPS across frames (lazy-loads the model on first call)."""
    global _lpips_fn
    try:
        import lpips
        if _lpips_fn is None:
            _lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        vals = []
        for o, e in zip(orig_frames[:n_max], edited_frames[:n_max]):
            o_t = torch.from_numpy(o / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            e_t = torch.from_numpy(e / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                vals.append(_lpips_fn(o_t * 2 - 1, e_t * 2 - 1).item())
        return float(np.mean(vals))
    except ImportError:
        return float("nan")


# ── Baseline edit functions ──────────────────────────────────────────────────

def baseline_global_blur(frame: np.ndarray, mask: np.ndarray,
                         sigma: float = 5.0) -> np.ndarray:
    """Gaussian blur the entire frame."""
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def baseline_boundary_blur(frame: np.ndarray, mask: np.ndarray,
                           sigma: float = 8.0) -> np.ndarray:
    """Gaussian blur restricted to the boundary ring (reuses pilot's version)."""
    # Map sigma to ring_width and blend_alpha for the existing function
    rw = max(4, int(round(sigma * 1.5)))
    return apply_boundary_blur(frame, mask, ring_width=rw, blend_alpha=0.85)


def baseline_pixelation(frame: np.ndarray, mask: np.ndarray,
                        block_size: int = 8) -> np.ndarray:
    """Downsample then upsample the boundary region (pixelation)."""
    if mask.sum() == 0:
        return frame.copy()
    H, W = frame.shape[:2]
    # Build boundary ring via morphological ops
    kw = max(block_size, 8)
    kern = np.ones((kw * 2 + 1, kw * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask, kern)
    eroded = cv2.erode(mask, kern)
    ring = ((dilated > 0) & (eroded == 0)).astype(np.float32)
    if ring.sum() == 0:
        return frame.copy()
    # Pixelate: downsample then upsample
    bs = max(2, block_size)
    small = cv2.resize(frame, (W // bs, H // bs), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    # Blend in ring only (soft edge)
    ring_smooth = cv2.GaussianBlur(ring, (0, 0), kw / 2.0)[:, :, None]
    out = frame.astype(np.float32) * (1 - ring_smooth) + pixelated.astype(np.float32) * ring_smooth
    return np.clip(out, 0, 255).astype(np.uint8)


def baseline_inpainting(frame: np.ndarray, mask: np.ndarray,
                        radius: int = 5) -> np.ndarray:
    """cv2.inpaint on the mask boundary region."""
    if mask.sum() == 0:
        return frame.copy()
    kw = max(8, radius)
    kern = np.ones((kw * 2 + 1, kw * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask, kern)
    eroded = cv2.erode(mask, kern)
    ring = ((dilated > 0) & (eroded == 0)).astype(np.uint8)
    if ring.sum() == 0:
        return frame.copy()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, ring, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def baseline_morph_fill(frame: np.ndarray, mask: np.ndarray,
                        ring_width: int = 12) -> np.ndarray:
    """Erode mask by ring_width, fill boundary with mean color."""
    if mask.sum() == 0:
        return frame.copy()
    kern = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), np.uint8)
    eroded = cv2.erode(mask, kern)
    ring = ((mask > 0) & (eroded == 0)).astype(np.uint8)
    if ring.sum() == 0:
        return frame.copy()
    # Mean color of background near the object
    dilated = cv2.dilate(mask, kern)
    bg_mask = (dilated > 0) & (mask == 0)
    if bg_mask.sum() > 0:
        mean_color = frame[bg_mask].mean(axis=0)
    else:
        mean_color = frame.mean(axis=(0, 1))
    out = frame.copy()
    out[ring > 0] = mean_color.astype(np.uint8)
    # Light blur to soften edges
    ring_f = ring.astype(np.float32)
    ring_smooth = cv2.GaussianBlur(ring_f, (0, 0), ring_width / 3.0)[:, :, None]
    result = frame.astype(np.float32) * (1 - ring_smooth) + out.astype(np.float32) * ring_smooth
    return np.clip(result, 0, 255).astype(np.uint8)


def baseline_brs(frame: np.ndarray, mask: np.ndarray,
                 blend_alpha: float = 0.6) -> np.ndarray:
    """Our BRS method (flat-mean proxy, the effective version) as reference."""
    return apply_old_boundary_suppression(frame, mask, ring_width=24,
                                          blend_alpha=blend_alpha)


# ── Baseline registry: name -> (function, param_name, sweep_values) ──────────

BASELINES = {
    "global_blur":    (baseline_global_blur,    "sigma",      [2, 4, 6, 8, 12, 16, 24]),
    "boundary_blur":  (baseline_boundary_blur,  "sigma",      [3, 5, 8, 12, 16, 24]),
    "pixelation":     (baseline_pixelation,     "block_size", [4, 6, 8, 12, 16, 24, 32]),
    "inpainting":     (baseline_inpainting,     "radius",     [3, 5, 8, 12, 16, 24]),
    "morph_fill":     (baseline_morph_fill,     "ring_width", [4, 8, 12, 16, 24, 32]),
    "brs":            (baseline_brs,            "blend_alpha", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
}


# ── SSIM-matching sweep ─────────────────────────────────────────────────────

def ssim_match_sweep(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    edit_fn,
    param_name: str,
    param_values: list,
    target_ssim: float,
) -> Tuple[float, float, list]:
    """
    Try each parameter value, compute mean SSIM on ALL frames,
    return (best_param, achieved_ssim_full, edited_frames_at_best_param).

    Uses all frames for SSIM computation (not a sample) to ensure
    the reported SSIM is the actual full-video value.
    """
    best_param, best_ssim, best_dist = None, None, float("inf")
    for pval in param_values:
        ssims = []
        for i in range(len(frames)):
            edited = edit_fn(frames[i], masks[i], **{param_name: pval})
            s, _ = frame_quality(frames[i], edited)
            ssims.append(s)
        mean_s = float(np.mean(ssims))
        dist = abs(mean_s - target_ssim)
        if dist < best_dist:
            best_dist = dist
            best_param = pval
            best_ssim = mean_s
    # Apply best param to all frames (already computed above, but
    # we need the list; re-apply is cheap vs SAM2 tracking)
    edited_all = [edit_fn(f, m, **{param_name: best_param})
                  for f, m in zip(frames, masks)]
    return best_param, best_ssim, edited_all


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline comparison at matched SSIM")
    p.add_argument("--videos", default="",
                   help="Comma-separated video names (empty = DAVIS_MINI_VAL)")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="baselines_v1")
    p.add_argument("--prompt", default="point", choices=["point", "mask"])
    p.add_argument("--target_ssim", type=float, default=0.93)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] or DAVIS_MINI_VAL

    out_dir = Path("results_v100/baselines") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baselines] videos={videos}, target_SSIM={args.target_ssim}")
    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    all_results = []

    for vid in videos:
        print(f"\n{'='*60}\n  {vid}\n{'='*60}")
        frames, masks, _ = load_single_video(args.davis_root, vid,
                                             max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed"); continue

        # Clean baseline
        _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, args.prompt)
        print(f"  clean JF = {jf_clean:.4f}")
        if jf_clean < 0.3:
            print(f"  [skip] JF too low"); continue

        # Codec-clean baseline (must succeed for fair comparison)
        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_frames is None:
            print(f"  [skip] codec round-trip failed on clean frames for {vid}")
            continue
        _, jf_cc, _, _ = run_tracking(codec_frames, masks, predictor, device, args.prompt)
        print(f"  codec-clean JF = {jf_cc:.4f}")

        vid_row = {"video": vid, "jf_clean": jf_clean, "jf_codec_clean": jf_cc}

        for bname, (bfn, pname, pvals) in BASELINES.items():
            t0 = time.time()
            best_p, achieved_ssim, edited = ssim_match_sweep(
                frames, masks, bfn, pname, pvals, args.target_ssim)
            # Codec round-trip then track
            codec_edited = codec_round_trip(edited, args.ffmpeg_path, args.crf)
            if codec_edited is None:
                print(f"  [skip] {bname} codec failed on {vid}")
                continue  # skip this baseline — do not mix codec/non-codec pipelines
            _, jf_adv, j_adv, f_adv = run_tracking(
                codec_edited, masks, predictor, device, args.prompt)
            delta = jf_cc - jf_adv  # both are post-codec (fair comparison guaranteed)
            elapsed = time.time() - t0

            # LPIPS (perceptual distance — catches pixelation artifacts SSIM misses)
            lp = compute_lpips(frames, edited, device=str(device))

            vid_row[f"{bname}_param"] = best_p
            vid_row[f"{bname}_ssim"] = achieved_ssim
            vid_row[f"{bname}_lpips"] = lp
            vid_row[f"{bname}_jf"] = jf_adv
            vid_row[f"{bname}_delta"] = delta

            print(f"  {bname:16s}  {pname}={best_p:<6}  "
                  f"SSIM={achieved_ssim:.4f}  LPIPS={lp:.4f}  JF={jf_adv:.4f}  "
                  f"ΔJF={delta:+.4f}  ({elapsed:.1f}s)")

        all_results.append(vid_row)
        # Intermediate save
        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # ── Aggregate summary ────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print(f"AGGREGATE ({len(all_results)} videos, target SSIM={args.target_ssim})")
        print(f"{'='*70}")
        print(f"{'method':16s}  {'SSIM':>6s}  {'LPIPS':>6s}  {'mean_dJF':>9s}  {'med_dJF':>9s}")
        print("-" * 55)
        for bname in BASELINES:
            ssims  = [r[f"{bname}_ssim"]  for r in all_results if f"{bname}_ssim" in r]
            lpipss = [r[f"{bname}_lpips"] for r in all_results if f"{bname}_lpips" in r]
            deltas = [r[f"{bname}_delta"] for r in all_results if f"{bname}_delta" in r]
            if deltas:
                lp_str = f"{np.nanmean(lpipss):6.3f}" if lpipss else "   N/A"
                print(f"{bname:16s}  {np.mean(ssims):6.3f}  {lp_str}  "
                      f"{np.mean(deltas)*100:+8.1f}pp  {np.median(deltas)*100:+8.1f}pp")

    # Final save
    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
