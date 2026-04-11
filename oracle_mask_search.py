"""
oracle_mask_search.py — Non-Ring Oracle Gap Test

Tests whether a spatially-variable (non-ring) feathering mask can beat BRS
at the same SSIM constraint. This is the go/no-go experiment for the
learned-feathering direction.

Parametrization:
  Angular-sector weights: divide the boundary annulus into N_a angular sectors
  (based on angle from mask centroid), each with an independent blend alpha.
  The radial profile is fixed (same raised-cosine as BRS).

  This isolates the question: "Does non-uniform angular weighting help?"
  If yes, there is an oracle gap worth pursuing with a learned model.

Oracle evaluation:
  Real H.264 round-trip + SAM2 inference (no proxy loss).

Usage (on GPU server):
  python oracle_mask_search.py \\
    --videos bear,camel,dog \\
    --n_search 30 \\
    --ring_width 28 \\
    --n_angular 8 \\
    --ssim_floor 0.92 \\
    --device cuda \\
    --max_frames 10 \\
    --tag oracle_test_v1

Results saved to:
  results_v100/oracle_gap/<tag>/results.json
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

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH, DAVIS_MINI_TRAIN
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor, run_tracking, codec_round_trip, frame_quality,
    apply_old_boundary_suppression,   # flat-mean BRS (the effective version)
    _apply_old_brs_proxy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Parametrized suppression: angular-sector weights with flat-mean fill
#
# The flat-mean proxy (old BRS) triggers H.264 DCT artifacts at the boundary
# by creating a low-entropy region that causes DC mismatch ringing on decode.
# We keep this proxy but vary the blend alpha per angular sector.
# ─────────────────────────────────────────────────────────────────────────────

def _build_sector_geometry(
    mask: np.ndarray,
    n_angular: int,
    ring_width: int,
    smooth_sigma: float = 3.0,
):
    """
    Precompute reusable per-frame geometry for sector suppression.

    Returns:
        boundary_ring: smoothed ring weight map (HxW float32)
        sector_masks_smoothed: list of HxW float32 indicator-of-sector maps
                               (after gaussian smoothing) — one per sector
        bg_proxy: flat-mean background proxy (HxWx3 float32)
        ring_areas: per-sector "area under smoothed ring" used as budget weight
    """
    H, W = mask.shape

    kernel = np.ones((ring_width * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    boundary_ring = ((dilated > 0) & (eroded == 0)).astype(np.uint8)

    if boundary_ring.sum() == 0:
        return None

    ring_smooth = cv2.GaussianBlur(boundary_ring.astype(np.float32),
                                   (0, 0), ring_width / 2.0)

    ys, xs = np.where(mask > 0)
    cy, cx = ys.mean(), xs.mean()
    Y, X = np.mgrid[0:H, 0:W]
    angle = np.arctan2(Y - cy, X - cx)
    angle_norm = (angle + np.pi) / (2.0 * np.pi)
    a_idx = np.clip((angle_norm * n_angular).astype(int), 0, n_angular - 1)

    sector_masks_smoothed = []
    ring_areas = np.zeros(n_angular, dtype=np.float64)
    for k in range(n_angular):
        sk = (a_idx == k).astype(np.float32)
        if smooth_sigma > 0:
            sk = cv2.GaussianBlur(sk, (0, 0), smooth_sigma)
        sector_masks_smoothed.append(sk)
        # area = mass of (smoothed sector indicator × smoothed ring)
        ring_areas[k] = float((sk * ring_smooth).sum())

    return {
        "ring_smooth": ring_smooth,
        "sector_masks": sector_masks_smoothed,
        "ring_areas": ring_areas,
        "centroid": (cy, cx),
        "n_angular": n_angular,
        "ring_width": ring_width,
        "smooth_sigma": smooth_sigma,
    }


def apply_sector_suppression(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    sector_alphas: np.ndarray,   # shape (n_angular,) ∈ [0, 1]
    ring_width: int = 24,
    smooth_sigma: float = 3.0,
    geometry: dict | None = None,
) -> np.ndarray:
    """
    Apply flat-mean BRS with per-angular-sector blend alpha.

    Uses the same flat-mean proxy as idea1_old (which triggers H.264 artifacts),
    but varies the blend alpha by angular sector around the mask centroid.

    The pixel blend weight is computed as:
        weight(x,y) = ring_smooth(x,y) * sum_k(alpha_k * sector_mask_smoothed_k(x,y))

    sector_alphas: shape (N_a,), values ∈ [0, 1]
    geometry: optional precomputed sector geometry; if None, will be computed.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    n_a = len(sector_alphas)

    if geometry is None:
        geometry = _build_sector_geometry(mask, n_a, ring_width, smooth_sigma)
        if geometry is None:
            return frame_rgb.copy()

    ring_smooth = geometry["ring_smooth"]
    sector_masks = geometry["sector_masks"]

    # alpha_map(x,y) = sum_k alpha_k * sector_mask_smoothed_k(x,y)
    alpha_map = np.zeros_like(ring_smooth, dtype=np.float32)
    for k in range(n_a):
        alpha_map += float(sector_alphas[k]) * sector_masks[k]

    weight = np.clip(ring_smooth * alpha_map, 0.0, 1.0)

    bg_proxy = _apply_old_brs_proxy(frame_rgb, mask, dilation_px=ring_width * 2)

    f = frame_rgb.astype(np.float32)
    w = weight[:, :, None]
    edited = f * (1.0 - w) + bg_proxy.astype(np.float32) * w
    return np.clip(edited, 0, 255).astype(np.uint8)


def apply_sector_suppression_frames(
    frames: list,
    masks: list,
    sector_alphas: np.ndarray,
    ring_width: int = 24,
    smooth_sigma: float = 3.0,
    target_budget: float | None = None,
) -> list:
    """Apply sector suppression to a list of frames.

    If target_budget is given, re-project sector_alphas per frame based on
    that frame's ring_areas to maintain strict iso-budget across the video
    (fixes Bug #1: budget drift due to mask shape changes over time).
    """
    out = []
    for f, m in zip(frames, masks):
        if target_budget is not None and m.sum() > 0:
            geom = _build_sector_geometry(m, len(sector_alphas), ring_width, smooth_sigma)
            if geom is not None:
                frame_alphas = project_to_budget(
                    sector_alphas.copy(), target_budget, geom["ring_areas"]
                )
                out.append(apply_sector_suppression(
                    f, m, frame_alphas, ring_width, smooth_sigma, geometry=geom
                ))
                continue
        out.append(apply_sector_suppression(f, m, sector_alphas, ring_width, smooth_sigma))
    return out


def project_to_budget(
    sector_alphas: np.ndarray,
    target_budget: float,
    ring_areas: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Project sector_alphas to satisfy:
        - alpha_k ∈ [0, 1]
        - sum_k(ring_areas[k] * alpha_k) / sum(ring_areas) == target_budget

    Uses iterative shift+clip projection (water-filling style).
    """
    a = np.clip(sector_alphas.astype(np.float64), 0.0, 1.0)
    total_area = float(ring_areas.sum())
    if total_area <= 0:
        return a

    # Weighted mean
    def w_mean(x):
        return float((ring_areas * x).sum() / total_area)

    for _ in range(max_iter):
        diff = target_budget - w_mean(a)
        if abs(diff) < 1e-5:
            break
        # add diff uniformly, then clip
        a = a + diff
        a = np.clip(a, 0.0, 1.0)

    return a


# ─────────────────────────────────────────────────────────────────────────────
# Oracle evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_oracle(
    edited_frames: list,
    orig_frames: list,
    masks: list,
    predictor,
    device: torch.device,
    ffmpeg_path: str,
    crf: int = 23,
    prompt: str = "point",
    jf_codec_clean: float | None = None,
) -> tuple[float, float, float | None]:
    """
    Returns (delta_jf_codec, mean_ssim, jf_codec_clean_or_none).

    ΔJF = jf_codec_clean − jf_codec_adv  (matches pilot_fancy_eval.py formula).
    jf_codec_clean is SAM2 on codec-decoded ORIGINAL frames (computed once per video).

    If jf_codec_clean is provided, skip the clean-codec SAM2 run.
    """
    # SSIM (mean over ALL frames — not capped, to ensure reported SSIM reflects full video)
    ssim_vals = [frame_quality(o, e)[0] for o, e in zip(orig_frames, edited_frames)]
    mean_ssim = float(np.mean(ssim_vals))

    # Codec round-trip on EDITED frames
    decoded = codec_round_trip(edited_frames, ffmpeg_path, crf)
    if decoded is None:
        return 0.0, mean_ssim, jf_codec_clean

    # Codec-clean JF (computed once): SAM2 on codec(original) to match fancy_eval baseline
    returned_jf_codec_clean = jf_codec_clean
    if jf_codec_clean is None:
        codec_orig = codec_round_trip(orig_frames, ffmpeg_path, crf)
        if codec_orig is None:
            return 0.0, mean_ssim, None
        _, jfc, _, _ = run_tracking(codec_orig, masks, predictor, device, prompt=prompt)
        returned_jf_codec_clean = float(jfc)
        print(f"    JF_codec_clean = {returned_jf_codec_clean:.4f}")

    # Codec-adv JF: SAM2 on codec(edited)
    _, jf_codec_adv, _, _ = run_tracking(decoded, masks, predictor, device, prompt=prompt)
    jf_codec_adv = float(jf_codec_adv)
    delta_jf = returned_jf_codec_clean - jf_codec_adv

    return delta_jf, mean_ssim, returned_jf_codec_clean


# ─────────────────────────────────────────────────────────────────────────────
# BRS baseline
# ─────────────────────────────────────────────────────────────────────────────

def brs_baseline(
    frames: list,
    masks: list,
    predictor,
    device: torch.device,
    ffmpeg_path: str,
    crf: int = 23,
    ring_width: int = 24,
    alpha: float = 0.80,
    prompt: str = "point",
) -> tuple[float, float, float]:
    """Run flat-mean BRS(rw, alpha) and return (delta_jf, ssim, jf_codec_clean)."""
    edited = [
        apply_old_boundary_suppression(f, m, ring_width=ring_width, blend_alpha=alpha)
        for f, m in zip(frames, masks)
    ]
    delta_jf, ssim, jf_codec_clean = evaluate_oracle(
        edited, frames, masks, predictor, device, ffmpeg_path, crf,
        prompt=prompt, jf_codec_clean=None
    )
    return delta_jf, ssim, jf_codec_clean


# ─────────────────────────────────────────────────────────────────────────────
# Random search + COBYLA refinement
# ─────────────────────────────────────────────────────────────────────────────

def random_search(
    frames: list,
    masks: list,
    predictor,
    device: torch.device,
    ffmpeg_path: str,
    n_search: int = 30,
    n_angular: int = 8,
    ring_width: int = 24,
    ssim_floor: float = 0.90,
    crf: int = 23,
    prompt: str = "point",
    jf_codec_clean: float | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    target_budget: float = 0.80,
    ring_areas: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, list, float]:
    """
    Random search over N_a angular sector weights.

    Returns:
        best_alphas, best_delta_jf, best_ssim, best_obj, all_results, jf_codec_clean
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if ring_areas is None:
        # fallback: assume uniform sector areas
        ring_areas = np.ones(n_angular)

    best_alphas = np.full(n_angular, target_budget)  # BRS-equivalent initialization
    best_delta_jf = -999.0
    best_ssim = 0.0
    best_obj = -999.0
    all_results = []

    for i in range(n_search):
        # Sampling strategy:
        # - First 5: sweep uniform alphas (0.70, 0.80, 0.85, 0.90, 0.93) — BRS family
        # - Rest: random with bias toward [0.80, 0.95]
        # ISO-BUDGET COMPARISON: every pattern is projected so that the
        # *area-weighted* sector budget equals the BRS uniform target_budget.
        # This isolates SPATIAL ANISOTROPY from total distortion budget.
        if i == 0:
            alphas = np.full(n_angular, target_budget)        # uniform baseline
        elif i == 1:
            # Checkerboard: alternate high/low (safe for any n_angular)
            alphas = np.array([target_budget + 0.2 if k % 2 == 0 else target_budget - 0.2
                               for k in range(n_angular)])
        elif i == 2:
            # Semicircle: first half stronger, second half weaker
            half = n_angular // 2
            alphas = np.concatenate([
                np.full(half, target_budget + 0.15),
                np.full(n_angular - half, target_budget - 0.15),
            ])
        elif i == 3:
            alphas = np.full(n_angular, target_budget - 0.05)
            alphas[0] = target_budget + 0.30
        elif i == 4:
            alphas = np.full(n_angular, target_budget - 0.10)
            alphas[0] = target_budget + 0.40
            alphas[min(n_angular // 2, n_angular - 1)] = target_budget + 0.40
        else:
            alphas = rng.uniform(0.20, 1.30, size=n_angular)

        # Project onto exact area-weighted budget
        alphas = project_to_budget(alphas, target_budget, ring_areas)

        edited = apply_sector_suppression_frames(
            frames, masks, alphas, ring_width, target_budget=target_budget)
        delta_jf, ssim, jf_codec_clean = evaluate_oracle(
            edited, frames, masks, predictor, device, ffmpeg_path, crf,
            prompt=prompt, jf_codec_clean=jf_codec_clean
        )
        # Objective: ΔJF (in pp scale) - penalty for SSIM violation
        delta_jf_pp = delta_jf * 100.0
        ssim_penalty = max(0.0, ssim_floor - ssim) * 600.0  # large penalty in pp units
        obj = delta_jf_pp - ssim_penalty

        all_results.append({
            "iter": i,
            "alphas": alphas.tolist(),
            "delta_jf": float(delta_jf),
            "ssim": float(ssim),
            "obj": float(obj),
        })

        if verbose:
            print(f"    [{i+1:2d}/{n_search}] ΔJF={delta_jf_pp:.2f}pp  SSIM={ssim:.3f}  "
                  f"obj={obj:.2f}  alphas=[{', '.join(f'{a:.2f}' for a in alphas[:4])}...]")

        if obj > best_obj:
            best_obj = obj
            best_delta_jf = delta_jf
            best_ssim = ssim
            best_alphas = alphas.copy()

    return best_alphas, best_delta_jf, best_ssim, best_obj, all_results, jf_codec_clean


def cobyla_refine(
    frames: list,
    masks: list,
    predictor,
    device: torch.device,
    ffmpeg_path: str,
    init_alphas: np.ndarray,
    n_steps: int = 10,
    ring_width: int = 24,
    ssim_floor: float = 0.90,
    crf: int = 23,
    prompt: str = "point",
    jf_codec_clean: float | None = None,
    verbose: bool = True,
    target_budget: float = 0.80,
    ring_areas: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Budget-projected coordinate-wise local search starting from init_alphas.
    Every candidate is projected onto the exact area-weighted budget, so the
    iso-budget guarantee is preserved.
    """
    if ring_areas is None:
        ring_areas = np.ones(len(init_alphas))

    n_a = len(init_alphas)
    current = project_to_budget(init_alphas.copy(), target_budget, ring_areas)
    best = {"alphas": current.copy(), "delta_jf": 0.0, "ssim": 0.0, "obj": -999.0}
    call_count = [0]

    def evaluate(alphas: np.ndarray) -> tuple[float, float, float]:
        nonlocal jf_codec_clean
        edited = apply_sector_suppression_frames(
            frames, masks, alphas, ring_width, target_budget=target_budget)
        delta_jf, ssim, jf_codec_clean = evaluate_oracle(
            edited, frames, masks, predictor, device, ffmpeg_path, crf,
            prompt=prompt, jf_codec_clean=jf_codec_clean
        )
        delta_jf_pp = delta_jf * 100.0
        ssim_penalty = max(0.0, ssim_floor - ssim) * 600.0
        obj = delta_jf_pp - ssim_penalty
        call_count[0] += 1
        if verbose:
            print(f"    [REFINE {call_count[0]:2d}] ΔJF={delta_jf_pp:.2f}pp  "
                  f"SSIM={ssim:.3f}  obj={obj:.2f}  budget={float((ring_areas*alphas).sum()/ring_areas.sum()):.3f}")
        return obj, delta_jf, ssim

    # Initial baseline
    obj0, dj0, ss0 = evaluate(current)
    if obj0 > best["obj"]:
        best.update({"alphas": current.copy(), "delta_jf": dj0, "ssim": ss0, "obj": obj0})

    # Coordinate-pair swap descent: pick (i, j), shift mass from i to j
    delta = 0.15
    rng = np.random.default_rng(123)
    for step in range(n_steps):
        i, j = rng.choice(n_a, size=2, replace=False)
        trial = current.copy()
        # Move budget from i to j, area-weighted
        # We want: new_i*A_i + new_j*A_j == old_i*A_i + old_j*A_j
        # So: shift α_i by -d, then α_j by +d * A_i/A_j
        if ring_areas[j] < 1e-6:
            continue
        d = delta * (1.0 + 0.5 * (rng.random() - 0.5))
        trial[i] = trial[i] - d
        trial[j] = trial[j] + d * ring_areas[i] / ring_areas[j]
        trial = np.clip(trial, 0.0, 1.0)
        # Re-project to enforce exact budget
        trial = project_to_budget(trial, target_budget, ring_areas)
        obj, dj, ss = evaluate(trial)
        if obj > best["obj"]:
            best.update({"alphas": trial.copy(), "delta_jf": dj, "ssim": ss, "obj": obj})
            current = trial.copy()
        # else: keep current

    return best["alphas"], best["delta_jf"], best["ssim"], best["obj"], jf_codec_clean


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,camel,dog",
                   help="Comma-separated DAVIS video names")
    p.add_argument("--n_search", type=int, default=30,
                   help="Number of random search evaluations")
    p.add_argument("--cobyla_steps", type=int, default=10,
                   help="COBYLA refinement steps after random search")
    p.add_argument("--n_angular", type=int, default=8,
                   help="Number of angular sectors")
    p.add_argument("--ring_width", type=int, default=24,
                   help="Boundary ring width in pixels (default: 24, matching flat-mean BRS baseline)")
    p.add_argument("--ssim_floor", type=float, default=0.92,
                   help="SSIM floor constraint")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point", choices=["point", "mask"])
    p.add_argument("--max_frames", type=int, default=10,
                   help="Max frames per video")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="oracle_test_v1")
    p.add_argument("--save_dir", default="results_v100/oracle_gap")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    if not videos:
        videos = DAVIS_MINI_TRAIN[:3]

    save_dir = Path(ROOT) / args.save_dir / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Oracle Gap Test ===")
    print(f"Videos: {videos}")
    print(f"n_search={args.n_search}, cobyla_steps={args.cobyla_steps}, "
          f"n_angular={args.n_angular}, ring_width={args.ring_width}")
    print(f"SSIM floor: {args.ssim_floor}, CRF: {args.crf}")

    # Build SAM2 predictor
    print("Loading SAM2...")
    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)

    all_results = []
    summary_rows = []

    for vid in videos:
        print(f"\n{'='*60}")
        print(f"Video: {vid}")
        print(f"{'='*60}")

        try:
            frames, masks, _ = load_single_video(
                DAVIS_ROOT, vid, max_frames=args.max_frames
            )
        except Exception as e:
            print(f"  [skip] Failed to load {vid}: {e}")
            continue

        if not frames or not masks:
            print(f"  [skip] Empty video or masks: {vid}")
            continue

        frames = frames[:args.max_frames]
        masks = masks[:args.max_frames]
        print(f"  Loaded {len(frames)} frames, frame size {frames[0].shape[:2]}")

        t0 = time.time()

        # 1. BRS baseline (flat-mean, rw=24, α=0.80) — matches fancy_v1
        print(f"  [1/3] BRS baseline (rw={args.ring_width}, α=0.80, flat-mean, prompt={args.prompt})...")
        try:
            brs_delta_jf, brs_ssim, jf_codec_clean = brs_baseline(
                frames, masks, predictor, device, FFMPEG_PATH,
                crf=args.crf, ring_width=args.ring_width, alpha=0.80,
                prompt=args.prompt,
            )
        except Exception as e:
            print(f"  [error] BRS failed: {e}")
            continue

        if brs_delta_jf is None or jf_codec_clean is None:
            print(f"  [skip] BRS returned None (codec failure?) for {vid}")
            continue

        brs_delta_jf_pp = brs_delta_jf * 100.0
        print(f"  BRS: ΔJF={brs_delta_jf_pp:.2f}pp  SSIM={brs_ssim:.3f}  "
              f"JF_codec_clean={jf_codec_clean:.4f}")

        # Compute first-frame sector geometry → ring_areas (for budget projection)
        first_geom = _build_sector_geometry(
            masks[0], args.n_angular, args.ring_width, smooth_sigma=3.0
        )
        if first_geom is None:
            print(f"  [skip] cannot build sector geometry for {vid}")
            continue
        ring_areas = first_geom["ring_areas"]
        target_budget = 0.80   # match BRS uniform alpha
        print(f"  Ring areas (sector budgets): {[f'{a/ring_areas.sum():.3f}' for a in ring_areas]}")
        print(f"  Target budget: {target_budget:.2f}")

        # 2. Random search for non-ring oracle (iso-budget enforced)
        print(f"  [2/3] Random search ({args.n_search} evals, iso-budget)...")
        try:
            best_alphas, rs_delta_jf, rs_ssim, rs_obj, rs_history, jf_codec_clean = random_search(
                frames, masks, predictor, device, FFMPEG_PATH,
                n_search=args.n_search,
                n_angular=args.n_angular,
                ring_width=args.ring_width,
                ssim_floor=args.ssim_floor,
                crf=args.crf,
                prompt=args.prompt,
                jf_codec_clean=jf_codec_clean,
                verbose=True,
                target_budget=target_budget,
                ring_areas=ring_areas,
            )
        except Exception as e:
            print(f"  [error] Random search failed: {e}")
            continue

        print(f"  Best random: ΔJF={rs_delta_jf*100:.2f}pp  SSIM={rs_ssim:.3f}  "
              f"alphas={[f'{a:.2f}' for a in best_alphas]}")

        # 3. Budget-projected refinement from best random
        print(f"  [3/3] Refinement ({args.cobyla_steps} steps, iso-budget enforced)...")
        try:
            oracle_alphas, oracle_delta_jf, oracle_ssim, oracle_obj, jf_codec_clean = cobyla_refine(
                frames, masks, predictor, device, FFMPEG_PATH,
                init_alphas=best_alphas,
                n_steps=args.cobyla_steps,
                ring_width=args.ring_width,
                ssim_floor=args.ssim_floor,
                crf=args.crf,
                prompt=args.prompt,
                jf_codec_clean=jf_codec_clean,
                verbose=True,
                target_budget=target_budget,
                ring_areas=ring_areas,
            )
        except Exception as e:
            print(f"  [error] COBYLA failed: {e}, using random search result")
            oracle_alphas = best_alphas
            oracle_delta_jf = rs_delta_jf
            oracle_ssim = rs_ssim

        gap_pp = (oracle_delta_jf - brs_delta_jf) * 100.0

        t1 = time.time()
        print(f"\n  === {vid} SUMMARY ===")
        print(f"  BRS(rw={args.ring_width},flat-mean,α=0.80): ΔJF={brs_delta_jf*100:.2f}pp  SSIM={brs_ssim:.3f}")
        print(f"  Oracle (sector):      ΔJF={oracle_delta_jf*100:.2f}pp  SSIM={oracle_ssim:.3f}")
        print(f"  Gap: {gap_pp:+.2f}pp  ({'ORACLE WINS' if gap_pp > 2.0 else 'no gap' if gap_pp > -2.0 else 'BRS wins'})")
        print(f"  Oracle alphas: {[f'{a:.2f}' for a in oracle_alphas]}")
        print(f"  Elapsed: {t1-t0:.0f}s")

        # Verify final budget compliance
        oracle_budget = float((ring_areas * oracle_alphas).sum() / ring_areas.sum())
        print(f"  Oracle final area-weighted budget: {oracle_budget:.4f} (target {target_budget:.4f})")

        vid_result = {
            "video": vid,
            "n_frames": len(frames),
            "jf_codec_clean": jf_codec_clean,
            "target_budget": float(target_budget),
            "ring_areas": ring_areas.tolist(),
            "oracle_final_budget": oracle_budget,
            "brs": {
                "ring_width": args.ring_width,
                "alpha": 0.80,
                "delta_jf": brs_delta_jf,
                "ssim": brs_ssim,
            },
            "oracle": {
                "n_search": args.n_search,
                "cobyla_steps": args.cobyla_steps,
                "n_angular": args.n_angular,
                "delta_jf": oracle_delta_jf,
                "ssim": oracle_ssim,
                "alphas": oracle_alphas.tolist(),
                "gap_vs_brs": gap_pp,
            },
            "search_history": rs_history,
            "elapsed_s": t1 - t0,
        }
        all_results.append(vid_result)
        summary_rows.append({
            "video": vid,
            "brs_delta_jf": brs_delta_jf,
            "brs_ssim": brs_ssim,
            "oracle_delta_jf": oracle_delta_jf,
            "oracle_ssim": oracle_ssim,
            "gap": oracle_delta_jf - brs_delta_jf,
        })

    # Aggregate
    if summary_rows:
        gaps_pp = [r["gap"] * 100.0 for r in summary_rows]
        oracle_jfs_pp = [r["oracle_delta_jf"] * 100.0 for r in summary_rows]
        brs_jfs_pp = [r["brs_delta_jf"] * 100.0 for r in summary_rows]
        mean_gap = float(np.mean(gaps_pp))
        mean_oracle_jf = float(np.mean(oracle_jfs_pp))
        mean_brs_jf = float(np.mean(brs_jfs_pp))

        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(summary_rows)} videos)")
        print(f"  BRS mean ΔJF: {mean_brs_jf:.2f}pp")
        print(f"  Oracle mean ΔJF: {mean_oracle_jf:.2f}pp")
        print(f"  Mean gap: {mean_gap:+.2f}pp")
        if mean_gap > 5.0:
            verdict = "ORACLE GAP EXISTS — proceed to learned model"
        elif mean_gap > 2.0:
            verdict = "SMALL GAP — marginal evidence for learned model"
        else:
            verdict = "NO ORACLE GAP — do not proceed with learned model"
        print(f"  Verdict: {verdict}")

        final = {
            "aggregate": {
                "n_videos": len(summary_rows),
                "mean_brs_delta_jf": mean_brs_jf,
                "mean_oracle_delta_jf": mean_oracle_jf,
                "mean_gap_pp": mean_gap,
                "verdict": verdict,
            },
            "per_video": all_results,
            "args": vars(args),
        }
    else:
        final = {"per_video": [], "aggregate": {"n_videos": 0, "verdict": "no data"}}

    out_path = save_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
