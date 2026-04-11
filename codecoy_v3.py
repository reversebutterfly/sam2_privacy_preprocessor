"""
codecoy_v3.py — CoDecoy v3: DINOv2 Feature-Space Differentiable Decoy

Core idea: optimize a low-frequency perturbation in the background so that
DINOv2 features of the decoy region MATCH the target region's features.

Unlike v1/v2 which only matched pixel-space color, this optimizes in
DINOv2's semantic feature space — the same kind of features SAM2's Hiera
encoder produces.

Combined with BRS boundary weakening on the real target.

Architecture:
  - Frozen DINOv2-ViT-S/14 as feature extractor
  - Learnable low-frequency perturbation δ in YUV space (32x32 resolution)
  - Upsampled + blended into background decoy region
  - Loss: cosine similarity in DINOv2 patch-token space
  - Constraint: SSIM >= 0.90, high-frequency penalty

Usage:
  python codecoy_v3.py --videos bear,blackswan,dog --device cuda --tag v3
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
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor, run_tracking, codec_round_trip, frame_quality,
    apply_old_boundary_suppression,
)

# ─────────────────────────────────────────────────────────────────────────────
# DINOv2
# ─────────────────────────────────────────────────────────────────────────────

_dino = None

def get_dino(device):
    global _dino
    if _dino is None:
        import importlib
        cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.exists(cache_dir):
            # Load from local cache without network
            sys.path.insert(0, cache_dir)
            hub_module = importlib.import_module("hubconf")
            _dino = hub_module.dinov2_vits14(pretrained=True).to(device).eval()
            sys.path.pop(0)
        else:
            _dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    verbose=False).to(device).eval()
        for p in _dino.parameters():
            p.requires_grad_(False)
    return _dino

DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINO_STD = torch.tensor([0.229, 0.224, 0.225])

def dino_preprocess(img_t, device):
    """img_t: (1, 3, H, W) in [0, 1] → DINOv2 input."""
    img_r = F.interpolate(img_t, size=(518, 518), mode="bilinear", align_corners=False)
    mean = DINO_MEAN.to(device).view(1, 3, 1, 1)
    std = DINO_STD.to(device).view(1, 3, 1, 1)
    return (img_r - mean) / std

def dino_patch_features(img_t, device, grad=False):
    """Extract DINOv2 patch tokens. img_t: (1, 3, H, W) in [0,1]. Returns (1, 37*37, 384).
    Set grad=True when optimizing through DINOv2 (backprop to input)."""
    dino = get_dino(device)
    inp = dino_preprocess(img_t, device)
    if grad:
        out = dino.forward_features(inp)
    else:
        with torch.no_grad():
            out = dino.forward_features(inp)
    return out["x_norm_patchtokens"]  # (1, 1369, 384)

def get_target_prototype(frame_t, mask_t, device):
    """Get DINOv2 feature prototype for the target region."""
    feats = dino_patch_features(frame_t, device)  # (1, 1369, 384)
    # Map mask to patch space (37x37)
    mask_r = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0).float(),
                            size=(37, 37), mode="nearest")[0, 0]  # (37, 37)
    mask_flat = mask_r.flatten() > 0.5  # (1369,)
    if mask_flat.sum() < 1:
        return feats.mean(dim=1)  # fallback: global mean
    target_feats = feats[0, mask_flat]  # (K, 384)
    return target_feats.mean(dim=0, keepdim=True)  # (1, 384)


# ─────────────────────────────────────────────────────────────────────────────
# Decoy location
# ─────────────────────────────────────────────────────────────────────────────

def find_decoy_region(mask, min_dist=80):
    """Find background region for decoy. Returns (cy, cx) or None."""
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    obj_cy, obj_cx = ys.mean(), xs.mean()
    Y, X = np.mgrid[0:H, 0:W]
    dist = np.sqrt((Y - obj_cy)**2 + (X - obj_cx)**2)
    valid = (mask == 0) & (dist > min_dist)
    pad = 60
    valid[:pad, :] = False; valid[-pad:, :] = False
    valid[:, :pad] = False; valid[:, -pad:] = False
    if not valid.any():
        return None
    dist[~valid] = 0
    idx = np.unravel_index(dist.argmax(), dist.shape)
    return int(idx[0]), int(idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 Feature-Space Decoy Optimization
# ─────────────────────────────────────────────────────────────────────────────

def optimize_decoy(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    decoy_cy: int,
    decoy_cx: int,
    device: torch.device,
    decoy_h: int = 80,
    decoy_w: int = 80,
    n_iter: int = 200,
    lr: float = 0.02,
    alpha_max: float = 0.5,
    lf_res: int = 32,
) -> np.ndarray:
    """
    Optimize a low-frequency perturbation at (decoy_cy, decoy_cx) so that
    DINOv2 features of the decoy region match the target's features.

    Returns: edited frame (np.ndarray uint8).
    """
    H, W = frame_rgb.shape[:2]
    frame_t = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)

    # Target prototype
    target_proto = get_target_prototype(frame_t, mask_t, device)  # (1, 384)

    # Learnable low-frequency perturbation (in low resolution)
    delta_lf = torch.zeros(1, 3, lf_res, lf_res, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_lf], lr=lr)

    # Decoy region bounds
    dy0 = max(0, decoy_cy - decoy_h // 2)
    dy1 = min(H, decoy_cy + decoy_h // 2)
    dx0 = max(0, decoy_cx - decoy_w // 2)
    dx1 = min(W, decoy_cx + decoy_w // 2)
    rh, rw = dy1 - dy0, dx1 - dx0
    if rh < 20 or rw < 20:
        return frame_rgb.copy()

    # Soft alpha mask for decoy region (raised cosine)
    yy = torch.linspace(-1, 1, rh, device=device)
    xx = torch.linspace(-1, 1, rw, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    dist_sq = gy**2 + gx**2
    alpha_mask = torch.clamp(1.0 - dist_sq, 0, 1) ** 2 * alpha_max  # (rh, rw)
    alpha_mask = alpha_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, rh, rw)

    # Decoy patch coordinates in DINO's 37x37 patch space
    # Map pixel coords to patch indices
    py0 = int(dy0 / H * 37)
    py1 = int(dy1 / H * 37)
    px0 = int(dx0 / W * 37)
    px1 = int(dx1 / W * 37)
    py1 = max(py1, py0 + 1)
    px1 = max(px1, px0 + 1)

    best_loss = float("inf")
    best_delta = None

    for step in range(n_iter):
        optimizer.zero_grad()

        # Upsample low-freq delta to decoy region size
        delta_up = F.interpolate(delta_lf, size=(rh, rw), mode="bilinear",
                                  align_corners=False)  # (1, 3, rh, rw)

        # Apply perturbation
        edited = frame_t.clone()
        patch = edited[:, :, dy0:dy1, dx0:dx1]
        edited[:, :, dy0:dy1, dx0:dx1] = patch + delta_up * alpha_mask
        edited = edited.clamp(0, 1)

        # Extract DINOv2 features of decoy region
        feats = dino_patch_features(edited, device, grad=True)  # (1, 1369, 384) WITH gradients
        feats_2d = feats[0].view(37, 37, -1)  # (37, 37, 384)
        decoy_feats = feats_2d[py0:py1, px0:px1].reshape(-1, 384)  # (K, 384)
        decoy_proto = decoy_feats.mean(dim=0, keepdim=True)  # (1, 384)

        # Feature alignment loss: maximize cosine similarity
        cos_sim = F.cosine_similarity(decoy_proto, target_proto, dim=-1)
        feat_loss = 1.0 - cos_sim.mean()

        # High-frequency penalty (encourage low-freq perturbation)
        hf_penalty = (delta_lf[:, :, 1:, :] - delta_lf[:, :, :-1, :]).abs().mean() + \
                      (delta_lf[:, :, :, 1:] - delta_lf[:, :, :, :-1]).abs().mean()

        # Total loss
        loss = feat_loss + 0.5 * hf_penalty

        loss.backward()
        optimizer.step()

        # Clamp delta to prevent artifacts
        with torch.no_grad():
            delta_lf.clamp_(-0.3, 0.3)

        if step % 50 == 0 or step == n_iter - 1:
            print(f"      [opt {step:3d}] cos_sim={cos_sim.item():.4f}  "
                  f"hf={hf_penalty.item():.4f}  loss={loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta_lf.detach().clone()

    # Apply best perturbation
    with torch.no_grad():
        delta_up = F.interpolate(best_delta, size=(rh, rw), mode="bilinear",
                                  align_corners=False)
        result = frame_t.clone()
        patch = result[:, :, dy0:dy1, dx0:dx1]
        result[:, :, dy0:dy1, dx0:dx1] = (patch + delta_up * alpha_mask).clamp(0, 1)
        out = (result[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    return out, best_delta.cpu(), (dy0, dy1, dx0, dx1), alpha_mask.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog")
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--brs_alpha", type=float, default=0.80)
    p.add_argument("--decoy_alpha", type=float, default=0.5)
    p.add_argument("--n_opt_iter", type=int, default=200)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="v3")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "codecoy" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CoDecoy v3: DINOv2 Feature-Space Optimization ===")
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

        decoy_loc = find_decoy_region(masks[0])
        if decoy_loc is None:
            print(f"  [skip] no decoy location")
            continue
        print(f"  Decoy: ({decoy_loc[0]}, {decoy_loc[1]})")

        t0 = time.time()

        # Optimize decoy on first frame only, reuse delta for all frames
        print(f"  Optimizing DINOv2 decoy (first frame, {args.n_opt_iter} steps)...")
        decoy_frame0, best_delta, bounds, alpha_m = optimize_decoy(
            frames[0], masks[0], decoy_loc[0], decoy_loc[1],
            device, n_iter=args.n_opt_iter, alpha_max=args.decoy_alpha)

        dy0, dy1, dx0, dx1 = bounds
        rh, rw = dy1 - dy0, dx1 - dx0

        # Apply optimized delta to all frames (fast, no re-optimization)
        print(f"  Applying optimized delta to all {len(frames)} frames...")
        edited_frames = []
        delta_up = F.interpolate(best_delta.to(device), size=(rh, rw),
                                  mode="bilinear", align_corners=False)
        alpha_gpu = alpha_m.to(device)

        for i, (f, m) in enumerate(zip(frames, masks)):
            with torch.no_grad():
                f_t = torch.from_numpy(f / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
                f_t[:, :, dy0:dy1, dx0:dx1] = (
                    f_t[:, :, dy0:dy1, dx0:dx1] + delta_up * alpha_gpu
                ).clamp(0, 1)
                ef = (f_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # Also apply BRS
            ef = apply_old_boundary_suppression(ef, m, ring_width=24, blend_alpha=args.brs_alpha)
            edited_frames.append(ef)

        # Quality
        ssims = [frame_quality(o, e)[0] for o, e in zip(frames[:10], edited_frames[:10])]
        mean_ssim = float(np.mean(ssims))

        # BRS-only baseline for comparison
        brs_frames = [apply_old_boundary_suppression(f, m, ring_width=24, blend_alpha=args.brs_alpha)
                      for f, m in zip(frames, masks)]
        codec_brs = codec_round_trip(brs_frames, FFMPEG_PATH, args.crf)
        codec_combo = codec_round_trip(edited_frames, FFMPEG_PATH, args.crf)
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)

        if codec_clean is None or codec_brs is None or codec_combo is None:
            print(f"  [skip] codec failed")
            continue

        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)
        _, jf_brs, _, _ = run_tracking(codec_brs, masks, predictor, device, args.prompt)
        _, jf_combo, _, _ = run_tracking(codec_combo, masks, predictor, device, args.prompt)

        djf_brs = (jf_clean - jf_brs) * 100
        djf_combo = (jf_clean - jf_combo) * 100
        combo_gain = djf_combo - djf_brs
        elapsed = time.time() - t0

        print(f"\n  === {vid} RESULTS ===")
        print(f"  Clean JF:     {jf_clean:.4f}")
        print(f"  BRS only:     ΔJF={djf_brs:+.1f}pp")
        print(f"  BRS+DINOdecoy: ΔJF={djf_combo:+.1f}pp  SSIM={mean_ssim:.4f}")
        print(f"  Decoy gain:   {combo_gain:+.1f}pp")
        print(f"  Elapsed:      {elapsed:.0f}s")

        result = {
            "video": vid,
            "jf_clean": float(jf_clean),
            "jf_brs": float(jf_brs),
            "jf_combo": float(jf_combo),
            "djf_brs_pp": float(djf_brs),
            "djf_combo_pp": float(djf_combo),
            "combo_gain_pp": float(combo_gain),
            "ssim": mean_ssim,
            "elapsed_s": elapsed,
        }
        all_results.append(result)
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    if all_results:
        gains = [r["combo_gain_pp"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={len(all_results)})")
        print(f"  Mean BRS ΔJF: {np.mean([r['djf_brs_pp'] for r in all_results]):+.1f}pp")
        print(f"  Mean Combo ΔJF: {np.mean([r['djf_combo_pp'] for r in all_results]):+.1f}pp")
        print(f"  Mean Decoy Gain: {np.mean(gains):+.1f}pp")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)


if __name__ == "__main__":
    main()
