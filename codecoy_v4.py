"""
codecoy_v4.py — CoDecoy v4: Fixed DINOv2 Feature Decoy

Fixes from v3:
  1. Decoy size: 80→200px (covers ~40 DINO patches vs target's ~150)
  2. Location: "near target but not overlapping" instead of far corner
  3. Loss: patch-level alpha-weighted prototype (not rect average)
  4. Codec proxy: GaussianBlur + quant noise in optimization loop
  5. Early return fix

Usage:
  python codecoy_v4.py --videos bear,blackswan,dog --device cuda --tag v4
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

# ── DINOv2 ──────────────────────────────────────────────────────────────────

_dino = None

def get_dino(device):
    global _dino
    if _dino is None:
        import importlib
        cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.exists(cache_dir):
            sys.path.insert(0, cache_dir)
            hub = importlib.import_module("hubconf")
            _dino = hub.dinov2_vits14(pretrained=True).to(device).eval()
            sys.path.pop(0)
        else:
            _dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    verbose=False).to(device).eval()
        for p in _dino.parameters():
            p.requires_grad_(False)
    return _dino

DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINO_STD = torch.tensor([0.229, 0.224, 0.225])

def dino_features(img_t, device, grad=False):
    """img_t: (1,3,H,W) [0,1] → patch tokens (1, 37*37, 384)."""
    dino = get_dino(device)
    img_r = F.interpolate(img_t, size=(518, 518), mode="bilinear", align_corners=False)
    mean = DINO_MEAN.to(device).view(1, 3, 1, 1)
    std = DINO_STD.to(device).view(1, 3, 1, 1)
    inp = (img_r - mean) / std
    if grad:
        return dino.forward_features(inp)["x_norm_patchtokens"]
    with torch.no_grad():
        return dino.forward_features(inp)["x_norm_patchtokens"]

def codec_proxy(x):
    """Differentiable H.264 proxy: blur + quantization noise."""
    blurred = F.avg_pool2d(x, 2, 2)
    blurred = F.interpolate(blurred, size=x.shape[2:], mode="bilinear", align_corners=False)
    noise = torch.randn_like(x) * 0.005
    return (0.7 * x + 0.3 * blurred + noise).clamp(0, 1)


# ── Location ─────────────────────────────────────────────────────────────────

def find_near_decoy(mask, margin=30, decoy_size=200):
    """Find location NEAR the target but NOT overlapping — much better than far corner."""
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    # Dilate mask to create exclusion zone
    excl_radius = decoy_size // 2 + margin
    kern = np.ones((excl_radius * 2 + 1,) * 2, np.uint8)
    excluded = cv2.dilate(mask.astype(np.uint8), kern)

    # Distance from mask boundary (want close but not overlapping)
    dist_from_mask = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)

    # Valid: not excluded, not too close to edges, within reasonable distance
    half = decoy_size // 2
    valid = (excluded == 0).astype(np.float32)
    valid[:half, :] = 0; valid[-half:, :] = 0
    valid[:, :half] = 0; valid[:, -half:] = 0

    # Score: prefer close to target (but not overlapping)
    # Ideal distance: margin + decoy_size//2 (just outside exclusion zone)
    ideal_dist = float(excl_radius)
    score = valid * np.exp(-((dist_from_mask - ideal_dist) / 50.0) ** 2)

    if score.max() < 1e-6:
        return None

    idx = np.unravel_index(score.argmax(), score.shape)
    return int(idx[0]), int(idx[1])


# ── Optimization ─────────────────────────────────────────────────────────────

def optimize_decoy_v4(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    decoy_cy: int,
    decoy_cx: int,
    device: torch.device,
    decoy_size: int = 200,
    n_iter: int = 200,
    lr: float = 0.03,
    alpha_max: float = 0.6,
    lf_res: int = 48,
):
    """Optimize decoy in DINOv2 feature space with all v3 fixes."""
    H, W = frame_rgb.shape[:2]
    frame_t = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)

    # Target prototype (alpha-weighted by mask in DINO patch space)
    with torch.no_grad():
        target_feats = dino_features(frame_t, device)  # (1, 1369, 384)
        mask_37 = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0),
                                 size=(37, 37), mode="nearest")[0, 0]  # (37, 37)
        mask_flat = mask_37.flatten()  # (1369,)
        if mask_flat.sum() < 1:
            mask_flat = torch.ones_like(mask_flat)
        # Weighted target prototype
        weights = mask_flat / mask_flat.sum()
        target_proto = (target_feats[0] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)  # (1, 384)
        target_proto = F.normalize(target_proto, dim=-1)

    # Decoy bounds
    half = decoy_size // 2
    dy0, dy1 = max(0, decoy_cy - half), min(H, decoy_cy + half)
    dx0, dx1 = max(0, decoy_cx - half), min(W, decoy_cx + half)
    rh, rw = dy1 - dy0, dx1 - dx0
    if rh < 40 or rw < 40:
        return frame_rgb.copy(), None, None, None

    # Soft alpha (elliptical)
    yy = torch.linspace(-1, 1, rh, device=device)
    xx = torch.linspace(-1, 1, rw, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    alpha = (torch.clamp(1.0 - (gy**2 + gx**2), 0, 1) ** 1.5 * alpha_max).unsqueeze(0).unsqueeze(0)

    # DINO patch-space decoy mask (for weighted loss)
    py0, py1 = int(dy0 / H * 37), max(int(dy1 / H * 37), int(dy0 / H * 37) + 1)
    px0, px1 = int(dx0 / W * 37), max(int(dx1 / W * 37), int(dx0 / W * 37) + 1)

    # Build patch-level weights from alpha (downsample to 37x37)
    alpha_37 = F.interpolate(alpha, size=(37, 37), mode="bilinear", align_corners=False)[0, 0]
    # Zero outside decoy region
    patch_weights = torch.zeros(37, 37, device=device)
    patch_weights[py0:py1, px0:px1] = alpha_37[py0:py1, px0:px1]
    pw_flat = patch_weights.flatten()  # (1369,)
    pw_sum = pw_flat.sum().clamp(min=1e-6)

    # Learnable delta
    delta = torch.zeros(1, 3, lf_res, lf_res, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    best_loss, best_delta = float("inf"), None

    for step in range(n_iter):
        opt.zero_grad()

        delta_up = F.interpolate(delta, size=(rh, rw), mode="bilinear", align_corners=False)
        edited = frame_t.clone()
        edited[:, :, dy0:dy1, dx0:dx1] = (edited[:, :, dy0:dy1, dx0:dx1] + delta_up * alpha).clamp(0, 1)

        # Codec proxy (differentiable)
        edited_codec = codec_proxy(edited)

        # DINOv2 features with gradient
        feats = dino_features(edited_codec, device, grad=True)  # (1, 1369, 384)

        # Alpha-weighted decoy prototype
        decoy_proto = (feats[0] * pw_flat.unsqueeze(-1)).sum(dim=0, keepdim=True) / pw_sum
        decoy_proto = F.normalize(decoy_proto, dim=-1)

        # Cosine alignment loss
        cos_sim = F.cosine_similarity(decoy_proto, target_proto, dim=-1).mean()
        feat_loss = 1.0 - cos_sim

        # Smoothness (low-freq)
        tv = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean() + \
             (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()

        loss = feat_loss + 0.3 * tv
        loss.backward()
        opt.step()

        with torch.no_grad():
            delta.clamp_(-0.4, 0.4)

        if step % 50 == 0 or step == n_iter - 1:
            print(f"      [opt {step:3d}] cos={cos_sim.item():.4f} tv={tv.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()

    # Apply
    with torch.no_grad():
        d_up = F.interpolate(best_delta, size=(rh, rw), mode="bilinear", align_corners=False)
        result = frame_t.clone()
        result[:, :, dy0:dy1, dx0:dx1] = (result[:, :, dy0:dy1, dx0:dx1] + d_up * alpha).clamp(0, 1)
        out = (result[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return out, best_delta.cpu(), (dy0, dy1, dx0, dx1), alpha.cpu()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog")
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--brs_alpha", type=float, default=0.80)
    p.add_argument("--decoy_size", type=int, default=200)
    p.add_argument("--decoy_alpha", type=float, default=0.6)
    p.add_argument("--n_opt", type=int, default=200)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="v4")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    save_dir = Path(ROOT) / "results_v100" / "codecoy" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CoDecoy v4 (all fixes) ===")
    print(f"  decoy_size={args.decoy_size}, alpha={args.decoy_alpha}, n_opt={args.n_opt}")

    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    results = []

    for vid in videos:
        print(f"\n{'='*60}\n{vid}\n{'='*60}")
        try:
            frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] {e}"); continue
        if not frames: continue
        frames, masks = frames[:args.max_frames], masks[:args.max_frames]

        loc = find_near_decoy(masks[0], decoy_size=args.decoy_size)
        if loc is None:
            print(f"  [skip] no location"); continue
        print(f"  Decoy near target: ({loc[0]}, {loc[1]})")

        t0 = time.time()

        # Optimize on frame 0
        print(f"  Optimizing ({args.n_opt} steps)...")
        _, best_delta, bounds, alpha_m = optimize_decoy_v4(
            frames[0], masks[0], loc[0], loc[1], device,
            decoy_size=args.decoy_size, n_iter=args.n_opt, alpha_max=args.decoy_alpha)

        if best_delta is None:
            print(f"  [skip] optimization failed"); continue

        dy0, dy1, dx0, dx1 = bounds
        rh, rw = dy1 - dy0, dx1 - dx0
        d_up = F.interpolate(best_delta.to(device), size=(rh, rw),
                              mode="bilinear", align_corners=False)
        a_gpu = alpha_m.to(device)

        # Apply to all frames + BRS
        edited = []
        brs_only = []
        for f, m in zip(frames, masks):
            with torch.no_grad():
                ft = torch.from_numpy(f / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
                ft[:, :, dy0:dy1, dx0:dx1] = (ft[:, :, dy0:dy1, dx0:dx1] + d_up * a_gpu).clamp(0, 1)
                ef = (ft[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            ef = apply_old_boundary_suppression(ef, m, ring_width=24, blend_alpha=args.brs_alpha)
            edited.append(ef)
            brs_only.append(apply_old_boundary_suppression(f, m, ring_width=24, blend_alpha=args.brs_alpha))

        ssims = [frame_quality(o, e)[0] for o, e in zip(frames[:10], edited[:10])]

        cc = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        cb = codec_round_trip(brs_only, FFMPEG_PATH, args.crf)
        ce = codec_round_trip(edited, FFMPEG_PATH, args.crf)
        if not all([cc, cb, ce]):
            print(f"  [skip] codec failed"); continue

        _, jf_c, _, _ = run_tracking(cc, masks, predictor, device, args.prompt)
        _, jf_b, _, _ = run_tracking(cb, masks, predictor, device, args.prompt)
        _, jf_e, _, _ = run_tracking(ce, masks, predictor, device, args.prompt)

        djf_b = (jf_c - jf_b) * 100
        djf_e = (jf_c - jf_e) * 100
        gain = djf_e - djf_b
        elapsed = time.time() - t0

        print(f"\n  BRS only:    ΔJF={djf_b:+.1f}pp")
        print(f"  BRS+Decoy:   ΔJF={djf_e:+.1f}pp  SSIM={np.mean(ssims):.4f}")
        print(f"  Decoy gain:  {gain:+.1f}pp  ({elapsed:.0f}s)")

        results.append({
            "video": vid, "jf_clean": float(jf_c), "jf_brs": float(jf_b), "jf_combo": float(jf_e),
            "djf_brs": float(djf_b), "djf_combo": float(djf_e), "gain": float(gain),
            "ssim": float(np.mean(ssims)), "elapsed": elapsed,
        })
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)

    if results:
        print(f"\n{'='*60}\nAGGREGATE (n={len(results)})")
        print(f"  BRS:   {np.mean([r['djf_brs'] for r in results]):+.1f}pp")
        print(f"  Combo: {np.mean([r['djf_combo'] for r in results]):+.1f}pp")
        print(f"  Gain:  {np.mean([r['gain'] for r in results]):+.1f}pp")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
