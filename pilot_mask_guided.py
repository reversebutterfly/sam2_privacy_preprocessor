"""
Pilot: Mask-Guided Semantic Obfuscation for SAM2 Video Tracking

Publisher-side preprocessing that uses per-frame GT masks (oracle tier)
to apply persistent, low-frequency semantic edits that corrupt SAM2's
memory bank across all frames.

Three edit types:
  A (idea1): Alpha-matte boundary suppression — blend 8-24px boundary ring
             toward local background, weakening the true object contour.
  B (idea2): Exterior echo contour — add a faint smooth shadow/highlight ring
             just outside the mask, creating a competing false boundary.
  C (combo):  A + B together (primary hypothesis).

All edits are:
  - Low-frequency (broad, smooth — survives H.264 DCT)
  - Persistent (applied to every frame using GT per-frame mask)
  - Object-local (only affect mask region ± halo)
  - No ML model required — pure numpy/cv2

Usage:
  python pilot_mask_guided.py \\
      --edit_type combo \\
      --videos dog-agility,elephant,flamingo \\
      --ring_width 16 --blend_alpha 0.6 \\
      --halo_offset 8 --halo_width 12 --halo_strength 0.4 \\
      --max_frames 50 --crf 23 \\
      --tag pilot_v1 --save_dir results_v100/mask_guided

Outputs:
  results_v100/mask_guided/<tag>/results.json   (per-video + aggregate)
  results_v100/mask_guided/<tag>/summary.csv
  logs_v100/mask_guided_<tag>.log
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264, encode_decode_hevc
from src.brs_utils import multiband_background_proxy, sdf_shell_weights
from src.metrics import jf_score, mean_jf


# ── Edit Functions ─────────────────────────────────────────────────────────────

def _background_proxy(frame_rgb: np.ndarray, mask: np.ndarray,
                      dilation_px: int = 24,
                      low_sigma: Optional[float] = None,
                      band_sigma_small: Optional[float] = None,
                      band_sigma_large: Optional[float] = None,
                      mid_gain: float = 0.25,
                      guard_px: int = 2) -> np.ndarray:
    """
    Multi-band normalized-convolution background proxy.

    Replaces the old flat-color fill with:
      1. a low-frequency normalized-convolution RGB field, and
      2. a transported mid-band residual for local texture/color variation.
    """
    return multiband_background_proxy(
        frame_rgb,
        mask,
        dilation_px=dilation_px,
        low_sigma=low_sigma,
        band_sigma_small=band_sigma_small,
        band_sigma_large=band_sigma_large,
        mid_gain=mid_gain,
        guard_px=guard_px,
    )


def apply_boundary_suppression(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    ring_width: int = 16,
    blend_alpha: float = 0.6,
    outer_ring_width: Optional[int] = None,
    outer_alpha: Optional[float] = None,
    proxy_low_sigma: Optional[float] = None,
    proxy_band_small_sigma: Optional[float] = None,
    proxy_band_large_sigma: Optional[float] = None,
    proxy_mid_gain: float = 0.25,
    proxy_guard_px: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Idea 1: Alpha-matte boundary suppression.

    Blend an SDF shell around the mask boundary toward a multi-band
    normalized-convolution background proxy, reducing figure-ground
    separability while avoiding the old flat-color blob artifact.

    ring_width:       inside-shell width in pixels
    blend_alpha:      peak inside-shell blend strength
    outer_ring_width: outside-shell width; defaults to a narrower shell
    outer_alpha:      outside-shell peak blend; defaults to a weaker shell
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    outer_ring_width = (
        max(2, int(round(ring_width * 0.55)))
        if outer_ring_width is None
        else int(outer_ring_width)
    )
    outer_alpha = (
        float(blend_alpha * 0.65)
        if outer_alpha is None
        else float(outer_alpha)
    )

    ring_weight, boundary_ring, _, _ = sdf_shell_weights(
        mask,
        inner_width_px=ring_width,
        outer_width_px=outer_ring_width,
        inner_alpha=blend_alpha,
        outer_alpha=outer_alpha,
    )
    if boundary_ring.sum() == 0:
        return frame_rgb.copy()

    bg_proxy = _background_proxy(
        frame_rgb,
        mask,
        dilation_px=max(ring_width, outer_ring_width) * 2,
        low_sigma=proxy_low_sigma,
        band_sigma_small=proxy_band_small_sigma,
        band_sigma_large=proxy_band_large_sigma,
        mid_gain=proxy_mid_gain,
        guard_px=proxy_guard_px,
    )

    f = frame_rgb.astype(np.float32)
    w = ring_weight[:, :, None]
    edited = f * (1 - w) + bg_proxy * w
    return np.clip(edited, 0, 255).astype(np.uint8)


def apply_echo_contour(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    halo_offset: int = 8,
    halo_width: int = 12,
    halo_strength: float = 0.4,
    **kwargs,
) -> np.ndarray:
    """
    Idea 2: Exterior echo contour.

    Add a smooth shadow/highlight ring just outside the mask boundary
    to create a competing false contour visible to SAM2 patch tokens.

    halo_offset:   pixels outside the mask where the halo starts
    halo_width:    thickness of the halo ring in pixels
    halo_strength: additive luma shift as fraction of Y range [0,1].
                   Peak delta-Y = 255 * halo_strength.
                   Positive = brighten, negative = darken. Default 0.4 → peak +102Y.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    # Build the halo zone: between dilate(offset) and dilate(offset + width)
    k_inner = np.ones((halo_offset * 2 + 1,) * 2, np.uint8)
    k_outer = np.ones(((halo_offset + halo_width) * 2 + 1,) * 2, np.uint8)
    inner_dil = cv2.dilate(mask, k_inner)
    outer_dil = cv2.dilate(mask, k_outer)
    halo_zone = ((outer_dil > 0) & (inner_dil == 0)).astype(np.float32)

    if halo_zone.sum() == 0:
        return frame_rgb.copy()

    # Smooth halo (low-frequency)
    # Build an unsigned weight map in [0, 1], then apply the signed strength once.
    # This avoids the clip-with-negative-upper-bound bug when halo_strength < 0.
    halo_weight = np.clip(cv2.GaussianBlur(halo_zone, (0, 0), halo_width / 2.0), 0.0, 1.0)

    # Additive luma shift: peak delta-Y = 255 * halo_strength
    #   positive halo_strength → brightens the ring (light halo)
    #   negative halo_strength → darkens the ring (shadow halo)
    frame_ycbcr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    frame_ycbcr[:, :, 0] = np.clip(
        frame_ycbcr[:, :, 0] + halo_weight * (255.0 * halo_strength),
        0, 255
    )
    edited_ycbcr = np.clip(frame_ycbcr, 0, 255).astype(np.uint8)
    edited_rgb = cv2.cvtColor(edited_ycbcr, cv2.COLOR_YCrCb2RGB)

    return edited_rgb


def apply_combo(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    ring_width: int = 16,
    blend_alpha: float = 0.6,
    halo_offset: int = 8,
    halo_width: int = 12,
    halo_strength: float = 0.4,
    **kwargs,
) -> np.ndarray:
    """Idea 1 + Idea 2 combined."""
    edited = apply_boundary_suppression(frame_rgb, mask, ring_width, blend_alpha)
    edited = apply_echo_contour(edited, mask, halo_offset, halo_width, halo_strength)
    return edited


def apply_global_blur(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    ring_width: int = 16,
    blend_alpha: float = 0.6,
    **kwargs,
) -> np.ndarray:
    """
    Global baseline: apply Gaussian blur to the entire frame (no mask guidance).
    Uses `ring_width` as blur sigma and `blend_alpha` as blend strength.
    This tests whether mask guidance is necessary or any low-freq global edit works.
    """
    sigma = max(ring_width / 2.0, 2.0)
    blurred = cv2.GaussianBlur(frame_rgb.astype(np.float32), (0, 0), sigma)
    result = frame_rgb.astype(np.float32) * (1 - blend_alpha) + blurred * blend_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_global_brightness(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    blend_alpha: float = 0.6,
    **kwargs,
) -> np.ndarray:
    """
    Global baseline: multiplicative brightness boost (scale > 1) across the entire frame.
    Uses blend_alpha to set scale: scale = 1.0 + blend_alpha * 0.5
    """
    scale = 1.0 + blend_alpha * 0.5
    result = np.clip(frame_rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return result


def apply_boundary_blur(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    ring_width: int = 24,
    blend_alpha: float = 0.8,
    **kwargs,
) -> np.ndarray:
    """
    Fair baseline: Gaussian blur restricted to the boundary ring.
    Same spatial support as combo_strong, different edit operation.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()
    ring_weight, ring, _, _ = sdf_shell_weights(
        mask,
        inner_width_px=ring_width,
        outer_width_px=max(2, int(round(ring_width * 0.55))),
        inner_alpha=blend_alpha,
        outer_alpha=blend_alpha * 0.65,
    )
    if ring.sum() == 0:
        return frame_rgb.copy()
    sigma   = max(ring_width / 2.0, 3.0)
    blurred = cv2.GaussianBlur(frame_rgb.astype(np.float32), (0, 0), sigma)
    ring_w  = ring_weight[:, :, None]
    result = frame_rgb.astype(np.float32) * (1 - ring_w) + blurred * ring_w
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_interior_feather(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    ring_width: int = 24,
    blend_alpha: float = 0.8,
    **kwargs,
) -> np.ndarray:
    """
    Fair baseline: Gaussian blur applied to mask interior only.
    Tests whether interior vs boundary targeting matters.
    ring_width used as blur sigma for comparable distortion.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()
    sigma    = max(ring_width / 2.0, 3.0)
    blurred  = cv2.GaussianBlur(frame_rgb.astype(np.float32), (0, 0), sigma)
    interior_w = np.clip(
        cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma / 2.0) * blend_alpha,
        0.0, blend_alpha,
    )[:, :, None]
    result = frame_rgb.astype(np.float32) * (1 - interior_w) + blurred * interior_w
    return np.clip(result, 0, 255).astype(np.uint8)


EDIT_FNS = {
    "idea1":            apply_boundary_suppression,
    "idea2":            apply_echo_contour,
    "combo":            apply_combo,
    "global_blur":      apply_global_blur,
    "global_bright":    apply_global_brightness,
    "boundary_blur":    apply_boundary_blur,
    "interior_feather": apply_interior_feather,
}


# ── Imperfect-mask perturbation ────────────────────────────────────────────────

def perturb_mask(mask: np.ndarray, mode: str, px: int = 8) -> np.ndarray:
    """
    Simulate imperfect GT masks for robustness testing.
    mode: 'dilate' | 'erode' | 'noise' | 'none'
    px:   perturbation amount in pixels (dilation/erosion kernel half-size,
          or percentage of mask pixels flipped for noise mode)
    """
    if mode == "none" or px == 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    if mode == "dilate":
        return cv2.dilate(mask, kernel)
    elif mode == "erode":
        return cv2.erode(mask, kernel)
    elif mode == "noise":
        noisy = mask.copy()
        n_flip = max(1, int(mask.sum() * (px / 100.0)))
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            idx = np.random.choice(len(ys), min(n_flip, len(ys)), replace=False)
            noisy[ys[idx], xs[idx]] = 0
        # Also flip some background pixels ON
        ys0, xs0 = np.where(mask == 0)
        if len(ys0) > 0:
            idx0 = np.random.choice(len(ys0), min(n_flip, len(ys0)), replace=False)
            noisy[ys0[idx0], xs0[idx0]] = 1
        return noisy
    return mask


# ── Scale-normalized ring width ────────────────────────────────────────────────

def scale_norm_ring_width(mask: np.ndarray, rho: float,
                           min_w: int = 6, max_w: int = 32) -> int:
    """
    Compute a per-frame scale-normalised ring width.

    ring_width = clip(round(rho * sqrt(mask_area / pi)), min_w, max_w)

    rho=0.10 means the ring covers ~10% of the object's equivalent radius.
    Typical useful range: rho in {0.06, 0.10, 0.14}.
    """
    area = int(mask.sum())
    if area == 0:
        return min_w
    r_eq = float(area / np.pi) ** 0.5
    w = int(np.clip(round(rho * r_eq), min_w, max_w))
    return w


# ── Apply edit to all frames ───────────────────────────────────────────────────

def apply_edit_to_video(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    edit_type: str,
    params: dict,
    ring_width_mode: str = "fixed",
    ring_width_rho: float = 0.10,
) -> List[np.ndarray]:
    """
    Apply the chosen edit to every frame using its corresponding GT mask.

    ring_width_mode:
        "fixed"      — use params["ring_width"] as-is (original behaviour)
        "scale_norm" — compute per-frame ring_width from mask area and rho;
                       overrides params["ring_width"] for each frame
    ring_width_rho:
        Scale factor used when ring_width_mode="scale_norm" (default 0.10).
    """
    fn = EDIT_FNS[edit_type]
    edited = []
    for frame, mask in zip(frames, masks):
        if ring_width_mode == "scale_norm":
            w = scale_norm_ring_width(mask, rho=ring_width_rho)
            frame_params = dict(params, ring_width=w)
        else:
            frame_params = params
        edited.append(fn(frame, mask, **frame_params))
    return edited


# ── SAM2 tracking helper ───────────────────────────────────────────────────────

def build_predictor(checkpoint: str, config: str, device: torch.device):
    from sam2.build_sam import build_sam2_video_predictor
    pred = build_sam2_video_predictor(config, checkpoint, device=device)
    pred.eval()
    return pred


def run_tracking(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    predictor,
    device: torch.device,
    prompt: str = "point",
) -> Tuple[List[np.ndarray], float, float, float]:
    """
    Run VideoPredictor with first-frame prompt.
    prompt: "point" (centroid) or "mask" (full GT mask).
    Returns (pred_masks, mean_jf, mean_j, mean_f).
    """
    H, W = frames[0].shape[:2]
    gt_bool = [m.astype(bool) for m in masks]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fr in enumerate(frames):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)

            first_mask = masks[0].astype(bool)
            ys, xs = np.where(first_mask)

            if prompt == "mask" and len(ys) > 0:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=first_mask)
            elif len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=np.array([[cx, cy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=first_mask)

            pred = [None] * len(frames)
            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                if 1 in obj_ids:
                    idx = list(obj_ids).index(1)
                    pred[fi] = (logits[idx, 0] > 0.0).cpu().numpy()
                else:
                    pred[fi] = np.zeros((H, W), dtype=bool)

    for i in range(len(pred)):
        if pred[i] is None:
            pred[i] = np.zeros((H, W), dtype=bool)

    mjf, mj, mf = mean_jf(pred, gt_bool)
    return pred, mjf, mj, mf


# ── Quality metrics ───────────────────────────────────────────────────────────

def frame_quality(orig: np.ndarray, edit: np.ndarray) -> Tuple[float, float]:
    """Return (SSIM, PSNR) between original and edited frame."""
    from skimage.metrics import structural_similarity as ssim_fn
    s = ssim_fn(orig, edit, channel_axis=2, data_range=255)
    mse = np.mean((orig.astype(float) - edit.astype(float)) ** 2)
    psnr = float('inf') if mse < 1e-10 else 10 * np.log10(255 ** 2 / mse)
    return float(s), float(psnr)


def codec_round_trip(
    frames: List[np.ndarray],
    ffmpeg_path: str,
    crf: int,
    codec: str = "h264",
) -> Optional[List[np.ndarray]]:
    try:
        if codec == "hevc":
            return encode_decode_hevc(frames, ffmpeg_path=ffmpeg_path, crf=crf)
        return encode_decode_h264(frames, ffmpeg_path=ffmpeg_path, crf=crf)
    except Exception as e:
        print(f"  [codec] error: {e}")
        return None


# ── Main pilot ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edit_type", default="combo",
                   choices=["idea1", "idea2", "combo", "global_blur", "global_bright"],
                   help="Which edit to apply: idea1, idea2, or combo")
    p.add_argument("--videos", default="",
                   help="Comma-separated video names (empty = DAVIS_MINI_VAL)")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="mask", choices=["point", "mask"],
                   help="SAM2 first-frame prompt type (mask = oracle GT, point = centroid)")
    p.add_argument("--min_jf_clean", type=float, default=0.3,
                   help="Skip videos where clean JF < this threshold")
    # Idea 1 params
    p.add_argument("--ring_width",   type=int,   default=16)
    p.add_argument("--blend_alpha",  type=float, default=0.6)
    p.add_argument("--outer_ring_width", type=int, default=0,
                   help="Outside-shell width in pixels; <=0 uses auto ratio from ring_width.")
    p.add_argument("--outer_alpha", type=float, default=-1.0,
                   help="Outside-shell peak blend alpha; <0 uses auto ratio from blend_alpha.")
    p.add_argument("--proxy_mid_gain", type=float, default=0.25,
                   help="Mid-band texture gain for the normalized-convolution proxy.")
    p.add_argument("--proxy_guard_px", type=int, default=2,
                   help="Guard band outside the mask excluded from proxy sampling.")
    # Idea 2 params
    p.add_argument("--halo_offset",   type=int,   default=8)
    p.add_argument("--halo_width",    type=int,   default=12)
    p.add_argument("--halo_strength", type=float, default=0.4)
    # Infrastructure
    p.add_argument("--davis_root",    default=DAVIS_ROOT)
    p.add_argument("--checkpoint",    default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",   default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",   default=FFMPEG_PATH)
    p.add_argument("--save_dir",      default="results_v100/mask_guided")
    p.add_argument("--tag",           default="pilot_v1")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--sanity",        action="store_true",
                   help="Sanity mode: stop after first video, check pipeline")
    # Imperfect mask robustness
    p.add_argument("--mask_perturb", default="none",
                   choices=["none", "dilate", "erode", "noise"],
                   help="Perturb GT masks before editing to test robustness. "
                        "none=oracle, dilate/erode=boundary error, noise=random flip.")
    p.add_argument("--mask_perturb_px", type=int, default=8,
                   help="Perturbation amount: dilation/erosion kernel half-size (px), "
                        "or percentage of mask pixels flipped for noise mode.")
    # Scale-normalised ring width
    p.add_argument("--ring_width_mode", default="fixed",
                   choices=["fixed", "scale_norm"],
                   help="fixed: use --ring_width as-is.  "
                        "scale_norm: per-frame ring_width = clip(round(rho*sqrt(area/pi)), 6, 32).")
    p.add_argument("--ring_width_rho", type=float, default=0.10,
                   help="Scale factor for scale_norm mode (typical: 0.06/0.10/0.14).")
    return p.parse_args()


def edit_params(args) -> dict:
    """Collect edit parameters from args into a dict for EDIT_FNS."""
    idea1_params = {
        "ring_width": args.ring_width,
        "blend_alpha": args.blend_alpha,
        "outer_ring_width": None if args.outer_ring_width <= 0 else args.outer_ring_width,
        "outer_alpha": None if args.outer_alpha < 0 else args.outer_alpha,
        "proxy_mid_gain": args.proxy_mid_gain,
        "proxy_guard_px": args.proxy_guard_px,
    }
    if args.edit_type == "idea1":
        return idea1_params
    elif args.edit_type == "idea2":
        return {"halo_offset": args.halo_offset, "halo_width": args.halo_width,
                "halo_strength": args.halo_strength}
    elif args.edit_type in ("global_blur", "global_bright"):
        return {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}
    else:  # combo
        return {
            **idea1_params,
            "halo_offset": args.halo_offset, "halo_width": args.halo_width,
            "halo_strength": args.halo_strength,
        }


def main():
    args = parse_args()
    device = torch.device(args.device)
    _raw = [v.strip() for v in args.videos.split(",") if v.strip()]
    if _raw == ["all"]:
        from pathlib import Path as _P
        _img = _P(args.davis_root) / "JPEGImages" / "480p"
        videos = sorted(d.name for d in _img.iterdir() if d.is_dir()) if _img.exists() else DAVIS_MINI_VAL
    else:
        videos = _raw or DAVIS_MINI_VAL
    params = edit_params(args)

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pilot.log"

    print(f"[pilot] edit_type={args.edit_type}, prompt={args.prompt}")
    print(f"[pilot] params={params}")
    print(f"[pilot] videos={videos}")
    print(f"[pilot] output -> {out_dir}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    for vid in videos:
        print(f"\n=== {vid} ===")
        frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue

        # ── Clean baseline ──
        _, jf_clean, j_clean, f_clean = run_tracking(frames, masks, predictor, device, args.prompt)
        print(f"  clean: JF={jf_clean:.4f}  J={j_clean:.4f}  F={f_clean:.4f}")

        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue

        # ── Codec-clean baseline ──
        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_frames:
            _, jf_codec_clean, j_cc, f_cc = run_tracking(codec_frames, masks, predictor, device, args.prompt)
            print(f"  codec_clean: JF={jf_codec_clean:.4f}")
        else:
            jf_codec_clean = float("nan")

        # ── Apply edit (with optional mask perturbation for robustness testing) ──
        edit_masks = masks
        if args.mask_perturb != "none":
            edit_masks = [perturb_mask(m, args.mask_perturb, args.mask_perturb_px)
                          for m in masks]
        edited_frames = apply_edit_to_video(
            frames, edit_masks, args.edit_type, params,
            ring_width_mode=args.ring_width_mode,
            ring_width_rho=args.ring_width_rho,
        )

        # Quality check (sampled on first 5 frames)
        ssim_vals, psnr_vals = [], []
        for fo, fe in zip(frames[:5], edited_frames[:5]):
            s, p = frame_quality(fo, fe)
            ssim_vals.append(s)
            psnr_vals.append(p)
        mean_ssim = float(np.mean(ssim_vals))
        mean_psnr = float(np.nanmean([p for p in psnr_vals if p != float("inf")]))
        print(f"  quality: SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.1f}dB")

        # ── Pre-codec adversarial ──
        _, jf_adv, j_adv, f_adv = run_tracking(edited_frames, masks, predictor, device, args.prompt)
        delta_adv = jf_clean - jf_adv
        print(f"  adv (pre-codec): JF={jf_adv:.4f}  ΔJF={delta_adv:+.4f}")

        # ── Post-codec adversarial ──
        codec_edited = codec_round_trip(edited_frames, args.ffmpeg_path, args.crf)
        if codec_edited:
            _, jf_codec_adv, j_ca, f_ca = run_tracking(codec_edited, masks, predictor, device, args.prompt)
            delta_codec = jf_codec_clean - jf_codec_adv
            print(f"  adv (post-codec CRF{args.crf}): JF={jf_codec_adv:.4f}  ΔJF={delta_codec:+.4f}")
        else:
            jf_codec_adv = float("nan")
            delta_codec = float("nan")

        row = {
            "video": vid, "n_frames": len(frames),
            "jf_clean": jf_clean, "j_clean": j_clean, "f_clean": f_clean,
            "jf_codec_clean": jf_codec_clean,
            "jf_adv": jf_adv, "j_adv": j_adv, "f_adv": f_adv,
            "jf_codec_adv": jf_codec_adv,
            "delta_jf_adv": delta_adv,
            "delta_jf_codec": delta_codec,
            "mean_ssim": mean_ssim,
            "mean_psnr": mean_psnr,
            "mask_perturb": args.mask_perturb,
            "mask_perturb_px": args.mask_perturb_px,
            "ring_width_mode": args.ring_width_mode,
            "ring_width_rho": args.ring_width_rho,
        }
        all_results.append(row)

        with open(log_path, "a") as lf:
            lf.write(f"{vid} | adv ΔJF={delta_adv:+.4f} | codec ΔJF={delta_codec:+.4f} | SSIM={mean_ssim:.4f}\n")

        # Intermediate save
        _save(out_dir, args, all_results)

        if args.sanity:
            print("\n[SANITY MODE] Stopping after first video.")
            break

    # Final save + summary
    _save(out_dir, args, all_results)
    _print_summary(all_results)


def _save(out_dir, args, results):
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    # CSV
    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w") as f:
        f.write("video,jf_clean,jf_adv,jf_codec_adv,delta_jf_adv,delta_jf_codec,mean_ssim\n")
        for r in results:
            f.write(f"{r['video']},{r['jf_clean']:.6f},{r['jf_adv']:.6f},"
                    f"{r['jf_codec_adv']:.6f},{r['delta_jf_adv']:.6f},"
                    f"{r['delta_jf_codec']:.6f},{r['mean_ssim']:.6f}\n")


def _print_summary(results):
    valid = [r for r in results if not (
        isinstance(r["delta_jf_codec"], float) and
        (r["delta_jf_codec"] != r["delta_jf_codec"])  # NaN check
    )]
    if not valid:
        print("\n[SUMMARY] No valid results.")
        return

    mean_delta = np.mean([r["delta_jf_codec"] for r in valid])
    median_delta = np.median([r["delta_jf_codec"] for r in valid])
    above5 = sum(1 for r in valid if r["delta_jf_codec"] >= 0.05)
    above8 = sum(1 for r in valid if r["delta_jf_codec"] >= 0.08)
    mean_ssim = np.mean([r["mean_ssim"] for r in valid])

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(valid)} videos)")
    print(f"  mean  post-codec ΔJF = {mean_delta:+.4f} ({mean_delta*100:+.1f}pp)")
    print(f"  median post-codec ΔJF = {median_delta:+.4f}")
    print(f"  videos ≥ 5pp:  {above5}/{len(valid)}")
    print(f"  videos ≥ 8pp:  {above8}/{len(valid)}")
    print(f"  mean SSIM:     {mean_ssim:.4f}")
    print(f"{'='*60}")

    if mean_delta >= 0.08:
        print("  *** STRONG POSITIVE SIGNAL (≥8pp) — expand to auto-mask + memory ablation ***")
    elif mean_delta >= 0.05:
        print("  ** POSITIVE SIGNAL (≥5pp) — optimize params, continue ***")
    elif mean_delta >= 0.02:
        print("  * WEAK SIGNAL (2-5pp) — try stronger params (higher blend_alpha, wider ring)")
    else:
        print("  -- NEGATIVE — try: ring_width↑, blend_alpha↑, halo_strength↑, or combo with larger halo")


if __name__ == "__main__":
    main()
