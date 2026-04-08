"""
Generate side-by-side visual comparison of BRS variants for paper figures.

Output per video:
  results_v100/figures/brs_compare/<video>_frame<N>.png
  — 4-column strip: Original | Old BRS (rw=24,α=0.93) | New BRS (rw=24,α=0.93) | New BRS (rw=18,α=0.72)

Old BRS: flat-mean proxy + square morphological kernel (archived implementation)
New BRS: multi-band normalized-conv proxy + SDF shell (current apply_boundary_suppression)

Usage:
  python gen_brs_comparison.py [--videos bear,elephant,...] [--frame_idx 5] [--out_dir ...]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DAVIS_ROOT
from src.dataset import load_single_video
from pilot_mask_guided import apply_boundary_suppression


# ── Old BRS (archived) ─────────────────────────────────────────────────────────

def _old_background_proxy(frame_rgb: np.ndarray, mask: np.ndarray, dilation_px: int = 24) -> np.ndarray:
    """Original flat-mean proxy (pre-improvement)."""
    kernel = np.ones((dilation_px * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    bg_mask = (dilated > 0) & (mask == 0)

    proxy = frame_rgb.astype(np.float32).copy()
    if bg_mask.sum() < 10:
        bg_color = frame_rgb.mean(axis=(0, 1))
        proxy[:] = bg_color
    else:
        bg_mean = frame_rgb[bg_mask].mean(axis=0)
        proxy[mask > 0] = bg_mean

    sigma = max(dilation_px // 2, 5)
    proxy = cv2.GaussianBlur(proxy, (0, 0), sigma)
    return proxy


def apply_old_brs(frame_rgb: np.ndarray, mask: np.ndarray,
                  ring_width: int = 24, blend_alpha: float = 0.93) -> np.ndarray:
    """Old BRS: square kernel ring + flat-mean proxy."""
    if mask.sum() == 0:
        return frame_rgb.copy()

    kernel = np.ones((ring_width * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded  = cv2.erode(mask, kernel)
    boundary_ring = ((dilated > 0) & (eroded == 0)).astype(np.uint8)

    if boundary_ring.sum() == 0:
        return frame_rgb.copy()

    bg_proxy = _old_background_proxy(frame_rgb, mask, dilation_px=ring_width * 2)

    ring_float = boundary_ring.astype(np.float32)
    ring_smooth = cv2.GaussianBlur(ring_float, (0, 0), ring_width / 2.0)
    ring_smooth = np.clip(ring_smooth * blend_alpha, 0.0, blend_alpha)

    f = frame_rgb.astype(np.float32)
    w = ring_smooth[:, :, None]
    edited = f * (1 - w) + bg_proxy * w
    return np.clip(edited, 0, 255).astype(np.uint8)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Fast per-channel mean SSIM approximation."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_ab = mu_a * mu_b
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    sig_a2 = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    ssim_map = num / (den + 1e-8)
    return float(ssim_map.mean())


# ── Annotation helpers ─────────────────────────────────────────────────────────

def add_label(img: np.ndarray, text: str, ssim: float | None = None) -> np.ndarray:
    """Add a label bar at the top of an image."""
    H, W = img.shape[:2]
    bar = np.zeros((36, W, 3), dtype=np.uint8)
    label = f"{text}  SSIM={ssim:.3f}" if ssim is not None else text
    cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def draw_mask_contour(img: np.ndarray, mask: np.ndarray,
                      color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Overlay mask contour on image."""
    out = img.copy()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

DEFAULT_VIDEOS = [
    "bear", "elephant", "dog", "horsejump-high", "scooter-black",
    "motocross-jump", "flamingo", "breakdance",
]


def process_video(video: str, frame_idx: int, out_dir: Path,
                  show_contour: bool = True) -> dict:
    frames, masks, _ = load_single_video(
        DAVIS_ROOT, video, max_frames=-1,
    )
    if not frames:
        print(f"  [skip] {video}: load failed")
        return {}

    # Pick a frame that has a non-empty mask
    fi = min(frame_idx, len(frames) - 1)
    while fi < len(frames) and masks[fi].sum() == 0:
        fi += 1
    if fi >= len(frames):
        print(f"  [skip] {video}: no valid mask")
        return {}

    frame = frames[fi]
    mask  = masks[fi].astype(np.uint8)

    # Generate variants
    orig     = frame.copy()
    old_high = apply_old_brs(frame, mask, ring_width=24, blend_alpha=0.93)
    new_high = apply_boundary_suppression(frame, mask, ring_width=24, blend_alpha=0.93,
                                          proxy_mid_gain=0.25)
    new_bal  = apply_boundary_suppression(frame, mask, ring_width=18, blend_alpha=0.72,
                                          outer_ring_width=10, outer_alpha=0.46,
                                          proxy_mid_gain=0.22)

    ssim_old_high = compute_ssim(orig, old_high)
    ssim_new_high = compute_ssim(orig, new_high)
    ssim_new_bal  = compute_ssim(orig, new_bal)

    if show_contour:
        orig_ann     = draw_mask_contour(orig,     mask)
        old_high_ann = draw_mask_contour(old_high, mask)
        new_high_ann = draw_mask_contour(new_high, mask)
        new_bal_ann  = draw_mask_contour(new_bal,  mask)
    else:
        orig_ann, old_high_ann, new_high_ann, new_bal_ann = orig, old_high, new_high, new_bal

    cols = [
        add_label(orig_ann,     "Original"),
        add_label(old_high_ann, "Old BRS rw=24 α=0.93", ssim_old_high),
        add_label(new_high_ann, "New BRS rw=24 α=0.93", ssim_new_high),
        add_label(new_bal_ann,  "New BRS rw=18 α=0.72", ssim_new_bal),
    ]

    # Ensure same height
    max_h = max(c.shape[0] for c in cols)
    padded = []
    for c in cols:
        if c.shape[0] < max_h:
            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.vstack([c, pad])
        padded.append(c)

    strip = np.hstack(padded)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{video}_frame{fi:03d}.png"
    cv2.imwrite(str(fname), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  saved: {fname.name}  "
          f"old={ssim_old_high:.3f}  new_high={ssim_new_high:.3f}  new_bal={ssim_new_bal:.3f}")

    return {
        "video": video, "frame": fi,
        "ssim_old_high": ssim_old_high,
        "ssim_new_high": ssim_new_high,
        "ssim_new_bal":  ssim_new_bal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS),
                    help="Comma-separated DAVIS video names")
    ap.add_argument("--frame_idx", type=int, default=5,
                    help="Frame index to visualise (0-based, picks next non-empty if empty)")
    ap.add_argument("--out_dir", default="results_v100/figures/brs_compare",
                    help="Output directory")
    ap.add_argument("--no_contour", action="store_true",
                    help="Suppress mask contour overlay")
    args = ap.parse_args()

    videos  = [v.strip() for v in args.videos.split(",") if v.strip()]
    out_dir = Path(args.out_dir)
    results = []

    print(f"Generating BRS comparison for {len(videos)} videos → {out_dir}")
    for vid in videos:
        print(f"\n[{vid}]")
        r = process_video(vid, args.frame_idx, out_dir,
                          show_contour=not args.no_contour)
        if r:
            results.append(r)

    if results:
        print("\n── Summary ──────────────────────────────────────────────────")
        print(f"{'Video':<25} {'OldBRS':>8} {'NewHigh':>8} {'NewBal':>8}")
        for r in results:
            print(f"{r['video']:<25} {r['ssim_old_high']:>8.3f} "
                  f"{r['ssim_new_high']:>8.3f} {r['ssim_new_bal']:>8.3f}")
        avg_old  = sum(r['ssim_old_high'] for r in results) / len(results)
        avg_newh = sum(r['ssim_new_high'] for r in results) / len(results)
        avg_newb = sum(r['ssim_new_bal']  for r in results) / len(results)
        print(f"{'MEAN':<25} {avg_old:>8.3f} {avg_newh:>8.3f} {avg_newb:>8.3f}")
        print(f"\nImages saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
