"""
Codec Amplification Mechanism Analysis.

Generates the mechanism evidence figure showing:
  1. Boundary-region DCT energy before/after edit (low-freq shift)
  2. Boundary contrast (gradient magnitude) before/after edit and after codec
  3. Edit is preserved/amplified by H.264 in the low-frequency band

Usage:
  python analyze_codec_amplification.py \\
      --video blackswan \\
      --out_dir figures/codec_amplification

  # Full analysis (multiple videos):
  python analyze_codec_amplification.py \\
      --videos blackswan,dog,car-roundabout \\
      --out_dir figures/codec_amplification
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DAVIS_ROOT, SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    apply_edit_to_video,
    codec_round_trip,
)


# ── DCT energy analysis ────────────────────────────────────────────────────────

def boundary_ring_mask(mask: np.ndarray, ring_width: int = 24) -> np.ndarray:
    """Return the boundary ring region as a bool mask."""
    kernel = np.ones((ring_width * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded  = cv2.erode(mask, kernel)
    return (dilated > 0) & (eroded == 0)


def dct_energy_bands(patch: np.ndarray, n_bands: int = 4) -> np.ndarray:
    """
    Compute DCT energy in n_bands frequency bands for a grayscale patch.
    Band 0 = lowest frequencies (DC + very low), Band n-1 = highest.
    Returns normalised energy per band summing to 1.
    """
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        patch = patch.astype(np.float32)

    if patch.size == 0:
        return np.ones(n_bands) / n_bands

    # Resize to fixed block for consistent DCT
    block_sz = min(64, min(patch.shape[:2]))
    patch_r  = cv2.resize(patch, (block_sz, block_sz), interpolation=cv2.INTER_AREA)
    dct_c    = cv2.dct(patch_r)
    energy   = dct_c ** 2

    # Split into n_bands frequency shells (diagonal distance from DC)
    H, W = energy.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((ys / H) ** 2 + (xs / W) ** 2)  # 0=DC, 1.41=highest

    band_edges = np.linspace(0, np.sqrt(2), n_bands + 1)
    band_energy = np.array([
        energy[(dist >= band_edges[b]) & (dist < band_edges[b + 1])].sum()
        for b in range(n_bands)
    ], dtype=np.float64)

    total = band_energy.sum()
    return band_energy / total if total > 0 else np.ones(n_bands) / n_bands


def gradient_magnitude_in_ring(frame: np.ndarray, ring: np.ndarray) -> float:
    """Mean gradient magnitude (Sobel) inside the ring region."""
    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag[ring].mean()) if ring.sum() > 0 else 0.0


def analyse_video(frames, masks, ffmpeg_path, crf, ring_width=24, n_bands=4):
    """
    Compute per-frame:
      - DCT energy bands in boundary ring (orig, edited, codec_edited)
      - Gradient magnitude in boundary ring (orig, edited, codec_edited)

    Returns dict with aggregated means.
    """
    COMBO_PARAMS = {
        "ring_width": ring_width, "blend_alpha": 0.8,
        "halo_offset": 12, "halo_width": 16, "halo_strength": 0.6,
    }
    edited_frames = apply_edit_to_video(frames, masks, "combo", COMBO_PARAMS)
    codec_edited  = codec_round_trip(edited_frames, ffmpeg_path, crf)
    if codec_edited is None:
        return None

    orig_dct, edit_dct, ce_dct = [], [], []
    orig_grad, edit_grad, ce_grad = [], [], []

    for orig, edit, ce, mask in zip(frames, edited_frames, codec_edited, masks):
        ring = boundary_ring_mask(mask, ring_width)
        if ring.sum() < 50:
            continue

        # Extract boundary patch as bounding-box crop
        ys, xs = np.where(ring)
        y1, y2 = max(0, ys.min()), min(orig.shape[0], ys.max() + 1)
        x1, x2 = max(0, xs.min()), min(orig.shape[1], xs.max() + 1)

        orig_patch = orig[y1:y2, x1:x2]
        edit_patch = edit[y1:y2, x1:x2]
        ce_patch   = ce[y1:y2, x1:x2]

        if orig_patch.size == 0:
            continue

        orig_dct.append(dct_energy_bands(orig_patch, n_bands))
        edit_dct.append(dct_energy_bands(edit_patch, n_bands))
        ce_dct.append(dct_energy_bands(ce_patch, n_bands))

        orig_grad.append(gradient_magnitude_in_ring(orig, ring))
        edit_grad.append(gradient_magnitude_in_ring(edit, ring))
        ce_grad.append(gradient_magnitude_in_ring(ce, ring))

    if not orig_dct:
        return None

    return {
        "n_frames":  len(orig_dct),
        "orig_dct":  np.mean(orig_dct, axis=0).tolist(),
        "edit_dct":  np.mean(edit_dct, axis=0).tolist(),
        "ce_dct":    np.mean(ce_dct,  axis=0).tolist(),
        "orig_grad": float(np.mean(orig_grad)),
        "edit_grad": float(np.mean(edit_grad)),
        "ce_grad":   float(np.mean(ce_grad)),
    }


def make_figure(results_by_video: dict, out_dir: Path, n_bands: int = 4):
    """Generate and save the codec amplification figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib not available — skipping figure generation")
        return

    all_orig_dct = []
    all_edit_dct = []
    all_ce_dct   = []
    all_orig_grad, all_edit_grad, all_ce_grad = [], [], []

    for r in results_by_video.values():
        all_orig_dct.append(r["orig_dct"])
        all_edit_dct.append(r["edit_dct"])
        all_ce_dct.append(r["ce_dct"])
        all_orig_grad.append(r["orig_grad"])
        all_edit_grad.append(r["edit_grad"])
        all_ce_grad.append(r["ce_grad"])

    orig_dct = np.mean(all_orig_dct, axis=0)
    edit_dct = np.mean(all_edit_dct, axis=0)
    ce_dct   = np.mean(all_ce_dct,  axis=0)

    band_labels = [f"B{i+1}" for i in range(n_bands)]
    band_labels[0]  = "Low\n(B1)"
    band_labels[-1] = "High\n(B4)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Codec Amplification Mechanism", fontsize=13, fontweight="bold")

    # ── Panel 1: DCT energy bands ──────────────────────────────────────────────
    x = np.arange(n_bands)
    w = 0.25
    ax1.bar(x - w, orig_dct, w, label="Original",       color="#4878CF")
    ax1.bar(x,     edit_dct, w, label="After edit",      color="#6ACC65")
    ax1.bar(x + w, ce_dct,  w, label="After edit+codec", color="#D65F5F")
    ax1.set_xlabel("Frequency band (boundary region)")
    ax1.set_ylabel("Normalised DCT energy")
    ax1.set_title("(a) Frequency energy shift in boundary ring")
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1)

    # ── Panel 2: Gradient magnitude ────────────────────────────────────────────
    conditions = ["Original", "After edit", "After\nedit+codec"]
    means  = [np.mean(all_orig_grad), np.mean(all_edit_grad), np.mean(all_ce_grad)]
    stds   = [np.std(all_orig_grad),  np.std(all_edit_grad),  np.std(all_ce_grad)]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    ax2.bar(range(3), means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel("Mean gradient magnitude (boundary ring)")
    ax2.set_title("(b) Boundary contrast suppression")
    ax2.set_ylim(0, None)

    # Annotate % drop
    if means[0] > 0:
        drop_edit  = (means[0] - means[1]) / means[0] * 100
        drop_codec = (means[0] - means[2]) / means[0] * 100
        ax2.annotate(f"−{drop_edit:.0f}%",  xy=(1, means[1] + stds[1] + 0.5),
                     ha="center", fontsize=9, color="#6ACC65")
        ax2.annotate(f"−{drop_codec:.0f}%", xy=(2, means[2] + stds[2] + 0.5),
                     ha="center", fontsize=9, color="#D65F5F")

    plt.tight_layout()
    out_path = out_dir / "codec_amplification.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    out_png = out_dir / "codec_amplification.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[fig] saved → {out_path}")
    print(f"[fig] saved → {out_png}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos",       default="blackswan,dog,car-roundabout,flamingo,dance-twirl")
    p.add_argument("--max_frames",   type=int, default=30)
    p.add_argument("--crf",          type=int, default=23)
    p.add_argument("--ring_width",   type=int, default=24)
    p.add_argument("--n_bands",      type=int, default=4)
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--ffmpeg_path",  default=FFMPEG_PATH)
    p.add_argument("--out_dir",      default="figures/codec_amplification")
    return p.parse_args()


def main():
    args = parse_args()
    video_list = [v.strip() for v in args.videos.split(",") if v.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_by_video = {}

    for vid in video_list:
        print(f"\n=== {vid} ===")
        frames, masks, _ = load_single_video(args.davis_root, vid,
                                             max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue

        r = analyse_video(frames, masks, args.ffmpeg_path, args.crf,
                          ring_width=args.ring_width, n_bands=args.n_bands)
        if r is None:
            print(f"  [skip] analysis failed")
            continue

        print(f"  frames_analysed={r['n_frames']}")
        print(f"  orig_dct ={[f'{v:.3f}' for v in r['orig_dct']]}")
        print(f"  edit_dct ={[f'{v:.3f}' for v in r['edit_dct']]}")
        print(f"  ce_dct   ={[f'{v:.3f}' for v in r['ce_dct']]}")
        print(f"  grad: orig={r['orig_grad']:.2f}  edit={r['edit_grad']:.2f}  "
              f"ce={r['ce_grad']:.2f}  "
              f"Δ(edit)={r['orig_grad']-r['edit_grad']:+.2f}  "
              f"Δ(codec)={r['orig_grad']-r['ce_grad']:+.2f}")
        results_by_video[vid] = r

    # Save raw numbers
    with open(out_dir / "codec_amplification_data.json", "w") as f:
        json.dump(results_by_video, f, indent=2)
    print(f"\n[data] saved → {out_dir}/codec_amplification_data.json")

    # Print aggregate
    if results_by_video:
        all_orig_g = [v["orig_grad"] for v in results_by_video.values()]
        all_edit_g = [v["edit_grad"] for v in results_by_video.values()]
        all_ce_g   = [v["ce_grad"]   for v in results_by_video.values()]
        n = len(all_orig_g)
        print("\n" + "=" * 60)
        print(f"Aggregate over {n} videos:")
        print(f"  Orig grad:       {np.mean(all_orig_g):.2f} ± {np.std(all_orig_g):.2f}")
        print(f"  Edit grad:       {np.mean(all_edit_g):.2f} ± {np.std(all_edit_g):.2f}  "
              f"(Δ={np.mean(all_orig_g)-np.mean(all_edit_g):+.2f})")
        print(f"  Edit+codec grad: {np.mean(all_ce_g):.2f} ± {np.std(all_ce_g):.2f}  "
              f"(Δ={np.mean(all_orig_g)-np.mean(all_ce_g):+.2f})")

        # DCT bands
        orig_dct = np.mean([v["orig_dct"] for v in results_by_video.values()], axis=0)
        edit_dct = np.mean([v["edit_dct"] for v in results_by_video.values()], axis=0)
        ce_dct   = np.mean([v["ce_dct"]   for v in results_by_video.values()], axis=0)
        print(f"\n  Orig DCT bands:       {[f'{v:.3f}' for v in orig_dct]}")
        print(f"  Edit DCT bands:       {[f'{v:.3f}' for v in edit_dct]}")
        print(f"  Edit+codec DCT bands: {[f'{v:.3f}' for v in ce_dct]}")
        lf_shift = edit_dct[0] - orig_dct[0]
        codec_amp = ce_dct[0] - orig_dct[0]
        print(f"\n  Low-freq shift (edit):  {lf_shift:+.4f}")
        print(f"  Low-freq shift (codec): {codec_amp:+.4f}  "
              f"({'amplified' if codec_amp > lf_shift else 'attenuated'} by codec)")
        print("=" * 60)

        make_figure(results_by_video, out_dir, args.n_bands)


if __name__ == "__main__":
    main()
