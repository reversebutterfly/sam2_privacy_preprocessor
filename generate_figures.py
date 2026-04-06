"""
Generate qualitative figures for the paper.

For each selected video, saves:
  1. Original frame (frame 5)
  2. Edited frame (combo_strong applied)
  3. Side-by-side comparison with zoomed boundary region
  4. Difference map (amplified 5×)

Also generates:
  - Per-video ΔJF bar chart
  - Pre-codec vs post-codec scatter plot

Usage:
  python generate_figures.py \\
      --results_tag full_combo_strong \\
      --results_dir results_v100/mask_guided \\
      --videos bike-packing,bmx-trees,blackswan,dancing,elephant \\
      --frame_idx 5 \\
      --out figures/

  # Auto-select top/mid/bottom effect videos:
  python generate_figures.py --auto_select --n_per_group 2
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

from config import DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import apply_edit_to_video, codec_round_trip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_tag",  default="full_combo_strong")
    p.add_argument("--results_dir",  default="results_v100/mask_guided")
    p.add_argument("--videos",       default="",
                   help="Comma-separated video names. Empty = auto-select.")
    p.add_argument("--auto_select",  action="store_true",
                   help="Auto-select top/mid/bottom effect videos from results")
    p.add_argument("--n_per_group",  type=int, default=2,
                   help="Videos per group when --auto_select is used")
    p.add_argument("--frame_idx",    type=int, default=5)
    p.add_argument("--max_frames",   type=int, default=20)
    p.add_argument("--crf",          type=int, default=23)
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--ffmpeg_path",  default=FFMPEG_PATH)
    p.add_argument("--out",          default="figures/")
    return p.parse_args()


def zoom_boundary(frame: np.ndarray, mask: np.ndarray,
                  margin: int = 60, size: int = 256) -> np.ndarray:
    """Crop a region around the mask boundary and resize to (size, size)."""
    if mask.sum() == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    cy, cx = ys.mean(), xs.mean()
    y1 = max(0, int(cy) - margin)
    y2 = min(frame.shape[0], int(cy) + margin)
    x1 = max(0, int(cx) - margin)
    x2 = min(frame.shape[1], int(cx) + margin)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def make_comparison_panel(orig: np.ndarray, edited: np.ndarray,
                          orig_codec: np.ndarray, edited_codec: np.ndarray,
                          mask: np.ndarray, video: str, delta_pp: float) -> np.ndarray:
    """
    4-panel figure:
      [original | edited | zoomed orig boundary | zoomed edited boundary]
    with annotation.
    """
    H, W = orig.shape[:2]
    zoom_size = 256

    z_orig  = zoom_boundary(orig,  mask, margin=80, size=zoom_size)
    z_edit  = zoom_boundary(edited, mask, margin=80, size=zoom_size)

    # Resize main frames to consistent height
    target_h = 256
    scale = target_h / H
    w_resized = int(W * scale)
    orig_r  = cv2.resize(orig,  (w_resized, target_h))
    edit_r  = cv2.resize(edited,(w_resized, target_h))

    # Diff map (5× amplified)
    diff = np.abs(orig.astype(float) - edited.astype(float))
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    diff_r = cv2.resize(diff_amplified, (w_resized, target_h))

    # Codec diff
    if orig_codec is not None and edited_codec is not None:
        cdiff = np.abs(orig_codec.astype(float) - edited_codec.astype(float))
        cdiff_amplified = np.clip(cdiff * 5, 0, 255).astype(np.uint8)
        cdiff_r = cv2.resize(cdiff_amplified, (w_resized, target_h))
    else:
        cdiff_r = np.zeros_like(diff_r)

    # Combine: [orig | edited | diff | codec_diff] top row
    #          [zoom_orig | zoom_edit | padding] bottom row
    top = np.concatenate([orig_r, edit_r, diff_r, cdiff_r], axis=1)

    # Bottom row: zoomed boundaries side by side
    pad_w = top.shape[1] - 2 * zoom_size
    bottom_l = np.concatenate([z_orig, z_edit], axis=1)
    bottom_r = np.zeros((zoom_size, pad_w, 3), dtype=np.uint8)
    # Add text to bottom_r
    cv2.putText(bottom_r, f"{video}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(bottom_r, f"DeltaJF_codec: {delta_pp:+.1f}pp", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(bottom_r, "Left: original  Right: edited (boundary suppressed)", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    bottom = np.concatenate([bottom_l, bottom_r], axis=1)

    panel = np.concatenate([top, bottom], axis=0)

    # Column labels on top
    label_h = 24
    label_row = np.zeros((label_h, panel.shape[1], 3), dtype=np.uint8)
    labels = ["Original", "Edited (combo_strong)", "Diff x5 (pre-codec)", "Diff x5 (post-codec)"]
    col_w = w_resized
    for i, lbl in enumerate(labels):
        cv2.putText(label_row, lbl, (i * col_w + 5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return np.concatenate([label_row, panel], axis=0)


def make_bar_chart(results: list, out_path: Path):
    """Simple bar chart of per-video ΔJF_codec using only numpy/cv2."""
    valid = sorted(
        [r for r in results if isinstance(r.get("delta_jf_codec"), float)
         and r["delta_jf_codec"] == r["delta_jf_codec"]],
        key=lambda r: r["delta_jf_codec"], reverse=True
    )
    if not valid:
        return

    n = len(valid)
    W, H = max(1200, n * 12), 400
    img = np.ones((H, W, 3), dtype=np.uint8) * 30  # dark bg

    max_val = max(r["delta_jf_codec"] for r in valid) * 1.1
    bar_w = max(2, W // n - 1)

    mean_val = sum(r["delta_jf_codec"] for r in valid) / n

    for i, r in enumerate(valid):
        x = i * (bar_w + 1)
        val = r["delta_jf_codec"]
        bar_h = int(val / max_val * (H - 80))
        y1 = H - 40 - bar_h
        # Color: green gradient by effect strength
        g = int(50 + 200 * (val / max_val))
        cv2.rectangle(img, (x, y1), (x + bar_w, H - 40), (30, g, 80), -1)

    # Mean line
    mean_y = H - 40 - int(mean_val / max_val * (H - 80))
    cv2.line(img, (0, mean_y), (W, mean_y), (255, 200, 0), 2)
    cv2.putText(img, f"mean={mean_val*100:.1f}pp", (10, mean_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    # Title
    cv2.putText(img, f"Per-video DeltaJF_codec (combo_strong, n={n})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"  saved bar chart: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep results
    res_json = Path(args.results_dir) / args.results_tag / "results.json"
    with open(res_json) as f:
        sweep = json.load(f)
    sweep_args = sweep["args"]
    results = sweep["results"]

    # Generate bar chart
    make_bar_chart(results, out_dir / "bar_delta_jf_codec.png")

    # Determine which videos to visualize
    if args.videos:
        video_list = [v.strip() for v in args.videos.split(",") if v.strip()]
    elif args.auto_select:
        valid = sorted(
            [r for r in results if isinstance(r.get("delta_jf_codec"), float)
             and r["delta_jf_codec"] == r["delta_jf_codec"]
             and r.get("jf_clean", 0) >= 0.3],
            key=lambda r: r["delta_jf_codec"], reverse=True
        )
        n = args.n_per_group
        # top N, middle N, bottom N
        mid_start = len(valid) // 2 - n // 2
        video_list = (
            [r["video"] for r in valid[:n]] +
            [r["video"] for r in valid[mid_start:mid_start + n]] +
            [r["video"] for r in valid[-n:]]
        )
    else:
        video_list = ["bike-packing", "bmx-trees", "blackswan", "dancing", "elephant"]

    # Get edit params from sweep
    edit_params = {
        "ring_width":    sweep_args.get("ring_width", 24),
        "blend_alpha":   sweep_args.get("blend_alpha", 0.8),
        "halo_offset":   sweep_args.get("halo_offset", 12),
        "halo_width":    sweep_args.get("halo_width",  16),
        "halo_strength": sweep_args.get("halo_strength", 0.6),
    }
    edit_type = sweep_args.get("edit_type", "combo")

    # Result lookup
    result_map = {r["video"]: r for r in results}

    for vid in video_list:
        print(f"\n  === {vid} ===")
        frames, masks, _ = load_single_video(args.davis_root, vid,
                                             max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue

        edited = apply_edit_to_video(frames, masks, edit_type, edit_params)

        # Codec versions
        try:
            orig_codec   = codec_round_trip(frames,  args.ffmpeg_path, args.crf)
            edited_codec = codec_round_trip(edited,  args.ffmpeg_path, args.crf)
        except Exception as e:
            print(f"  [codec error] {e}")
            orig_codec = edited_codec = None

        fi = min(args.frame_idx, len(frames) - 1)
        orig_f   = frames[fi]
        edited_f = edited[fi]
        mask_f   = masks[fi]
        orig_c   = orig_codec[fi]   if orig_codec   else None
        edited_c = edited_codec[fi] if edited_codec else None

        delta_pp = result_map.get(vid, {}).get("delta_jf_codec", 0) * 100

        # Comparison panel
        panel = make_comparison_panel(orig_f, edited_f, orig_c, edited_c,
                                      mask_f, vid, delta_pp)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"compare_{vid}_f{fi:02d}.png"
        cv2.imwrite(str(out_path), panel_bgr)
        print(f"  saved: {out_path} | ΔJF_codec={delta_pp:+.1f}pp")

        # Also save clean originals for paper
        orig_bgr   = cv2.cvtColor(orig_f, cv2.COLOR_RGB2BGR)
        edited_bgr = cv2.cvtColor(edited_f, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{vid}_orig_f{fi:02d}.png"), orig_bgr)
        cv2.imwrite(str(out_dir / f"{vid}_edited_f{fi:02d}.png"), edited_bgr)

    print(f"\nFigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
