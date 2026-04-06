"""
Parameter Sensitivity Sweep — ring_width × blend_alpha Pareto Grid

Runs combo_strong on a small subset of videos across a grid of (ring_width, blend_alpha)
to characterize the effectiveness-quality Pareto frontier.

Generates:
  results_v100/param_sweep/<tag>/results.json   — per-cell, per-video results
  results_v100/param_sweep/<tag>/pareto.csv     — aggregated Pareto table
  results_v100/param_sweep/<tag>/pareto.png     — Pareto plot (SSIM vs ΔJF_codec)

Usage:
  python pilot_param_sweep.py \\
      --ring_widths 8,16,24,32 \\
      --blend_alphas 0.4,0.6,0.8 \\
      --videos bear,bike-packing,blackswan,bmx-trees,boat,breakdance,bus,car-roundabout,camel,cat-girl \\
      --max_frames 50 --crf 23 --prompt point \\
      --tag param_sweep_v1
"""

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor, run_tracking, apply_edit_to_video,
    codec_round_trip, frame_quality, edit_params,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ring_widths",   default="8,16,24,32",
                   help="Comma-separated list of ring_width values")
    p.add_argument("--blend_alphas",  default="0.4,0.6,0.8",
                   help="Comma-separated list of blend_alpha values")
    p.add_argument("--videos",        default="",
                   help="Comma-separated video names (empty = DAVIS_MINI_VAL)")
    p.add_argument("--max_frames",    type=int, default=50)
    p.add_argument("--crf",           type=int, default=23)
    p.add_argument("--prompt",        default="point", choices=["point", "mask"])
    p.add_argument("--min_jf_clean",  type=float, default=0.3)
    # Idea-2 halo params (held fixed at defaults for param sensitivity)
    p.add_argument("--halo_offset",   type=int,   default=8)
    p.add_argument("--halo_width",    type=int,   default=12)
    p.add_argument("--halo_strength", type=float, default=0.4)
    p.add_argument("--davis_root",    default=DAVIS_ROOT)
    p.add_argument("--checkpoint",    default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",   default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",   default=FFMPEG_PATH)
    p.add_argument("--save_dir",      default="results_v100/param_sweep")
    p.add_argument("--tag",           default="param_sweep_v1")
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ring_widths  = [int(x)   for x in args.ring_widths.split(",")  if x.strip()]
    blend_alphas = [float(x) for x in args.blend_alphas.split(",") if x.strip()]
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] or DAVIS_MINI_VAL

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[param_sweep] ring_widths={ring_widths}")
    print(f"[param_sweep] blend_alphas={blend_alphas}")
    print(f"[param_sweep] videos={videos} ({len(videos)} total)")
    print(f"[param_sweep] grid cells = {len(ring_widths) * len(blend_alphas)}")
    print(f"[param_sweep] total runs = {len(ring_widths) * len(blend_alphas) * len(videos)}")
    print(f"[param_sweep] output -> {out_dir}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    # Pre-load frames once per video
    video_data = {}
    for vid in videos:
        frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] {vid}: load failed")
            continue
        video_data[vid] = (frames, masks)
    print(f"[param_sweep] loaded {len(video_data)} videos")

    # Pre-compute clean and codec-clean baselines per video
    baselines = {}
    for vid, (frames, masks) in video_data.items():
        _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, args.prompt)
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] {vid}: JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue
        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_frames is None:
            continue
        _, jf_codec_clean, _, _ = run_tracking(codec_frames, masks, predictor, device, args.prompt)
        baselines[vid] = {"jf_clean": jf_clean, "jf_codec_clean": jf_codec_clean,
                          "frames": frames, "masks": masks}
        print(f"  baseline {vid}: JF_clean={jf_clean:.4f}  JF_codec_clean={jf_codec_clean:.4f}")

    all_results = []

    for ring_width, blend_alpha in product(ring_widths, blend_alphas):
        cell_key = f"rw{ring_width}_ba{blend_alpha:.2f}"
        print(f"\n{'='*50}")
        print(f"CELL: ring_width={ring_width}, blend_alpha={blend_alpha}")
        cell_results = []

        for vid, bsl in baselines.items():
            frames, masks = bsl["frames"], bsl["masks"]
            params = {
                "ring_width": ring_width,
                "blend_alpha": blend_alpha,
                "halo_offset": args.halo_offset,
                "halo_width":  args.halo_width,
                "halo_strength": args.halo_strength,
            }
            edited = apply_edit_to_video(frames, masks, "combo", params)

            # Quality
            ssim_vals, psnr_vals = [], []
            for fo, fe in zip(frames[:5], edited[:5]):
                s, p = frame_quality(fo, fe)
                ssim_vals.append(s)
                psnr_vals.append(p)
            mean_ssim = float(np.mean(ssim_vals))
            mean_psnr = float(np.nanmean([p for p in psnr_vals if p != float("inf")]))

            # Post-codec adversarial
            codec_edited = codec_round_trip(edited, args.ffmpeg_path, args.crf)
            if codec_edited is None:
                continue
            _, jf_codec_adv, _, _ = run_tracking(codec_edited, masks, predictor, device, args.prompt)
            delta_codec = bsl["jf_codec_clean"] - jf_codec_adv

            row = {
                "ring_width": ring_width,
                "blend_alpha": blend_alpha,
                "video": vid,
                "jf_clean": bsl["jf_clean"],
                "jf_codec_clean": bsl["jf_codec_clean"],
                "jf_codec_adv": jf_codec_adv,
                "delta_jf_codec": delta_codec,
                "mean_ssim": mean_ssim,
                "mean_psnr": mean_psnr,
            }
            cell_results.append(row)
            all_results.append(row)
            print(f"  {vid}: ΔJF_codec={delta_codec*100:+.1f}pp  SSIM={mean_ssim:.4f}")

        if cell_results:
            mean_d = np.mean([r["delta_jf_codec"] for r in cell_results])
            mean_s = np.mean([r["mean_ssim"] for r in cell_results])
            print(f"  → CELL mean: ΔJF_codec={mean_d*100:+.1f}pp  SSIM={mean_s:.4f}")

        # Intermediate save
        _save(out_dir, args, all_results)

    _save(out_dir, args, all_results)
    _print_pareto(out_dir, all_results, ring_widths, blend_alphas)


def _save(out_dir, args, results):
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    # Pareto CSV
    _write_pareto_csv(out_dir, results)


def _write_pareto_csv(out_dir, results):
    if not results:
        return
    # Aggregate by (ring_width, blend_alpha)
    cells = {}
    for r in results:
        key = (r["ring_width"], r["blend_alpha"])
        cells.setdefault(key, []).append(r)

    csv_path = out_dir / "pareto.csv"
    with open(csv_path, "w") as f:
        f.write("ring_width,blend_alpha,n_videos,mean_delta_jf_codec_pp,median_delta_jf_codec_pp,"
                "mean_ssim,mean_psnr,frac_ge5pp,frac_ge8pp,frac_ge12pp\n")
        for (rw, ba), rows in sorted(cells.items()):
            deltas = [r["delta_jf_codec"] for r in rows]
            ssims  = [r["mean_ssim"] for r in rows]
            psnrs  = [r["mean_psnr"] for r in rows]
            f.write(
                f"{rw},{ba:.2f},{len(rows)},"
                f"{np.mean(deltas)*100:.2f},{np.median(deltas)*100:.2f},"
                f"{np.mean(ssims):.4f},{np.nanmean(psnrs):.1f},"
                f"{sum(d >= 0.05 for d in deltas)/len(deltas):.3f},"
                f"{sum(d >= 0.08 for d in deltas)/len(deltas):.3f},"
                f"{sum(d >= 0.12 for d in deltas)/len(deltas):.3f}\n"
            )


def _print_pareto(out_dir, results, ring_widths, blend_alphas):
    if not results:
        return

    cells = {}
    for r in results:
        key = (r["ring_width"], r["blend_alpha"])
        cells.setdefault(key, []).append(r)

    print(f"\n{'='*70}")
    print(f"PARETO TABLE (mean ΔJF_codec pp | mean SSIM)")
    header = "ring\\alpha  " + "  ".join(f"{a:.2f}" for a in blend_alphas)
    print(header)
    for rw in ring_widths:
        row_str = f"  rw={rw:2d}    "
        for ba in blend_alphas:
            rows = cells.get((rw, ba), [])
            if rows:
                d = np.mean([r["delta_jf_codec"] for r in rows]) * 100
                s = np.mean([r["mean_ssim"] for r in rows])
                row_str += f" {d:+5.1f}pp/{s:.3f}"
            else:
                row_str += "         —     "
        print(row_str)
    print(f"{'='*70}")
    print(f"Pareto CSV saved: {out_dir / 'pareto.csv'}")


if __name__ == "__main__":
    main()
