"""
YouTube-VOS Generalisation Pilot — combo_strong edit.

Runs the same combo_strong boundary suppression on YouTube-VOS 2019 to
demonstrate that the effect generalises beyond DAVIS 2017.

Expected YouTube-VOS layout:
  data/youtube_vos/
    valid_all_frames/
      JPEGImages/<video_id>/<frame>.jpg   (dense frames, used for tracking)
    valid/
      Annotations/<video_id>/<frame>.png  (sparse GT, used for editing)
      meta.json

Usage:
  # Full sweep (all videos with annotations):
  python pilot_ytbvos.py \\
      --ytvos_root data/youtube_vos \\
      --prompt mask \\
      --ring_width 24 --blend_alpha 0.8 \\
      --tag ytbvos_combo_strong \\
      --save_dir results_v100/ytbvos

  # Sanity check (1 video):
  python pilot_ytbvos.py --videos <video_id> --sanity

Outputs:
  results_v100/ytbvos/<tag>/results.json
  results_v100/ytbvos/<tag>/summary.csv
  results_v100/ytbvos/<tag>/pilot.log
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video_ytvos, list_ytvos_videos
from pilot_mask_guided import (
    apply_edit_to_video,
    run_tracking,
    frame_quality,
    build_predictor,
    codec_round_trip,
    scale_norm_ring_width,
)

DEFAULT_YTVOS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "youtube_vos"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ytvos_root",    default=DEFAULT_YTVOS_ROOT)
    p.add_argument("--split",         default="valid",
                   help="Sub-folder with JPEG frames ('valid_all_frames' if dense split available)")
    p.add_argument("--anno_split",    default="valid",
                   help="Sub-folder with annotations")
    p.add_argument("--videos",        default="",
                   help="Comma-separated video IDs (empty = all annotated videos)")
    p.add_argument("--max_frames",    type=int, default=50)
    p.add_argument("--crf",           type=int, default=23)
    p.add_argument("--prompt",        default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean",  type=float, default=0.3)
    # combo_strong params — aligned with pilot_mask_guided.py defaults
    p.add_argument("--ring_width",    type=int,   default=24)
    p.add_argument("--blend_alpha",   type=float, default=0.8)
    p.add_argument("--halo_offset",   type=int,   default=8)    # was 12 — fixed
    p.add_argument("--halo_width",    type=int,   default=12)   # was 16 — fixed
    p.add_argument("--halo_strength", type=float, default=0.4)  # was 0.6 — fixed
    # Scale-normalised ring width
    p.add_argument("--ring_width_mode", default="fixed",
                   choices=["fixed", "scale_norm"],
                   help="fixed: use --ring_width as-is.  "
                        "scale_norm: per-frame ring_width = clip(round(rho*sqrt(area/pi)), 6, 32).")
    p.add_argument("--ring_width_rho", type=float, default=0.10,
                   help="Scale factor for scale_norm mode (typical: 0.06/0.10/0.14).")
    # Infrastructure
    p.add_argument("--checkpoint",    default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",   default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",   default=FFMPEG_PATH)
    p.add_argument("--save_dir",      default="results_v100/ytbvos")
    p.add_argument("--tag",           default="ytbvos_combo_strong")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--sanity",        action="store_true")
    return p.parse_args()


def _save(out_dir: Path, args, results: list):
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "dataset": "youtube_vos", "results": results},
                  f, indent=2)

    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w") as f:
        f.write("video,jf_clean,jf_codec_clean,jf_codec_adv,"
                "delta_jf_adv,delta_jf_codec,mean_ssim,mean_psnr\n")
        for r in results:
            def fmt(v, spec=".4f"):
                return format(v, spec) if isinstance(v, float) and v == v else "nan"
            f.write(
                f"{r['video']},"
                f"{fmt(r.get('jf_clean', float('nan')))},"
                f"{fmt(r.get('jf_codec_clean', float('nan')))},"
                f"{fmt(r.get('jf_codec_adv', float('nan')))},"
                f"{fmt(r.get('delta_jf_adv', float('nan')))},"
                f"{fmt(r.get('delta_jf_codec', float('nan')))},"
                f"{fmt(r.get('mean_ssim', float('nan')))},"
                f"{fmt(r.get('mean_psnr', float('nan')), '.1f')}\n"
            )


def _print_summary(results: list):
    valid = [
        r for r in results
        if isinstance(r.get("delta_jf_codec"), float)
        and r["delta_jf_codec"] == r["delta_jf_codec"]
    ]
    if not valid:
        print("[summary] No valid results yet.")
        return
    deltas = [r["delta_jf_codec"] for r in valid]
    n      = len(deltas)
    mean_d = sum(deltas) / n
    std_d  = (sum((d - mean_d) ** 2 for d in deltas) / n) ** 0.5
    ci95   = 1.96 * std_d / n ** 0.5
    print("\n" + "=" * 60)
    print(f"YouTube-VOS combo_strong — n={n}")
    print(f"  Mean ΔJF_codec = {mean_d*100:+.2f}pp  CI95 ±{ci95*100:.2f}pp")
    ssims = [r["mean_ssim"] for r in valid if isinstance(r.get("mean_ssim"), float)]
    if ssims:
        print(f"  Mean SSIM = {sum(ssims)/len(ssims):.4f}")
    print("=" * 60)


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # Resolve video list
    if args.videos:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        videos = list_ytvos_videos(
            args.ytvos_root,
            split=args.split,
            anno_split=args.anno_split,
            min_annotated_frames=1,
        )
        if not videos:
            raise FileNotFoundError(
                f"No annotated videos found in {args.ytvos_root}. "
                "Run scripts/download_ytbvos.sh first."
            )

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pilot.log"

    params = {
        "ring_width":    args.ring_width,
        "blend_alpha":   args.blend_alpha,
        "halo_offset":   args.halo_offset,
        "halo_width":    args.halo_width,
        "halo_strength": args.halo_strength,
    }

    print(f"[ytbvos] {len(videos)} videos  prompt={args.prompt}  CRF={args.crf}")
    print(f"[ytbvos] params={params}")
    print(f"[ytbvos] output -> {out_dir}")

    predictor   = build_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    # Resume support
    res_json = out_dir / "results.json"
    done     = set()
    if res_json.exists():
        with open(res_json) as f:
            saved = json.load(f)
        all_results = saved.get("results", [])
        done        = {r["video"] for r in all_results}
        print(f"[ytbvos] Resuming: {len(done)} videos already done")

    for vid in videos:
        if vid in done:
            continue

        frames, masks, _ = load_single_video_ytvos(
            args.ytvos_root, vid,
            split=args.split,
            anno_split=args.anno_split,
            max_frames=args.max_frames,
        )
        if not frames:
            print(f"  [skip] {vid}: load failed")
            continue

        # At least one annotated mask is required for editing
        if not any(m.sum() > 0 for m in masks):
            print(f"  [skip] {vid}: no annotated masks found")
            continue

        print(f"\n=== {vid}  ({len(frames)} frames) ===")

        _, jf_clean, j_clean, f_clean = run_tracking(
            frames, masks, predictor, device, args.prompt)
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue

        codec_clean_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_clean_frames:
            _, jf_codec_clean, _, _ = run_tracking(
                codec_clean_frames, masks, predictor, device, args.prompt)
        else:
            jf_codec_clean = float("nan")

        edited_frames = apply_edit_to_video(
            frames, masks, "combo", params,
            ring_width_mode=args.ring_width_mode,
            ring_width_rho=args.ring_width_rho,
        )

        ssim_vals, psnr_vals = [], []
        for fo, fe in zip(frames[:5], edited_frames[:5]):
            s, p = frame_quality(fo, fe)
            ssim_vals.append(s)
            psnr_vals.append(p)
        mean_ssim = float(np.mean(ssim_vals))
        mean_psnr = float(np.nanmean([p for p in psnr_vals if p != float("inf")]))

        _, jf_adv, _, _ = run_tracking(
            edited_frames, masks, predictor, device, args.prompt)
        delta_adv = jf_clean - jf_adv

        codec_edited = codec_round_trip(edited_frames, args.ffmpeg_path, args.crf)
        if codec_edited:
            _, jf_codec_adv, _, _ = run_tracking(
                codec_edited, masks, predictor, device, args.prompt)
            delta_codec = jf_codec_clean - jf_codec_adv
        else:
            jf_codec_adv = float("nan")
            delta_codec  = float("nan")

        print(
            f"  clean={jf_clean:.4f}  adv={jf_adv:.4f}  ΔJF_adv={delta_adv:+.4f}"
            f"  ΔJF_codec={delta_codec:+.4f}  SSIM={mean_ssim:.4f}"
        )

        row = {
            "video":            vid,
            "n_frames":         len(frames),
            "jf_clean":         jf_clean,
            "jf_codec_clean":   jf_codec_clean,
            "jf_adv":           jf_adv,
            "jf_codec_adv":     jf_codec_adv,
            "delta_jf_adv":     delta_adv,
            "delta_jf_codec":   delta_codec,
            "mean_ssim":        mean_ssim,
            "mean_psnr":        mean_psnr,
            "ring_width_mode":  args.ring_width_mode,
            "ring_width_rho":   args.ring_width_rho,
        }
        all_results.append(row)
        done.add(vid)

        with open(log_path, "a") as lf:
            lf.write(
                f"{vid}  ΔJF_adv={delta_adv:+.4f}  "
                f"ΔJF_codec={delta_codec:+.4f}  SSIM={mean_ssim:.4f}\n"
            )

        _save(out_dir, args, all_results)

        if args.sanity:
            print("\n[SANITY] Stopping after first video.")
            break

    _save(out_dir, args, all_results)
    _print_summary(all_results)


if __name__ == "__main__":
    main()
