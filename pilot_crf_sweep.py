"""
CRF Robustness Sweep for combo_strong edit.

Tests ΔJF_codec across CRF ∈ {18, 23, 28} to demonstrate that the boundary
suppression effect is robust to H.264 quality setting — not a CRF=23 artefact.

Each CRF value is an independent codec round-trip from the *same* edited frames,
so we can directly attribute differences to codec strength, not edit variability.

Usage:
  python pilot_crf_sweep.py \\
      --videos all \\
      --crfs 18,23,28 \\
      --ring_width 24 --blend_alpha 0.8 \\
      --prompt mask \\
      --tag crf_sweep_v1 \\
      --save_dir results_v100/crf_sweep

  # Sanity (1 video, fast):
  python pilot_crf_sweep.py --videos blackswan --sanity

Outputs:
  results_v100/crf_sweep/<tag>/results.json   (per-video × per-CRF rows)
  results_v100/crf_sweep/<tag>/summary.csv
  results_v100/crf_sweep/<tag>/sweep.log
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

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    apply_edit_to_video,
    run_tracking,
    frame_quality,
    build_predictor,
    codec_round_trip,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="all",
                   help="Comma-separated video names, or 'all' for full DAVIS 2017")
    p.add_argument("--crfs", default="18,23,28",
                   help="Comma-separated CRF values to sweep")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--prompt", default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    # combo_strong edit params (match full_combo_strong defaults)
    p.add_argument("--ring_width",    type=int,   default=24)
    p.add_argument("--blend_alpha",   type=float, default=0.8)
    p.add_argument("--halo_offset",   type=int,   default=12)
    p.add_argument("--halo_width",    type=int,   default=16)
    p.add_argument("--halo_strength", type=float, default=0.6)
    # Infrastructure
    p.add_argument("--davis_root",  default=DAVIS_ROOT)
    p.add_argument("--checkpoint",  default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir",    default="results_v100/crf_sweep")
    p.add_argument("--tag",         default="crf_sweep_v1")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--sanity",      action="store_true",
                   help="Stop after first video (pipeline check)")
    return p.parse_args()


def _get_all_davis_videos(davis_root: str) -> List[str]:
    img_root = Path(davis_root) / "JPEGImages" / "480p"
    if not img_root.exists():
        raise FileNotFoundError(f"DAVIS not found: {img_root}")
    return sorted(d.name for d in img_root.iterdir() if d.is_dir())


def _save(out_dir: Path, args, results: list, crfs: list):
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "crfs": crfs, "results": results}, f, indent=2)

    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w") as f:
        f.write("video,crf,jf_clean,jf_codec_clean,jf_codec_adv,"
                "delta_jf_codec,mean_ssim,mean_psnr\n")
        for r in results:
            def fmt(v, spec=".4f"):
                return format(v, spec) if isinstance(v, float) and v == v else "nan"
            f.write(
                f"{r['video']},{r['crf']},"
                f"{fmt(r.get('jf_clean', float('nan')))},"
                f"{fmt(r.get('jf_codec_clean', float('nan')))},"
                f"{fmt(r.get('jf_codec_adv', float('nan')))},"
                f"{fmt(r.get('delta_jf_codec', float('nan')))},"
                f"{fmt(r.get('mean_ssim', float('nan')))},"
                f"{fmt(r.get('mean_psnr', float('nan')), '.1f')}\n"
            )


def _print_summary(results: list, crfs: list):
    print("\n" + "=" * 60)
    print("CRF ROBUSTNESS SWEEP — SUMMARY")
    print("=" * 60)
    print(f"{'CRF':>5}  {'n':>4}  {'mean ΔJF_codec':>16}  {'CI95':>8}  {'mean SSIM':>10}")
    print("-" * 60)
    for crf in crfs:
        rows = [
            r for r in results
            if r.get("crf") == crf
            and isinstance(r.get("delta_jf_codec"), float)
            and r["delta_jf_codec"] == r["delta_jf_codec"]
        ]
        if not rows:
            print(f"  CRF {crf:2d}: no data")
            continue
        deltas = [r["delta_jf_codec"] for r in rows]
        n      = len(deltas)
        mean_d = sum(deltas) / n
        std_d  = (sum((d - mean_d) ** 2 for d in deltas) / n) ** 0.5
        ci95   = 1.96 * std_d / n ** 0.5
        ssims  = [r["mean_ssim"] for r in rows if isinstance(r.get("mean_ssim"), float)]
        mean_s = sum(ssims) / len(ssims) if ssims else float("nan")
        print(f"  {crf:3d}   {n:4d}   {mean_d*100:+8.2f}pp      ±{ci95*100:.2f}pp   {mean_s:.4f}")
    print("=" * 60)


def main():
    args   = parse_args()
    device = torch.device(args.device)
    crfs   = [int(c.strip()) for c in args.crfs.split(",") if c.strip()]

    if args.videos == "all":
        videos = _get_all_davis_videos(args.davis_root)
    else:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sweep.log"

    params = {
        "ring_width":    args.ring_width,
        "blend_alpha":   args.blend_alpha,
        "halo_offset":   args.halo_offset,
        "halo_width":    args.halo_width,
        "halo_strength": args.halo_strength,
    }

    print(f"[crf_sweep] CRFs={crfs}  prompt={args.prompt}")
    print(f"[crf_sweep] edit=combo_strong  params={params}")
    print(f"[crf_sweep] {len(videos)} videos -> {out_dir}")

    predictor   = build_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    # Resume: skip already-completed (video, crf) pairs
    res_json  = out_dir / "results.json"
    done_keys: set = set()
    if res_json.exists():
        with open(res_json) as f:
            saved = json.load(f)
        all_results = saved.get("results", [])
        done_keys   = {(r["video"], r["crf"]) for r in all_results}
        print(f"[crf_sweep] Resuming: {len(done_keys)} (video,crf) pairs already saved")

    for vid in videos:
        frames, masks, _ = load_single_video(
            args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] {vid}: load failed")
            continue

        # Clean JF — shared across all CRFs for this video
        _, jf_clean, j_clean, f_clean = run_tracking(
            frames, masks, predictor, device, args.prompt)
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] {vid}: JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue

        # Apply edit once — same edited_frames reused for all CRF codec round-trips
        edited_frames = apply_edit_to_video(frames, masks, "combo", params)

        # Per-frame quality (first 5 frames, independent of CRF)
        ssim_vals, psnr_vals = [], []
        for fo, fe in zip(frames[:5], edited_frames[:5]):
            s, p = frame_quality(fo, fe)
            ssim_vals.append(s)
            psnr_vals.append(p)
        mean_ssim = float(np.mean(ssim_vals))
        mean_psnr = float(np.nanmean([p for p in psnr_vals if p != float("inf")]))

        print(f"\n=== {vid}  JF_clean={jf_clean:.4f}  SSIM={mean_ssim:.4f} ===")

        for crf in crfs:
            if (vid, crf) in done_keys:
                print(f"  CRF{crf}: already done")
                continue

            # Codec-clean baseline at this CRF
            codec_clean = codec_round_trip(frames, args.ffmpeg_path, crf)
            if codec_clean:
                _, jf_codec_clean, _, _ = run_tracking(
                    codec_clean, masks, predictor, device, args.prompt)
            else:
                jf_codec_clean = float("nan")

            # Codec-adversarial at this CRF
            codec_edited = codec_round_trip(edited_frames, args.ffmpeg_path, crf)
            if codec_edited:
                _, jf_codec_adv, _, _ = run_tracking(
                    codec_edited, masks, predictor, device, args.prompt)
                delta_codec = jf_codec_clean - jf_codec_adv
            else:
                jf_codec_adv = float("nan")
                delta_codec  = float("nan")

            print(
                f"  CRF{crf}: codec_clean={jf_codec_clean:.4f}  "
                f"codec_adv={jf_codec_adv:.4f}  ΔJF={delta_codec:+.4f}"
            )

            row = {
                "video":          vid,
                "crf":            crf,
                "n_frames":       len(frames),
                "jf_clean":       jf_clean,
                "jf_codec_clean": jf_codec_clean,
                "jf_codec_adv":   jf_codec_adv,
                "delta_jf_codec": delta_codec,
                "mean_ssim":      mean_ssim,
                "mean_psnr":      mean_psnr,
            }
            all_results.append(row)
            done_keys.add((vid, crf))

            with open(log_path, "a") as lf:
                lf.write(
                    f"{vid}  CRF{crf}  ΔJF_codec={delta_codec:+.4f}  "
                    f"SSIM={mean_ssim:.4f}\n"
                )

            _save(out_dir, args, all_results, crfs)

        if args.sanity:
            print("\n[SANITY] Pipeline OK — stopping after first video.")
            break

    _save(out_dir, args, all_results, crfs)
    _print_summary(all_results, crfs)


if __name__ == "__main__":
    main()
