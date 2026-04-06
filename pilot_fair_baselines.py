"""
Fair Baseline Comparison — matched-distortion boundary edits.

Implements two baselines evaluated at the same ring region and blend
strength as combo_strong (rw=24, alpha=0.8), for a fair comparison:

  boundary_blur:    Gaussian blur restricted to the same boundary ring.
                    Same spatial support as our method, different edit op.
  interior_feather: Gaussian blur applied to mask interior only.
                    Tests whether interior vs boundary targeting matters.

Both baselines are compared against our combo_strong at matched SSIM.

Usage (sanity):
  python pilot_fair_baselines.py --edit_type boundary_blur --sanity

Usage (full sweep):
  python pilot_fair_baselines.py \\
      --edit_type boundary_blur \\
      --ring_width 24 --blend_alpha 0.8 \\
      --videos all --tag fair_boundary_blur_full

  python pilot_fair_baselines.py \\
      --edit_type interior_feather \\
      --blend_alpha 0.8 \\
      --videos all --tag fair_interior_feather_full
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor,
    run_tracking,
    codec_round_trip,
    frame_quality,
    EDIT_FNS,
)

# Only expose the two fair-baseline edit types from this script
_FAIR_EDIT_TYPES = ["boundary_blur", "interior_feather"]


def apply_edit_to_video(frames, masks, edit_type, params):
    fn = EDIT_FNS[edit_type]
    return [fn(f, m, **params) for f, m in zip(frames, masks)]


# ── Save / summary helpers ─────────────────────────────────────────────────────

def _save(out_dir: Path, args, results: list):
    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    with open(out_dir / "summary.csv", "w") as f:
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
    import math
    valid = [r for r in results
             if isinstance(r.get("delta_jf_codec"), float)
             and not math.isnan(r["delta_jf_codec"])]
    if not valid:
        print("[summary] No valid results.")
        return
    deltas = [r["delta_jf_codec"] for r in valid]
    n      = len(deltas)
    mean_d = sum(deltas) / n
    std_d  = (sum((d - mean_d) ** 2 for d in deltas) / n) ** 0.5
    ci95   = 1.96 * std_d / n ** 0.5
    ssims  = [r["mean_ssim"] for r in valid if isinstance(r.get("mean_ssim"), float)]
    mean_s = sum(ssims) / len(ssims) if ssims else float("nan")
    print("\n" + "=" * 60)
    print(f"Fair baseline ({valid[0].get('edit_type','?')}) — n={n}")
    print(f"  Mean ΔJF_codec = {mean_d*100:+.2f}pp  CI95 ±{ci95*100:.2f}pp")
    print(f"  Mean SSIM      = {mean_s:.4f}")
    print("=" * 60)


# ── Argparse ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edit_type", default="boundary_blur",
                   choices=_FAIR_EDIT_TYPES)
    p.add_argument("--videos",       default="",
                   help="Comma-separated video names or 'all' for full DAVIS")
    p.add_argument("--max_frames",   type=int,   default=50)
    p.add_argument("--crf",          type=int,   default=23)
    p.add_argument("--prompt",       default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    # Edit params (matched to combo_strong defaults)
    p.add_argument("--ring_width",   type=int,   default=24)
    p.add_argument("--blend_alpha",  type=float, default=0.8)
    # Infrastructure
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--checkpoint",   default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",  default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",  default=FFMPEG_PATH)
    p.add_argument("--save_dir",     default="results_v100/fair_baselines")
    p.add_argument("--tag",          default="fair_boundary_blur")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--sanity",       action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # Resolve video list
    _raw = [v.strip() for v in args.videos.split(",") if v.strip()]
    if _raw == ["all"]:
        from pathlib import Path as _P
        _img = _P(args.davis_root) / "JPEGImages" / "480p"
        videos = sorted(d.name for d in _img.iterdir() if d.is_dir()) \
                 if _img.exists() else DAVIS_MINI_VAL
    else:
        videos = _raw or DAVIS_MINI_VAL

    params = {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pilot.log"

    print(f"[fair_baselines] edit={args.edit_type}  prompt={args.prompt}  CRF={args.crf}")
    print(f"[fair_baselines] params={params}")
    print(f"[fair_baselines] videos ({len(videos)}) -> {out_dir}")

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
        print(f"[fair_baselines] Resuming: {len(done)} done")

    for vid in videos:
        if vid in done:
            continue

        frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] {vid}: load failed")
            continue

        print(f"\n=== {vid}  ({len(frames)} frames) ===")

        _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, args.prompt)
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue
        print(f"  clean: JF={jf_clean:.4f}")

        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_frames:
            _, jf_codec_clean, _, _ = run_tracking(
                codec_frames, masks, predictor, device, args.prompt)
            print(f"  codec_clean: JF={jf_codec_clean:.4f}")
        else:
            jf_codec_clean = float("nan")

        edited_frames = apply_edit_to_video(frames, masks, args.edit_type, params)

        ssim_vals, psnr_vals = [], []
        for fo, fe in zip(frames[:5], edited_frames[:5]):
            s, p = frame_quality(fo, fe)
            ssim_vals.append(s)
            psnr_vals.append(p)
        mean_ssim = float(np.mean(ssim_vals))
        mean_psnr = float(np.nanmean([p for p in psnr_vals if p != float("inf")]))
        print(f"  quality: SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.1f}dB")

        _, jf_adv, _, _ = run_tracking(edited_frames, masks, predictor, device, args.prompt)
        delta_adv = jf_clean - jf_adv
        print(f"  adv (pre-codec):  JF={jf_adv:.4f}  ΔJF={delta_adv:+.4f}")

        codec_edited = codec_round_trip(edited_frames, args.ffmpeg_path, args.crf)
        if codec_edited:
            _, jf_codec_adv, _, _ = run_tracking(
                codec_edited, masks, predictor, device, args.prompt)
            delta_codec = jf_codec_clean - jf_codec_adv
            print(f"  adv (post-codec): JF={jf_codec_adv:.4f}  ΔJF={delta_codec:+.4f}")
        else:
            jf_codec_adv = float("nan")
            delta_codec  = float("nan")

        row = {
            "video":          vid,
            "edit_type":      args.edit_type,
            "n_frames":       len(frames),
            "jf_clean":       jf_clean,
            "jf_codec_clean": jf_codec_clean,
            "jf_adv":         jf_adv,
            "jf_codec_adv":   jf_codec_adv,
            "delta_jf_adv":   delta_adv,
            "delta_jf_codec": delta_codec,
            "mean_ssim":      mean_ssim,
            "mean_psnr":      mean_psnr,
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
