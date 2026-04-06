"""
Pilot: Fancy Suppression Methods Evaluation

Compares baseline (idea1) against three enhanced variants:
  msb      — Multi-Scale Boundary suppression
  adv_opt  — Adversarial Parameter Optimisation
  lfnet    — Learned Feathering Network
  full     — All three chained

Metrics reported per video and in aggregate:
  ΔJF_codec  — post-H.264 J&F degradation (primary metric)
  SSIM       — visual quality (secondary constraint)
  PSNR       — dB
  ring_width / blend_alpha  — optimised params (adv_opt / full only)

Usage (on GPU server after conda activate sam2_privacy_preprocessor):

  # Quick sanity (1 video, all methods):
  python pilot_fancy_eval.py --videos bear --sanity --device cuda

  # Full DAVIS mini-val sweep (10 videos):
  python pilot_fancy_eval.py --videos "" --methods idea1,msb,adv_opt,lfnet --device cuda

  # Full comparison on all DAVIS:
  python pilot_fancy_eval.py --videos all --methods idea1,msb,adv_opt,full \\
      --max_frames 50 --crf 23 --prompt point --tag fancy_v1 --device cuda

Results saved to:
  results_v100/fancy_eval/<tag>/results.json
  results_v100/fancy_eval/<tag>/summary.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor, run_tracking, codec_round_trip, frame_quality,
    apply_boundary_suppression,   # idea1
)
from src.fancy_suppression import (
    apply_multiscale_suppression,
    apply_adv_opt_suppression,
    apply_lfnet_suppression,
    apply_full_fancy,
    train_lfnet_per_video,
    optimize_adv_params,
)


# ── Method dispatch ────────────────────────────────────────────────────────────

def apply_method(
    method: str,
    frames: list,
    masks: list,
    args,
    device: str,
) -> tuple:
    """
    Apply `method` to all frames, return (edited_frames, per_frame_extra_info).
    extra_info: list of dicts (may include optimised params, timing, etc.)
    """
    edited, infos = [], []

    if method == "idea1":
        for frame, mask in zip(frames, masks):
            edited.append(apply_boundary_suppression(
                frame, mask,
                ring_width=args.ring_width, blend_alpha=args.blend_alpha))
            infos.append({"ring_width": args.ring_width, "blend_alpha": args.blend_alpha})

    elif method == "msb":
        scales = [int(x) for x in args.msb_scales.split(",")]
        alphas = [float(x) for x in args.msb_alphas.split(",")]
        for frame, mask in zip(frames, masks):
            edited.append(apply_multiscale_suppression(frame, mask, scales=scales, alphas=alphas))
            infos.append({"scales": scales, "alphas": alphas})

    elif method == "adv_opt":
        # Optimise per-video using the first annotated frame (fast proxy)
        first_mask = next((m for m in masks if m.sum() > 0), masks[0])
        first_frame = frames[0]
        t0 = time.time()
        rw, alpha = optimize_adv_params(
            first_frame, first_mask,
            n_iter=args.adv_n_iter, lr=args.adv_lr,
            ssim_floor=args.ssim_floor, device=device)
        adv_time = time.time() - t0
        print(f"    [adv_opt] optimised: ring_width={rw}, blend_alpha={alpha:.3f} ({adv_time:.1f}s)")
        for frame, mask in zip(frames, masks):
            edited.append(apply_boundary_suppression(frame, mask, ring_width=rw, blend_alpha=alpha))
            infos.append({"ring_width": rw, "blend_alpha": alpha, "adv_opt_time": adv_time})

    elif method == "lfnet":
        # Fine-tune LFNet per-video (using first annotated frame as representative)
        first_mask  = next((m for m in masks if m.sum() > 0), masks[0])
        first_frame = frames[0]
        t0 = time.time()
        model = train_lfnet_per_video(
            first_frame, first_mask,
            n_iter=args.lfnet_n_iter, ssim_floor=args.ssim_floor, device=device)
        train_time = time.time() - t0
        print(f"    [lfnet] trained: {train_time:.1f}s")
        for frame, mask in zip(frames, masks):
            edited.append(apply_lfnet_suppression(frame, mask, model=model, device=device))
            infos.append({"lfnet_train_time": train_time})

    elif method == "full":
        # Full pipeline: AdvOpt → MSB → LFNet (all per-video)
        first_mask  = next((m for m in masks if m.sum() > 0), masks[0])
        first_frame = frames[0]
        t0 = time.time()
        rw, alpha = optimize_adv_params(
            first_frame, first_mask,
            n_iter=args.adv_n_iter, lr=args.adv_lr,
            ssim_floor=args.ssim_floor, device=device)

        # MSB with AdvOpt-tuned params
        scales_use = [max(4, rw // 2), rw, min(40, rw + rw // 2)]
        alphas_use = [min(0.95, alpha), min(0.85, alpha * 0.88), min(0.75, alpha * 0.75)]
        msb_edited = [
            apply_multiscale_suppression(f, m, scales=scales_use, alphas=alphas_use)
            for f, m in zip(frames, masks)
        ]

        # LFNet fine-tune on first MSB-edited frame
        model = train_lfnet_per_video(
            msb_edited[0], first_mask,
            n_iter=args.lfnet_n_iter, ssim_floor=args.ssim_floor, device=device)

        full_time = time.time() - t0
        print(f"    [full] pipeline done: {full_time:.1f}s  rw={rw}, alpha={alpha:.3f}")

        for msb_f, mask in zip(msb_edited, masks):
            edited.append(apply_lfnet_suppression(msb_f, mask, model=model, device=device))
            infos.append({"ring_width": rw, "blend_alpha": alpha,
                          "scales": scales_use, "full_time": full_time})
    else:
        raise ValueError(f"Unknown method: {method}")

    return edited, infos


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="idea1,msb,adv_opt,lfnet",
                   help="Comma-separated methods to evaluate")
    p.add_argument("--videos",  default="",
                   help="Comma-separated video names (empty=DAVIS_MINI_VAL, 'all'=all DAVIS)")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf",        type=int, default=23)
    p.add_argument("--prompt",     default="point", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    # Baseline idea1 params
    p.add_argument("--ring_width",  type=int,   default=24)
    p.add_argument("--blend_alpha", type=float, default=0.80)
    # MSB params
    p.add_argument("--msb_scales", default="8,16,24",
                   help="Comma-separated ring scales for MSB")
    p.add_argument("--msb_alphas", default="0.85,0.75,0.60",
                   help="Comma-separated alpha values for MSB (same count as scales)")
    # AdvOpt params
    p.add_argument("--adv_n_iter",  type=int,   default=80)
    p.add_argument("--adv_lr",      type=float, default=0.08)
    p.add_argument("--ssim_floor",  type=float, default=0.92)
    # LFNet params
    p.add_argument("--lfnet_n_iter", type=int, default=150)
    # Infrastructure
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--checkpoint",   default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",  default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",  default=FFMPEG_PATH)
    p.add_argument("--save_dir",     default="results_v100/fancy_eval")
    p.add_argument("--tag",          default="fancy_v1")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--sanity",       action="store_true",
                   help="Stop after first video")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = args.device
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    raw = [v.strip() for v in args.videos.split(",") if v.strip()]
    if raw == ["all"]:
        img_root = Path(args.davis_root) / "JPEGImages" / "480p"
        videos = sorted(d.name for d in img_root.iterdir() if d.is_dir()) if img_root.exists() else DAVIS_MINI_VAL
    else:
        videos = raw or DAVIS_MINI_VAL

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fancy_eval] methods  = {methods}")
    print(f"[fancy_eval] videos   = {videos} ({len(videos)} total)")
    print(f"[fancy_eval] output   → {out_dir}")

    predictor   = build_predictor(args.checkpoint, args.sam2_config, torch.device(device))
    all_results = []

    for vid in videos:
        print(f"\n{'='*60}")
        print(f"VIDEO: {vid}")
        frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue

        # Clean baseline
        _, jf_clean, j_clean, f_clean = run_tracking(
            frames, masks, predictor, torch.device(device), args.prompt)
        print(f"  clean: JF={jf_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean < {args.min_jf_clean}")
            continue

        # Codec-clean baseline
        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_frames is None:
            print(f"  [skip] codec round-trip failed")
            continue
        _, jf_codec_clean, _, _ = run_tracking(
            codec_frames, masks, predictor, torch.device(device), args.prompt)
        print(f"  codec_clean: JF={jf_codec_clean:.4f}")

        for method in methods:
            print(f"\n  --- {method} ---")
            try:
                t0 = time.time()
                edited_frames, infos = apply_method(method, frames, masks, args, device)
                method_time = time.time() - t0

                # Quality (first 5 frames)
                ssim_vals, psnr_vals = [], []
                for fo, fe in zip(frames[:5], edited_frames[:5]):
                    s, p = frame_quality(fo, fe)
                    ssim_vals.append(s)
                    psnr_vals.append(p)
                mean_ssim = float(np.mean(ssim_vals))
                mean_psnr = float(np.nanmean([p for p in psnr_vals if p != float("inf")]))

                # Post-codec adversarial
                codec_edited = codec_round_trip(edited_frames, args.ffmpeg_path, args.crf)
                if codec_edited is None:
                    print(f"    [skip] codec failed on edited frames")
                    continue
                _, jf_codec_adv, _, _ = run_tracking(
                    codec_edited, masks, predictor, torch.device(device), args.prompt)
                delta_codec = jf_codec_clean - jf_codec_adv

                print(f"    ΔJF_codec={delta_codec*100:+.1f}pp  SSIM={mean_ssim:.4f}  "
                      f"PSNR={mean_psnr:.1f}dB  time={method_time:.1f}s")

                row = {
                    "video": vid, "method": method, "n_frames": len(frames),
                    "jf_clean": jf_clean, "jf_codec_clean": jf_codec_clean,
                    "jf_codec_adv": jf_codec_adv,
                    "delta_jf_codec": delta_codec,
                    "mean_ssim": mean_ssim, "mean_psnr": mean_psnr,
                    "method_time_s": method_time,
                    "info": infos[0] if infos else {},
                }
                all_results.append(row)

            except Exception as e:
                print(f"    [error] {method} on {vid}: {e}")
                import traceback; traceback.print_exc()

        _save(out_dir, args, all_results)

        if args.sanity:
            print("\n[SANITY MODE] Stopping after first video.")
            break

    _save(out_dir, args, all_results)
    _print_comparison(all_results, methods)


def _save(out_dir, args, results):
    (out_dir / "results.json").write_text(
        json.dumps({"args": vars(args), "results": results}, indent=2))
    # CSV
    rows = ["video,method,delta_jf_codec_pp,mean_ssim,mean_psnr,jf_codec_clean,jf_codec_adv"]
    for r in results:
        rows.append(f"{r['video']},{r['method']},"
                    f"{r['delta_jf_codec']*100:.2f},{r['mean_ssim']:.4f},"
                    f"{r['mean_psnr']:.1f},"
                    f"{r['jf_codec_clean']:.4f},{r['jf_codec_adv']:.4f}")
    (out_dir / "summary.csv").write_text("\n".join(rows))


def _print_comparison(results, methods):
    if not results:
        return
    print(f"\n{'='*70}")
    print(f"{'Method':<12}  {'n':>3}  {'mean ΔJF_codec':>14}  "
          f"{'≥10pp':>5}  {'≥15pp':>5}  {'SSIM':>6}")
    print("-" * 70)
    for method in methods:
        rows = [r for r in results if r["method"] == method and
                not (isinstance(r["delta_jf_codec"], float) and r["delta_jf_codec"] != r["delta_jf_codec"])]
        if not rows:
            continue
        deltas = [r["delta_jf_codec"] for r in rows]
        ssims  = [r["mean_ssim"]       for r in rows]
        n = len(rows)
        print(f"{method:<12}  {n:>3}  {np.mean(deltas)*100:>+13.1f}pp  "
              f"{sum(d>=0.10 for d in deltas):>5}  "
              f"{sum(d>=0.15 for d in deltas):>5}  "
              f"{np.mean(ssims):>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
