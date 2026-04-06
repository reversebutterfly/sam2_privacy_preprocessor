"""
YouTube-VOS AdvOpt Pilot — paired idea1 vs adv_opt comparison.

Runs both idea1 (fixed params) and adv_opt (first-frame constrained optimisation)
on YouTube-VOS 2019 to test whether per-video parameter adaptation helps beyond
DAVIS. This is an explicit distribution-shift stress test.

NOTE: Prior gate analysis on YT-VOS (n=497) showed the +4pp gap vs DAVIS is
content-distributional, not parameter-driven. This experiment verifies whether
adv_opt changes that conclusion.

Usage:
  # Sanity (1 video):
  python pilot_ytbvos_adv.py --sanity

  # Partial sweep (50 videos):
  python pilot_ytbvos_adv.py --max_videos 50 --tag ytvos_adv_n50

  # Full sweep:
  python pilot_ytbvos_adv.py --tag ytvos_adv_full

Outputs:
  results_v100/ytbvos_adv/<tag>/results.json   — per-video paired metrics
  results_v100/ytbvos_adv/<tag>/summary.csv    — flat table
  results_v100/ytbvos_adv/<tag>/pilot.log
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video_ytvos, list_ytvos_videos
from src.fancy_suppression import optimize_adv_params
from pilot_mask_guided import (
    apply_boundary_suppression,
    run_tracking,
    frame_quality,
    build_predictor,
    codec_round_trip,
)

DEFAULT_YTVOS_ROOT = os.path.join(ROOT, "data", "youtube_vos")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ytvos_root",  default=DEFAULT_YTVOS_ROOT)
    p.add_argument("--split",       default="valid",
                   help="Sub-folder with JPEG frames")
    p.add_argument("--anno_split",  default="valid",
                   help="Sub-folder with annotations")
    p.add_argument("--videos",      default="",
                   help="Comma-separated video IDs (empty = all annotated)")
    p.add_argument("--max_videos",  type=int, default=0,
                   help="Cap on number of videos to process (0 = no cap)")
    p.add_argument("--max_frames",  type=int, default=50)
    p.add_argument("--crf",         type=int, default=23)
    p.add_argument("--prompt",      default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.30)
    # idea1 fixed baseline params
    p.add_argument("--idea1_ring_width", type=int,   default=24)
    p.add_argument("--idea1_alpha",      type=float, default=0.80)
    # adv_opt params
    p.add_argument("--adv_n_iter",  type=int,   default=80)
    p.add_argument("--adv_lr",      type=float, default=0.08)
    p.add_argument("--ssim_floor",  type=float, default=0.92)
    # infra
    p.add_argument("--checkpoint",  default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir",    default="results_v100/ytbvos_adv")
    p.add_argument("--tag",         default="ytvos_adv_v1")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--sanity",      action="store_true",
                   help="Stop after first video")
    return p.parse_args()


def _save(out_dir: Path, args, results: list):
    (out_dir / "results.json").write_text(
        json.dumps({"args": vars(args), "results": results}, indent=2))
    rows = ["video,jf_codec_clean,"
            "idea1_jf_codec_adv,idea1_delta_jf_codec,idea1_ssim,"
            "adv_jf_codec_adv,adv_delta_jf_codec,adv_ssim,adv_rw,adv_alpha,"
            "paired_gain_pp"]
    for r in results:
        rows.append(
            f"{r['video']},"
            f"{r['jf_codec_clean']:.4f},"
            f"{r['idea1_jf_codec_adv']:.4f},{r['idea1_delta_jf_codec']*100:.2f},{r['idea1_ssim']:.4f},"
            f"{r['adv_jf_codec_adv']:.4f},{r['adv_delta_jf_codec']*100:.2f},{r['adv_ssim']:.4f},"
            f"{r['adv_ring_width']},{r['adv_alpha']:.4f},"
            f"{(r['adv_delta_jf_codec']-r['idea1_delta_jf_codec'])*100:.2f}"
        )
    (out_dir / "summary.csv").write_text("\n".join(rows))


def _print_summary(results):
    valid = [r for r in results
             if isinstance(r.get("adv_delta_jf_codec"), float)
             and r["adv_delta_jf_codec"] == r["adv_delta_jf_codec"]]
    if not valid:
        print("[summary] No valid results yet.")
        return
    import statistics
    idea1_d = [r["idea1_delta_jf_codec"] * 100 for r in valid]
    adv_d   = [r["adv_delta_jf_codec"]   * 100 for r in valid]
    gains   = [a - i for a, i in zip(adv_d, idea1_d)]
    wins    = sum(1 for g in gains if g > 0)
    print(f"\n{'='*60}")
    print(f"SUMMARY  n={len(valid)}")
    print(f"  idea1 (rw=24,α=0.80):  mean={statistics.mean(idea1_d):+.1f}pp")
    print(f"  adv_opt (1st-frame):   mean={statistics.mean(adv_d):+.1f}pp")
    print(f"  paired gain:           mean={statistics.mean(gains):+.1f}pp")
    print(f"  win-rate:              {wins}/{len(valid)} = {wins/len(valid):.1%}")
    print(f"{'='*60}\n")


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
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ytbvos_adv] {len(videos)} videos  prompt={args.prompt}  CRF={args.crf}")
    print(f"[ytbvos_adv] idea1: rw={args.idea1_ring_width}, α={args.idea1_alpha}")
    print(f"[ytbvos_adv] adv_opt: n_iter={args.adv_n_iter}, ssim_floor={args.ssim_floor}")
    print(f"[ytbvos_adv] output -> {out_dir}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    # Resume support
    res_json = out_dir / "results.json"
    done = set()
    if res_json.exists():
        with open(res_json) as f:
            saved = json.load(f)
        all_results = saved.get("results", [])
        done = {r["video"] for r in all_results}
        print(f"[ytbvos_adv] Resuming: {len(done)} videos already done")

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
        if not any(np.asarray(m).sum() > 0 for m in masks):
            print(f"  [skip] {vid}: no annotated masks")
            continue

        print(f"\n{'='*60}\nVIDEO: {vid}  ({len(frames)} frames)")

        # Clean tracking
        _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, args.prompt)
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue

        # Post-codec clean baseline
        codec_clean = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_clean:
            _, jf_codec_clean, _, _ = run_tracking(
                codec_clean, masks, predictor, device, args.prompt)
        else:
            jf_codec_clean = float("nan")

        # ── idea1 (fixed params) ──────────────────────────────────────────
        idea1_frames = [
            apply_boundary_suppression(f, m,
                ring_width=args.idea1_ring_width,
                blend_alpha=args.idea1_alpha)
            for f, m in zip(frames, masks)
        ]
        idea1_ssims = [frame_quality(fo, fe)[0]
                       for fo, fe in zip(frames, idea1_frames)]
        codec_idea1 = codec_round_trip(idea1_frames, args.ffmpeg_path, args.crf)
        if codec_idea1:
            _, jf_codec_idea1, _, _ = run_tracking(
                codec_idea1, masks, predictor, device, args.prompt)
            delta_idea1 = jf_codec_clean - jf_codec_idea1
        else:
            jf_codec_idea1 = delta_idea1 = float("nan")
        print(f"  idea1:   ΔJF_codec={delta_idea1*100:+.1f}pp  "
              f"SSIM={float(np.mean(idea1_ssims)):.3f}")

        # ── adv_opt (first-frame adaptation) ─────────────────────────────
        first_mask  = next((m for m in masks if np.asarray(m).sum() > 0), masks[0])
        first_frame = frames[0]
        adv_rw, adv_alpha = optimize_adv_params(
            first_frame, first_mask,
            n_iter=args.adv_n_iter,
            lr=args.adv_lr,
            ssim_floor=args.ssim_floor,
            device=str(device),
        )
        print(f"  adv_opt: rw={adv_rw}, α={adv_alpha:.3f}")
        adv_frames = [
            apply_boundary_suppression(f, m,
                ring_width=adv_rw, blend_alpha=adv_alpha)
            for f, m in zip(frames, masks)
        ]
        adv_ssims = [frame_quality(fo, fe)[0]
                     for fo, fe in zip(frames, adv_frames)]
        codec_adv = codec_round_trip(adv_frames, args.ffmpeg_path, args.crf)
        if codec_adv:
            _, jf_codec_adv, _, _ = run_tracking(
                codec_adv, masks, predictor, device, args.prompt)
            delta_adv = jf_codec_clean - jf_codec_adv
        else:
            jf_codec_adv = delta_adv = float("nan")
        print(f"  adv_opt: ΔJF_codec={delta_adv*100:+.1f}pp  "
              f"SSIM={float(np.mean(adv_ssims)):.3f}  "
              f"gain={( delta_adv - delta_idea1)*100:+.1f}pp")

        row = {
            "video":               vid,
            "jf_clean":            float(jf_clean),
            "jf_codec_clean":      float(jf_codec_clean),
            "idea1_jf_codec_adv":  float(jf_codec_idea1),
            "idea1_delta_jf_codec": float(delta_idea1),
            "idea1_ssim":          float(np.mean(idea1_ssims)),
            "adv_jf_codec_adv":    float(jf_codec_adv),
            "adv_delta_jf_codec":  float(delta_adv),
            "adv_ssim":            float(np.mean(adv_ssims)),
            "adv_ring_width":      adv_rw,
            "adv_alpha":           float(adv_alpha),
        }
        all_results.append(row)
        _save(out_dir, args, all_results)

        if args.sanity:
            print("\n[SANITY] Stopping after first video.")
            break

    _save(out_dir, args, all_results)
    _print_summary(all_results)


if __name__ == "__main__":
    main()
