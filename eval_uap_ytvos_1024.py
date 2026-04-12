"""
eval_uap_ytvos_1024.py — UAP-SAM2 in-distribution validation on YouTube-VOS.

PURPOSE: Verify our 1024-protocol UAP eval pipeline by reproducing the official
UAP-SAM2 paper's mIoU drop on YouTube-VOS with point prompts. The official
JPEG-eval reproduction (uap_eval_heldout_jpeg.py) reports ~24pp mIoU drop;
ours should be in the same ballpark if the pipeline is correct.

Key differences from eval_uap_davis_1024.py:
  - YouTube-VOS data loader (load_single_video_ytvos)
  - Default --prompt point (matches UAP-SAM2 training)
  - Reports ΔmIoU prominently (= ΔJ, the paper's metric)
  - Otherwise identical: 1024×1024 protocol, JPEG QF=95, NN mask resize

Usage:
  # Sanity (1 video, 5 frames):
  python eval_uap_ytvos_1024.py --uap_path .../YOUTUBE.pth --max_videos 1 --max_frames 5 --sanity --tag uap1024_yt_sanity

  # Full reproduction (matches official protocol):
  python eval_uap_ytvos_1024.py --uap_path .../YOUTUBE.pth --max_videos 100 --max_frames 15 --tag uap1024_yt_repro
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video_ytvos, list_ytvos_videos
from src.codec_eot import encode_decode_h264
from src.metrics import quality_summary, mean_jf
from pilot_mask_guided import build_predictor
from eval_uap_davis_1024 import (
    JPEG_QUALITY,
    resize_frame_to_1024,
    resize_mask_to_1024,
    apply_uap_1024,
    apply_random_1024,
    apply_none,
    run_tracking_1024,
    codec_round_trip_1024,
    frame_quality_1024,
)

DEFAULT_YTVOS_ROOT = os.path.join(ROOT, "data", "youtube_vos")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="uap", choices=["uap", "random", "none"])
    p.add_argument("--uap_path", default="")
    p.add_argument("--eps", type=int, default=10)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--ytvos_root", default=DEFAULT_YTVOS_ROOT)
    p.add_argument("--split", default="valid")
    p.add_argument("--anno_split", default="valid")
    p.add_argument("--videos", default="")
    p.add_argument("--max_videos", type=int, default=100,
                   help="Cap on number of videos (matches official limit_img=100)")
    p.add_argument("--max_frames", type=int, default=15,
                   help="Frames per video (matches official limit_frames=15)")
    p.add_argument("--min_jf_clean", type=float, default=0.30)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point", choices=["point", "mask"],
                   help="Default 'point' to match UAP-SAM2 training prompt regime")
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--tag", default="uap1024_yt_test")
    p.add_argument("--save_dir", default="results_v100/uap_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sanity", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[eval-yt-1024] attack={args.attack}, prompt={args.prompt}, crf={args.crf}, "
          f"max_frames={args.max_frames}, max_videos={args.max_videos}, JPEG_Q={JPEG_QUALITY}")

    uap = None
    if args.attack == "uap":
        if not args.uap_path:
            raise ValueError("--uap_path required for --attack uap")
        uap_tensor = torch.load(args.uap_path, map_location="cpu", weights_only=False)
        if uap_tensor.dim() == 4:
            uap_tensor = uap_tensor[0]
        uap = uap_tensor.numpy()
        assert uap.shape == (3, 1024, 1024), f"UAP shape: {uap.shape}"
        assert np.abs(uap).max() <= 0.05, f"|uap|.max()={np.abs(uap).max():.4f}"
        print(f"[eval-yt-1024] UAP shape={uap.shape}, range=[{uap.min():.4f},{uap.max():.4f}], "
              f"abs_mean={np.abs(uap).mean():.4f}")
    elif args.attack == "random":
        print(f"[eval-yt-1024] random sign ±{args.eps}/255 (seed={args.seed})")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    if args.videos:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        videos = list_ytvos_videos(args.ytvos_root, split=args.anno_split)
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"[eval-yt-1024] {len(videos)} videos")

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_skip = 0
    t0 = time.time()

    for vname in videos:
        print(f"\n=== {vname} ===")
        try:
            frames, masks, _ = load_single_video_ytvos(
                args.ytvos_root, vname,
                split=args.split, anno_split=args.anno_split,
                max_frames=args.max_frames,
            )
        except Exception as e:
            print(f"  [skip] load error: {e}")
            n_skip += 1
            continue
        if len(frames) == 0 or len(masks) == 0:
            print(f"  [skip] no frames/masks")
            n_skip += 1
            continue

        frames_1024 = [resize_frame_to_1024(fr) for fr in frames]
        masks_1024 = [resize_mask_to_1024(m) for m in masks]

        try:
            _, jf_clean, j_clean, f_clean = run_tracking_1024(
                frames_1024, masks_1024, predictor, device, args.prompt)
        except Exception as e:
            print(f"  [skip] tracking error: {e}")
            n_skip += 1
            continue
        print(f"  clean@1024: JF={jf_clean:.4f}  J(=mIoU)={j_clean:.4f}  F={f_clean:.4f}")

        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean@1024={jf_clean:.3f} < {args.min_jf_clean}")
            n_skip += 1
            continue

        codec_frames = codec_round_trip_1024(frames_1024, args.ffmpeg_path, args.crf)
        if codec_frames is None:
            n_skip += 1
            continue
        assert len(codec_frames) == len(frames_1024)
        _, jf_codec_clean, j_codec_clean, _ = run_tracking_1024(
            codec_frames, masks_1024, predictor, device, args.prompt)
        print(f"  codec_clean@1024: JF={jf_codec_clean:.4f}  J={j_codec_clean:.4f}")

        if args.attack == "uap":
            adv_frames = apply_uap_1024(frames_1024, uap)
        elif args.attack == "random":
            adv_frames = apply_random_1024(frames_1024, args.eps, args.seed)
        else:
            adv_frames = apply_none(frames_1024)

        q = frame_quality_1024(frames_1024, adv_frames)
        mean_ssim = q["mean_ssim"]
        mean_psnr = q["mean_psnr"]
        mean_lpips = q.get("mean_lpips")
        lp_str = f"  LPIPS={mean_lpips:.4f}" if mean_lpips is not None else ""
        print(f"  quality@1024: SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.1f}dB{lp_str}")

        _, jf_adv, j_adv, f_adv = run_tracking_1024(
            adv_frames, masks_1024, predictor, device, args.prompt)
        delta_jf_adv = jf_clean - jf_adv
        delta_j_adv = j_clean - j_adv
        print(f"  adv (pre-codec)@1024: JF={jf_adv:.4f}  J={j_adv:.4f}  "
              f"ΔJF={delta_jf_adv:+.4f}  ΔmIoU={delta_j_adv:+.4f}")

        codec_adv = codec_round_trip_1024(adv_frames, args.ffmpeg_path, args.crf)
        if codec_adv is None:
            n_skip += 1
            continue
        assert len(codec_adv) == len(adv_frames)
        _, jf_codec_adv, j_codec_adv, _ = run_tracking_1024(
            codec_adv, masks_1024, predictor, device, args.prompt)
        delta_jf_codec = jf_codec_clean - jf_codec_adv
        delta_j_codec = j_codec_clean - j_codec_adv
        print(f"  adv (post-codec CRF{args.crf})@1024: JF={jf_codec_adv:.4f}  "
              f"J={j_codec_adv:.4f}  ΔJF={delta_jf_codec:+.4f}  ΔmIoU={delta_j_codec:+.4f}")

        results.append({
            "video": vname,
            "jf_clean_1024": jf_clean, "j_clean_1024": j_clean, "f_clean_1024": f_clean,
            "jf_codec_clean_1024": jf_codec_clean, "j_codec_clean_1024": j_codec_clean,
            "jf_adv_1024": jf_adv, "j_adv_1024": j_adv,
            "jf_codec_adv_1024": jf_codec_adv, "j_codec_adv_1024": j_codec_adv,
            "delta_jf_adv": delta_jf_adv, "delta_j_adv": delta_j_adv,
            "delta_jf_codec": delta_jf_codec, "delta_j_codec": delta_j_codec,
            "mean_ssim_1024": mean_ssim, "mean_psnr_1024": mean_psnr,
            "mean_lpips_1024": mean_lpips,
            "n_frames": len(frames),
        })

        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "dataset": "youtube_vos_1024",
                       "jpeg_quality": JPEG_QUALITY, "results": results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY MODE] Stopping after first valid video.")
            break

    elapsed = time.time() - t0
    print(f"\n[eval-yt-1024] elapsed: {elapsed:.1f}s")

    if results:
        deltas_pre = [r["delta_jf_adv"] for r in results]
        deltas_post = [r["delta_jf_codec"] for r in results]
        ssims = [r["mean_ssim_1024"] for r in results]
        mean_j_clean = float(np.mean([r["j_clean_1024"] for r in results]))
        mean_j_adv = float(np.mean([r["j_adv_1024"] for r in results]))
        mean_j_codec_clean = float(np.mean([r["j_codec_clean_1024"] for r in results]))
        mean_j_codec_adv = float(np.mean([r["j_codec_adv_1024"] for r in results]))
        delta_miou_pre = (mean_j_adv - mean_j_clean) * 100
        delta_miou_post = (mean_j_codec_adv - mean_j_codec_clean) * 100
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(results)} valid, {n_skip} skipped)  attack={args.attack}  prompt={args.prompt}")
        print(f"  ── J&F (deployment metric) ──")
        print(f"  mean pre-codec  ΔJF = {np.mean(deltas_pre)*100:+.2f}pp")
        print(f"  mean post-codec ΔJF = {np.mean(deltas_post)*100:+.2f}pp")
        print(f"  ── mIoU (= J, paper-comparable) ──")
        print(f"  mIoU clean       = {mean_j_clean*100:.2f}%")
        print(f"  mIoU adv         = {mean_j_adv*100:.2f}%   ΔmIoU = {delta_miou_pre:+.2f}pp")
        print(f"  mIoU codec_clean = {mean_j_codec_clean*100:.2f}%")
        print(f"  mIoU codec_adv   = {mean_j_codec_adv*100:.2f}%   ΔmIoU(codec) = {delta_miou_post:+.2f}pp")
        print(f"  ── reference: official UAP-SAM2 reproduction (their JPEG eval): ΔmIoU ≈ -24.22pp")
        print(f"  ── reference: published paper claim:                            ΔmIoU ≈ -45.77pp")
        print(f"{'='*60}")

        with open(out_dir / "summary.csv", "w") as f:
            f.write("video,j_clean,jf_clean,j_codec_clean,jf_codec_clean,j_adv,jf_adv,"
                    "j_codec_adv,jf_codec_adv,delta_jf_adv,delta_jf_codec,delta_j_adv,delta_j_codec,"
                    "mean_ssim,mean_psnr,mean_lpips,n_frames\n")
            for r in results:
                lp = r.get("mean_lpips_1024")
                lp_str = f"{lp:.4f}" if lp is not None else "nan"
                f.write(
                    f"{r['video']},{r['j_clean_1024']:.4f},{r['jf_clean_1024']:.4f},"
                    f"{r['j_codec_clean_1024']:.4f},{r['jf_codec_clean_1024']:.4f},"
                    f"{r['j_adv_1024']:.4f},{r['jf_adv_1024']:.4f},"
                    f"{r['j_codec_adv_1024']:.4f},{r['jf_codec_adv_1024']:.4f},"
                    f"{r['delta_jf_adv']:.4f},{r['delta_jf_codec']:.4f},"
                    f"{r['delta_j_adv']:.4f},{r['delta_j_codec']:.4f},"
                    f"{r['mean_ssim_1024']:.4f},{r['mean_psnr_1024']:.1f},{lp_str},{r['n_frames']}\n"
                )
        print(f"\nResults → {out_dir}/results.json")
    else:
        print("\n[warn] No valid videos evaluated.")


if __name__ == "__main__":
    main()
