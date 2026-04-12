"""
eval_uap_ytvos.py — Evaluate UAP-SAM2 perturbation on YouTube-VOS using J&F + codec.

Applies the pre-trained UAP (YOUTUBE.pth, L_inf=10/255) to YT-VOS 2019 frames,
runs SAM2.1 tracking, and reports J&F pre-codec and post-H.264 CRF=23.

Protocol matches our main AdvOpt experiments:
  - GT mask prompt (first-frame oracle)
  - H.264 CRF=23
  - max_frames=50
  - min_jf_clean=0.30

Usage:
  # Sanity (1 video):
  python eval_uap_ytvos.py --uap_path .../YOUTUBE.pth --sanity

  # Full YT-VOS sweep:
  python eval_uap_ytvos.py --uap_path .../YOUTUBE.pth --tag uap_ytvos_full

Outputs:
  results_v100/uap_eval/<tag>/results.json
  results_v100/uap_eval/<tag>/summary.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video_ytvos, list_ytvos_videos
from pilot_mask_guided import run_tracking, frame_quality, build_predictor, codec_round_trip

DEFAULT_YTVOS_ROOT = os.path.join(ROOT, "data", "youtube_vos")


def apply_uap(frames: List[np.ndarray], uap: np.ndarray) -> List[np.ndarray]:
    """Apply UAP perturbation to RGB uint8 frames.

    uap: float32 numpy [3, H, W] in [-eps, +eps] normalised [0,1] space.
    Resizes to frame resolution if needed.
    """
    out = []
    for fr in frames:
        H, W = fr.shape[:2]
        uap_hw = uap
        if uap_hw.shape[1] != H or uap_hw.shape[2] != W:
            uap_resized = cv2.resize(
                uap_hw.transpose(1, 2, 0),
                (W, H),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)
            uap_hw = uap_resized
        fr_f = fr.astype(np.float32) / 255.0
        adv_f = np.clip(fr_f + uap_hw.transpose(1, 2, 0), 0.0, 1.0)
        out.append((adv_f * 255.0).astype(np.uint8))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uap_path",      required=True,
                   help="Path to UAP .pth file ([1,3,H,W] or [3,H,W] float32 in [-eps,eps])")
    p.add_argument("--ytvos_root",    default=DEFAULT_YTVOS_ROOT)
    p.add_argument("--split",         default="valid")
    p.add_argument("--anno_split",    default="valid")
    p.add_argument("--videos",        default="",
                   help="Comma-separated video IDs (empty = all annotated)")
    p.add_argument("--max_videos",    type=int, default=0,
                   help="Cap on number of videos (0 = no cap)")
    p.add_argument("--max_frames",    type=int, default=50)
    p.add_argument("--crf",           type=int, default=23)
    p.add_argument("--prompt",        default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean",  type=float, default=0.30)
    p.add_argument("--checkpoint",    default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",   default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",   default=FFMPEG_PATH)
    p.add_argument("--save_dir",      default="results_v100/uap_eval")
    p.add_argument("--tag",           default="uap_ytvos_full")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--sanity",        action="store_true",
                   help="Stop after first valid video")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load UAP
    uap_tensor = torch.load(args.uap_path, map_location="cpu", weights_only=False)
    if uap_tensor.dim() == 4:
        uap_tensor = uap_tensor[0]  # [3, H, W]
    uap = uap_tensor.numpy()  # float32, [-eps, +eps] in [0,1] space
    print(f"[eval] UAP shape={uap.shape}, range=[{uap.min():.4f},{uap.max():.4f}]")

    # Build SAM2
    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    # Video list
    if args.videos:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        videos = list_ytvos_videos(
            args.ytvos_root,
            split=args.anno_split,
        )
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"[eval] {len(videos)} videos, max_frames={args.max_frames}, "
          f"CRF={args.crf}, prompt={args.prompt}")

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_skip = 0

    for vname in videos:
        print(f"\n=== {vname} ===")
        try:
            frames, masks, _ = load_single_video_ytvos(
                args.ytvos_root,
                vname,
                split=args.split,
                anno_split=args.anno_split,
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

        # Clean JF
        try:
            _, jf_clean, j_clean, f_clean = run_tracking(
                frames, masks, predictor, device, args.prompt)
        except Exception as e:
            print(f"  [skip] tracking error: {e}")
            n_skip += 1
            continue

        print(f"  clean: JF={jf_clean:.4f}  J={j_clean:.4f}  F={f_clean:.4f}")

        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            n_skip += 1
            continue

        # Codec-clean JF
        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        _, jf_codec_clean, _, _ = run_tracking(
            codec_frames, masks, predictor, device, args.prompt)
        print(f"  codec_clean: JF={jf_codec_clean:.4f}")

        # Quality (SSIM/PSNR of UAP vs original)
        adv_frames = apply_uap(frames, uap)
        ssims, psnrs = [], []
        for orig, adv in zip(frames, adv_frames):
            s, ps = frame_quality(orig, adv)
            ssims.append(s)
            psnrs.append(ps)
        mean_ssim = float(np.mean(ssims))
        mean_psnr = float(np.mean(psnrs))
        print(f"  quality: SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.1f}dB")

        # Pre-codec adversarial JF
        _, jf_adv, j_adv, f_adv = run_tracking(
            adv_frames, masks, predictor, device, args.prompt)
        delta_jf_adv = jf_codec_clean - jf_adv
        print(f"  adv (pre-codec): JF={jf_adv:.4f}  ΔJF={delta_jf_adv:+.4f}")

        # Post-codec adversarial JF
        codec_adv = codec_round_trip(adv_frames, args.ffmpeg_path, args.crf)
        _, jf_codec_adv, _, _ = run_tracking(
            codec_adv, masks, predictor, device, args.prompt)
        delta_jf_codec = jf_codec_clean - jf_codec_adv
        print(f"  adv (post-codec CRF{args.crf}): JF={jf_codec_adv:.4f}  ΔJF={delta_jf_codec:+.4f}")

        rec = {
            "video": vname,
            "jf_clean": jf_clean,
            "j_clean": j_clean,
            "f_clean": f_clean,
            "jf_codec_clean": jf_codec_clean,
            "jf_adv": jf_adv,
            "delta_jf_adv": delta_jf_adv,
            "jf_codec_adv": jf_codec_adv,
            "delta_jf_codec": delta_jf_codec,
            "mean_ssim": mean_ssim,
            "mean_psnr": mean_psnr,
            "n_frames": len(frames),
        }
        results.append(rec)

        # Save incrementally
        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "dataset": "youtube_vos",
                       "results": results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY MODE] Stopping after first valid video.")
            break

    # Summary
    if results:
        deltas = [r["delta_jf_codec"] for r in results]
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        mean_ssim_all = float(np.mean([r["mean_ssim"] for r in results]))
        n5 = sum(1 for d in deltas if d >= 0.05)
        n8 = sum(1 for d in deltas if d >= 0.08)
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(results)} valid videos, {n_skip} skipped)")
        print(f"  mean  post-codec ΔJF = {mean_delta:+.4f} ({mean_delta*100:+.1f}pp)")
        print(f"  median post-codec ΔJF = {median_delta:+.4f}")
        print(f"  videos ≥ 5pp:  {n5}/{len(results)}")
        print(f"  videos ≥ 8pp:  {n8}/{len(results)}")
        print(f"  mean SSIM:     {mean_ssim_all:.4f}")
        print(f"{'='*60}")

        with open(out_dir / "summary.csv", "w") as f:
            f.write("video,jf_clean,jf_codec_clean,jf_adv,jf_codec_adv,"
                    "delta_jf_adv,delta_jf_codec,mean_ssim,mean_psnr,n_frames\n")
            for r in results:
                f.write(
                    f"{r['video']},{r['jf_clean']:.4f},{r['jf_codec_clean']:.4f},"
                    f"{r['jf_adv']:.4f},{r['jf_codec_adv']:.4f},"
                    f"{r['delta_jf_adv']:.4f},{r['delta_jf_codec']:.4f},"
                    f"{r['mean_ssim']:.4f},{r['mean_psnr']:.1f},{r['n_frames']}\n"
                )
        print(f"\nResults → {out_dir}/results.json")
    else:
        print("\n[warn] No valid videos evaluated.")


if __name__ == "__main__":
    main()
