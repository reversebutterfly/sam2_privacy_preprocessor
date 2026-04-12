"""
eval_uap_davis_1024.py — Corrected UAP-SAM2 baseline eval at native 1024x1024 protocol.

Fixes the resolution-mismatch bug in eval_uap_davis.py:
  - Old (broken): downsample 1024 UAP -> 480p with cv2.INTER_LINEAR, apply at 480p,
    then SAM2 internally re-resizes 480p -> 1024 (double resampling kills high-freq pattern).
  - New (correct): square-resize each frame 854x480 -> 1024x1024 (matches SAM2Transforms),
    apply UAP at 1024x1024, save 1024x1024 JPEG QF=95 (matches official UAP-SAM2 protocol).

Also includes:
  - Random sign ±10/255 positive control (--attack random)
  - Zero perturbation reference (--attack none)
  - Explicit JPEG quality flag (no implicit OpenCV default)
  - Nearest-neighbor mask resize (never bilinear)

Usage:
  # Sanity (1 video, 5 frames):
  python eval_uap_davis_1024.py --uap_path .../YOUTUBE.pth \
      --videos bear --max_frames 5 --sanity --tag uap_1024_sanity

  # Full DAVIS:
  python eval_uap_davis_1024.py --uap_path .../YOUTUBE.pth \
      --max_frames 50 --tag uap_1024_full

  # Random control (full DAVIS):
  python eval_uap_davis_1024.py --attack random --eps 10 \
      --max_frames 50 --tag random_pm10_1024_full

Outputs:
  results_v100/uap_eval/<tag>/results.json
  results_v100/uap_eval/<tag>/summary.csv
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264
from src.metrics import jf_score, mean_jf, quality_summary
from pilot_mask_guided import build_predictor

JPEG_QUALITY = 95  # explicit, matches UAP-SAM2 official default


# ── Resolution helpers ─────────────────────────────────────────────────────────

def resize_frame_to_1024(frame_rgb: np.ndarray) -> np.ndarray:
    """Square bilinear resize to 1024x1024 — matches SAM2Transforms.Resize((1024,1024))."""
    return cv2.resize(frame_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)


def resize_mask_to_1024(mask: np.ndarray) -> np.ndarray:
    """Square nearest-neighbor resize to 1024x1024 — preserves binary GT mask."""
    m = mask.astype(np.uint8) if mask.dtype == bool else mask
    return cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)


# ── Attack functions ──────────────────────────────────────────────────────────

def apply_uap_1024(frames_1024: List[np.ndarray], uap: np.ndarray) -> List[np.ndarray]:
    """Apply UAP at native 1024x1024 resolution.

    frames_1024: list of [1024, 1024, 3] uint8 RGB
    uap:         [3, 1024, 1024] float32 in [-eps, +eps] (in [0,1] space)
    """
    assert uap.shape == (3, 1024, 1024), f"UAP must be [3,1024,1024], got {uap.shape}"
    delta = uap.transpose(1, 2, 0)  # [1024, 1024, 3]
    out = []
    for fr in frames_1024:
        assert fr.shape == (1024, 1024, 3), f"Frame must be [1024,1024,3], got {fr.shape}"
        fr_f = fr.astype(np.float32) / 255.0
        adv_f = np.clip(fr_f + delta, 0.0, 1.0)
        out.append((adv_f * 255.0).astype(np.uint8))
    return out


def apply_random_1024(frames_1024: List[np.ndarray], eps_int: int, seed: int) -> List[np.ndarray]:
    """Random sign ±eps/255 perturbation at 1024x1024 (positive control).

    Uses ONE fixed pattern across all frames (universal), to be comparable to UAP.
    """
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], size=(1024, 1024, 3)).astype(np.float32)
    delta = sign * (eps_int / 255.0)  # [-eps/255, +eps/255]
    out = []
    for fr in frames_1024:
        fr_f = fr.astype(np.float32) / 255.0
        adv_f = np.clip(fr_f + delta, 0.0, 1.0)
        out.append((adv_f * 255.0).astype(np.uint8))
    return out


def apply_none(frames_1024: List[np.ndarray], **kw) -> List[np.ndarray]:
    """No perturbation — clean reference."""
    return [fr.copy() for fr in frames_1024]


# ── Tracking at 1024x1024 ─────────────────────────────────────────────────────

def run_tracking_1024(
    frames_1024: List[np.ndarray],
    masks_1024: List[np.ndarray],
    predictor,
    device: torch.device,
    prompt: str = "mask",
) -> Tuple[List[np.ndarray], float, float, float]:
    """Run SAM2 video tracking on 1024x1024 frames + masks. JPEG QF=95 ingress."""
    H, W = 1024, 1024
    gt_bool = [m.astype(bool) for m in masks_1024]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fr in enumerate(frames_1024):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(tmp_dir, f"{i:05d}.jpg"),
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)
            first_mask = masks_1024[0].astype(bool)
            ys, xs = np.where(first_mask)

            if prompt == "mask" and len(ys) > 0:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=first_mask)
            elif len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=np.array([[cx, cy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=first_mask)

            pred = [None] * len(frames_1024)
            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                if 1 in obj_ids:
                    idx = list(obj_ids).index(1)
                    pred[fi] = (logits[idx, 0] > 0.0).cpu().numpy()
                else:
                    pred[fi] = np.zeros((H, W), dtype=bool)

    for i in range(len(pred)):
        if pred[i] is None:
            pred[i] = np.zeros((H, W), dtype=bool)

    mjf, mj, mf = mean_jf(pred, gt_bool)
    return pred, mjf, mj, mf


def codec_round_trip_1024(frames_1024, ffmpeg_path, crf):
    try:
        return encode_decode_h264(frames_1024, ffmpeg_path=ffmpeg_path, crf=crf)
    except Exception as e:
        print(f"  [codec] error: {e}")
        return None


# ── Quality metrics at 1024 (delegates to project-standard quality_summary) ──

def frame_quality_1024(frames_1024_clean, frames_1024_adv) -> dict:
    """Project-standard color SSIM + PSNR (+ LPIPS if available) at 1024x1024.

    Returns dict with keys: mean_ssim, mean_psnr, mean_lpips.
    """
    return quality_summary(frames_1024_clean, frames_1024_adv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="uap", choices=["uap", "random", "none"],
                   help="uap = released UAP weights; random = ±eps sign control; none = clean reference")
    p.add_argument("--uap_path", default="",
                   help="Path to UAP .pth file (required for --attack uap)")
    p.add_argument("--eps", type=int, default=10,
                   help="±eps/255 for --attack random (default 10, matches UAP-SAM2)")
    p.add_argument("--seed", type=int, default=30,
                   help="Random seed for --attack random (matches UAP-SAM2 official seed)")
    p.add_argument("--videos", default="",
                   help="Comma-separated video names; empty = all DAVIS videos")
    p.add_argument("--max_videos", type=int, default=-1,
                   help="Max number of videos to evaluate (-1 = all)")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--min_jf_clean", type=float, default=0.30)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="mask", choices=["point", "mask"])
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--tag", default="uap_1024_test")
    p.add_argument("--save_dir", default="results_v100/uap_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sanity", action="store_true",
                   help="Stop after first valid video")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── Load attack ──
    print(f"[eval-1024] attack={args.attack}, prompt={args.prompt}, crf={args.crf}, "
          f"max_frames={args.max_frames}, JPEG_Q={JPEG_QUALITY}")

    uap = None
    if args.attack == "uap":
        if not args.uap_path:
            raise ValueError("--uap_path required for --attack uap")
        uap_tensor = torch.load(args.uap_path, map_location="cpu", weights_only=False)
        if uap_tensor.dim() == 4:
            uap_tensor = uap_tensor[0]
        uap = uap_tensor.numpy()
        assert uap.shape == (3, 1024, 1024), f"UAP shape must be [3,1024,1024], got {uap.shape}"
        assert np.abs(uap).max() <= 0.05, (
            f"UAP value range too large: |uap|.max()={np.abs(uap).max():.4f}; "
            f"expected ≤ 0.05 (10/255 in [0,1] space)"
        )
        print(f"[eval-1024] UAP shape={uap.shape}, range=[{uap.min():.4f},{uap.max():.4f}], "
              f"abs_mean={np.abs(uap).mean():.4f}")
    elif args.attack == "random":
        print(f"[eval-1024] random sign ±{args.eps}/255 (seed={args.seed}, universal pattern)")
    else:
        print("[eval-1024] no attack — clean reference run")

    # ── SAM2 ──
    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    # ── Video list ──
    davis_img = Path(args.davis_root) / "JPEGImages" / "480p"
    if args.videos:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        videos = sorted(d.name for d in davis_img.iterdir() if d.is_dir())
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"[eval-1024] {len(videos)} videos")

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_skip = 0
    t0 = time.time()

    for vname in videos:
        print(f"\n=== {vname} ===")
        try:
            frames, masks, _ = load_single_video(args.davis_root, vname, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] load error: {e}")
            n_skip += 1
            continue
        if len(frames) == 0:
            print(f"  [skip] no frames")
            n_skip += 1
            continue

        # Resize to 1024x1024
        frames_1024 = [resize_frame_to_1024(fr) for fr in frames]
        masks_1024 = [resize_mask_to_1024(m) for m in masks]

        # Clean JF at 1024
        try:
            _, jf_clean, j_clean, f_clean = run_tracking_1024(
                frames_1024, masks_1024, predictor, device, args.prompt)
        except Exception as e:
            print(f"  [skip] tracking error: {e}")
            n_skip += 1
            continue
        print(f"  clean@1024: JF={jf_clean:.4f}  J={j_clean:.4f}  F={f_clean:.4f}")

        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean@1024={jf_clean:.3f} < {args.min_jf_clean}")
            n_skip += 1
            continue

        # Codec-clean JF at 1024
        codec_frames = codec_round_trip_1024(frames_1024, args.ffmpeg_path, args.crf)
        if codec_frames is None:
            n_skip += 1
            continue
        assert len(codec_frames) == len(frames_1024), (
            f"codec dropped frames: {len(codec_frames)} vs {len(frames_1024)}"
        )
        _, jf_codec_clean, j_codec_clean_saved, f_codec_clean = run_tracking_1024(
            codec_frames, masks_1024, predictor, device, args.prompt)
        print(f"  codec_clean@1024: JF={jf_codec_clean:.4f}  J={j_codec_clean_saved:.4f}")

        # Apply attack
        if args.attack == "uap":
            adv_frames = apply_uap_1024(frames_1024, uap)
        elif args.attack == "random":
            adv_frames = apply_random_1024(frames_1024, args.eps, args.seed)
        else:
            adv_frames = apply_none(frames_1024)

        # Quality (project-standard color SSIM + PSNR + LPIPS)
        q = frame_quality_1024(frames_1024, adv_frames)
        mean_ssim = q["mean_ssim"]
        mean_psnr = q["mean_psnr"]
        mean_lpips = q.get("mean_lpips")
        lp_str = f"  LPIPS={mean_lpips:.4f}" if mean_lpips is not None else ""
        print(f"  quality@1024: SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.1f}dB{lp_str}")

        # Pre-codec adv JF (true pre-codec: subtracted from raw clean, NOT codec_clean)
        _, jf_adv, j_adv, f_adv = run_tracking_1024(
            adv_frames, masks_1024, predictor, device, args.prompt)
        delta_jf_adv = jf_clean - jf_adv  # FIXED: was jf_codec_clean - jf_adv (mixed in codec drop)
        delta_j_adv = j_clean - j_adv     # mIoU drop (= UAP-SAM2 paper metric)
        print(f"  adv (pre-codec)@1024: JF={jf_adv:.4f}  ΔJF={delta_jf_adv:+.4f}  "
              f"ΔmIoU={delta_j_adv:+.4f}")

        # Post-codec adv JF (post-codec: subtracted from codec_clean, deployment-conditioned)
        codec_adv = codec_round_trip_1024(adv_frames, args.ffmpeg_path, args.crf)
        if codec_adv is None:
            n_skip += 1
            continue
        assert len(codec_adv) == len(adv_frames), (
            f"codec dropped adv frames: {len(codec_adv)} vs {len(adv_frames)}"
        )
        _, jf_codec_adv, j_codec_adv, _ = run_tracking_1024(
            codec_adv, masks_1024, predictor, device, args.prompt)
        delta_jf_codec = jf_codec_clean - jf_codec_adv
        # For codec ΔmIoU: need j_codec_clean too — re-extract from earlier call
        # (we stored only jf_codec_clean; recompute j_codec_clean here)
        # Actually run_tracking_1024 already returned j; we just didn't save it. Re-run is wasteful.
        # Workaround: store j_codec_clean from the codec_clean run by refactoring above.
        delta_j_codec = j_codec_clean_saved - j_codec_adv
        print(f"  adv (post-codec CRF{args.crf})@1024: JF={jf_codec_adv:.4f}  ΔJF={delta_jf_codec:+.4f}  "
              f"ΔmIoU={delta_j_codec:+.4f}")

        rec = {
            "video": vname,
            "jf_clean_1024": jf_clean,
            "j_clean_1024": j_clean,
            "f_clean_1024": f_clean,
            "jf_codec_clean_1024": jf_codec_clean,
            "j_codec_clean_1024": j_codec_clean_saved,
            "jf_adv_1024": jf_adv,
            "j_adv_1024": j_adv,
            "delta_jf_adv": delta_jf_adv,
            "delta_j_adv": delta_j_adv,
            "jf_codec_adv_1024": jf_codec_adv,
            "j_codec_adv_1024": j_codec_adv,
            "delta_jf_codec": delta_jf_codec,
            "delta_j_codec": delta_j_codec,
            "mean_ssim_1024": mean_ssim,
            "mean_psnr_1024": mean_psnr,
            "mean_lpips_1024": mean_lpips,
            "n_frames": len(frames),
        }
        results.append(rec)

        # Save incrementally
        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "dataset": "davis_1024",
                       "jpeg_quality": JPEG_QUALITY, "results": results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY MODE] Stopping after first valid video.")
            break

    elapsed = time.time() - t0
    print(f"\n[eval-1024] elapsed: {elapsed:.1f}s")

    # ── Summary ──
    if results:
        deltas_pre = [r["delta_jf_adv"] for r in results]
        deltas_post = [r["delta_jf_codec"] for r in results]
        deltas_j_pre = [r["delta_j_adv"] for r in results]
        deltas_j_post = [r["delta_j_codec"] for r in results]
        ssims = [r["mean_ssim_1024"] for r in results]
        # Aggregate mIoU (= mean J across all videos): paper-comparable metric
        mean_j_clean = float(np.mean([r["j_clean_1024"] for r in results]))
        mean_j_adv = float(np.mean([r["j_adv_1024"] for r in results]))
        mean_j_codec_clean = float(np.mean([r["j_codec_clean_1024"] for r in results]))
        mean_j_codec_adv = float(np.mean([r["j_codec_adv_1024"] for r in results]))
        n5 = sum(1 for d in deltas_post if d >= 0.05)
        n8 = sum(1 for d in deltas_post if d >= 0.08)
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(results)} valid, {n_skip} skipped)  attack={args.attack}")
        print(f"  mean pre-codec  ΔJF = {np.mean(deltas_pre)*100:+.2f}pp")
        print(f"  mean post-codec ΔJF = {np.mean(deltas_post)*100:+.2f}pp")
        print(f"  median post-codec ΔJF = {np.median(deltas_post)*100:+.2f}pp")
        print(f"  videos ≥ 5pp: {n5}/{len(results)}")
        print(f"  videos ≥ 8pp: {n8}/{len(results)}")
        print(f"  mean SSIM@1024: {np.mean(ssims):.4f}")
        print(f"  ── mIoU (paper-comparable) ──")
        print(f"  mIoU clean       = {mean_j_clean*100:.2f}%")
        print(f"  mIoU adv         = {mean_j_adv*100:.2f}%   ΔmIoU = {(mean_j_adv-mean_j_clean)*100:+.2f}pp")
        print(f"  mIoU codec_clean = {mean_j_codec_clean*100:.2f}%")
        print(f"  mIoU codec_adv   = {mean_j_codec_adv*100:.2f}%   ΔmIoU(codec) = {(mean_j_codec_adv-mean_j_codec_clean)*100:+.2f}pp")
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
