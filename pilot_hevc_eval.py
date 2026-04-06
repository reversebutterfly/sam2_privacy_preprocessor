"""
pilot_hevc_eval.py — Evaluate combo_strong with H.265/HEVC codec.

Tests whether the codec-amplified boundary suppression effect is specific to
H.264 or generalises to H.265/HEVC (different DCT/transform coding).

Usage:
    python pilot_hevc_eval.py \\
        --videos all --max_frames 50 --crf 28 \\
        --tag hevc_combo_strong_v1 \\
        --save_dir results_v100/mask_guided

H.265 CRF note: quality is roughly equivalent to H.264 at CRF+5.
  CRF 23 (H.265) ≈ CRF 18 (H.264) — very high quality
  CRF 28 (H.265) ≈ CRF 23 (H.264) — standard quality  ← default
  CRF 33 (H.265) ≈ CRF 28 (H.264) — lower quality

Outputs:
    results_v100/mask_guided/<tag>/results.json
    results_v100/mask_guided/<tag>/summary.csv
"""

import argparse
import json
import os
import subprocess
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
from src.metrics import jf_score, mean_jf
# Reuse edit functions from pilot_mask_guided
from pilot_mask_guided import (
    apply_boundary_suppression, apply_combo,
    apply_edit_to_video, run_tracking, build_predictor,
    frame_quality, _save, _print_summary,
)


# ── HEVC encode/decode ─────────────────────────────────────────────────────────

def encode_decode_hevc(
    frames: List[np.ndarray],
    crf: int = 28,
    fps: int = 25,
    ffmpeg_path: str = "ffmpeg",
) -> List[np.ndarray]:
    """
    Encode a list of frames to H.265/HEVC with FFmpeg and decode back.
    Non-differentiable; used at evaluation time only.

    CRF 28 (HEVC) ≈ CRF 23 (H.264) in visual quality.
    """
    if not frames:
        return []

    H, W = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), bgr)

        in_pattern  = os.path.join(tmp, "%05d.png")
        out_video   = os.path.join(tmp, "encoded_hevc.mp4")
        out_pattern = os.path.join(tmp, "decoded_%05d.png")

        # Encode H.265
        cmd_enc = [
            ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", in_pattern,
            "-vcodec", "libx265",
            "-crf", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-x265-params", "log-level=0",
            out_video,
        ]
        try:
            subprocess.run(cmd_enc, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"HEVC encode failed: {e.stderr}") from e

        # Decode
        cmd_dec = [
            ffmpeg_path, "-y",
            "-i", out_video,
            out_pattern,
        ]
        try:
            subprocess.run(cmd_dec, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"HEVC decode failed: {e.stderr}") from e

        decoded = []
        idx = 1
        while True:
            path = os.path.join(tmp, f"decoded_{idx:05d}.png")
            if not os.path.exists(path):
                break
            bgr = cv2.imread(path)
            if bgr is None:
                break
            decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx += 1

    return decoded


def codec_round_trip_hevc(frames, ffmpeg_path, crf):
    try:
        return encode_decode_hevc(frames, crf=crf, ffmpeg_path=ffmpeg_path)
    except Exception as e:
        print(f"  [hevc] error: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="",
                   help="Comma-separated video names, 'all' = full DAVIS, empty = mini-val")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=28,
                   help="HEVC CRF (28≈H264 CRF23 in quality)")
    p.add_argument("--crf_h264", type=int, default=23,
                   help="H.264 CRF for comparison")
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--blend_alpha", type=float, default=0.8)
    p.add_argument("--edit_type", default="combo", choices=["idea1", "idea2", "combo"],
                   help="Edit type: idea1=boundary suppression, combo=idea1+halo")
    p.add_argument("--prompt", default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir", default="results_v100/mask_guided")
    p.add_argument("--tag", default="hevc_combo_strong_v1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sanity", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    _raw = [v.strip() for v in args.videos.split(",") if v.strip()]
    if _raw == ["all"]:
        img_dir = Path(args.davis_root) / "JPEGImages" / "480p"
        videos = sorted(d.name for d in img_dir.iterdir() if d.is_dir())
    else:
        from config import DAVIS_MINI_VAL
        videos = _raw or DAVIS_MINI_VAL

    params = {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}
    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hevc_eval] combo_strong rw={args.ring_width} alpha={args.blend_alpha}")
    print(f"[hevc_eval] HEVC CRF={args.crf}  H264 CRF={args.crf_h264}")
    print(f"[hevc_eval] {len(videos)} videos -> {out_dir}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    for vid in videos:
        print(f"\n=== {vid} ===")
        try:
            frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] load error: {e}")
            continue
        if not frames:
            print(f"  [skip] empty")
            continue

        # Clean
        _, jf_clean, j_clean, f_clean = run_tracking(frames, masks, predictor, device, args.prompt)
        print(f"  clean: JF={jf_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue

        # Codec-clean H.264
        from src.codec_eot import encode_decode_h264
        h264_clean = None
        try:
            h264_clean = encode_decode_h264(frames, crf=args.crf_h264, ffmpeg_path=args.ffmpeg_path)
        except Exception as e:
            print(f"  [h264] error: {e}")
        jf_h264_clean = float("nan")
        if h264_clean:
            _, jf_h264_clean, _, _ = run_tracking(h264_clean, masks, predictor, device, args.prompt)
            print(f"  h264_clean CRF{args.crf_h264}: JF={jf_h264_clean:.4f}")

        # Codec-clean HEVC
        hevc_clean = codec_round_trip_hevc(frames, args.ffmpeg_path, args.crf)
        jf_hevc_clean = float("nan")
        if hevc_clean:
            _, jf_hevc_clean, _, _ = run_tracking(hevc_clean, masks, predictor, device, args.prompt)
            print(f"  hevc_clean CRF{args.crf}: JF={jf_hevc_clean:.4f}")

        # Edit
        edited = apply_edit_to_video(frames, masks, args.edit_type, params)

        # Quality
        ssims = [frame_quality(fo, fe)[0] for fo, fe in zip(frames[:5], edited[:5])]
        mean_ssim = float(np.mean(ssims))

        # Pre-codec adv
        _, jf_adv, _, _ = run_tracking(edited, masks, predictor, device, args.prompt)
        print(f"  adv (pre-codec): JF={jf_adv:.4f}  ΔJF={jf_clean-jf_adv:+.4f}")

        # H.264 adv
        h264_adv = None
        try:
            h264_adv = encode_decode_h264(edited, crf=args.crf_h264, ffmpeg_path=args.ffmpeg_path)
        except Exception as e:
            print(f"  [h264_adv] error: {e}")
        jf_h264_adv = float("nan")
        delta_h264 = float("nan")
        if h264_adv:
            _, jf_h264_adv, _, _ = run_tracking(h264_adv, masks, predictor, device, args.prompt)
            delta_h264 = jf_h264_clean - jf_h264_adv
            print(f"  adv H.264 CRF{args.crf_h264}: JF={jf_h264_adv:.4f}  ΔJF={delta_h264:+.4f}")

        # HEVC adv
        hevc_adv = codec_round_trip_hevc(edited, args.ffmpeg_path, args.crf)
        jf_hevc_adv = float("nan")
        delta_hevc = float("nan")
        if hevc_adv:
            _, jf_hevc_adv, _, _ = run_tracking(hevc_adv, masks, predictor, device, args.prompt)
            delta_hevc = jf_hevc_clean - jf_hevc_adv
            print(f"  adv HEVC CRF{args.crf}: JF={jf_hevc_adv:.4f}  ΔJF={delta_hevc:+.4f}")

        row = {
            "video": vid,
            "n_frames": len(frames),
            "jf_clean": round(jf_clean, 4),
            "jf_h264_clean": round(jf_h264_clean, 4) if not np.isnan(jf_h264_clean) else None,
            "jf_hevc_clean": round(jf_hevc_clean, 4) if not np.isnan(jf_hevc_clean) else None,
            "jf_adv": round(jf_adv, 4),
            "jf_h264_adv": round(jf_h264_adv, 4) if not np.isnan(jf_h264_adv) else None,
            "jf_hevc_adv": round(jf_hevc_adv, 4) if not np.isnan(jf_hevc_adv) else None,
            "delta_jf_h264": round(delta_h264, 4) if not np.isnan(delta_h264) else None,
            "delta_jf_hevc": round(delta_hevc, 4) if not np.isnan(delta_hevc) else None,
            "mean_ssim": round(mean_ssim, 4),
            "crf_hevc": args.crf,
            "crf_h264": args.crf_h264,
        }
        all_results.append(row)

        # Save incrementally
        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY] Stopping after first video.")
            break

    # Final aggregate
    valid = [r for r in all_results
             if r["delta_jf_h264"] is not None and r["delta_jf_hevc"] is not None]
    if valid:
        mean_h264 = np.mean([r["delta_jf_h264"] for r in valid])
        mean_hevc = np.mean([r["delta_jf_hevc"] for r in valid])
        print(f"\n{'='*60}")
        print(f"[aggregate] n={len(valid)}")
        print(f"  H.264 CRF{args.crf_h264}: ΔJF_codec = {mean_h264*100:+.2f}pp")
        print(f"  HEVC  CRF{args.crf}:  ΔJF_codec = {mean_hevc*100:+.2f}pp")
        print(f"  Ratio HEVC/H264: {mean_hevc/mean_h264:.2f}x")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"[saved] {out_dir}/results.json")


if __name__ == "__main__":
    main()
