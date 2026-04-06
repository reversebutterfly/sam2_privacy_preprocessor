"""
pilot_adaptive_lite.py — Adaptive-lite attacker robustness test.

Tests whether simple counter-processing (unsharp mask, deblur, edge enhance)
applied BEFORE SAM2 can recover tracking performance on defended videos.

This determines the "privacy defense" vs "codec artifact" framing:
  - If ΔJF_counter < ΔJF_codec/2: method is ROBUST → "privacy defense" claim holds
  - If ΔJF_counter ≈ 0: counter-processing breaks the method → "codec artifact" framing only

Flow:
  1. Apply combo_strong edit → edited_frames
  2. Encode with H.264 → defended_codec_frames   (the "released" video)
  3. Apply adaptive-lite counter-processing → counter_frames
  4. Run SAM2 on counter_frames → jf_counter
  5. Compare: delta_counter = jf_codec_clean - jf_counter
     (positive = editing still degrades, negative = counter-processing recovered)

Counter-processing variants:
  A: unsharp_mild   — sigma=1.5, amount=1.0  (mild sharpening)
  B: unsharp_strong — sigma=2.0, amount=2.0  (aggressive sharpening)
  C: clahe          — histogram equalization for contrast boost
  D: edge_enhance   — Laplacian-based edge sharpening

Usage:
    python pilot_adaptive_lite.py \\
        --videos all --max_frames 50 --crf 23 \\
        --tag adaptive_lite_v1 \\
        --save_dir results_v100/mask_guided

Outputs:
    results_v100/mask_guided/<tag>/results.json
"""

import argparse
import json
import os
import sys
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
from src.metrics import jf_score, mean_jf
from pilot_mask_guided import (
    apply_edit_to_video, run_tracking, build_predictor, frame_quality,
)


# ── Counter-processing functions ───────────────────────────────────────────────

def unsharp_mask(frame: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: sharpen by subtracting blurred version."""
    blurred = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigma)
    sharpened = frame.astype(np.float32) + amount * (frame.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def clahe_enhance(frame: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast boost."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def edge_enhance(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Laplacian-based edge enhancement."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_3ch = np.stack([lap, lap, lap], axis=2)
    enhanced = frame.astype(np.float32) - strength * lap_3ch
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def apply_counter(frames: List[np.ndarray], mode: str) -> List[np.ndarray]:
    """Apply adaptive-lite counter-processing to a list of frames."""
    if mode == "unsharp_mild":
        return [unsharp_mask(f, sigma=1.5, amount=1.0) for f in frames]
    elif mode == "unsharp_strong":
        return [unsharp_mask(f, sigma=2.0, amount=2.0) for f in frames]
    elif mode == "clahe":
        return [clahe_enhance(f, clip_limit=2.0, tile=8) for f in frames]
    elif mode == "edge_enhance":
        return [edge_enhance(f, strength=0.5) for f in frames]
    elif mode == "combined":
        # unsharp + clahe
        return [clahe_enhance(unsharp_mask(f, sigma=1.5, amount=1.5)) for f in frames]
    else:
        raise ValueError(f"Unknown counter mode: {mode}")


def codec_round_trip(frames, ffmpeg_path, crf):
    try:
        return encode_decode_h264(frames, crf=crf, ffmpeg_path=ffmpeg_path)
    except Exception as e:
        print(f"  [codec] error: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="",
                   help="Comma-separated video names, 'all' = full DAVIS, empty = mini-val")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--blend_alpha", type=float, default=0.8)
    p.add_argument("--edit_type", default="combo", choices=["idea1", "idea2", "combo"],
                   help="Edit type: idea1=boundary suppression, combo=idea1+halo")
    p.add_argument("--prompt", default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    p.add_argument("--counter_modes", default="unsharp_mild,unsharp_strong,clahe,edge_enhance,combined",
                   help="Comma-separated list of counter-processing modes to test")
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir", default="results_v100/mask_guided")
    p.add_argument("--tag", default="adaptive_lite_v1")
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

    counter_modes = [m.strip() for m in args.counter_modes.split(",") if m.strip()]
    params = {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[adaptive_lite] combo_strong rw={args.ring_width} alpha={args.blend_alpha}")
    print(f"[adaptive_lite] counter modes: {counter_modes}")
    print(f"[adaptive_lite] CRF={args.crf}, {len(videos)} videos -> {out_dir}")

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

        # Clean baseline
        _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, args.prompt)
        print(f"  clean: JF={jf_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean < {args.min_jf_clean}")
            continue

        # Codec-clean baseline (what the downstream user receives, unedited)
        codec_clean = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        jf_codec_clean = float("nan")
        if codec_clean:
            _, jf_codec_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)
            print(f"  codec_clean CRF{args.crf}: JF={jf_codec_clean:.4f}")

        # Apply combo_strong edit
        edited = apply_edit_to_video(frames, masks, args.edit_type, params)
        ssims = [frame_quality(fo, fe)[0] for fo, fe in zip(frames[:5], edited[:5])]
        mean_ssim = float(np.mean(ssims))

        # Pre-codec adversarial
        _, jf_adv, _, _ = run_tracking(edited, masks, predictor, device, args.prompt)
        print(f"  adv (pre-codec): JF={jf_adv:.4f}  ΔJF={jf_clean-jf_adv:+.4f}")

        # Post-codec adversarial (the "released" defended video)
        codec_adv = codec_round_trip(edited, args.ffmpeg_path, args.crf)
        jf_codec_adv = float("nan")
        delta_codec = float("nan")
        if codec_adv:
            _, jf_codec_adv, _, _ = run_tracking(codec_adv, masks, predictor, device, args.prompt)
            delta_codec = jf_codec_clean - jf_codec_adv
            print(f"  adv (post-codec): JF={jf_codec_adv:.4f}  ΔJF_codec={delta_codec:+.4f}")

        # Now apply each counter-processing mode to the defended codec video
        counter_results = {}
        if codec_adv:
            for mode in counter_modes:
                try:
                    counter_frames = apply_counter(codec_adv, mode)
                    _, jf_counter, _, _ = run_tracking(counter_frames, masks, predictor, device, args.prompt)
                    # Residual effect after counter-processing
                    # positive = editing still degrades tracker despite counter-processing
                    delta_counter = jf_codec_clean - jf_counter
                    # Recovery fraction: how much did counter-processing recover?
                    recovery_frac = 1.0 - (delta_counter / delta_codec) if not np.isnan(delta_codec) and abs(delta_codec) > 0.001 else float("nan")
                    counter_results[mode] = {
                        "jf_counter": round(jf_counter, 4),
                        "delta_counter": round(delta_counter, 4),
                        "recovery_frac": round(recovery_frac, 3) if not np.isnan(recovery_frac) else None,
                    }
                    print(f"  [{mode}]: JF={jf_counter:.4f}  ΔJF={delta_counter:+.4f}  "
                          f"recovery={recovery_frac:.1%}" if not np.isnan(recovery_frac) else
                          f"  [{mode}]: JF={jf_counter:.4f}  ΔJF={delta_counter:+.4f}")
                except Exception as e:
                    print(f"  [{mode}] error: {e}")
                    counter_results[mode] = {"error": str(e)}

        row = {
            "video": vid,
            "n_frames": len(frames),
            "jf_clean": round(jf_clean, 4),
            "jf_codec_clean": round(jf_codec_clean, 4) if not np.isnan(jf_codec_clean) else None,
            "jf_adv": round(jf_adv, 4),
            "jf_codec_adv": round(jf_codec_adv, 4) if not np.isnan(jf_codec_adv) else None,
            "delta_jf_codec": round(delta_codec, 4) if not np.isnan(delta_codec) else None,
            "mean_ssim": round(mean_ssim, 4),
            "counter_results": counter_results,
        }
        all_results.append(row)

        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY] Stopping after first video.")
            break

    # Aggregate summary
    valid = [r for r in all_results if r["delta_jf_codec"] is not None]
    if valid:
        mean_codec = np.mean([r["delta_jf_codec"] for r in valid])
        print(f"\n{'='*70}")
        print(f"[aggregate] n={len(valid)}  ΔJF_codec={mean_codec*100:+.2f}pp")
        for mode in counter_modes:
            deltas = [r["counter_results"].get(mode, {}).get("delta_counter")
                      for r in valid if r["counter_results"].get(mode, {}).get("delta_counter") is not None]
            recoveries = [r["counter_results"].get(mode, {}).get("recovery_frac")
                          for r in valid if r["counter_results"].get(mode, {}).get("recovery_frac") is not None]
            if deltas:
                md = np.mean(deltas)
                mr = np.mean(recoveries) if recoveries else float("nan")
                print(f"  [{mode}]: ΔJF={md*100:+.2f}pp  recovery={mr:.1%}" if not np.isnan(mr) else
                      f"  [{mode}]: ΔJF={md*100:+.2f}pp")
                if md > 0.005:
                    print(f"    → ROBUST: {md*100:.1f}pp degradation survives {mode}")
                else:
                    print(f"    → BROKEN: counter-processing neutralizes the effect")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"[saved] {out_dir}/results.json")


if __name__ == "__main__":
    main()
