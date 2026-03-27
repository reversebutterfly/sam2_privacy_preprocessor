"""
Pilot: Semantic Sweep — can non-pixel-noise edits on a single frame
(t=ATTACK_FRAME) break SAM2 tracking on future frames?

Perturbation types:
  brightness   — multiply pixel values by scale factor
  blur         — Gaussian blur (kernel size)
  color_shift  — fixed per-channel RGB offset
  hue_shift    — HSV hue rotation (degrees)
  saturation   — HSV saturation scale
  freq_mask    — keep only low-frequency DCT components + random phase on
                 high-freq portion (codec-surviving edit)
  jpeg         — JPEG re-compress at low quality then decode back

Only frame t=ATTACK_FRAME is modified; all other frames are left clean.
Results are split into:
  jf_all      — mean J&F over frames 1..T
  jf_attacked — J&F on frame ATTACK_FRAME only
  jf_future   — mean J&F on frames ATTACK_FRAME+1..T

Usage:
  python pilot_semantic_sweep.py \
      --videos bike-packing,blackswan,car-roundabout,dog-agility,elephant,flamingo \
      --attack_frame 2 \
      --max_frames 50 \
      --crf 23 \
      --tag pilot_v1 \
      --save_dir results_v100/semantic_sweep

Outputs:
  results_v100/semantic_sweep/<tag>/results.json
  results_v100/semantic_sweep/<tag>/summary.txt
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264, tensor_to_frames, frames_to_tensor
from src.metrics import jf_score, mean_jf


# ── Perturbation definitions ──────────────────────────────────────────────────

PERTURBATION_GRID = [
    # (name, ptype, strength)
    # Brightness boost
    ("bright_1.3",   "brightness", 1.3),
    ("bright_1.5",   "brightness", 1.5),
    ("bright_2.0",   "brightness", 2.0),
    # Gaussian blur (kernel size = 2*k+1)
    ("blur_k5",      "blur", 5),
    ("blur_k11",     "blur", 11),
    ("blur_k21",     "blur", 21),
    # Per-channel RGB shift (strength = max absolute shift / 255)
    ("color_s30",    "color_shift", 30),
    ("color_s60",    "color_shift", 60),
    ("color_s100",   "color_shift", 100),
    # Hue rotation in degrees
    ("hue_30",       "hue_shift", 30),
    ("hue_60",       "hue_shift", 60),
    ("hue_90",       "hue_shift", 90),
    # Saturation scale
    ("sat_0.3",      "saturation", 0.3),
    ("sat_0.0",      "saturation", 0.0),
    # JPEG re-compression
    ("jpeg_q15",     "jpeg", 15),
    ("jpeg_q5",      "jpeg", 5),
    ("jpeg_q2",      "jpeg", 2),
    # Low-frequency DCT mask + high-freq phase randomisation
    ("freq_k8",      "freq_mask", 8),
    ("freq_k16",     "freq_mask", 16),
    ("freq_k32",     "freq_mask", 32),
]


def perturb_frame(frame_np: np.ndarray, ptype: str, strength) -> np.ndarray:
    """Apply one perturbation to a uint8 HxWx3 frame."""
    f = frame_np.astype(np.float32)

    if ptype == "brightness":
        return np.clip(f * float(strength), 0, 255).astype(np.uint8)

    elif ptype == "blur":
        k = int(strength)
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(frame_np, (k, k), 0)

    elif ptype == "color_shift":
        # deterministic fixed offsets scaled by strength
        offsets = np.array([float(strength), -float(strength) * 0.6, float(strength) * 0.4])
        return np.clip(f + offsets[None, None, :], 0, 255).astype(np.uint8)

    elif ptype == "hue_shift":
        hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(strength) // 2) % 180  # OpenCV hue 0-179
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    elif ptype == "saturation":
        hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(strength), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    elif ptype == "jpeg":
        quality = int(strength)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif ptype == "freq_mask":
        # Keep DCT components within a top-left k×k block; zero rest
        k = int(strength)
        H, W, C = f.shape
        out = np.zeros_like(f)
        for c in range(C):
            dct_ch = cv2.dct(f[:, :, c])
            mask = np.zeros_like(dct_ch)
            mask[:k, :k] = 1.0
            out[:, :, c] = cv2.idct(dct_ch * mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown ptype: {ptype}")


# ── SAM2 tracking helper ──────────────────────────────────────────────────────

def build_sam2_predictor(checkpoint: str, config: str, device: torch.device):
    from sam2.build_sam import build_sam2_video_predictor
    pred = build_sam2_video_predictor(config, checkpoint, device=device)
    pred.eval()
    return pred


def run_tracking(
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    predictor,
    device: torch.device,
) -> List[np.ndarray]:
    """
    Run VideoPredictor with GT first-frame prompt.
    Returns list of predicted binary masks (one per frame).
    """
    H, W = frames_uint8[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fr in enumerate(frames_uint8):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)

            first_mask = masks_uint8[0].astype(bool)
            ys, xs = np.where(first_mask)
            if len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=np.array([[cx, cy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=first_mask)

            pred = [None] * len(frames_uint8)
            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                if 1 in obj_ids:
                    idx = list(obj_ids).index(1)
                    pred[fi] = (logits[idx, 0] > 0.0).cpu().numpy()
                else:
                    pred[fi] = np.zeros((H, W), dtype=bool)

    for i in range(len(pred)):
        if pred[i] is None:
            pred[i] = np.zeros((H, W), dtype=bool)
    return pred


def compute_jf_splits(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    attack_frame: int,
) -> Dict[str, float]:
    """
    Compute JF on three subsets:
      all:      frames 1..T (skip init frame 0)
      attacked: frame ATTACK_FRAME
      future:   frames ATTACK_FRAME+1..T
    """
    T = len(pred_masks)
    gt_bool = [m.astype(bool) for m in gt_masks]

    jf_all, _, _ = mean_jf(pred_masks[1:], gt_bool[1:])

    if attack_frame < T:
        jf_att, _, _ = mean_jf([pred_masks[attack_frame]], [gt_bool[attack_frame]])
    else:
        jf_att = float("nan")

    future_start = attack_frame + 1
    if future_start < T:
        jf_fut, _, _ = mean_jf(pred_masks[future_start:], gt_bool[future_start:])
    else:
        jf_fut = float("nan")

    return {"jf_all": jf_all, "jf_attacked": jf_att, "jf_future": jf_fut}


def codec_round_trip(
    frames_uint8: List[np.ndarray],
    ffmpeg_path: str,
    crf: int,
) -> Optional[List[np.ndarray]]:
    """H.264 encode + decode. Returns None on failure."""
    try:
        tensor = frames_to_tensor(frames_uint8)
        tensor_rt = encode_decode_h264(tensor, ffmpeg_path=ffmpeg_path, crf=crf)
        return tensor_to_frames(tensor_rt)
    except Exception as e:
        print(f"  [codec] error: {e}")
        return None


# ── Main pilot ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="",
                   help="Comma-separated video names. Empty = DAVIS_MINI_VAL")
    p.add_argument("--attack_frame", type=int, default=2,
                   help="Index of frame to attack (default: 2)")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--min_jf_clean", type=float, default=0.5,
                   help="Skip videos where clean JF < this threshold")
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir", default="results_v100/semantic_sweep")
    p.add_argument("--tag", default="pilot_v1")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    videos = [v.strip() for v in args.videos.split(",") if v.strip()] or DAVIS_MINI_VAL
    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pilot.log"

    print(f"[pilot] videos={videos}")
    print(f"[pilot] attack_frame={args.attack_frame}, crf={args.crf}")
    print(f"[pilot] output -> {out_dir}")

    predictor = build_sam2_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    for vid in videos:
        print(f"\n=== {vid} ===")
        frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue
        if args.attack_frame >= len(frames):
            print(f"  [skip] attack_frame={args.attack_frame} >= n_frames={len(frames)}")
            continue

        gt_bool = [m.astype(bool) for m in masks]

        # Clean baseline
        pred_clean = run_tracking(frames, masks, predictor, device)
        clean_splits = compute_jf_splits(pred_clean, masks, args.attack_frame)
        print(f"  clean: jf_all={clean_splits['jf_all']:.4f}  "
              f"jf_attacked={clean_splits['jf_attacked']:.4f}  "
              f"jf_future={clean_splits['jf_future']:.4f}")

        if clean_splits["jf_all"] < args.min_jf_clean:
            print(f"  [skip] jf_clean={clean_splits['jf_all']:.3f} < {args.min_jf_clean}")
            continue

        # Codec clean baseline
        codec_clean_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_clean_frames:
            pred_codec_clean = run_tracking(codec_clean_frames, masks, predictor, device)
            codec_clean_splits = compute_jf_splits(pred_codec_clean, masks, args.attack_frame)
        else:
            codec_clean_splits = {"jf_all": float("nan"), "jf_attacked": float("nan"), "jf_future": float("nan")}

        print(f"  codec_clean: jf_all={codec_clean_splits['jf_all']:.4f}  "
              f"jf_future={codec_clean_splits['jf_future']:.4f}")

        vid_results = {
            "video": vid,
            "n_frames": len(frames),
            "clean": clean_splits,
            "codec_clean": codec_clean_splits,
            "perturbations": [],
        }

        for pname, ptype, strength in PERTURBATION_GRID:
            t0 = time.time()

            # Apply perturbation to attack frame only
            adv_frames = list(frames)
            adv_frames[args.attack_frame] = perturb_frame(
                frames[args.attack_frame], ptype, strength)

            # Pre-codec adversarial
            pred_adv = run_tracking(adv_frames, masks, predictor, device)
            adv_splits = compute_jf_splits(pred_adv, masks, args.attack_frame)

            # Post-codec adversarial
            codec_adv_frames = codec_round_trip(adv_frames, args.ffmpeg_path, args.crf)
            if codec_adv_frames:
                pred_codec_adv = run_tracking(codec_adv_frames, masks, predictor, device)
                codec_adv_splits = compute_jf_splits(pred_codec_adv, masks, args.attack_frame)
            else:
                codec_adv_splits = {"jf_all": float("nan"), "jf_attacked": float("nan"), "jf_future": float("nan")}

            row = {
                "name": pname,
                "ptype": ptype,
                "strength": strength,
                "adv": adv_splits,
                "codec_adv": codec_adv_splits,
                "delta_jf_future_adv": clean_splits["jf_future"] - adv_splits["jf_future"],
                "delta_jf_future_codec": codec_clean_splits["jf_future"] - codec_adv_splits["jf_future"],
                "attacked_frame_drop_adv": clean_splits["jf_attacked"] - adv_splits["jf_attacked"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            vid_results["perturbations"].append(row)

            print(
                f"  {pname:20s} | "
                f"adv_future={adv_splits['jf_future']:.4f} (Δ{row['delta_jf_future_adv']:+.4f}) | "
                f"codec_future={codec_adv_splits['jf_future']:.4f} (Δ{row['delta_jf_future_codec']:+.4f}) | "
                f"att_drop={row['attacked_frame_drop_adv']:+.4f}"
            )

            # Flush log
            with open(log_path, "a") as lf:
                lf.write(
                    f"{vid} | {pname:20s} | "
                    f"adv_future Δ={row['delta_jf_future_adv']:+.4f} | "
                    f"codec_future Δ={row['delta_jf_future_codec']:+.4f}\n"
                )

        all_results.append(vid_results)

        # Intermediate save
        out_json = out_dir / "results.json"
        with open(out_json, "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Final save
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved -> {out_json}")

    # Summary table
    print("\n=== SUMMARY: best codec_future ΔJF per video ===")
    for vr in all_results:
        if not vr["perturbations"]:
            continue
        best = max(vr["perturbations"], key=lambda r: r["delta_jf_future_codec"])
        print(
            f"  {vr['video']:20s}  clean_future={vr['clean']['jf_future']:.4f}  "
            f"best={best['name']} Δcodec={best['delta_jf_future_codec']:+.4f}  "
            f"att_drop={best['attacked_frame_drop_adv']:+.4f}"
        )

    # Write summary file
    with open(out_dir / "summary.txt", "w") as sf:
        sf.write("video,ptype,strength,clean_jf_future,adv_jf_future,codec_jf_future,"
                 "delta_jf_future_adv,delta_jf_future_codec,attacked_frame_drop\n")
        for vr in all_results:
            for row in vr["perturbations"]:
                sf.write(
                    f"{vr['video']},{row['ptype']},{row['strength']},"
                    f"{vr['clean']['jf_future']:.6f},"
                    f"{row['adv']['jf_future']:.6f},"
                    f"{row['codec_adv']['jf_future']:.6f},"
                    f"{row['delta_jf_future_adv']:.6f},"
                    f"{row['delta_jf_future_codec']:.6f},"
                    f"{row['attacked_frame_drop_adv']:.6f}\n"
                )


if __name__ == "__main__":
    main()
