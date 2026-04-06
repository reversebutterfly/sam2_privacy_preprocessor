"""
eval_uap_davis.py — Evaluate UAP-SAM2 perturbation on DAVIS using J&F + codec.

Loads a pre-trained UAP (.pth, shape [1,3,1024,1024], float32, pixel space [0,1])
from the UAP-SAM2 repo, applies it to each DAVIS frame, runs SAM2 tracking,
and reports J&F (pre-codec and post-H.264) matching the format of pilot_mask_guided.py.

Usage (sanity, 5 videos):
  python eval_uap_davis.py \
      --uap_path /path/to/uap_file/YOUTUBE.pth \
      --videos "" --max_videos 5 --max_frames 15 \
      --crf 23 --prompt point \
      --tag uap_orig_davis5

Usage (full DAVIS eval):
  python eval_uap_davis.py \
      --uap_path /path/to/uap_file/YOUTUBE_fixed.pth \
      --max_frames 15 --crf 23 --prompt point \
      --tag uap_fixed_davis_full

Outputs:
  results/<tag>/uap_davis_results.json
  results/<tag>/uap_davis_summary.csv
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

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264
from src.metrics import jf_score, mean_jf


# ── UAP application ────────────────────────────────────────────────────────────

def apply_uap(frames: List[np.ndarray], uap: np.ndarray) -> List[np.ndarray]:
    """
    Apply UAP perturbation to a list of RGB uint8 frames.

    uap: float32 numpy array [3, H, W] in [−eps, +eps] normalised to [0,1] space.
    Resizes uap to each frame's resolution if needed.
    """
    out = []
    for fr in frames:
        H, W = fr.shape[:2]
        uap_hw = uap
        if uap_hw.shape[1] != H or uap_hw.shape[2] != W:
            # resize perturbation to frame resolution
            uap_resized = cv2.resize(
                uap_hw.transpose(1, 2, 0),   # [H, W, 3]
                (W, H),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)             # back to [3, H, W]
            uap_hw = uap_resized
        fr_f = fr.astype(np.float32) / 255.0
        adv_f = np.clip(fr_f + uap_hw.transpose(1, 2, 0), 0.0, 1.0)
        out.append((adv_f * 255.0).astype(np.uint8))
    return out


# ── Tracking (reused from pilot_mask_guided.py) ────────────────────────────────

def run_tracking(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    predictor,
    device: torch.device,
    prompt: str = "point",
) -> Tuple[List[np.ndarray], float, float, float]:
    H, W = frames[0].shape[:2]
    gt_bool = [m.astype(bool) for m in masks]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fr in enumerate(frames):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)
            first_mask = masks[0].astype(bool)
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

            pred = [None] * len(frames)
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


def codec_round_trip(frames, ffmpeg_path, crf):
    try:
        return encode_decode_h264(frames, ffmpeg_path=ffmpeg_path, crf=crf)
    except Exception as e:
        print(f"  [codec] error: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uap_path", required=True,
                   help="Path to UAP .pth file (shape [1,3,1024,1024])")
    p.add_argument("--videos", default="",
                   help="Comma-separated video names; empty = all DAVIS videos")
    p.add_argument("--max_videos", type=int, default=-1,
                   help="Max number of videos to evaluate (-1 = all)")
    p.add_argument("--max_frames", type=int, default=15)
    p.add_argument("--min_jf_clean", type=float, default=0.3,
                   help="Skip video if clean JF < this threshold")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point", choices=["point", "mask"])
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--config",     default=SAM2_CONFIG)
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--tag", default="uap_davis")
    p.add_argument("--save_dir", default="results")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load SAM2 ──
    from sam2.build_sam import build_sam2_video_predictor
    print(f"[model] Loading SAM2 from {args.checkpoint}")
    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)

    # ── Load UAP ──
    print(f"[uap] Loading {args.uap_path}")
    uap_tensor = torch.load(args.uap_path, map_location="cpu")
    if uap_tensor.ndim == 4:
        uap_tensor = uap_tensor.squeeze(0)   # [1,3,H,W] → [3,H,W]
    uap = uap_tensor.numpy()                  # [3,1024,1024], float32, [−eps,+eps]
    print(f"  shape={uap.shape}  L-inf={np.abs(uap).max():.4f}  "
          f"std={uap.std():.4f}")

    # ── Select videos ──
    davis_img = Path(args.davis_root) / "JPEGImages" / "480p"
    if args.videos:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        videos = sorted(d.name for d in davis_img.iterdir() if d.is_dir())
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"[eval] {len(videos)} videos, max_frames={args.max_frames}, "
          f"CRF={args.crf}, prompt={args.prompt}")

    # ── Output dir ──
    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_valid = 0
    sum_jf_clean = 0.0
    sum_jf_adv   = 0.0
    sum_jf_codec_clean = 0.0
    sum_jf_codec_adv   = 0.0

    for vname in videos:
        t0 = time.time()
        print(f"\n── {vname} ──")
        try:
            frames, masks, _ = load_single_video(
                args.davis_root, vname, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] load error: {e}")
            continue

        if not frames or not any(m.any() for m in masks):
            print(f"  [skip] no valid masks")
            continue

        # ── Clean tracking ──
        _, jf_clean, j_clean, f_clean = run_tracking(
            frames, masks, predictor, device, args.prompt)
        print(f"  clean: JF={jf_clean:.4f}  J={j_clean:.4f}  F={f_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean={jf_clean:.3f} < {args.min_jf_clean}")
            continue

        # ── Codec-clean ──
        codec_frames = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_frames:
            _, jf_codec_clean, _, _ = run_tracking(
                codec_frames, masks, predictor, device, args.prompt)
            print(f"  codec_clean: JF={jf_codec_clean:.4f}")
        else:
            jf_codec_clean = float("nan")

        # ── Adversarial (pre-codec) ──
        adv_frames = apply_uap(frames, uap)
        _, jf_adv, j_adv, f_adv = run_tracking(
            adv_frames, masks, predictor, device, args.prompt)
        delta_adv = jf_clean - jf_adv
        print(f"  adv (pre-codec):  JF={jf_adv:.4f}  ΔJF={delta_adv:+.4f}")

        # ── Adversarial (post-codec) ──
        codec_adv = codec_round_trip(adv_frames, args.ffmpeg_path, args.crf)
        if codec_adv:
            _, jf_codec_adv, _, _ = run_tracking(
                codec_adv, masks, predictor, device, args.prompt)
            delta_codec = jf_codec_clean - jf_codec_adv
            print(f"  adv (post-codec): JF={jf_codec_adv:.4f}  ΔJF={delta_codec:+.4f}")
        else:
            jf_codec_adv = float("nan")
            delta_codec  = float("nan")

        elapsed = time.time() - t0
        row = {
            "video": vname,
            "jf_clean":        round(jf_clean, 4),
            "j_clean":         round(j_clean, 4),
            "f_clean":         round(f_clean, 4),
            "jf_codec_clean":  round(jf_codec_clean, 4),
            "jf_adv":          round(jf_adv, 4),
            "j_adv":           round(j_adv, 4),
            "f_adv":           round(f_adv, 4),
            "jf_codec_adv":    round(jf_codec_adv, 4),
            "delta_jf_adv":    round(delta_adv, 4),
            "delta_jf_codec":  round(delta_codec, 4) if not np.isnan(delta_codec) else None,
            "n_frames":        len(frames),
            "elapsed_s":       round(elapsed, 1),
        }
        results.append(row)
        n_valid += 1
        sum_jf_clean       += jf_clean
        sum_jf_adv         += jf_adv
        if not np.isnan(jf_codec_clean): sum_jf_codec_clean += jf_codec_clean
        if not np.isnan(jf_codec_adv):   sum_jf_codec_adv   += jf_codec_adv

        # save incrementally
        with open(out_dir / "uap_davis_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ── Aggregate ──
    if n_valid == 0:
        print("\n[warn] No valid videos evaluated.")
        return

    agg = {
        "uap_path":         args.uap_path,
        "n_valid":          n_valid,
        "max_frames":       args.max_frames,
        "crf":              args.crf,
        "prompt":           args.prompt,
        "mean_jf_clean":        round(sum_jf_clean       / n_valid, 4),
        "mean_jf_adv":          round(sum_jf_adv         / n_valid, 4),
        "mean_jf_codec_clean":  round(sum_jf_codec_clean / n_valid, 4),
        "mean_jf_codec_adv":    round(sum_jf_codec_adv   / n_valid, 4),
        "mean_delta_jf_adv":    round((sum_jf_clean - sum_jf_adv) / n_valid, 4),
        "mean_delta_jf_codec":  round((sum_jf_codec_clean - sum_jf_codec_adv) / n_valid, 4),
    }
    print("\n" + "=" * 60)
    print(f"[aggregate] n_valid={n_valid}")
    print(f"  JF_clean:       {agg['mean_jf_clean']:.4f}")
    print(f"  JF_adv:         {agg['mean_jf_adv']:.4f}  "
          f"ΔJF_adv={agg['mean_delta_jf_adv']:+.4f}")
    print(f"  JF_codec_clean: {agg['mean_jf_codec_clean']:.4f}")
    print(f"  JF_codec_adv:   {agg['mean_jf_codec_adv']:.4f}  "
          f"ΔJF_codec={agg['mean_delta_jf_codec']:+.4f}")

    full = {"aggregate": agg, "per_video": results}
    with open(out_dir / "uap_davis_results.json", "w") as f:
        json.dump(full, f, indent=2)

    # CSV
    import csv
    with open(out_dir / "uap_davis_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\n[saved] {out_dir}/uap_davis_results.json")
    print(f"[saved] {out_dir}/uap_davis_summary.csv")
    import sys; sys.stdout.flush()


if __name__ == "__main__":
    main()
