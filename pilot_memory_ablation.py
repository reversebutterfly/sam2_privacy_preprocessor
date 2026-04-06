"""
Memory Ablation Pilot — Single-Frame Attack + Memory Reset Proof

Applies the mask-guided edit to ONLY one frame (attack_frame) and measures
how failure propagates to subsequent frames. Also runs a "memory reset"
condition: clear SAM2 state at frame attack_frame+1 using clean frames,
proving the failure is due to corrupted memory state, not visual confusion.

Experimental conditions per video:
  1. clean               — original video, full propagation
  2. adv_all             — edit ALL frames (from pilot_mask_guided)
  3. adv_frame_t         — edit ONLY frame t; all other frames clean
  4. adv_frame_t_reset   — edit ONLY frame t; reset SAM2 state at t+1
                           (re-initialize from clean frame t+1 using GT mask)

Metrics reported:
  - J&F on frames [0..attack_frame-1]  (pre-attack; should be clean=adv)
  - J&F on frames [attack_frame]       (attacked frame itself)
  - J&F on frames [attack_frame+1..T]  (future frames — shows memory poisoning)

Usage:
  python pilot_memory_ablation.py \\
      --edit_type combo \\
      --videos dog-agility,bike-packing,blackswan \\
      --attack_frame 2 \\
      --ring_width 16 --blend_alpha 0.6 \\
      --halo_offset 8 --halo_width 12 --halo_strength 0.4 \\
      --prompt mask --max_frames 60 --crf 23 \\
      --tag ablation_v1 --save_dir results_v100/memory_ablation
"""

import argparse
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
from src.codec_eot import encode_decode_h264
from src.metrics import jf_score, mean_jf
from pilot_mask_guided import (
    apply_boundary_suppression, apply_echo_contour, apply_combo,
    EDIT_FNS, build_predictor, frame_quality
)


def codec_round_trip(frames, ffmpeg_path, crf):
    try:
        return encode_decode_h264(frames, ffmpeg_path=ffmpeg_path, crf=crf)
    except Exception as e:
        print(f"  [codec] error: {e}")
        return None


def write_frames_to_dir(frames: List[np.ndarray], tmp_dir: str):
    for i, fr in enumerate(frames):
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def run_tracking_standard(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    predictor,
    device: torch.device,
    prompt: str = "mask",
) -> Tuple[List[np.ndarray], List[float], float, float]:
    """Standard SAM2 tracking. Returns (pred_masks, per_frame_jf, mean_jf, mean_j)."""
    H, W = frames[0].shape[:2]
    gt_bool = [m.astype(bool) for m in masks]

    with tempfile.TemporaryDirectory() as tmp_dir:
        write_frames_to_dir(frames, tmp_dir)
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)
            _add_prompt(predictor, state, masks[0], prompt)
            pred = _propagate(predictor, state, len(frames), H, W)

    per_frame = [float(jf_score(p, g)[0]) for p, g in zip(pred, gt_bool)]
    _, mj, mf = _mean_jf_from_masks(pred, gt_bool)
    mjf = float(np.mean(per_frame))
    return pred, per_frame, mjf, mj


def run_tracking_with_reset(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    predictor,
    device: torch.device,
    reset_at_frame: int,
    prompt: str = "mask",
) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Run SAM2 tracking, but at `reset_at_frame` reinitialize SAM2 state
    using the clean frame and its GT mask. This simulates memory reset.
    """
    H, W = frames[0].shape[:2]
    gt_bool = [m.astype(bool) for m in masks]
    pred_all = [np.zeros((H, W), dtype=bool)] * len(frames)

    # Phase 1: frames 0..reset_at_frame-1
    phase1_frames = frames[:reset_at_frame]
    with tempfile.TemporaryDirectory() as tmp_dir:
        write_frames_to_dir(phase1_frames, tmp_dir)
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)
            _add_prompt(predictor, state, masks[0], prompt)
            pred1 = _propagate(predictor, state, len(phase1_frames), H, W)
    for i, p in enumerate(pred1):
        pred_all[i] = p

    # Phase 2: from reset_at_frame onward, reinit from clean frame
    phase2_frames = frames[reset_at_frame:]
    if len(phase2_frames) > 0:
        with tempfile.TemporaryDirectory() as tmp_dir:
            write_frames_to_dir(phase2_frames, tmp_dir)
            with torch.inference_mode():
                state = predictor.init_state(video_path=tmp_dir)
                # Use GT mask at the reset frame as the new prompt
                reset_mask = masks[reset_at_frame].astype(bool)
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=reset_mask)
                pred2 = _propagate(predictor, state, len(phase2_frames), H, W)
        for i, p in enumerate(pred2):
            pred_all[reset_at_frame + i] = p

    per_frame = [float(jf_score(p, g)[0]) for p, g in zip(pred_all, gt_bool)]
    mjf = float(np.mean(per_frame))
    return pred_all, per_frame, mjf


def _add_prompt(predictor, state, mask0, prompt):
    first_mask = mask0.astype(bool)
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


def _propagate(predictor, state, n_frames, H, W):
    pred = [None] * n_frames
    for fi, obj_ids, logits in predictor.propagate_in_video(state):
        if 1 in obj_ids:
            idx = list(obj_ids).index(1)
            pred[fi] = (logits[idx, 0] > 0.0).cpu().numpy()
        else:
            pred[fi] = np.zeros((H, W), dtype=bool)
    for i in range(n_frames):
        if pred[i] is None:
            pred[i] = np.zeros((H, W), dtype=bool)
    return pred


def _mean_jf_from_masks(preds, gts):
    jfs = [jf_score(p, g)[0] for p, g in zip(preds, gts)]
    return float(np.mean(jfs)), float(np.mean(jfs)), float(np.mean(jfs))


def split_jf(per_frame_jf, attack_frame, n_frames):
    """Split per-frame J&F into pre / attacked / future."""
    pre    = per_frame_jf[:attack_frame] if attack_frame > 0 else []
    at     = [per_frame_jf[attack_frame]] if attack_frame < n_frames else []
    future = per_frame_jf[attack_frame + 1:] if attack_frame + 1 < n_frames else []
    return (
        float(np.mean(pre))    if pre    else float("nan"),
        float(np.mean(at))     if at     else float("nan"),
        float(np.mean(future)) if future else float("nan"),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edit_type", default="combo", choices=["idea1", "idea2", "combo"])
    p.add_argument("--videos", default="")
    p.add_argument("--attack_frame", type=int, default=2,
                   help="Frame index to attack (0-indexed; default=2 = second memory write)")
    p.add_argument("--max_frames", type=int, default=60)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="mask", choices=["point", "mask"])
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    p.add_argument("--ring_width",   type=int,   default=16)
    p.add_argument("--blend_alpha",  type=float, default=0.6)
    p.add_argument("--halo_offset",   type=int,   default=8)
    p.add_argument("--halo_width",    type=int,   default=12)
    p.add_argument("--halo_strength", type=float, default=0.4)
    p.add_argument("--davis_root",    default=DAVIS_ROOT)
    p.add_argument("--checkpoint",    default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",   default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",   default=FFMPEG_PATH)
    p.add_argument("--save_dir",      default="results_v100/memory_ablation")
    p.add_argument("--tag",           default="ablation_v1")
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


def get_params(args):
    if args.edit_type == "idea1":
        return {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}
    elif args.edit_type == "idea2":
        return {"halo_offset": args.halo_offset, "halo_width": args.halo_width,
                "halo_strength": args.halo_strength}
    else:
        return {
            "ring_width": args.ring_width, "blend_alpha": args.blend_alpha,
            "halo_offset": args.halo_offset, "halo_width": args.halo_width,
            "halo_strength": args.halo_strength,
        }


def main():
    args = parse_args()
    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] or DAVIS_MINI_VAL
    params = get_params(args)
    edit_fn = EDIT_FNS[args.edit_type]

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] edit_type={args.edit_type}, attack_frame={args.attack_frame}")
    print(f"[ablation] params={params}")
    print(f"[ablation] videos={videos}")
    print(f"[ablation] output -> {out_dir}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)
    all_results = []

    for vid in videos:
        print(f"\n=== {vid} ===")
        t0 = time.time()
        frames, masks, _ = load_single_video(
            args.davis_root, vid, max_frames=args.max_frames)
        if not frames or len(frames) <= args.attack_frame + 2:
            print(f"  [skip] too few frames ({len(frames) if frames else 0})")
            continue

        T = len(frames)
        af = args.attack_frame

        # 1. Clean
        _, pf_clean, jf_clean, _ = run_tracking_standard(frames, masks, predictor, device, args.prompt)
        print(f"  clean: JF={jf_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean < {args.min_jf_clean}")
            continue

        # 2. adv_all (all frames edited)
        edited_all = [edit_fn(fr, m, **params) for fr, m in zip(frames, masks)]
        codec_adv_all = codec_round_trip(edited_all, args.ffmpeg_path, args.crf)
        if codec_adv_all:
            _, pf_adv_all, jf_adv_all, _ = run_tracking_standard(
                codec_adv_all, masks, predictor, device, args.prompt)
        else:
            pf_adv_all = [float("nan")] * T
            jf_adv_all = float("nan")

        # 3. adv_frame_t (only frame t edited, codec round-trip)
        edited_single = list(frames)  # copy
        edited_single[af] = edit_fn(frames[af], masks[af], **params)
        codec_adv_single = codec_round_trip(edited_single, args.ffmpeg_path, args.crf)
        if codec_adv_single:
            _, pf_adv_single, jf_adv_single, _ = run_tracking_standard(
                codec_adv_single, masks, predictor, device, args.prompt)
        else:
            pf_adv_single = [float("nan")] * T
            jf_adv_single = float("nan")

        # 4. adv_frame_t_reset (edit frame t, then reset memory at t+1 with clean GT)
        if codec_adv_single:
            _, pf_reset, jf_reset = run_tracking_with_reset(
                codec_adv_single, masks, predictor, device,
                reset_at_frame=af + 1, prompt=args.prompt)
        else:
            pf_reset = [float("nan")] * T
            jf_reset = float("nan")

        # Per-segment J&F
        jf_pre_clean,  jf_at_clean,  jf_fut_clean  = split_jf(pf_clean,      af, T)
        jf_pre_all,    jf_at_all,    jf_fut_all    = split_jf(pf_adv_all,    af, T)
        jf_pre_single, jf_at_single, jf_fut_single = split_jf(pf_adv_single, af, T)
        jf_pre_reset,  jf_at_reset,  jf_fut_reset  = split_jf(pf_reset,      af, T)

        # Quality
        s, p = frame_quality(frames[af], edited_single[af])
        print(f"  attack_frame={af}: SSIM={s:.4f}  PSNR={p:.1f}dB")
        print(f"  clean:      pre={jf_pre_clean:.3f}  at={jf_at_clean:.3f}  fut={jf_fut_clean:.3f}")
        print(f"  adv_all:    pre={jf_pre_all:.3f}    at={jf_at_all:.3f}    fut={jf_fut_all:.3f}"
              f"  (Δfut={jf_fut_clean - jf_fut_all:+.3f})")
        print(f"  adv_t:      pre={jf_pre_single:.3f}  at={jf_at_single:.3f}  fut={jf_fut_single:.3f}"
              f"  (Δfut={jf_fut_clean - jf_fut_single:+.3f})")
        print(f"  adv_t+reset:pre={jf_pre_reset:.3f}  at={jf_at_reset:.3f}   fut={jf_fut_reset:.3f}"
              f"  (Δfut={jf_fut_clean - jf_fut_reset:+.3f})")

        fut_delta_adv_all    = jf_fut_clean - jf_fut_all
        fut_delta_adv_single = jf_fut_clean - jf_fut_single
        fut_delta_reset      = jf_fut_clean - jf_fut_reset

        # Memory proof: single-frame attack causes future failure, but reset recovers
        memory_proof = (fut_delta_adv_single > 0.03) and (fut_delta_reset < fut_delta_adv_single * 0.5)
        print(f"  MEMORY PROOF: {'YES' if memory_proof else 'NO'}"
              f" (single Δfut={fut_delta_adv_single:+.3f}"
              f", reset Δfut={fut_delta_reset:+.3f})")

        result = {
            "video": vid,
            "n_frames": T,
            "attack_frame": af,
            "jf_clean": jf_clean,
            "ssim_attack_frame": s,
            "psnr_attack_frame": p,
            # Full-video JF
            "jf_adv_all": jf_adv_all,
            "jf_adv_single": jf_adv_single,
            "jf_adv_single_reset": jf_reset,
            # Pre / at / future breakdown
            "jf_pre_clean": jf_pre_clean,     "jf_at_clean": jf_at_clean,     "jf_fut_clean": jf_fut_clean,
            "jf_pre_adv_all": jf_pre_all,     "jf_at_adv_all": jf_at_all,     "jf_fut_adv_all": jf_fut_all,
            "jf_pre_adv_single": jf_pre_single,"jf_at_adv_single": jf_at_single,"jf_fut_adv_single": jf_fut_single,
            "jf_pre_reset": jf_pre_reset,     "jf_at_reset": jf_at_reset,     "jf_fut_reset": jf_fut_reset,
            # Deltas
            "delta_fut_adv_all": float(fut_delta_adv_all),
            "delta_fut_adv_single": float(fut_delta_adv_single),
            "delta_fut_reset": float(fut_delta_reset),
            "memory_proof": bool(memory_proof),
            "per_frame_clean": pf_clean,
            "per_frame_adv_single": pf_adv_single,
            "per_frame_reset": pf_reset,
            "elapsed_s": time.time() - t0,
        }
        all_results.append(result)

    # Save
    n = len(all_results)
    if n > 0:
        mean_fut_single = sum(r["delta_fut_adv_single"] for r in all_results) / n
        mean_fut_reset  = sum(r["delta_fut_reset"]      for r in all_results) / n
        proof_count     = sum(1 for r in all_results if r["memory_proof"])
    else:
        mean_fut_single = mean_fut_reset = 0.0
        proof_count = 0

    output = {
        "args": vars(args),
        "aggregate": {
            "n_videos": n,
            "mean_delta_fut_adv_single": mean_fut_single,
            "mean_delta_fut_reset": mean_fut_reset,
            "memory_proof_count": proof_count,
            "memory_proof_frac": proof_count / n if n > 0 else 0.0,
        },
        "results": all_results,
    }

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n} videos)")
    print(f"  single-frame attack Δfut = +{mean_fut_single:.4f}")
    print(f"  memory reset       Δfut = +{mean_fut_reset:.4f}")
    print(f"  memory proof       = {proof_count}/{n} ({proof_count/n:.0%})" if n > 0 else "  no videos")
    if mean_fut_reset < mean_fut_single * 0.5:
        print(f"  *** MEMORY CAUSALITY CONFIRMED: reset reduces future failure by ≥50% ***")
    print(f"{'='*60}")

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
