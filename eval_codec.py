"""
Post-codec evaluation for SAM2 Privacy Preprocessor.
Runs real FFmpeg H.264 encode/decode, then evaluates SAM2 tracking J&F.

Usage:
  python eval_codec.py --checkpoint results/ours_s1_steps3000/g_theta_final.pt --videos bear,breakdance
  python eval_codec.py --mode uap --uap_delta results/uap.../uap_delta_final.pt --videos bear
  python eval_codec.py --mode clean --videos bear,breakdance   # baseline (no attack)
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    SAM2_CHECKPOINT, SAM2_CONFIG,
    DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH, RESULTS_DIR,
)
from src.preprocessor import ResidualPreprocessor, Stage4Preprocessor
from src.codec_eot import encode_decode_h264, tensor_to_frames, frames_to_tensor
from src.metrics import mean_jf, quality_summary
from src.dataset import load_single_video


def run_sam2_video_tracking(
    frames_uint8: List[np.ndarray],
    masks_uint8:  List[np.ndarray],
    sam2_checkpoint: str,
    sam2_config: str,
    device: torch.device,
) -> Tuple[List[np.ndarray], float, float, float]:
    """
    Run SAM2 video predictor with GT first-frame prompt, return predicted masks.
    Writes frames to a temp JPEG directory so the official init_state(video_path) API works.
    Returns (pred_masks, mean_jf, mean_j, mean_f)
    """
    import cv2 as _cv2
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
    predictor.eval()

    H, W = frames_uint8[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write frames as JPEG (SAM2 video predictor loads from a directory of images)
        for i, fr in enumerate(frames_uint8):
            bgr = _cv2.cvtColor(fr, _cv2.COLOR_RGB2BGR)
            _cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr,
                         [_cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode():
            inference_state = predictor.init_state(video_path=tmp_dir)

            # Add first-frame GT centroid point as prompt
            first_mask = masks_uint8[0].astype(bool)
            ys, xs = np.where(first_mask)
            if len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.array([[cx, cy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    mask=first_mask,
                )

            # Propagate through all frames
            pred_masks = [None] * len(frames_uint8)
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                if 1 in obj_ids:
                    idx = list(obj_ids).index(1)
                    pred_masks[frame_idx] = (mask_logits[idx, 0] > 0.0).cpu().numpy()
                else:
                    pred_masks[frame_idx] = np.zeros((H, W), dtype=bool)

    # Fill any gaps (frames not returned by propagate)
    for i in range(len(pred_masks)):
        if pred_masks[i] is None:
            pred_masks[i] = np.zeros((H, W), dtype=bool)

    gt_bool = [m.astype(bool) for m in masks_uint8]
    mjf, mj, mf = mean_jf(pred_masks, gt_bool)
    return pred_masks, mjf, mj, mf


def apply_preprocessor(
    frames_uint8: List[np.ndarray],
    device: torch.device,
    mode: str = "clean",
    g_theta: Optional[nn.Module] = None,
    uap_delta: Optional[torch.Tensor] = None,
    input_size: int = 1024,
) -> List[np.ndarray]:
    """Apply g_theta or UAP delta to frames, return adversarial uint8 frames."""
    adv_frames = []
    for frame_np in frames_uint8:
        x = torch.from_numpy(frame_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        H, W = frame_np.shape[:2]
        x1024 = torch.nn.functional.interpolate(x, size=(input_size, input_size),
                                                  mode="bilinear", align_corners=False)
        with torch.no_grad():
            if mode == "clean":
                x_adv_1024 = x1024
            elif mode == "uap" and uap_delta is not None:
                x_adv_1024 = torch.clamp(x1024 + uap_delta.to(device), 0, 1)
            elif mode == "ours" and g_theta is not None:
                g_theta.eval()
                x_adv_1024 = g_theta(x1024)[0]
            else:
                x_adv_1024 = x1024

        # Resize back to original
        x_adv = torch.nn.functional.interpolate(x_adv_1024, size=(H, W),
                                                  mode="bilinear", align_corners=False)
        adv_np = (x_adv[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        adv_frames.append(adv_np)
    return adv_frames


def eval_video(
    video_name: str,
    davis_root: str,
    device: torch.device,
    sam2_checkpoint: str,
    sam2_config: str,
    mode: str = "clean",
    g_theta: Optional[nn.Module] = None,
    uap_delta: Optional[torch.Tensor] = None,
    crf_values: List[int] = None,
    ffmpeg_path: str = "ffmpeg",
    max_frames: int = 50,
) -> Dict:
    """
    Evaluate one video: clean SAM2 -> adversarial (pre-codec) -> post-codec at each CRF.
    crf_values: list of H.264 CRF values to sweep, e.g. [18, 23, 28]
    """
    if crf_values is None:
        crf_values = [23]

    frames, masks, _ = load_single_video(davis_root, video_name, max_frames=max_frames)
    if not frames:
        return {}

    print(f"  {video_name}: {len(frames)} frames")

    # --- Clean baseline ---
    _, jf_clean, j_clean, f_clean = run_sam2_video_tracking(
        frames, masks, sam2_checkpoint, sam2_config, device
    )

    # --- Adversarial (pre-codec) ---
    adv_frames = apply_preprocessor(frames, device, mode, g_theta, uap_delta)
    _, jf_adv, j_adv, f_adv = run_sam2_video_tracking(
        adv_frames, masks, sam2_checkpoint, sam2_config, device
    )

    # --- Post-codec sweep over CRF values ---
    codec_results = {}
    for crf in crf_values:
        try:
            codec_frames = encode_decode_h264(adv_frames, crf=crf, fps=24, ffmpeg_path=ffmpeg_path)
            _, jf_c, j_c, f_c = run_sam2_video_tracking(
                codec_frames, masks, sam2_checkpoint, sam2_config, device
            )
            codec_results[crf] = {"jf": jf_c, "j": j_c, "f": f_c}
        except Exception as e:
            print(f"    [WARN] FFmpeg CRF={crf} failed: {e}")
            codec_results[crf] = {"jf": -1.0, "j": -1.0, "f": -1.0}

    # Primary CRF = first in list (default 23)
    primary_crf  = crf_values[0]
    jf_codec     = codec_results[primary_crf]["jf"]
    j_codec      = codec_results[primary_crf]["j"]
    f_codec      = codec_results[primary_crf]["f"]

    # --- Quality metrics ---
    qual = quality_summary(frames, adv_frames)

    result = {
        "video":            video_name,
        "n_frames":         len(frames),
        "jf_clean":         jf_clean,
        "j_clean":          j_clean,
        "f_clean":          f_clean,
        "jf_adv":           jf_adv,
        "j_adv":            j_adv,
        "f_adv":            f_adv,
        "jf_codec":         jf_codec,       # primary CRF
        "j_codec":          j_codec,
        "f_codec":          f_codec,
        "delta_jf_adv":     jf_clean - jf_adv,
        "delta_jf_codec":   jf_clean - jf_codec if jf_codec >= 0 else -1.0,
        "mean_ssim":        qual["mean_ssim"],
        "mean_psnr":        qual["mean_psnr"],
        "mean_lpips":       qual["mean_lpips"],
        "codec_sweep":      {f"crf{k}": v for k, v in codec_results.items()},
    }
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Post-codec evaluation")
    p.add_argument("--mode",     default="clean", choices=["clean", "ours", "uap"])
    p.add_argument("--checkpoint",  default=None, help="g_theta checkpoint (.pt)")
    p.add_argument("--uap_delta",   default=None, help="UAP delta (.pt)")
    p.add_argument("--stage",       type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--channels",    type=int, default=32)
    p.add_argument("--num_blocks",  type=int, default=4)
    p.add_argument("--max_delta",   type=float, default=8.0/255.0)
    p.add_argument("--davis_root",  default=DAVIS_ROOT)
    p.add_argument("--videos",      default=None)
    p.add_argument("--max_frames",  type=int, default=50)
    p.add_argument("--crf",         type=int, nargs="+", default=[23],
                   help="H.264 CRF value(s) to sweep, e.g. --crf 18 23 28")
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",     default=SAM2_CONFIG)
    p.add_argument("--save_dir",    default=RESULTS_DIR)
    p.add_argument("--tag",         default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    g_theta   = None
    uap_delta = None
    if args.mode == "ours" and args.checkpoint:
        if args.stage == 4:
            g_theta = Stage4Preprocessor(args.channels, args.num_blocks, args.max_delta).to(device)
        else:
            g_theta = ResidualPreprocessor(args.channels, args.num_blocks, args.max_delta).to(device)
        g_theta.load_state_dict(torch.load(args.checkpoint, map_location=device))
        g_theta.eval()
        print(f"  Loaded g_theta from {args.checkpoint}")
    elif args.mode == "uap" and args.uap_delta:
        uap_delta = torch.load(args.uap_delta, map_location=device)
        print(f"  Loaded UAP delta from {args.uap_delta}")

    video_names = [v.strip() for v in args.videos.split(",")] if args.videos else DAVIS_MINI_VAL[:5]
    print(f"Videos: {video_names}")

    results = []
    for vid in tqdm(video_names, desc="Eval"):
        try:
            r = eval_video(
                vid, args.davis_root, device,
                args.sam2_checkpoint, args.sam2_config,
                mode=args.mode, g_theta=g_theta, uap_delta=uap_delta,
                crf_values=args.crf, ffmpeg_path=args.ffmpeg_path, max_frames=args.max_frames,
            )
            if r:
                results.append(r)
                print(f"  {vid}: JF_clean={r['jf_clean']:.3f} "
                      f"JF_adv={r['jf_adv']:.3f} JF_codec={r['jf_codec']:.3f} "
                      f"SSIM={r['mean_ssim']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {vid}: {e}")

    # Summary
    if results:
        for key in ["jf_clean", "jf_adv", "jf_codec", "delta_jf_adv", "delta_jf_codec",
                    "mean_ssim", "mean_psnr"]:
            vals = [r[key] for r in results if r.get(key, -1) >= 0]
            if vals:
                print(f"  mean_{key}: {np.mean(vals):.4f}")

    # Save
    run_name = (f"{args.tag}_" if args.tag else "") + f"eval_codec_{args.mode}"
    out_dir  = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
