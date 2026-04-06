"""
pilot_xmem_eval.py — Evaluate combo_strong defense against XMem tracker.

Tests if the codec-amplified boundary suppression effect generalizes beyond SAM2.
Uses XMem (Cheng et al., ECCV 2022) as a second tracker.

Flow:
  1. Load DAVIS frames + GT masks
  2. Run XMem on clean frames → JF_clean
  3. Apply combo_strong + H.264 → defended frames
  4. Run XMem on defended frames → JF_defended
  5. ΔJF_codec = JF_clean - JF_defended

Usage:
    python pilot_xmem_eval.py \\
        --videos "" --max_frames 50 --crf 23 \\
        --xmem_root /IMBR_Data/Student-home/2025M_LvShaoting/XMem \\
        --xmem_checkpoint /IMBR_Data/Student-home/2025M_LvShaoting/XMem/saves/XMem.pth \\
        --tag xmem_combo_v1 --device cuda:3

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
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH, DAVIS_MINI_VAL
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264
from pilot_mask_guided import apply_edit_to_video, frame_quality


# ── XMem J&F metric helpers ────────────────────────────────────────────────────

def compute_j_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Jaccard (IoU) between binary masks."""
    pred_b = pred_mask > 0
    gt_b = gt_mask > 0
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def compute_f_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """F-measure (boundary F1) between binary masks."""
    def _boundary(mask, dilation_ratio=0.02):
        h, w = mask.shape
        d = max(1, int(round(dilation_ratio * min(h, w))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        mask_u8 = mask.astype(np.uint8) * 255
        dil = cv2.dilate(mask_u8, kernel)
        ero = cv2.erode(mask_u8, kernel)
        return (dil - ero) > 0

    pred_b = pred_mask > 0
    gt_b = gt_mask > 0
    pred_bnd = _boundary(pred_b)
    gt_bnd = _boundary(gt_b)
    inter_p = np.logical_and(pred_bnd, cv2.dilate(gt_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0).sum()
    inter_r = np.logical_and(gt_bnd, cv2.dilate(pred_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0).sum()
    if pred_bnd.sum() == 0:
        precision = 1.0 if gt_bnd.sum() == 0 else 0.0
    else:
        precision = float(inter_p) / float(pred_bnd.sum())
    if gt_bnd.sum() == 0:
        recall = 1.0
    else:
        recall = float(inter_r) / float(gt_bnd.sum())
    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def jf_from_masks(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[float, float, float]:
    """Compute mean J, mean F, mean J&F over a sequence."""
    js, fs = [], []
    for pred, gt in zip(pred_masks, gt_masks):
        js.append(compute_j_score(pred, gt))
        fs.append(compute_f_score(pred, gt))
    j = float(np.mean(js))
    f = float(np.mean(fs))
    return (j + f) / 2, j, f


# ── XMem inference wrapper ─────────────────────────────────────────────────────

def build_xmem(xmem_root: str, checkpoint: str, device: torch.device):
    """Load XMem model."""
    sys.path.insert(0, xmem_root)
    from model.network import XMem as XMemNet
    config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'num_prototypes': 128,
        'min_mid_term_frames': 5,
        'max_mid_term_frames': 10,
        'max_long_term_elements': 2000,  # reduced from 10000 to save VRAM
        'hidden_dim': 64,
        'key_dim': 64,
        'value_dim': 512,
        'single_object': False,  # XMem.pth is multi-object checkpoint
    }
    network = XMemNet(config, checkpoint, map_location='cpu').eval().to(device)
    return network, config


@torch.no_grad()
def run_xmem_tracking(
    frames: List[np.ndarray],
    gt_masks: List[np.ndarray],
    network,
    config: dict,
    device: torch.device,
) -> Tuple[List[np.ndarray], float, float, float]:
    """
    Run XMem on a video sequence.
    frames: list of H×W×3 uint8 RGB
    gt_masks: list of H×W uint8 (0/255)
    Returns: (pred_masks, jf, j, f)
    """
    from inference.inference_core import InferenceCore

    processor = InferenceCore(network, config)
    processor.set_all_labels([1])  # single object with label id=1

    pred_masks = []
    for t, (frame, gt_mask) in enumerate(zip(frames, gt_masks)):
        # Convert frame to tensor [3, H, W] float in [0, 1]
        img_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_t = img_t.to(device)

        if t == 0:
            # First frame: provide ground-truth mask [1, H, W] as float
            mask_t = torch.from_numpy((gt_mask > 0).astype(np.float32)).unsqueeze(0).to(device)
            prob = processor.step(img_t, mask_t, [1])
        else:
            prob = processor.step(img_t)

        # prob: [2, H, W] — channel 0=background, channel 1=object
        out_idx = torch.max(prob, dim=0).indices  # [H, W], values 0 or 1
        pred_binary = (out_idx == 1).cpu().numpy().astype(np.uint8) * 255
        pred_masks.append(pred_binary)

        # Explicitly free GPU tensors each frame to prevent accumulation
        del img_t, prob, out_idx
        if t == 0:
            del mask_t

    # Clean up processor memory
    del processor
    torch.cuda.empty_cache()

    # Ensure length matches gt
    pred_masks = pred_masks[:len(gt_masks)]
    while len(pred_masks) < len(gt_masks):
        pred_masks.append(np.zeros_like(gt_masks[0]))

    jf, j, f = jf_from_masks(pred_masks, gt_masks)
    return pred_masks, jf, j, f


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
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    p.add_argument("--edit_type", default="combo", choices=["idea1", "idea2", "combo"],
                   help="Edit type: idea1=boundary suppression, combo=idea1+halo")
    p.add_argument("--xmem_root", default="/IMBR_Data/Student-home/2025M_LvShaoting/XMem",
                   help="Path to XMem repository root")
    p.add_argument("--xmem_checkpoint", default="/IMBR_Data/Student-home/2025M_LvShaoting/XMem/saves/XMem.pth")
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir", default="results_v100/mask_guided")
    p.add_argument("--tag", default="xmem_combo_v1")
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
        videos = _raw or DAVIS_MINI_VAL

    params = {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}
    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[xmem_eval] combo_strong rw={args.ring_width} alpha={args.blend_alpha}")
    print(f"[xmem_eval] CRF={args.crf}, {len(videos)} videos -> {out_dir}")
    print(f"[xmem_eval] Loading XMem from {args.xmem_checkpoint}")

    network, config = build_xmem(args.xmem_root, args.xmem_checkpoint, device)
    all_results = []

    for vid in videos:
        print(f"\n=== {vid} ===")
        torch.cuda.empty_cache()
        try:
            frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] load error: {e}")
            continue
        if not frames:
            print(f"  [skip] empty")
            continue

        # Clean baseline with XMem
        try:
            _, jf_clean, j_clean, f_clean = run_xmem_tracking(frames, masks, network, config, device)
        except RuntimeError as e:
            print(f"  [skip] OOM: {e}")
            torch.cuda.empty_cache()
            continue
        print(f"  clean: JF={jf_clean:.4f}  J={j_clean:.4f}  F={f_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean < {args.min_jf_clean}")
            continue

        # Codec-clean baseline (unedited + H.264)
        codec_clean = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        jf_codec_clean = float("nan")
        if codec_clean:
            _, jf_codec_clean, _, _ = run_xmem_tracking(codec_clean, masks, network, config, device)
            print(f"  codec_clean CRF{args.crf}: JF={jf_codec_clean:.4f}")

        # Apply edit
        edited = apply_edit_to_video(frames, masks, args.edit_type, params)
        ssims = [frame_quality(fo, fe)[0] for fo, fe in zip(frames[:5], edited[:5])]
        mean_ssim = float(np.mean(ssims))

        # Pre-codec adversarial
        _, jf_adv, _, _ = run_xmem_tracking(edited, masks, network, config, device)
        print(f"  adv (pre-codec): JF={jf_adv:.4f}  ΔJF={jf_clean - jf_adv:+.4f}")

        # Post-codec adversarial (the defended released video)
        codec_adv = codec_round_trip(edited, args.ffmpeg_path, args.crf)
        jf_codec_adv = float("nan")
        delta_codec = float("nan")
        if codec_adv:
            _, jf_codec_adv, _, _ = run_xmem_tracking(codec_adv, masks, network, config, device)
            delta_codec = jf_codec_clean - jf_codec_adv
            print(f"  adv (post-codec): JF={jf_codec_adv:.4f}  ΔJF_codec={delta_codec:+.4f}")

        row = {
            "video": vid,
            "n_frames": len(frames),
            "jf_clean": round(jf_clean, 4),
            "jf_codec_clean": round(jf_codec_clean, 4) if not np.isnan(jf_codec_clean) else None,
            "jf_adv": round(jf_adv, 4),
            "jf_codec_adv": round(jf_codec_adv, 4) if not np.isnan(jf_codec_adv) else None,
            "delta_jf_codec": round(delta_codec, 4) if not np.isnan(delta_codec) else None,
            "mean_ssim": round(mean_ssim, 4),
        }
        all_results.append(row)

        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY] Stopping after first video.")
            break

    # Aggregate
    valid = [r for r in all_results if r["delta_jf_codec"] is not None and
             np.isfinite(r["delta_jf_codec"])]
    if valid:
        mean_delta = np.mean([r["delta_jf_codec"] for r in valid])
        mean_ssim = np.mean([r["mean_ssim"] for r in valid])
        n_neg = sum(1 for r in valid if r["delta_jf_codec"] < 0)
        print(f"\n{'='*60}")
        print(f"[XMem aggregate] n={len(valid)}")
        print(f"  ΔJF_codec = {mean_delta*100:+.2f}pp")
        print(f"  SSIM = {mean_ssim:.3f}")
        print(f"  Negatives = {n_neg}/{len(valid)}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"[saved] {out_dir}/results.json")


if __name__ == "__main__":
    main()
