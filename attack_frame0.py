"""
attack_frame0.py — Feature-Space Attack: Frame-0-Only Decisive Experiment
==========================================================================
Implements the smallest decisive experiment from refine-logs/FEATURE_ATTACK_PLAN.md:

  Attack only frame 0. All later frames stay clean.
  Evaluate with official SAM2VideoPredictor + real H.264 codec (CRF 23).

Loss modes (--attack_mode):
  cd  [PRIMARY]  C+D: J_mem_write + J_mem_match + J_ptr
                 Targets maskmem_features (C) and obj_ptr (D).
  b   [BACKUP]   B: FPN finest-level feature shift.
                 Replicates UAP-SAM2 objective; tests codec survival of FPN features.

Kill criteria (from plan):
  EARLY KILL: mean dJF_codec < 0.03 on >= 6 valid videos → pivot to negative-result paper
  PROCEED:    mean dJF_codec >= 0.05 on valid videos → launch full g_theta training

Usage (remote V100 server):
  # Primary experiment (C+D)
  python attack_frame0.py --attack_mode cd --steps 300 --restarts 2 \\
    --videos bike-packing,blackswan,bus,car-roundabout,car-turn,classic-car,color-run,cows,crossing \\
    --crf 23 --max_frames 50 --tag F0_CD --alpha 1.0 --beta 2.0 --gamma 0.25

  # Low-cost baseline (B)
  python attack_frame0.py --attack_mode b --steps 300 --restarts 2 \\
    --videos bike-packing,blackswan,bus,car-roundabout,car-turn,classic-car \\
    --crf 23 --max_frames 50 --tag F0_B

  # Quick smoke test (1 video, 50 steps)
  python attack_frame0.py --attack_mode cd --videos bike-packing --steps 50 --restarts 1 --tag F0_smoke
"""

import argparse
import json
import os
import random
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    DAVIS_ROOT, DAVIS_MINI_VAL, FFMPEG_PATH,
    RESULTS_DIR, SAM2_CHECKPOINT, SAM2_CONFIG,
)
from src.codec_eot import codec_proxy_transform, encode_decode_h264
from src.dataset import load_single_video
from src.losses import PerceptualLoss, SSIMConstraint, compute_ssim
from src.metrics import mean_jf, quality_summary
from train import SAM2VideoMemoryAttacker


# ---------------------------------------------------------------------------
# Feature-space loss helpers
# ---------------------------------------------------------------------------

def _cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance, inputs broadcast-flattened to [N, C]."""
    a_f = a.reshape(-1, a.shape[-1])
    b_f = b.reshape(-1, b.shape[-1])
    return (1.0 - F.cosine_similarity(a_f, b_f.detach(), dim=-1)).mean()


def _codec_pair(
    x_adv: torch.Tensor, x_cln: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply codec_proxy_transform with the SAME Python-random settings to both branches.
    This removes transform noise from the pairwise relational losses (Fix #4)."""
    state = random.getstate()
    x_adv_c = codec_proxy_transform(x_adv)
    random.setstate(state)
    with torch.no_grad():
        x_cln_c = codec_proxy_transform(x_cln)
    return x_adv_c, x_cln_c


def _build_maskmem_with_grad(
    attacker: SAM2VideoMemoryAttacker,
    vision_feats: List[torch.Tensor],
    feat_sizes: List[Tuple[int, int]],
    logits: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Like SAM2VideoMemoryAttacker._build_memory but does NOT detach the
    logits mask from the gradient tape (mask conditioning path gets gradient).
    Returns (mem_flat [Hm*Wm, B, Cm], pos_flat [Hm*Wm, B, Cp]) or (None, None).
    """
    try:
        B = vision_feats[-1].size(1)
        H_f, W_f = feat_sizes[-1]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, -1, H_f, W_f)

        masks_1024 = F.interpolate(
            logits.detach(),          # detach mask-logit path (same as original)
            size=(attacker.INPUT_SIZE, attacker.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )

        maskmem_features, maskmem_pos_enc = attacker.sam2.memory_encoder(
            pix_feat, masks_1024, skip_mask_sigmoid=False,
        )
        pos = maskmem_pos_enc[0] if isinstance(maskmem_pos_enc, (list, tuple)) \
            else maskmem_pos_enc

        _B, _C, _H, _W = maskmem_features.shape
        mem_flat = maskmem_features.view(_B, _C, _H * _W).permute(2, 0, 1)
        _B2, _C2, _H2, _W2 = pos.shape
        pos_flat = pos.view(_B2, _C2, _H2 * _W2).permute(2, 0, 1)
        return mem_flat, pos_flat
    except Exception as e:
        print(f"  [WARN] _build_maskmem_with_grad: {e}")
        return None, None


def compute_cd_losses(
    attacker: SAM2VideoMemoryAttacker,
    x_adv_1024: torch.Tensor,
    x_0_1024_clean: torch.Tensor,
    x_1_1024_clean: Optional[torch.Tensor],
    coords_np: np.ndarray,
    labels_np: np.ndarray,
    orig_hw: Tuple[int, int],
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.25,
    delta_mask: float = 0.05,
    use_mem_match: bool = True,
    tau: float = 0.1,
    gt_mask_1024: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the C+D feature-space attack losses.

    Returns (total_loss, diagnostics_dict).
    """
    device = x_adv_1024.device

    # Fix #3: shared codec sample — same random state applied to both branches
    x_adv_c, x_0_c = _codec_pair(x_adv_1024, x_0_1024_clean)

    # ── Adversarial forward (frame 0) ──────────────────────────────────────
    # Fix #2: add no_mem_embed as SAM2VideoPredictor does for the initial frame
    vf_adv, vpe_adv, fs_adv = attacker._encode_backbone(x_adv_c)
    if getattr(attacker.sam2, "directly_add_no_mem_embed", False):
        vf_adv = list(vf_adv)
        vf_adv[-1] = vf_adv[-1] + attacker.sam2.no_mem_embed
    logits_adv, ptr_adv = attacker._decode(
        vf_adv, coords_np, labels_np, prior_mask=None, orig_hw=orig_hw,
    )
    mem_flat_adv, pos_flat_adv = _build_maskmem_with_grad(
        attacker, vf_adv, fs_adv, logits_adv,
    )

    # ── Clean forward (frame 0, no gradient) ──────────────────────────────
    with torch.no_grad():
        vf_cln, vpe_cln, fs_cln = attacker._encode_backbone(x_0_c)
        if getattr(attacker.sam2, "directly_add_no_mem_embed", False):
            vf_cln = list(vf_cln)
            vf_cln[-1] = vf_cln[-1] + attacker.sam2.no_mem_embed
        logits_cln, ptr_cln = attacker._decode(
            vf_cln, coords_np, labels_np, prior_mask=None, orig_hw=orig_hw,
        )
        mem_flat_cln, pos_flat_cln = _build_maskmem_with_grad(
            attacker, vf_cln, fs_cln, logits_cln,
        )

    # ── J_mem_write (C) ───────────────────────────────────────────────────
    J_mw = torch.tensor(0.0, device=device)
    if mem_flat_adv is not None and mem_flat_cln is not None:
        J_mw = _cosine_dist(mem_flat_adv, mem_flat_cln)

    # ── J_ptr (D) ─────────────────────────────────────────────────────────
    J_ptr = torch.tensor(0.0, device=device)
    if ptr_adv is not None and ptr_cln is not None:
        J_ptr = _cosine_dist(ptr_adv, ptr_cln)

    # ── J_mem_match (attention output shift on frame 1) ───────────────────
    J_ma = torch.tensor(0.0, device=device)
    if use_mem_match and x_1_1024_clean is not None and mem_flat_adv is not None:
        try:
            with torch.no_grad():
                x_1_c = codec_proxy_transform(x_1_1024_clean)
                vf1, vpe1, _ = attacker._encode_backbone(x_1_c)
                # Reference: clean memory → clean attention output
                vf1_ref = attacker._apply_memory_attention(
                    list(vf1), vpe1,
                    mem_flat_cln, pos_flat_cln,
                    obj_ptr=ptr_cln, frame_dist=1,
                )
                att_cln_ref = vf1_ref[-1].detach()  # [HW, B, C]

            # Adversarial memory → attention output (grad flows through mem_flat_adv)
            vf1_adv = attacker._apply_memory_attention(
                list(vf1), vpe1,
                mem_flat_adv, pos_flat_adv,
                obj_ptr=ptr_adv, frame_dist=1,
            )
            att_adv_out = vf1_adv[-1]  # [HW, B, C], has grad

            J_ma = F.mse_loss(att_adv_out, att_cln_ref)
        except Exception as e:
            # Fix #5: log J_ma failure once and disable to avoid silent zero loss
            if not getattr(attacker, "_jma_warned", False):
                print(f"  [WARN] J_mem_match disabled (error: {e}). "
                      "Re-run with --no_mem_match to suppress.")
                attacker._jma_warned = True

    # ── J_mask (weak auxiliary) ───────────────────────────────────────────
    J_mask = torch.tensor(0.0, device=device)
    if gt_mask_1024 is not None and delta_mask > 0:
        # Fix #1: resize gt_mask to match logits_adv which is at orig_hw, not 1024x1024
        gt_f = F.interpolate(
            gt_mask_1024.float().to(device),
            size=logits_adv.shape[-2:], mode="nearest",
        )
        target = torch.zeros_like(logits_adv)
        bce = F.binary_cross_entropy_with_logits(
            logits_adv * gt_f, target, reduction="none",
        )
        J_mask = (bce * gt_f).sum() / (gt_f.sum() + 1e-6)

    # ── Total attack objective (maximize → negate for minimization) ───────
    J_total = alpha * J_mw + beta * J_ma + gamma * J_ptr + delta_mask * J_mask
    loss = -J_total

    diag = {
        "J_mw":   J_mw.item(),
        "J_ma":   J_ma.item(),
        "J_ptr":  J_ptr.item(),
        "J_mask": J_mask.item(),
        "J_total": J_total.item(),
    }
    return loss, diag


def compute_b_losses(
    attacker: SAM2VideoMemoryAttacker,
    x_adv_1024: torch.Tensor,
    x_0_1024_clean: torch.Tensor,
    x_1_1024_clean: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Mode B: FPN finest-level feature shift (UAP-SAM2 style) + optional
    temporal misalignment between consecutive adversarial-memory frames.
    """
    # Fix #3: shared codec sample for mode-B too
    x_adv_c, x_0_c = _codec_pair(x_adv_1024, x_0_1024_clean)

    vf_adv, vpe_adv, _ = attacker._encode_backbone(x_adv_c)
    with torch.no_grad():
        vf_cln, _, _ = attacker._encode_backbone(x_0_c)

    # FPN finest level: vision_feats[-1] in [HW, B, C]
    J_b = F.mse_loss(vf_adv[-1], vf_cln[-1].detach())

    # Optional inter-frame temporal misalignment (frame 0 vs frame 1)
    J_temporal = torch.tensor(0.0, device=x_adv_1024.device)
    if x_1_1024_clean is not None:
        try:
            with torch.no_grad():
                x_1_c = codec_proxy_transform(x_1_1024_clean)
                vf1_cln, _, _ = attacker._encode_backbone(x_1_c)
            J_temporal = -F.mse_loss(vf_adv[-1], vf1_cln[-1].detach())
        except Exception:
            pass

    J_total = J_b + 0.5 * J_temporal
    loss = -J_total

    diag = {
        "J_fpn":      J_b.item(),
        "J_temporal": J_temporal.item(),
        "J_total":    J_total.item(),
    }
    return loss, diag


# ---------------------------------------------------------------------------
# Per-video optimization
# ---------------------------------------------------------------------------

def optimize_frame0(
    video_name: str,
    davis_root: str,
    attacker: SAM2VideoMemoryAttacker,
    perceptual_loss: PerceptualLoss,
    ssim_constraint: SSIMConstraint,
    args,
) -> Optional[np.ndarray]:
    """
    Optimize δ_0 for a single video. Returns adversarial frame 0 (uint8 numpy).
    Runs (args.restarts) independent Adam runs and picks the one with highest J_total.
    """
    device = attacker.device

    frames_uint8, masks_uint8, _ = load_single_video(
        davis_root, video_name, max_frames=args.max_frames,
    )
    if not frames_uint8:
        print(f"  [{video_name}] SKIP: no frames loaded")
        return None

    # Prepare frame tensors at 1024×1024
    def to_1024(img_uint8):
        x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        return F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

    x_0_1024 = to_1024(frames_uint8[0]).detach()
    x_1_1024 = to_1024(frames_uint8[1]).detach() if len(frames_uint8) > 1 else None
    orig_hw = frames_uint8[0].shape[:2]

    # GT centroid prompt for frame 0
    first_mask = masks_uint8[0].astype(bool)
    ys, xs = np.where(first_mask)
    if len(ys) == 0:
        print(f"  [{video_name}] SKIP: empty first-frame mask")
        return None
    cx, cy = int(xs.mean()), int(ys.mean())
    coords_np = np.array([[cx, cy]], dtype=np.float32)
    labels_np = np.array([1], dtype=np.int32)

    # GT mask as 1024×1024 tensor for J_mask auxiliary
    gt_mask_1024 = torch.from_numpy(first_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_mask_1024 = F.interpolate(gt_mask_1024, size=(1024, 1024), mode="nearest").to(device)

    eps = args.eps / 255.0
    best_delta = None
    best_J = -float("inf")

    for restart in range(args.restarts):
        # Initialize δ_0 from uniform[-eps, eps]
        delta = torch.empty_like(x_0_1024).uniform_(-eps, eps).requires_grad_(True)

        optimizer = torch.optim.Adam([delta], lr=1.0 / 255.0)

        pbar = tqdm(range(args.steps), desc=f"  {video_name} restart={restart+1}", leave=False)
        for step in pbar:
            optimizer.zero_grad()

            # Project to eps-ball before forward pass
            with torch.no_grad():
                delta.data.clamp_(-eps, eps)
            x_adv_1024 = (x_0_1024 + delta).clamp(0.0, 1.0)

            if args.attack_mode == "cd":
                attack_loss, diag = compute_cd_losses(
                    attacker, x_adv_1024, x_0_1024,
                    x_1_1024, coords_np, labels_np, orig_hw,
                    alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                    delta_mask=args.delta_mask,
                    use_mem_match=args.use_mem_match,
                    gt_mask_1024=gt_mask_1024,
                )
            else:  # mode b
                attack_loss, diag = compute_b_losses(
                    attacker, x_adv_1024, x_0_1024, x_1_1024,
                )

            # Perceptual constraint penalty
            lpips_loss = perceptual_loss(x_0_1024, x_adv_1024)
            ssim_loss  = ssim_constraint(x_0_1024, x_adv_1024)
            total_loss = attack_loss + args.lambda_lpips * lpips_loss + args.lambda_ssim * ssim_loss

            total_loss.backward()
            optimizer.step()

            # Project δ back into eps-ball
            with torch.no_grad():
                delta.data.clamp_(-eps, eps)

            if step % 50 == 0 or step == args.steps - 1:
                J = diag["J_total"]
                pbar.set_postfix({
                    "J": f"{J:.4f}",
                    "lpips": f"{lpips_loss.item():.3f}",
                    "ssim": f"{ssim_loss.item():.3f}",
                })

        # Track best restart by J_total
        with torch.no_grad():
            delta.data.clamp_(-eps, eps)
            x_adv_final = (x_0_1024 + delta).clamp(0.0, 1.0)
            if args.attack_mode == "cd":
                _, diag = compute_cd_losses(
                    attacker, x_adv_final, x_0_1024,
                    x_1_1024, coords_np, labels_np, orig_hw,
                    alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                    delta_mask=args.delta_mask, use_mem_match=args.use_mem_match,
                    gt_mask_1024=gt_mask_1024,
                )
            else:
                _, diag = compute_b_losses(attacker, x_adv_final, x_0_1024, x_1_1024)

            J_final = diag["J_total"]
            if J_final > best_J:
                best_J = J_final
                best_delta = delta.detach().clone()
            print(f"  {video_name} restart={restart+1} J_final={J_final:.4f}")

    if best_delta is None:
        return None

    # Produce adversarial frame 0 at original resolution
    with torch.no_grad():
        best_delta.clamp_(-eps, eps)
        x_adv_1024_best = (x_0_1024 + best_delta).clamp(0.0, 1.0)
        # Resize back to original resolution
        x_adv_orig = F.interpolate(
            x_adv_1024_best, size=orig_hw, mode="bilinear", align_corners=False,
        )
        adv_np = (x_adv_orig[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Log perceptual quality
    lpips_val = perceptual_loss.measure(
        x_0_1024,
        F.interpolate(x_adv_orig, size=(1024, 1024), mode="bilinear", align_corners=False),
    )
    from src.metrics import compute_ssim
    ssim_val = compute_ssim(frames_uint8[0], adv_np)
    print(f"  {video_name} frame-0: best_J={best_J:.4f}  SSIM={ssim_val:.3f}  LPIPS={lpips_val:.3f}")

    return adv_np


# ---------------------------------------------------------------------------
# Evaluation: VideoPredictor with adversarial frame 0 + clean others
# ---------------------------------------------------------------------------

def eval_frame0_attack(
    video_name: str,
    adv_frame0: np.ndarray,
    davis_root: str,
    sam2_checkpoint: str,
    sam2_config: str,
    device: torch.device,
    crf: int = 23,
    ffmpeg_path: str = "ffmpeg",
    max_frames: int = 50,
) -> Dict:
    """
    Evaluate the frame-0 attack with official SAM2VideoPredictor.
    Builds a video with adversarial frame 0 + clean frames 1+, then runs codec.
    """
    from sam2.build_sam import build_sam2_video_predictor

    frames_uint8, masks_uint8, _ = load_single_video(davis_root, video_name, max_frames=max_frames)
    if not frames_uint8:
        return {}

    def run_predictor(frames_list, masks_list):
        predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
        predictor.eval()
        H, W = frames_list[0].shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            # Server's SAM2 load_video_frames_from_jpg_images only finds .jpg files.
            # JPEG q=95 artifacts are negligible and cancel in dJF since all branches
            # (clean/adv/codec) go through the same run_predictor path.
            for i, fr in enumerate(frames_list):
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            with torch.inference_mode():
                state = predictor.init_state(video_path=tmp)
                first_m = masks_list[0].astype(bool)
                ys, xs = np.where(first_m)
                if len(ys) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        points=np.array([[cx, cy]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                else:
                    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=first_m)

                pred = [None] * len(frames_list)
                for fidx, obj_ids, logits in predictor.propagate_in_video(state):
                    if 1 in obj_ids:
                        idx = list(obj_ids).index(1)
                        pred[fidx] = (logits[idx, 0] > 0.0).cpu().numpy()
                    else:
                        pred[fidx] = np.zeros((H, W), dtype=bool)
        for i in range(len(pred)):
            if pred[i] is None:
                pred[i] = np.zeros((H, W), dtype=bool)
        gt_bool = [m.astype(bool) for m in masks_list]
        mjf, mj, mf = mean_jf(pred, gt_bool)
        return mjf, mj, mf

    # 1. Clean (no codec, no attack) — pure baseline
    jf_clean, j_clean, f_clean = run_predictor(frames_uint8, masks_uint8)

    # 2. Clean-codec baseline: clean frames compressed with CRF, no attack
    # Fix #6: needed to isolate attack damage from codec damage
    jf_clean_codec = -1.0
    try:
        clean_codec_frames = encode_decode_h264(
            frames_uint8, crf=crf, fps=24, ffmpeg_path=ffmpeg_path,
        )
        jf_clean_codec, _, _ = run_predictor(clean_codec_frames, masks_uint8)
    except Exception as e:
        print(f"  [{video_name}] clean-codec FFmpeg failed: {e}")

    # 3. Adversarial pre-codec: adversarial frame 0 + clean frames 1+ (no H.264)
    adv_frames = [adv_frame0] + list(frames_uint8[1:])
    jf_adv, j_adv, f_adv = run_predictor(adv_frames, masks_uint8)

    # 4. Adversarial post-codec: H.264 applied to entire adversarial video
    jf_adv_codec = -1.0
    try:
        adv_codec_frames = encode_decode_h264(adv_frames, crf=crf, fps=24, ffmpeg_path=ffmpeg_path)
        jf_adv_codec, _, _ = run_predictor(adv_codec_frames, masks_uint8)
    except Exception as e:
        print(f"  [{video_name}] adv-codec FFmpeg failed: {e}")

    # dJF metrics:
    #   dJF_adv       = jf_clean - jf_adv            (attack damage, no codec)
    #   dJF_codec     = jf_clean - jf_adv_codec       (attack damage + codec)
    #   dJF_attack_under_codec = jf_clean_codec - jf_adv_codec  (attack damage, codec-normalised)
    dJF_adv   = jf_clean - jf_adv
    dJF_codec = jf_clean - jf_adv_codec if jf_adv_codec >= 0 else float("nan")
    dJF_attack_under_codec = (
        jf_clean_codec - jf_adv_codec
        if jf_clean_codec >= 0 and jf_adv_codec >= 0 else float("nan")
    )

    # 5. Perceptual quality on frame 0
    ssim_f0  = quality_summary([frames_uint8[0]], [adv_frame0])["mean_ssim"]
    lpips_f0 = quality_summary([frames_uint8[0]], [adv_frame0])["mean_lpips"]

    result = {
        "video":                video_name,
        "jf_clean":             jf_clean,
        "jf_clean_codec":       jf_clean_codec,
        "jf_adv":               jf_adv,
        "jf_adv_codec":         jf_adv_codec,
        "dJF_adv":              dJF_adv,
        "dJF_codec":            dJF_codec,
        "dJF_attack_under_codec": dJF_attack_under_codec,
        "ssim_f0":              ssim_f0,
        "lpips_f0":             lpips_f0,
        "valid":                jf_clean >= 0.5,
        "crf":                  crf,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Frame-0-only feature-space attack on SAM2")
    p.add_argument("--attack_mode", choices=["cd", "b"], default="cd",
                   help="cd=C+D (maskmem+ptr, primary), b=FPN feature shift (backup)")

    # Videos
    p.add_argument("--videos", type=str, default=",".join(DAVIS_MINI_VAL[:9]),
                   help="Comma-separated video names (default: first 9 DAVIS val videos)")
    p.add_argument("--max_frames", type=int, default=50)

    # Optimization
    p.add_argument("--steps",   type=int,   default=300,   help="Adam steps per restart")
    p.add_argument("--restarts",type=int,   default=2,     help="Independent restarts (best kept)")
    p.add_argument("--eps",     type=float, default=8.0,   help="Linf budget in /255 units")

    # C+D loss weights
    p.add_argument("--alpha",      type=float, default=1.0,  help="J_mem_write weight")
    p.add_argument("--beta",       type=float, default=2.0,  help="J_mem_match weight")
    p.add_argument("--gamma",      type=float, default=0.25, help="J_ptr weight")
    p.add_argument("--delta_mask", type=float, default=0.05, help="J_mask weight (weak)")
    p.add_argument("--no_mem_match", action="store_true",
                   help="Disable J_mem_match (use J_mw + J_ptr only; faster, less memory)")

    # Perceptual constraints
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--lambda_ssim",  type=float, default=1.0)
    p.add_argument("--max_lpips",    type=float, default=0.10)
    p.add_argument("--max_ssim_loss",type=float, default=0.05,
                   help="Max allowed (1-SSIM); 0.05 → SSIM >= 0.95")

    # Evaluation
    p.add_argument("--crf",      type=int,    default=23)
    p.add_argument("--min_jf_clean", type=float, default=0.5,
                   help="Valid video threshold (JF_clean >= this)")

    # Paths
    p.add_argument("--davis_root",      default=DAVIS_ROOT)
    p.add_argument("--checkpoint",      default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",     default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",     default=FFMPEG_PATH)
    p.add_argument("--results_dir",     default=RESULTS_DIR)
    p.add_argument("--tag",             default="F0_CD")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--eval_only",       action="store_true",
                   help="Skip optimization; load saved *_adv_f0.png and run eval only")

    return p.parse_args()


def main():
    args = parse_args()
    args.use_mem_match = not args.no_mem_match

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    # Output directory
    out_dir = os.path.join(args.results_dir, f"attack_frame0_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[attack_frame0] mode={args.attack_mode}  videos={len(videos)}  "
          f"steps={args.steps}×{args.restarts}  eps={args.eps}/255  "
          f"crf={args.crf}  tag={args.tag}")
    print(f"  C+D weights: α={args.alpha} β={args.beta} γ={args.gamma}")
    print(f"  output: {out_dir}")

    # Load SAM2VideoMemoryAttacker (frozen SAM2)
    attacker = SAM2VideoMemoryAttacker(args.checkpoint, args.sam2_config, device)
    perceptual_loss = PerceptualLoss(threshold=args.max_lpips, device=str(device))
    ssim_constraint = SSIMConstraint(threshold=args.max_ssim_loss)

    all_results = []
    adv_frames_cache: Dict[str, np.ndarray] = {}

    # ── Phase 1: Optimize δ_0 for each video ─────────────────────────────
    if args.eval_only:
        print("\n── Phase 1: SKIPPED (--eval_only) — loading saved adv frames ───────")
        for video_name in videos:
            png_path = os.path.join(out_dir, f"{video_name}_adv_f0.png")
            if os.path.exists(png_path):
                bgr = cv2.imread(png_path)
                if bgr is not None:
                    adv_frames_cache[video_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    print(f"  [{video_name}] loaded from {png_path}")
                else:
                    print(f"  [{video_name}] WARNING: failed to read {png_path}")
            else:
                print(f"  [{video_name}] WARNING: no saved adv frame at {png_path}")
    else:
        print("\n── Phase 1: Optimize δ_0 ────────────────────────────────────────")
        for video_name in videos:
            print(f"\n[{video_name}]")
            adv_f0 = optimize_frame0(
                video_name, args.davis_root, attacker,
                perceptual_loss, ssim_constraint, args,
            )
            if adv_f0 is not None:
                adv_frames_cache[video_name] = adv_f0
                # Save adversarial frame 0 as PNG
                adv_bgr = cv2.cvtColor(adv_f0, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir, f"{video_name}_adv_f0.png"), adv_bgr)

    # ── Phase 2: Evaluate with official VideoPredictor + codec ────────────
    print("\n── Phase 2: Evaluate with VideoPredictor ────────────────────────")
    for video_name, adv_f0 in adv_frames_cache.items():
        print(f"\n[{video_name}] evaluating...")
        result = eval_frame0_attack(
            video_name, adv_f0,
            davis_root=args.davis_root,
            sam2_checkpoint=args.checkpoint,
            sam2_config=args.sam2_config,
            device=device,
            crf=args.crf,
            ffmpeg_path=args.ffmpeg_path,
            max_frames=args.max_frames,
        )
        if result:
            all_results.append(result)
            print(f"  jf_clean={result['jf_clean']:.3f}  "
                  f"dJF_adv={result['dJF_adv']:+.3f}  "
                  f"dJF_codec={result['dJF_codec']:+.3f}  "
                  f"ssim={result['ssim_f0']:.3f}  "
                  f"valid={result['valid']}")

    # ── Phase 3: Summary ──────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    valid = [r for r in all_results if r.get("valid", False)]
    all_valid = [r for r in all_results if r.get("jf_codec", -1) >= 0]

    def stats(lst, key):
        if not lst:
            return 0.0, 0.0
        vals = [r[key] for r in lst]
        return float(np.mean(vals)), float(np.std(vals))

    mean_djf_adv_valid,    std_djf_adv_valid   = stats(valid, "dJF_adv")
    mean_djf_codec_valid,  std_djf_codec_valid  = stats(valid, "dJF_codec")
    mean_djf_auc_valid,    _                    = stats(
        [r for r in valid if not np.isnan(r.get("dJF_attack_under_codec", float("nan")))],
        "dJF_attack_under_codec",
    )
    mean_djf_adv_all,      _                    = stats(all_valid, "dJF_adv")
    mean_djf_codec_all,    _                    = stats(all_valid, "dJF_codec")
    mean_ssim,             _                    = stats(all_valid, "ssim_f0")

    # Kill criteria use dJF_attack_under_codec (codec-normalised attack damage)
    n_valid = len(valid)
    # Primary kill metric: attack damage UNDER codec (jf_clean_codec - jf_adv_codec)
    valid_auc = [r for r in valid
                 if not np.isnan(r.get("dJF_attack_under_codec", float("nan")))]
    kill_metric = mean_djf_auc_valid if valid_auc else mean_djf_codec_valid
    n_valid_harmed = sum(1 for r in valid if r.get("dJF_codec", 0) > 0)
    pct_harmed = n_valid_harmed / max(n_valid, 1)

    print(f"\n  Videos: {len(all_results)} total, {n_valid} valid (JF_clean >= {args.min_jf_clean})")
    print(f"  Valid subset:  mean dJF_adv={mean_djf_adv_valid:+.4f}±{std_djf_adv_valid:.4f}")
    print(f"                 mean dJF_codec(CRF{args.crf})={mean_djf_codec_valid:+.4f}±{std_djf_codec_valid:.4f}")
    print(f"                 mean dJF_attack_under_codec={mean_djf_auc_valid:+.4f}  [PRIMARY kill metric]")
    print(f"                 % harmed (dJF_codec>0)={pct_harmed:.0%}")
    print(f"  All valid:     mean dJF_codec={mean_djf_codec_all:+.4f}")
    print(f"  mean SSIM(f0): {mean_ssim:.3f}")

    # Kill criteria (from FEATURE_ATTACK_PLAN.md)
    print("\n── Kill Criteria Check ──────────────────────────────────────────")
    if n_valid >= 6:
        if kill_metric < 0.03:
            print(f"  [EARLY KILL] kill_metric={kill_metric:+.4f} < 0.03 on valid subset (n={n_valid}).")
            print("     Conclusion: feature-space attack does NOT survive CRF23.")
            print("     Action: Pivot to negative-result paper.")
            kill_status = "early_kill"
        elif kill_metric >= 0.05:
            print(f"  [PROCEED] kill_metric={kill_metric:+.4f} >= 0.05. Attack survives codec.")
            print("     Action: Launch full g_theta training (C+D losses, multi-frame).")
            kill_status = "proceed"
        else:
            print(f"  [MARGINAL] kill_metric={kill_metric:+.4f} (target >=0.05, kill <0.03).")
            print("     Recommendation: inspect per-video results; may proceed with caution.")
            kill_status = "marginal"
    else:
        print(f"  [INSUFFICIENT] Only {n_valid} valid videos (need >= 6 for kill criterion).")
        kill_status = "insufficient_videos"

    if pct_harmed < 0.70 and n_valid >= 6:
        print(f"  [WARNING] Only {pct_harmed:.0%} valid videos harmed (target >= 70%).")

    # Save results
    summary = {
        "tag":          args.tag,
        "attack_mode":  args.attack_mode,
        "crf":          args.crf,
        "eps":          args.eps,
        "steps":        args.steps,
        "restarts":     args.restarts,
        "alpha":        args.alpha,
        "beta":         args.beta,
        "gamma":        args.gamma,
        "n_videos":     len(all_results),
        "n_valid":      n_valid,
        "mean_djf_adv_valid":           mean_djf_adv_valid,
        "mean_djf_codec_valid":         mean_djf_codec_valid,
        "std_djf_codec_valid":          std_djf_codec_valid,
        "mean_djf_attack_under_codec":  mean_djf_auc_valid,
        "kill_metric":                  kill_metric,
        "mean_djf_adv_all":             mean_djf_adv_all,
        "mean_djf_codec_all":           mean_djf_codec_all,
        "mean_ssim_f0":                 mean_ssim,
        "pct_harmed_valid":             pct_harmed,
        "kill_status":                  kill_status,
        "per_video":    all_results,
    }
    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n  Results saved: {out_json}")

    # Per-video table
    print("\n  Per-video breakdown:")
    print(f"  {'Video':<20} {'JF_clean':>8} {'dJF_adv':>8} {'dJF_codec':>10} {'dJF_auc':>8} {'SSIM':>6} {'Valid':>5}")
    print("  " + "-" * 72)
    for r in sorted(all_results, key=lambda x: x["jf_clean"], reverse=True):
        auc = r.get("dJF_attack_under_codec", float("nan"))
        auc_s = f"{auc:+8.3f}" if not np.isnan(auc) else "     nan"
        print(f"  {r['video']:<20} {r['jf_clean']:>8.3f} {r['dJF_adv']:>+8.3f} "
              f"{r['dJF_codec']:>+10.3f} {auc_s} {r['ssim_f0']:>6.3f} {'✓' if r['valid'] else '✗':>5}")


if __name__ == "__main__":
    main()
