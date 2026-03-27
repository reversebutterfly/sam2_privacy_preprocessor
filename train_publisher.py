#!/usr/bin/env python
"""
Publisher-side frame-0 preprocessing for SAM2 privacy protection.

Threat model / framing:
  - Publisher modifies ONLY the first frame (frame-0) of a video before release.
  - All subsequent frames are released CLEAN (unmodified).
  - The perturbation poisons SAM2's memory bank during its initialization step.
  - Downstream SAM2VideoPredictor then fails on subsequent CLEAN frames.
  - Publisher has NO access to or modification of downstream SAM2 deployments.

Key differences from the prior "attacker" framing:
  1. Only frame-0 is modified — not all frames.
  2. Training loss targets future-frame tracking failure (frames 1+), not frame-0 itself.
  3. Surrogate uses the VideoPredictor memory path (SAM2VideoMemoryAttacker.forward_clip)
     instead of the image predictor — fixing the train/eval mismatch identified in
     /research-review.
  4. g_theta is a one-pass preprocessing function, not a per-video iterative optimizer.

Gradient path (why this is different from prior pixel attacks):
  future_logits[1+] → memory_attention → maskmem_features[0] → pix_feat (backbone of frame-0_adv)
  → delta (from g_theta) — gradient DOES flow, unlike the prior image-predictor surrogate.

Fair UAP comparison (publisher framing):
  Use eval_codec.py --mode uap --frame0_only to apply an existing UAP delta to frame-0 only,
  giving the UAP the same publisher-side constraint as g_theta.

Usage:
  # Sanity: overfit 1 video, 500 steps — exits early on fail/pass
  python train_publisher.py --videos bear --num_steps 500 --sanity --tag sanity

  # Quick eval: 3 videos, 1000 steps
  python train_publisher.py --videos bear,breakdance,car-shadow --num_steps 1000 --tag quick

  # Post-training codec eval (add --frame0_only for publisher framing):
  python eval_codec.py --mode ours \\
      --checkpoint results/publisher/pub_frame0_steps1000_gs256/g_theta_final.pt \\
      --g_theta_size 256 --frame0_only --videos bear,breakdance,car-shadow --crf 23
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, RESULTS_DIR
from src.preprocessor import ResidualPreprocessor
from src.losses import soft_iou_loss, PerceptualLoss, SSIMConstraint
from src.metrics import jf_score
from src.dataset import load_single_video
from train import SAM2VideoMemoryAttacker, get_centroid_prompt, build_clip_pool

INPUT_SIZE = 1024  # SAM2 input resolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_frames(
    frames_np: List[np.ndarray],
    device: torch.device,
) -> List[torch.Tensor]:
    """Convert list of numpy [H,W,3] uint8 frames → list of [1,3,1024,1024] float tensors."""
    result = []
    for f in frames_np:
        x = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        result.append(x)
    return result


def apply_g_theta_frame0(
    g_theta: ResidualPreprocessor,
    frames_01: List[torch.Tensor],
    g_theta_size: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Apply g_theta to frame-0 ONLY. All other frames remain clean.

    The delta is produced at g_theta_size resolution, upsampled to INPUT_SIZE,
    then clamped to max_delta and applied to frame-0.

    Returns:
        adv_frames:  list of tensors — frame-0 is modified, frames 1+ are clean.
        delta_small: [1,3,g_theta_size,g_theta_size] raw delta (for SSIM/LPIPS at small res).
        frame0_adv:  [1,3,INPUT_SIZE,INPUT_SIZE] modified frame-0 (for metrics).
    """
    frame0 = frames_01[0]  # [1,3,1024,1024]

    x_small = F.interpolate(frame0, size=(g_theta_size, g_theta_size),
                             mode="bilinear", align_corners=False)
    _, delta_small = g_theta(x_small)

    delta_full = F.interpolate(delta_small, size=(INPUT_SIZE, INPUT_SIZE),
                                mode="bilinear", align_corners=False)
    delta_full = delta_full.clamp(-g_theta.max_delta, g_theta.max_delta)
    frame0_adv = (frame0 + delta_full).clamp(0.0, 1.0)

    # Only frame-0 is adversarial; the rest are untouched
    adv_frames = [frame0_adv] + list(frames_01[1:])
    return adv_frames, delta_small, frame0_adv


# ---------------------------------------------------------------------------
# Evaluation (surrogate-based, no codec — for training-time feedback)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_publisher_quick(
    g_theta: ResidualPreprocessor,
    video_attacker: SAM2VideoMemoryAttacker,
    videos: List[str],
    davis_root: str,
    device: torch.device,
    g_theta_size: int,
    clip_len: int = 6,
    max_frames: int = 20,
    max_eval_videos: int = 3,
) -> Dict:
    """
    Quick evaluation on up to max_eval_videos using the surrogate (no real H.264).

    Metrics are computed ONLY on frames 1+ (future frames after the poisoned init),
    which is the correct metric for publisher-side causal poisoning.

    Returns dict with mean_jf_clean, mean_jf_adv, delta_jf (surrogate, pre-codec),
    and per-video breakdown.
    """
    jf_clean_all, jf_adv_all = [], []
    per_video = {}
    g_theta.eval()

    for vid in videos[:max_eval_videos]:
        try:
            frames_np, masks_np, _ = load_single_video(davis_root, vid, max_frames=max_frames)
        except Exception as e:
            print(f"  [eval] could not load {vid}: {e}")
            continue
        if len(frames_np) < 2 or masks_np[0].sum() < 100:
            continue

        n = min(clip_len, len(frames_np))
        frames_01 = encode_frames(frames_np[:n], device)
        masks_clipped = masks_np[:n]
        coords, labels = get_centroid_prompt(masks_np[0])

        # --- Clean forward ---
        all_logits_clean = video_attacker.forward_clip(frames_01, masks_clipped, coords, labels)

        # --- Adversarial forward (frame-0 only) ---
        adv_frames, _, _ = apply_g_theta_frame0(g_theta, frames_01, g_theta_size)
        all_logits_adv = video_attacker.forward_clip(adv_frames, masks_clipped, coords, labels)

        # Metrics on frames 1+ (future tracking, not frame-0 itself)
        vid_jf_c, vid_jf_a = [], []
        for i in range(1, len(all_logits_clean)):
            gt_np = masks_clipped[i]
            if gt_np.sum() < 50:
                continue
            orig_hw = gt_np.shape[:2]

            logits_c = F.interpolate(all_logits_clean[i], size=orig_hw,
                                     mode="bilinear", align_corners=False)
            logits_a = F.interpolate(all_logits_adv[i], size=orig_hw,
                                     mode="bilinear", align_corners=False)

            pred_c = (torch.sigmoid(logits_c[0, 0]) > 0.5).cpu().numpy()
            pred_a = (torch.sigmoid(logits_a[0, 0]) > 0.5).cpu().numpy()
            gt_bool = gt_np.astype(bool)

            jf_c, _, _ = jf_score(pred_c, gt_bool)
            jf_a, _, _ = jf_score(pred_a, gt_bool)
            vid_jf_c.append(jf_c)
            vid_jf_a.append(jf_a)

        if vid_jf_c:
            mjf_c = float(np.mean(vid_jf_c))
            mjf_a = float(np.mean(vid_jf_a))
            per_video[vid] = {"jf_clean": mjf_c, "jf_adv": mjf_a,
                               "delta_jf": mjf_c - mjf_a}
            jf_clean_all.extend(vid_jf_c)
            jf_adv_all.extend(vid_jf_a)

    g_theta.train()

    if not jf_clean_all:
        return {"mean_jf_clean": 0.0, "mean_jf_adv": 0.0, "delta_jf": 0.0,
                "n_frames": 0, "per_video": {}}

    mjf_c = float(np.mean(jf_clean_all))
    mjf_a = float(np.mean(jf_adv_all))
    return {
        "mean_jf_clean": mjf_c,
        "mean_jf_adv":   mjf_a,
        "delta_jf":      mjf_c - mjf_a,
        "n_frames":      len(jf_clean_all),
        "per_video":     per_video,
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(out_dir: str, run_name: str, history: List[Dict], args_dict: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{run_name}.json")
    with open(json_path, "w") as f:
        json.dump({"args": args_dict, "history": history}, f, indent=2)
    if history:
        all_keys: list = []
        seen_keys: set = set()
        for row in history:
            for k in row.keys():
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)
        csv_path = os.path.join(out_dir, f"{run_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(history)
    print(f"  Results saved -> {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Publisher-side frame-0 preprocessing (causal memory poisoning)"
    )
    # Data / model
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--checkpoint",   default=SAM2_CHECKPOINT,
                   help="SAM2 model checkpoint (.pt)")
    p.add_argument("--sam2_config",  default=SAM2_CONFIG)
    p.add_argument("--videos",       default="bear",
                   help="Comma-separated training video names")
    p.add_argument("--val_videos",   default="",
                   help="Comma-separated val video names (empty = use train videos)")
    # Training
    p.add_argument("--num_steps",    type=int,   default=2000)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--g_accum_steps",type=int,   default=4)
    p.add_argument("--clip_len",     type=int,   default=6,
                   help="Number of frames per training clip (frame-0 + 5 future frames)")
    # g_theta architecture
    p.add_argument("--max_delta",    type=float, default=8.0 / 255)
    p.add_argument("--g_theta_size", type=int,   default=256,
                   help="g_theta internal resolution (frames are downsampled to this before CNN)")
    p.add_argument("--channels",     type=int,   default=32)
    p.add_argument("--num_blocks",   type=int,   default=4)
    # Loss weights
    p.add_argument("--lambda_perc",  type=float, default=0.5,
                   help="Weight for LPIPS hinge loss on frame-0")
    p.add_argument("--lambda_ssim",  type=float, default=0.5,
                   help="Weight for SSIM hinge loss on frame-0")
    p.add_argument("--max_lpips",    type=float, default=0.10)
    p.add_argument("--max_ssim_loss",type=float, default=0.10,
                   help="Maximum allowed (1-SSIM); threshold=0.10 → SSIM >= 0.90")
    # Logging / saving
    p.add_argument("--save_dir",     default=os.path.join(RESULTS_DIR, "publisher"))
    p.add_argument("--tag",          default="pub")
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--eval_every",   type=int, default=200)
    p.add_argument("--save_every",   type=int, default=500)
    # Sanity / resume
    p.add_argument("--sanity",       action="store_true",
                   help="Enable sanity gate at step 500: exit 1 if dJF < 0.10 on train set")
    p.add_argument("--sanity_threshold", type=float, default=0.10,
                   help="Minimum dJF on TRAIN set at step 500 to pass sanity check")
    p.add_argument("--resume",       default="",
                   help="Path to g_theta checkpoint to resume from")
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[publisher] device={device}")
    print(f"  g_theta_size={args.g_theta_size}, clip_len={args.clip_len}, steps={args.num_steps}")
    print(f"  lambda_perc={args.lambda_perc}, lambda_ssim={args.lambda_ssim}")

    # ── Models ──────────────────────────────────────────────────────────────
    print("[publisher] Loading SAM2VideoMemoryAttacker (frozen)...")
    video_attacker = SAM2VideoMemoryAttacker(args.checkpoint, args.sam2_config, device)
    video_attacker.eval()
    for p in video_attacker.parameters():
        p.requires_grad_(False)

    g_theta = ResidualPreprocessor(
        channels=args.channels,
        num_blocks=args.num_blocks,
        max_delta=args.max_delta,
    ).to(device)
    if args.resume:
        g_theta.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"  Resumed from {args.resume}")
    n_params = sum(p.numel() for p in g_theta.parameters())
    print(f"  g_theta: {g_theta.__class__.__name__}, params={n_params:,}")

    # ── Losses ──────────────────────────────────────────────────────────────
    perc_fn = PerceptualLoss(threshold=args.max_lpips, device=device)
    ssim_fn = SSIMConstraint(threshold=args.max_ssim_loss)

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(g_theta.parameters(), lr=args.lr, weight_decay=1e-4)
    t_max = max(1, args.num_steps // args.g_accum_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=args.lr * 0.1
    )

    # ── Data ────────────────────────────────────────────────────────────────
    train_videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    val_videos = (
        [v.strip() for v in args.val_videos.split(",") if v.strip()]
        if args.val_videos
        else train_videos
    )
    print(f"  Train: {train_videos}")
    print(f"  Val:   {val_videos}")

    clip_pool_raw = build_clip_pool(
        train_videos, args.davis_root,
        max_frames=30, clip_len=args.clip_len,
    )
    # MAJOR fix: require frame-0 AND ≥1 future frame with visible object.
    # Clips where only frame-0 is visible produce n_future=0 → no attack gradient.
    clip_pool = [
        c for c in clip_pool_raw
        if c["masks"][0].sum() >= 100
        and any(m.sum() >= 50 for m in c["masks"][1:])
    ]
    if not clip_pool:
        print("[ERROR] No valid clips found (need frame-0 AND future frames with object).")
        sys.exit(1)
    print(f"  Clip pool: {len(clip_pool)}/{len(clip_pool_raw)} clips "
          f"(filtered to require future-frame object visibility)")

    # ── Run name ────────────────────────────────────────────────────────────
    run_name = f"{args.tag}_frame0_steps{args.num_steps}_gs{args.g_theta_size}"
    out_dir  = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    history: List[Dict] = []
    optimizer.zero_grad()
    pbar = tqdm(range(1, args.num_steps + 1), desc="publisher-frame0")

    for step in pbar:
        clip      = random.choice(clip_pool)
        frames_np = clip["frames"]
        masks_np  = clip["masks"]

        if masks_np[0].sum() < 100:
            continue

        coords, labels = get_centroid_prompt(masks_np[0])

        # Encode all frames to [1,3,1024,1024] float tensors
        frames_01 = encode_frames(frames_np, device)

        g_theta.train()

        # === Publisher operation: apply g_theta to frame-0 ONLY ===
        adv_frames, delta_small, frame0_adv = apply_g_theta_frame0(
            g_theta, frames_01, args.g_theta_size
        )

        # === Run video predictor surrogate ===
        # Frame-0 is adversarial; frames 1+ are clean.
        # The memory bank is initialised from the poisoned frame-0.
        # The loss measures tracking failure on frames 1+ (causal poisoning signal).
        all_logits = video_attacker.forward_clip(
            adv_frames, masks_np, coords, labels
        )

        # === CRITICAL fix: verify gradient path at step 1 ===
        # forward_clip silently falls back to prompt-only if memory_attention fails.
        # In that case, future logits have NO gradient path to frame0_adv.
        if step == 1 and len(all_logits) > 1:
            probe_grad = torch.autograd.grad(
                all_logits[1].sum(), frame0_adv,
                allow_unused=True, retain_graph=True,
            )
            if probe_grad[0] is None:
                print(
                    "\n[CRITICAL] Gradient from logits[1] → frame0_adv is None!\n"
                    "  SAM2VideoMemoryAttacker memory path is not connected.\n"
                    "  Likely cause: _build_memory or _apply_memory_attention raised an exception\n"
                    "  and silently fell back to prompt-only propagation.\n"
                    "  Training cannot learn causal memory poisoning without this path.\n"
                    "  Check SAM2VideoMemoryAttacker in train.py for exception swallowing."
                )
                sys.exit(1)
            else:
                print("\n[CRITICAL CHECK PASSED] Gradient path frame0_adv → future logits is intact.")

        # === Attack loss: tracking failure on future frames ===
        l_future = torch.tensor(0.0, device=device)
        n_future = 0
        for i in range(1, len(all_logits)):
            gt_np = masks_np[i]
            if gt_np.sum() < 50:
                continue
            gt = (
                torch.from_numpy(gt_np.astype(np.float32))
                .unsqueeze(0).unsqueeze(0).to(device)
            )
            logits_i = all_logits[i]
            if gt.shape[-2:] != logits_i.shape[-2:]:
                gt = F.interpolate(gt, size=logits_i.shape[-2:], mode="nearest")
            l_future = l_future + soft_iou_loss(logits_i, gt)
            n_future += 1

        if n_future == 0:
            # MAJOR fix: no attack gradient this step — skip to avoid pure regularization update.
            # (This should be rare after clip_pool filtering above.)
            optimizer.zero_grad()
            continue
        l_future = l_future / n_future

        # === Perceptual constraints on frame-0 only ===
        frame0_orig = frames_01[0]
        frame0_orig_small = F.interpolate(
            frame0_orig, size=(args.g_theta_size, args.g_theta_size),
            mode="bilinear", align_corners=False,
        )
        frame0_adv_small = F.interpolate(
            frame0_adv, size=(args.g_theta_size, args.g_theta_size),
            mode="bilinear", align_corners=False,
        )

        l_perc = perc_fn(frame0_orig_small, frame0_adv_small)
        l_ssim = ssim_fn(frame0_orig_small, frame0_adv_small)

        loss = l_future + args.lambda_perc * l_perc + args.lambda_ssim * l_ssim

        # === Gradient accumulation ===
        (loss / args.g_accum_steps).backward()

        if step % args.g_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(g_theta.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # === Logging ===
        if step % args.log_every == 0:
            history.append({
                "step":      step,
                "loss":      loss.item(),
                "l_future":  l_future.item(),
                "l_perc":    l_perc.item(),
                "l_ssim":    l_ssim.item(),
                "n_future":  n_future,
                "lr":        scheduler.get_last_lr()[0],
            })
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                fut=f"{l_future.item():.4f}",
                n=n_future,
            )

        # === Eval ===
        if step % args.eval_every == 0:
            ev = eval_publisher_quick(
                g_theta, video_attacker, val_videos,
                args.davis_root, device, args.g_theta_size, args.clip_len,
            )
            if history:
                history[-1].update({f"val_{k}": v for k, v in ev.items()
                                     if not isinstance(v, dict)})
            print(
                f"\n  step={step} | surrogate pre-codec"
                f" JF_clean={ev['mean_jf_clean']:.3f}"
                f" JF_adv={ev['mean_jf_adv']:.3f}"
                f" dJF={ev['delta_jf']:.3f}"
                f" (n={ev['n_frames']} future frames)"
            )
            for vid, vr in ev.get("per_video", {}).items():
                print(f"    {vid}: clean={vr['jf_clean']:.3f} adv={vr['jf_adv']:.3f}"
                      f" dJF={vr['delta_jf']:.3f}")

        # === Sanity gate at step 500 ===
        if args.sanity and step == 500:
            ev_train = eval_publisher_quick(
                g_theta, video_attacker, train_videos,
                args.davis_root, device, args.g_theta_size, args.clip_len,
            )
            dJF_train = ev_train["delta_jf"]
            print(f"\n[SANITY] step=500 train dJF={dJF_train:.3f} "
                  f"(threshold={args.sanity_threshold})")
            save_results(out_dir, run_name, history, vars(args))
            torch.save(g_theta.state_dict(),
                       os.path.join(out_dir, "g_theta_step500.pt"))
            if dJF_train < args.sanity_threshold:
                print(
                    f"[SANITY FAIL] dJF={dJF_train:.3f} < {args.sanity_threshold} on train set.\n"
                    f"  Memory poisoning via frame-0 (ε={args.max_delta*255:.0f}/255) "
                    f"shows no training signal in the surrogate.\n"
                    f"  INTERPRETATION: The VideoPredictor memory path gradient is too weak "
                    f"for g_theta to learn from frame-0 alone at this budget.\n"
                    f"  → Publisher-side framing with pixel-constrained frame-0 only "
                    f"may not be viable. Consider diffusion editing (IDEA_REPORT.md §Idea 1)."
                )
                sys.exit(1)
            else:
                print(f"[SANITY PASS] dJF={dJF_train:.3f} >= {args.sanity_threshold}."
                      f" Surrogate shows memory poisoning signal.")
                if args.num_steps == 500:
                    print("[SANITY MODE] num_steps=500 — stopping after sanity check.")
                    torch.save(g_theta.state_dict(),
                               os.path.join(out_dir, "g_theta_final.pt"))
                    sys.exit(0)

        # === Checkpoint ===
        if step % args.save_every == 0:
            torch.save(g_theta.state_dict(),
                       os.path.join(out_dir, f"g_theta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))

    # Final save
    torch.save(g_theta.state_dict(), os.path.join(out_dir, "g_theta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[publisher] Done -> {out_dir}")
    print(f"  Next: codec eval with frame-0-only mode:")
    print(
        f"  python eval_codec.py --mode ours"
        f" --checkpoint {os.path.join(out_dir, 'g_theta_final.pt')}"
        f" --g_theta_size {args.g_theta_size}"
        f" --frame0_only"
        f" --videos {args.val_videos or args.videos}"
        f" --crf 23 --min_jf_clean 0.5"
    )


if __name__ == "__main__":
    main()
