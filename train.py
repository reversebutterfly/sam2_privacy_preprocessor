"""
SAM2 Privacy Preprocessor - Main Training Script
Supports B0 (sanity), B1 (UAP baseline), B2 (Stage 1 ours).

Modes:
  --mode ours  : Learnable residual preprocessor (stages 1-4)
  --mode uap   : Universal adversarial perturbation baseline

Usage:
  # B0 Sanity
  python train.py --stage 1 --videos bear --num_steps 500 --sanity

  # B1 Baseline UAP
  python train.py --mode uap --videos bear,breakdance,car-shadow,dance-jump,dog --num_steps 2000

  # B2 Stage 1
  python train.py --mode ours --stage 1 --num_steps 3000
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
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    SAM2_CHECKPOINT, SAM2_CONFIG,
    DAVIS_ROOT, DAVIS_MINI_TRAIN, DAVIS_MINI_VAL,
    RESULTS_DIR,
)
from src.preprocessor import ResidualPreprocessor, Stage4Preprocessor
from src.losses import PerceptualLoss, soft_iou_loss, temporal_loss
from src.codec_eot import codec_proxy_transform
from src.metrics import jf_score
from src.dataset import load_single_video


class SAM2Attacker(nn.Module):
    """Frozen SAM2 wrapper. Only g_theta gradients flow back."""

    SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    SAM2_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    INPUT_SIZE = 1024

    def __init__(self, checkpoint: str, config_name: str, device: torch.device):
        super().__init__()
        self.device = device
        self._orig_hw: Optional[Tuple[int, int]] = None
        from sam2.build_sam import build_sam2
        sam2_model = build_sam2(config_name, checkpoint, device=device)
        sam2_model.eval()
        for p in sam2_model.parameters():
            p.requires_grad_(False)
        self.sam2 = sam2_model
        self.mean = self.SAM2_MEAN.to(device)
        self.std  = self.SAM2_STD.to(device)

    def encode_image(self, img_np: np.ndarray) -> torch.Tensor:
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        H, W = img_np.shape[:2]
        self._orig_hw = (H, W)
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = torch.nn.functional.interpolate(
            x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        return x  # [1,3,1024,1024] in [0,1]

    def forward(
        self,
        x01: torch.Tensor,
        point_coords_np: np.ndarray,
        point_labels_np: np.ndarray,
    ) -> torch.Tensor:
        x_norm = (x01 - self.mean) / self.std
        backbone_out = self.sam2.forward_image(x_norm)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)

        if self._orig_hw is not None:
            H, W = self._orig_hw
            sx, sy = self.INPUT_SIZE / W, self.INPUT_SIZE / H
        else:
            sx, sy = 1.0, 1.0

        pts = torch.from_numpy(point_coords_np).float().to(self.device)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        pts  = pts.unsqueeze(0)
        lbls = torch.from_numpy(point_labels_np).int().to(self.device).unsqueeze(0)

        sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
            points=(pts, lbls), boxes=None, masks=None,
        )
        high_res_feats = [f.to(self.device) for f in vision_feats[:-1]]
        image_embed    = vision_feats[-1].to(self.device)

        try:
            low_res_masks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
        except TypeError:
            low_res_masks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
            )

        if self._orig_hw is not None:
            logits = torch.nn.functional.interpolate(
                low_res_masks, size=self._orig_hw, mode="bilinear", align_corners=False,
            )
        else:
            logits = low_res_masks
        return logits  # [1,1,H,W]


def get_centroid_prompt(mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_np.astype(bool))
    if len(ys) == 0:
        H, W = mask_np.shape
        cx, cy = W // 2, H // 2
    else:
        cx, cy = int(xs.mean()), int(ys.mean())
    return np.array([[cx, cy]], dtype=np.float32), np.array([1], dtype=np.int32)


def build_frame_pool(video_names: List[str], davis_root: str, max_frames: int = 30) -> List[Dict]:
    pool = []
    for vid in video_names:
        try:
            frames, masks, _ = load_single_video(davis_root, vid, max_frames=max_frames)
            for i, (f, m) in enumerate(zip(frames, masks)):
                pool.append({"video": vid, "frame_idx": i, "frame_np": f, "mask_np": m})
        except Exception as e:
            print(f"  [WARN] Could not load {vid}: {e}")
    return pool


def eval_quick(
    attacker: SAM2Attacker,
    g_theta,
    val_videos: List[str],
    davis_root: str,
    device: torch.device,
    max_frames: int = 10,
    mode: str = "ours",
    uap_delta: Optional[torch.Tensor] = None,
) -> Dict:
    jf_clean_list, jf_adv_list = [], []
    for vid in val_videos[:3]:
        try:
            frames, masks, _ = load_single_video(davis_root, vid, max_frames=max_frames)
        except Exception:
            continue
        for frame_np, mask_np in zip(frames, masks):
            if mask_np.sum() < 100:
                continue
            coords, labels = get_centroid_prompt(mask_np)
            with torch.no_grad():
                x01 = attacker.encode_image(frame_np)
                logits_clean = attacker(x01, coords, labels)
                pred_clean = (torch.sigmoid(logits_clean[0, 0]) > 0.5).cpu().numpy()
                if mode == "uap" and uap_delta is not None:
                    x_adv = torch.clamp(x01 + uap_delta, 0, 1)
                elif mode == "ours" and g_theta is not None:
                    g_theta.eval()
                    x_adv = g_theta(x01)[0]
                    g_theta.train()
                else:
                    x_adv = x01
                logits_adv = attacker(x_adv, coords, labels)
                pred_adv = (torch.sigmoid(logits_adv[0, 0]) > 0.5).cpu().numpy()
            jf_c, _, _ = jf_score(pred_clean, mask_np.astype(bool))
            jf_a, _, _ = jf_score(pred_adv,   mask_np.astype(bool))
            jf_clean_list.append(jf_c)
            jf_adv_list.append(jf_a)
    if not jf_clean_list:
        return {"mean_jf_clean": 0.0, "mean_jf_adv": 0.0, "delta_jf": 0.0}
    mjf_c = float(np.mean(jf_clean_list))
    mjf_a = float(np.mean(jf_adv_list))
    return {"mean_jf_clean": mjf_c, "mean_jf_adv": mjf_a, "delta_jf": mjf_c - mjf_a}


def save_results(out_dir: str, run_name: str, history: List[Dict], args_dict: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{run_name}.json")
    with open(json_path, "w") as f:
        json.dump({"args": args_dict, "history": history}, f, indent=2)
    if history:
        csv_path = os.path.join(out_dir, f"{run_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    print(f"  Results saved -> {json_path}")


def train_uap(args, attacker: SAM2Attacker, frame_pool: List[Dict], device: torch.device) -> None:
    eps = args.max_delta
    uap_lr = args.uap_lr if args.uap_lr is not None else args.lr
    use_lpips = args.uap_lpips
    print(f"\n[UAP] eps={eps:.4f}  lr={uap_lr}  lpips_hinge={use_lpips}")
    delta = torch.zeros(1, 3, SAM2Attacker.INPUT_SIZE, SAM2Attacker.INPUT_SIZE,
                        device=device, requires_grad=True)
    optimizer = optim.Adam([delta], lr=uap_lr)
    perc_fn   = PerceptualLoss(threshold=args.max_lpips, device=device) if use_lpips else None
    run_name  = (f"{args.tag}_" if args.tag else "") + f"uap_eps{eps:.4f}_steps{args.num_steps}"
    if use_lpips:
        run_name += "_lpips"
    out_dir   = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history   = []
    pbar = tqdm(range(1, args.num_steps + 1), desc="UAP")
    for step in pbar:
        item = random.choice(frame_pool)
        if item["mask_np"].sum() < 50:
            continue
        coords, labels = get_centroid_prompt(item["mask_np"])
        x01 = attacker.encode_image(item["frame_np"])
        # Project delta before forward (for loss computation)
        delta_c = torch.clamp(delta, -eps, eps)
        x_adv   = torch.clamp(x01 + delta_c, 0.0, 1.0)
        logits  = attacker(x_adv, coords, labels)
        gt = torch.from_numpy(item["mask_np"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = torch.nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        loss = soft_iou_loss(logits, gt)
        if perc_fn is not None:
            loss = loss + args.lambda1 * perc_fn(x01, x_adv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Project back onto L-inf ball after Adam step
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if step % args.log_every == 0:
            row = {"step": step, "loss": loss.item()}
            history.append(row)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if step % args.eval_every == 0:
            # Use post-step projected delta for eval
            with torch.no_grad():
                delta_eval = torch.clamp(delta, -eps, eps)
            ev = eval_quick(attacker, None, DAVIS_MINI_VAL[:3], args.davis_root,
                            device, mode="uap", uap_delta=delta_eval)
            history[-1].update(ev)
            print(f"\n  step={step} JF_clean={ev['mean_jf_clean']:.3f} "
                  f"JF_adv={ev['mean_jf_adv']:.3f} dJF={ev['delta_jf']:.3f}")
        if step % args.save_every == 0:
            with torch.no_grad():
                delta_save = torch.clamp(delta, -eps, eps)
            torch.save(delta_save.cpu(), os.path.join(out_dir, f"uap_delta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))
    with torch.no_grad():
        delta_final = torch.clamp(delta, -eps, eps)
    torch.save(delta_final.cpu(), os.path.join(out_dir, "uap_delta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[UAP] Done -> {out_dir}")


def train_ours(args, attacker: SAM2Attacker, frame_pool: List[Dict], device: torch.device) -> None:
    print(f"\n[OURS] Stage {args.stage}, steps={args.num_steps}")
    if args.stage == 4:
        g_theta = Stage4Preprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                      max_delta=args.max_delta).to(device)
    else:
        g_theta = ResidualPreprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                        max_delta=args.max_delta).to(device)
    print(f"  {g_theta.__class__.__name__}  params={sum(p.numel() for p in g_theta.parameters()):,}")
    if args.checkpoint:
        g_theta.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Resumed from {args.checkpoint}")
    # AdamW for Stage 2+ (as planned), Adam for Stage 1
    _opt_cls = optim.AdamW if (args.optimizer == "adamw" or args.stage >= 2) else optim.Adam
    optimizer  = _opt_cls(g_theta.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=args.lr * 0.1)
    perc_fn    = PerceptualLoss(threshold=args.max_lpips, device=device)
    run_name   = (f"{args.tag}_" if args.tag else "") + f"ours_s{args.stage}_steps{args.num_steps}"
    out_dir    = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history    = []
    # Temporal loss state: track previous delta + its video/frame identity
    prev_delta:     Optional[torch.Tensor] = None
    prev_vid_key:   Optional[str]          = None   # "videoname:frame_idx"
    pbar = tqdm(range(1, args.num_steps + 1), desc=f"Stage{args.stage}")
    for step in pbar:
        item = random.choice(frame_pool)
        if item["mask_np"].sum() < 50:
            continue
        coords, labels = get_centroid_prompt(item["mask_np"])
        x01 = attacker.encode_image(item["frame_np"])
        g_theta.train()
        if args.stage == 4:
            x_adv, delta, _ = g_theta(x01)
        else:
            x_adv, delta = g_theta(x01)
        if args.stage >= 3 and random.random() < args.eot_prob:
            x_adv_in = codec_proxy_transform(x_adv, p_apply=1.0)
        else:
            x_adv_in = x_adv
        logits = attacker(x_adv_in, coords, labels)
        gt = torch.from_numpy(item["mask_np"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = torch.nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        l_attack = soft_iou_loss(logits, gt)
        l_perc   = perc_fn(x01, x_adv)
        loss     = l_attack + args.lambda1 * l_perc
        # Stage 2+: temporal consistency — only between adjacent frames in the same video
        l_temp   = torch.zeros(1, device=device)
        if args.stage >= 2 and step % 10 == 0 and prev_delta is not None:
            curr_key = f"{item['video']}:{item['frame_idx']}"
            prev_expected = f"{item['video']}:{item['frame_idx'] - 1}"
            if prev_vid_key == prev_expected and prev_delta.shape == delta.shape:
                l_temp = temporal_loss([prev_delta, delta])
                loss   = loss + args.lambda2 * l_temp
        prev_delta   = delta.detach()
        prev_vid_key = f"{item['video']}:{item['frame_idx']}"
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g_theta.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            history.append({
                "step": step, "loss": loss.item(),
                "l_attack": l_attack.item(), "l_perc": l_perc.item(),
                "l_temp": l_temp.item(), "lr": scheduler.get_last_lr()[0],
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}", att=f"{l_attack.item():.4f}")
        if step % args.eval_every == 0:
            # Eval on val videos (smoke-test; no memory bank)
            ev = eval_quick(attacker, g_theta, DAVIS_MINI_VAL[:3], args.davis_root, device)
            if history:
                history[-1].update(ev)
            print(f"\n  step={step} JF_clean={ev['mean_jf_clean']:.3f} "
                  f"JF_adv={ev['mean_jf_adv']:.3f} dJF={ev['delta_jf']:.3f}")
            # B0 sanity gate: must meet plan's threshold of ≥30% J&F drop on overfit video
            if args.sanity and step == 500:
                sanity_threshold = 0.30
                if ev["delta_jf"] < sanity_threshold:
                    print(f"\n[SANITY FAIL] dJF={ev['delta_jf']:.3f} < {sanity_threshold} at step 500.")
                    save_results(out_dir, run_name, history, vars(args))
                    sys.exit(1)
                else:
                    print(f"\n[SANITY PASS] dJF={ev['delta_jf']:.3f} >= {sanity_threshold}")
                    if args.num_steps == 500:
                        break
        if step % args.save_every == 0:
            torch.save(g_theta.state_dict(), os.path.join(out_dir, f"g_theta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))
    torch.save(g_theta.state_dict(), os.path.join(out_dir, "g_theta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[OURS] Done -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    default="ours", choices=["ours", "uap"])
    p.add_argument("--stage",   type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--sanity",  action="store_true")
    p.add_argument("--davis_root",  default=DAVIS_ROOT)
    p.add_argument("--videos",      default=None)
    p.add_argument("--max_frames",  type=int, default=30)
    p.add_argument("--channels",    type=int,   default=32)
    p.add_argument("--num_blocks",  type=int,   default=4)
    p.add_argument("--max_delta",   type=float, default=8.0/255.0)
    p.add_argument("--num_steps",   type=int,   default=3000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--uap_lr",      type=float, default=None,
                   help="UAP-specific lr (default: same as --lr)")
    p.add_argument("--optimizer",   default="adam", choices=["adam", "adamw"],
                   help="Optimizer for g_theta (Stage 2+ defaults to adamw)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--lambda1",     type=float, default=1.0)
    p.add_argument("--lambda2",     type=float, default=0.1)
    p.add_argument("--lambda3",     type=float, default=0.05)
    p.add_argument("--max_lpips",   type=float, default=0.10)
    p.add_argument("--uap_lpips",   action="store_true",
                   help="Add LPIPS hinge to UAP baseline (fair-budget Anti-C baseline)")
    p.add_argument("--eot_prob",    type=float, default=0.5)
    p.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",     default=SAM2_CONFIG)
    p.add_argument("--save_dir",    default=RESULTS_DIR)
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--tag",         default=None)
    p.add_argument("--log_every",   type=int, default=10)
    p.add_argument("--eval_every",  type=int, default=500)
    p.add_argument("--save_every",  type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
    video_names = [v.strip() for v in args.videos.split(",")] if args.videos else DAVIS_MINI_TRAIN
    print(f"Videos: {video_names}")
    frame_pool = build_frame_pool(video_names, args.davis_root, args.max_frames)
    if not frame_pool:
        print("[ERROR] No frames loaded.")
        sys.exit(1)
    print(f"Frame pool: {len(frame_pool)} frames")
    print("Loading SAM2...")
    attacker = SAM2Attacker(args.sam2_checkpoint, args.sam2_config, device)
    attacker.eval()
    if args.mode == "uap":
        train_uap(args, attacker, frame_pool, device)
    else:
        train_ours(args, attacker, frame_pool, device)


if __name__ == "__main__":
    main()
