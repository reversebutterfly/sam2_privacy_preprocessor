"""
1-GPU profiling pilot for SAM2 Privacy Preprocessor.

Measures peak VRAM, step time, and effective throughput at a given
g_theta_size to inform V100 GPU count / batch-size decisions.

Usage:
  python profile_gpu.py --g_theta_size 512 --num_steps 50 --stage 1
  python profile_gpu.py --g_theta_size 512 --num_steps 50 --stage 3
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    SAM2_CHECKPOINT, SAM2_CONFIG,
    DAVIS_ROOT, DAVIS_MINI_TRAIN,
)
from src.preprocessor import ResidualPreprocessor
from src.losses import PerceptualLoss, soft_iou_loss
from src.codec_eot import codec_proxy_transform
from src.dataset import load_single_video
from train import SAM2Attacker, get_centroid_prompt, build_frame_pool


def profile(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] No CUDA device found. Profiling on CPU (not representative).")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total_vram = (torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                  if device.type == "cuda" else 0)
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {total_vram} GB")
    print(f"Stage: {args.stage}  g_theta_size: {args.g_theta_size}  steps: {args.num_steps}")

    # Load SAM2
    print("\nLoading SAM2...")
    attacker = SAM2Attacker(args.sam2_checkpoint, args.sam2_config, device)
    attacker.eval()

    # Load g_theta
    g_theta = ResidualPreprocessor(max_delta=8.0 / 255.0).to(device)

    _opt_cls = torch.optim.AdamW if args.stage >= 2 else torch.optim.Adam
    optimizer = _opt_cls(g_theta.parameters(), lr=1e-4, weight_decay=1e-4)
    perc_fn = PerceptualLoss(threshold=0.10, device=device)

    # Build frame pool
    video_names = [v.strip() for v in args.videos.split(",")] if args.videos else DAVIS_MINI_TRAIN
    frame_pool = build_frame_pool(video_names, args.davis_root, max_frames=30)
    if not frame_pool:
        print("[ERROR] No frames loaded.")
        sys.exit(1)
    print(f"Frame pool: {len(frame_pool)} frames from {len(video_names)} videos")

    import random
    random.seed(0)

    # Warmup
    print("\nWarmup (5 steps)...")
    for _ in range(5):
        item = random.choice(frame_pool)
        if item["mask_np"].sum() < 50:
            continue
        coords, labels = get_centroid_prompt(item["mask_np"])
        x01 = attacker.encode_image(item["frame_np"])
        gs = args.g_theta_size
        x_small = nn.functional.interpolate(x01, size=(gs, gs), mode="bilinear", align_corners=False)
        x_adv_small, delta_small = g_theta(x_small)
        delta = nn.functional.interpolate(delta_small, size=(1024, 1024), mode="bilinear", align_corners=False)
        x_adv = (x01 + delta).clamp(0, 1)
        if args.stage >= 3:
            x_adv = codec_proxy_transform(x_adv, p_apply=1.0)
        logits = attacker(x_adv, coords, labels)
        gt = torch.from_numpy(item["mask_np"].astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        loss = soft_iou_loss(logits, gt) + perc_fn(x_small, x_adv_small)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed profiling
    print(f"\nProfiling {args.num_steps} steps...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_times = []
    for step in range(args.num_steps):
        item = random.choice(frame_pool)
        if item["mask_np"].sum() < 50:
            continue
        t0 = time.perf_counter()
        coords, labels = get_centroid_prompt(item["mask_np"])
        x01 = attacker.encode_image(item["frame_np"])
        gs = args.g_theta_size
        x_small = nn.functional.interpolate(x01, size=(gs, gs), mode="bilinear", align_corners=False)
        x_adv_small, delta_small = g_theta(x_small)
        delta = nn.functional.interpolate(delta_small, size=(1024, 1024), mode="bilinear", align_corners=False)
        x_adv = (x01 + delta).clamp(0, 1)
        if args.stage >= 3:
            x_adv = codec_proxy_transform(x_adv, p_apply=1.0)
        logits = attacker(x_adv, coords, labels)
        gt = torch.from_numpy(item["mask_np"].astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        loss = soft_iou_loss(logits, gt) + perc_fn(x_small, x_adv_small)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if device.type == "cuda" else 0.0)
    mean_step_s = sum(step_times) / len(step_times)
    it_per_s    = 1.0 / mean_step_s
    free_vram   = total_vram - peak_vram_gb

    print("\n" + "=" * 55)
    print(f"  Peak VRAM usage : {peak_vram_gb:.2f} GB / {total_vram} GB")
    print(f"  Free VRAM       : {free_vram:.2f} GB")
    print(f"  Mean step time  : {mean_step_s * 1000:.1f} ms")
    print(f"  Throughput      : {it_per_s:.2f} it/s")

    # Estimate wall-clock for V100 experiments (single GPU)
    experiments = {
        "B0 Sanity   (500 steps) ": 500,
        "B1 UAP      (5000 steps)": 5000,
        "B2 Stage1   (5000 steps)": 5000,
        "B2 Stage2   (3000 steps)": 3000,
        "B3 Stage3   (3000 steps)": 3000,
    }
    print("\n  Estimated wall-clock (single GPU, current step time):")
    total_min = 0.0
    for name, n in experiments.items():
        mins = n * mean_step_s / 60
        total_min += mins
        print(f"    {name}: {mins:.0f} min")
    print(f"    {'Total (sequential)':30s}: {total_min / 60:.1f} h")

    # Recommendation
    print("\n  Recommendation:")
    if peak_vram_gb < 12:
        print(f"    peak VRAM {peak_vram_gb:.1f} GB << 16 GB V100 → 1 GPU is sufficient.")
        print(f"    Consider --g_accum_steps 4 to simulate batch-4 training.")
        print(f"    max_parallel_runs: 1 (experiments are sequential by design)")
    elif peak_vram_gb < 14:
        print(f"    peak VRAM {peak_vram_gb:.1f} GB fits on 1×V100-16GB (tight).")
        print(f"    Use --g_accum_steps 4 with care; monitor VRAM mid-run.")
    else:
        print(f"    peak VRAM {peak_vram_gb:.1f} GB is too close to 16 GB limit.")
        print(f"    Consider V100-32GB or reduce g_theta_size.")
    print("=" * 55)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",        type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--g_theta_size", type=int, default=512)
    p.add_argument("--num_steps",    type=int, default=50)
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--videos",       default=None)
    p.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",     default=SAM2_CONFIG)
    return p.parse_args()


if __name__ == "__main__":
    profile(parse_args())
