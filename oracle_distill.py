"""
oracle_distill.py — Held-out Oracle Distillation Test

Trains a tiny sector-alpha predictor on oracle alphas from a TRAIN set,
evaluates on a HELD-OUT TEST set, and reports oracle gap closure:

    closure = (learned_ΔJF − BRS_ΔJF) / (oracle_ΔJF − BRS_ΔJF)

Architecture (intentionally minimal):
    Input features (per video, computed from first GT mask + RGB):
        - Mask area / image area
        - Mask aspect ratio
        - Per-sector ring area (8 dims)
        - Per-sector mean RGB (8 × 3 dims)
        - Per-sector mean gradient magnitude (8 dims)

    Predictor: 2-layer MLP → 8 sector logits → softmax × 8 (mean=1) → α scaled

Output α is projected onto the same iso-budget constraint used during oracle search.

Usage:
  python oracle_distill.py \\
      --train_results results_v100/oracle_gap/oracle_strict_p1/results.json,results_v100/oracle_gap/oracle_strict_p2/results.json \\
      --test_results  results_v100/oracle_gap/oracle_strict_p3/results.json \\
      --epochs 200 --device cuda --tag distill_v1

Reports per-test-video closure ratio + aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import (
    build_predictor, run_tracking, codec_round_trip, frame_quality,
    apply_old_boundary_suppression,
)
from oracle_mask_search import (
    apply_sector_suppression, apply_sector_suppression_frames,
    _build_sector_geometry, project_to_budget, evaluate_oracle, brs_baseline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-video feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    n_angular: int = 8,
    frame_rgb_next: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract per-video feature vector from first frame (+ optional second frame for motion).

    Features (total = 4 + 7*n_angular):
      Global (4): mask_area_ratio, aspect_ratio, mask_compactness, boundary_length_ratio
      Per-sector (7 * n_angular):
        - ring area fraction (n_angular)
        - mean RGB in ring (3 * n_angular)
        - mean gradient magnitude in ring (n_angular)
        - boundary curvature proxy: mean normal direction variance (n_angular)
        - motion magnitude: mean optical flow in ring sector (n_angular) — 0 if no next frame

    Returns:
        feat: 1D float32 array
    """
    H, W = mask.shape
    img_area = H * W
    mask_area = float(mask.sum())

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros(4 + 7 * n_angular, dtype=np.float32)

    # Global features
    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1
    aspect = float(bbox_w) / max(bbox_h, 1)
    # Compactness: 4π·area / perimeter²  (1.0 for circle)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) if contours else 1.0
    compactness = float(4.0 * 3.14159 * mask_area / max(perimeter ** 2, 1.0))
    boundary_length_ratio = float(perimeter) / max(2 * (bbox_h + bbox_w), 1.0)

    cy, cx = float(ys.mean()), float(xs.mean())

    # Sector indicator
    Y, X = np.mgrid[0:H, 0:W]
    angle = np.arctan2(Y - cy, X - cx)
    angle_norm = (angle + np.pi) / (2.0 * np.pi)
    a_idx = np.clip((angle_norm * n_angular).astype(int), 0, n_angular - 1)

    # Boundary ring
    rw = 24
    kernel = np.ones((rw * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    boundary_ring = ((dilated > 0) & (eroded == 0))

    # Gradient (image gradients → boundary contrast)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    # Gradient direction (for curvature proxy)
    grad_dir = np.arctan2(gy, gx + 1e-8)

    # Optical flow (motion) — Farneback between frame0 and frame1
    flow_mag_map = np.zeros((H, W), dtype=np.float32)
    if frame_rgb_next is not None:
        gray0 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame_rgb_next, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_mag_map = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Per-sector features
    ring_total = max(boundary_ring.sum(), 1)
    sector_areas = np.zeros(n_angular, dtype=np.float32)
    sector_rgb = np.zeros((n_angular, 3), dtype=np.float32)
    sector_grad = np.zeros(n_angular, dtype=np.float32)
    sector_curvature = np.zeros(n_angular, dtype=np.float32)  # gradient direction variance
    sector_motion = np.zeros(n_angular, dtype=np.float32)

    for k in range(n_angular):
        sec_mask = (a_idx == k) & boundary_ring
        cnt = sec_mask.sum()
        sector_areas[k] = float(cnt) / ring_total
        if cnt > 0:
            sector_rgb[k] = frame_rgb[sec_mask].mean(axis=0) / 255.0
            sector_grad[k] = float(grad_mag[sec_mask].mean()) / 255.0
            sector_curvature[k] = float(grad_dir[sec_mask].std())  # high variance = high curvature
            sector_motion[k] = float(flow_mag_map[sec_mask].mean()) / max(float(flow_mag_map.max()), 1.0)

    feat = np.concatenate([
        np.array([mask_area / img_area, aspect, compactness, boundary_length_ratio],
                 dtype=np.float32),
        sector_areas,
        sector_rgb.flatten(),
        sector_grad,
        sector_curvature,
        sector_motion,
    ])
    return feat.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor model
# ─────────────────────────────────────────────────────────────────────────────

class SectorPredictor(nn.Module):
    """
    Tiny MLP: features → 8 sector logits → projected to budget.
    """
    def __init__(self, in_dim: int, n_angular: int = 8, hidden: int = 64):
        super().__init__()
        self.n_angular = n_angular
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, n_angular),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        Returns raw logits (B, n_angular). Caller must project onto budget.
        """
        return self.net(x)


class TwoStageSectorPredictor(nn.Module):
    """
    Two-stage predictor:
      Stage 1 (gate): classify if this video benefits from anisotropic suppression
                       → if confident NO, fall back to uniform α (safe default)
      Stage 2 (sector): predict 8 sector alphas for positive cases

    This avoids catastrophic losses on "difficult regime" videos where the
    predictor would otherwise output harmful patterns.
    """
    def __init__(self, in_dim: int, n_angular: int = 8, hidden: int = 64):
        super().__init__()
        self.n_angular = n_angular
        # Gate: features → 1 (probability that anisotropic > uniform)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )
        # Sector predictor (same as before)
        self.sector = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, n_angular),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sector_logits, gate_prob)."""
        gate_logit = self.gate(x)  # (B, 1)
        gate_prob = torch.sigmoid(gate_logit)  # (B, 1)
        sector_logits = self.sector(x)  # (B, n_angular)
        return sector_logits, gate_prob


def torch_project_to_budget(
    alphas: torch.Tensor,
    target: float,
    ring_areas: torch.Tensor,
    n_iter: int = 30,
) -> torch.Tensor:
    """Differentiable iso-budget projection (clip + shift)."""
    a = torch.clamp(alphas, 0.0, 1.0)
    total_area = ring_areas.sum()
    if total_area < 1e-8:
        return a  # degenerate mask — no projection possible
    for _ in range(n_iter):
        weighted_mean = (ring_areas * a).sum() / total_area
        diff = target - weighted_mean
        if torch.abs(diff) < 1e-5:
            break
        a = torch.clamp(a + diff, 0.0, 1.0)
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Data loading from oracle search results
# ─────────────────────────────────────────────────────────────────────────────

def load_oracle_dataset(result_paths: list[str]) -> list[dict]:
    """
    Load per-video oracle search results into a flat list of training records.
    Each record:
        {video, oracle_alphas, brs_delta_jf, oracle_delta_jf, ring_areas,
         features (computed), feature_dim}
    """
    records = []
    for path in result_paths:
        if not os.path.exists(path):
            print(f"  [warn] {path} not found")
            continue
        d = json.load(open(path))
        per_video = d.get("per_video", [])
        for vid_r in per_video:
            vid = vid_r["video"]
            try:
                frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=3)
            except Exception as e:
                print(f"  [skip] cannot load video {vid}: {e}")
                continue
            if not frames or masks[0].sum() == 0:
                continue
            # Extract features with motion (if 2nd frame available)
            frame_next = frames[1] if len(frames) > 1 else None
            feat = extract_features(frames[0], masks[0], n_angular=8,
                                    frame_rgb_next=frame_next)
            oracle_alphas = np.array(vid_r["oracle"]["alphas"], dtype=np.float32)
            n_sectors = len(oracle_alphas)
            if n_sectors != 8:
                print(f"  [skip] {vid}: oracle has {n_sectors} sectors, expected 8")
                continue
            # Save source config for protocol consistency verification (Bug #2 fix)
            source_args = d.get("args", {})
            records.append({
                "video": vid,
                "oracle_alphas": oracle_alphas,
                "brs_delta_jf": float(vid_r["brs"]["delta_jf"]),
                "oracle_delta_jf": float(vid_r["oracle"]["delta_jf"]),
                "ring_areas": np.array(vid_r["ring_areas"], dtype=np.float32),
                "target_budget": float(vid_r.get("target_budget", 0.80)),
                "features": feat,
                "source_ring_width": source_args.get("ring_width", 24),
                "source_prompt": source_args.get("prompt", "point"),
                "source_max_frames": source_args.get("max_frames", 50),
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_predictor(
    train_records: list[dict],
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    two_stage: bool = False,
) -> SectorPredictor | TwoStageSectorPredictor:
    if not train_records:
        raise ValueError("No training records")

    in_dim = len(train_records[0]["features"])
    if two_stage:
        model = TwoStageSectorPredictor(in_dim, n_angular=8, hidden=64).to(device)
    else:
        model = SectorPredictor(in_dim, n_angular=8, hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    feats = torch.from_numpy(np.stack([r["features"] for r in train_records])).to(device)
    targets = torch.from_numpy(np.stack([r["oracle_alphas"] for r in train_records])).to(device)
    ring_areas_t = [torch.from_numpy(r["ring_areas"]).to(device) for r in train_records]
    targets_budget = [r["target_budget"] for r in train_records]

    # Gate labels: 1 if oracle is meaningfully better than BRS (gap > 5pp)
    if two_stage:
        gate_labels = torch.tensor(
            [1.0 if (r["oracle_delta_jf"] - r["brs_delta_jf"]) > 0.05 else 0.0
             for r in train_records],
            device=device,
        )

    for ep in range(epochs):
        opt.zero_grad()

        if two_stage:
            logits, gate_prob = model(feats)  # (N, 8), (N, 1)
            # Sector loss: MSE on projected alphas
            sector_loss = 0.0
            for i in range(len(train_records)):
                pred = torch.sigmoid(logits[i])
                pred_proj = torch_project_to_budget(pred, targets_budget[i], ring_areas_t[i])
                sector_loss = sector_loss + F.mse_loss(pred_proj, targets[i])
            sector_loss = sector_loss / len(train_records)
            # Gate loss: BCE
            gate_loss = F.binary_cross_entropy(
                gate_prob.squeeze(-1), gate_labels
            )
            loss = sector_loss + 0.5 * gate_loss
        else:
            logits = model(feats)
            loss = 0.0
            for i in range(len(train_records)):
                pred = torch.sigmoid(logits[i])
                pred_proj = torch_project_to_budget(pred, targets_budget[i], ring_areas_t[i])
                loss = loss + F.mse_loss(pred_proj, targets[i])
            loss = loss / len(train_records)

        loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  [epoch {ep+1:3d}] loss={loss.item():.4f}")

    return model


@torch.no_grad()
def predict_alphas(model, features: np.ndarray,
                   ring_areas: np.ndarray, target_budget: float, device: str,
                   gate_threshold: float = 0.5) -> tuple[np.ndarray, float]:
    """
    Returns (alphas, gate_prob).
    If model is TwoStageSectorPredictor and gate_prob < threshold,
    returns uniform alphas (safe fallback).
    """
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    ring_areas_t = torch.from_numpy(ring_areas).to(device)

    if isinstance(model, TwoStageSectorPredictor):
        logits, gate_prob_t = model(feat_t)
        gate_prob = float(gate_prob_t[0, 0].cpu())
        if gate_prob < gate_threshold:
            # Fallback: uniform α = target_budget (safe = BRS equivalent)
            return np.full(len(ring_areas), target_budget, dtype=np.float32), gate_prob
        pred = torch.sigmoid(logits[0])
        pred_proj = torch_project_to_budget(pred, target_budget, ring_areas_t)
        return pred_proj.cpu().numpy(), gate_prob
    else:
        logits = model(feat_t)[0]
        pred = torch.sigmoid(logits)
        pred_proj = torch_project_to_budget(pred, target_budget, ring_areas_t)
        return pred_proj.cpu().numpy(), 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Held-out evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictor(
    model: SectorPredictor,
    test_records: list[dict],
    predictor_sam2,
    device: str,
    ffmpeg_path: str,
    crf: int = 23,
    max_frames: int = 50,
) -> list[dict]:
    results = []
    for rec in test_records:
        vid = rec["video"]
        try:
            frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=max_frames)
        except Exception as e:
            print(f"  [skip] {vid}: {e}")
            continue
        frames = frames[:max_frames]
        masks = masks[:max_frames]
        if not frames or masks[0].sum() == 0:
            continue

        # Recompute first-frame features with motion
        frame_next = frames[1] if len(frames) > 1 else None
        feat = extract_features(frames[0], masks[0], n_angular=8,
                                frame_rgb_next=frame_next)
        ring_areas = rec["ring_areas"]
        target_budget = rec["target_budget"]

        # 1. BRS baseline (uniform 0.80)
        try:
            brs_dj, brs_ssim, jfcc = brs_baseline(
                frames, masks, predictor_sam2, torch.device(device), ffmpeg_path,
                crf=crf,
                ring_width=rec.get("source_ring_width", 24),
                alpha=0.80,
                prompt=rec.get("source_prompt", "point"),
            )
        except Exception as e:
            print(f"  [{vid}] BRS error: {e}")
            continue

        # 2. Predictor inference
        learned_alphas, gate_prob = predict_alphas(model, feat, ring_areas, target_budget, device)
        rw = rec.get("source_ring_width", 24)
        edited = apply_sector_suppression_frames(
            frames, masks, learned_alphas, rw, target_budget=target_budget)
        learned_dj, learned_ssim, _ = evaluate_oracle(
            edited, frames, masks, predictor_sam2, torch.device(device), ffmpeg_path,
            crf=crf, prompt=rec.get("source_prompt", "point"), jf_codec_clean=jfcc,
        )

        # 3. Oracle (from training data, recorded value)
        oracle_dj = rec["oracle_delta_jf"]

        gap = oracle_dj - brs_dj
        learned_gain = learned_dj - brs_dj
        closure = (learned_gain / gap) if abs(gap) > 1e-6 else float("nan")

        gated_str = f" gate={gate_prob:.2f}" if isinstance(model, TwoStageSectorPredictor) else ""
        print(f"  [{vid}] BRS={brs_dj*100:.1f}pp Learned={learned_dj*100:.1f}pp "
              f"Oracle={oracle_dj*100:.1f}pp closure={closure*100:.1f}% "
              f"SSIM_l={learned_ssim:.3f}{gated_str}")

        results.append({
            "video": vid,
            "brs_delta_jf": float(brs_dj),
            "brs_ssim": float(brs_ssim),
            "learned_delta_jf": float(learned_dj),
            "learned_ssim": float(learned_ssim),
            "oracle_delta_jf": float(oracle_dj),
            "gap": float(gap),
            "learned_gain": float(learned_gain),
            "closure": float(closure),
            "learned_alphas": learned_alphas.tolist(),
            "oracle_alphas": rec["oracle_alphas"].tolist(),
            "gate_prob": float(gate_prob),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_results", required=True,
                   help="Comma-separated paths to oracle gap result.json files (training set)")
    p.add_argument("--test_results", required=True,
                   help="Comma-separated paths to oracle gap result.json files (held-out)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="distill_v1")
    p.add_argument("--save_dir", default="results_v100/oracle_distill")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=23)
    return p.parse_args()


def main():
    args = parse_args()

    train_paths = [p.strip() for p in args.train_results.split(",") if p.strip()]
    test_paths = [p.strip() for p in args.test_results.split(",") if p.strip()]

    print(f"=== Oracle Distillation Test ===")
    print(f"Train results: {train_paths}")
    print(f"Test results: {test_paths}")

    train_records = load_oracle_dataset(train_paths)
    test_records = load_oracle_dataset(test_paths)
    print(f"\nLoaded {len(train_records)} train, {len(test_records)} test records")

    # Bug #3 fix: check for train/test overlap
    train_videos = {r["video"] for r in train_records}
    test_videos = {r["video"] for r in test_records}
    overlap = train_videos & test_videos
    if overlap:
        print(f"  [ERROR] Train/test overlap detected: {sorted(overlap)}")
        print(f"  Removing overlapping videos from test set.")
        test_records = [r for r in test_records if r["video"] not in overlap]
        print(f"  After dedup: {len(test_records)} test records")

    if len(train_records) == 0 or len(test_records) == 0:
        print("Insufficient data for distillation")
        return

    # Train
    print(f"\nTraining predictor (epochs={args.epochs})...")
    model = train_predictor(train_records, epochs=args.epochs, lr=args.lr, device=args.device)

    # Evaluate held-out
    print(f"\nLoading SAM2 predictor for held-out evaluation...")
    sam2 = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, torch.device(args.device))

    print(f"\n=== Held-out Evaluation ===")
    results = evaluate_predictor(
        model, test_records, sam2, args.device, FFMPEG_PATH,
        crf=args.crf, max_frames=args.max_frames,
    )

    # Aggregate
    if results:
        closures = [r["closure"] for r in results if not np.isnan(r["closure"])]
        wins = sum(1 for r in results if r["learned_gain"] > 0)
        learned_djs = [r["learned_delta_jf"] for r in results]
        brs_djs = [r["brs_delta_jf"] for r in results]
        oracle_djs = [r["oracle_delta_jf"] for r in results]

        agg = {
            "n": len(results),
            "mean_brs_delta_jf_pp": float(np.mean(brs_djs)) * 100,
            "mean_learned_delta_jf_pp": float(np.mean(learned_djs)) * 100,
            "mean_oracle_delta_jf_pp": float(np.mean(oracle_djs)) * 100,
            "mean_gap_pp": float(np.mean([r["gap"] for r in results])) * 100,
            "mean_learned_gain_pp": float(np.mean([r["learned_gain"] for r in results])) * 100,
            "mean_closure": float(np.mean(closures)),
            "win_rate": wins / len(results),
        }

        print(f"\n{'='*60}")
        print(f"AGGREGATE (n={agg['n']} held-out videos)")
        print(f"  BRS:     {agg['mean_brs_delta_jf_pp']:.2f}pp")
        print(f"  Learned: {agg['mean_learned_delta_jf_pp']:.2f}pp")
        print(f"  Oracle:  {agg['mean_oracle_delta_jf_pp']:.2f}pp")
        print(f"  Mean gap (oracle - BRS): {agg['mean_gap_pp']:+.2f}pp")
        print(f"  Mean learned gain: {agg['mean_learned_gain_pp']:+.2f}pp")
        print(f"  **Mean closure: {agg['mean_closure']*100:.1f}%**")
        print(f"  Win-rate vs BRS: {agg['win_rate']*100:.1f}%")
        if agg['mean_closure'] >= 0.30 and agg['win_rate'] >= 0.70:
            print(f"  Verdict: SUCCESS (closure ≥ 30%, win-rate ≥ 70%)")
        else:
            print(f"  Verdict: NOT SUCCESS")

        save_dir = Path(ROOT) / args.save_dir / args.tag
        save_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "args": vars(args),
            "aggregate": agg,
            "per_video": results,
        }
        with open(save_dir / "results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
