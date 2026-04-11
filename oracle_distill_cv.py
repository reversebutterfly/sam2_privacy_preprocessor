"""
oracle_distill_cv.py — Repeated Held-Out Cross-Validation

Stress-test the SectorPredictor by running multiple random train/test splits
and reporting closure distribution. This addresses the n=4 split-instability
risk identified in Round 3 review.

For each split:
  1. Pick K test videos uniformly at random
  2. Train on the remaining (16 - K)
  3. Evaluate on test set with real H.264 + SAM2
  4. Record per-video closure ratios

Aggregate over splits:
  - Mean / median / std closure
  - Worst-split closure
  - Per-video closure across splits (where each video shows up in multiple test sets)
  - Win-rate distribution

Usage:
  python oracle_distill_cv.py \\
      --result_dirs results_v100/oracle_gap/oracle_strict_p1,results_v100/oracle_gap/oracle_strict_p2,\\
                    results_v100/oracle_gap/oracle_strict_p3,results_v100/oracle_gap/oracle_strict_p4 \\
      --n_splits 5 --test_size 4 --seed 0 --device cuda --tag cv_v1
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

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import build_predictor
from oracle_distill import (
    extract_features, SectorPredictor, train_predictor,
    predict_alphas, evaluate_predictor, load_oracle_dataset,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dirs", required=True,
                   help="Comma-separated list of oracle gap result.json files (combined dataset)")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="cv_v1")
    p.add_argument("--save_dir", default="results_v100/oracle_distill_cv")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--two_stage", action="store_true",
                   help="Use TwoStageSectorPredictor with gate (fallback to uniform on uncertain videos)")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    paths = [p.strip() for p in args.result_dirs.split(",") if p.strip()]
    print(f"=== Oracle Distillation CV ===")
    print(f"Result paths: {paths}")
    print(f"Splits: {args.n_splits}  test_size: {args.test_size}")

    all_records = load_oracle_dataset(paths)
    print(f"Loaded {len(all_records)} videos total")

    if len(all_records) < args.test_size + 4:
        print("Not enough videos for CV")
        return

    save_dir = Path(ROOT) / args.save_dir / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading SAM2 predictor...")
    sam2 = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, torch.device(args.device))

    n = len(all_records)
    all_results = []
    per_video_closures = {r["video"]: [] for r in all_records}

    for split_idx in range(args.n_splits):
        print(f"\n{'='*60}")
        print(f"SPLIT {split_idx+1}/{args.n_splits}")
        print(f"{'='*60}")

        # Random shuffle and split
        idx = rng.permutation(n)
        test_idx = idx[:args.test_size]
        train_idx = idx[args.test_size:]

        train_records = [all_records[i] for i in train_idx]
        test_records = [all_records[i] for i in test_idx]
        test_videos = [r["video"] for r in test_records]
        print(f"Train: {len(train_records)} videos")
        print(f"Test:  {test_videos}")

        # Train predictor
        model = train_predictor(train_records, epochs=args.epochs,
                                lr=args.lr, device=args.device,
                                two_stage=getattr(args, 'two_stage', False))

        # Evaluate held-out
        results = evaluate_predictor(
            model, test_records, sam2, args.device, FFMPEG_PATH,
            crf=23, max_frames=args.max_frames,
        )

        if not results:
            print("[skip split] no results")
            continue

        closures = [r["closure"] for r in results if not np.isnan(r["closure"])]
        wins = sum(1 for r in results if r["learned_gain"] > 0)
        agg = {
            "split": split_idx,
            "test_videos": test_videos,
            "n": len(results),
            "mean_closure": float(np.mean(closures)),
            "median_closure": float(np.median(closures)),
            "win_rate": wins / len(results),
            "mean_brs_pp": float(np.mean([r["brs_delta_jf"] for r in results])) * 100,
            "mean_learned_pp": float(np.mean([r["learned_delta_jf"] for r in results])) * 100,
            "mean_oracle_pp": float(np.mean([r["oracle_delta_jf"] for r in results])) * 100,
            "per_video": results,
        }
        all_results.append(agg)

        for r in results:
            per_video_closures[r["video"]].append(r["closure"])

        print(f"\nSplit {split_idx+1} aggregate:")
        print(f"  Mean closure: {agg['mean_closure']*100:.1f}%")
        print(f"  Median closure: {agg['median_closure']*100:.1f}%")
        print(f"  Win-rate: {agg['win_rate']*100:.1f}%")
        print(f"  BRS: {agg['mean_brs_pp']:.1f}pp  Learned: {agg['mean_learned_pp']:.1f}pp  "
              f"Oracle: {agg['mean_oracle_pp']:.1f}pp")

    # Aggregate over splits
    if all_results:
        all_closures_per_split = [a["mean_closure"] for a in all_results]
        all_winrates = [a["win_rate"] for a in all_results]
        worst_split_closure = min(all_closures_per_split)
        best_split_closure = max(all_closures_per_split)

        # Per-video closure (videos that appeared in test sets)
        per_video_summary = {}
        for vid, c_list in per_video_closures.items():
            if c_list:
                per_video_summary[vid] = {
                    "n_appearances": len(c_list),
                    "mean_closure": float(np.mean(c_list)),
                    "min_closure": float(np.min(c_list)),
                    "max_closure": float(np.max(c_list)),
                }

        cv_agg = {
            "n_splits": len(all_results),
            "mean_split_closure": float(np.mean(all_closures_per_split)),
            "median_split_closure": float(np.median(all_closures_per_split)),
            "std_split_closure": float(np.std(all_closures_per_split)),
            "worst_split_closure": float(worst_split_closure),
            "best_split_closure": float(best_split_closure),
            "mean_split_winrate": float(np.mean(all_winrates)),
            "min_split_winrate": float(np.min(all_winrates)),
            "per_video_closures": per_video_summary,
        }

        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION AGGREGATE ({len(all_results)} splits)")
        print(f"{'='*60}")
        print(f"  Mean split closure: {cv_agg['mean_split_closure']*100:.1f}% "
              f"± {cv_agg['std_split_closure']*100:.1f}%")
        print(f"  Median split closure: {cv_agg['median_split_closure']*100:.1f}%")
        print(f"  WORST split closure: {cv_agg['worst_split_closure']*100:.1f}%")
        print(f"  Best split closure: {cv_agg['best_split_closure']*100:.1f}%")
        print(f"  Mean split win-rate: {cv_agg['mean_split_winrate']*100:.1f}%")
        print(f"  Min split win-rate: {cv_agg['min_split_winrate']*100:.1f}%")

        out = {
            "args": vars(args),
            "cv_aggregate": cv_agg,
            "per_split": all_results,
        }
        with open(save_dir / "results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
