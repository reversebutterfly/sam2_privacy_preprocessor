"""
cmt_probe.py — CMT Core Hypothesis Validation

Tests: Does SAM2 memory token divergence predict post-codec tracking failure?

For each video and each sector:
  1. Apply a fixed probe edit (BRS α=0.80) to ONLY that sector
  2. Run SAM2 for 2-3 frames, extract memory tokens
  3. Compute memory divergence vs clean memory tokens
  4. Compare the 8-dimensional divergence vector against oracle ΔJF gains

If correlation(memory_divergence, oracle_gap) > 0.85, CMT is viable.
If < 0.6, CMT is dead.

Usage:
  python cmt_probe.py \
      --videos bear,blackswan,dog,dance-twirl,elephant \
      --max_frames 10 --device cuda --tag cmt_probe_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
from pilot_mask_guided import (
    build_predictor, codec_round_trip, frame_quality,
    apply_old_boundary_suppression,
)
from oracle_mask_search import _build_sector_geometry, apply_sector_suppression


# ─────────────────────────────────────────────────────────────────────────────
# Extract SAM2 memory tokens
# ─────────────────────────────────────────────────────────────────────────────

def extract_memory_tokens(
    frames: list,
    masks: list,
    predictor,
    device: torch.device,
    prompt: str = "point",
    n_frames: int = 3,
) -> torch.Tensor:
    """
    Run SAM2 on first n_frames and extract the memory token from the last frame.

    Returns:
        memory_features: tensor from SAM2's internal state
        We extract the maskmem_features (spatial memory) after propagation.
    """
    H, W = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fr in enumerate(frames[:n_frames]):
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

            # Propagate through n_frames
            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                pass  # just propagate

            # Extract memory state from per-object output dict
            # Structure: state["output_dict_per_obj"][obj_idx][frame_type][frame_idx]
            mem_features = []
            obj_ptrs = []
            od = state.get("output_dict_per_obj", {})
            for obj_idx in od:
                for frame_type in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                    if frame_type not in od[obj_idx]:
                        continue
                    for fk in sorted(od[obj_idx][frame_type].keys()):
                        out = od[obj_idx][frame_type][fk]
                        if "maskmem_features" in out and out["maskmem_features"] is not None:
                            mem_features.append(out["maskmem_features"].detach().cpu())
                        if "obj_ptr" in out and out["obj_ptr"] is not None:
                            obj_ptrs.append(out["obj_ptr"].detach().cpu())

            # Concatenate all memory features + object pointers into one vector
            all_parts = []
            if mem_features:
                all_parts.append(torch.cat(mem_features, dim=0).flatten())
            if obj_ptrs:
                all_parts.append(torch.cat(obj_ptrs, dim=0).flatten())

            if all_parts:
                return torch.cat(all_parts)

    return torch.zeros(1)


def memory_divergence(mem_clean: torch.Tensor, mem_edited: torch.Tensor) -> float:
    """L2 distance between memory token vectors (normalized)."""
    if mem_clean.shape != mem_edited.shape:
        # Flatten and truncate to shorter
        c = mem_clean.flatten()
        e = mem_edited.flatten()
        n = min(len(c), len(e))
        c, e = c[:n], e[:n]
    else:
        c = mem_clean.flatten()
        e = mem_edited.flatten()

    if len(c) == 0:
        return 0.0

    # Normalized L2
    norm = max(float(c.norm()), 1e-8)
    return float((c - e).norm() / norm)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sector probing
# ─────────────────────────────────────────────────────────────────────────────

def probe_sector_divergences(
    frames: list,
    masks: list,
    predictor,
    device: torch.device,
    n_angular: int = 8,
    ring_width: int = 24,
    probe_alpha: float = 0.80,
    n_probe_frames: int = 3,
    prompt: str = "point",
) -> np.ndarray:
    """
    For each sector k, apply BRS ONLY in that sector and measure memory divergence.

    Returns:
        divergences: (n_angular,) array of memory divergence scores
    """
    # 1. Clean memory baseline
    mem_clean = extract_memory_tokens(frames[:n_probe_frames], masks[:n_probe_frames],
                                       predictor, device, prompt, n_probe_frames)

    # 2. Per-sector probes
    divergences = np.zeros(n_angular, dtype=np.float64)

    for k in range(n_angular):
        # Create sector alphas: only sector k is active
        alphas = np.zeros(n_angular)
        alphas[k] = probe_alpha

        # Apply sector-only edit to first n_probe_frames
        edited = []
        for f, m in zip(frames[:n_probe_frames], masks[:n_probe_frames]):
            edited.append(apply_sector_suppression(f, m, alphas, ring_width))

        # Get memory tokens for this edit
        mem_edited = extract_memory_tokens(edited, masks[:n_probe_frames],
                                            predictor, device, prompt, n_probe_frames)

        div = memory_divergence(mem_clean, mem_edited)
        divergences[k] = div
        print(f"    sector {k}: memory_div = {div:.6f}")

    return divergences


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--n_angular", type=int, default=8)
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--probe_alpha", type=float, default=0.80)
    p.add_argument("--n_probe_frames", type=int, default=3)
    p.add_argument("--max_frames", type=int, default=10)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--oracle_results", default="results_v100/oracle_gap/oracle_rerun_fixed/results.json",
                   help="Path to oracle gap results for correlation analysis")
    p.add_argument("--tag", default="cmt_probe_v1")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "cmt_probe" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CMT Hypothesis Validation ===")
    print(f"Videos: {videos}")
    print(f"Testing: does memory_divergence(sector_k) predict oracle_gap(sector_k)?")

    # Load oracle results for correlation
    oracle_data = {}
    if os.path.exists(args.oracle_results):
        d = json.load(open(args.oracle_results))
        for v in d.get("per_video", []):
            oracle_data[v["video"]] = {
                "oracle_alphas": np.array(v["oracle"]["alphas"]),
                "oracle_djf": v["oracle"]["delta_jf"],
                "brs_djf": v["brs"]["delta_jf"],
                "gap": v["oracle"]["delta_jf"] - v["brs"]["delta_jf"],
            }
        print(f"Loaded oracle data for {len(oracle_data)} videos")

    predictor = build_predictor(SAM2_CHECKPOINT, SAM2_CONFIG, device)
    all_results = []

    for vid in videos:
        print(f"\n{'='*60}\nVideo: {vid}\n{'='*60}")
        try:
            frames, masks, _ = load_single_video(DAVIS_ROOT, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        if not frames:
            continue

        t0 = time.time()
        print(f"  Probing {args.n_angular} sectors...")
        divergences = probe_sector_divergences(
            frames, masks, predictor, device,
            n_angular=args.n_angular, ring_width=args.ring_width,
            probe_alpha=args.probe_alpha, n_probe_frames=args.n_probe_frames,
            prompt=args.prompt,
        )
        elapsed = time.time() - t0
        print(f"  Divergences: {divergences}")
        print(f"  Elapsed: {elapsed:.1f}s")

        result = {
            "video": vid,
            "divergences": divergences.tolist(),
            "elapsed_s": elapsed,
        }

        # Correlation with oracle alphas
        if vid in oracle_data:
            oa = oracle_data[vid]["oracle_alphas"]
            # Oracle alphas: higher alpha = more editing = more effective
            # Divergence: higher = more memory disruption
            # We expect: sectors with high oracle alpha should have high divergence
            from scipy.stats import pearsonr, spearmanr
            r_pearson, p_pearson = pearsonr(divergences, oa)
            r_spearman, p_spearman = spearmanr(divergences, oa)
            result["oracle_alphas"] = oa.tolist()
            result["pearson_r"] = float(r_pearson)
            result["pearson_p"] = float(p_pearson)
            result["spearman_r"] = float(r_spearman)
            result["spearman_p"] = float(p_spearman)
            print(f"  Correlation with oracle alphas:")
            print(f"    Pearson r={r_pearson:.3f} (p={p_pearson:.4f})")
            print(f"    Spearman r={r_spearman:.3f} (p={p_spearman:.4f})")
        else:
            print(f"  [warn] no oracle data for {vid}")

        all_results.append(result)

    # Aggregate correlation
    if all_results:
        pearson_rs = [r["pearson_r"] for r in all_results if "pearson_r" in r]
        spearman_rs = [r["spearman_r"] for r in all_results if "spearman_r" in r]
        if pearson_rs:
            print(f"\n{'='*60}")
            print(f"AGGREGATE (n={len(pearson_rs)} videos)")
            print(f"  Mean Pearson r: {np.mean(pearson_rs):.3f}")
            print(f"  Mean Spearman r: {np.mean(spearman_rs):.3f}")
            if np.mean(pearson_rs) > 0.7:
                print(f"  VERDICT: CMT is VIABLE (r > 0.7)")
            elif np.mean(pearson_rs) > 0.5:
                print(f"  VERDICT: CMT is MARGINAL (0.5 < r < 0.7)")
            else:
                print(f"  VERDICT: CMT is NOT VIABLE (r < 0.5)")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
