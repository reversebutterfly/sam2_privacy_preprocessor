"""
cmt_probe.py v2 — CMT Hypothesis Validation (bug-fixed)

Fixes from v1:
  1. Compare divergence vs MARGINAL post-codec ΔJF (not oracle_alphas)
  2. Add H.264 codec round-trip before SAM2 inference
  3. Extract ONLY last non-cond frame's maskmem_features (not mixed all frames)
  4. Normalize divergence by sector ring area

For each video and each sector k:
  1. Apply BRS (α=0.80) to ONLY sector k
  2. H.264 encode/decode (CRF=23)
  3. Run SAM2 on codec output, extract last-frame memory token
  4. Compute memory divergence vs clean-codec memory
  5. Also compute marginal ΔJF (post-codec tracking drop from editing only sector k)

Then correlate: memory_divergence(k) vs marginal_ΔJF(k)

Usage:
  python cmt_probe.py --videos bear,blackswan,dog --device cuda --tag cmt_v2
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
    build_predictor, run_tracking, codec_round_trip, frame_quality,
)
from oracle_mask_search import _build_sector_geometry, apply_sector_suppression


def extract_last_memory(
    frames: list,
    masks: list,
    predictor,
    device: torch.device,
    prompt: str = "point",
) -> torch.Tensor:
    """
    Run SAM2, extract ONLY the last non-cond frame's maskmem_features.
    Returns a flat tensor. Falls back to obj_ptr if maskmem not available.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fr in enumerate(frames):
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
                    labels=np.array([1], dtype=np.int32))
            else:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=first_mask)

            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                pass

            # Extract ONLY last non-cond frame's memory
            od = state.get("output_dict_per_obj", {})
            for obj_idx in od:
                ncf = od[obj_idx].get("non_cond_frame_outputs", {})
                if ncf:
                    last_fk = max(ncf.keys())
                    out = ncf[last_fk]
                    if "maskmem_features" in out and out["maskmem_features"] is not None:
                        return out["maskmem_features"].detach().cpu().flatten()
                    if "obj_ptr" in out and out["obj_ptr"] is not None:
                        return out["obj_ptr"].detach().cpu().flatten()

    return torch.zeros(1)


def memory_divergence(mem_a: torch.Tensor, mem_b: torch.Tensor) -> float:
    """Cosine distance between two memory vectors (1 - cosine_similarity)."""
    if len(mem_a) < 2 or len(mem_b) < 2:
        return 0.0
    # Ensure same length
    n = min(len(mem_a), len(mem_b))
    a, b = mem_a[:n].float(), mem_b[:n].float()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return 1.0 - cos  # 0 = identical, 2 = opposite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,dog,dance-twirl,elephant")
    p.add_argument("--n_angular", type=int, default=8)
    p.add_argument("--ring_width", type=int, default=24)
    p.add_argument("--probe_alpha", type=float, default=0.80)
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--prompt", default="point")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="cmt_v2")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    save_dir = Path(ROOT) / "results_v100" / "cmt_probe" / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CMT Hypothesis Validation v2 (bug-fixed) ===")
    print(f"Protocol: per-sector BRS probe → H.264 CRF={args.crf} → SAM2 → memory + ΔJF")

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
        frames = frames[:args.max_frames]
        masks = masks[:args.max_frames]

        # Compute sector geometry for area normalization
        geom = _build_sector_geometry(masks[0], args.n_angular, args.ring_width)
        if geom is None:
            print(f"  [skip] no geometry")
            continue
        ring_areas = geom["ring_areas"]
        area_weights = ring_areas / max(ring_areas.sum(), 1e-8)

        t0 = time.time()

        # 1. Clean baseline: codec(original) → SAM2
        print(f"  [0/{args.n_angular}] Clean codec baseline...")
        codec_clean = codec_round_trip(frames, FFMPEG_PATH, args.crf)
        if codec_clean is None:
            print(f"  [skip] codec failed")
            continue
        mem_clean = extract_last_memory(codec_clean, masks, predictor, device, args.prompt)
        _, jf_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)
        print(f"  Clean: JF={jf_clean:.4f}, mem_size={len(mem_clean)}")

        # 2. Per-sector probes: edit ONLY sector k → codec → SAM2
        divergences = np.zeros(args.n_angular, dtype=np.float64)
        marginal_djfs = np.zeros(args.n_angular, dtype=np.float64)

        for k in range(args.n_angular):
            alphas = np.zeros(args.n_angular)
            alphas[k] = args.probe_alpha

            edited = [apply_sector_suppression(f, m, alphas, args.ring_width)
                      for f, m in zip(frames, masks)]

            # Codec round-trip (matching oracle protocol)
            codec_edited = codec_round_trip(edited, FFMPEG_PATH, args.crf)
            if codec_edited is None:
                print(f"    sector {k}: codec failed")
                continue

            # Memory divergence
            mem_edited = extract_last_memory(codec_edited, masks, predictor, device, args.prompt)
            div = memory_divergence(mem_clean, mem_edited)
            # Area-normalized divergence
            div_norm = div / max(float(area_weights[k]), 1e-6)
            divergences[k] = div_norm

            # Marginal ΔJF (the TRUE label for CMT)
            _, jf_edited, _, _ = run_tracking(codec_edited, masks, predictor, device, args.prompt)
            marginal_djf = jf_clean - jf_edited
            marginal_djfs[k] = marginal_djf

            print(f"    sector {k}: div={div:.6f}  div_norm={div_norm:.4f}  "
                  f"marginal_ΔJF={marginal_djf*100:+.2f}pp  area={area_weights[k]:.3f}")

        elapsed = time.time() - t0

        # Correlation: divergence vs marginal ΔJF
        result = {
            "video": vid,
            "divergences": divergences.tolist(),
            "marginal_djfs": marginal_djfs.tolist(),
            "ring_area_weights": area_weights.tolist(),
            "jf_clean": float(jf_clean),
            "elapsed_s": elapsed,
        }

        active = marginal_djfs != 0  # skip sectors with no data
        if active.sum() >= 4:
            from scipy.stats import pearsonr, spearmanr
            d_active = divergences[active]
            m_active = marginal_djfs[active]
            r_p, p_p = pearsonr(d_active, m_active)
            r_s, p_s = spearmanr(d_active, m_active)
            result["pearson_r"] = float(r_p)
            result["pearson_p"] = float(p_p)
            result["spearman_r"] = float(r_s)
            result["spearman_p"] = float(p_s)
            print(f"\n  Correlation (div vs marginal_ΔJF):")
            print(f"    Pearson r={r_p:.3f} (p={p_p:.4f})")
            print(f"    Spearman r={r_s:.3f} (p={p_s:.4f})")

        all_results.append(result)
        with open(save_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # Aggregate
    if all_results:
        pr = [r["pearson_r"] for r in all_results if "pearson_r" in r]
        sr = [r["spearman_r"] for r in all_results if "spearman_r" in r]
        if pr:
            mean_pr = np.mean(pr)
            mean_sr = np.mean(sr)
            print(f"\n{'='*60}")
            print(f"AGGREGATE (n={len(pr)} videos)")
            print(f"  Mean Pearson r (div vs marginal_ΔJF): {mean_pr:.3f}")
            print(f"  Mean Spearman r: {mean_sr:.3f}")
            if mean_pr > 0.7:
                print(f"  VERDICT: CMT is VIABLE")
            elif mean_pr > 0.4:
                print(f"  VERDICT: CMT is MARGINAL — needs stronger features")
            else:
                print(f"  VERDICT: CMT is NOT VIABLE")

    with open(save_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
