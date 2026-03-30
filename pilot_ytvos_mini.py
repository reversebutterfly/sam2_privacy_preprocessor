"""
pilot_ytvos_mini.py  —  Step 3: Minimal YT-VOS gap diagnosis experiment

Compares on the SAME 20-30 videos:
  (A) fixed ring_width=24 / valid (sparse) frames     [original protocol]
  (B) scale_norm ring_width rho=0.10 / valid frames   [scale fix]
  (C) fixed ring_width=24 / valid_all_frames (dense)  [frame density fix]
  (D) scale_norm + dense frames                        [both fixes]

Conditions C and D require valid_all_frames split to be available.
If not available, they are skipped gracefully.

Usage (on server, ~1-2h for 25 videos × 4 conditions):
  python pilot_ytvos_mini.py \\
      --ytvos_root  data/youtube_vos \\
      --n_videos    25 \\
      --seed        42 \\
      --tag         mini_gap_diag \\
      --save_dir    results_v100/mini_gap \\
      --checkpoint  /path/to/sam2.1_hiera_tiny.pt \\
      --sam2_config sam2.1_hiera_tiny.yaml

  # Skip dense-frame conditions if valid_all_frames not available:
  python pilot_ytvos_mini.py ... --skip_dense

Output:
  results_v100/mini_gap/<tag>/results.json
  results_v100/mini_gap/<tag>/summary_table.md   (printed + saved)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video_ytvos, list_ytvos_videos
from pilot_mask_guided import (
    apply_edit_to_video,
    run_tracking,
    frame_quality,
    build_predictor,
    codec_round_trip,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def pick_videos(
    ytvos_root: str,
    split: str,
    anno_split: str,
    n: int,
    seed: int,
    min_jf_clean_approx: float = 0.0,
) -> List[str]:
    """Pick n random videos that have annotations."""
    all_vids = list_ytvos_videos(
        ytvos_root, split=split, anno_split=anno_split, min_annotated_frames=3)
    rng = random.Random(seed)
    rng.shuffle(all_vids)
    return all_vids[:n]


def run_one_video(
    vid: str,
    ytvos_root: str,
    split: str,
    anno_split: str,
    max_frames: int,
    params: dict,
    ring_width_mode: str,
    ring_width_rho: float,
    predictor,
    device: torch.device,
    ffmpeg_path: str,
    crf: int,
    prompt: str,
    min_jf_clean: float,
) -> Optional[Dict[str, Any]]:
    """Run one video for one condition. Returns result dict or None if skipped."""
    frames, masks, _ = load_single_video_ytvos(
        ytvos_root, vid,
        split=split, anno_split=anno_split,
        max_frames=max_frames,
    )
    if not frames:
        return None
    if not any(m.sum() > 0 for m in masks):
        return None

    _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, prompt)
    if jf_clean < min_jf_clean:
        return None

    codec_clean = codec_round_trip(frames, ffmpeg_path, crf)
    if codec_clean:
        _, jf_codec_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, prompt)
    else:
        jf_codec_clean = float("nan")

    edited = apply_edit_to_video(
        frames, masks, "combo", params,
        ring_width_mode=ring_width_mode,
        ring_width_rho=ring_width_rho,
    )

    ssim_vals = []
    for fo, fe in zip(frames[:5], edited[:5]):
        from pilot_mask_guided import frame_quality
        s, _ = frame_quality(fo, fe)
        ssim_vals.append(s)
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else float("nan")

    _, jf_adv, _, _ = run_tracking(edited, masks, predictor, device, prompt)
    delta_adv = jf_clean - jf_adv

    codec_edited = codec_round_trip(edited, ffmpeg_path, crf)
    if codec_edited:
        _, jf_codec_adv, _, _ = run_tracking(codec_edited, masks, predictor, device, prompt)
        delta_codec = jf_codec_clean - jf_codec_adv
    else:
        jf_codec_adv = float("nan")
        delta_codec  = float("nan")

    return {
        "video":            vid,
        "n_frames":         len(frames),
        "jf_clean":         jf_clean,
        "jf_codec_clean":   jf_codec_clean,
        "jf_adv":           jf_adv,
        "jf_codec_adv":     jf_codec_adv,
        "delta_jf_adv":     delta_adv,
        "delta_jf_codec":   delta_codec,
        "mean_ssim":        mean_ssim,
        "split":            split,
        "ring_width_mode":  ring_width_mode,
        "ring_width_rho":   ring_width_rho,
    }


def summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results
             if isinstance(r.get("delta_jf_codec"), float)
             and r["delta_jf_codec"] == r["delta_jf_codec"]]
    if not valid:
        return {"n": 0, "mean": float("nan"), "neg_rate": float("nan"),
                "ci95": float("nan"), "mean_ssim": float("nan")}
    deltas = [r["delta_jf_codec"] for r in valid]
    n = len(deltas)
    mean_d = float(np.mean(deltas))
    std_d  = float(np.std(deltas))
    ci95   = 1.96 * std_d / n ** 0.5
    neg_rate = sum(1 for d in deltas if d < 0) / n
    ssims  = [r["mean_ssim"] for r in valid
              if isinstance(r.get("mean_ssim"), float)
              and r["mean_ssim"] == r["mean_ssim"]]
    return {
        "n":         n,
        "mean":      mean_d,
        "ci95":      ci95,
        "neg_rate":  neg_rate,
        "mean_ssim": float(np.mean(ssims)) if ssims else float("nan"),
    }


def render_summary_table(condition_results: Dict[str, List[Dict]]) -> str:
    headers = ["Condition", "n", "ΔJF_codec (mean±ci95)", "neg_rate", "SSIM"]
    rows = []
    for cond_name, results in condition_results.items():
        s = summary_stats(results)
        rows.append([
            cond_name,
            str(s["n"]),
            f"{s['mean']*100:+.2f}pp ±{s['ci95']*100:.2f}pp",
            f"{s['neg_rate']*100:.1f}%",
            f"{s['mean_ssim']:.4f}",
        ])
    widths = [max(len(h), max(len(r[i]) for r in rows))
              for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ytvos_root",     default="data/youtube_vos")
    p.add_argument("--anno_split",     default="valid")
    p.add_argument("--n_videos",       type=int,   default=25)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max_frames",     type=int,   default=50)
    p.add_argument("--crf",            type=int,   default=23)
    p.add_argument("--prompt",         default="mask", choices=["mask", "point"])
    p.add_argument("--min_jf_clean",   type=float, default=0.3)
    # edit params (combo_strong)
    p.add_argument("--ring_width",     type=int,   default=24)
    p.add_argument("--blend_alpha",    type=float, default=0.8)
    p.add_argument("--halo_offset",    type=int,   default=8)
    p.add_argument("--halo_width",     type=int,   default=12)
    p.add_argument("--halo_strength",  type=float, default=0.4)
    p.add_argument("--ring_width_rho", type=float, default=0.10,
                   help="rho for scale_norm conditions")
    # conditions to run
    p.add_argument("--skip_dense",     action="store_true",
                   help="Skip conditions C/D (valid_all_frames) if not available")
    # infra
    p.add_argument("--checkpoint",     default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",    default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path",    default=FFMPEG_PATH)
    p.add_argument("--save_dir",       default="results_v100/mini_gap")
    p.add_argument("--tag",            default="mini_gap_diag")
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "ring_width":    args.ring_width,
        "blend_alpha":   args.blend_alpha,
        "halo_offset":   args.halo_offset,
        "halo_width":    args.halo_width,
        "halo_strength": args.halo_strength,
    }

    # Check dense split availability
    dense_split = "valid_all_frames"
    dense_available = (
        not args.skip_dense and
        (Path(args.ytvos_root) / dense_split / "JPEGImages").exists()
    )

    # Define conditions
    conditions = [
        # (label, split, ring_width_mode, rho)
        ("A_fixed_sparse",      "valid",             "fixed",      args.ring_width_rho),
        ("B_scalenorm_sparse",  "valid",             "scale_norm", args.ring_width_rho),
    ]
    if dense_available:
        conditions += [
            ("C_fixed_dense",       dense_split, "fixed",      args.ring_width_rho),
            ("D_scalenorm_dense",   dense_split, "scale_norm", args.ring_width_rho),
        ]
    else:
        print(f"[mini] valid_all_frames not found — skipping dense conditions C/D")

    # Pick video list from sparse split (so all conditions use the same videos)
    print(f"[mini] picking {args.n_videos} videos (seed={args.seed}) ...")
    videos = pick_videos(
        args.ytvos_root,
        split="valid",
        anno_split=args.anno_split,
        n=args.n_videos,
        seed=args.seed,
    )
    print(f"[mini] selected: {videos}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    # Resume support
    res_json = out_dir / "results.json"
    all_cond_results: Dict[str, List[Dict]] = {c[0]: [] for c in conditions}
    done_keys = set()

    if res_json.exists():
        with open(res_json) as f:
            saved = json.load(f)
        for cond_label, cond_results in saved.get("conditions", {}).items():
            if cond_label in all_cond_results:
                all_cond_results[cond_label] = cond_results
                for r in cond_results:
                    done_keys.add((cond_label, r["video"]))
        print(f"[mini] resuming — {len(done_keys)} (condition, video) pairs done")

    # Run
    for cond_label, split, rw_mode, rho in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_label}  split={split}  ring_mode={rw_mode}  rho={rho}")
        print(f"{'='*60}")

        for vid in videos:
            key = (cond_label, vid)
            if key in done_keys:
                print(f"  [skip] {vid} already done")
                continue

            print(f"  {vid} ...", end=" ", flush=True)
            result = run_one_video(
                vid=vid,
                ytvos_root=args.ytvos_root,
                split=split,
                anno_split=args.anno_split,
                max_frames=args.max_frames,
                params=params,
                ring_width_mode=rw_mode,
                ring_width_rho=rho,
                predictor=predictor,
                device=device,
                ffmpeg_path=args.ffmpeg_path,
                crf=args.crf,
                prompt=args.prompt,
                min_jf_clean=args.min_jf_clean,
            )
            if result is None:
                print("SKIP (load fail / low JF_clean)")
                continue

            d = result["delta_jf_codec"]
            d_str = f"{d*100:+.1f}pp" if d == d else "nan"
            print(f"ΔJF_codec={d_str}  SSIM={result['mean_ssim']:.3f}")

            all_cond_results[cond_label].append(result)
            done_keys.add(key)

            # Incremental save
            with open(res_json, "w") as f:
                json.dump({
                    "args":       vars(args),
                    "videos":     videos,
                    "conditions": all_cond_results,
                }, f, indent=2)

    # Final table
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    table = render_summary_table(all_cond_results)
    print(table)

    # Interpretation
    print("\nInterpretation:")
    s_A = summary_stats(all_cond_results.get("A_fixed_sparse", []))
    s_B = summary_stats(all_cond_results.get("B_scalenorm_sparse", []))

    if s_A["n"] > 0 and s_B["n"] > 0:
        delta_mean = (s_B["mean"] - s_A["mean"]) * 100
        delta_neg  = (s_B["neg_rate"] - s_A["neg_rate"]) * 100
        if abs(delta_mean) < 1.0 and abs(delta_neg) < 5.0:
            print(f"  Scale-norm vs fixed: Δmean={delta_mean:+.1f}pp, Δneg={delta_neg:+.1f}%")
            print("  → Scale normalisation does NOT materially help.")
            print("    Content distribution (not ring size) is the primary driver.")
        elif delta_mean > 2.0 or delta_neg < -5.0:
            print(f"  Scale-norm vs fixed: Δmean={delta_mean:+.1f}pp, Δneg={delta_neg:+.1f}%")
            print("  → Scale normalisation HELPS. Ring size mis-match was a real factor.")
        else:
            print(f"  Scale-norm vs fixed: Δmean={delta_mean:+.1f}pp, Δneg={delta_neg:+.1f}%")
            print("  → Marginal improvement. Ring size is a minor factor.")

    if dense_available:
        s_C = summary_stats(all_cond_results.get("C_fixed_dense", []))
        if s_A["n"] > 0 and s_C["n"] > 0:
            delta_mean = (s_C["mean"] - s_A["mean"]) * 100
            delta_neg  = (s_C["neg_rate"] - s_A["neg_rate"]) * 100
            if delta_mean > 3.0 or delta_neg < -8.0:
                print(f"\n  Dense vs sparse frames: Δmean={delta_mean:+.1f}pp, Δneg={delta_neg:+.1f}%")
                print("  → Frame density MATTERS. Sparse eval under-estimates attack effectiveness.")
            elif abs(delta_mean) < 1.5 and abs(delta_neg) < 5.0:
                print(f"\n  Dense vs sparse frames: Δmean={delta_mean:+.1f}pp, Δneg={delta_neg:+.1f}%")
                print("  → Frame density does NOT explain the gap.")
            else:
                print(f"\n  Dense vs sparse frames: Δmean={delta_mean:+.1f}pp, Δneg={delta_neg:+.1f}%")
                print("  → Moderate effect. Frame density is a partial factor.")

    # Save summary table
    md_path = out_dir / "summary_table.md"
    with open(md_path, "w") as f:
        f.write(f"# YT-VOS Mini Gap Diagnosis\n\n")
        f.write(f"Videos: n={args.n_videos}, seed={args.seed}\n\n")
        f.write(table + "\n")
    print(f"\n[mini] saved -> {md_path}")


if __name__ == "__main__":
    main()
