"""
Proxy Validation: Boundary Gradient Suppression vs. Post-Codec ΔJF

Tests the key scientific claim underlying AdvOpt:
  "Minimising boundary-region gradient magnitude (pre-codec) correlates with
   maximising post-codec J&F degradation (ΔJF_codec)."

This script:
  1. Loads existing idea1 results (results from pilot_mask_guided.py runs)
  2. For each video, recomputes:
     - Pre-edit boundary gradient magnitude (G_orig)
     - Post-edit boundary gradient magnitude (G_edit)
     - Gradient suppression ratio: (G_orig - G_edit) / G_orig
  3. Correlates gradient suppression with ΔJF_codec from the saved results
  4. Reports Pearson r and Spearman ρ with p-values

Usage:
  # Using idea1 full DAVIS results:
  python validate_proxy_correlation.py \\
      --results_json results_v100/mask_guided/idea1_full/results.json \\
      --davis_root /path/to/DAVIS \\
      --ring_width 24 --blend_alpha 0.80 --edit_type idea1

  # Or specify a results directory directly:
  python validate_proxy_correlation.py \\
      --results_json results/B2_eval_eval_codec_ours/results.json

Output:
  results_v100/proxy_validation/<tag>/correlation.json
  results_v100/proxy_validation/<tag>/correlation.png  (scatter plot)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DAVIS_ROOT
from src.dataset import load_single_video
from pilot_mask_guided import (
    apply_boundary_suppression, apply_edit_to_video,
    EDIT_FNS,
)


def boundary_grad_magnitude(frame_rgb: np.ndarray, mask: np.ndarray,
                             ring_width: int = 24) -> float:
    """
    Compute mean gradient magnitude in the boundary ring of the frame.

    This is the proxy value that AdvOpt minimises.  Lower values mean
    the boundary is less detectable by gradient-sensitive feature extractors.
    """
    if mask.sum() == 0:
        return float("nan")

    kernel = np.ones((ring_width * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded  = cv2.erode(mask,  kernel)
    ring    = (dilated > 0) & (eroded == 0)
    if ring.sum() == 0:
        return float("nan")

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)

    return float(gmag[ring].mean())


def grad_suppression_ratio(orig_frame: np.ndarray, edit_frame: np.ndarray,
                            mask: np.ndarray, ring_width: int = 24) -> float:
    """
    Fraction of boundary gradient removed by the edit.
    suppression = (G_orig - G_edit) / G_orig  ∈ (-∞, 1]
    Higher = more suppression = better for AdvOpt proxy.
    """
    g_orig = boundary_grad_magnitude(orig_frame, mask, ring_width)
    g_edit = boundary_grad_magnitude(edit_frame, mask, ring_width)
    if np.isnan(g_orig) or g_orig < 1e-8:
        return float("nan")
    return (g_orig - g_edit) / g_orig


def pearson_r(x, y):
    x, y = np.array(x), np.array(y)
    if len(x) < 3:
        return float("nan"), 1.0
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom < 1e-12:
        return 0.0, 1.0
    r = float((xm * ym).sum() / denom)
    # Two-tailed t-test approx
    from math import sqrt
    n = len(x)
    t = r * sqrt(n - 2) / sqrt(max(1 - r ** 2, 1e-12))
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), df=n - 2))
    return r, p


def spearman_rho(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json", required=True,
                   help="Path to results.json from a pilot_mask_guided / pilot_fancy_eval run")
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--ring_width",   type=int,   default=24)
    p.add_argument("--blend_alpha",  type=float, default=0.80)
    p.add_argument("--edit_type",    default="idea1",
                   help="Edit type to recompute edits for gradient measurement")
    p.add_argument("--max_frames",   type=int, default=5,
                   help="Number of frames to average gradient over (5 is fast and stable)")
    p.add_argument("--save_dir",     default="results_v100/proxy_validation")
    p.add_argument("--tag",          default="v1")
    p.add_argument("--no_plot",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"[error] results_json not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", data)
    if not isinstance(results, list):
        print(f"[error] results.json must contain a 'results' list")
        sys.exit(1)

    # Filter to valid rows
    valid = [r for r in results
             if isinstance(r.get("delta_jf_codec"), (int, float))
             and r["delta_jf_codec"] == r["delta_jf_codec"]  # not NaN
             and r["delta_jf_codec"] > -0.5]
    print(f"[proxy_val] Loaded {len(valid)} valid rows from {results_path}")

    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {"ring_width": args.ring_width, "blend_alpha": args.blend_alpha}

    rows = []
    for r in valid:
        vid = r["video"]
        print(f"  {vid} ...", end=" ", flush=True)

        frames, masks, _ = load_single_video(
            args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"[skip: load failed]")
            continue

        # Apply the edit
        from pilot_mask_guided import apply_edit_to_video
        edited = apply_edit_to_video(frames, masks, args.edit_type, params)

        # Compute gradient suppression per frame, then average
        supp_vals = []
        g_orig_vals, g_edit_vals = [], []
        for frame, edit_frame, mask in zip(frames, edited, masks):
            g_o = boundary_grad_magnitude(frame,       mask, args.ring_width)
            g_e = boundary_grad_magnitude(edit_frame,  mask, args.ring_width)
            if not (np.isnan(g_o) or np.isnan(g_e) or g_o < 1e-8):
                supp_vals.append((g_o - g_e) / g_o)
                g_orig_vals.append(g_o)
                g_edit_vals.append(g_e)

        if not supp_vals:
            print(f"[skip: no valid frames]")
            continue

        mean_supp   = float(np.mean(supp_vals))
        mean_g_orig = float(np.mean(g_orig_vals))
        mean_g_edit = float(np.mean(g_edit_vals))
        delta_codec = r["delta_jf_codec"]

        print(f"supp={mean_supp:+.3f}  G_orig={mean_g_orig:.4f}  ΔJF={delta_codec*100:+.1f}pp")

        rows.append({
            "video": vid,
            "grad_suppression": mean_supp,
            "g_orig": mean_g_orig,
            "g_edit": mean_g_edit,
            "delta_jf_codec": delta_codec,
            "mean_ssim": r.get("mean_ssim", float("nan")),
        })

    if len(rows) < 5:
        print(f"[error] Not enough valid videos ({len(rows)}) for correlation")
        sys.exit(1)

    # Correlations
    supp_arr   = [r["grad_suppression"] for r in rows]
    delta_arr  = [r["delta_jf_codec"]   for r in rows]
    g_orig_arr = [r["g_orig"]           for r in rows]

    r_supp,   p_supp   = pearson_r(supp_arr,   delta_arr)
    r_gorig,  p_gorig  = pearson_r(g_orig_arr, delta_arr)
    rho_supp, p_rho    = spearman_rho(supp_arr, delta_arr)

    print(f"\n{'='*60}")
    print(f"PROXY VALIDATION RESULTS (n={len(rows)})")
    print(f"  Pearson  r(suppression, ΔJF_codec) = {r_supp:+.3f}  p={p_supp:.4f}")
    print(f"  Spearman ρ(suppression, ΔJF_codec) = {rho_supp:+.3f}  p={p_rho:.4f}")
    print(f"  Pearson  r(G_orig,      ΔJF_codec) = {r_gorig:+.3f}  p={p_gorig:.4f}")
    if abs(r_supp) >= 0.4:
        print(f"  *** Moderate-strong correlation — proxy loss is INFORMATIVE ***")
    elif abs(r_supp) >= 0.2:
        print(f"  **  Weak-moderate correlation — proxy has partial validity")
    else:
        print(f"  --  Weak/no correlation — proxy may not predict ΔJF_codec")
    print(f"{'='*60}")

    # Save JSON
    out_data = {
        "n_videos": len(rows),
        "pearson_r_suppression_vs_delta_jf": r_supp,
        "pearson_p": p_supp,
        "spearman_rho_suppression_vs_delta_jf": rho_supp,
        "spearman_p": p_rho,
        "pearson_r_gorig_vs_delta_jf": r_gorig,
        "per_video": rows,
    }
    out_json = out_dir / "correlation.json"
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[saved] {out_json}")

    # Scatter plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            ax = axes[0]
            ax.scatter(supp_arr, [d * 100 for d in delta_arr], alpha=0.6)
            ax.set_xlabel("Gradient Suppression Ratio (proxy loss)")
            ax.set_ylabel("ΔJF_codec (pp)")
            ax.set_title(f"Suppression vs ΔJF  r={r_supp:+.3f} (p={p_supp:.3f})")
            # Regression line
            m, b = np.polyfit(supp_arr, [d * 100 for d in delta_arr], 1)
            xs = np.linspace(min(supp_arr), max(supp_arr), 50)
            ax.plot(xs, m * xs + b, "r--", lw=1.5)
            ax.axhline(0, color="gray", lw=0.5)

            ax2 = axes[1]
            ax2.scatter(g_orig_arr, [d * 100 for d in delta_arr], alpha=0.6, color="orange")
            ax2.set_xlabel("Original Boundary Gradient (G_orig)")
            ax2.set_ylabel("ΔJF_codec (pp)")
            ax2.set_title(f"G_orig vs ΔJF  r={r_gorig:+.3f} (p={p_gorig:.3f})")

            plt.tight_layout()
            out_png = out_dir / "correlation.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[saved] {out_png}")
        except Exception as e:
            print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
