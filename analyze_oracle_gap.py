"""
Oracle-Gap Analysis for AdvOpt Proxy Validation.

Compares:
  1. Proxy-chosen alpha (AdvOpt's gradient-suppression proxy)
  2. Oracle alpha (the alpha that actually maximises ΔJF_codec for each video)

Using param_sweep data where multiple alpha values are tested on the same videos,
we can compute:
  - proxy_alpha_k = argmax_alpha{ proxy_loss(alpha) s.t. SSIM ≥ floor }
  - oracle_alpha_k = argmax_alpha{ ΔJF_codec(alpha) s.t. SSIM ≥ floor }
  - gap = oracle_ΔJF - proxy_ΔJF (in pp)

If proxy ≈ oracle, the AdvOpt proxy is validated for parameter selection.

Usage:
  python analyze_oracle_gap.py \\
      --results_json results_v100/param_sweep/param_sweep_v1/results.json \\
      --ssim_floor 0.90

Output:
  results_v100/oracle_gap/oracle_gap.json
  results_v100/oracle_gap/oracle_gap.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json", required=True)
    p.add_argument("--ssim_floor",  type=float, default=0.90)
    p.add_argument("--save_dir",    default="results_v100/oracle_gap")
    p.add_argument("--no_plot",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.results_json))
    rows = data.get("results", data)

    # Group by (video, ring_width) — within each group, vary alpha
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r["delta_jf_codec"] != r["delta_jf_codec"]:  # NaN
            continue
        groups[(r["video"], r["ring_width"])].append(r)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for (vid, rw), group_rows in sorted(groups.items()):
        # Filter by SSIM floor
        valid = [r for r in group_rows if r["mean_ssim"] >= args.ssim_floor]
        if len(valid) < 2:
            continue

        # Sort by alpha
        valid.sort(key=lambda x: x["blend_alpha"])
        alphas = [r["blend_alpha"]  for r in valid]
        deltas = [r["delta_jf_codec"] for r in valid]
        ssims  = [r["mean_ssim"]    for r in valid]

        # Oracle: alpha that maximizes ΔJF_codec among SSIM-valid rows
        oracle_idx   = int(np.argmax(deltas))
        oracle_alpha = alphas[oracle_idx]
        oracle_delta = deltas[oracle_idx]

        # Proxy: highest alpha that satisfies SSIM floor (proxy picks max-alpha under constraint)
        proxy_idx   = len(valid) - 1   # highest alpha = strongest suppression
        proxy_alpha = alphas[proxy_idx]
        proxy_delta = deltas[proxy_idx]

        gap_pp = (oracle_delta - proxy_delta) * 100

        row = {
            "video": vid, "ring_width": rw,
            "oracle_alpha": oracle_alpha, "oracle_delta_pp": oracle_delta * 100,
            "proxy_alpha":  proxy_alpha,  "proxy_delta_pp":  proxy_delta  * 100,
            "gap_pp": gap_pp,
            "n_valid": len(valid),
        }
        results.append(row)
        print(f"  {vid} rw={rw}: oracle_alpha={oracle_alpha:.2f}({oracle_delta*100:.1f}pp) "
              f"proxy_alpha={proxy_alpha:.2f}({proxy_delta*100:.1f}pp) gap={gap_pp:+.1f}pp")

    if not results:
        print("[error] No results to analyse.")
        sys.exit(1)

    gaps    = [r["gap_pp"] for r in results]
    proxies = [r["proxy_delta_pp"] for r in results]
    oracles = [r["oracle_delta_pp"] for r in results]

    print(f"\n{'='*55}")
    print(f"ORACLE-GAP ANALYSIS (n={len(results)} video/rw pairs, SSIM ≥ {args.ssim_floor})")
    print(f"  Mean oracle ΔJF  = {np.mean(oracles):+.1f}pp")
    print(f"  Mean proxy  ΔJF  = {np.mean(proxies):+.1f}pp")
    print(f"  Mean gap         = {np.mean(gaps):+.1f}pp  (oracle - proxy)")
    print(f"  % oracle==proxy  = {sum(g == 0 for g in gaps) / len(gaps) * 100:.0f}%")
    print(f"  % |gap| ≤ 3pp   = {sum(abs(g) <= 3 for g in gaps) / len(gaps) * 100:.0f}%")
    print(f"  % |gap| ≤ 5pp   = {sum(abs(g) <= 5 for g in gaps) / len(gaps) * 100:.0f}%")
    if abs(np.mean(gaps)) <= 5:
        print("  *** Proxy-chosen alpha is near-oracle (gap ≤ 5pp avg) ***")
    print(f"{'='*55}")

    out_json = out_dir / "oracle_gap.json"
    with open(out_json, "w") as f:
        json.dump({
            "ssim_floor": args.ssim_floor,
            "n_pairs": len(results),
            "mean_oracle_pp": float(np.mean(oracles)),
            "mean_proxy_pp":  float(np.mean(proxies)),
            "mean_gap_pp":    float(np.mean(gaps)),
            "results": results,
        }, f, indent=2)
    print(f"[saved] {out_json}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(oracles, proxies, alpha=0.6)
            lim = [min(oracles + proxies) - 2, max(oracles + proxies) + 2]
            ax.plot(lim, lim, "r--", lw=1.5, label="oracle = proxy")
            ax.set_xlabel("Oracle ΔJF_codec (pp)")
            ax.set_ylabel("Proxy (AdvOpt) ΔJF_codec (pp)")
            ax.set_title(f"Oracle vs Proxy  mean gap={np.mean(gaps):+.1f}pp  (n={len(results)})")
            ax.legend()
            plt.tight_layout()
            out_png = out_dir / "oracle_gap.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[saved] {out_png}")
        except Exception as e:
            print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
