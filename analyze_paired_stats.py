"""
Paired statistical analysis: AdvOpt vs idea1.

Computes:
  - Mean ΔJF_codec for each method (± 95% CI via bootstrap)
  - Paired gain: adv_opt_delta - idea1_delta per video
  - Win-rate (fraction of videos where AdvOpt > idea1)
  - Paired Wilcoxon signed-rank test (non-parametric, paired)
  - Paired t-test (parametric)
  - SSIM-constrained subset (SSIM ≥ 0.90 for AdvOpt)
  - Pareto scatter: SSIM vs ΔJF for idea1 and adv_opt

Usage:
    python analyze_paired_stats.py \\
        --results_json results_v100/fancy_eval/fancy_v2/results.json \\
        --methods idea1,adv_opt \\
        --save_dir results_v100/paired_stats
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci(data, n_boot=5000, ci=0.95, stat=np.mean):
    data = np.array(data)
    boot_stats = [stat(np.random.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json", required=True)
    p.add_argument("--methods",   default="idea1,adv_opt")
    p.add_argument("--ssim_floor", type=float, default=0.90)
    p.add_argument("--save_dir",   default="results_v100/paired_stats")
    p.add_argument("--tag",        default="v1")
    p.add_argument("--no_plot",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    assert len(methods) == 2, "Specify exactly two methods for paired comparison"
    m1, m2 = methods

    data = json.load(open(args.results_json))
    rows = data.get("results", data)

    # Group by (video, method)
    by_vid_method = {}
    for r in rows:
        vid = r["video"]
        met = r["method"]
        if r.get("delta_jf_codec") != r.get("delta_jf_codec"):  # NaN check
            continue
        by_vid_method[(vid, met)] = r

    # Get paired videos (present in both methods)
    videos_m1 = {vid for (vid, met) in by_vid_method if met == m1}
    videos_m2 = {vid for (vid, met) in by_vid_method if met == m2}
    paired_vids = sorted(videos_m1 & videos_m2)
    print(f"[paired_stats] Paired videos: {len(paired_vids)}")

    d1 = [by_vid_method[(v, m1)]["delta_jf_codec"] * 100 for v in paired_vids]
    d2 = [by_vid_method[(v, m2)]["delta_jf_codec"] * 100 for v in paired_vids]
    s1 = [by_vid_method[(v, m1)].get("mean_ssim", 1.0) for v in paired_vids]
    s2 = [by_vid_method[(v, m2)].get("mean_ssim", 1.0) for v in paired_vids]

    gains = [x2 - x1 for x1, x2 in zip(d1, d2)]
    win_rate = sum(g > 0 for g in gains) / len(gains)
    tie_rate = sum(g == 0 for g in gains) / len(gains)

    # Bootstrap CIs
    ci1_lo, ci1_hi = bootstrap_ci(d1)
    ci2_lo, ci2_hi = bootstrap_ci(d2)

    # Significance tests
    from scipy import stats as scipy_stats
    wstat, wp = scipy_stats.wilcoxon(d2, d1, alternative="greater")
    tstat, tp = scipy_stats.ttest_rel(d2, d1, alternative="greater")

    # SSIM-constrained subset: adv_opt SSIM >= floor
    constrained_vids = [v for v, s in zip(paired_vids, s2) if s >= args.ssim_floor]
    if constrained_vids:
        cd1 = [by_vid_method[(v, m1)]["delta_jf_codec"] * 100 for v in constrained_vids]
        cd2 = [by_vid_method[(v, m2)]["delta_jf_codec"] * 100 for v in constrained_vids]
        cgains = [x2 - x1 for x1, x2 in zip(cd1, cd2)]
        cwin   = sum(g > 0 for g in cgains) / len(cgains)
    else:
        cd1, cd2, cgains, cwin = [], [], [], 0.0

    print(f"\n{'='*65}")
    print(f"PAIRED ANALYSIS: {m2} vs {m1}  (n={len(paired_vids)} videos)")
    print(f"  {m1}: mean={np.mean(d1):+.1f}pp  95%CI=[{ci1_lo:+.1f}, {ci1_hi:+.1f}]  SSIM={np.mean(s1):.3f}")
    print(f"  {m2}: mean={np.mean(d2):+.1f}pp  95%CI=[{ci2_lo:+.1f}, {ci2_hi:+.1f}]  SSIM={np.mean(s2):.3f}")
    print(f"  Ratio (mean):         {np.mean(d2)/max(np.mean(d1),0.01):.2f}×")
    print(f"  Paired gain (mean):   {np.mean(gains):+.1f}pp  (median {np.median(gains):+.1f}pp)")
    print(f"  Win-rate {m2}>{m1}:  {win_rate*100:.0f}%  ({sum(g>0 for g in gains)}/{len(gains)} videos)")
    print(f"  Tie-rate:             {tie_rate*100:.0f}%")
    print(f"  Wilcoxon (one-sided): W={wstat:.0f}  p={wp:.4f}")
    print(f"  t-test   (one-sided): t={tstat:.2f}  p={tp:.4f}")
    if constrained_vids:
        print(f"  SSIM≥{args.ssim_floor} subset (n={len(constrained_vids)}): {m2} mean={np.mean(cd2):+.1f}pp vs {m1} mean={np.mean(cd1):+.1f}pp")
        print(f"    Paired gain: {np.mean(cgains):+.1f}pp  win-rate: {cwin*100:.0f}%")
    print(f"{'='*65}\n")

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_data = {
        "method_baseline": m1,
        "method_proposed": m2,
        "n_paired": len(paired_vids),
        "mean_delta_baseline": float(np.mean(d1)),
        "mean_delta_proposed": float(np.mean(d2)),
        "ci95_baseline": [float(ci1_lo), float(ci1_hi)],
        "ci95_proposed": [float(ci2_lo), float(ci2_hi)],
        "mean_paired_gain": float(np.mean(gains)),
        "median_paired_gain": float(np.median(gains)),
        "win_rate": float(win_rate),
        "wilcoxon_p": float(wp),
        "ttest_p": float(tp),
        "ssim_floor": args.ssim_floor,
        "constrained_n": len(constrained_vids),
        "constrained_mean_proposed": float(np.mean(cd2)) if cd2 else None,
        "constrained_win_rate": float(cwin),
        "per_video": [
            {
                "video": v,
                m1 + "_pp": d1i, m2 + "_pp": d2i,
                m1 + "_ssim": s1i, m2 + "_ssim": s2i,
                "gain_pp": gi,
            }
            for v, d1i, d2i, s1i, s2i, gi in zip(paired_vids, d1, d2, s1, s2, gains)
        ],
    }
    out_json = out_dir / f"paired_{m1}_vs_{m2}_{args.tag}.json"
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[saved] {out_json}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Pareto scatter: SSIM vs ΔJF
            ax = axes[0]
            ax.scatter(d1, s1, alpha=0.7, label=m1, color="steelblue", s=40)
            ax.scatter(d2, s2, alpha=0.7, label=m2, color="coral",     s=40)
            ax.axhline(args.ssim_floor, color="gray", ls="--", lw=1, label=f"SSIM={args.ssim_floor}")
            ax.set_xlabel("ΔJF_codec (pp)")
            ax.set_ylabel("Mean SSIM")
            ax.set_title("Pareto: Privacy (ΔJF) vs Utility (SSIM)")
            ax.legend(fontsize=9)

            # Paired gain per video
            ax2 = axes[1]
            sorted_gains = sorted(enumerate(gains), key=lambda x: x[1], reverse=True)
            vidx = [x[0] for x in sorted_gains]
            gval = [x[1] for x in sorted_gains]
            colors = ["coral" if g >= 0 else "steelblue" for g in gval]
            ax2.barh(range(len(gval)), gval, color=colors)
            ax2.axvline(0, color="black", lw=0.8)
            ax2.set_yticks(range(len(gval)))
            ax2.set_yticklabels([paired_vids[i][:12] for i in vidx], fontsize=6)
            ax2.set_xlabel("Paired gain (pp)")
            ax2.set_title(f"{m2} - {m1}  per video  (mean={np.mean(gains):+.1f}pp)")

            plt.tight_layout()
            out_png = out_dir / f"paired_{m1}_vs_{m2}_{args.tag}.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[saved] {out_png}")
        except Exception as e:
            print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
