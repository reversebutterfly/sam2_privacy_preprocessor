"""
Analyze and compare mask-guided pilot experiment results.

Usage:
  python analyze_mask_guided.py --results_dir results_v100/mask_guided \
      --tags full_combo,full_idea1,full_idea2,full_combo_strong \
      --min_jf_clean 0.3 \
      --out results_v100/mask_guided/analysis.md
"""

import argparse
import json
import math
import os
import csv
from pathlib import Path


def load_results(results_dir: str, tag: str):
    path = Path(results_dir) / tag / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def _mean(xs):
    return sum(xs) / len(xs)

def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def _median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

def _ci95(xs):
    """95% CI half-width (t-based, approx normal for n>10)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    se = _std(xs) / math.sqrt(n)
    # Use 1.96 for normal approximation
    return 1.96 * se


def summarize(results, min_jf_clean: float = 0.3):
    videos = [r for r in results if r.get("jf_clean", 0) >= min_jf_clean
              and isinstance(r.get("delta_jf_codec"), float)
              and r["delta_jf_codec"] == r["delta_jf_codec"]]  # NaN guard
    if not videos:
        return None

    n = len(videos)
    deltas_codec = [r["delta_jf_codec"] for r in videos]
    deltas_adv   = [r["delta_jf_adv"]   for r in videos]
    ssims        = [r["mean_ssim"] for r in videos if r.get("mean_ssim")]
    psnrs        = [r["mean_psnr"] for r in videos
                    if r.get("mean_psnr") and r["mean_psnr"] != float("inf")]

    return {
        "n_videos": n,
        "mean_delta_codec":   _mean(deltas_codec),
        "median_delta_codec": _median(deltas_codec),
        "std_delta_codec":    _std(deltas_codec),
        "ci95_delta_codec":   _ci95(deltas_codec),
        "mean_delta_adv":     _mean(deltas_adv),
        "mean_ssim":          _mean(ssims) if ssims else float("nan"),
        "mean_psnr":          _mean(psnrs) if psnrs else float("nan"),
        "frac_5pp":  sum(1 for d in deltas_codec if d >= 0.05) / n,
        "frac_8pp":  sum(1 for d in deltas_codec if d >= 0.08) / n,
        "frac_12pp": sum(1 for d in deltas_codec if d >= 0.12) / n,
        "top5": [(r["video"], r["delta_jf_codec"]) for r in
                 sorted(videos, key=lambda r: r["delta_jf_codec"], reverse=True)[:5]],
        "bot5": [(r["video"], r["delta_jf_codec"]) for r in
                 sorted(videos, key=lambda r: r["delta_jf_codec"])[:5]],
        "videos": videos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_v100/mask_guided")
    parser.add_argument("--tags", default="full_combo,full_idea1,full_idea2,full_combo_strong")
    parser.add_argument("--min_jf_clean", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",")]

    lines = []
    lines.append("# Mask-Guided Pilot — Cross-Run Analysis\n")
    lines.append(f"Results dir: `{args.results_dir}`  |  min_jf_clean={args.min_jf_clean}\n")

    # Summary table
    lines.append("\n## Summary Table\n")
    lines.append("| Tag | N | ΔJF_adv | ΔJF_codec (mean±CI95) | median | SSIM | PSNR | ≥5pp | ≥8pp | ≥12pp |")
    lines.append("|-----|---|---------|----------------------|--------|------|------|------|------|-------|")

    summaries = {}
    for tag in tags:
        data = load_results(args.results_dir, tag)
        if data is None:
            lines.append(f"| {tag} | — | — | — | — | — | — | — | — | — |  *(not found)*")
            continue
        s = summarize(data["results"], args.min_jf_clean)
        if s is None:
            lines.append(f"| {tag} | 0 | — | — | — | — | — | — | — | — |")
            continue
        summaries[tag] = s
        ci = s['ci95_delta_codec']
        lines.append(
            f"| {tag} | {s['n_videos']} "
            f"| {s['mean_delta_adv']*100:+.1f}pp "
            f"| {s['mean_delta_codec']*100:+.1f}pp ± {ci*100:.1f}pp "
            f"| {s['median_delta_codec']*100:+.1f}pp "
            f"| {s['mean_ssim']:.3f} "
            f"| {s['mean_psnr']:.1f}dB "
            f"| {s['frac_5pp']:.0%} "
            f"| {s['frac_8pp']:.0%} "
            f"| {s['frac_12pp']:.0%} |"
        )

    # Per-video table for the best tag
    if summaries:
        best_tag = max(summaries, key=lambda t: summaries[t]["mean_delta_codec"])
        s = summaries[best_tag]
        lines.append(f"\n## Best Tag: `{best_tag}` — Per-Video Results (sorted by ΔJF_codec)\n")
        lines.append("| Video | JF_clean | JF_codec_clean | JF_adv | JF_codec_adv | ΔJF_adv | ΔJF_codec | SSIM |")
        lines.append("|-------|----------|---------------|--------|-------------|---------|-----------|------|")
        for r in sorted(s["videos"], key=lambda x: x["delta_jf_codec"], reverse=True):
            lines.append(
                f"| {r['video']} "
                f"| {r['jf_clean']:.3f} "
                f"| {r.get('jf_codec_clean', 0):.3f} "
                f"| {r['jf_adv']:.3f} "
                f"| {r['jf_codec_adv']:.3f} "
                f"| +{r['delta_jf_adv']:.3f} "
                f"| +{r['delta_jf_codec']:.3f} "
                f"| {r.get('mean_ssim', 0):.3f} |"
            )

        lines.append("\n### Top-5 Videos (highest ΔJF_codec)\n")
        for vid, d in s["top5"]:
            lines.append(f"- {vid}: +{d:.4f}")
        lines.append("\n### Bottom-5 Videos (lowest ΔJF_codec)\n")
        for vid, d in s["bot5"]:
            lines.append(f"- {vid}: +{d:.4f}")

    # Cross-run per-video comparison for all completed tags
    if len(summaries) >= 2:
        lines.append("\n## Cross-Run Per-Video Comparison\n")
        all_videos = set()
        for s in summaries.values():
            for r in s["videos"]:
                all_videos.add(r["video"])
        video_data = {v: {} for v in all_videos}
        for tag, s in summaries.items():
            for r in s["videos"]:
                video_data[r["video"]][tag] = r["delta_jf_codec"]

        header_cols = list(summaries.keys())
        lines.append("| Video | " + " | ".join(header_cols) + " |")
        lines.append("|-------|" + "|".join(["-------"] * len(header_cols)) + "|")
        for vid in sorted(all_videos):
            row = f"| {vid} |"
            for tag in header_cols:
                val = video_data[vid].get(tag)
                row += f" {'+' if val and val >= 0 else ''}{val:.3f} |" if val is not None else " — |"
            lines.append(row)

    report = "\n".join(lines)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
