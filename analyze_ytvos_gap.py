"""
analyze_ytvos_gap.py  —  Step 1 gap diagnosis (CPU only, no GPU)

Reads existing YT-VOS results.json and compares:
  - negative ΔJF_codec group vs positive group
  - n_frames, jf_clean, mask area (if available)

Also optionally scans annotation directories to compute median mask area.

Usage (on server):
  # Quick (results only):
  python analyze_ytvos_gap.py \
      --results_json results_v100/ytbvos/ytbvos_combo_strong_v1/results.json

  # With mask area scan (requires annotation dir):
  python analyze_ytvos_gap.py \
      --results_json results_v100/ytbvos/ytbvos_combo_strong_v1/results.json \
      --ytvos_root   data/youtube_vos \
      --anno_split   valid \
      --max_videos   200

Output:
  Console table + results_v100/ytbvos/ytbvos_combo_strong_v1/gap_analysis.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get("results", [])


def median_mask_area(
    video_id: str,
    ytvos_root: str,
    anno_split: str,
    palette_idx: int = 1,
    max_frames: int = 50,
) -> Optional[float]:
    """Return median mask area (fraction of frame) from annotation PNGs."""
    try:
        from PIL import Image
    except ImportError:
        return None

    anno_dir = Path(ytvos_root) / anno_split / "Annotations" / video_id
    if not anno_dir.exists():
        return None

    pngs = sorted(anno_dir.glob("*.png"))[:max_frames]
    areas = []
    for p in pngs:
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        H, W = arr.shape
        # auto-detect palette index if needed
        vals = np.unique(arr)
        nonbg = vals[vals > 0]
        if len(nonbg) == 0:
            continue
        # use the smallest non-zero value (first object palette index)
        pidx = int(nonbg[0])
        area = float((arr == pidx).sum()) / (H * W)
        areas.append(area)
    return float(np.median(areas)) if areas else None


def resolve_palette_idx(video_id: str, ytvos_root: str, anno_split: str) -> int:
    meta_path = Path(ytvos_root) / anno_split / "meta.json"
    if not meta_path.exists():
        return 1
    with open(meta_path) as f:
        meta = json.load(f)
    vid_meta = meta.get("videos", {}).get(video_id, {})
    obj_keys = sorted(vid_meta.get("objects", {}).keys(), key=lambda x: int(x))
    return int(obj_keys[0]) if obj_keys else 1


def stats(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {
        "n":      len(arr),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


def fmt(s: Dict[str, float], key: str = "mean") -> str:
    return f"{s[key]:.4f} (±{s['std']:.4f}, median={s['median']:.4f})"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), max(len(r[i]) for r in rows))
              for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json", required=True,
                   help="Path to results.json from pilot_ytbvos.py run")
    p.add_argument("--ytvos_root",   default="",
                   help="YT-VOS root (optional; enables mask area scan)")
    p.add_argument("--anno_split",   default="valid",
                   help="Annotation sub-folder (default: valid)")
    p.add_argument("--max_videos",   type=int, default=300,
                   help="Cap for mask-area scan (expensive per-video)")
    p.add_argument("--neg_thresh",   type=float, default=0.0,
                   help="ΔJF_codec threshold for 'negative' group")
    return p.parse_args()


def main():
    args  = parse_args()
    recs  = load_results(args.results_json)
    out_dir = Path(args.results_json).parent

    # filter to valid (non-NaN delta)
    valid = [r for r in recs
             if isinstance(r.get("delta_jf_codec"), (int, float))
             and r["delta_jf_codec"] == r["delta_jf_codec"]]

    print(f"\n[gap analysis]  total={len(recs)}  valid={len(valid)}")

    neg = [r for r in valid if r["delta_jf_codec"] <  args.neg_thresh]
    pos = [r for r in valid if r["delta_jf_codec"] >= args.neg_thresh]

    print(f"  negative (ΔJF<{args.neg_thresh}): n={len(neg)}")
    print(f"  positive (ΔJF≥{args.neg_thresh}): n={len(pos)}")

    # ── Section 1: n_frames ──────────────────────────────────────────────────
    neg_nf = [r["n_frames"] for r in neg if "n_frames" in r]
    pos_nf = [r["n_frames"] for r in pos if "n_frames" in r]

    # ── Section 2: jf_clean ──────────────────────────────────────────────────
    neg_jf = [r["jf_clean"] for r in neg if "jf_clean" in r]
    pos_jf = [r["jf_clean"] for r in pos if "jf_clean" in r]

    # ── Section 3: delta_jf_codec distribution ──────────────────────────────
    all_delta = [r["delta_jf_codec"] for r in valid]
    neg_delta = [r["delta_jf_codec"] for r in neg]
    pos_delta = [r["delta_jf_codec"] for r in pos]

    # ── Section 4: jf_codec_clean (baseline after codec) ────────────────────
    neg_codec = [r["jf_codec_clean"] for r in neg
                 if isinstance(r.get("jf_codec_clean"), float)
                 and r["jf_codec_clean"] == r["jf_codec_clean"]]
    pos_codec = [r["jf_codec_clean"] for r in pos
                 if isinstance(r.get("jf_codec_clean"), float)
                 and r["jf_codec_clean"] == r["jf_codec_clean"]]

    # ── Section 5: mask area (optional) ─────────────────────────────────────
    mask_areas: Dict[str, float] = {}
    if args.ytvos_root:
        print(f"\n[mask area scan]  scanning up to {args.max_videos} videos ...")
        vids_to_scan = [r["video"] for r in valid[:args.max_videos]]
        for i, vid in enumerate(vids_to_scan):
            pidx = resolve_palette_idx(vid, args.ytvos_root, args.anno_split)
            area = median_mask_area(vid, args.ytvos_root, args.anno_split,
                                    palette_idx=pidx)
            if area is not None:
                mask_areas[vid] = area
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(vids_to_scan)} scanned ...")
        print(f"  done: {len(mask_areas)} videos with mask area")

    # ── Print results ────────────────────────────────────────────────────────
    lines = []
    lines.append("# YT-VOS Gap Analysis\n")

    # Table 1: summary comparison
    rows = []
    if neg_nf and pos_nf:
        sn, sp = stats(neg_nf), stats(pos_nf)
        rows.append(["n_frames",     fmt(sn), fmt(sp),
                     f"{sp['mean'] - sn['mean']:+.2f}"])
    if neg_jf and pos_jf:
        sn, sp = stats(neg_jf), stats(pos_jf)
        rows.append(["jf_clean",     fmt(sn), fmt(sp),
                     f"{sp['mean'] - sn['mean']:+.4f}"])
    if neg_codec and pos_codec:
        sn, sp = stats(neg_codec), stats(pos_codec)
        rows.append(["jf_codec_clean", fmt(sn), fmt(sp),
                     f"{sp['mean'] - sn['mean']:+.4f}"])

    neg_ma = [mask_areas[r["video"]] for r in neg if r["video"] in mask_areas]
    pos_ma = [mask_areas[r["video"]] for r in pos if r["video"] in mask_areas]
    if neg_ma and pos_ma:
        sn, sp = stats(neg_ma), stats(pos_ma)
        rows.append(["mask_area (frac)", fmt(sn), fmt(sp),
                     f"{sp['mean'] - sn['mean']:+.4f}"])

    if rows:
        tbl = render_table(
            ["metric", f"neg (n={len(neg)})", f"pos (n={len(pos)})", "pos−neg"],
            rows,
        )
        lines.append("## Feature Comparison: Negative vs Positive Group\n")
        lines.append(tbl + "\n")
        print("\n" + tbl)

    # Table 2: delta distribution
    lines.append("\n## ΔJF_codec Distribution\n")
    buckets = [(-999, -0.10), (-0.10, -0.05), (-0.05, 0.0),
               (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 999)]
    bucket_rows = []
    for lo, hi in buckets:
        cnt = sum(1 for d in all_delta if lo <= d < hi)
        label = (f"[{lo:.2f},{hi:.2f})" if hi < 999
                 else f"[{lo:.2f},+∞)")
        label = label.replace("-999.00", "−∞")
        bucket_rows.append([label, str(cnt), f"{cnt/len(all_delta)*100:.1f}%"])
    dist_tbl = render_table(["ΔJF_codec bucket", "count", "pct"], bucket_rows)
    lines.append(dist_tbl + "\n")
    print("\n" + dist_tbl)

    # Table 3: n_frames distribution
    if neg_nf or pos_nf:
        lines.append("\n## Frame Count Distribution\n")
        all_nf = [r["n_frames"] for r in valid if "n_frames" in r]
        nf_buckets = [(1, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 50)]
        nf_rows = []
        for lo, hi in nf_buckets:
            all_c = sum(1 for n in all_nf  if lo <= n <= hi)
            neg_c = sum(1 for n in neg_nf  if lo <= n <= hi)
            pos_c = sum(1 for n in pos_nf  if lo <= n <= hi)
            neg_r = f"{neg_c/all_c*100:.0f}%" if all_c > 0 else "—"
            nf_rows.append([f"{lo}–{hi}", str(all_c), str(neg_c), neg_r])
        nf_tbl = render_table(
            ["n_frames", "total", "neg_count", "neg_rate"], nf_rows)
        lines.append(nf_tbl + "\n")
        print("\n" + nf_tbl)

    # Diagnosis
    lines.append("\n## Auto-Diagnosis\n")
    diag = []

    if neg_jf and pos_jf:
        gap_jf = np.mean(pos_jf) - np.mean(neg_jf)
        if gap_jf > 0.08:
            diag.append(f"- **JF_clean gap = {gap_jf:.3f}**: negative group has much lower baseline "
                        f"tracking quality → SAM2 already struggling on these videos; "
                        f"the edit cannot make it worse (floor effect).")
        elif gap_jf > 0.04:
            diag.append(f"- **JF_clean gap = {gap_jf:.3f}**: moderate; baseline tracking quality "
                        f"partially explains negatives.")
        else:
            diag.append(f"- **JF_clean gap = {gap_jf:.3f}**: small; baseline tracking quality "
                        f"does NOT explain the negative group.")

    if neg_nf and pos_nf:
        gap_nf = np.mean(pos_nf) - np.mean(neg_nf)
        if abs(gap_nf) < 1.0:
            diag.append(f"- **n_frames gap = {gap_nf:+.1f}**: frame count is NOT a driver.")
        else:
            diag.append(f"- **n_frames gap = {gap_nf:+.1f}**: negative group has "
                        f"{'fewer' if gap_nf > 0 else 'more'} frames.")

    if neg_ma and pos_ma:
        gap_ma = np.mean(pos_ma) - np.mean(neg_ma)
        if gap_ma > 0.03:
            diag.append(f"- **mask_area gap = {gap_ma:+.4f}**: positive group has LARGER objects "
                        f"→ supports scale-normalization hypothesis (fixed 24px ring "
                        f"covers proportionally more boundary on large objects).")
        elif gap_ma < -0.03:
            diag.append(f"- **mask_area gap = {gap_ma:+.4f}**: negative group has LARGER objects "
                        f"→ scale does not explain negatives in expected direction.")
        else:
            diag.append(f"- **mask_area gap = {gap_ma:+.4f}**: object scale similar between groups.")

    if not diag:
        diag.append("- Run with --ytvos_root to enable mask area analysis.")

    diag_text = "\n".join(diag)
    lines.append(diag_text + "\n")
    print("\nDiagnosis:")
    print(diag_text)

    # ── Save markdown ────────────────────────────────────────────────────────
    out_md = out_dir / "gap_analysis.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[gap analysis]  wrote -> {out_md}")


if __name__ == "__main__":
    main()
