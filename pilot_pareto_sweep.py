"""
pilot_pareto_sweep.py — Matched-utility Pareto sweep: ours vs boundary_blur baseline.

目标：在相同"发布视频质量"（SSIM on released video）下，比较
  - idea1   vs  boundary_blur  （严格公平对比，相同 ring 空间支持）
  - combo   vs  boundary_blur  （combo 空间支持更大，作为 enhanced variant 展示）

核心修正（相比 v1）：
  - SSIM 改为在 codec_adv 和 codec_clean 上算（发布资产的真实质量损失）
  - 全帧平均 SSIM（不再只用 frames[:5]）
  - alpha 网格对齐（所有方法扫同一 alpha 空间 [0.4,0.6,0.8,1.0]）
  - combo 的 halo 参数显式记录到 JSON
  - 输出真实 nondominated Pareto frontier

Usage:
    python pilot_pareto_sweep.py \\
        --videos "" --max_frames 50 --crf 23 \\
        --tag pareto_v1 --device cuda

    # sanity check (1 video × 3 configs)
    python pilot_pareto_sweep.py --sanity --tag pareto_sanity

Outputs:
    results_v100/mask_guided/<tag>/results.json    ← per-config per-video 数据
    results_v100/mask_guided/<tag>/pareto.json     ← Pareto frontier 分析
"""

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAM2_CHECKPOINT, SAM2_CONFIG, DAVIS_ROOT, FFMPEG_PATH, DAVIS_MINI_VAL
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264
from pilot_mask_guided import (
    apply_edit_to_video, run_tracking, build_predictor, frame_quality, codec_round_trip,
)


# ── Parameter grids（所有方法用相同 alpha 范围，保证对比公平）─────────────────

# alpha=1.0 对 boundary_blur 是"全强度模糊"，也给 idea1/combo 同样机会
ALPHA_GRID = [0.4, 0.6, 0.8, 1.0]
RW_GRID    = [8, 12, 16, 20, 24, 28, 32]

# combo 的 halo 固定为默认值（与论文主结果一致）
COMBO_HALO_DEFAULTS = {"halo_offset": 8, "halo_width": 12, "halo_strength": 0.4}


def make_configs(methods):
    """生成所有 (method, params) 组合。"""
    configs = []
    for method in methods:
        for rw, alpha in product(RW_GRID, ALPHA_GRID):
            params = {"ring_width": rw, "blend_alpha": alpha}
            if method == "combo":
                params.update(COMBO_HALO_DEFAULTS)
            configs.append({"method": method, "params": params})
    return configs


# ── Quality: codec-domain SSIM ─────────────────────────────────────────────────

def released_ssim(codec_clean_frames, codec_adv_frames):
    """
    SSIM 在"发布资产"上计算：codec_clean vs codec_adv，全帧平均。
    这才是受众（平台用户）感知到的视觉质量损失。
    """
    ssims = []
    for orig, edit in zip(codec_clean_frames, codec_adv_frames):
        s, _ = frame_quality(orig, edit)
        ssims.append(s)
    return float(np.mean(ssims))


# ── Pareto frontier ────────────────────────────────────────────────────────────

def nondominated(points):
    """
    计算 2D 非支配集（Pareto frontier）。
    points: list of (ssim_loss, delta_jf)，其中 ssim_loss = 1 - SSIM（越小越好），delta_jf（越大越好）。
    返回非支配点的 index 列表。
    """
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if: j 的 ssim_loss <= i 的 AND j 的 delta_jf >= i 的（至少一个严格）
            if (points[j][0] <= points[i][0] and points[j][1] >= points[i][1] and
                    (points[j][0] < points[i][0] or points[j][1] > points[i][1])):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def pareto_analysis(all_results, methods):
    """
    1. 在 SSIM bucket 内，找各方法的最优 ΔJF 点（用于表格）
    2. 在各方法内，计算真实 nondominated Pareto frontier
    3. 找"公平对比"：idea1 vs boundary_blur（相同 ring 支持）
    """
    # 按方法分组，过滤无效点
    by_method = {m: [] for m in methods}
    for r in all_results:
        m = r["method"]
        if m in by_method and r.get("mean_delta_jf_codec") is not None and r.get("mean_ssim_released") is not None:
            by_method[m].append(r)

    # --- 1. SSIM bucket 表格 ---
    buckets = [(0.70, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    bucket_table = []
    for blo, bhi in buckets:
        row = {"ssim_range": f"{blo:.2f}-{bhi:.2f}"}
        for method, entries in by_method.items():
            in_b = [e for e in entries if blo <= e["mean_ssim_released"] < bhi]
            if in_b:
                best = max(in_b, key=lambda x: x["mean_delta_jf_codec"])
                row[f"{method}_best_pp"] = round(best["mean_delta_jf_codec"] * 100, 2)
                row[f"{method}_best_ssim"] = round(best["mean_ssim_released"], 4)
                row[f"{method}_best_params"] = {k: v for k, v in best["params"].items()
                                                if k in ["ring_width", "blend_alpha"]}
            else:
                row[f"{method}_best_pp"] = None
        # 找 winner（在有数据的方法中）
        candidates = {m: row.get(f"{m}_best_pp") for m in methods if row.get(f"{m}_best_pp") is not None}
        if candidates:
            row["winner"] = max(candidates.items(), key=lambda x: x[1])[0]
        bucket_table.append(row)

    # --- 2. 各方法的 Pareto frontier（ssim_loss vs delta_jf）---
    frontiers = {}
    for method, entries in by_method.items():
        if not entries:
            continue
        pts = [(1.0 - e["mean_ssim_released"], e["mean_delta_jf_codec"]) for e in entries]
        front_idx = nondominated(pts)
        front_entries = sorted([entries[i] for i in front_idx],
                               key=lambda x: x["mean_ssim_released"], reverse=True)
        frontiers[method] = [
            {
                "ring_width": e["params"]["ring_width"],
                "blend_alpha": e["params"]["blend_alpha"],
                "mean_ssim_released": round(e["mean_ssim_released"], 4),
                "mean_delta_jf_codec_pp": round(e["mean_delta_jf_codec"] * 100, 2),
                "n_videos": len(e["per_video"]),
            }
            for e in front_entries
        ]

    # --- 3. idea1 vs boundary_blur 直接对比（相同参数下）---
    fair_compare = []
    for rw, alpha in product(RW_GRID, ALPHA_GRID):
        idea1_match = [e for e in by_method.get("idea1", [])
                       if e["params"]["ring_width"] == rw and e["params"]["blend_alpha"] == alpha]
        blur_match  = [e for e in by_method.get("boundary_blur", [])
                       if e["params"]["ring_width"] == rw and e["params"]["blend_alpha"] == alpha]
        if idea1_match and blur_match:
            i1 = idea1_match[0]
            bl = blur_match[0]
            fair_compare.append({
                "ring_width": rw, "blend_alpha": alpha,
                "idea1_pp": round(i1["mean_delta_jf_codec"] * 100, 2),
                "idea1_ssim": round(i1["mean_ssim_released"], 4),
                "blur_pp":   round(bl["mean_delta_jf_codec"] * 100, 2),
                "blur_ssim": round(bl["mean_ssim_released"], 4),
                "gap_pp":    round((i1["mean_delta_jf_codec"] - bl["mean_delta_jf_codec"]) * 100, 2),
                "winner":    "idea1" if i1["mean_delta_jf_codec"] > bl["mean_delta_jf_codec"] else "boundary_blur",
            })
    fair_compare.sort(key=lambda x: (x["ring_width"], x["blend_alpha"]))

    idea1_wins = sum(1 for r in fair_compare if r["winner"] == "idea1")
    blur_wins  = sum(1 for r in fair_compare if r["winner"] == "boundary_blur")

    return {
        "bucket_table": bucket_table,
        "frontiers": frontiers,
        "fair_compare_idea1_vs_blur": {
            "n_configs": len(fair_compare),
            "idea1_wins": idea1_wins,
            "blur_wins": blur_wins,
            "configs": fair_compare,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="",
                   help="逗号分隔的视频名，'all'=全 DAVIS，空=mini-val")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--min_jf_clean", type=float, default=0.3)
    p.add_argument("--methods", default="boundary_blur,idea1,combo",
                   help="要扫描的方法，逗号分隔")
    p.add_argument("--prompt", default="mask", choices=["point", "mask"])
    p.add_argument("--davis_root", default=DAVIS_ROOT)
    p.add_argument("--checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config", default=SAM2_CONFIG)
    p.add_argument("--ffmpeg_path", default=FFMPEG_PATH)
    p.add_argument("--save_dir", default="results_v100/mask_guided")
    p.add_argument("--tag", default="pareto_v1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sanity", action="store_true",
                   help="只跑1个视频×3个典型配置，验证流程")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    _raw = [v.strip() for v in args.videos.split(",") if v.strip()]
    if _raw == ["all"]:
        img_dir = Path(args.davis_root) / "JPEGImages" / "480p"
        videos = sorted(d.name for d in img_dir.iterdir() if d.is_dir())
    else:
        videos = _raw or DAVIS_MINI_VAL

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.save_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = make_configs(methods)

    if args.sanity:
        videos = videos[:1]
        configs = [
            {"method": "boundary_blur", "params": {"ring_width": 24, "blend_alpha": 0.8}},
            {"method": "idea1",         "params": {"ring_width": 24, "blend_alpha": 0.8}},
            {"method": "combo",         "params": {"ring_width": 24, "blend_alpha": 0.8, **COMBO_HALO_DEFAULTS}},
        ]
        print(f"[sanity] 1 video × {len(configs)} configs")

    print(f"[pareto] methods={methods}, configs={len(configs)}, videos={len(videos)}")
    print(f"[pareto] CRF={args.crf}, SSIM=on released video -> {out_dir}")

    predictor = build_predictor(args.checkpoint, args.sam2_config, device)

    # 构建 config 结果容器
    all_results = []
    config_map = {}   # (method, rw, alpha) -> index
    for cfg in configs:
        rw    = cfg["params"]["ring_width"]
        alpha = cfg["params"]["blend_alpha"]
        key   = (cfg["method"], rw, alpha)
        entry = {
            "method": cfg["method"],
            "params": cfg["params"],
            "per_video": [],
            "mean_delta_jf_codec": None,
            "mean_ssim_released": None,
        }
        config_map[key] = len(all_results)
        all_results.append(entry)

    for vid in videos:
        print(f"\n{'='*60}\n=== {vid} ===")
        try:
            frames, masks, _ = load_single_video(args.davis_root, vid, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [skip] load error: {e}")
            continue
        if not frames:
            print(f"  [skip] empty")
            continue

        # Clean baseline（原始帧 + SAM2）
        _, jf_clean, _, _ = run_tracking(frames, masks, predictor, device, args.prompt)
        print(f"  clean JF={jf_clean:.4f}")
        if jf_clean < args.min_jf_clean:
            print(f"  [skip] JF_clean < {args.min_jf_clean}")
            continue

        # Codec-clean baseline（原始帧 + H264 + SAM2）
        codec_clean = codec_round_trip(frames, args.ffmpeg_path, args.crf)
        if codec_clean is None:
            print(f"  [skip] codec error")
            continue
        _, jf_codec_clean, _, _ = run_tracking(codec_clean, masks, predictor, device, args.prompt)
        print(f"  codec_clean JF={jf_codec_clean:.4f}")

        # 扫描所有 configs
        for cfg in configs:
            method = cfg["method"]
            params = cfg["params"]
            rw     = params["ring_width"]
            alpha  = params["blend_alpha"]
            key    = (method, rw, alpha)

            try:
                edited = apply_edit_to_video(frames, masks, method, params)

                # H264 codec on edited frames
                codec_adv = codec_round_trip(edited, args.ffmpeg_path, args.crf)
                if codec_adv is None:
                    continue

                # SSIM on released video: codec_clean vs codec_adv（全帧）
                ssim_released = released_ssim(codec_clean, codec_adv)

                # Tracking on released video
                _, jf_codec_adv, _, _ = run_tracking(codec_adv, masks, predictor, device, args.prompt)
                delta_jf_codec = jf_codec_clean - jf_codec_adv

                vid_row = {
                    "video": vid,
                    "jf_clean": round(jf_clean, 4),
                    "jf_codec_clean": round(jf_codec_clean, 4),
                    "jf_codec_adv": round(jf_codec_adv, 4),
                    "delta_jf_codec": round(delta_jf_codec, 4),
                    "ssim_released": round(ssim_released, 4),
                }
                all_results[config_map[key]]["per_video"].append(vid_row)
                print(f"  [{method} rw={rw} a={alpha:.1f}] "
                      f"SSIM(released)={ssim_released:.4f} ΔJF={delta_jf_codec*100:+.2f}pp")

            except Exception as e:
                print(f"  [{method} rw={rw} a={alpha:.1f}] error: {e}")

        # 每个视频后更新汇总统计并保存
        for entry in all_results:
            deltas = [r["delta_jf_codec"] for r in entry["per_video"]]
            ssims  = [r["ssim_released"]  for r in entry["per_video"]]
            if deltas:
                entry["mean_delta_jf_codec"]  = round(float(np.mean(deltas)), 4)
                entry["mean_ssim_released"]    = round(float(np.mean(ssims)), 4)

        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)

        if args.sanity:
            print("\n[SANITY] Stopping after first video.")
            break

    # 最终 Pareto 分析
    pareto = pareto_analysis(all_results, methods)
    with open(out_dir / "pareto.json", "w") as f:
        json.dump(pareto, f, indent=2)

    # 打印 bucket 表格
    print(f"\n{'='*70}")
    print("[Pareto Bucket Table] 相同 SSIM(released) bucket 下各方法最优 ΔJF_codec")
    header = f"{'SSIM Range':<16}"
    for m in methods:
        header += f"{m + '_pp':<20}"
    header += "Winner"
    print(header)
    for row in pareto["bucket_table"]:
        line = f"{row['ssim_range']:<16}"
        for m in methods:
            val = row.get(f"{m}_best_pp")
            mark = " ★" if row.get("winner") == m else "  "
            line += f"{str(val) + mark:<20}" if val is not None else f"{'—':<20}"
        line += row.get("winner", "—")
        print(line)

    # 打印 idea1 vs blur 公平对比摘要
    fc = pareto["fair_compare_idea1_vs_blur"]
    if fc["n_configs"] > 0:
        print(f"\n[Fair Compare] idea1 vs boundary_blur（相同 ring 支持，相同参数）")
        print(f"  idea1 胜出: {fc['idea1_wins']}/{fc['n_configs']} 配置")
        print(f"  blur  胜出: {fc['blur_wins']}/{fc['n_configs']} 配置")
        # 展示几个典型行
        for r in fc["configs"][:6]:
            print(f"  rw={r['ring_width']:2d} a={r['blend_alpha']:.1f} | "
                  f"idea1={r['idea1_pp']:+.1f}pp SSIM={r['idea1_ssim']:.3f} | "
                  f"blur={r['blur_pp']:+.1f}pp SSIM={r['blur_ssim']:.3f} | "
                  f"gap={r['gap_pp']:+.1f}pp [{r['winner']}]")

    print(f"\n[saved] {out_dir}/results.json")
    print(f"[saved] {out_dir}/pareto.json")


if __name__ == "__main__":
    main()
