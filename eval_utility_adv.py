"""
Utility Preservation for AdvOpt: LPIPS + YOLO Detection Recall.

Non-tracking utility metrics to demonstrate that AdvOpt preserves video utility:
  - LPIPS  (perceptual distance to original, AlexNet backbone)
  - SSIM   (structural similarity, reference)
  - PSNR   (pixel-level fidelity)
  - YOLO detection recall (fraction of clean YOLO detections still detected after edit)

Answers: "Does AdvOpt make videos look different or break object detection?"
Expected answer: No — edits are near-imperceptible (LPIPS ≈ 0.01-0.03) and
detection recall ≈ 100%.

Usage:
    python eval_utility_adv.py \\
        --videos bear,blackswan,bmx-trees,breakdance,camel,car-roundabout,car-shadow,cows,dance-twirl,dog \\
        --device cuda --save_dir results_v100/utility_adv

Output:
    results_v100/utility_adv/utility_adv.json
    results_v100/utility_adv/utility_adv.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DAVIS_ROOT, FFMPEG_PATH
from src.dataset import load_single_video
from pilot_mask_guided import frame_quality
from src.fancy_suppression import optimize_adv_params
from pilot_mask_guided import apply_boundary_suppression


def to_lpips_tensor(frame_rgb, device):
    """Convert [H,W,3] uint8 RGB to [-1,1] tensor for LPIPS."""
    t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t.to(device)


def compute_lpips(net, orig_frames, edit_frames, device):
    scores = []
    with torch.no_grad():
        for fo, fe in zip(orig_frames, edit_frames):
            d = net(to_lpips_tensor(fo, device), to_lpips_tensor(fe, device))
            scores.append(float(d.mean()))
    return float(np.mean(scores))


def compute_yolo_recall(model, orig_frames, edit_frames, conf=0.25, iou_thresh=0.5):
    """
    Compute detection recall: fraction of clean-frame YOLO detections
    that are also detected in the edited frames (same class, IoU ≥ threshold).
    """
    total, recalled = 0, 0
    for fo, fe in zip(orig_frames, edit_frames):
        def detect(frame):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            res = model(bgr, conf=conf, verbose=False)
            boxes = []
            for r in res:
                for box in r.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                    cls  = int(box.cls.cpu().numpy().flatten()[0])
                    boxes.append((xyxy, cls))
            return boxes

        clean_boxes = detect(fo)
        adv_boxes   = detect(fe)

        for (cb, cc) in clean_boxes:
            total += 1
            matched = False
            for (ab, ac) in adv_boxes:
                if ac != cc:
                    continue
                xi1, yi1 = max(cb[0], ab[0]), max(cb[1], ab[1])
                xi2, yi2 = min(cb[2], ab[2]), min(cb[3], ab[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                a1 = (cb[2]-cb[0]) * (cb[3]-cb[1])
                a2 = (ab[2]-ab[0]) * (ab[3]-ab[1])
                union = a1 + a2 - inter
                if union > 0 and inter / union >= iou_thresh:
                    matched = True
                    break
            if matched:
                recalled += 1
    return recalled / total if total > 0 else 1.0, total, recalled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="bear,blackswan,bmx-trees,breakdance,camel,"
                   "car-roundabout,car-shadow,cows,dance-twirl,dog")
    p.add_argument("--ring_width",  type=int,   default=24)
    p.add_argument("--blend_alpha", type=float, default=0.80)
    p.add_argument("--adv_n_iter",  type=int,   default=80)
    p.add_argument("--ssim_floor",  type=float, default=0.92)
    p.add_argument("--max_frames",  type=int,   default=10)
    p.add_argument("--davis_root",  default=DAVIS_ROOT)
    p.add_argument("--save_dir",    default="results_v100/utility_adv")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--no_plot",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    device = args.device

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LPIPS
    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net="alex").to(device)
    lpips_net.eval()
    print("[utility] LPIPS (AlexNet) loaded")

    # Load YOLO
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    print("[utility] YOLOv8n loaded")

    results = []
    for vid in videos:
        print(f"\n=== {vid} ===")
        frames, masks, _ = load_single_video(
            args.davis_root, vid, max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue

        # Apply AdvOpt: first-frame adaptation, then apply to all frames
        import numpy as _np
        first_mask = next((m for m in masks if _np.asarray(m).sum() > 0), masks[0])
        first_frame = frames[0]
        opt_rw, opt_alpha = optimize_adv_params(
            first_frame, first_mask,
            n_iter=args.adv_n_iter,
            ssim_floor=args.ssim_floor,
            device=device,
        )
        print(f"  adv_opt: rw={opt_rw}, α={opt_alpha:.3f}")
        edited_frames = [
            apply_boundary_suppression(f, m, ring_width=opt_rw, blend_alpha=opt_alpha)
            for f, m in zip(frames, masks)
        ]

        # Also compute idea1 baseline for comparison
        idea1_frames = [
            apply_boundary_suppression(f, m, args.ring_width, args.blend_alpha)
            for f, m in zip(frames, masks)
        ]

        # Quality metrics
        ssim_vals, psnr_vals = [], []
        for fo, fe in zip(frames, edited_frames):
            s, p = frame_quality(fo, fe)
            ssim_vals.append(s)
            if p != float("inf"):
                psnr_vals.append(p)
        mean_ssim = float(np.mean(ssim_vals))
        mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else float("nan")

        # LPIPS
        mean_lpips = compute_lpips(lpips_net, frames, edited_frames, device)
        idea1_lpips = compute_lpips(lpips_net, frames, idea1_frames, device)

        # YOLO recall
        recall, n_total, n_recalled = compute_yolo_recall(yolo, frames, edited_frames)
        idea1_recall, _, _ = compute_yolo_recall(yolo, frames, idea1_frames)

        print(f"  AdvOpt: SSIM={mean_ssim:.4f} PSNR={mean_psnr:.1f}dB "
              f"LPIPS={mean_lpips:.4f} YOLO_recall={recall:.3f} ({n_recalled}/{n_total})")
        print(f"  idea1:  LPIPS={idea1_lpips:.4f} YOLO_recall={idea1_recall:.3f}")

        results.append({
            "video": vid,
            "adv_opt": {
                "mean_ssim": mean_ssim, "mean_psnr": mean_psnr,
                "mean_lpips": mean_lpips, "yolo_recall": recall,
                "yolo_n_total": n_total, "yolo_n_recalled": n_recalled,
                "opt_ring_width": opt_rw, "opt_alpha": opt_alpha,
            },
            "idea1": {
                "mean_lpips": idea1_lpips, "yolo_recall": idea1_recall,
            },
        })

    if not results:
        print("[error] No results")
        sys.exit(1)

    # Aggregate
    adv_ssims  = [r["adv_opt"]["mean_ssim"]  for r in results]
    adv_psnrs  = [r["adv_opt"]["mean_psnr"]  for r in results if r["adv_opt"]["mean_psnr"] == r["adv_opt"]["mean_psnr"]]
    adv_lpips  = [r["adv_opt"]["mean_lpips"] for r in results]
    adv_recall = [r["adv_opt"]["yolo_recall"] for r in results]
    i1_lpips   = [r["idea1"]["mean_lpips"]   for r in results]
    i1_recall  = [r["idea1"]["yolo_recall"]  for r in results]

    print(f"\n{'='*60}")
    print(f"UTILITY SUMMARY  AdvOpt  (n={len(results)} videos)")
    print(f"  SSIM:         {np.mean(adv_ssims):.4f}  (all ≥ {min(adv_ssims):.3f})")
    print(f"  PSNR:         {np.mean(adv_psnrs):.1f} dB")
    print(f"  LPIPS:        {np.mean(adv_lpips):.4f}  (idea1={np.mean(i1_lpips):.4f})")
    print(f"    LPIPS < 0.05 (near-imperceptible): {sum(l<0.05 for l in adv_lpips)}/{len(adv_lpips)}")
    print(f"  YOLO recall:  {np.mean(adv_recall)*100:.1f}%  (idea1={np.mean(i1_recall)*100:.1f}%)")
    print(f"    YOLO ≥95%:  {sum(r>=0.95 for r in adv_recall)}/{len(adv_recall)} videos")
    print(f"{'='*60}")

    out_data = {
        "n_videos": len(results),
        "mean_ssim":        float(np.mean(adv_ssims)),
        "mean_psnr":        float(np.mean(adv_psnrs)) if adv_psnrs else None,
        "mean_lpips":       float(np.mean(adv_lpips)),
        "mean_yolo_recall": float(np.mean(adv_recall)),
        "idea1_mean_lpips": float(np.mean(i1_lpips)),
        "idea1_mean_recall": float(np.mean(i1_recall)),
        "results": results,
    }
    out_json = out_dir / "utility_adv.json"
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[saved] {out_json}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))

            ax = axes[0]
            vids = [r["video"][:12] for r in results]
            ax.barh(vids, adv_lpips, color="coral", label="AdvOpt")
            ax.barh(vids, i1_lpips, color="steelblue", alpha=0.5, label="idea1")
            ax.axvline(0.05, color="gray", ls="--", lw=1, label="imperceptible threshold")
            ax.set_xlabel("LPIPS (↓ better)")
            ax.set_title(f"Perceptual Distance  AdvOpt={np.mean(adv_lpips):.4f}")
            ax.legend(fontsize=8)

            ax2 = axes[1]
            ax2.bar(vids, [r*100 for r in adv_recall], color="steelblue", label="AdvOpt")
            ax2.axhline(95, color="red", ls="--", lw=1, label="95% recall")
            ax2.set_ylim(0, 110)
            ax2.set_xticklabels(vids, rotation=45, ha="right", fontsize=7)
            ax2.set_ylabel("YOLO Detection Recall (%)")
            ax2.set_title(f"Object Detection Recall  mean={np.mean(adv_recall)*100:.1f}%")
            ax2.legend(fontsize=8)

            plt.tight_layout()
            out_png = out_dir / "utility_adv.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[saved] {out_png}")
        except Exception as e:
            print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
