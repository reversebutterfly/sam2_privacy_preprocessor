"""
Utility Preservation Evaluation

Measures how much downstream utility is preserved after mask-guided boundary suppression.
Runs YOLOv8n object detection on original vs preprocessed frames and computes:
  - Detection AP50 (or recall@0.5 IoU threshold against YOLO's own clean detections)
  - Relative detection recall change (ΔRecall)

Since DAVIS has no COCO-format detection annotations, we use a self-consistency
approach: run YOLO on clean frames to get pseudo-GT boxes, then run YOLO on
preprocessed frames and measure how many pseudo-GT boxes are still detected.

This answers: "Does preprocessing make objects disappear from YOLO's perspective?"

Usage:
  python eval_utility.py \\
      --results_tag full_combo_strong \\
      --results_dir results_v100/mask_guided \\
      --max_frames 20 \\
      --conf_thresh 0.3 \\
      --iou_thresh 0.5 \\
      --out results_v100/utility/full_combo_strong_utility.json

Outputs:
  Per-video: n_clean_dets, n_adv_dets, recall, delta_recall
  Aggregate: mean_recall_clean, mean_recall_adv, mean_delta_recall
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DAVIS_ROOT, SAM2_CHECKPOINT, SAM2_CONFIG, FFMPEG_PATH
from src.dataset import load_single_video
from src.codec_eot import encode_decode_h264
from pilot_mask_guided import apply_edit_to_video


def iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def detect_frames(model, frames: List[np.ndarray], conf: float) -> List[List]:
    """Run YOLO on a list of RGB frames; returns per-frame list of [x1,y1,x2,y2,score,cls]."""
    results = []
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        res = model(bgr, conf=conf, verbose=False)
        boxes = []
        for r in res:
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                score = float(box.conf.cpu().numpy().flatten()[0])
                cls = int(box.cls.cpu().numpy().flatten()[0])
                boxes.append(xyxy + [score, cls])
        results.append(boxes)
    return results


def compute_recall(clean_dets: List[List], adv_dets: List[List],
                   iou_thresh: float = 0.5) -> Tuple[float, int, int]:
    """
    For each clean detection, check if it's 'recalled' in the adv detections.
    Returns (recall, n_clean_dets_total, n_recalled_total).
    """
    total = 0
    recalled = 0
    for clean_frame, adv_frame in zip(clean_dets, adv_dets):
        for c_box in clean_frame:
            total += 1
            # Check if any adv detection matches
            matched = any(
                iou(c_box[:4], a_box[:4]) >= iou_thresh
                for a_box in adv_frame
            )
            if matched:
                recalled += 1
    recall = recalled / total if total > 0 else 1.0
    return recall, total, recalled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_tag",  default="full_combo_strong",
                   help="Tag to evaluate (loads results from results_dir/tag/results.json)")
    p.add_argument("--results_dir",  default="results_v100/mask_guided")
    p.add_argument("--eval_mode",    default="codec",  choices=["adv", "codec"],
                   help="Evaluate pre-codec (adv) or post-codec (codec) preprocessed frames")
    p.add_argument("--max_videos",   type=int, default=30,
                   help="Max number of videos to evaluate (top by JF_clean)")
    p.add_argument("--max_frames",   type=int, default=20,
                   help="Max frames per video for YOLO eval")
    p.add_argument("--conf_thresh",  type=float, default=0.25)
    p.add_argument("--iou_thresh",   type=float, default=0.5)
    p.add_argument("--davis_root",   default=DAVIS_ROOT)
    p.add_argument("--ffmpeg_path",  default=FFMPEG_PATH)
    p.add_argument("--crf",          type=int, default=23)
    p.add_argument("--yolo_model",   default="yolov8n.pt",
                   help="YOLO model variant: yolov8n.pt / yolov8s.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    # Load sweep results to get edit params and video list
    res_json = Path(args.results_dir) / args.results_tag / "results.json"
    with open(res_json) as f:
        sweep_data = json.load(f)

    sweep_args = sweep_data["args"]
    results = sweep_data["results"]
    # Sort by JF_clean descending; take top max_videos
    results_valid = [r for r in results if r.get("jf_clean", 0) >= 0.3]
    results_valid.sort(key=lambda r: r.get("jf_clean", 0), reverse=True)
    results_valid = results_valid[:args.max_videos]

    edit_type  = sweep_args.get("edit_type", "combo")
    ring_width = sweep_args.get("ring_width", 24)
    blend_alpha = sweep_args.get("blend_alpha", 0.8)
    halo_offset  = sweep_args.get("halo_offset", 12)
    halo_width   = sweep_args.get("halo_width",  16)
    halo_strength = sweep_args.get("halo_strength", 0.6)
    edit_params = {
        "ring_width": ring_width, "blend_alpha": blend_alpha,
        "halo_offset": halo_offset, "halo_width": halo_width,
        "halo_strength": halo_strength,
    }

    print(f"[eval_utility] tag={args.results_tag}, mode={args.eval_mode}")
    print(f"[eval_utility] eval {len(results_valid)} videos, max_frames={args.max_frames}")
    print(f"[eval_utility] YOLO model: {args.yolo_model}, conf={args.conf_thresh}, IoU={args.iou_thresh}")

    from ultralytics import YOLO
    model = YOLO(args.yolo_model)
    model.to(args.device)

    all_results = []

    for row in results_valid:
        vid = row["video"]
        print(f"\n  === {vid} (JF_clean={row['jf_clean']:.3f}) ===")

        frames, masks, _ = load_single_video(args.davis_root, vid,
                                             max_frames=args.max_frames)
        if not frames:
            print(f"  [skip] load failed")
            continue

        # Apply edit
        edited = apply_edit_to_video(frames, masks, edit_type, edit_params)

        if args.eval_mode == "codec":
            clean_eval_frames = None
            adv_eval_frames = None
            try:
                from src.codec_eot import encode_decode_h264
                clean_eval_frames = encode_decode_h264(frames, ffmpeg_path=args.ffmpeg_path,
                                                        crf=args.crf)
                adv_eval_frames   = encode_decode_h264(edited, ffmpeg_path=args.ffmpeg_path,
                                                        crf=args.crf)
            except Exception as e:
                print(f"  [codec error] {e}")
            if clean_eval_frames is None or adv_eval_frames is None:
                clean_eval_frames = frames
                adv_eval_frames = edited
        else:
            clean_eval_frames = frames
            adv_eval_frames   = edited

        # YOLO detection
        clean_dets = detect_frames(model, clean_eval_frames, args.conf_thresh)
        adv_dets   = detect_frames(model, adv_eval_frames,   args.conf_thresh)

        n_clean = sum(len(d) for d in clean_dets)
        n_adv   = sum(len(d) for d in adv_dets)
        recall, total, recalled = compute_recall(clean_dets, adv_dets, args.iou_thresh)
        print(f"  clean dets: {n_clean}, adv dets: {n_adv}, recall: {recall:.3f} ({recalled}/{total})")

        all_results.append({
            "video": vid,
            "jf_clean": row["jf_clean"],
            "delta_jf_codec": row.get("delta_jf_codec"),
            "n_clean_dets": n_clean,
            "n_adv_dets":   n_adv,
            "recall":       recall,
            "n_total_clean_dets": total,
            "n_recalled":         recalled,
        })

    # Aggregate
    if all_results:
        recalls = [r["recall"] for r in all_results if r["n_total_clean_dets"] > 0]
        print(f"\n{'='*50}")
        print(f"UTILITY SUMMARY ({len(all_results)} videos, mode={args.eval_mode})")
        if recalls:
            mean_recall = sum(recalls) / len(recalls)
            print(f"  mean detection recall (adv vs clean): {mean_recall:.4f} ({mean_recall*100:.1f}%)")
            above95 = sum(1 for r in recalls if r >= 0.95) / len(recalls)
            print(f"  fraction ≥95% recall: {above95:.0%}")
        print(f"{'='*50}")

    out_data = {
        "tag": args.results_tag,
        "eval_mode": args.eval_mode,
        "conf_thresh": args.conf_thresh,
        "iou_thresh": args.iou_thresh,
        "results": all_results,
        "summary": {
            "n_videos": len(all_results),
            "mean_recall": sum(r["recall"] for r in all_results
                               if r["n_total_clean_dets"] > 0) / max(1,
                sum(1 for r in all_results if r["n_total_clean_dets"] > 0)),
        }
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved to {args.out}")
    else:
        print(json.dumps(out_data["summary"], indent=2))


if __name__ == "__main__":
    main()
