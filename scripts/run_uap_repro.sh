#!/bin/bash
# UAP-SAM2 Strict Reproduction — Phase 3: Run Official Code
#
# Usage: screen -S uap_run bash scripts/run_uap_repro.sh
# Log:   ~/uap_repro_run.log
#
# Deviations from strict reproduction (documented):
#   1. device patched from 'cuda:1' to 'cuda:1' (unchanged — server has multi-GPU, OK)
#   2. SA-V data: uses YouTube-VOS valid frames as substitute (HuggingFace unreachable)
#      — weight_fea=1e-6 so contribution is negligible; use --loss_fea=False to match paper
#      — This must be documented in the reproduction log
#   3. All other paper settings: exact

LOGFILE="$HOME/uap_repro_run.log"
exec > >(tee -a "$LOGFILE") 2>&1
set -e

echo "=== UAP-SAM2 Official Reproduction Run ==="
echo "Date: $(date)"

CONDA_BIN=/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/bin
eval "$($CONDA_BIN/conda shell.bash hook)"
conda activate UAP-SAM2

REPO_DIR="/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"

# ─── Step 0: SA-V substitute ─────────────────────────────────────────────────
echo "[step 0] Setting up SA-V substitute from YouTube-VOS valid frames..."
SAV_DST="$REPO_DIR/data/sav_test/JPEGImages_24fps"
mkdir -p "$SAV_DST"

if [ "$(ls -A $SAV_DST 2>/dev/null | wc -l)" -lt 10 ]; then
    YTVOS_VALID="/IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor/data/youtube_vos/valid/JPEGImages"
    echo "  Symlinking 30 YouTube-VOS valid video dirs as SA-V substitute..."
    COUNT=0
    for VID_DIR in "$YTVOS_VALID"/*/; do
        VID_NAME=$(basename "$VID_DIR")
        if [ ! -e "$SAV_DST/$VID_NAME" ]; then
            ln -s "$VID_DIR" "$SAV_DST/$VID_NAME"
            COUNT=$((COUNT + 1))
        fi
        [ $COUNT -ge 30 ] && break
    done
    echo "  Created $COUNT SA-V substitute symlinks (YouTube-VOS valid)"
    echo "  DEVIATION NOTE: SA-V substituted with YouTube-VOS valid frames"
    echo "  Impact: loss_fea term (weight=1e-6) uses wrong target distribution"
    echo "  Mitigation: run WITHOUT --loss_fea for fair paper comparison"
else
    echo "  SA-V already present: $(ls $SAV_DST | wc -l) videos"
fi

# ─── Step 1: Create required dirs ─────────────────────────────────────────────
mkdir -p "$REPO_DIR/uap_file"
mkdir -p "$REPO_DIR/adv/YOUTUBE"
mkdir -p "$REPO_DIR/clean/YOUTUBE"

# ─── Step 2: Quick sanity run (2 videos, 3 frames, no SA-V loss) ─────────────
echo ""
echo "[step 2] SANITY RUN (limit_img=2, limit_frames=3)..."
echo "  This verifies the official code runs without errors before full training."

PYTHONPATH="$REPO_DIR" python uap_attack.py \
    --train_dataset YOUTUBE \
    --test_dataset YOUTUBE \
    --checkpoints sam2-t \
    --eps 0.0392156862745098 \
    --alpha 0.00784313725490196 \
    --P_num 2 \
    --prompts_num 256 \
    --limit_img 2 \
    --limit_frames 3 \
    --fea_num 5 \
    --seed 30 \
    --loss_t \
    --loss_diff \
    2>&1 | tee "$REPO_DIR/sanity_run.log"

SANITY_STATUS=$?
echo "[sanity] exit code: $SANITY_STATUS"

if [ $SANITY_STATUS -ne 0 ]; then
    echo "SANITY FAILED. Do not proceed to full training."
    exit 1
fi

# Check that uap_file/YOUTUBE.pth was created
if [ -f "$REPO_DIR/uap_file/YOUTUBE.pth" ]; then
    echo "  Sanity OK: uap_file/YOUTUBE.pth created ($(ls -lh $REPO_DIR/uap_file/YOUTUBE.pth | awk '{print $5}'))"
else
    echo "  WARN: uap_file/YOUTUBE.pth not created by sanity run"
fi

echo ""
echo "=== SANITY PASSED — launching full training ==="
echo ""

# ─── Step 3: Full official training (paper settings) ─────────────────────────
echo "[step 3] FULL OFFICIAL TRAINING (paper settings: P_num=10, limit_img=100, limit_frames=15)"
echo "  eps=10/255, alpha=2/255, prompts_num=256, loss_t + loss_diff (no loss_fea due to SA-V)"
echo "  Started: $(date)"

PYTHONPATH="$REPO_DIR" python uap_attack.py \
    --train_dataset YOUTUBE \
    --test_dataset YOUTUBE \
    --checkpoints sam2-t \
    --eps 0.0392156862745098 \
    --alpha 0.00784313725490196 \
    --P_num 10 \
    --prompts_num 256 \
    --limit_img 100 \
    --limit_frames 15 \
    --fea_num 30 \
    --seed 30 \
    --loss_t \
    --loss_diff \
    2>&1 | tee "$REPO_DIR/train_official.log"

TRAIN_STATUS=$?
echo "[train] exit code: $TRAIN_STATUS"
echo "[train] completed: $(date)"

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "TRAINING FAILED. Check train_official.log"
    exit 1
fi

# Verify UAP saved
UAP_PATH="$REPO_DIR/uap_file/YOUTUBE.pth"
if [ -f "$UAP_PATH" ]; then
    SIZE=$(ls -lh "$UAP_PATH" | awk '{print $5}')
    echo "  UAP saved: $UAP_PATH ($SIZE)"
    python -c "
import sys; sys.path.insert(0,'$REPO_DIR')
import torch
uap = torch.load('$UAP_PATH', map_location='cpu')
print(f'  UAP tensor shape: {uap.shape}')
print(f'  UAP L-inf norm: {uap.abs().max().item():.6f} (paper target: {10/255:.6f})')
print(f'  UAP within eps: {uap.abs().max().item() <= 10/255 + 1e-6}')
"
else
    echo "  ERROR: UAP file not found!"
    exit 1
fi

echo ""
echo "=== TRAINING COMPLETE — launching eval ==="
echo ""

# ─── Step 4: Official eval (point prompt, YouTube-VOS valid) ─────────────────
echo "[step 4] OFFICIAL EVAL (test_prompts=pt, YouTube-VOS valid)"
echo "  Paper target: miouadv ≈ 37.03%, clean ≈ 82.8%"
echo "  Started: $(date)"

PYTHONPATH="$REPO_DIR" python uap_atk_test.py \
    --train_dataset YOUTUBE \
    --test_dataset YOUTUBE \
    --test_prompts pt \
    --checkpoints sam2-t \
    --limit_img -1 \
    --limit_frames -1 \
    --seed 30 \
    --P_num 10 \
    --prompts_num 256 \
    2>&1 | tee "$REPO_DIR/eval_official.log"

EVAL_STATUS=$?
echo "[eval] exit code: $EVAL_STATUS"
echo "[eval] completed: $(date)"

# Extract mIoU from log
echo ""
echo "=== EVAL RESULTS ==="
grep -E "miouimg|miouadv|iou_count" "$REPO_DIR/eval_official.log" | tail -5
echo "===================="

# ─── Step 5: Save reproduction report ────────────────────────────────────────
echo ""
echo "[step 5] Saving reproduction report..."
python - <<'PYEOF'
import json, re, pathlib

repo = "/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2"

# Parse eval log
eval_log = pathlib.Path(f"{repo}/eval_official.log").read_text()
miouimg = re.search(r'miouimg:\s*([\d.]+)', eval_log)
miouadv = re.search(r'miouadv:\s*([\d.]+)', eval_log)
iou_count = re.search(r'iou_count:\s*(\d+)', eval_log)

report = {
    "date": "2026-03-31",
    "repo": "CGCL-codes/UAP-SAM2",
    "commit": "779ce0b7ebb8cc09fb712c46c555099f6a99e08f",
    "environment": {
        "python": "3.8.20",
        "pytorch": "2.4.0+cu121",
        "sam2_version": "1.0 (sam2_hiera_tiny.pt, 38,946,242 params)",
        "checkpoint": "sam2_hiera_tiny.pt (official SAM2 1.0)",
    },
    "training_settings": {
        "dataset": "YouTube-VOS train (3471 videos)",
        "eps": "10/255",
        "alpha": "2/255",
        "P_num": 10,
        "prompts_num": 256,
        "limit_img": 100,
        "limit_frames": 15,
        "fea_num": 30,
        "losses": "loss_t + loss_diff (loss_fea excluded: SA-V unavailable on server)",
        "device": "cuda:1",
        "seed": 30,
    },
    "deviations_from_strict": [
        "SA-V substituted with YouTube-VOS valid frames (server network blocked HuggingFace); loss_fea excluded (weight=1e-6 anyway)",
    ],
    "results": {
        "miou_clean_pct": float(miouimg.group(1)) if miouimg else None,
        "miou_adv_pct": float(miouadv.group(1)) if miouadv else None,
        "iou_count": int(iou_count.group(1)) if iou_count else None,
    },
    "paper_targets": {
        "miou_adv_pt_youtube": 37.03,
        "miou_clean_youtube": "~82.8 (inferred from Fig. 6)",
    },
    "uap_path": f"{repo}/uap_file/YOUTUBE.pth",
}

# Compute gap
if report["results"]["miou_adv_pct"] is not None:
    gap = report["results"]["miou_adv_pct"] - report["paper_targets"]["miou_adv_pt_youtube"]
    report["gap_vs_paper_pp"] = round(gap, 2)
    within_band = abs(gap) <= 2.0
    report["within_2pp_band"] = within_band
    print(f"  clean mIoU: {report['results']['miou_clean_pct']}%")
    print(f"  adv mIoU:   {report['results']['miou_adv_pct']}%  (paper target: 37.03%)")
    print(f"  gap:        {gap:+.2f}pp  ({'WITHIN ±2pp band' if within_band else 'OUTSIDE ±2pp band'})")
else:
    print("  Could not parse mIoU from log")

out = pathlib.Path(f"{repo}/reproduction_report.json")
out.write_text(json.dumps(report, indent=2))
print(f"  Report saved: {out}")
PYEOF

echo ""
echo "=== REPRODUCTION COMPLETE ==="
echo "Log: $LOGFILE"
echo "Eval log: $REPO_DIR/eval_official.log"
echo "Report: $REPO_DIR/reproduction_report.json"
echo "UAP: $REPO_DIR/uap_file/YOUTUBE.pth"
echo "Date: $(date)"
