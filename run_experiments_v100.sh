#!/usr/bin/env bash
# ============================================================
# run_experiments_v100.sh
# Full experiment sequence for V100 server (16/32 GB VRAM).
#
# Assumes:
#   - eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate sam2_privacy_preprocessor
#   - DAVIS data at: /path/to/DAVIS  (update DAVIS_ROOT below)
#   - SAM2.1-tiny checkpoint at: checkpoints/sam2.1_hiera_tiny.pt
#   - Code at: /home/2025M_LvShaoting/sam2_privacy_preprocessor
#
# Run inside screen/tmux:
#   screen -S sam2_exp
#   bash run_experiments_v100.sh 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
# ============================================================

set -euo pipefail

# ── Paths (auto-derived from script location, no edits needed) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAVIS_ROOT="$SCRIPT_DIR/data/davis"        # mirrors config.py logic
CKPT="$SCRIPT_DIR/checkpoints/sam2.1_hiera_tiny.pt"
CFG="configs/sam2.1/sam2.1_hiera_t.yaml"
FFMPEG="ffmpeg"
RESULTS="$SCRIPT_DIR/results_v100"

# ── Training hyperparameters ──────────────────────────────────
G_SIZE=512         # g_theta resolution (vs 256 locally)
ACCUM=4            # gradient accumulation (effective batch = 4)
MAX_FRAMES=50      # frames per video for training
MAX_FRAMES_EVAL=80 # frames per video for eval

# Training videos: first 20 from DAVIS_TRAIN_VIDEOS_ALL
TRAIN_VIDS="bear,bike-packing,blackswan,bmx-bumps,bmx-trees,boat,breakdance,breakdance-flare,bus,car-roundabout,car-shadow,car-turn,cat-girl,classic-car,color-run,cows,crossing,dance-jump,dance-twirl,dog"

# Eval videos: remaining 9 from DAVIS_MINI_VAL (hold-out)
EVAL_VIDS="dog-agility,drone,elephant,flamingo,drift-chicane,drift-straight,drift-turn,dogs-scale,dog-gooses"

mkdir -p "$SCRIPT_DIR/logs" "$RESULTS"

echo "============================================================"
echo "SAM2 Privacy Preprocessor — V100 Experiment Suite"
echo "Start: $(date)"
echo "============================================================"

# ── Step 0: Profiling pilot ───────────────────────────────────
echo ""
echo "[PROFILE] 1-GPU profiling pilot at g_theta_size=$G_SIZE ..."
python profile_gpu.py \
    --stage 1 \
    --g_theta_size "$G_SIZE" \
    --num_steps 80 \
    --davis_root "$DAVIS_ROOT" \
    --videos "bear,breakdance,car-shadow,dance-jump,dog" \
    --sam2_checkpoint "$CKPT" \
    --sam2_config "$CFG" \
    2>&1 | tee logs/profile.log
echo "[PROFILE] Done."

# ── B0: Sanity overfit ────────────────────────────────────────
echo ""
echo "[B0] Sanity overfit on single video ..."
python train.py \
    --mode ours --stage 1 --sanity \
    --tag B0_v100 \
    --videos bear \
    --max_frames 30 \
    --num_steps 500 \
    --lr 1e-4 \
    --g_theta_size "$G_SIZE" \
    --g_accum_steps 1 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/B0.log
echo "[B0] Done."

# ── B1a: UAP baseline ─────────────────────────────────────────
echo ""
echo "[B1a] UAP baseline (5000 steps, 20 videos) ..."
python train.py \
    --mode uap \
    --tag B1a_v100 \
    --videos "$TRAIN_VIDS" \
    --max_frames "$MAX_FRAMES" \
    --num_steps 5000 \
    --lr 0.01 \
    --max_delta 0.03137 \
    --g_accum_steps 1 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/B1a.log
echo "[B1a] Done."

# ── B1b: UAP + LPIPS fair-budget baseline ─────────────────────
echo ""
echo "[B1b] UAP + LPIPS baseline (5000 steps, 20 videos) ..."
python train.py \
    --mode uap --uap_lpips \
    --tag B1b_v100 \
    --videos "$TRAIN_VIDS" \
    --max_frames "$MAX_FRAMES" \
    --num_steps 5000 \
    --lr 0.01 \
    --max_delta 0.03137 \
    --g_accum_steps 1 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/B1b.log
echo "[B1b] Done."

# ── B2 Stage 1: g_theta, no codec ────────────────────────────
echo ""
echo "[B2 Stage1] g_theta Stage 1 (5000 steps, 20 videos) ..."
python train.py \
    --mode ours --stage 1 \
    --tag B2_v100 \
    --videos "$TRAIN_VIDS" \
    --max_frames "$MAX_FRAMES" \
    --num_steps 5000 \
    --lr 1e-4 \
    --g_theta_size "$G_SIZE" \
    --g_accum_steps "$ACCUM" \
    --lambda1 1.0 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/B2_stage1.log

B2S1_CKPT="$RESULTS/B2_v100_ours_s1_steps5000/g_theta_final.pt"
echo "[B2 Stage1] Done. Checkpoint: $B2S1_CKPT"

# ── B2 Stage 2: + temporal consistency ───────────────────────
echo ""
echo "[B2 Stage2] g_theta Stage 2 (3000 steps, from B2S1 ckpt) ..."
python train.py \
    --mode ours --stage 2 \
    --tag B2_v100 \
    --checkpoint "$B2S1_CKPT" \
    --videos "$TRAIN_VIDS" \
    --max_frames "$MAX_FRAMES" \
    --num_steps 3000 \
    --lr 5e-5 \
    --g_theta_size "$G_SIZE" \
    --g_accum_steps "$ACCUM" \
    --lambda1 1.0 --lambda2 0.1 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/B2_stage2.log

B2S2_CKPT="$RESULTS/B2_v100_ours_s2_steps3000/g_theta_final.pt"
echo "[B2 Stage2] Done. Checkpoint: $B2S2_CKPT"

# ── B3 Stage 3: + codec-proxy EOT ────────────────────────────
echo ""
echo "[B3 Stage3] g_theta Stage 3 EOT (3000 steps, from B2S2 ckpt) ..."
python train.py \
    --mode ours --stage 3 \
    --tag B3_v100 \
    --checkpoint "$B2S2_CKPT" \
    --videos "$TRAIN_VIDS" \
    --max_frames "$MAX_FRAMES" \
    --num_steps 3000 \
    --lr 5e-5 \
    --g_theta_size "$G_SIZE" \
    --g_accum_steps "$ACCUM" \
    --lambda1 1.0 --lambda2 0.1 \
    --eot_prob 0.5 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/B3_stage3.log

B3S3_CKPT="$RESULTS/B3_v100_ours_s3_steps3000/g_theta_final.pt"
echo "[B3 Stage3] Done. Checkpoint: $B3S3_CKPT"

# ── Eval: B1a UAP ─────────────────────────────────────────────
echo ""
echo "[EVAL B1a] UAP codec eval ..."
python eval_codec.py \
    --mode uap \
    --tag B1a_v100 \
    --uap_delta "$RESULTS/B1a_v100_uap_eps0.0314_steps5000/uap_delta_final.pt" \
    --videos "$EVAL_VIDS" \
    --max_frames "$MAX_FRAMES_EVAL" \
    --crf 18 23 28 35 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --ffmpeg_path "$FFMPEG" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/eval_B1a.log

# ── Eval: B1b UAP+LPIPS ───────────────────────────────────────
echo ""
echo "[EVAL B1b] UAP+LPIPS codec eval ..."
python eval_codec.py \
    --mode uap \
    --tag B1b_v100 \
    --uap_delta "$RESULTS/B1b_v100_uap_eps0.0314_steps5000_lpips/uap_delta_final.pt" \
    --videos "$EVAL_VIDS" \
    --max_frames "$MAX_FRAMES_EVAL" \
    --crf 18 23 28 35 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --ffmpeg_path "$FFMPEG" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/eval_B1b.log

# ── Eval: B2 Stage 1 (ablation: no temporal, no codec) ────────
echo ""
echo "[EVAL B2S1] Stage1 codec eval (ablation) ..."
python eval_codec.py \
    --mode ours --stage 1 \
    --tag B2S1_v100 \
    --checkpoint "$B2S1_CKPT" \
    --g_theta_size "$G_SIZE" \
    --videos "$EVAL_VIDS" \
    --max_frames "$MAX_FRAMES_EVAL" \
    --crf 18 23 28 35 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --ffmpeg_path "$FFMPEG" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/eval_B2S1.log

# ── Eval: B2 Stage 2 (ablation: temporal, no codec) ───────────
echo ""
echo "[EVAL B2S2] Stage2 codec eval (ablation: temporal, no EOT) ..."
python eval_codec.py \
    --mode ours --stage 2 \
    --tag B2S2_v100 \
    --checkpoint "$B2S2_CKPT" \
    --g_theta_size "$G_SIZE" \
    --videos "$EVAL_VIDS" \
    --max_frames "$MAX_FRAMES_EVAL" \
    --crf 18 23 28 35 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --ffmpeg_path "$FFMPEG" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/eval_B2S2.log

# ── Eval: B3 Stage 3 (full method) ───────────────────────────
echo ""
echo "[EVAL B3S3] Stage3 EOT codec eval (full method) ..."
python eval_codec.py \
    --mode ours --stage 3 \
    --tag B3S3_v100 \
    --checkpoint "$B3S3_CKPT" \
    --g_theta_size "$G_SIZE" \
    --videos "$EVAL_VIDS" \
    --max_frames "$MAX_FRAMES_EVAL" \
    --crf 18 23 28 35 \
    --save_dir "$RESULTS" \
    --davis_root "$DAVIS_ROOT" \
    --ffmpeg_path "$FFMPEG" \
    --sam2_checkpoint "$CKPT" --sam2_config "$CFG" \
    2>&1 | tee logs/eval_B3S3.log

echo ""
echo "============================================================"
echo "All experiments done: $(date)"
echo "Results in: $RESULTS/"
echo "Logs in:    logs/"
echo "============================================================"
