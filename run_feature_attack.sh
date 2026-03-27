#!/bin/bash
# run_feature_attack.sh — Deploy feature-space attack experiments on V100 server
# Usage: bash run_feature_attack.sh
# Run from: /home/2025M_LvShaoting/sam2_privacy_preprocessor

set -e
eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate sam2_privacy_preprocessor

RESULTS_DIR="results_v100"
LOGS_DIR="logs_v100"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

VAL_VIDEOS="bike-packing,blackswan,bus,car-roundabout,car-turn,classic-car,color-run,cows,crossing"

echo "============================================================"
echo " Feature-Space Frame-0 Attack Experiments"
echo " $(date)"
echo "============================================================"

# ── Experiment F0_B: Low-cost baseline (B mode, FPN feature shift) ────────────
# Should finish in ~2h, gives early kill/proceed signal for mode B
echo ""
echo ">> Launching F0_B (FPN feature shift baseline)..."
screen -dmS F0_B bash -c "
  python attack_frame0.py \
    --attack_mode b \
    --videos $VAL_VIDEOS \
    --steps 300 --restarts 2 \
    --crf 23 --max_frames 50 \
    --tag F0_B \
    --results_dir $RESULTS_DIR \
  2>&1 | tee $LOGS_DIR/F0_B.log
  echo 'F0_B DONE'
"
echo "   screen F0_B launched"

# ── Experiment F0_CD: Primary C+D attack ─────────────────────────────────────
# Main decisive experiment, ~4h
echo ""
echo ">> Launching F0_CD (C+D: maskmem + obj_ptr)..."
screen -dmS F0_CD bash -c "
  python attack_frame0.py \
    --attack_mode cd \
    --videos $VAL_VIDEOS \
    --steps 300 --restarts 2 \
    --alpha 1.0 --beta 2.0 --gamma 0.25 --delta_mask 0.05 \
    --crf 23 --max_frames 50 \
    --tag F0_CD \
    --results_dir $RESULTS_DIR \
  2>&1 | tee $LOGS_DIR/F0_CD.log
  echo 'F0_CD DONE'
"
echo "   screen F0_CD launched"

# ── Experiment F0_CD_no_match: C+D without J_mem_match (faster, memory-light) ─
echo ""
echo ">> Launching F0_CD_nm (C+D without J_mem_match, quick signal)..."
screen -dmS F0_CD_nm bash -c "
  python attack_frame0.py \
    --attack_mode cd \
    --no_mem_match \
    --videos $VAL_VIDEOS \
    --steps 300 --restarts 2 \
    --alpha 1.0 --beta 0.0 --gamma 0.25 --delta_mask 0.05 \
    --crf 23 --max_frames 50 \
    --tag F0_CD_nm \
    --results_dir $RESULTS_DIR \
  2>&1 | tee $LOGS_DIR/F0_CD_nm.log
  echo 'F0_CD_nm DONE'
"
echo "   screen F0_CD_nm launched"

echo ""
echo "All experiments launched in screen sessions."
echo "Monitor: screen -ls"
echo "Check B:  screen -r F0_B"
echo "Check CD: screen -r F0_CD"
echo "Check nm: screen -r F0_CD_nm"
echo ""
echo "Results will be in: $RESULTS_DIR/"
echo "  attack_frame0_F0_B/results.json"
echo "  attack_frame0_F0_CD/results.json"
echo "  attack_frame0_F0_CD_nm/results.json"
echo ""
echo "Kill criteria (from FEATURE_ATTACK_PLAN.md):"
echo "  kill_metric < 0.03 → EARLY KILL (negative result paper)"
echo "  kill_metric >= 0.05 → PROCEED (full g_theta training)"
