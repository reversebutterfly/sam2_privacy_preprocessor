# =============================================================================
# SAM2 Privacy Preprocessor — Experiment Run Script
# Run in PowerShell: .\run_experiments.ps1
#
# Run order: R001 (sanity) → R002/R003 (baselines) → R004→R005→R006 (main)
# After each run, check results/<RUN_ID>/results.json for success criteria.
# =============================================================================

$PY  = "D:\Users\glitterrr\anaconda3\envs\sam2_privacy_preprocessor\python.exe"
$ROOT = $PSScriptRoot

function Run-Exp($label, $cmd) {
    Write-Host ""
    Write-Host "[$label] Starting..." -ForegroundColor Cyan
    Write-Host "  $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$label] FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "[$label] Done" -ForegroundColor Green
}

# =============================================================================
# M0: SANITY CHECK (B0) — ~30 min on RTX 3060
# Success: eval_J < 0.70 on overfit video (bear), no gradient errors
# =============================================================================
Run-Exp "R001-SANITY" "& '$PY' '$ROOT\train.py' ``
    --run_id R001 ``
    --mode ours --stage 1 ``
    --videos bear ``
    --num_steps 500 ``
    --lr 5e-4 --lambda1 1.0 ``
    --eval_every 50 ``
    --sanity"

Write-Host ""
Write-Host "R001 done. Check results/R001/results.json — if eval_J < 0.70, proceed." -ForegroundColor Yellow
Write-Host "Press Enter to continue to M1 (baselines), or Ctrl+C to stop." -ForegroundColor Yellow
Read-Host

# =============================================================================
# M1: BASELINES (B1) — ~2-4h total
# R002: UAP (lp-norm only, no perceptual)
# R003: UAP + LPIPS constraint (fair-budget)
# =============================================================================
Run-Exp "R002-UAP-BASELINE" "& '$PY' '$ROOT\train.py' ``
    --run_id R002 ``
    --mode uap ``
    --videos bear,breakdance,car-shadow,dance-jump,dog ``
    --num_steps 2000 ``
    --eps 0.03137 --alpha 0.00784 ``
    --lambda1 0.0 ``
    --eval_every 200"

Run-Exp "R003-UAP-LPIPS" "& '$PY' '$ROOT\train.py' ``
    --run_id R003 ``
    --mode uap ``
    --videos bear,breakdance,car-shadow,dance-jump,dog ``
    --num_steps 2000 ``
    --eps 0.03137 --alpha 0.00784 ``
    --lambda1 1.0 --lpips_thresh 0.10 ``
    --eval_every 200"

# Evaluate baselines post-codec
Run-Exp "R002-EVAL" "& '$PY' '$ROOT\eval_codec.py' ``
    --run_id R002_eval ``
    --mode uap --checkpoint '$ROOT\results\R002\checkpoint.pth' ``
    --videos mini_val ``
    --crf 18,23,28"

Run-Exp "R003-EVAL" "& '$PY' '$ROOT\eval_codec.py' ``
    --run_id R003_eval ``
    --mode uap --checkpoint '$ROOT\results\R003\checkpoint.pth' ``
    --videos mini_val ``
    --crf 18,23,28"

Write-Host ""
Write-Host "Baseline evals done. Check results/R002_eval and R003_eval." -ForegroundColor Yellow
Write-Host "Expect: post-H264-CRF23 J drop < 3 pp (codec kills lp-norm attack)." -ForegroundColor Yellow
Write-Host "Press Enter to continue to M2 (main method), or Ctrl+C to stop." -ForegroundColor Yellow
Read-Host

# =============================================================================
# M2: MAIN METHOD (B2 Stages 1→2→3) — ~6h total
# R004: Stage 1 (residual + perceptual)
# R005: Stage 2 (+ temporal consistency)
# R006: Stage 3 (+ codec-aware EOT) ← core paper result
# =============================================================================
Run-Exp "R004-STAGE1" "& '$PY' '$ROOT\train.py' ``
    --run_id R004 ``
    --mode ours --stage 1 ``
    --videos bear,breakdance,car-shadow,dance-jump,dog ``
    --num_steps 1000 ``
    --lr 5e-4 --lambda1 1.0 --lpips_thresh 0.10 ``
    --eval_every 100"

Run-Exp "R005-STAGE2" "& '$PY' '$ROOT\train.py' ``
    --run_id R005 ``
    --mode ours --stage 2 ``
    --checkpoint '$ROOT\results\R004\checkpoint.pth' ``
    --videos bear,breakdance,car-shadow,dance-jump,dog ``
    --num_steps 1000 ``
    --lr 2e-4 --lambda1 1.0 --lambda2 0.1 ``
    --eval_every 100"

Run-Exp "R006-STAGE3" "& '$PY' '$ROOT\train.py' ``
    --run_id R006 ``
    --mode ours --stage 3 ``
    --checkpoint '$ROOT\results\R005\checkpoint.pth' ``
    --videos bear,breakdance,car-shadow,dance-jump,dog ``
    --num_steps 1000 ``
    --lr 1e-4 --lambda1 1.0 --lambda2 0.1 ``
    --eval_every 100"

# Evaluate Stage 2 and Stage 3 post-codec (for B3 ablation)
Run-Exp "R005-EVAL" "& '$PY' '$ROOT\eval_codec.py' ``
    --run_id R005_eval ``
    --mode ours --checkpoint '$ROOT\results\R005\checkpoint.pth' ``
    --videos mini_val ``
    --crf 18,23,28"

Run-Exp "R006-EVAL" "& '$PY' '$ROOT\eval_codec.py' ``
    --run_id R006_eval ``
    --mode ours --checkpoint '$ROOT\results\R006\checkpoint.pth' ``
    --videos mini_val ``
    --crf 18,23,28"

# Clean baseline (reference J&F)
Run-Exp "R000-CLEAN" "& '$PY' '$ROOT\eval_codec.py' ``
    --run_id R000_clean ``
    --mode clean ``
    --videos mini_val ``
    --crf 23"

Write-Host ""
Write-Host "=== All must-run experiments complete ===" -ForegroundColor Green
Write-Host "Key files to check:"
Write-Host "  results/R006_eval/results.json  ← Stage 3 post-codec J drop (target ≥12 pp)"
Write-Host "  results/R005_eval/results.json  ← Stage 2 post-codec (should be worse than R006)"
Write-Host "  results/R002_eval/results.json  ← UAP baseline (should collapse post-codec)"
Write-Host "  results/R000_clean/results.json ← clean reference"
