# SAM2 Privacy Preprocessor - Minimal Run Script
# Runs B0 (sanity), B1 (UAP baseline), B2 (Stage 1 ours)
# Usage: conda activate sam2_privacy_preprocessor; .\run_b0_b1_b2.ps1

$PYTHON = "D:\Users\glitterrr\anaconda3\envs\sam2_privacy_preprocessor\python.exe"
$ROOT   = $PSScriptRoot

Set-Location $ROOT

Write-Host "=== SAM2 Privacy Preprocessor - B0/B1/B2 Run ===" -ForegroundColor Cyan
Write-Host "Python: $PYTHON"
Write-Host ""

# -- B0: Sanity check (500 steps, 1 video, ~30 min) --------------------------
Write-Host "[B0] Sanity check (500 steps, bear)..." -ForegroundColor Yellow
& $PYTHON train.py `
    --mode ours --stage 1 `
    --videos bear `
    --num_steps 500 --sanity `
    --lr 1e-3 --seed 42 `
    --tag B0_sanity

if ($LASTEXITCODE -ne 0) {
    Write-Host "[B0] FAILED (exit $LASTEXITCODE). Fix issues before proceeding." -ForegroundColor Red
    exit 1
}
Write-Host "[B0] PASSED" -ForegroundColor Green
Write-Host ""

# -- B1a: UAP Baseline (2000 steps, 5 videos, ~2 hr) -------------------------
Write-Host "[B1a] UAP baseline (2000 steps, 5 videos)..." -ForegroundColor Yellow
& $PYTHON train.py `
    --mode uap `
    --videos bear,breakdance,car-shadow,dance-jump,dog `
    --num_steps 2000 `
    --uap_lr 1e-2 --lr 1e-2 --seed 42 `
    --tag B1a_uap

if ($LASTEXITCODE -ne 0) {
    Write-Host "[B1a] FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "[B1a] DONE" -ForegroundColor Green
Write-Host ""

# -- B1b: UAP + LPIPS fair-budget baseline (Anti-C) --------------------------
Write-Host "[B1b] UAP+LPIPS fair-budget baseline (Anti-C)..." -ForegroundColor Yellow
& $PYTHON train.py `
    --mode uap `
    --videos bear,breakdance,car-shadow,dance-jump,dog `
    --num_steps 2000 `
    --uap_lr 1e-2 --lr 1e-2 --seed 42 `
    --uap_lpips --max_lpips 0.10 --lambda1 1.0 `
    --tag B1b_uap_lpips

if ($LASTEXITCODE -ne 0) {
    Write-Host "[B1b] FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "[B1b] DONE" -ForegroundColor Green
Write-Host ""

# -- B2: Stage 1 Ours (3000 steps, 5 videos, ~3 hr) --------------------------
Write-Host "[B2] Stage 1 ours (3000 steps, 5 videos)..." -ForegroundColor Yellow
& $PYTHON train.py `
    --mode ours --stage 1 `
    --videos bear,breakdance,car-shadow,dance-jump,dog `
    --num_steps 3000 `
    --lr 1e-3 --optimizer adam --seed 42 `
    --lambda1 1.0 --max_lpips 0.10 `
    --tag B2_stage1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[B2] FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "[B2] DONE" -ForegroundColor Green
Write-Host ""

Write-Host "=== Training done. Running post-codec eval (CRF sweep 18/23/28)... ===" -ForegroundColor Cyan
Write-Host ""

# -- Eval B1a post-codec -------------------------------------------------------
$B1A_DELTA = (Get-ChildItem "$ROOT\results\B1a_uap_*\uap_delta_final.pt" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
if ($B1A_DELTA) {
    Write-Host "[Eval B1a] $B1A_DELTA" -ForegroundColor Yellow
    & $PYTHON eval_codec.py --mode uap --uap_delta "$B1A_DELTA" `
        --videos bike-packing,blackswan,bmx-bumps,boat,bus `
        --crf 18 23 28 --tag B1a_eval
}

# -- Eval B2 post-codec --------------------------------------------------------
$B2_CKPT = (Get-ChildItem "$ROOT\results\B2_stage1_*\g_theta_final.pt" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
if ($B2_CKPT) {
    Write-Host "[Eval B2] $B2_CKPT" -ForegroundColor Yellow
    & $PYTHON eval_codec.py --mode ours --checkpoint "$B2_CKPT" `
        --videos bike-packing,blackswan,bmx-bumps,boat,bus `
        --crf 18 23 28 --tag B2_eval
}

Write-Host ""
Write-Host "=== All done. Results in results/ ===" -ForegroundColor Green
