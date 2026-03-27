# =============================================================================
# SAM2 Privacy Preprocessor — Windows Setup Script (PowerShell)
# Run in PowerShell as: .\setup.ps1
#
# Steps:
#   1. Install Python packages into conda env
#   2. Download SAM2-T checkpoint
#   3. Download DAVIS 2017 dataset
# =============================================================================

$ENV_NAME = "sam2_privacy_preprocessor"
$PROJECT_ROOT = $PSScriptRoot
$UAP_SAM2_DIR = Join-Path $PROJECT_ROOT "..\UAP-SAM2-main\UAP-SAM2-main"
$CKPT_DIR = Join-Path $UAP_SAM2_DIR "sam2\checkpoints"
$DATA_DIR = Join-Path $PROJECT_ROOT "data\davis"

Write-Host "=== SAM2 Privacy Preprocessor Setup ===" -ForegroundColor Cyan
Write-Host "Project root : $PROJECT_ROOT"
Write-Host "UAP-SAM2 dir : $UAP_SAM2_DIR"
Write-Host "Env name     : $ENV_NAME"
Write-Host ""

# ── Step 1: Install Python packages ──────────────────────────────────────────
Write-Host "[1/3] Installing Python packages..." -ForegroundColor Yellow

# PyTorch with CUDA 12.1 (system has CUDA 12.3, driver 546.xx — cu121 is compatible)
# RTX 3060 Laptop (6 GB VRAM) — tested with torch 2.1.2+cu121
& "D:\Users\glitterrr\anaconda3\envs\$ENV_NAME\python.exe" -m pip install `
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
& "D:\Users\glitterrr\anaconda3\envs\$ENV_NAME\python.exe" -m pip install `
    "numpy==1.24.4" `
    "Pillow>=9.0.0" `
    "opencv-python>=4.7.0" `
    "tqdm>=4.65.0" `
    "lpips>=0.1.4" `
    "scikit-image>=0.20.0" `
    "scipy>=1.10.0" `
    "PyWavelets>=1.4.0" `
    "hydra-core>=1.2.0" `
    "omegaconf>=2.3.0" `
    "pycocotools>=2.0.7"

Write-Host "[1/3] Done." -ForegroundColor Green

# ── Step 2: Download SAM2-T checkpoint ───────────────────────────────────────
Write-Host "[2/3] Downloading SAM2-T checkpoint..." -ForegroundColor Yellow
$CKPT_FILE = Join-Path $CKPT_DIR "sam2_hiera_tiny.pt"

if (Test-Path $CKPT_FILE) {
    Write-Host "  Checkpoint already exists: $CKPT_FILE" -ForegroundColor Green
} else {
    # Official Meta checkpoint URL
    $CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    # Fallback (sam2 v1.0):
    # $CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    Write-Host "  Downloading from $CKPT_URL"
    Invoke-WebRequest -Uri $CKPT_URL -OutFile $CKPT_FILE -UseBasicParsing
    Write-Host "  Saved to $CKPT_FILE" -ForegroundColor Green
}

# ── Step 3: DAVIS 2017 dataset ───────────────────────────────────────────────
Write-Host "[3/3] DAVIS 2017 dataset setup..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  MANUAL STEP REQUIRED:" -ForegroundColor Red
Write-Host "  DAVIS 2017 requires registration. Download here:"
Write-Host "  https://davischallenge.org/davis2017/code.html"
Write-Host ""
Write-Host "  Download: DAVIS-2017-trainval-480p.zip"
Write-Host "  Extract to: $DATA_DIR"
Write-Host ""
Write-Host "  Expected structure:"
Write-Host "    $DATA_DIR\"
Write-Host "      Annotations\480p\<video>\00000.png ..."
Write-Host "      JPEGImages\480p\<video>\00000.jpg ..."
Write-Host "      ImageSets\2017\train.txt"
Write-Host "      ImageSets\2017\val.txt"
Write-Host ""

New-Item -ItemType Directory -Path $DATA_DIR -Force | Out-Null
Write-Host "  Created placeholder: $DATA_DIR" -ForegroundColor DarkGray

# ── Step 4: Verify FFmpeg ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4] Checking FFmpeg..." -ForegroundColor Yellow
$FFMPEG = "C:\ffmpeg\bin\ffmpeg.exe"
if (Test-Path $FFMPEG) {
    $ver = & $FFMPEG -version 2>&1 | Select-Object -First 1
    Write-Host "  Found: $ver" -ForegroundColor Green
} else {
    Write-Host "  FFmpeg not found at $FFMPEG" -ForegroundColor Red
    Write-Host "  Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    Write-Host "  Extract and place ffmpeg.exe at C:\ffmpeg\bin\ffmpeg.exe"
    Write-Host "  OR update FFMPEG_PATH in config.py"
}

# ── Step 5: Run environment check ────────────────────────────────────────────
Write-Host ""
Write-Host "[5] Running environment check..." -ForegroundColor Yellow
conda run -n $ENV_NAME python (Join-Path $PROJECT_ROOT "check_env.py")

Write-Host ""
Write-Host "=== Setup complete. If check_env.py passes, run: ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # B0 Sanity (500 steps, 1 video, ~30 min on RTX 3060)"
Write-Host "  conda activate $ENV_NAME"
Write-Host "  python train.py --stage 1 --videos bear --num_steps 500 --sanity"
Write-Host ""
Write-Host "  # B1 Baseline (UAP-SAM2 reimpl)"
Write-Host "  python train.py --mode uap --videos bear,breakdance,car-shadow,dance-jump,dog --num_steps 2000"
Write-Host ""
