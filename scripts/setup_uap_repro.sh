#!/bin/bash
# UAP-SAM2 Strict Reproduction Setup
# Runs on server. Execute with:
#   screen -S uap_setup bash scripts/setup_uap_repro.sh
# Logs to: ~/uap_repro_setup.log

set -e
LOGFILE="$HOME/uap_repro_setup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== UAP-SAM2 Strict Reproduction Setup ==="
echo "Date: $(date)"
echo ""

CONDA_BIN=/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/bin
eval "$($CONDA_BIN/conda shell.bash hook)"
conda activate UAP-SAM2

REPO_DIR="$HOME/UAP-SAM2"
DATA_LINK="$REPO_DIR/data"
CKPT_DIR="$REPO_DIR/checkpoints"
YTVOS_SRC="/IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor/data/youtube_vos"
SAV_DST="$REPO_DIR/data/sav_test/JPEGImages_24fps"

# ─── Step 1: Clone official repo ─────────────────────────────────────────────
echo "[step 1] Cloning CGCL-codes/UAP-SAM2..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Already exists, pulling latest..."
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/CGCL-codes/UAP-SAM2.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Record commit hash
COMMIT_HASH=$(git rev-parse HEAD)
echo ""
echo "=== VERSION INFO ==="
echo "UAP-SAM2 commit: $COMMIT_HASH"
echo "Repo URL: https://github.com/CGCL-codes/UAP-SAM2"
echo "Clone date: $(date)"
echo "===================="
echo ""

# ─── Step 2: Install SAM2 from vendored copy ─────────────────────────────────
echo "[step 2] Installing SAM2 from vendored copy inside UAP-SAM2..."
if [ -d "$REPO_DIR/sam2" ]; then
    cd "$REPO_DIR/sam2"
    SAM2_COMMIT=$(git log --oneline -1 2>/dev/null || echo "not a git repo")
    echo "  SAM2 sub-repo: $SAM2_COMMIT"
    pip install -e . --quiet
    cd "$REPO_DIR"
    echo "  SAM2 installed from: $REPO_DIR/sam2"
else
    echo "  No sam2/ subdir found. Checking submodule..."
    git submodule update --init --recursive
    if [ -d "$REPO_DIR/sam2" ]; then
        cd "$REPO_DIR/sam2"
        SAM2_COMMIT=$(git log --oneline -1)
        echo "  SAM2 submodule: $SAM2_COMMIT"
        pip install -e . --quiet
        cd "$REPO_DIR"
    else
        echo "  ERROR: sam2/ directory not found in repo. Manual install needed."
    fi
fi

# Verify SAM2 importable
python -c "import sam2; print('  sam2 version:', getattr(sam2, '__version__', 'unknown'))" || \
    echo "  WARNING: sam2 not importable after install"

# ─── Step 3: Download SAM2 1.0 checkpoint ────────────────────────────────────
echo "[step 3] Downloading SAM2 1.0 hiera_tiny checkpoint..."
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_DIR/sam2_hiera_tiny.pt" ]; then
    echo "  Already exists: $CKPT_DIR/sam2_hiera_tiny.pt"
    ls -lh "$CKPT_DIR/sam2_hiera_tiny.pt"
else
    echo "  Downloading from Meta CDN (SAM2 1.0 release 2024-07-28)..."
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" \
        -O "$CKPT_DIR/sam2_hiera_tiny.pt"
    echo "  Downloaded: $(ls -lh $CKPT_DIR/sam2_hiera_tiny.pt)"
fi

# Also check for download_ckpts.sh approach
if [ -f "$CKPT_DIR/download_ckpts.sh" ]; then
    echo "  Found download_ckpts.sh, using it as reference..."
    cat "$CKPT_DIR/download_ckpts.sh"
fi

echo ""
echo "=== CHECKPOINT INFO ==="
ls -lh "$CKPT_DIR/" 2>/dev/null || echo "No checkpoints found"
echo "======================"

# ─── Step 4: Align data paths ────────────────────────────────────────────────
echo "[step 4] Aligning data paths..."
mkdir -p "$REPO_DIR/data"

# YouTube-VOS → data/YOUTUBE
if [ ! -e "$REPO_DIR/data/YOUTUBE" ]; then
    echo "  Creating symlink: data/YOUTUBE -> $YTVOS_SRC"
    ln -s "$YTVOS_SRC" "$REPO_DIR/data/YOUTUBE"
else
    echo "  data/YOUTUBE already exists: $(ls -la $REPO_DIR/data/YOUTUBE | head -1)"
fi

# Verify data/YOUTUBE/train/JPEGImages structure
echo "  Verifying YouTube-VOS structure..."
N_VIDEOS=$(ls "$REPO_DIR/data/YOUTUBE/train/JPEGImages/" 2>/dev/null | wc -l)
echo "  data/YOUTUBE/train/JPEGImages: $N_VIDEOS videos"
N_ANNO=$(ls "$REPO_DIR/data/YOUTUBE/train/Annotations/" 2>/dev/null | wc -l)
echo "  data/YOUTUBE/train/Annotations: $N_ANNO videos"

# SA-V: check if available, download subset if not
echo "[step 4b] Checking SA-V test data..."
if [ -d "$SAV_DST" ] && [ "$(ls -A $SAV_DST 2>/dev/null | wc -l)" -gt 0 ]; then
    N_SAV=$(ls "$SAV_DST" | wc -l)
    echo "  SA-V already present: $N_SAV videos under $SAV_DST"
else
    echo "  SA-V NOT found at $SAV_DST"
    echo "  Attempting to download SA-V test frames (minimal subset for fea_num=30)..."
    mkdir -p "$SAV_DST"
    # Try downloading SA-V test set from HuggingFace mirror
    python3 - <<'PYEOF'
import os, urllib.request, json

sav_dst = os.environ.get("SAV_DST", "")
# Try SA-V minimal download via huggingface
# SA-V test videos listing from official SAM2 repo
try:
    print("  Trying HuggingFace SA-V download...")
    import subprocess
    result = subprocess.run(
        ["pip", "install", "huggingface_hub", "-q"],
        capture_output=True, text=True
    )
    from huggingface_hub import snapshot_download
    # Only download JPEGImages_24fps
    snapshot_download(
        repo_id="facebook/sam2-video-benchmark-sav",
        repo_type="dataset",
        local_dir="/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/data/sav_test",
        allow_patterns=["JPEGImages_24fps/*/*.jpg"],
        max_workers=4,
    )
    print("  SA-V download complete via HuggingFace Hub")
except Exception as e:
    print(f"  HuggingFace download failed: {e}")
    print("  SA-V data must be downloaded manually.")
    print("  See: https://ai.meta.com/datasets/segment-anything-video-downloads/")
    print("  Alternatively, we can use a placeholder (changes loss_fea term only).")
PYEOF
fi

# ─── Step 5: Install missing Python packages ─────────────────────────────────
echo "[step 5] Installing any missing packages..."
pip install pillow matplotlib scikit-learn -q 2>/dev/null || true
pip install imagecorruptions -q 2>/dev/null || true

# ─── Step 6: Verify all imports work ─────────────────────────────────────────
echo "[step 6] Testing official module imports..."
cd "$REPO_DIR"
python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

results = {}
for mod in ['attack_setting', 'dataset_YOUTUBE', 'sam2_util', 'sam2.build_sam']:
    try:
        __import__(mod)
        results[mod] = 'OK'
    except Exception as e:
        results[mod] = f'FAIL: {e}'

print("\n=== MODULE IMPORT CHECK ===")
for mod, status in results.items():
    print(f"  {mod:30s}: {status}")
print("===========================\n")

all_ok = all(v == 'OK' for v in results.values())
print(f"All imports OK: {all_ok}")
if not all_ok:
    sys.exit(1)
PYEOF
IMPORT_STATUS=$?

# ─── Step 7: Verify checkpoint can be loaded ─────────────────────────────────
echo "[step 7] Verifying SAM2 1.0 checkpoint loads..."
cd "$REPO_DIR"
python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import torch

ckpt = "checkpoints/sam2_hiera_tiny.pt"
try:
    data = torch.load(ckpt, map_location="cpu")
    keys = list(data.keys()) if isinstance(data, dict) else ["<tensor>"]
    print(f"  Checkpoint loaded OK. Keys: {keys[:3]}...")
    if isinstance(data, dict) and 'model' in data:
        n_params = sum(v.numel() for v in data['model'].values() if hasattr(v, 'numel'))
        print(f"  Model params: {n_params:,}")
except Exception as e:
    print(f"  ERROR loading checkpoint: {e}")
    sys.exit(1)
PYEOF

# ─── Step 8: Run sanity import check for attack_setting ──────────────────────
echo "[step 8] Checking attack_setting config..."
cd "$REPO_DIR"
python - <<'PYEOF'
import sys; sys.path.insert(0, '.')
from attack_setting import SAM_MASK_THRESH, CUDA_DEVICES, make_multi_prompts
print(f"  SAM_MASK_THRESH: {SAM_MASK_THRESH}")
print(f"  CUDA_DEVICES: {CUDA_DEVICES}")
prompts = make_multi_prompts(prompt_num=256)
print(f"  make_multi_prompts(256): {len(prompts)} prompts")
print(f"  Prompt types: {set(p.get('type') for p in prompts)}")
PYEOF

echo ""
echo "=== SETUP SUMMARY ==="
echo "UAP-SAM2 repo: $REPO_DIR"
echo "Commit: $(cd $REPO_DIR && git rev-parse HEAD)"
echo "Conda env: UAP-SAM2 (Python 3.8, PyTorch 2.4.0)"
echo "SAM2 1.0 ckpt: $CKPT_DIR/sam2_hiera_tiny.pt"
echo "YouTube-VOS: $REPO_DIR/data/YOUTUBE -> $YTVOS_SRC"
echo "SA-V: $([ -d $SAV_DST ] && echo 'PRESENT' || echo 'MISSING - blocker for strict repro')"
echo "Import check: $([ $IMPORT_STATUS -eq 0 ] && echo 'ALL OK' || echo 'SOME FAILURES')"
echo "====================="
echo ""
echo "Setup complete. $(date)"
echo "Log: $LOGFILE"
