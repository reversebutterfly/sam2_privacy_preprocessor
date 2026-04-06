#!/bin/bash
# UAP-SAM2 Strict Reproduction Setup (fixed)
# sam2/ inside UAP-SAM2 is a flat package — no pip install needed.
# Just run from UAP-SAM2 root with PYTHONPATH=.

set -e
LOGFILE="$HOME/uap_repro_setup2.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== UAP-SAM2 Setup (fixed) ==="
echo "Date: $(date)"

CONDA_BIN=/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/bin
eval "$($CONDA_BIN/conda shell.bash hook)"
conda activate UAP-SAM2

REPO_DIR="/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2"
cd "$REPO_DIR"

echo ""
echo "=== VERSION INFO ==="
echo "UAP-SAM2 commit: $(git rev-parse HEAD)"
echo "SAM2 flat copy: $REPO_DIR/sam2 (no setup.py — importable via PYTHONPATH=.)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "===================="

# ─── Step 1: Download SAM2 1.0 checkpoint ────────────────────────────────────
echo ""
echo "[step 1] Downloading SAM2 1.0 checkpoint..."
CKPT="$REPO_DIR/checkpoints/sam2_hiera_tiny.pt"
mkdir -p "$REPO_DIR/checkpoints"
if [ -f "$CKPT" ]; then
    echo "  Already exists: $CKPT ($(ls -lh $CKPT | awk '{print $5}'))"
else
    echo "  Downloading from Meta CDN..."
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" \
        -O "$CKPT"
    echo "  Downloaded: $CKPT ($(ls -lh $CKPT | awk '{print $5}'))"
fi

# ─── Step 2: Install missing pip packages ────────────────────────────────────
echo ""
echo "[step 2] Installing missing packages..."
pip install PyWavelets pillow matplotlib scikit-learn imagecorruptions -q 2>&1 | tail -3

# ─── Step 3: Align data paths ────────────────────────────────────────────────
echo ""
echo "[step 3] Aligning data paths..."
YTVOS_SRC="/IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor/data/youtube_vos"

mkdir -p "$REPO_DIR/data"

# YouTube-VOS → data/YOUTUBE
if [ ! -e "$REPO_DIR/data/YOUTUBE" ]; then
    ln -s "$YTVOS_SRC" "$REPO_DIR/data/YOUTUBE"
    echo "  Symlink: data/YOUTUBE -> $YTVOS_SRC"
else
    echo "  data/YOUTUBE: already exists"
fi

N_VID=$(ls "$REPO_DIR/data/YOUTUBE/train/JPEGImages/" | wc -l)
echo "  YouTube-VOS train videos: $N_VID"

# SA-V: create placeholder with a warning (strict repro needs real SA-V)
SAV_DST="$REPO_DIR/data/sav_test/JPEGImages_24fps"
if [ -d "$SAV_DST" ] && [ "$(ls -A $SAV_DST 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  SA-V: present ($(ls $SAV_DST | wc -l) videos)"
else
    echo "  SA-V: NOT present. Attempting HuggingFace download..."
    mkdir -p "$SAV_DST"
    python - <<'PYEOF'
import subprocess, sys, os

# Try to download SA-V via HuggingFace hub
try:
    subprocess.run(["pip", "install", "huggingface_hub", "-q"], check=True, capture_output=True)
    from huggingface_hub import hf_hub_download, list_repo_files
    import pathlib

    dst = pathlib.Path("/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/data/sav_test/JPEGImages_24fps")
    dst.mkdir(parents=True, exist_ok=True)

    # SA-V is hosted at facebook/segment-anything-v (need to check exact repo)
    print("  Checking HuggingFace for SA-V dataset...")
    # The dataset might be accessible at facebook/sam2
    # Try downloading a minimal set of test frames
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id="facebook/sam2-video-benchmark-sav",
        repo_type="dataset",
        local_dir=str(dst.parent.parent),
        allow_patterns=["JPEGImages_24fps/*/00000.jpg"],  # just first frame of each video
        max_workers=4,
    )
    frames = list(dst.rglob("*.jpg"))
    print(f"  Downloaded {len(frames)} SA-V frames")
except Exception as e:
    print(f"  HuggingFace SA-V download failed: {e}")
    print("  SA-V is a BLOCKER for strict loss_fea. Will work around in Phase 5b.")
PYEOF
fi

# ─── Step 4: Test all imports from UAP-SAM2 root ─────────────────────────────
echo ""
echo "[step 4] Testing all imports (PYTHONPATH=.)..."
PYTHONPATH="$REPO_DIR" python - <<'PYEOF'
import sys
sys.path.insert(0, '/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2')
os_import_ok = {}

# Test each official module
for mod in ['sam2.build_sam', 'sam2.utils.misc', 'sam2.modeling.sam2_base',
            'attack_setting', 'dataset_YOUTUBE', 'sam2_util']:
    try:
        __import__(mod)
        os_import_ok[mod] = 'OK'
    except Exception as e:
        os_import_ok[mod] = f'FAIL: {e}'

print("\n=== MODULE IMPORT CHECK ===")
ok_count = 0
for mod, status in os_import_ok.items():
    print(f"  {mod:40s}: {status}")
    if status == 'OK':
        ok_count += 1
print(f"===========================")
print(f"  {ok_count}/{len(os_import_ok)} imports OK")

if ok_count < len(os_import_ok):
    import sys; sys.exit(1)
PYEOF
IMPORT_STATUS=$?

# ─── Step 5: Verify checkpoint loads ─────────────────────────────────────────
echo ""
echo "[step 5] Verifying SAM2 1.0 checkpoint..."
PYTHONPATH="$REPO_DIR" python - <<'PYEOF'
import sys
sys.path.insert(0, '/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2')
import torch
ckpt_path = "/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/checkpoints/sam2_hiera_tiny.pt"
data = torch.load(ckpt_path, map_location="cpu")
if isinstance(data, dict) and 'model' in data:
    n = sum(v.numel() for v in data['model'].values() if hasattr(v, 'numel'))
    print(f"  SAM2 1.0 ckpt loaded OK — {n:,} parameters")
else:
    print(f"  Loaded (type={type(data)})")
PYEOF

# ─── Step 6: Test build_sam2_video_predictor ─────────────────────────────────
echo ""
echo "[step 6] Testing build_sam2_video_predictor with SAM2 1.0..."
PYTHONPATH="$REPO_DIR" python - <<'PYEOF'
import sys, os
os.chdir('/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2')
sys.path.insert(0, '/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2')

from sam2.build_sam import build_sam2_video_predictor
try:
    model = build_sam2_video_predictor(
        config_file="configs/sam2/sam2_hiera_t.yaml",
        ckpt_path="checkpoints/sam2_hiera_tiny.pt",
        device="cpu",
    )
    print(f"  build_sam2_video_predictor OK (device=cpu)")
    print(f"  image_size: {model.image_size}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)
PYEOF
MODEL_STATUS=$?

# ─── Step 7: Test load_model from sam2_util ───────────────────────────────────
echo ""
echo "[step 7] Testing sam2_util.load_model (device=cpu for sanity)..."
PYTHONPATH="$REPO_DIR" python - <<'PYEOF'
import sys, os
os.chdir('/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2')
sys.path.insert(0, '/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2')
from argparse import Namespace
from sam2_util import load_model

try:
    # Patch device to cpu for this sanity check
    import sam2_util as su
    orig = su.load_model
    def patched_load(args, device='cpu'):
        return orig(args, device='cpu')

    args = Namespace(checkpoints='sam2-t')
    result = patched_load(args)
    print(f"  load_model OK: {type(result)}")
except Exception as e:
    print(f"  FAIL: {e}")
PYEOF

echo ""
echo "=== FINAL STATUS ==="
echo "Repo commit:  $(cd $REPO_DIR && git rev-parse HEAD)"
echo "Checkpoint:   $CKPT ($(ls -lh $CKPT 2>/dev/null | awk '{print $5}' || echo 'MISSING'))"
echo "YouTube-VOS:  $N_VID videos in train/JPEGImages"
echo "SA-V:         $([ -d $SAV_DST ] && ls $SAV_DST | wc -l || echo 0) videos at $SAV_DST"
echo "Imports:      $([ $IMPORT_STATUS -eq 0 ] && echo 'ALL OK' || echo 'SOME FAILED')"
echo "Model build:  $([ $MODEL_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED')"
echo "===================="
echo ""
echo "Done: $(date)"
