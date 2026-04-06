#!/usr/bin/env bash
# =============================================================================
# YouTube-VOS 2019 Download Script
# Target directory: data/youtube_vos/
# =============================================================================
#
# YouTube-VOS requires a free registration at the competition website.
# Follow the steps below on the server.
#
# Official data page:
#   https://codalab.lisn.upsaclay.fr/competitions/7685
#   (YouTube-VOS 2019, Semi-supervised Video Object Segmentation track)
#
# What we need (for validation only):
#   valid.zip            — sparse JPEG frames + sparse GT annotations + meta.json
#   valid_all_frames.zip — *all* JPEG frames (dense, for running inference)
#
# We do NOT need train.zip (too large, ~41 GB) for this paper.
# =============================================================================

set -euo pipefail

YTVOS_ROOT="${1:-data/youtube_vos}"
echo "==> YouTube-VOS data will be placed in: ${YTVOS_ROOT}"
mkdir -p "${YTVOS_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: Download using gdown (Google Drive) — requires registration token
# ---------------------------------------------------------------------------
# After registering on CodaLab, you will receive download links like:
#   https://drive.google.com/file/d/<FILE_ID>/view
#
# Set these IDs here once you have them:
VALID_ID=""              # Google Drive ID for valid.zip
VALID_ALL_ID=""          # Google Drive ID for valid_all_frames.zip

# ---------------------------------------------------------------------------
# Step 2: Install gdown if not present
# ---------------------------------------------------------------------------
if ! command -v gdown &>/dev/null; then
    echo "==> Installing gdown..."
    pip install gdown -q
fi

# ---------------------------------------------------------------------------
# Step 3: Download (skip if already present)
# ---------------------------------------------------------------------------
cd "${YTVOS_ROOT}"

if [ -n "${VALID_ID}" ] && [ ! -f "valid.zip" ]; then
    echo "==> Downloading valid.zip ..."
    gdown "https://drive.google.com/uc?id=${VALID_ID}" -O valid.zip
else
    echo "==> valid.zip: already present or VALID_ID not set — skipping"
fi

if [ -n "${VALID_ALL_ID}" ] && [ ! -f "valid_all_frames.zip" ]; then
    echo "==> Downloading valid_all_frames.zip ..."
    gdown "https://drive.google.com/uc?id=${VALID_ALL_ID}" -O valid_all_frames.zip
else
    echo "==> valid_all_frames.zip: already present or VALID_ALL_ID not set — skipping"
fi

# ---------------------------------------------------------------------------
# Step 4: Extract (skip if already extracted)
# ---------------------------------------------------------------------------
if [ -f "valid.zip" ] && [ ! -d "valid" ]; then
    echo "==> Extracting valid.zip ..."
    unzip -q valid.zip
    echo "==> Extracted valid/"
fi

if [ -f "valid_all_frames.zip" ] && [ ! -d "valid_all_frames" ]; then
    echo "==> Extracting valid_all_frames.zip ..."
    unzip -q valid_all_frames.zip
    echo "==> Extracted valid_all_frames/"
fi

# ---------------------------------------------------------------------------
# Step 5: Verify directory structure
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking directory structure..."
EXPECTED_DIRS=(
    "valid/Annotations"
    "valid/JPEGImages"
    "valid_all_frames/JPEGImages"
)
OK=true
for d in "${EXPECTED_DIRS[@]}"; do
    if [ -d "${d}" ]; then
        N=$(ls "${d}" | wc -l)
        echo "  [OK] ${d}/ (${N} items)"
    else
        echo "  [MISSING] ${d}/"
        OK=false
    fi
done

if [ -f "valid/meta.json" ]; then
    echo "  [OK] valid/meta.json"
else
    echo "  [MISSING] valid/meta.json"
    OK=false
fi

if [ "${OK}" = "true" ]; then
    echo ""
    echo "==> YouTube-VOS setup complete."
    echo "==> Run the pilot:"
    echo "      python pilot_ytbvos.py --ytvos_root ${YTVOS_ROOT} --sanity"
else
    echo ""
    echo "==> Some files are missing. Please download them manually from:"
    echo "    https://codalab.lisn.upsaclay.fr/competitions/7685"
    echo ""
    echo "  Alternative: use the Roboflow / academic mirrors if available."
fi
