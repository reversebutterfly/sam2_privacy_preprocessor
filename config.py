"""
Central path configuration for SAM2 Privacy Preprocessor.
Edit FFMPEG_PATH and DAVIS_ROOT to match your setup.
"""
import os
import sys

# ── SAM2 (installed via pip install git+https://github.com/facebookresearch/sam2) ──
SAM2_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sam2.1_hiera_tiny.pt")
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"  # SAM2.1 tiny; hydra resolves via package

# ── FFmpeg (Windows) ─────────────────────────────────────────────────────────
# Edit this if ffmpeg is installed in a different location.
# Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
FFMPEG_PATH  = "ffmpeg"   # in PATH; change to r"C:\ffmpeg\bin\ffmpeg.exe" if needed
FFPROBE_PATH = "ffprobe"

# ── Conda env Python (for setup scripts) ─────────────────────────────────────
CONDA_PYTHON = r"D:\Users\glitterrr\anaconda3\envs\sam2_privacy_preprocessor\python.exe"

# ── DAVIS dataset ─────────────────────────────────────────────────────────────
DAVIS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "davis")
DAVIS_RES  = "480p"

# DAVIS person-category train videos (DAVIS 2017 semi-supervised, person only)
# Full list — use --videos flag to select a subset for quick runs.
DAVIS_TRAIN_VIDEOS_ALL = [
    "bear", "bike-packing", "blackswan", "bmx-bumps", "bmx-trees",
    "boat", "breakdance", "breakdance-flare", "bus", "car-roundabout",
    "car-shadow", "car-turn", "cat-girl", "classic-car", "color-run",
    "cows", "crossing", "dance-jump", "dance-twirl", "dog",
    "dog-agility", "dog-gooses", "dogs-scale", "drift-chicane",
    "drift-straight", "drift-turn", "drone", "elephant", "flamingo",
]
# Minimal subsets for quick experiments
DAVIS_MINI_TRAIN = ["bear", "breakdance", "car-shadow", "dance-jump", "dog"]
DAVIS_MINI_VAL   = [
    "bike-packing", "blackswan", "bmx-bumps", "boat", "bus",
    "car-roundabout", "car-turn", "cat-girl", "classic-car", "color-run",
    "cows", "crossing", "dance-twirl", "dog-agility", "drone",
    "elephant", "flamingo", "drift-chicane", "drift-straight", "drift-turn",
]

# ── Results directory ─────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
