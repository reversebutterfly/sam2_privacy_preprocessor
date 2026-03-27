"""
Environment readiness check for SAM2 Privacy Preprocessor.
Run this BEFORE training to catch missing dependencies early.

Usage:
    conda activate sam2_privacy_preprocessor
    python check_env.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

OK     = "[OK]  "
FAIL   = "[FAIL]"
WARN   = "[WARN]"

results = {}


def check(name, fn):
    try:
        msg = fn()
        tag = OK
        results[name] = True
    except Exception as e:
        msg = str(e)
        tag = FAIL
        results[name] = False
    print(f"  {tag} {name}: {msg}")
    return results[name]


print("=" * 60)
print(" SAM2 Privacy Preprocessor — Environment Check")
print("=" * 60)

# ── Python ────────────────────────────────────────────────────
print("\n[1] Python")
check("version", lambda: f"{sys.version.split()[0]} (need ≥3.9)")

# ── PyTorch + CUDA ────────────────────────────────────────────
print("\n[2] PyTorch & CUDA")

def _torch_version():
    import torch
    v = torch.__version__
    cuda = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if cuda else "NO GPU"
    mem = (torch.cuda.get_device_properties(0).total_memory // (1024**3)) if cuda else 0
    if not cuda:
        raise RuntimeError("CUDA not available")
    if mem < 5:
        raise RuntimeError(f"GPU has only {mem} GB VRAM — need ≥ 6 GB for SAM2-T training")
    return f"{v} | CUDA: {torch.version.cuda} | GPU: {dev} ({mem} GB)"

check("torch", _torch_version)

# ── Required packages ─────────────────────────────────────────
print("\n[3] Required packages")
REQUIRED = [
    ("numpy",       "numpy"),
    ("PIL",         "Pillow"),
    ("cv2",         "opencv-python"),
    ("tqdm",        "tqdm"),
    ("lpips",       "lpips"),
    ("skimage",     "scikit-image"),
    ("scipy",       "scipy"),
    ("pywt",        "PyWavelets"),
    ("hydra",       "hydra-core"),
    ("omegaconf",   "omegaconf"),
]

for mod, pkg in REQUIRED:
    def _import(m=mod):
        __import__(m)
        return "installed"
    check(pkg, _import)

# ── SAM2 package (pip-installed) ─────────────────────────────
print("\n[4] SAM2 package")

def _check_sam2_import():
    import sam2
    return f"importable from {sam2.__file__}"
check("sam2 importable", _check_sam2_import)

# ── SAM2 checkpoint ───────────────────────────────────────────
print("\n[5] SAM2 checkpoint")
CKPT_PATH = os.path.join(ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")

def _check_ckpt():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint missing: {CKPT_PATH}")
    size_mb = os.path.getsize(CKPT_PATH) / (1024**2)
    if size_mb < 10:
        raise RuntimeError(f"Checkpoint seems incomplete ({size_mb:.1f} MB)")
    return f"{size_mb:.0f} MB at {CKPT_PATH}"
check("sam2.1_hiera_tiny.pt", _check_ckpt)

# ── FFmpeg ────────────────────────────────────────────────────
print("\n[6] FFmpeg (for post-codec evaluation)")
FFMPEG_CANDIDATES = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    "ffmpeg",  # in PATH
]

def _check_ffmpeg():
    import subprocess
    for path in FFMPEG_CANDIDATES:
        try:
            r = subprocess.run([path, "-version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                first_line = r.stdout.decode().split("\n")[0]
                return f"{first_line}  [path: {path}]"
        except (FileNotFoundError, Exception):
            continue
    raise FileNotFoundError(
        f"ffmpeg not found. Tried: {FFMPEG_CANDIDATES}"
    )
check("ffmpeg", _check_ffmpeg)

# ── DAVIS dataset ─────────────────────────────────────────────
print("\n[7] DAVIS 2017 dataset")
DAVIS_ROOT = os.path.join(ROOT, "data", "davis")

def _check_davis():
    img_root = os.path.join(DAVIS_ROOT, "JPEGImages", "480p")
    anno_root = os.path.join(DAVIS_ROOT, "Annotations", "480p")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"JPEGImages not found at {img_root}")
    if not os.path.isdir(anno_root):
        raise FileNotFoundError(f"Annotations not found at {anno_root}")
    n_vids = len([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
    return f"{n_vids} videos in {DAVIS_ROOT}"
check("davis/JPEGImages + Annotations", _check_davis)

# ── Project src/ modules ──────────────────────────────────────
print("\n[8] Project source modules")

def _check_src():
    sys.path.insert(0, ROOT)
    from src.preprocessor import ResidualPreprocessor
    from src.losses import PerceptualLoss, soft_iou_loss
    from src.metrics import jf_score
    from src.dataset import DAVISDataset
    from src.codec_eot import codec_proxy_transform
    return "all modules importable"
check("src/", _check_src)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
n_pass = sum(results.values())
n_total = len(results)
print(f"\n RESULT: {n_pass}/{n_total} checks passed\n")

FAILED = [k for k, v in results.items() if not v]
if FAILED:
    print(f" FAILED: {FAILED}")
    print("\n See setup.ps1 for install commands.\n")
    sys.exit(1)
else:
    print(" All checks passed. Ready to run experiments.\n")
    sys.exit(0)
