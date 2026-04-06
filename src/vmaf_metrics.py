"""
VMAF perceptual quality metrics for the SAM2 Privacy Preprocessor.

Computes VMAF (Video Multi-method Assessment Fusion) using FFmpeg's libvmaf filter.
Falls back to numpy PSNR/SSIM when libvmaf is not compiled into the local FFmpeg.

Typical usage:
  from src.vmaf_metrics import compute_vmaf_frames

  result = compute_vmaf_frames(orig_frames, edited_frames, ffmpeg_path)
  # result = {'vmaf': 92.1, 'psnr': 34.2, 'ssim': 0.96, 'method': 'libvmaf'}
  # or      = {'vmaf': None, 'psnr': 34.2, 'ssim': 0.96, 'method': 'fallback_ssim'}

Integration with sweep pipeline:
  from src.vmaf_metrics import compute_vmaf_frames
  quality = compute_vmaf_frames(frames, edited_frames, args.ffmpeg_path)
  row["vmaf"]   = quality["vmaf"]
  row["method"] = quality["method"]
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_lossless_mp4(
    frames: List[np.ndarray],
    path: str,
    ffmpeg_path: str,
    fps: int = 25,
) -> None:
    """Write [H,W,3] uint8 RGB frame list to a lossless H.264 MP4."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), bgr)
        cmd = [
            ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "%05d.png"),
            "-vcodec", "libx264",
            "-crf", "0",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg write failed (exit {result.returncode}): "
                f"{result.stderr[-400:]}"
            )


def _has_libvmaf(ffmpeg_path: str) -> bool:
    """Return True if the ffmpeg binary was compiled with libvmaf support."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-filters"],
            capture_output=True, text=True, timeout=10,
        )
        return "libvmaf" in result.stdout
    except Exception:
        return False


def _parse_vmaf_score(stderr: str) -> Optional[float]:
    """Extract the aggregate VMAF score from ffmpeg stderr."""
    for line in stderr.splitlines():
        # libvmaf ≥ 2.x:  "VMAF score: 92.123456"
        if "VMAF score:" in line:
            try:
                return float(line.split("VMAF score:")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        # libvmaf 3.x:    "VMAF score = 92.123456"
        if "VMAF score =" in line:
            try:
                return float(line.split("=")[-1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        # Some versions: "[libvmaf @ ...] VMAF score: 92.12"
        if "vmaf" in line.lower() and "score" in line.lower():
            parts = line.lower().split("score")
            if len(parts) > 1:
                token = parts[-1].strip().lstrip(":= ").split()[0]
                try:
                    return float(token)
                except ValueError:
                    pass
    return None


def _parse_extra_metrics(stderr: str) -> Dict[str, Optional[float]]:
    """
    Extract PSNR and SSIM from ffmpeg stderr when using
    libvmaf=psnr=1:ssim=1 or separate psnr/ssim filters.
    """
    psnr = ssim = None
    for line in stderr.splitlines():
        lo = line.lower()
        # libvmaf PSNR: "average: 34.21 min: ..."
        if "average:" in lo and ("psnr" in lo or "y_avg" in lo):
            try:
                psnr = float(lo.split("average:")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        # libvmaf SSIM: "All:0.9654" or "ssim_y ... mean:0.9654"
        if "ssim" in lo and ("all:" in lo or "mean:" in lo):
            for sep in ("all:", "mean:"):
                if sep in lo:
                    try:
                        ssim = float(lo.split(sep)[1].strip().split()[0].rstrip(","))
                        break
                    except (IndexError, ValueError):
                        pass
    return {"psnr": psnr, "ssim": ssim}


# ── Public API ────────────────────────────────────────────────────────────────

def compute_vmaf_frames(
    ref_frames: List[np.ndarray],
    dist_frames: List[np.ndarray],
    ffmpeg_path: str = "ffmpeg",
    fps: int = 25,
) -> Dict[str, object]:
    """
    Compute VMAF (and PSNR/SSIM) between reference and distorted frame sequences.

    Tries libvmaf first (full VMAF score).  Falls back to numpy SSIM/PSNR if
    libvmaf is not compiled into the ffmpeg binary.

    Args:
        ref_frames:   original frames, list of [H, W, 3] uint8 RGB
        dist_frames:  edited/distorted frames, same format
        ffmpeg_path:  path to ffmpeg executable
        fps:          frame-rate for intermediate lossless MP4 files

    Returns dict:
        vmaf   (float | None):  VMAF score 0–100; None if libvmaf unavailable
        psnr   (float | None):  mean PSNR in dB
        ssim   (float | None):  mean SSIM 0–1
        method (str):           'libvmaf' | 'fallback_ssim'
    """
    if not ref_frames or not dist_frames:
        return {"vmaf": None, "psnr": None, "ssim": None, "method": "empty"}

    n           = min(len(ref_frames), len(dist_frames))
    ref_frames  = ref_frames[:n]
    dist_frames = dist_frames[:n]

    if _has_libvmaf(ffmpeg_path):
        try:
            return _vmaf_via_libvmaf(ref_frames, dist_frames, ffmpeg_path, fps)
        except Exception as exc:
            print(f"  [vmaf] libvmaf path failed ({exc}); using numpy fallback")

    return _vmaf_fallback(ref_frames, dist_frames)


# ── Internal implementations ──────────────────────────────────────────────────

def _vmaf_via_libvmaf(
    ref_frames: List[np.ndarray],
    dist_frames: List[np.ndarray],
    ffmpeg_path: str,
    fps: int,
) -> Dict[str, object]:
    """Compute VMAF using ffmpeg libvmaf filter (requires libvmaf support)."""
    with tempfile.TemporaryDirectory() as tmp:
        ref_path  = os.path.join(tmp, "ref.mp4")
        dist_path = os.path.join(tmp, "dist.mp4")
        _write_lossless_mp4(ref_frames,  ref_path,  ffmpeg_path, fps)
        _write_lossless_mp4(dist_frames, dist_path, ffmpeg_path, fps)

        # dist is input 0, ref is input 1 (libvmaf convention: [distorted][reference])
        cmd = [
            ffmpeg_path, "-y",
            "-i", dist_path,
            "-i", ref_path,
            "-filter_complex",
            "[0:v][1:v]libvmaf=psnr=1:ssim=1:log_fmt=xml:log_path=/dev/null",
            "-f", "null", "-",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )

        vmaf   = _parse_vmaf_score(result.stderr)
        extras = _parse_extra_metrics(result.stderr)

    return {
        "vmaf":   vmaf,
        "psnr":   extras.get("psnr"),
        "ssim":   extras.get("ssim"),
        "method": "libvmaf",
    }


def _vmaf_fallback(
    ref_frames: List[np.ndarray],
    dist_frames: List[np.ndarray],
) -> Dict[str, object]:
    """Compute PSNR and SSIM with numpy/skimage (no ffmpeg dependency)."""
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []

    for ref, dist in zip(ref_frames, dist_frames):
        mse  = np.mean((ref.astype(float) - dist.astype(float)) ** 2)
        psnr = float("inf") if mse < 1e-10 else 10 * np.log10(255 ** 2 / mse)
        psnr_vals.append(psnr)

        try:
            from skimage.metrics import structural_similarity as _ssim
            ssim_vals.append(float(_ssim(ref, dist, channel_axis=2, data_range=255)))
        except ImportError:
            pass

    finite = [p for p in psnr_vals if p != float("inf")]
    return {
        "vmaf":   None,
        "psnr":   float(np.mean(finite))    if finite    else None,
        "ssim":   float(np.mean(ssim_vals)) if ssim_vals else None,
        "method": "fallback_ssim",
    }
