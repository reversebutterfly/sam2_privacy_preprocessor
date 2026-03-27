"""
Codec-aware Expected Over Transformations (EOT) for Stage 3.

Training: differentiable proxy of H.264 compression artifacts.
Evaluation: real FFmpeg H.264 encode → decode pipeline.
"""

import os
import random
import subprocess
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ── Differentiable codec proxy ────────────────────────────────────────────────
#
# H.264 / YUV 4:2:0 artifacts (in order of visual impact):
#   1. Chroma subsampling (yuv420p)  ← new in this version
#   2. Gaussian blur / DCT low-pass
#   3. Additive quantisation noise
#   4. Spatial scale rounding
#

def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """Build a 2D Gaussian kernel tensor [1, 1, K, K]."""
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = max(ksize, 3)
    if ksize % 2 == 0:
        ksize += 1
    x = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2
    g1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = g1d.outer(g1d)
    return kernel.view(1, 1, ksize, ksize)


def differentiable_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply per-channel Gaussian blur (differentiable)."""
    if sigma <= 0:
        return x
    k = _gaussian_kernel(sigma, x.device)
    pad = k.shape[-1] // 2
    B, C, H, W = x.shape
    # Apply per channel
    k3 = k.expand(C, 1, -1, -1)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_padded, k3, groups=C)


def differentiable_resize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Downsample and upsample to simulate spatial quantization artifacts."""
    if abs(scale - 1.0) < 1e-3:
        return x
    B, C, H, W = x.shape
    H2, W2 = max(1, int(H * scale)), max(1, int(W * scale))
    x_down = F.interpolate(x, (H2, W2), mode="bilinear", align_corners=False)
    x_up   = F.interpolate(x_down, (H, W), mode="bilinear", align_corners=False)
    return x_up


def simulate_yuv420p(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable simulation of H.264 YUV 4:2:0 chroma subsampling.

    Converts RGB → YCbCr, halves chroma resolution (2× downsample/upsample),
    then converts back to RGB.  This is the dominant artifact in H.264 because
    the codec always writes yuv420p, even when the source is full-colour.

    Args:
        x: [B, 3, H, W] in [0, 1]
    Returns:
        x_420: [B, 3, H, W] in [0, 1]  (luma unchanged, chroma blurred)
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    # BT.601 RGB → YCbCr
    y  =  0.299  * r + 0.587  * g + 0.114  * b
    cb = -0.16874 * r - 0.33126 * g + 0.500   * b + 0.5
    cr =  0.500   * r - 0.41869 * g - 0.08131 * b + 0.5

    # 4:2:0 — halve chroma resolution then restore
    cb_sub = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr_sub = F.avg_pool2d(cr, kernel_size=2, stride=2)
    B, _, H, W = x.shape
    cb_up = F.interpolate(cb_sub, size=(H, W), mode="bilinear", align_corners=False)
    cr_up = F.interpolate(cr_sub, size=(H, W), mode="bilinear", align_corners=False)

    # BT.601 YCbCr → RGB
    r_out = y + 1.402   * (cr_up - 0.5)
    g_out = y - 0.34414 * (cb_up - 0.5) - 0.71414 * (cr_up - 0.5)
    b_out = y + 1.772   * (cb_up - 0.5)

    return torch.cat([r_out, g_out, b_out], dim=1).clamp(0.0, 1.0)


def codec_proxy_transform(
    x: torch.Tensor,
    blur_sigmas: Tuple[float, ...] = (0.0, 0.5, 1.0),
    noise_stds:  Tuple[float, ...] = (0.0, 0.005, 0.01),
    resize_scales: Tuple[float, ...] = (1.0, 0.90, 0.95),
    p_apply: float = 0.8,
    p_yuv420: float = 0.8,
) -> torch.Tensor:
    """
    Differentiable H.264 proxy: YUV420p chroma subsampling + blur + noise + resize.
    Used during Stage 3 training (EOT).

    Changes vs. original:
    - Added simulate_yuv420p (dominant H.264 artifact, applied first)
    - p_yuv420 controls chroma-subsampling probability

    Args:
        x:             [B, 3, H, W] in [0, 1]
        blur_sigmas:   pool of Gaussian sigma values to sample from
        noise_stds:    pool of noise standard deviations
        resize_scales: pool of downscale factors
        p_apply:       probability of applying blur / noise / resize
        p_yuv420:      probability of applying YUV 4:2:0 chroma subsampling
    Returns:
        x_transformed: [B, 3, H, W] in [0, 1]
    """
    out = x

    # 1. Chroma subsampling (dominant H.264 artifact)
    if random.random() < p_yuv420:
        out = simulate_yuv420p(out)

    # 2. Low-pass blur (DCT quantisation proxy)
    if random.random() < p_apply:
        sigma = random.choice(blur_sigmas)
        if sigma > 0:
            out = differentiable_blur(out, sigma)

    # 3. Quantisation noise
    if random.random() < p_apply:
        std = random.choice(noise_stds)
        if std > 0:
            out = (out + torch.randn_like(out) * std).clamp(0.0, 1.0)

    # 4. Spatial resolution rounding
    if random.random() < p_apply:
        scale = random.choice(resize_scales)
        out = differentiable_resize(out, scale)

    return out.clamp(0.0, 1.0)


# ── Real FFmpeg H.264 encode/decode ───────────────────────────────────────────

def encode_decode_h264(
    frames: List[np.ndarray],
    crf: int = 23,
    fps: int = 25,
    ffmpeg_path: str = r"C:\ffmpeg\bin\ffmpeg.exe",
) -> List[np.ndarray]:
    """
    Encode a list of frames to H.264 with FFmpeg and decode back.
    Non-differentiable; used at evaluation time only.

    Args:
        frames:      list of np.ndarray [H, W, 3] uint8 (RGB)
        crf:         H.264 CRF quality (lower = better; 18/23/28)
        fps:         frames per second for the intermediate video
        ffmpeg_path: path to ffmpeg executable
    Returns:
        decoded frames: list of np.ndarray [H, W, 3] uint8 (RGB)
    """
    if not frames:
        return []

    H, W = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmp:
        # Write frames as PNG sequence
        for i, fr in enumerate(frames):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), bgr)

        in_pattern  = os.path.join(tmp, "%05d.png")
        out_video   = os.path.join(tmp, "encoded.mp4")
        out_pattern = os.path.join(tmp, "decoded_%05d.png")

        # Encode
        cmd_enc = [
            ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", in_pattern,
            "-vcodec", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            out_video,
        ]
        _run_ffmpeg(cmd_enc)

        # Decode
        cmd_dec = [
            ffmpeg_path, "-y",
            "-i", out_video,
            out_pattern,
        ]
        _run_ffmpeg(cmd_dec)

        # Read decoded frames
        decoded = []
        idx = 1  # ffmpeg output starts at 00001.png
        while True:
            path = os.path.join(tmp, f"decoded_{idx:05d}.png")
            if not os.path.exists(path):
                break
            bgr = cv2.imread(path)
            if bgr is None:
                break
            decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx += 1

    return decoded


def _run_ffmpeg(cmd: List[str]) -> None:
    """Run an FFmpeg command, raise RuntimeError on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg failed:\nCMD: {' '.join(cmd)}\nSTDERR: {e.stderr}"
        ) from e
    except FileNotFoundError:
        raise FileNotFoundError(
            f"FFmpeg not found at: {cmd[0]}\n"
            "Update FFMPEG_PATH in config.py or pass --ffmpeg_path."
        )


def tensor_to_frames(x: torch.Tensor) -> List[np.ndarray]:
    """Convert [T, 3, H, W] float [0,1] tensor to list of uint8 numpy arrays."""
    out = []
    for t in range(x.shape[0]):
        frame = (x[t].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out.append(frame)
    return out


def frames_to_tensor(frames: List[np.ndarray], device: str = "cpu") -> torch.Tensor:
    """Convert list of [H, W, 3] uint8 arrays to [T, 3, H, W] float [0,1] tensor."""
    out = []
    for fr in frames:
        t = torch.from_numpy(fr.astype(np.float32) / 255.0).permute(2, 0, 1)
        out.append(t)
    return torch.stack(out).to(device)
