from __future__ import annotations

import cv2
import numpy as np


_EPS = 1e-6


def binary_mask_u8(mask: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask in {0, 1}."""
    return (np.asarray(mask) > 0).astype(np.uint8)


def ellipse_kernel(radius: int) -> np.ndarray:
    """Elliptical structuring element with a pixel radius."""
    radius = max(int(radius), 0)
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )


def masked_normalized_blur(
    image: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalized convolution using Gaussian weights and a binary validity mask.

    output = G(image * valid) / (G(valid) + eps)
    """
    image_f = image.astype(np.float32, copy=False)
    valid_f = np.asarray(valid_mask, dtype=np.float32)

    if fallback is None:
        if image_f.ndim == 3:
            valid_pixels = image_f[valid_f > 0.5]
            if valid_pixels.size:
                fill = valid_pixels.reshape(-1, image_f.shape[2]).mean(axis=0, dtype=np.float64)
            else:
                fill = image_f.mean(axis=(0, 1), dtype=np.float64)
            fallback = np.broadcast_to(
                np.asarray(fill, dtype=np.float32)[None, None, :],
                image_f.shape,
            ).copy()
        else:
            fill = float(image_f[valid_f > 0.5].mean()) if np.any(valid_f > 0.5) else float(image_f.mean())
            fallback = np.full_like(image_f, fill, dtype=np.float32)
    else:
        fallback = fallback.astype(np.float32, copy=False)

    if sigma <= 0:
        return image_f.copy()

    weight = valid_f if image_f.ndim == 2 else valid_f[:, :, None]
    num = cv2.GaussianBlur(
        image_f * weight,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    den = cv2.GaussianBlur(
        valid_f,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    if image_f.ndim == 3:
        den = den[:, :, None]

    return np.where(den > _EPS, num / np.maximum(den, _EPS), fallback)


def multiband_background_proxy(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    dilation_px: int = 24,
    low_sigma: float | None = None,
    band_sigma_small: float | None = None,
    band_sigma_large: float | None = None,
    mid_gain: float = 0.25,
    guard_px: int = 2,
) -> np.ndarray:
    """
    Multi-band normalized-convolution background proxy.

    Low band:
        masked normalized Gaussian interpolation of RGB background.
    Mid band:
        masked transport of a band-pass residual to avoid flat-color blobs.
    """
    frame_f = frame_rgb.astype(np.float32, copy=False)
    mask_u8 = binary_mask_u8(mask)
    if mask_u8.sum() == 0:
        return frame_rgb.copy()

    bg_pixels = frame_f[mask_u8 == 0]
    if bg_pixels.size:
        bg_fill = bg_pixels.reshape(-1, 3).mean(axis=0, dtype=np.float64).astype(np.float32)
    else:
        bg_fill = frame_f.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    fallback = np.broadcast_to(bg_fill[None, None, :], frame_f.shape).copy()

    bg_valid = (mask_u8 == 0).astype(np.float32)
    if guard_px > 0:
        guarded = cv2.dilate(mask_u8, ellipse_kernel(guard_px))
        guarded_valid = (guarded == 0).astype(np.float32)
        if guarded_valid.sum() >= 32:
            bg_valid = guarded_valid

    low_sigma = float(low_sigma) if low_sigma is not None else max(float(dilation_px) / 3.0, 4.0)
    band_sigma_small = (
        float(band_sigma_small)
        if band_sigma_small is not None
        else max(float(dilation_px) / 10.0, 1.2)
    )
    band_sigma_large = (
        float(band_sigma_large)
        if band_sigma_large is not None
        else max(float(dilation_px) / 3.5, band_sigma_small + 1.5)
    )

    low_proxy = masked_normalized_blur(frame_f, bg_valid, low_sigma, fallback=fallback)

    blur_small = cv2.GaussianBlur(
        frame_f,
        (0, 0),
        sigmaX=band_sigma_small,
        sigmaY=band_sigma_small,
        borderType=cv2.BORDER_REFLECT,
    )
    blur_large = cv2.GaussianBlur(
        frame_f,
        (0, 0),
        sigmaX=band_sigma_large,
        sigmaY=band_sigma_large,
        borderType=cv2.BORDER_REFLECT,
    )
    band_pass = blur_small - blur_large
    band_proxy = masked_normalized_blur(
        band_pass,
        bg_valid,
        band_sigma_small,
        fallback=np.zeros_like(frame_f),
    )

    proxy = np.clip(low_proxy + float(mid_gain) * band_proxy, 0.0, 255.0)
    proxy[bg_valid > 0.5] = frame_f[bg_valid > 0.5]
    return proxy.astype(np.float32)


def normal_transport_proxy(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    max_out: int = 28,
    smooth_sigma: float = 1.5,
    mid_gain: float = 0.20,
    _dist_in: np.ndarray | None = None,
    _dist_out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normal-transport background proxy.

    For each pixel inside the mask, sample background texture by remapping
    along the outward surface normal direction (dist_in + 2 px further out).
    This gives real background texture instead of a blurred mean color.

    Falls back to multiband_background_proxy for pixels whose source
    location still lands inside the mask.

    _dist_in / _dist_out: optional pre-computed distance fields to avoid
    redundant cv2.distanceTransform calls.
    """
    mask_u8 = binary_mask_u8(mask)
    if mask_u8.sum() == 0:
        return frame_rgb.copy()

    if _dist_in is not None and _dist_out is not None:
        dist_in, dist_out = _dist_in, _dist_out
    else:
        dist_in, dist_out = mask_distance_fields(mask_u8)

    # Smooth SDF to get stable normals
    sdf = dist_out.astype(np.float32) - dist_in.astype(np.float32)
    sdf_s = cv2.GaussianBlur(sdf, (0, 0), smooth_sigma, borderType=cv2.BORDER_REFLECT)

    gx = cv2.Sobel(sdf_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf_s, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx * gx + gy * gy) + _EPS
    nx, ny = gx / norm, gy / norm  # outward unit normals

    H, W = mask_u8.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Step each interior pixel outward by (dist_in + 2) px to land in background
    step = np.clip(dist_in + 2.0, 2.0, float(max_out)).astype(np.float32)
    map_x = (xx + nx * step).astype(np.float32)
    map_y = (yy + ny * step).astype(np.float32)

    proxy = cv2.remap(
        frame_rgb.astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )

    # Where remap source still lands inside mask → use multiband fallback
    src_in_mask = cv2.remap(
        mask_u8.astype(np.float32), map_x, map_y,
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    fallback = multiband_background_proxy(
        frame_rgb, mask_u8,
        dilation_px=max_out * 2,
        mid_gain=mid_gain,
    ).astype(np.float32)
    proxy[src_in_mask > 0.5] = fallback[src_in_mask > 0.5]
    return proxy


def mask_distance_fields(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inside/outside Euclidean distance fields in pixels."""
    mask_u8 = binary_mask_u8(mask)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5).astype(np.float32)
    return dist_in, dist_out


def raised_cosine_falloff(distance_px: np.ndarray, width_px: float) -> np.ndarray:
    """Half-cosine taper that is 1 at the boundary and 0 at width_px."""
    width_px = float(width_px)
    if width_px <= 0:
        return np.zeros_like(distance_px, dtype=np.float32)

    ratio = np.clip(distance_px.astype(np.float32) / width_px, 0.0, 1.0)
    weight = 0.5 * (1.0 + np.cos(np.pi * ratio))
    weight[distance_px > width_px] = 0.0
    return weight.astype(np.float32)


def sdf_shell_weights(
    mask: np.ndarray,
    inner_width_px: float,
    outer_width_px: float | None = None,
    inner_alpha: float = 0.8,
    outer_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an SDF shell and distance-domain blend weights.

    Returns:
        weight_map: [H, W] float32 in [0, max(alpha)]
        shell_mask: [H, W] uint8
        dist_in:    [H, W] float32
        dist_out:   [H, W] float32
    """
    mask_u8 = binary_mask_u8(mask)
    outer_width_px = float(inner_width_px if outer_width_px is None else outer_width_px)
    outer_alpha = float(inner_alpha if outer_alpha is None else outer_alpha)
    inner_alpha = float(inner_alpha)

    dist_in, dist_out = mask_distance_fields(mask_u8)
    inside = (mask_u8 > 0).astype(np.float32)
    outside = 1.0 - inside

    w_in = raised_cosine_falloff(dist_in, inner_width_px) * inside
    w_out = raised_cosine_falloff(dist_out, outer_width_px) * outside

    weight = np.clip(
        inner_alpha * w_in + outer_alpha * w_out,
        0.0,
        max(inner_alpha, outer_alpha),
    ).astype(np.float32)
    shell = ((w_in > 0) | (w_out > 0)).astype(np.uint8)
    return weight, shell, dist_in, dist_out


def add_boundary_block_bias(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    max_dist: int = 4,
    amp_y: float = 2.0,
    amp_c: float = 5.0,
    block: int = 8,
) -> np.ndarray:
    """
    Block-aligned low-frequency chroma/luma split at boundary-crossing DCT blocks.

    Fully vectorized (no Python loops) using morphological ops + sliding window
    variance for texture masking.
    """
    mask_u8 = binary_mask_u8(mask)
    if mask_u8.sum() == 0:
        return frame_rgb.copy()

    dist_in, dist_out = mask_distance_fields(mask_u8)
    signed = dist_out.astype(np.float32) - dist_in.astype(np.float32)

    # Support and basis (pixel-level)
    support = (np.abs(signed) <= float(max_dist)).astype(np.float32)
    basis = np.clip(-signed / float(max_dist), -1.0, 1.0) * support

    # Boundary-crossing blocks: both fg AND bg exist within block neighbourhood
    kernel_b = np.ones((block, block), np.uint8)
    block_has_fg = cv2.dilate(mask_u8, kernel_b).astype(np.float32)
    block_has_bg = cv2.dilate(1 - mask_u8, kernel_b).astype(np.float32)
    crossing = (block_has_fg > 0) & (block_has_bg > 0)

    # Texture masking: local variance via box filter (fast, fully vectorized)
    ycc = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_ch = ycc[:, :, 0]
    ksize = (block, block)
    y_mean = cv2.blur(y_ch, ksize)
    y_sq_mean = cv2.blur(y_ch * y_ch, ksize)
    local_var = np.maximum(y_sq_mean - y_mean * y_mean, 0.0)
    tm = np.clip(local_var / 64.0, 0.35, 1.0)

    bias = basis * tm * crossing.astype(np.float32)
    ycc[:, :, 0] = np.clip(ycc[:, :, 0] + amp_y * bias, 0, 255)
    ycc[:, :, 1] = np.clip(ycc[:, :, 1] + amp_c * bias, 0, 255)
    ycc[:, :, 2] = np.clip(ycc[:, :, 2] - amp_c * bias, 0, 255)

    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
