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
