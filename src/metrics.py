"""
Evaluation metrics for SAM2 Privacy Preprocessor.

J&F  — standard VOS metric (region similarity + boundary accuracy).
SSIM — structural similarity (quality constraint check).
LPIPS — perceptual distance (optional, requires lpips package).
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.metrics import structural_similarity as _ssim


# ── J (Jaccard / region similarity) ──────────────────────────────────────────

def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection-over-union between binary masks.

    Args:
        pred, gt: binary np.ndarray (bool or 0/1 int)
    Returns:
        IoU in [0, 1]
    """
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    inter  = np.logical_and(pred_b, gt_b).sum()
    union  = np.logical_or(pred_b, gt_b).sum()
    return float(inter) / max(float(union), 1e-9)


# ── F (boundary accuracy) ─────────────────────────────────────────────────────

def _boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Extract binary boundary with morphological dilation."""
    h, w = mask.shape[-2:]
    pixels = max(1, int(round(dilation_ratio * max(h, w))))
    struct = np.ones((pixels, pixels), dtype=bool)
    eroded   = binary_erosion(mask, struct)
    boundary = np.logical_xor(mask, eroded)
    return boundary


def f_measure(pred: np.ndarray, gt: np.ndarray, bound_thresh: float = 0.02) -> float:
    """
    Boundary F-measure (precision / recall on dilated boundaries).
    """
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    gt_boundary   = _boundary(gt_b,   bound_thresh)
    pred_boundary = _boundary(pred_b, bound_thresh)

    # Precision: how many pred boundary pixels are near GT boundary
    tp_pred = np.logical_and(pred_boundary, binary_dilation(gt_boundary)).sum()
    prec    = float(tp_pred) / max(pred_boundary.sum(), 1e-9)

    # Recall: how many GT boundary pixels are near pred boundary
    tp_gt = np.logical_and(gt_boundary, binary_dilation(pred_boundary)).sum()
    rec   = float(tp_gt) / max(gt_boundary.sum(), 1e-9)

    if prec + rec < 1e-9:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ── J&F combined ─────────────────────────────────────────────────────────────

def jf_score(
    pred: np.ndarray,
    gt:   np.ndarray,
    bound_thresh: float = 0.02,
) -> Tuple[float, float, float]:
    """
    Compute J&F score.

    Returns:
        (jf, j, f)  all in [0, 1]
    """
    j = jaccard(pred, gt)
    f = f_measure(pred, gt, bound_thresh)
    return (j + f) / 2.0, j, f


def mean_jf(
    pred_sequence: list,
    gt_sequence:   list,
) -> Tuple[float, float, float]:
    """
    Average J&F over a sequence of frame masks.

    Args:
        pred_sequence: list of np.ndarray binary masks
        gt_sequence:   list of np.ndarray binary masks (same length)
    Returns:
        (mean_jf, mean_j, mean_f)
    """
    scores = [jf_score(p, g) for p, g in zip(pred_sequence, gt_sequence)]
    jf_vals = [s[0] for s in scores]
    j_vals  = [s[1] for s in scores]
    f_vals  = [s[2] for s in scores]
    return float(np.mean(jf_vals)), float(np.mean(j_vals)), float(np.mean(f_vals))


def jf_curve(
    pred_sequence: list,
    gt_sequence:   list,
    frame_indices: Optional[list] = None,
) -> dict:
    """
    Per-frame J&F curve (useful for B4 long-term tracking analysis).

    Returns:
        dict with keys 'frames', 'jf', 'j', 'f'
    """
    if frame_indices is None:
        frame_indices = list(range(len(pred_sequence)))
    scores = [jf_score(p, g) for p, g in zip(pred_sequence, gt_sequence)]
    return {
        "frames": frame_indices,
        "jf": [s[0] for s in scores],
        "j":  [s[1] for s in scores],
        "f":  [s[2] for s in scores],
    }


# ── Perceptual quality metrics ────────────────────────────────────────────────

def compute_ssim(orig: np.ndarray, adv: np.ndarray) -> float:
    """
    SSIM between two [H, W, 3] uint8 images.
    Higher is better; target ≥ 0.95.
    """
    return float(_ssim(orig, adv, channel_axis=2, data_range=255))


def compute_psnr(orig: np.ndarray, adv: np.ndarray) -> float:
    """PSNR in dB."""
    mse = np.mean((orig.astype(np.float32) - adv.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_lpips_np(orig: np.ndarray, adv: np.ndarray) -> Optional[float]:
    """
    LPIPS between two [H, W, 3] uint8 images (requires lpips + torch).
    Returns None if lpips is not installed.
    """
    try:
        import lpips
        import torch
        _fn = getattr(compute_lpips_np, "_fn", None)
        if _fn is None:
            compute_lpips_np._fn = lpips.LPIPS(net="alex")
            compute_lpips_np._fn.eval()
            _fn = compute_lpips_np._fn

        def _to_t(img):
            t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            return float(_fn(_to_t(orig), _to_t(adv)).item())
    except ImportError:
        return None


def quality_summary(
    orig_frames: list,
    adv_frames:  list,
) -> dict:
    """
    Aggregate quality metrics over a sequence of frame pairs.

    Args:
        orig_frames, adv_frames: lists of [H, W, 3] uint8 np.ndarray
    Returns:
        dict with mean_ssim, mean_psnr, mean_lpips (None if unavailable)
    """
    ssims, psnrs, lpips_vals = [], [], []
    for o, a in zip(orig_frames, adv_frames):
        ssims.append(compute_ssim(o, a))
        psnrs.append(compute_psnr(o, a))
        lp = compute_lpips_np(o, a)
        if lp is not None:
            lpips_vals.append(lp)
    return {
        "mean_ssim":  float(np.mean(ssims)),
        "mean_psnr":  float(np.mean(psnrs)),
        "mean_lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
    }
