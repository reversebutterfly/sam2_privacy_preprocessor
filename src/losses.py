"""
Loss functions for SAM2 Privacy Preprocessor training.

L_total = L_attack + λ1·L_perceptual + λ2·L_temporal + λ3·L_decoy + λ4·L_ssim

All losses return scalar tensors (differentiable).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Attack loss ───────────────────────────────────────────────────────────────

def soft_iou_loss(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Soft IoU between sigmoid(pred_logits) and binary gt_masks.
    Minimizing this = maximizing SAM2 confusion on the target object.

    Args:
        pred_logits: [B, 1, H, W]  raw SAM2 mask logits
        gt_masks:    [B, 1, H, W]  binary GT masks (float 0/1)
    Returns:
        scalar loss in [0, 1]  (minimize to attack)
    """
    pred_prob = torch.sigmoid(pred_logits)
    gt = gt_masks.float()
    inter = (pred_prob * gt).sum(dim=(1, 2, 3))
    union = (pred_prob + gt - pred_prob * gt).sum(dim=(1, 2, 3)) + 1e-6
    return (inter / union).mean()


def bce_attack_loss(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Masked BCE attack loss (alternative to soft-IoU, closer to UAP-SAM2 style).
    Pushes predicted probability to 0 on target object regions.
    """
    gt = gt_masks.float()
    # Target: predict "no object" (0) everywhere
    target = torch.zeros_like(pred_logits)
    loss = F.binary_cross_entropy_with_logits(pred_logits * gt, target * gt, reduction="none")
    return (loss * gt).sum() / (gt.sum() + 1e-6)


# ── Perceptual loss ───────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    LPIPS-based perceptual hinge loss: max(0, LPIPS(x, x_adv) - threshold).

    Ensures the attack stays visually imperceptible.
    """

    def __init__(self, net: str = "alex", threshold: float = 0.10, device: str = "cuda"):
        super().__init__()
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
            self.available = True
        except ImportError:
            print("[PerceptualLoss] lpips not installed, using L2 fallback")
            self.lpips_fn = None
            self.available = False
        self.threshold = threshold

    def forward(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_orig, x_adv: [B, 3, H, W] in [0, 1]
        Returns:
            scalar hinge loss
        """
        if self.available:
            # LPIPS expects [-1, 1]
            a = x_orig * 2.0 - 1.0
            b = x_adv  * 2.0 - 1.0
            lp = self.lpips_fn(a, b)
            return F.relu(lp - self.threshold).mean()
        else:
            # L2 fallback: hinge on MSE
            mse = F.mse_loss(x_adv, x_orig, reduction="none").mean(dim=(1, 2, 3))
            return F.relu(mse - (self.threshold ** 2)).mean()

    @torch.no_grad()
    def measure(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        """Pure LPIPS value (no hinge), for logging."""
        if self.available:
            a = x_orig * 2.0 - 1.0
            b = x_adv  * 2.0 - 1.0
            return self.lpips_fn(a, b).mean().item()
        return F.mse_loss(x_adv, x_orig).item() ** 0.5


# ── SSIM constraint ───────────────────────────────────────────────────────────

def compute_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Differentiable SSIM between two [B, C, H, W] tensors in [0, 1].
    Returns mean SSIM scalar (higher = more similar, max 1.0).
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5
    B, C, H, W = x.shape

    # Build Gaussian window
    k = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    window_1d = torch.exp(-k ** 2 / (2 * sigma ** 2))
    window_1d = window_1d / window_1d.sum()
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)  # [W, W]
    window = window_2d.view(1, 1, window_size, window_size).expand(C, 1, -1, -1)

    pad = window_size // 2

    def blur(t):
        return F.conv2d(F.pad(t, [pad] * 4, mode="reflect"), window, groups=C)

    mu_x = blur(x)
    mu_y = blur(y)
    mu_x2  = mu_x * mu_x
    mu_y2  = mu_y * mu_y
    mu_xy  = mu_x * mu_y
    sig_x2 = blur(x * x) - mu_x2
    sig_y2 = blur(y * y) - mu_y2
    sig_xy = blur(x * y) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()


class SSIMConstraint(nn.Module):
    """
    SSIM hinge loss: max(0, ssim_loss - threshold) where ssim_loss = 1 - SSIM.

    Enforces SSIM(x_orig, x_adv) >= 1 - threshold (default: SSIM >= 0.90).
    Complements the LPIPS hinge to enforce structural similarity.
    """

    def __init__(self, threshold: float = 0.10):
        """
        Args:
            threshold: max allowed (1 - SSIM), i.e. SSIM >= 1 - threshold.
                       0.05 → SSIM ≥ 0.95 (tight); 0.10 → SSIM ≥ 0.90 (moderate).
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_orig, x_adv: [B, 3, H, W] in [0, 1]
        Returns:
            scalar hinge loss
        """
        ssim_val  = compute_ssim(x_orig, x_adv)
        ssim_loss = 1.0 - ssim_val          # lower is better; we want this ≤ threshold
        return F.relu(ssim_loss - self.threshold)

    @torch.no_grad()
    def measure(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        """Returns raw SSIM value (no hinge), for logging."""
        return compute_ssim(x_orig, x_adv).item()


# ── Temporal consistency loss ─────────────────────────────────────────────────

def temporal_loss(deltas: List[torch.Tensor]) -> torch.Tensor:
    """
    L2 norm of consecutive delta differences.
    Encourages smooth, non-flickering perturbations over time.

    Args:
        deltas: list of [B, 3, H, W] residual tensors (ordered by frame index)
    Returns:
        scalar loss
    """
    if len(deltas) < 2:
        return torch.tensor(0.0, device=deltas[0].device if deltas else "cpu")
    diffs = [F.mse_loss(deltas[i + 1], deltas[i]) for i in range(len(deltas) - 1)]
    return torch.stack(diffs).mean()


# ── Decoy loss (Stage 4) ──────────────────────────────────────────────────────

def decoy_loss(
    sam_fwder,
    x_adv: torch.Tensor,
    bg_point_coords: torch.Tensor,
    bg_point_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Maximize SAM2 confidence on background (decoy) regions.

    Injects a "foreground-like" signal into background, competing for
    SAM2 memory slots with the actual target object.

    Args:
        sam_fwder:         SamForwarder instance (frozen)
        x_adv:             [1, 3, H, W] adversarial image in [0, 1]
        bg_point_coords:   [1, K, 2] background prompt coords (pixel space)
        bg_point_labels:   [1, K]    all-ones (foreground labels)
    Returns:
        scalar loss (minimize → background looks like foreground to SAM2)
    """
    logits = sam_fwder.forward(x_adv, bg_point_coords, bg_point_labels, None, None)
    # We want high confidence in the background regions → maximize sigmoid(logits)
    # so we minimize the negative mean
    return -torch.sigmoid(logits).mean()


def sample_background_point(
    gt_mask: torch.Tensor,
    n_points: int = 1,
) -> Optional[torch.Tensor]:
    """
    Sample n random background points (complement of GT mask).

    Args:
        gt_mask: [1, 1, H, W] binary mask (on CPU is fine)
        n_points: number of points to sample
    Returns:
        coords: [1, n_points, 2] float tensor (x, y pixel coordinates)
                or None if background is empty
    """
    bg = (gt_mask[0, 0] == 0).nonzero(as_tuple=False)  # [N, 2] (row, col)
    if bg.shape[0] < n_points:
        return None
    idx = torch.randperm(bg.shape[0])[:n_points]
    pts = bg[idx]  # [n_points, 2] in (row, col)
    # SAM2 expects (x, y) = (col, row)
    coords = pts[:, [1, 0]].float().unsqueeze(0)  # [1, n_points, 2]
    return coords
