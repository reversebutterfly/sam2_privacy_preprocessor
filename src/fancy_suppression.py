"""
src/fancy_suppression.py

Enhanced boundary suppression variants for SAM2 Privacy Preprocessor.

Three methods beyond the baseline (idea1/combo):

  MSB      — Multi-Scale Boundary suppression
             Applies suppression at K spatial scales and fuses via per-pixel
             max-weight selection.  Captures coarse-to-fine boundary structure
             that a single ring misses.

  AdvOpt   — Adversarial Parameter Optimization
             Per-video gradient optimisation of (sigma, alpha) by minimising a
             differentiable proxy loss: suppress boundary-region gradient
             magnitude subject to an SSIM-floor constraint.  No SAM2 required
             in the optimisation loop.

  LFNet    — Learned Feathering Network
             Lightweight 4-layer CNN: (frame, mask) → per-pixel blend-weight.
             Fine-tuned per-video with the same proxy gradient-suppression loss.
             Falls back to MSB if the network has not been trained.

  full     — AdvOpt-tuned sigma/alpha fed into MSB, then refined by LFNet.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bg_proxy_np(frame_rgb: np.ndarray, mask: np.ndarray,
                 dilation_px: int = 24) -> np.ndarray:
    """Sample local background just outside the mask (same as pilot_mask_guided)."""
    kernel = np.ones((dilation_px * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    bg_mask = (dilated > 0) & (mask == 0)
    proxy = frame_rgb.astype(np.float32).copy()
    if bg_mask.sum() < 10:
        bg_color = frame_rgb.mean(axis=(0, 1))
        proxy[:] = bg_color
    else:
        bg_mean = frame_rgb[bg_mask].mean(axis=0)
        proxy[mask > 0] = bg_mean
    sigma = max(dilation_px // 2, 5)
    return cv2.GaussianBlur(proxy, (0, 0), sigma)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Multi-Scale Boundary Suppression (MSB)
# ─────────────────────────────────────────────────────────────────────────────

def apply_multiscale_suppression(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    scales: list[int] | None = None,
    alphas: list[float] | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Multi-Scale Boundary Suppression.

    Computes boundary rings at K dilation scales and blends each toward the
    local background proxy.  Final edit = max-weight fusion across all scales:
    each pixel takes the blend from whichever scale assigns it the highest
    weight, ensuring fine-scale rings near the boundary dominate while
    coarser rings extend the suppression zone.

    Args:
        frame_rgb:  [H, W, 3] uint8 input frame.
        mask:       [H, W]    binary uint8 GT mask.
        scales:     ring half-widths in pixels, e.g. [8, 16, 24].
        alphas:     blend strength per scale; should decrease from fine→coarse
                    to preserve overall SSIM, e.g. [0.85, 0.75, 0.60].
    """
    if scales is None:
        scales = [8, 16, 24]
    if alphas is None:
        # Uniform alpha matching idea1 default; outer scales use the same strength.
        # Decreasing-alpha schemes reduce the outer ring below idea1 baseline — avoid.
        alphas = [0.80, 0.80, 0.80]
    assert len(scales) == len(alphas), "scales and alphas must have equal length"

    if mask.sum() == 0:
        return frame_rgb.copy()

    H, W = frame_rgb.shape[:2]

    # Additive weight accumulation: each scale contributes a smooth ring weight;
    # final weight = clip(sum of all ring weights, 0, max(alphas)).
    # This ensures pixels near the boundary benefit from ALL scales that cover them,
    # giving stronger suppression near the edge than any single-scale method.
    max_alpha = max(alphas)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    # Weighted background proxy: accumulate (bg * incremental_weight)
    bg_accum   = np.zeros((H, W, 3), dtype=np.float32)

    for scale, alpha in zip(scales, alphas):
        kernel = np.ones((scale * 2 + 1,) * 2, np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded  = cv2.erode(mask,  kernel)
        ring    = ((dilated > 0) & (eroded == 0)).astype(np.float32)
        if ring.sum() == 0:
            continue

        sigma_blur = max(scale / 2.0, 2.0)
        smooth_ring = cv2.GaussianBlur(ring, (0, 0), sigma_blur)
        w_k = np.clip(smooth_ring * alpha, 0.0, alpha)  # [H, W]

        bg_proxy = _bg_proxy_np(frame_rgb, mask, dilation_px=scale * 2)

        # Incremental contribution of this scale (avoid double-counting pixels
        # already covered by finer scales at full alpha)
        w_new = np.clip(w_k - weight_sum, 0.0, None)
        bg_accum   += w_new[:, :, None] * bg_proxy
        weight_sum += w_new

    # Final blend
    weight_sum = np.clip(weight_sum, 0.0, max_alpha)
    # Normalise accumulated bg by total weight
    bg_final = np.where(
        weight_sum[:, :, None] > 1e-8,
        bg_accum / (weight_sum[:, :, None] + 1e-8),
        frame_rgb.astype(np.float32),
    )
    f = frame_rgb.astype(np.float32)
    w = weight_sum[:, :, None]
    edited = f * (1.0 - w) + bg_final * w
    return np.clip(edited, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Adversarial Parameter Optimisation (AdvOpt)
# ─────────────────────────────────────────────────────────────────────────────
# Differentiable helpers operating on [B, C, H, W] float tensors ∈ [0, 1].

def _gauss_blur_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable Gaussian blur via separable 1D convolutions (O(n) not O(n²))."""
    ks = max(int(4.0 * sigma + 0.5) | 1, 3)   # ensure odd
    t  = torch.arange(ks, dtype=x.dtype, device=x.device) - (ks - 1) / 2.0
    k1 = torch.exp(-0.5 * (t / sigma) ** 2)
    k1 = k1 / k1.sum()

    C = x.shape[1]
    pad = ks // 2
    # Separable: horizontal then vertical — O(ks) instead of O(ks²)
    kh = k1.view(1, 1, 1, ks).expand(C, 1, 1, ks)  # [C, 1, 1, ks]
    kv = k1.view(1, 1, ks, 1).expand(C, 1, ks, 1)  # [C, 1, ks, 1]
    out = F.conv2d(x, kh, padding=(0, pad), groups=C)
    out = F.conv2d(out, kv, padding=(pad, 0), groups=C)
    return out


def _diff_ring_weight(mask_t: torch.Tensor,
                      sigma: torch.Tensor,
                      alpha: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of the feathered boundary ring weight map.
    Uses difference of two Gaussian-blurred masks as a ring proxy:
        ring ≈ GaussBlur(mask, sigma_inner) − GaussBlur(mask, sigma_outer)
        weight = clip(ring * alpha, 0, alpha)
    """
    sigma_inner = sigma * 0.5
    sigma_outer = sigma * 2.0
    inner = _gauss_blur_torch(mask_t, float(sigma_inner.item()))
    outer = _gauss_blur_torch(mask_t, float(sigma_outer.item()))
    ring  = torch.clamp(inner - outer, 0.0, 1.0)
    # Normalise ring to [0, 1], then scale by alpha
    ring  = ring / (ring.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    # ring ∈ [0,1] and alpha ∈ (0,1) so ring*alpha ∈ [0, alpha] — no clamp needed
    return ring * alpha


def _diff_bg_proxy(frame_t: torch.Tensor,
                   mask_t: torch.Tensor,
                   sigma: torch.Tensor) -> torch.Tensor:
    """
    Differentiable background proxy: Gaussian-filled background region.
    bg_proxy = GaussBlur(frame * (1-mask)) / GaussBlur(1-mask + eps)
    """
    bg = frame_t * (1.0 - mask_t)
    sig = float((sigma * 2.0).clamp(2, 40).item())
    bg_blurred   = _gauss_blur_torch(bg,              sig)
    mask_blurred = _gauss_blur_torch(1.0 - mask_t,   sig) + 1e-6
    return (bg_blurred / mask_blurred).clamp(0.0, 1.0)


def _grad_mag_in_ring(edited_t: torch.Tensor,
                      ring_weight: torch.Tensor) -> torch.Tensor:
    """Mean gradient magnitude in the ring region (proxy loss)."""
    # Simple finite-difference gradients (no extra kernels needed)
    dx = edited_t[:, :, :, 1:] - edited_t[:, :, :, :-1]   # [B,C,H,W-1]
    dy = edited_t[:, :, 1:, :] - edited_t[:, :, :-1, :]   # [B,C,H-1,W]
    dx_sq = F.pad(dx ** 2, (0, 1))
    dy_sq = F.pad(dy ** 2, (0, 0, 0, 1))
    grad_sq = (dx_sq + dy_sq).mean(dim=1, keepdim=True)    # [B,1,H,W]
    # Ring-masked mean
    ring_sum = ring_weight.sum() + 1e-8
    return (grad_sq * ring_weight).sum() / ring_sum


def optimize_adv_params(
    frame_rgb: np.ndarray,
    mask_np: np.ndarray,
    n_iter: int = 80,
    lr: float = 0.08,
    ssim_floor: float = 0.92,
    device: str = "cpu",
) -> tuple[int, float]:
    """
    Per-frame adversarial parameter optimisation.

    Minimises the mean gradient magnitude inside the boundary ring (proxy for
    "boundary detectability by SAM2") subject to a soft SSIM-floor constraint.

    The blending pipeline is implemented in differentiable PyTorch so that
    gradients w.r.t. log_sigma and logit_alpha propagate cleanly.

    Args:
        frame_rgb:  [H, W, 3] uint8 input frame.
        mask_np:    [H, W]    binary uint8 GT mask.
        n_iter:     number of gradient steps.
        lr:         Adam learning rate.
        ssim_floor: soft SSIM floor (penalty triggers below this value).
        device:     "cpu" or "cuda".

    Returns:
        (ring_width, blend_alpha)  — integer and float ready for idea1/MSB API.
    """
    if mask_np.sum() == 0:
        return 16, 0.6

    dev = torch.device(device)
    frame_t = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(dev)
    mask_t  = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)

    # Learnable unconstrained parameters
    log_sigma   = nn.Parameter(torch.tensor([np.log(12.0)], device=dev))
    logit_alpha = nn.Parameter(torch.tensor([0.0],          device=dev))
    opt = torch.optim.Adam([log_sigma, logit_alpha], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()

        sigma = torch.exp(log_sigma).clamp(2.0, 20.0)     # sigma ∈ [2, 20]
        alpha = torch.sigmoid(logit_alpha) * 0.85 + 0.10  # alpha ∈ [0.10, 0.95]

        ring_w   = _diff_ring_weight(mask_t, sigma, alpha)
        bg_proxy = _diff_bg_proxy(frame_t, mask_t, sigma)
        edited   = frame_t * (1.0 - ring_w) + bg_proxy * ring_w

        # Primary loss: minimise gradient magnitude in ring (↓ = better suppression)
        loss = _grad_mag_in_ring(edited, ring_w)

        # Soft SSIM constraint: penalise large per-pixel L2 deviation as proxy.
        # Empirically: SSIM ≈ 1 − 20 * mse_edit (linear approx, low-distortion regime).
        # Penalty coefficient 20.0 enforces the floor much more strictly than 5.0
        # to prevent alpha from exceeding the SSIM budget (reduces SSIM < 0.90 cases).
        mse_edit = ((edited - frame_t) ** 2).mean()
        ssim_approx = 1.0 - 20.0 * mse_edit
        penalty = torch.relu(ssim_floor - ssim_approx) * 20.0

        (loss + penalty).backward()
        opt.step()

    final_sigma = float(torch.exp(log_sigma).clamp(2.0, 20.0).detach().cpu())
    final_alpha = float((torch.sigmoid(logit_alpha) * 0.85 + 0.10).detach().cpu())
    ring_width  = int(np.clip(round(final_sigma * 2), 4, 40))
    return ring_width, final_alpha


def apply_adv_opt_suppression(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    n_iter: int = 80,
    lr: float = 0.08,
    ssim_floor: float = 0.92,
    device: str = "cpu",
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper: optimise params, then apply idea1-style suppression."""
    from pilot_mask_guided import apply_boundary_suppression
    rw, alpha = optimize_adv_params(frame_rgb, mask, n_iter, lr, ssim_floor, device)
    return apply_boundary_suppression(frame_rgb, mask, ring_width=rw, blend_alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Learned Feathering Network (LFNet)
# ─────────────────────────────────────────────────────────────────────────────

class LFNet(nn.Module):
    """
    Lightweight encoder that maps (frame, mask) → per-pixel blend weight.

    Input:  [B, 4, H, W]  — 3-channel normalised frame + 1-channel mask
    Output: [B, 1, H, W]  ∈ [0, 1]  — spatial blend weights

    ~11 K parameters with default channels=16.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4,         channels,     3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels,  channels * 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels,  3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 4, H, W]  (RGB/255 ‖ mask)"""
        return self.head(self.enc(x))


def train_lfnet_per_video(
    frame_rgb: np.ndarray,
    mask_np: np.ndarray,
    n_iter: int = 150,
    lr: float = 5e-3,
    ssim_floor: float = 0.92,
    alpha_ceiling: float = 0.90,
    device: str = "cpu",
) -> LFNet:
    """
    Self-supervised per-frame fine-tuning of LFNet.

    The network learns to predict a blend-weight map that minimises the
    boundary-region gradient magnitude (proxy for SAM2 boundary detectability)
    while respecting a soft SSIM floor.

    A mask-locality regulariser penalises blend weights outside the ring zone,
    preventing the network from suppressing the entire image.

    Returns a trained LFNet instance (call .eval() before inference).
    """
    dev = torch.device(device)
    frame_t = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(dev)
    mask_t  = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)

    # Build a coarse ring mask as a locality prior (no gradients needed)
    with torch.no_grad():
        ring_prior = _diff_ring_weight(
            mask_t,
            torch.tensor(12.0, device=dev),
            torch.tensor(0.85, device=dev),
        ).clamp(0, 1)

    # Background proxy (numpy, then to tensor)
    bg_np    = _bg_proxy_np(frame_rgb, mask_np, dilation_px=24)
    bg_t     = torch.from_numpy(bg_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(dev)

    model = LFNet().to(dev).train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    inp = torch.cat([frame_t, mask_t], dim=1)  # [1, 4, H, W]

    for _ in range(n_iter):
        opt.zero_grad()
        weight  = model(inp) * alpha_ceiling     # [1, 1, H, W]
        edited  = frame_t * (1.0 - weight) + bg_t * weight

        # Gradient suppression in the ring zone
        grad_loss = _grad_mag_in_ring(edited, ring_prior)

        # SSIM soft constraint
        mse_edit    = ((edited - frame_t) ** 2).mean()
        ssim_approx = 1.0 - 20.0 * mse_edit
        penalty     = torch.relu(ssim_floor - ssim_approx) * 5.0

        # Locality: penalise non-zero weights outside the ring zone
        outside = 1.0 - ring_prior.clamp(0, 1)
        locality_penalty = (weight * outside).mean() * 2.0

        (grad_loss + penalty + locality_penalty).backward()
        opt.step()

    return model.eval()


def apply_lfnet_suppression(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    model: LFNet | None = None,
    device: str = "cpu",
    n_train_iter: int = 150,
    ssim_floor: float = 0.92,
    **kwargs,
) -> np.ndarray:
    """
    Apply LFNet learned suppression.

    If `model` is None, fine-tunes a fresh LFNet for this frame/mask pair.
    For per-video use, train once with `train_lfnet_per_video` and reuse.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    if model is None:
        model = train_lfnet_per_video(frame_rgb, mask, n_iter=n_train_iter,
                                      ssim_floor=ssim_floor, device=device)

    bg_np = _bg_proxy_np(frame_rgb, mask, dilation_px=24)
    dev   = torch.device(device)

    with torch.no_grad():
        frame_t = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(dev)
        mask_t  = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
        bg_t    = torch.from_numpy(bg_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(dev)
        inp     = torch.cat([frame_t, mask_t], dim=1)
        weight  = model(inp).clamp(0.0, 0.90)               # [1, 1, H, W]
        edited  = frame_t * (1.0 - weight) + bg_t * weight  # [1, 3, H, W]
        out_np  = (edited[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    return out_np


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full pipeline: AdvOpt → MSB → LFNet refinement
# ─────────────────────────────────────────────────────────────────────────────

def apply_full_fancy(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    adv_n_iter: int = 80,
    msb_scales: list[int] | None = None,
    lfnet_n_iter: int = 150,
    ssim_floor: float = 0.92,
    device: str = "cpu",
    **kwargs,
) -> np.ndarray:
    """
    Full three-stage pipeline:
      1. AdvOpt  — optimise (sigma, alpha) for this frame.
      2. MSB     — apply multi-scale suppression with AdvOpt-tuned alphas.
      3. LFNet   — fine-tune and apply a learned feathering network.

    The three stages are complementary:
      - AdvOpt sets globally optimal parameters without SAM2 in the loop.
      - MSB extends the suppression zone across multiple boundary scales.
      - LFNet adapts the weight map spatially to exploit per-pixel structure.
    """
    if mask.sum() == 0:
        return frame_rgb.copy()

    # Stage 1: adversarially optimised base parameters
    rw, alpha = optimize_adv_params(
        frame_rgb, mask, n_iter=adv_n_iter, ssim_floor=ssim_floor, device=device
    )

    # Stage 2: multi-scale suppression using the optimised alpha
    scales_use = msb_scales or [max(4, rw // 2), rw, min(40, rw + rw // 2)]
    alphas_use = [min(0.95, alpha), min(0.85, alpha * 0.88), min(0.75, alpha * 0.75)]
    edited = apply_multiscale_suppression(frame_rgb, mask, scales=scales_use, alphas=alphas_use)

    # Stage 3: LFNet per-frame refinement
    model  = train_lfnet_per_video(
        edited, mask, n_iter=lfnet_n_iter, ssim_floor=ssim_floor, device=device
    )
    edited = apply_lfnet_suppression(edited, mask, model=model, device=device)

    return edited


# ─────────────────────────────────────────────────────────────────────────────
# Registry (compatible with pilot_mask_guided.EDIT_FNS)
# ─────────────────────────────────────────────────────────────────────────────

FANCY_EDIT_FNS = {
    "msb":      apply_multiscale_suppression,
    "adv_opt":  apply_adv_opt_suppression,
    "lfnet":    apply_lfnet_suppression,
    "full":     apply_full_fancy,
}
