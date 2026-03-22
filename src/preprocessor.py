"""
Learnable residual preprocessor g_θ and optional DecoyHead.

g_θ: lightweight residual CNN that produces frame-specific adversarial residuals.
     Input:  [B, 3, H, W]  in [0, 1]
     Output: (x_adv, delta)  both in the same shape as input
             x_adv = clamp(x + delta, 0, 1)
             |delta|_inf ≤ max_delta

DecoyHead: small network predicting background perturbation magnitudes for Stage 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class ResidualPreprocessor(nn.Module):
    """
    Stage 1–3 g_θ: input-dependent residual attack network.

    Architecture: entry conv → N residual blocks → exit conv → Tanh scale.
    No downsampling: operates at full input resolution.
    ~250 K parameters with default settings (channels=32, num_blocks=4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        num_blocks: int = 4,
        max_delta: float = 8 / 255,
    ):
        super().__init__()
        self.max_delta = max_delta

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(channels) for _ in range(num_blocks)])
        self.exit = nn.Sequential(
            nn.Conv2d(channels, in_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] float in [0, 1]
        Returns:
            x_adv: [B, 3, H, W] float in [0, 1]
            delta:  [B, 3, H, W] float in [-max_delta, max_delta]
        """
        feat = self.entry(x)
        feat = self.blocks(feat)
        delta = self.exit(feat) * self.max_delta
        x_adv = (x + delta).clamp(0.0, 1.0)
        return x_adv, delta


class DecoyHead(nn.Module):
    """
    Stage 4: predicts background perturbation weight map.

    Takes the same [B, 3, H, W] input as ResidualPreprocessor and outputs
    a spatial weight mask in [0, 1] for the background (decoy) regions.
    The main preprocessor's delta is then amplified in these regions.
    """

    def __init__(self, in_channels: int = 3, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns weight map [B, 1, H, W] in [0, 1]."""
        return self.net(x)


class Stage4Preprocessor(nn.Module):
    """
    Combines ResidualPreprocessor with DecoyHead for Stage 4.

    The decoy mask amplifies perturbations in background regions so that
    SAM2's memory bank competes between the target object and the decoys.
    """

    def __init__(
        self,
        channels: int = 32,
        num_blocks: int = 4,
        max_delta: float = 8 / 255,
        decoy_channels: int = 16,
        decoy_amplify: float = 1.5,
    ):
        super().__init__()
        self.base = ResidualPreprocessor(3, channels, num_blocks, max_delta)
        self.decoy_head = DecoyHead(3, decoy_channels)
        self.decoy_amplify = decoy_amplify

    def forward(self, x: torch.Tensor, gt_mask: torch.Tensor = None):
        """
        Args:
            x:       [B, 3, H, W] in [0, 1]
            gt_mask: [B, 1, H, W] binary GT mask (optional).
                     If given, decoy is restricted to background regions.
        Returns:
            x_adv: perturbed image
            delta: final residual
            decoy_weight: [B, 1, H, W] decoy weight map
        """
        x_adv_base, delta_base = self.base(x)
        decoy_weight = self.decoy_head(x)  # [B,1,H,W] in [0,1]

        if gt_mask is not None:
            bg_mask = (1.0 - gt_mask.float())
            decoy_weight = decoy_weight * bg_mask

        # Amplify delta in decoy regions
        amplify = 1.0 + (self.decoy_amplify - 1.0) * decoy_weight
        delta = delta_base * amplify
        x_adv = (x + delta).clamp(0.0, 1.0)
        return x_adv, delta, decoy_weight
