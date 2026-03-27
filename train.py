"""
SAM2 Privacy Preprocessor - Main Training Script
Supports B0 (sanity), B1 (UAP baseline), B2 (Stage 1 ours).

Modes:
  --mode ours  : Learnable residual preprocessor (stages 1-4)
  --mode uap   : Universal adversarial perturbation baseline

Usage:
  # B0 Sanity
  python train.py --stage 1 --videos bear --num_steps 500 --sanity

  # B1 Baseline UAP
  python train.py --mode uap --videos bear,breakdance,car-shadow,dance-jump,dog --num_steps 2000

  # B2 Stage 1
  python train.py --mode ours --stage 1 --num_steps 3000
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    SAM2_CHECKPOINT, SAM2_CONFIG,
    DAVIS_ROOT, DAVIS_MINI_TRAIN, DAVIS_MINI_VAL,
    DAVIS_TRAIN_VIDEOS_ALL, RESULTS_DIR,
)
from src.preprocessor import ResidualPreprocessor, Stage4Preprocessor
from src.losses import PerceptualLoss, SSIMConstraint, soft_iou_loss, temporal_loss
from src.codec_eot import codec_proxy_transform
from src.metrics import jf_score
from src.dataset import load_single_video


class SAM2Attacker(nn.Module):
    """Frozen SAM2 wrapper. Only g_theta gradients flow back."""

    SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    SAM2_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    INPUT_SIZE = 1024

    def __init__(self, checkpoint: str, config_name: str, device: torch.device):
        super().__init__()
        self.device = device
        self._orig_hw: Optional[Tuple[int, int]] = None
        from sam2.build_sam import build_sam2
        sam2_model = build_sam2(config_name, checkpoint, device=device)
        sam2_model.eval()
        for p in sam2_model.parameters():
            p.requires_grad_(False)
        self.sam2 = sam2_model
        self.mean = self.SAM2_MEAN.to(device)
        self.std  = self.SAM2_STD.to(device)

    def encode_image(self, img_np: np.ndarray) -> torch.Tensor:
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        H, W = img_np.shape[:2]
        self._orig_hw = (H, W)
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = torch.nn.functional.interpolate(
            x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        return x  # [1,3,1024,1024] in [0,1]

    # Backbone output spatial sizes — matches SAM2ImagePredictor._bb_feat_sizes
    # vision_feats from _prepare_backbone_features are in [HW, B, C] sequence format
    # and must be reshaped to [B, C, H, W] before passing to the mask decoder.
    _BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]

    def forward(
        self,
        x01: torch.Tensor,
        point_coords_np: np.ndarray,
        point_labels_np: np.ndarray,
    ) -> torch.Tensor:
        x_norm = (x01 - self.mean) / self.std
        backbone_out = self.sam2.forward_image(x_norm)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)

        # Optionally add no-memory embedding (matches set_image in SAM2ImagePredictor)
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed

        # Reshape [HW, B, C] → [B, C, H, W] for each scale level
        # vision_feats is ordered large→small scale; _BB_FEAT_SIZES is also large→small
        # The predictor zips reversed lists then reverses back to preserve order
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._BB_FEAT_SIZES[::-1])
        ][::-1]
        # feats[-1]: image embed  [1, C, 64, 64]
        # feats[:-1]: high-res feats [[1,C,256,256], [1,C,128,128]]
        image_embed    = feats[-1]
        high_res_feats = feats[:-1]

        if self._orig_hw is not None:
            H, W = self._orig_hw
            sx, sy = self.INPUT_SIZE / W, self.INPUT_SIZE / H
        else:
            sx, sy = 1.0, 1.0

        pts = torch.from_numpy(point_coords_np).float().to(self.device)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        pts  = pts.unsqueeze(0)
        lbls = torch.from_numpy(point_labels_np).int().to(self.device).unsqueeze(0)

        sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
            points=(pts, lbls), boxes=None, masks=None,
        )

        try:
            low_res_masks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
        except TypeError:
            # Fallback for older SAM2 builds without high_res_features arg
            low_res_masks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
            )

        if self._orig_hw is not None:
            logits = torch.nn.functional.interpolate(
                low_res_masks, size=self._orig_hw, mode="bilinear", align_corners=False,
            )
        else:
            logits = low_res_masks
        return logits  # [1,1,H,W]

    def forward_with_prior(
        self,
        x01: torch.Tensor,
        point_coords_np: Optional[np.ndarray],
        point_labels_np: Optional[np.ndarray],
        prior_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        SAM2 forward pass that accepts an optional dense mask prompt.

        Used for clip-level cascaded training: frame 0 gets a GT point prompt;
        subsequent frames get the previous frame's predicted mask as a dense prompt
        (simulating how SAM2VideoPredictor propagates via first-frame prompting).

        Args:
            x01:              [1, 3, 1024, 1024] in [0, 1]
            point_coords_np:  [N, 2] float32  (None for frames 1+)
            point_labels_np:  [N]   int32      (None for frames 1+)
            prior_mask:       [1, 1, H, W] soft mask in [0, 1] from previous frame
                              (None for frame 0)
        Returns:
            logits: [1, 1, H, W]
        """
        x_norm = (x01 - self.mean) / self.std
        backbone_out = self.sam2.forward_image(x_norm)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)

        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._BB_FEAT_SIZES[::-1])
        ][::-1]
        image_embed    = feats[-1]
        high_res_feats = feats[:-1]

        # Convert soft mask from previous frame to SAM2 mask-prompt format.
        # SAM2 expects masks at input_image_size // 4 = 256×256, in logit space.
        if prior_mask is not None:
            # prior_mask is in [0,1]; convert to logits with a linear mapping
            mask_input = torch.nn.functional.interpolate(
                prior_mask, size=(256, 256), mode="bilinear", align_corners=False,
            )
            mask_input = mask_input * 20.0 - 10.0  # sigmoid-inverse approximation
        else:
            mask_input = None

        if point_coords_np is not None:
            if self._orig_hw is not None:
                H, W = self._orig_hw
                sx, sy = self.INPUT_SIZE / W, self.INPUT_SIZE / H
            else:
                sx, sy = 1.0, 1.0
            pts  = torch.from_numpy(point_coords_np).float().to(self.device)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            pts  = pts.unsqueeze(0)
            lbls = torch.from_numpy(point_labels_np).int().to(self.device).unsqueeze(0)
            points_arg = (pts, lbls)
        else:
            points_arg = None

        sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
            points=points_arg, boxes=None, masks=mask_input,
        )

        try:
            low_res_masks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
        except TypeError:
            low_res_masks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
            )

        if self._orig_hw is not None:
            logits = torch.nn.functional.interpolate(
                low_res_masks, size=self._orig_hw, mode="bilinear", align_corners=False,
            )
        else:
            logits = low_res_masks
        return logits


class SAM2VideoMemoryAttacker(nn.Module):
    """
    SAM2 video-predictor surrogate that uses the real memory_attention path.

    Fix A (Round 3): Close the train/eval mismatch by simulating how
    SAM2VideoPredictor conditions each frame on stored memories:
      - Frame 0: backbone → no_mem_embed → decode with GT point prompt
      - Frame 0 memory: encode via memory_encoder → store maskmem_features
      - Frame 1+: backbone → memory_attention(stored_mem) → decode with
                  prior soft-mask prompt (belt + suspenders conditioning)

    Falls back to forward_with_prior (prompt-encoder only) if memory_attention
    raises a shape or API error, so it degrades gracefully across SAM2 builds.
    """

    SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    SAM2_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    INPUT_SIZE = 1024
    _BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]

    def __init__(self, checkpoint: str, config_name: str, device: torch.device):
        super().__init__()
        self.device = device
        from sam2.build_sam import build_sam2
        sam2_model = build_sam2(config_name, checkpoint, device=device)
        sam2_model.eval()
        for p in sam2_model.parameters():
            p.requires_grad_(False)
        self.sam2 = sam2_model
        self.mean = self.SAM2_MEAN.to(device)
        self.std  = self.SAM2_STD.to(device)
        # Test whether memory_attention is callable on this build
        self._mem_attn_available: Optional[bool] = None
        # Stored by encode_image for per-frame forward compatibility
        self._orig_hw: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_backbone(self, x01: torch.Tensor):
        """Returns (vision_feats, vision_pos_embeds, feat_sizes)."""
        x_norm = (x01 - self.mean) / self.std
        backbone_out = self.sam2.forward_image(x_norm)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self.sam2._prepare_backbone_features(backbone_out)
        )
        return vision_feats, vision_pos_embeds, feat_sizes

    def _feats_to_image_embed(self, vision_feats):
        """Reshape [HW, B, C] sequence feats → [B, C, H, W] per-scale list."""
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *sz)
            for feat, sz in zip(vision_feats[::-1], self._BB_FEAT_SIZES[::-1])
        ][::-1]
        return feats[-1], feats[:-1]  # image_embed, high_res_feats

    def _decode(
        self,
        vision_feats,
        coords_np: Optional[np.ndarray],
        labels_np: Optional[np.ndarray],
        prior_mask: Optional[torch.Tensor],
        orig_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prompt-encoder + mask-decoder forward.
        Returns (logits [1,1,H,W], obj_ptr [B,hidden_dim] or None).
        """
        image_embed, high_res_feats = self._feats_to_image_embed(vision_feats)

        mask_input = None
        if prior_mask is not None:
            mask_input = torch.nn.functional.interpolate(
                prior_mask, size=(256, 256), mode="bilinear", align_corners=False,
            )
            mask_input = mask_input * 20.0 - 10.0  # logit approximation

        points_arg = None
        if coords_np is not None:
            H, W = orig_hw
            sx, sy = self.INPUT_SIZE / W, self.INPUT_SIZE / H
            pts  = torch.from_numpy(coords_np).float().to(self.device)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            pts  = pts.unsqueeze(0)
            lbls = torch.from_numpy(labels_np).int().to(self.device).unsqueeze(0)
            points_arg = (pts, lbls)

        sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
            points=points_arg, boxes=None, masks=mask_input,
        )
        try:
            dec_out = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
        except TypeError:
            dec_out = self.sam2.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
                repeat_image=False,
            )
        low_res_masks = dec_out[0]
        # Index 2 is sam_tokens_out (object pointer token pre-projection)
        sam_tokens_out = dec_out[2] if len(dec_out) > 2 else None

        # Extract object pointer for memory storage (Fix E — Round 4)
        obj_ptr: Optional[torch.Tensor] = None
        if sam_tokens_out is not None and hasattr(self.sam2, "obj_ptr_proj"):
            try:
                obj_ptr = self.sam2.obj_ptr_proj(sam_tokens_out[:, 0])  # [B, hidden_dim]
            except Exception:
                pass

        logits = torch.nn.functional.interpolate(
            low_res_masks, size=orig_hw, mode="bilinear", align_corners=False,
        )
        return logits, obj_ptr  # ([1,1,H,W], [B,C] or None)

    def _build_memory(self, vision_feats, feat_sizes, logits):
        """
        Encode current frame into memory tokens using SAM2's memory_encoder.

        Returns (maskmem_features [B,C_m,H_m,W_m],
                 maskmem_pos_flat  [H_m*W_m, B, C_m])
        or (None, None) if memory_encoder is unavailable/broken.
        """
        try:
            B = vision_feats[-1].size(1)
            H_f, W_f = feat_sizes[-1]
            pix_feat = (
                vision_feats[-1].permute(1, 2, 0).view(B, -1, H_f, W_f)
            )  # [B, C, H_f, W_f]

            # memory_encoder expects masks at full input resolution
            masks_1024 = torch.nn.functional.interpolate(
                logits.detach(),
                size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )

            maskmem_features, maskmem_pos_enc = self.sam2.memory_encoder(
                pix_feat, masks_1024, skip_mask_sigmoid=False,
            )
            # maskmem_features: [B, C_m, H_m, W_m]
            # maskmem_pos_enc: list[Tensor]; take index 0
            pos = maskmem_pos_enc[0] if isinstance(maskmem_pos_enc, (list, tuple)) \
                else maskmem_pos_enc
            # Flatten spatial dims: [B, C, H, W] → [H*W, B, C]
            _B, _C, _H, _W = maskmem_features.shape
            mem_flat = maskmem_features.view(_B, _C, _H * _W).permute(2, 0, 1)
            _B2, _C2, _H2, _W2 = pos.shape
            pos_flat = pos.view(_B2, _C2, _H2 * _W2).permute(2, 0, 1)
            return mem_flat, pos_flat
        except Exception as e:
            return None, None

    def _apply_memory_attention(
        self,
        vision_feats,
        vision_pos_embeds,
        mem_flat: torch.Tensor,
        mem_pos_flat: torch.Tensor,
        obj_ptr: Optional[torch.Tensor] = None,
        frame_dist: int = 1,
    ):
        """
        Run memory_attention to condition backbone features on stored memory.

        Fix E (Round 4): also appends object pointer tokens from previous frame
        with temporal positional encoding, matching the actual SAM2 video path.

        Returns updated vision_feats list (only [-1] level is modified).
        """
        curr     = vision_feats[-1]       # [HW, B, C]
        curr_pos = vision_pos_embeds[-1]  # [HW, B, C]
        B = curr.shape[1]
        device = curr.device

        # Add temporal positional encoding to spatial memory pos
        # SAM2 uses maskmem_tpos_enc[num_maskmem - t_pos - 1] where t_pos=1 for most recent frame
        mem_pos_with_tpos = mem_pos_flat
        try:
            n_maskmem = self.sam2.maskmem_tpos_enc.shape[0]  # e.g. 7
            t_pos = min(frame_dist, n_maskmem - 1)
            tpos_enc = self.sam2.maskmem_tpos_enc[n_maskmem - t_pos - 1]  # [1, 1, mem_dim]
            mem_pos_with_tpos = mem_pos_flat + tpos_enc.view(1, 1, -1)
        except Exception:
            pass

        # Append object pointer token (Fix E)
        num_obj_ptr_tokens = 0
        memory     = mem_flat
        memory_pos = mem_pos_with_tpos
        if obj_ptr is not None and getattr(self.sam2, "use_obj_ptrs_in_encoder", False):
            try:
                hidden_dim = obj_ptr.shape[-1]
                mem_dim    = mem_flat.shape[-1]
                obj_ptr_seq = obj_ptr.unsqueeze(0)  # [1, B, hidden_dim]

                # Temporal positional encoding for the object pointer
                try:
                    from sam2.modeling.sam2_utils import get_1d_sine_pe
                    max_ptrs  = getattr(self.sam2, "max_obj_ptrs_in_encoder", 16)
                    t_diff_max = max(max_ptrs - 1, 1)
                    tpos_dim   = hidden_dim if getattr(
                        self.sam2, "proj_tpos_enc_in_obj_ptrs", False) else mem_dim
                    t_val = torch.tensor(
                        [min(frame_dist, t_diff_max) / t_diff_max],
                        device=device, dtype=obj_ptr.dtype,
                    )
                    obj_pos_raw = get_1d_sine_pe(t_val, dim=tpos_dim)  # [1, tpos_dim]
                    obj_pos = self.sam2.obj_ptr_tpos_proj(obj_pos_raw)  # [1, mem_dim]
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)   # [1, B, mem_dim]
                except Exception:
                    obj_pos = obj_ptr_seq.new_zeros(1, B, mem_dim)

                # Handle dim split: if mem_dim < hidden_dim, split pointer into tokens
                if mem_dim < hidden_dim:
                    ratio = hidden_dim // mem_dim
                    obj_ptr_seq = (
                        obj_ptr_seq.view(-1, B, ratio, mem_dim)
                        .permute(0, 2, 1, 3).flatten(0, 1)
                    )  # [ratio, B, mem_dim]
                    obj_pos = obj_pos.repeat_interleave(ratio, dim=0)
                    num_obj_ptr_tokens = ratio
                else:
                    num_obj_ptr_tokens = 1

                memory     = torch.cat([mem_flat,          obj_ptr_seq], dim=0)
                memory_pos = torch.cat([mem_pos_with_tpos, obj_pos    ], dim=0)
            except Exception:
                num_obj_ptr_tokens = 0

        updated = self.sam2.memory_attention(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )  # [HW, B, C]

        new_feats = list(vision_feats)
        new_feats[-1] = updated
        return new_feats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, img_np: np.ndarray) -> torch.Tensor:
        """Matches SAM2Attacker.encode_image — returns [1,3,1024,1024] in [0,1]."""
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        H, W = img_np.shape[:2]
        self._orig_hw = (H, W)
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = torch.nn.functional.interpolate(
            x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        return x

    def forward(
        self,
        x01: torch.Tensor,
        point_coords_np: np.ndarray,
        point_labels_np: np.ndarray,
    ) -> torch.Tensor:
        """
        Per-frame image predictor forward (no memory), compatible with SAM2Attacker.forward.
        Allows SAM2VideoMemoryAttacker to be passed to eval_quick without loading a second model.
        """
        orig_hw = self._orig_hw if self._orig_hw is not None else (x01.shape[-2], x01.shape[-1])
        vision_feats, vision_pos_embeds, feat_sizes = self._encode_backbone(x01)
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        logits, _ = self._decode(vision_feats, point_coords_np, point_labels_np,
                                 prior_mask=None, orig_hw=orig_hw)
        return logits

    def forward_clip(
        self,
        frames_x01: List[torch.Tensor],
        masks_np: List[np.ndarray],
        coords_np: np.ndarray,
        labels_np: np.ndarray,
    ) -> List[torch.Tensor]:
        """
        Process a clip with real memory-attention propagation.

        Frame 0: no-mem-embed + GT point prompt
        Frame 1+: memory_attention conditioning + prior soft-mask prompt
                  (falls back to prompt-only if memory_attention unavailable)
        """
        all_logits: List[torch.Tensor] = []
        mem_flat:     Optional[torch.Tensor] = None
        mem_pos_flat: Optional[torch.Tensor] = None
        obj_ptr:      Optional[torch.Tensor] = None  # Fix E: carry object pointer
        prior_mask:   Optional[torch.Tensor] = None

        for i, x01 in enumerate(frames_x01):
            orig_hw = masks_np[i].shape[:2]  # (H, W)
            vision_feats, vision_pos_embeds, feat_sizes = self._encode_backbone(x01)

            if i == 0:
                # Frame 0: add no-memory embedding (matches SAM2ImagePredictor)
                if self.sam2.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
                logits, obj_ptr = self._decode(vision_feats, coords_np, labels_np,
                                               prior_mask=None, orig_hw=orig_hw)
            else:
                # Frame 1+: try memory attention (with obj_ptr), fall back to prompt-only
                if mem_flat is not None:
                    try:
                        vision_feats = self._apply_memory_attention(
                            vision_feats, vision_pos_embeds, mem_flat, mem_pos_flat,
                            obj_ptr=obj_ptr, frame_dist=i,
                        )
                    except Exception:
                        # memory_attention unavailable → prompt-only
                        if self.sam2.directly_add_no_mem_embed:
                            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
                else:
                    if self.sam2.directly_add_no_mem_embed:
                        vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
                # Prior soft mask as additional prompt
                logits, obj_ptr = self._decode(vision_feats, None, None,
                                               prior_mask=prior_mask, orig_hw=orig_hw)

            # Update memory bank and prior mask for next frame
            mem_flat, mem_pos_flat = self._build_memory(vision_feats, feat_sizes, logits)
            prior_mask = torch.sigmoid(logits).detach()
            all_logits.append(logits)

        return all_logits


def get_centroid_prompt(mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_np.astype(bool))
    if len(ys) == 0:
        H, W = mask_np.shape
        cx, cy = W // 2, H // 2
    else:
        cx, cy = int(xs.mean()), int(ys.mean())
    return np.array([[cx, cy]], dtype=np.float32), np.array([1], dtype=np.int32)


def build_frame_pool(video_names: List[str], davis_root: str, max_frames: int = 30) -> List[Dict]:
    pool = []
    for vid in video_names:
        try:
            frames, masks, _ = load_single_video(davis_root, vid, max_frames=max_frames)
            for i, (f, m) in enumerate(zip(frames, masks)):
                pool.append({"video": vid, "frame_idx": i, "frame_np": f, "mask_np": m})
        except Exception as e:
            print(f"  [WARN] Could not load {vid}: {e}")
    return pool


def build_clip_pool(
    video_names: List[str],
    davis_root: str,
    max_frames: int = 30,
    clip_len: int = 4,
) -> List[Dict]:
    """
    Build a pool of overlapping clips (clip_len consecutive frames per clip).
    Used for clip-level cascaded SAM2 training (Fix 3 of Round 2 review).
    """
    clips = []
    stride = max(1, clip_len // 2)
    for vid in video_names:
        try:
            frames, masks, _ = load_single_video(davis_root, vid, max_frames=max_frames)
            for start in range(0, len(frames) - clip_len + 1, stride):
                clip_frames = frames[start: start + clip_len]
                clip_masks  = masks[start:  start + clip_len]
                # Only include clips where at least one frame has a visible object
                if any(m.sum() >= 100 for m in clip_masks):
                    clips.append({
                        "video":  vid,
                        "start":  start,
                        "frames": clip_frames,
                        "masks":  clip_masks,
                    })
        except Exception as e:
            print(f"  [WARN] Could not load {vid}: {e}")
    return clips


def eval_quick(
    attacker: SAM2Attacker,
    g_theta,
    val_videos: List[str],
    davis_root: str,
    device: torch.device,
    max_frames: int = 10,
    mode: str = "ours",
    uap_delta: Optional[torch.Tensor] = None,
    g_theta_size: int = 1024,
) -> Dict:
    jf_clean_list, jf_adv_list = [], []
    for vid in val_videos[:3]:
        try:
            frames, masks, _ = load_single_video(davis_root, vid, max_frames=max_frames)
        except Exception:
            continue
        for frame_np, mask_np in zip(frames, masks):
            if mask_np.sum() < 100:
                continue
            coords, labels = get_centroid_prompt(mask_np)
            with torch.no_grad():
                x01 = attacker.encode_image(frame_np)
                logits_clean = attacker(x01, coords, labels)
                pred_clean = (torch.sigmoid(logits_clean[0, 0]) > 0.5).cpu().numpy()
                if mode == "uap" and uap_delta is not None:
                    x_adv = torch.clamp(x01 + uap_delta, 0, 1)
                elif mode == "ours" and g_theta is not None:
                    g_theta.eval()
                    gs = g_theta_size
                    if gs < SAM2Attacker.INPUT_SIZE:
                        x_sm = torch.nn.functional.interpolate(
                            x01, size=(gs, gs), mode="bilinear", align_corners=False)
                        delta_sm = g_theta(x_sm)[1]
                        delta_up = torch.nn.functional.interpolate(
                            delta_sm, size=(SAM2Attacker.INPUT_SIZE, SAM2Attacker.INPUT_SIZE),
                            mode="bilinear", align_corners=False)
                        x_adv = (x01 + delta_up).clamp(0, 1)
                    else:
                        x_adv = g_theta(x01)[0]
                    g_theta.train()
                else:
                    x_adv = x01
                logits_adv = attacker(x_adv, coords, labels)
                pred_adv = (torch.sigmoid(logits_adv[0, 0]) > 0.5).cpu().numpy()
            jf_c, _, _ = jf_score(pred_clean, mask_np.astype(bool))
            jf_a, _, _ = jf_score(pred_adv,   mask_np.astype(bool))
            jf_clean_list.append(jf_c)
            jf_adv_list.append(jf_a)
    if not jf_clean_list:
        return {"mean_jf_clean": 0.0, "mean_jf_adv": 0.0, "delta_jf": 0.0}
    mjf_c = float(np.mean(jf_clean_list))
    mjf_a = float(np.mean(jf_adv_list))
    return {"mean_jf_clean": mjf_c, "mean_jf_adv": mjf_a, "delta_jf": mjf_c - mjf_a}


def eval_video_quick(
    g_theta,
    val_videos: List[str],
    davis_root: str,
    device: torch.device,
    sam2_checkpoint: str,
    sam2_config: str,
    g_theta_size: int = 256,
    max_frames: int = 8,
    min_jf_clean: float = 0.5,
) -> Dict:
    """
    Fix F (Round 4): Quick eval using the official SAM2VideoPredictor path.

    Unlike eval_quick (which uses SAM2ImagePredictor per-frame), this function
    uses SAM2VideoPredictor with first-frame GT point prompt + memory propagation
    — identical to the paper's evaluation protocol.

    Uses at most 2 videos and max_frames frames to keep it fast during training.
    """
    import tempfile
    import cv2 as _cv2
    jf_clean_list, jf_adv_list = [], []

    try:
        from sam2.build_sam import build_sam2_video_predictor
        video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
    except Exception as e:
        print(f"  [eval_video_quick] VideoPredictor unavailable: {e}")
        return {"vid_mean_jf_clean": 0.0, "vid_mean_jf_adv": 0.0, "vid_delta_jf": 0.0}

    for vid in val_videos[:2]:
        try:
            frames, masks, _ = load_single_video(davis_root, vid, max_frames=max_frames)
        except Exception:
            continue
        if not frames or masks[0].sum() < 100:
            continue

        def _run_video_predictor(frame_tensors_01):
            """Write frames to temp dir, run video predictor, return JF score."""
            with tempfile.TemporaryDirectory() as tmp:
                for fi, t in enumerate(frame_tensors_01):
                    arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    # Downscale for speed
                    arr_small = _cv2.resize(arr, (480, 270))
                    _cv2.imwrite(os.path.join(tmp, f"{fi:05d}.jpg"), arr_small[:, :, ::-1])
                with torch.inference_mode():
                    state = video_predictor.init_state(video_path=tmp)
                    # First-frame GT point prompt
                    mask0 = masks[0]
                    ys, xs = np.where(mask0.astype(bool))
                    if len(ys) == 0:
                        video_predictor.reset_state(state)
                        return None
                    cx, cy = int(xs.mean()), int(ys.mean())
                    pts = np.array([[cx, cy]], dtype=np.float32)
                    lbls = np.array([1], dtype=np.int32)
                    # Scale to small resolution
                    H0, W0 = mask0.shape
                    pts[:, 0] = pts[:, 0] * 480 / W0
                    pts[:, 1] = pts[:, 1] * 270 / H0
                    video_predictor.add_new_points_or_box(state, frame_idx=0,
                                                          obj_id=1, points=pts, labels=lbls)
                    pred_masks = {}
                    for fi, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
                        pred_masks[fi] = (mask_logits[0, 0] > 0).cpu().numpy()
                    video_predictor.reset_state(state)

                scores = []
                for fi, gt_mask in enumerate(masks):
                    if fi not in pred_masks:
                        continue
                    pm = pred_masks[fi]
                    if pm.shape != gt_mask.shape:
                        pm = _cv2.resize(pm.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]),
                                         interpolation=_cv2.INTER_NEAREST).astype(bool)
                    jf, _, _ = jf_score(pm, gt_mask.astype(bool))
                    scores.append(jf)
                return float(np.mean(scores)) if scores else None

        # Encode frames at SAM2 input size
        dummy_attacker = SAM2Attacker.__new__(SAM2Attacker)
        dummy_attacker.device = device
        dummy_attacker.INPUT_SIZE = 1024
        dummy_attacker._orig_hw = None

        frames_01 = []
        for f in frames:
            t = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            t = torch.nn.functional.interpolate(t, size=(1024, 1024), mode="bilinear", align_corners=False)
            frames_01.append(t)

        # Clean JF
        jf_c = _run_video_predictor(frames_01)
        if jf_c is None or jf_c < min_jf_clean:
            continue

        # Adversarial JF: apply g_theta
        g_theta.eval()
        with torch.no_grad():
            adv_frames = []
            for t in frames_01:
                t_small = torch.nn.functional.interpolate(
                    t, size=(g_theta_size, g_theta_size), mode="bilinear", align_corners=False)
                out = g_theta(t_small)
                delta_small = out[1] if len(out) >= 2 else out[0]
                delta = torch.nn.functional.interpolate(
                    delta_small, size=(1024, 1024), mode="bilinear", align_corners=False,
                ).clamp(-8.0/255, 8.0/255)
                adv_frames.append((t + delta).clamp(0, 1))
        g_theta.train()

        jf_a = _run_video_predictor(adv_frames)
        if jf_a is None:
            continue

        jf_clean_list.append(jf_c)
        jf_adv_list.append(jf_a)

    if not jf_clean_list:
        return {"vid_mean_jf_clean": 0.0, "vid_mean_jf_adv": 0.0, "vid_delta_jf": 0.0}
    mjf_c = float(np.mean(jf_clean_list))
    mjf_a = float(np.mean(jf_adv_list))
    return {"vid_mean_jf_clean": mjf_c, "vid_mean_jf_adv": mjf_a, "vid_delta_jf": mjf_c - mjf_a}


def save_results(out_dir: str, run_name: str, history: List[Dict], args_dict: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{run_name}.json")
    with open(json_path, "w") as f:
        json.dump({"args": args_dict, "history": history}, f, indent=2)
    if history:
        # Collect all keys that appear across any row (rows may have different keys)
        all_keys: list = []
        seen_keys: set = set()
        for row in history:
            for k in row.keys():
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)
        csv_path = os.path.join(out_dir, f"{run_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore",
                                    restval="")
            writer.writeheader()
            writer.writerows(history)
    print(f"  Results saved -> {json_path}")


def train_uap(args, attacker: SAM2Attacker, frame_pool: List[Dict], device: torch.device) -> None:
    eps = args.max_delta
    uap_lr = args.uap_lr if args.uap_lr is not None else args.lr
    use_lpips = args.uap_lpips
    print(f"\n[UAP] eps={eps:.4f}  lr={uap_lr}  lpips_hinge={use_lpips}")
    delta = torch.zeros(1, 3, SAM2Attacker.INPUT_SIZE, SAM2Attacker.INPUT_SIZE,
                        device=device, requires_grad=True)
    optimizer = optim.Adam([delta], lr=uap_lr)
    perc_fn   = PerceptualLoss(threshold=args.max_lpips, device=device) if use_lpips else None
    run_name  = (f"{args.tag}_" if args.tag else "") + f"uap_eps{eps:.4f}_steps{args.num_steps}"
    if use_lpips:
        run_name += "_lpips"
    out_dir   = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history   = []
    pbar = tqdm(range(1, args.num_steps + 1), desc="UAP")
    for step in pbar:
        item = random.choice(frame_pool)
        if item["mask_np"].sum() < 50:
            continue
        coords, labels = get_centroid_prompt(item["mask_np"])
        x01 = attacker.encode_image(item["frame_np"])
        # Project delta before forward (for loss computation)
        delta_c = torch.clamp(delta, -eps, eps)
        x_adv   = torch.clamp(x01 + delta_c, 0.0, 1.0)
        logits  = attacker(x_adv, coords, labels)
        gt = torch.from_numpy(item["mask_np"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = torch.nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        loss = soft_iou_loss(logits, gt)
        if perc_fn is not None:
            loss = loss + args.lambda1 * perc_fn(x01, x_adv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Project back onto L-inf ball after Adam step
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if step % args.log_every == 0:
            row = {"step": step, "loss": loss.item()}
            history.append(row)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if step % args.eval_every == 0:
            # Use post-step projected delta for eval
            with torch.no_grad():
                delta_eval = torch.clamp(delta, -eps, eps)
            ev = eval_quick(attacker, None, DAVIS_MINI_VAL[:3], args.davis_root,
                            device, mode="uap", uap_delta=delta_eval)
            history[-1].update(ev)
            print(f"\n  step={step} JF_clean={ev['mean_jf_clean']:.3f} "
                  f"JF_adv={ev['mean_jf_adv']:.3f} dJF={ev['delta_jf']:.3f}")
        if step % args.save_every == 0:
            with torch.no_grad():
                delta_save = torch.clamp(delta, -eps, eps)
            torch.save(delta_save.cpu(), os.path.join(out_dir, f"uap_delta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))
    with torch.no_grad():
        delta_final = torch.clamp(delta, -eps, eps)
    torch.save(delta_final.cpu(), os.path.join(out_dir, "uap_delta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[UAP] Done -> {out_dir}")


def train_ours(args, attacker: SAM2Attacker, frame_pool: List[Dict], device: torch.device,
               video_names: Optional[List[str]] = None) -> None:
    print(f"\n[OURS] Stage {args.stage}, steps={args.num_steps}, "
          f"g_accum={args.g_accum_steps}, g_theta_size={args.g_theta_size}")
    if args.stage == 4:
        g_theta = Stage4Preprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                      max_delta=args.max_delta).to(device)
    else:
        g_theta = ResidualPreprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                        max_delta=args.max_delta).to(device)
    print(f"  {g_theta.__class__.__name__}  params={sum(p.numel() for p in g_theta.parameters()):,}")
    if args.checkpoint:
        g_theta.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Resumed from {args.checkpoint}")
    # AdamW for Stage 2+ (as planned), Adam for Stage 1
    _opt_cls = optim.AdamW if (args.optimizer == "adamw" or args.stage >= 2) else optim.Adam
    optimizer  = _opt_cls(g_theta.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max in effective optimizer steps
    t_max = max(1, args.num_steps // args.g_accum_steps)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr * 0.1)
    perc_fn    = PerceptualLoss(threshold=args.max_lpips, device=device)
    ssim_fn    = SSIMConstraint(threshold=args.max_ssim_loss) if args.use_ssim else None
    run_name   = (f"{args.tag}_" if args.tag else "") + f"ours_s{args.stage}_steps{args.num_steps}"
    out_dir    = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history    = []

    # Stage 2+: sequential frame iterator so consecutive frames appear naturally,
    # enabling temporal loss to fire reliably.
    if args.stage >= 2:
        frame_pool_seq = sorted(frame_pool, key=lambda x: (x["video"], x["frame_idx"]))
        frame_iter: itertools.cycle = itertools.cycle(frame_pool_seq)
    else:
        frame_iter = None

    # Temporal loss state
    prev_delta:   Optional[torch.Tensor] = None
    prev_vid_key: Optional[str]          = None  # "videoname:frame_idx"

    optimizer.zero_grad()
    pbar = tqdm(range(1, args.num_steps + 1), desc=f"Stage{args.stage}")
    for step in pbar:
        if frame_iter is not None:
            item = next(frame_iter)
        else:
            item = random.choice(frame_pool)
        if item["mask_np"].sum() < 50:
            continue
        coords, labels = get_centroid_prompt(item["mask_np"])
        x01 = attacker.encode_image(item["frame_np"])
        g_theta.train()
        # Run g_theta at reduced resolution to save VRAM
        gs = args.g_theta_size
        x_small = torch.nn.functional.interpolate(x01, size=(gs, gs), mode="bilinear", align_corners=False)
        if args.stage == 4:
            x_adv_small, delta_small, _ = g_theta(x_small)
        else:
            x_adv_small, delta_small = g_theta(x_small)
        # Upsample delta to SAM2 input resolution and apply
        delta = torch.nn.functional.interpolate(delta_small, size=(1024, 1024), mode="bilinear", align_corners=False)
        delta = delta.clamp(-args.max_delta, args.max_delta)
        x_adv = (x01 + delta).clamp(0.0, 1.0)
        if args.stage >= 3 and random.random() < args.eot_prob:
            x_adv_in = codec_proxy_transform(x_adv, p_apply=1.0, p_yuv420=1.0)
        else:
            x_adv_in = x_adv
        logits = attacker(x_adv_in, coords, labels)
        gt = torch.from_numpy(item["mask_np"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = torch.nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        l_attack = soft_iou_loss(logits, gt)
        l_perc   = perc_fn(x_small, x_adv_small)
        loss     = l_attack + args.lambda1 * l_perc
        if ssim_fn is not None:
            l_ssim = ssim_fn(x_small, x_adv_small)
            loss   = loss + args.lambda4 * l_ssim
        # Stage 2+: temporal consistency between consecutive frames of the same video
        l_temp = torch.zeros(1, device=device)
        if args.stage >= 2 and prev_delta is not None:
            curr_key     = f"{item['video']}:{item['frame_idx']}"
            prev_expected = f"{item['video']}:{item['frame_idx'] - 1}"
            if prev_vid_key == prev_expected and prev_delta.shape == delta_small.shape:
                l_temp = temporal_loss([prev_delta, delta_small])
                loss   = loss + args.lambda2 * l_temp
        prev_delta   = delta_small.detach()
        prev_vid_key = f"{item['video']}:{item['frame_idx']}"

        # Gradient accumulation: normalise and accumulate
        (loss / args.g_accum_steps).backward()

        if step % args.g_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(g_theta.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_every == 0:
            history.append({
                "step": step, "loss": loss.item(),
                "l_attack": l_attack.item(), "l_perc": l_perc.item(),
                "l_temp": l_temp.item(), "lr": scheduler.get_last_lr()[0],
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}", att=f"{l_attack.item():.4f}")
        if step % args.eval_every == 0:
            ev_val = eval_quick(attacker, g_theta, DAVIS_MINI_VAL[:3], args.davis_root, device,
                                g_theta_size=args.g_theta_size)
            if history:
                history[-1].update({f"val_{k}": v for k, v in ev_val.items()})
            print(f"\n  step={step} [val] JF_clean={ev_val['mean_jf_clean']:.3f} "
                  f"JF_adv={ev_val['mean_jf_adv']:.3f} dJF={ev_val['delta_jf']:.3f}")
            if args.sanity and step == 500:
                train_vids = video_names if video_names else list({it["video"] for it in frame_pool})
                ev_train = eval_quick(attacker, g_theta, train_vids, args.davis_root, device,
                                      g_theta_size=args.g_theta_size)
                sanity_threshold = 0.20
                print(f"  step=500 [train] JF_clean={ev_train['mean_jf_clean']:.3f} "
                      f"JF_adv={ev_train['mean_jf_adv']:.3f} dJF={ev_train['delta_jf']:.3f}")
                if ev_train["delta_jf"] < sanity_threshold:
                    print(f"\n[SANITY FAIL] train dJF={ev_train['delta_jf']:.3f} < {sanity_threshold}.")
                    save_results(out_dir, run_name, history, vars(args))
                    sys.exit(1)
                else:
                    print(f"\n[SANITY PASS] train dJF={ev_train['delta_jf']:.3f} >= {sanity_threshold}")
                    if args.num_steps == 500:
                        break
        if step % args.save_every == 0:
            torch.save(g_theta.state_dict(), os.path.join(out_dir, f"g_theta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))
    torch.save(g_theta.state_dict(), os.path.join(out_dir, "g_theta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[OURS] Done -> {out_dir}")


def train_ours_clip(
    args,
    attacker: SAM2Attacker,
    clip_pool: List[Dict],
    device: torch.device,
    video_names: Optional[List[str]] = None,
) -> None:
    """
    Clip-level cascaded training against SAM2 video propagation (Fix 3).

    For each clip of `clip_len` consecutive frames:
      - Frame 0: GT centroid point prompt (first-frame prompting, same as eval)
      - Frame 1+: previous frame's predicted soft mask as dense prompt (propagation)
    This directly matches how SAM2VideoPredictor is used at evaluation time.

    Loss = mean over frames of soft_iou_loss(predicted_mask, gt_mask)
    """
    print(f"\n[OURS-CLIP] Stage {args.stage}, steps={args.num_steps}, "
          f"clip_len={args.clip_len}, g_theta_size={args.g_theta_size}")
    if args.stage == 4:
        g_theta = Stage4Preprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                      max_delta=args.max_delta).to(device)
    else:
        g_theta = ResidualPreprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                        max_delta=args.max_delta).to(device)
    print(f"  {g_theta.__class__.__name__}  params={sum(p.numel() for p in g_theta.parameters()):,}")
    if args.checkpoint:
        g_theta.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Resumed from {args.checkpoint}")

    _opt_cls = optim.AdamW if (args.optimizer == "adamw" or args.stage >= 2) else optim.Adam
    optimizer = _opt_cls(g_theta.parameters(), lr=args.lr, weight_decay=1e-4)
    t_max     = max(1, args.num_steps // args.g_accum_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr * 0.1)
    perc_fn   = PerceptualLoss(threshold=args.max_lpips, device=device)
    ssim_fn   = SSIMConstraint(threshold=args.max_ssim_loss) if args.use_ssim else None

    run_name = (f"{args.tag}_" if args.tag else "") + f"ours_clip_s{args.stage}_steps{args.num_steps}"
    out_dir  = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history: List[Dict] = []

    optimizer.zero_grad()
    pbar = tqdm(range(1, args.num_steps + 1), desc=f"Clip-S{args.stage}")
    for step in pbar:
        clip = random.choice(clip_pool)

        # Apply g_theta to every frame in the clip
        gs = args.g_theta_size
        x01s = [attacker.encode_image(f) for f in clip["frames"]]
        x_smalls, deltas, x_advs = [], [], []
        g_theta.train()
        for x01 in x01s:
            x_small = torch.nn.functional.interpolate(
                x01, size=(gs, gs), mode="bilinear", align_corners=False)
            if args.stage == 4:
                x_adv_small, delta_small, _ = g_theta(x_small)
            else:
                x_adv_small, delta_small = g_theta(x_small)
            delta = torch.nn.functional.interpolate(
                delta_small, size=(1024, 1024), mode="bilinear", align_corners=False,
            ).clamp(-args.max_delta, args.max_delta)
            x_adv = (x01 + delta).clamp(0.0, 1.0)
            if args.stage >= 3 and random.random() < args.eot_prob:
                x_adv = codec_proxy_transform(x_adv, p_apply=1.0, p_yuv420=1.0)
            x_smalls.append(x_small)
            deltas.append(delta_small)
            x_advs.append(x_adv)

        # Cascaded SAM2 forward: frame 0 with GT point, frame 1+ with prior mask
        l_attack  = torch.tensor(0.0, device=device)
        prior_mask: Optional[torch.Tensor] = None
        coords, labels = get_centroid_prompt(clip["masks"][0])
        for i, (x_adv, mask_np) in enumerate(zip(x_advs, clip["masks"])):
            attacker.encode_image(clip["frames"][i])  # set _orig_hw
            if i == 0:
                logits = attacker.forward_with_prior(x_adv, coords, labels, prior_mask=None)
            else:
                logits = attacker.forward_with_prior(x_adv, None, None, prior_mask=prior_mask)
            gt = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if gt.shape[-2:] != logits.shape[-2:]:
                gt = torch.nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
            l_attack = l_attack + soft_iou_loss(logits, gt)
            # Pass soft predicted mask as prior (detach to avoid unbounded backprop chain)
            prior_mask = torch.sigmoid(logits).detach()

        l_attack = l_attack / len(clip["frames"])

        # Perceptual constraints (on g_theta resolution)
        l_perc = torch.stack([perc_fn(x_smalls[i], x_smalls[i] + deltas[i])
                               for i in range(len(x_smalls))]).mean()
        loss = l_attack + args.lambda1 * l_perc

        if ssim_fn is not None:
            l_ssim = torch.stack([ssim_fn(x_smalls[i], x_smalls[i] + deltas[i])
                                   for i in range(len(x_smalls))]).mean()
            loss = loss + args.lambda4 * l_ssim
        else:
            l_ssim = torch.tensor(0.0, device=device)

        # Stage 2+: temporal consistency across clip
        l_temp = torch.tensor(0.0, device=device)
        if args.stage >= 2 and len(deltas) >= 2:
            l_temp = temporal_loss(deltas)
            loss   = loss + args.lambda2 * l_temp

        (loss / args.g_accum_steps).backward()
        if step % args.g_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(g_theta.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_every == 0:
            history.append({
                "step": step, "loss": loss.item(),
                "l_attack": l_attack.item(), "l_perc": l_perc.item(),
                "l_ssim": l_ssim.item(), "l_temp": l_temp.item(),
                "lr": scheduler.get_last_lr()[0],
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}", att=f"{l_attack.item():.4f}")

        if step % args.eval_every == 0:
            ev_val = eval_quick(attacker, g_theta, DAVIS_MINI_VAL[:3], args.davis_root, device,
                                g_theta_size=args.g_theta_size)
            if history:
                history[-1].update({f"val_{k}": v for k, v in ev_val.items()})
            print(f"\n  step={step} [val] JF_clean={ev_val['mean_jf_clean']:.3f} "
                  f"JF_adv={ev_val['mean_jf_adv']:.3f} dJF={ev_val['delta_jf']:.3f}")

        if step % args.save_every == 0:
            torch.save(g_theta.state_dict(), os.path.join(out_dir, f"g_theta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))

    torch.save(g_theta.state_dict(), os.path.join(out_dir, "g_theta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[OURS-CLIP] Done -> {out_dir}")


def train_ours_video_clip(
    args,
    vid_attacker: SAM2VideoMemoryAttacker,
    clip_pool: List[Dict],
    device: torch.device,
    video_names: Optional[List[str]] = None,
) -> None:
    """
    Fix A (Round 3): Train g_theta against SAM2VideoMemoryAttacker.

    Uses the real memory_attention path for frame 1+ conditioning, directly
    matching how SAM2VideoPredictor operates at evaluation time.  Falls back
    transparently to prior-mask-only conditioning if memory_attention raises
    an exception on the installed SAM2 build.
    """
    print(f"\n[OURS-VIDEO] Stage {args.stage}, steps={args.num_steps}, "
          f"clip_len={args.clip_len}, g_theta_size={args.g_theta_size}")
    if args.stage == 4:
        g_theta = Stage4Preprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                      max_delta=args.max_delta).to(device)
    else:
        g_theta = ResidualPreprocessor(channels=args.channels, num_blocks=args.num_blocks,
                                        max_delta=args.max_delta).to(device)
    print(f"  {g_theta.__class__.__name__}  params={sum(p.numel() for p in g_theta.parameters()):,}")
    if args.checkpoint:
        g_theta.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Resumed from {args.checkpoint}")

    _opt_cls  = optim.AdamW if (args.optimizer == "adamw" or args.stage >= 2) else optim.Adam
    optimizer = _opt_cls(g_theta.parameters(), lr=args.lr, weight_decay=1e-4)
    t_max     = max(1, args.num_steps // args.g_accum_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr * 0.1)
    perc_fn   = PerceptualLoss(threshold=args.max_lpips, device=device)
    ssim_fn   = SSIMConstraint(threshold=args.max_ssim_loss) if args.use_ssim else None

    run_name = (f"{args.tag}_" if args.tag else "") + f"ours_video_s{args.stage}_steps{args.num_steps}"
    out_dir  = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    history: List[Dict] = []

    optimizer.zero_grad()
    pbar = tqdm(range(1, args.num_steps + 1), desc=f"Video-S{args.stage}")
    for step in pbar:
        clip = random.choice(clip_pool)
        gs   = args.g_theta_size

        # Apply g_theta to all frames in the clip
        x01s   = [vid_attacker.encode_image(f) for f in clip["frames"]]
        x_smalls, deltas, x_advs = [], [], []
        g_theta.train()
        for x01 in x01s:
            x_small = torch.nn.functional.interpolate(
                x01, size=(gs, gs), mode="bilinear", align_corners=False)
            if args.stage == 4:
                x_adv_small, delta_small, _ = g_theta(x_small)
            else:
                x_adv_small, delta_small = g_theta(x_small)
            delta = torch.nn.functional.interpolate(
                delta_small, size=(SAM2Attacker.INPUT_SIZE, SAM2Attacker.INPUT_SIZE),
                mode="bilinear", align_corners=False,
            ).clamp(-args.max_delta, args.max_delta)
            x_adv = (x01 + delta).clamp(0.0, 1.0)
            if args.stage >= 3 and random.random() < args.eot_prob:
                x_adv = codec_proxy_transform(x_adv, p_apply=1.0, p_yuv420=1.0)
            x_smalls.append(x_small)
            deltas.append(delta_small)
            x_advs.append(x_adv)

        # Forward through SAM2VideoMemoryAttacker (memory attention + prior mask)
        coords, labels = get_centroid_prompt(clip["masks"][0])
        all_logits = vid_attacker.forward_clip(x_advs, clip["masks"], coords, labels)

        # Attack loss: suppression across all frames
        l_attack = torch.tensor(0.0, device=device)
        for logits, mask_np in zip(all_logits, clip["masks"]):
            gt = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if gt.shape[-2:] != logits.shape[-2:]:
                gt = torch.nn.functional.interpolate(gt, size=logits.shape[-2:], mode="nearest")
            l_attack = l_attack + soft_iou_loss(logits, gt)
        l_attack = l_attack / len(all_logits)

        # Perceptual constraints at g_theta resolution
        l_perc = torch.stack([perc_fn(x_smalls[i], x_smalls[i] + deltas[i])
                               for i in range(len(x_smalls))]).mean()
        loss = l_attack + args.lambda1 * l_perc

        if ssim_fn is not None:
            l_ssim = torch.stack([ssim_fn(x_smalls[i], x_smalls[i] + deltas[i])
                                   for i in range(len(x_smalls))]).mean()
            loss = loss + args.lambda4 * l_ssim
        else:
            l_ssim = torch.tensor(0.0, device=device)

        # Temporal consistency across clip
        l_temp = torch.tensor(0.0, device=device)
        if args.stage >= 2 and len(deltas) >= 2:
            l_temp = temporal_loss(deltas)
            loss   = loss + args.lambda2 * l_temp

        (loss / args.g_accum_steps).backward()
        if step % args.g_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(g_theta.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_every == 0:
            history.append({
                "step": step, "loss": loss.item(),
                "l_attack": l_attack.item(), "l_perc": l_perc.item(),
                "l_ssim": l_ssim.item(), "l_temp": l_temp.item(),
                "lr": scheduler.get_last_lr()[0],
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}", att=f"{l_attack.item():.4f}")

        if step % args.eval_every == 0:
            # Image-predictor quick eval (fast proxy)
            ev_val = eval_quick(vid_attacker, g_theta, DAVIS_MINI_VAL[:3],
                                args.davis_root, device, g_theta_size=args.g_theta_size)
            if history:
                history[-1].update({f"val_{k}": v for k, v in ev_val.items()})
            print(f"\n  step={step} [val-img] JF_clean={ev_val['mean_jf_clean']:.3f} "
                  f"JF_adv={ev_val['mean_jf_adv']:.3f} dJF={ev_val['delta_jf']:.3f}")
            # Fix F: video-predictor eval (actual paper metric, 2-video quick estimate)
            if getattr(args, "eval_video", False):
                ev_vid = eval_video_quick(
                    g_theta, DAVIS_MINI_VAL[:2], args.davis_root, device,
                    sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
                    g_theta_size=args.g_theta_size,
                )
                if history:
                    history[-1].update({f"vid_{k}": v for k, v in ev_vid.items()})
                print(f"  step={step} [val-vid] JF_clean={ev_vid['vid_mean_jf_clean']:.3f} "
                      f"JF_adv={ev_vid['vid_mean_jf_adv']:.3f} dJF={ev_vid['vid_delta_jf']:.3f}")

        if step % args.save_every == 0:
            torch.save(g_theta.state_dict(), os.path.join(out_dir, f"g_theta_step{step}.pt"))
            save_results(out_dir, run_name, history, vars(args))

    torch.save(g_theta.state_dict(), os.path.join(out_dir, "g_theta_final.pt"))
    save_results(out_dir, run_name, history, vars(args))
    print(f"\n[OURS-VIDEO] Done -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    default="ours", choices=["ours", "uap"])
    p.add_argument("--stage",   type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--sanity",  action="store_true")
    p.add_argument("--davis_root",  default=DAVIS_ROOT)
    p.add_argument("--videos",      default=None)
    p.add_argument("--max_frames",  type=int, default=30)
    p.add_argument("--channels",    type=int,   default=32)
    p.add_argument("--num_blocks",  type=int,   default=4)
    p.add_argument("--max_delta",   type=float, default=8.0/255.0)
    p.add_argument("--num_steps",   type=int,   default=3000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--uap_lr",      type=float, default=None,
                   help="UAP-specific lr (default: same as --lr)")
    p.add_argument("--optimizer",   default="adam", choices=["adam", "adamw"],
                   help="Optimizer for g_theta (Stage 2+ defaults to adamw)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--lambda1",     type=float, default=1.0)
    p.add_argument("--lambda2",     type=float, default=0.1)
    p.add_argument("--lambda3",     type=float, default=0.05)
    p.add_argument("--max_lpips",   type=float, default=0.10)
    p.add_argument("--uap_lpips",   action="store_true",
                   help="Add LPIPS hinge to UAP baseline (fair-budget Anti-C baseline)")
    p.add_argument("--eot_prob",    type=float, default=0.5)
    p.add_argument("--g_theta_size", type=int, default=256,
                   help="g_theta runs at this resolution; delta is upsampled to 1024 before SAM2")
    p.add_argument("--g_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = g_accum_steps × 1)")
    p.add_argument("--use_full_train", action="store_true",
                   help="Use DAVIS_TRAIN_VIDEOS_ALL instead of DAVIS_MINI_TRAIN")
    # Clip-level / video-memory training (Fix 3 Round 2 / Fix A Round 3)
    p.add_argument("--train_mode", default="frame", choices=["frame", "clip", "video"],
                   help="frame: per-frame image predictor; "
                        "clip: cascaded prompt-propagation (round 2); "
                        "video: real memory_attention path (round 3, recommended)")
    p.add_argument("--clip_len", type=int, default=4,
                   help="Number of consecutive frames per clip (clip train_mode only)")
    # Fix F: video-predictor eval during training (actual paper metric proxy)
    p.add_argument("--eval_video", action="store_true",
                   help="Run SAM2VideoPredictor eval (2 videos) at every eval checkpoint. "
                        "Slower but directly measures the target metric.")
    # SSIM constraint (Fix 2)
    p.add_argument("--use_ssim",      action="store_true",
                   help="Add SSIM hinge to perceptual loss (enforces SSIM ≥ 1 - max_ssim_loss)")
    p.add_argument("--max_ssim_loss", type=float, default=0.05,
                   help="Max allowed 1-SSIM (threshold for SSIMConstraint hinge); "
                        "0.05 → SSIM ≥ 0.95 (matches paper claim); 0.10 → SSIM ≥ 0.90")
    p.add_argument("--lambda4",       type=float, default=1.0,
                   help="Weight of SSIM constraint term")
    p.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--sam2_config",     default=SAM2_CONFIG)
    p.add_argument("--save_dir",    default=RESULTS_DIR)
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--tag",         default=None)
    p.add_argument("--log_every",   type=int, default=10)
    p.add_argument("--eval_every",  type=int, default=500)
    p.add_argument("--save_every",  type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
    if args.videos:
        video_names = [v.strip() for v in args.videos.split(",")]
    elif args.use_full_train:
        video_names = DAVIS_TRAIN_VIDEOS_ALL
    else:
        video_names = DAVIS_MINI_TRAIN
    print(f"Videos: {video_names}")
    frame_pool = build_frame_pool(video_names, args.davis_root, args.max_frames)
    if not frame_pool:
        print("[ERROR] No frames loaded.")
        sys.exit(1)
    print(f"Frame pool: {len(frame_pool)} frames")
    print("Loading SAM2...")
    if args.mode == "uap":
        attacker = SAM2Attacker(args.sam2_checkpoint, args.sam2_config, device)
        attacker.eval()
        train_uap(args, attacker, frame_pool, device)
    elif args.train_mode == "video":
        clip_pool = build_clip_pool(video_names, args.davis_root, args.max_frames,
                                    clip_len=args.clip_len)
        if not clip_pool:
            print("[ERROR] No clips built — check --videos and --clip_len.")
            sys.exit(1)
        print(f"Clip pool: {len(clip_pool)} clips (len={args.clip_len})")
        vid_attacker = SAM2VideoMemoryAttacker(args.sam2_checkpoint, args.sam2_config, device)
        vid_attacker.eval()
        train_ours_video_clip(args, vid_attacker, clip_pool, device, video_names=video_names)
    elif args.train_mode == "clip":
        attacker = SAM2Attacker(args.sam2_checkpoint, args.sam2_config, device)
        attacker.eval()
        clip_pool = build_clip_pool(video_names, args.davis_root, args.max_frames,
                                    clip_len=args.clip_len)
        if not clip_pool:
            print("[ERROR] No clips built — check --videos and --clip_len.")
            sys.exit(1)
        print(f"Clip pool: {len(clip_pool)} clips (len={args.clip_len})")
        train_ours_clip(args, attacker, clip_pool, device, video_names=video_names)
    else:
        attacker = SAM2Attacker(args.sam2_checkpoint, args.sam2_config, device)
        attacker.eval()
        train_ours(args, attacker, frame_pool, device, video_names=video_names)


if __name__ == "__main__":
    main()
