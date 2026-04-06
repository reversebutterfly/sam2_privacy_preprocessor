"""
Strict reproduction scaffold for:
    "Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2"

This file is intentionally isolated from the local B1a/DAVIS/codec codepath.
Its purpose is to enforce the paper's original setting:

- YouTube-VOS data
- SA-V target features
- SAM2 1.0 hiera tiny checkpoint
- 1024x1024 inputs
- MI-FGSM + EMA perturbation update
- point+box mixed prompt bank (256 prompts)
- paper composite loss
- mIoU evaluation only

The outline assumes you have the official UAP-SAM2 repository and its helper
modules available on PYTHONPATH, or that you port those modules into this
workspace without changing their semantics.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ReproConfig:
    dataset_root: str = "./data/YOUTUBE"
    sav_root: str = "./data/sav_test/JPEGImages_24fps"
    sam2_checkpoint: str = "./checkpoints/sam2_hiera_tiny.pt"
    sam2_config: str = "configs/sam2/sam2_hiera_t.yaml"
    output_dir: str = "./uap_file"
    dataset_id: str = "YOUTUBE"
    image_limit: int = 100
    frame_limit: int = 15
    prompt_num: int = 256
    p_num: int = 10
    fea_num: int = 30
    epsilon: float = 10.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 10
    weight_loss_t: float = 1.0
    weight_loss_diff: float = 1.0
    weight_fea: float = 1e-6
    seed: int = 30
    device: str = "cuda:0"
    maskmem: int = 7
    input_size: int = 1024
    sam_mask_thresh: float = 0.0
    ema_decay: float = 0.95
    ema_mix: float = 0.05
    train_split: str = "train"
    eval_split: str = "valid"
    mode: str = "train_eval"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Strict reproduction scaffold for UAP-SAM2."
    )
    p.add_argument("--dataset_root", default="./data/YOUTUBE")
    p.add_argument("--sav_root", default="./data/sav_test/JPEGImages_24fps")
    p.add_argument("--sam2_checkpoint", default="./checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--sam2_config", default="configs/sam2/sam2_hiera_t.yaml")
    p.add_argument("--output_dir", default="./uap_file")
    p.add_argument("--dataset_id", default="YOUTUBE")
    p.add_argument("--image_limit", type=int, default=100)
    p.add_argument("--frame_limit", type=int, default=15)
    p.add_argument("--prompt_num", type=int, default=256)
    p.add_argument("--p_num", type=int, default=10)
    p.add_argument("--fea_num", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=10.0 / 255.0)
    p.add_argument("--alpha", type=float, default=2.0 / 255.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--weight_loss_t", type=float, default=1.0)
    p.add_argument("--weight_loss_diff", type=float, default=1.0)
    p.add_argument("--weight_fea", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--maskmem", type=int, default=7)
    p.add_argument("--input_size", type=int, default=1024)
    p.add_argument("--sam_mask_thresh", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.95)
    p.add_argument("--ema_mix", type=float, default=0.05)
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split", default="valid")
    p.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    return p


def parse_args() -> ReproConfig:
    args = build_parser().parse_args()
    return ReproConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_strict_reproduction(cfg: ReproConfig) -> None:
    if abs(cfg.epsilon - (10.0 / 255.0)) > 1e-12:
        raise ValueError("Strict reproduction requires epsilon = 10/255.")
    if abs(cfg.alpha - (2.0 / 255.0)) > 1e-12:
        raise ValueError("Strict reproduction requires alpha = 2/255.")
    if cfg.prompt_num != 256:
        raise ValueError("Strict reproduction requires prompt_num = 256.")
    if cfg.p_num != 10:
        raise ValueError("Strict reproduction requires P_num = 10.")
    if cfg.frame_limit != 15:
        raise ValueError("Strict reproduction requires frame_limit = 15.")
    if cfg.image_limit != 100:
        raise ValueError("Strict reproduction requires image_limit = 100.")
    if cfg.input_size != 1024:
        raise ValueError("Strict reproduction requires 1024x1024 inputs.")
    if cfg.sam_mask_thresh != 0.0:
        raise ValueError("Strict reproduction requires SAM_MASK_THRESH = 0.0.")
    if cfg.maskmem != 7:
        raise ValueError("Strict reproduction requires maskmem = 7.")
    if cfg.fea_num != 30:
        raise ValueError("Strict reproduction requires fea_num = 30.")
    if abs(cfg.weight_fea - 1e-6) > 1e-12:
        raise ValueError("Strict reproduction requires weight_fea = 1e-6.")
    if "sam2_hiera_tiny.pt" not in Path(cfg.sam2_checkpoint).name:
        raise ValueError("Strict reproduction requires the SAM2 1.0 tiny checkpoint.")


def assert_paths_exist(cfg: ReproConfig) -> None:
    required = [
        cfg.dataset_root,
        cfg.sav_root,
        cfg.sam2_checkpoint,
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required reproduction assets:\n" + "\n".join(missing)
        )


def ensure_official_modules_available() -> None:
    """
    The official code exposes critical logic in modules such as:
      - attack_setting.py
      - dataset_YOUTUBE.py
      - sam2_util.py
      - uap_attack.py
      - uap_atk_test.py

    Import them here to ensure you are using the official semantics instead of
    substituting local approximations.
    """
    try:
        import attack_setting  # noqa: F401
        import dataset_YOUTUBE  # noqa: F401
        import sam2_util  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Official UAP-SAM2 modules are not importable. "
            "Clone CGCL-codes/UAP-SAM2 and add it to PYTHONPATH, or run this "
            "script from inside that repo."
        ) from exc


def make_prompt_bank(prompt_num: int) -> Sequence[Dict]:
    """
    The paper uses attack_setting.make_multi_prompts(prompt_num=256), which
    mixes point and box prompts. Do not replace this with centroid-only points.
    """
    from attack_setting import make_multi_prompts

    return make_multi_prompts(prompt_num=prompt_num)


def build_train_dataset(cfg: ReproConfig):
    """
    Strict path: use the official YouTube-VOS loader from dataset_YOUTUBE.py.
    """
    from dataset_YOUTUBE import YoutubeDataset

    return YoutubeDataset(
        data_root=cfg.dataset_root,
        max_num=cfg.image_limit,
        frame_limit=cfg.frame_limit,
        split=cfg.train_split,
        img_size=cfg.input_size,
    )


def build_eval_dataset(cfg: ReproConfig):
    from dataset_YOUTUBE import YoutubeDataset

    return YoutubeDataset(
        data_root=cfg.dataset_root,
        max_num=-1,
        frame_limit=-1,
        split=cfg.eval_split,
        img_size=cfg.input_size,
    )


def load_sav_target_features(cfg: ReproConfig, device: torch.device) -> torch.Tensor:
    """
    The paper samples fea_num=30 target features from SA-V frames.
    Use the official feature extraction path from sam2_util.sam_fwder.
    """
    sav_root = Path(cfg.sav_root)
    frame_paths = sorted(p for p in sav_root.rglob("*.jpg"))
    if len(frame_paths) < cfg.fea_num:
        raise ValueError(
            f"SA-V root has only {len(frame_paths)} frames, expected at least {cfg.fea_num}."
        )

    sam_fwder = build_sam_forwarder(cfg, device)
    selected = frame_paths[:cfg.fea_num]
    feats: List[torch.Tensor] = []
    for path in selected:
        feat = sam_fwder.get_image_feature(str(path))
        feats.append(F.normalize(feat.flatten(1), dim=1))
    return torch.cat(feats, dim=0)


def build_sam_forwarder(cfg: ReproConfig, device: torch.device):
    """
    The official implementation uses SAM2 1.0 helpers exposed through sam2_util.
    """
    from sam2_util import sam_fwder

    return sam_fwder(
        model_cfg=cfg.sam2_config,
        sam2_checkpoint=cfg.sam2_checkpoint,
        device=device,
        maskmem=cfg.maskmem,
    )


def build_video_predictor(cfg: ReproConfig, device: torch.device):
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(cfg.sam2_config, cfg.sam2_checkpoint, device=device)


def masked_semantic_confusion_loss(
    pred_logits: torch.Tensor,
    attacked_region_mask: torch.Tensor,
) -> torch.Tensor:
    target = torch.zeros_like(pred_logits)
    raw = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    masked = raw * attacked_region_mask
    return masked.sum() / (attacked_region_mask.sum() + 1e-6)


def info_nce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)

    pos_logits = (query * positive).sum(dim=1, keepdim=True) / temperature
    neg_logits = query @ negatives.t() / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def consecutive_feature_cosine_loss(features: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(features) < 2:
        return torch.zeros((), device=features[0].device if features else "cpu")
    pair_losses = []
    for prev_f, curr_f in zip(features[:-1], features[1:]):
        prev_n = F.normalize(prev_f.flatten(1), dim=1)
        curr_n = F.normalize(curr_f.flatten(1), dim=1)
        pair_losses.append(F.cosine_similarity(prev_n, curr_n, dim=1).mean())
    return torch.stack(pair_losses).mean()


def compose_total_loss(
    loss_t: torch.Tensor,
    loss_ft: torch.Tensor,
    feature_diff: torch.Tensor,
    loss_fea: torch.Tensor,
    cfg: ReproConfig,
) -> torch.Tensor:
    return (
        cfg.weight_loss_t * loss_t
        + 0.01 * loss_ft
        + cfg.weight_loss_diff * feature_diff
        + cfg.weight_fea * loss_fea
    )


def mi_fgsm_ema_update(
    perturbation: torch.Tensor,
    grad: torch.Tensor,
    ema_grad: torch.Tensor,
    step: int,
    cfg: ReproConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ema_grad = cfg.ema_decay * ema_grad + cfg.ema_mix * grad.detach()
    avg_gradient = ema_grad / (1.0 - cfg.ema_decay ** step)
    perturbation = (perturbation - avg_gradient.sign() * cfg.alpha).clamp(
        -cfg.epsilon, cfg.epsilon
    )
    return perturbation.detach(), ema_grad.detach()


def extract_attack_features(
    sam_fwder,
    adv_frames: Sequence[torch.Tensor],
) -> Sequence[torch.Tensor]:
    feats = []
    for frame in adv_frames:
        feats.append(sam_fwder.get_feature(frame))
    return feats


def select_prompt_batch(
    prompt_bank: Sequence[Dict],
    p_num: int,
) -> Sequence[Dict]:
    indices = np.random.choice(len(prompt_bank), size=p_num, replace=False)
    return [prompt_bank[int(i)] for i in indices]


def run_attack_episode(
    batch: Dict,
    perturbation: torch.Tensor,
    prompt_bank: Sequence[Dict],
    target_features: torch.Tensor,
    sam_fwder,
    video_predictor,
    cfg: ReproConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    This function mirrors the official training episode at a high level:

    1. Resize frames to 1024x1024.
    2. Apply the current universal perturbation to every selected frame.
    3. Scan random prompt subsets over attacked regions (point+box mix).
    4. Run SAM2 video propagation.
    5. Accumulate the four paper losses.

    The frame packing and prompt-to-region mapping should be copied from the
    official repo rather than rewritten from scratch.
    """
    prompts = select_prompt_batch(prompt_bank, cfg.p_num)
    frames = batch["images"].to(device)
    masks = batch["masks"].to(device)
    adv_frames = (frames + perturbation).clamp(0.0, 1.0)

    attack_outputs = forward_with_official_prompt_scanning(
        batch=batch,
        prompts=prompts,
        adv_frames=adv_frames,
        sam_fwder=sam_fwder,
        video_predictor=video_predictor,
        cfg=cfg,
    )

    loss_t = masked_semantic_confusion_loss(
        attack_outputs["pred_logits"],
        attack_outputs["attacked_region_mask"],
    )
    loss_ft = info_nce_loss(
        attack_outputs["frame_features"],
        attack_outputs["prototype_features"],
        attack_outputs["negative_features"],
    )
    feature_diff = consecutive_feature_cosine_loss(attack_outputs["memory_features"])
    loss_fea = info_nce_loss(
        attack_outputs["frame_features"],
        attack_outputs["target_positive_features"],
        target_features,
    )
    total_loss = compose_total_loss(loss_t, loss_ft, feature_diff, loss_fea, cfg)
    metrics = {
        "loss_t": loss_t.detach(),
        "loss_ft": loss_ft.detach(),
        "feature_diff": feature_diff.detach(),
        "loss_fea": loss_fea.detach(),
        "loss": total_loss.detach(),
    }
    return total_loss, metrics


def forward_with_official_prompt_scanning(
    batch: Dict,
    prompts: Sequence[Dict],
    adv_frames: torch.Tensor,
    sam_fwder,
    video_predictor,
    cfg: ReproConfig,
) -> Dict[str, torch.Tensor]:
    """
    This is the only part you should *not* improvise.

    Port the logic directly from the official:
      - target region division / random prompt allocation
      - mixed point+box prompting
      - predictor.add_new_points_or_box(...)
      - predictor.propagate_in_video(...)
      - feature extraction used for loss_ft, feature_diff, loss_fea

    Return keys expected by run_attack_episode().
    """
    raise NotImplementedError(
        "Copy the official target-scanning + feature-extraction logic here verbatim."
    )


def train_uap(cfg: ReproConfig) -> Path:
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    assert_strict_reproduction(cfg)
    assert_paths_exist(cfg)
    ensure_official_modules_available()

    train_dataset = build_train_dataset(cfg)
    prompt_bank = make_prompt_bank(cfg.prompt_num)
    target_features = load_sav_target_features(cfg, device)
    sam_fwder = build_sam_forwarder(cfg, device)
    video_predictor = build_video_predictor(cfg, device)

    perturbation = torch.zeros(
        1, 3, cfg.input_size, cfg.input_size, device=device, dtype=torch.float32
    )
    ema_grad = torch.zeros_like(perturbation)

    for step in range(1, cfg.steps + 1):
        batch = train_dataset[np.random.randint(0, len(train_dataset))]
        perturbation = perturbation.detach().requires_grad_(True)
        total_loss, metrics = run_attack_episode(
            batch=batch,
            perturbation=perturbation,
            prompt_bank=prompt_bank,
            target_features=target_features,
            sam_fwder=sam_fwder,
            video_predictor=video_predictor,
            cfg=cfg,
            device=device,
        )
        grad = torch.autograd.grad(total_loss, perturbation)[0]
        perturbation, ema_grad = mi_fgsm_ema_update(
            perturbation=perturbation,
            grad=grad,
            ema_grad=ema_grad,
            step=step,
            cfg=cfg,
        )
        print(
            f"[train] step={step:04d} "
            f"loss={metrics['loss'].item():.5f} "
            f"loss_t={metrics['loss_t'].item():.5f} "
            f"loss_ft={metrics['loss_ft'].item():.5f} "
            f"diff={metrics['feature_diff'].item():.5f} "
            f"loss_fea={metrics['loss_fea'].item():.5f}"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{cfg.dataset_id}.pth"
    torch.save(perturbation.detach().cpu(), save_path)
    print(f"[train] saved perturbation -> {save_path}")
    return save_path


def compute_binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter) / float(union)


def evaluate_miou(cfg: ReproConfig, perturbation_path: Optional[str] = None) -> Dict[str, float]:
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    assert_strict_reproduction(cfg)
    assert_paths_exist(cfg)
    ensure_official_modules_available()

    if perturbation_path is None:
        perturbation_path = str(Path(cfg.output_dir) / f"{cfg.dataset_id}.pth")
    perturbation = torch.load(perturbation_path, map_location=device).to(device)

    eval_dataset = build_eval_dataset(cfg)
    sam_fwder = build_sam_forwarder(cfg, device)
    video_predictor = build_video_predictor(cfg, device)

    clean_ious: List[float] = []
    adv_ious: List[float] = []
    for index in range(len(eval_dataset)):
        batch = eval_dataset[index]
        result = run_eval_video(
            batch=batch,
            perturbation=perturbation,
            sam_fwder=sam_fwder,
            video_predictor=video_predictor,
            cfg=cfg,
            device=device,
        )
        clean_ious.extend(result["clean_ious"])
        adv_ious.extend(result["adv_ious"])

    miou_clean = 100.0 * float(np.mean(clean_ious))
    miou_adv = 100.0 * float(np.mean(adv_ious))
    print(f"miouimg: {miou_clean:.2f}%, miouadv: {miou_adv:.2f}%")
    return {"miou_clean": miou_clean, "miou_adv": miou_adv}


def run_eval_video(
    batch: Dict,
    perturbation: torch.Tensor,
    sam_fwder,
    video_predictor,
    cfg: ReproConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Strict evaluation path:
      - same YouTube-VOS format
      - same 1024x1024 preprocessing
      - same prompt initialization API
      - same video propagation API
      - mIoU only
    """
    raise NotImplementedError(
        "Port the official uap_atk_test.py evaluation loop here verbatim."
    )


def main() -> None:
    cfg = parse_args()
    if cfg.mode in {"train", "train_eval"}:
        train_uap(cfg)
    if cfg.mode in {"eval", "train_eval"}:
        evaluate_miou(cfg)


if __name__ == "__main__":
    main()
