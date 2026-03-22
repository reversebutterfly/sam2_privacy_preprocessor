"""
DAVIS 2017 dataset loader for SAM2 Privacy Preprocessor.

Expected directory structure:
  <davis_root>/
    Annotations/480p/<video>/<frame:05d>.png   (indexed PNG, 0=bg, >0=object)
    JPEGImages/480p/<video>/<frame:05d>.jpg    (RGB JPEG)
    ImageSets/2017/train.txt
    ImageSets/2017/val.txt

Usage:
    ds = DAVISDataset(davis_root, split="train", video_names=["bear", "dog"],
                      max_frames=30)
    frames, masks, meta = ds[0]
    # frames: [T, 3, H, W] float [0,1]
    # masks:  [T, 1, H, W] float binary
    # meta:   dict with 'video', 'frame_indices'
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DAVISDataset(Dataset):
    """
    Loads fixed-length clips from DAVIS 2017 semi-supervised set.

    Each item is a contiguous clip of `max_frames` consecutive frames
    from a single video, along with the first-object binary mask.
    """

    def __init__(
        self,
        davis_root:    str,
        split:         str = "train",
        video_names:   Optional[List[str]] = None,
        max_frames:    int = 30,
        resolution:    str = "480p",
        obj_id:        int = 1,          # which object to track (1 = first object)
        stride:        int = 1,          # frame stride when building clips
    ):
        """
        Args:
            davis_root:  path to DAVIS root (contains Annotations/, JPEGImages/)
            split:       'train' or 'val' (used to filter videos when video_names is None)
            video_names: explicit list of videos to use; if None, uses split file
            max_frames:  number of frames per clip; clips are sampled from the start
            resolution:  '480p' or 'Full-Resolution'
            obj_id:      object ID in the annotation PNG to treat as target
            stride:      frame stride (1 = every frame, 2 = every other, etc.)
        """
        self.davis_root  = Path(davis_root)
        self.anno_root   = self.davis_root / "Annotations" / resolution
        self.img_root    = self.davis_root / "JPEGImages"  / resolution
        self.max_frames  = max_frames
        self.obj_id      = obj_id
        self.stride      = stride

        if video_names is not None:
            self.videos = [v for v in video_names if (self.img_root / v).is_dir()]
        else:
            split_file = self.davis_root / "ImageSets" / "2017" / f"{split}.txt"
            if split_file.exists():
                with open(split_file) as f:
                    self.videos = [line.strip() for line in f if line.strip()]
            else:
                # Fall back: list all directories
                self.videos = sorted(
                    d.name for d in self.img_root.iterdir() if d.is_dir()
                )

        if not self.videos:
            raise ValueError(
                f"No videos found in {self.img_root}. "
                "Check DAVIS_ROOT in config.py and download DAVIS 2017."
            )

        # Build clip index: (video_name, start_frame)
        self.clips: List[Tuple[str, int]] = []
        self._frame_lists: dict = {}  # video_name -> sorted list of frame stems
        self._build_clips()

    def _build_clips(self) -> None:
        for vid in self.videos:
            img_dir  = self.img_root  / vid
            anno_dir = self.anno_root / vid
            if not img_dir.is_dir():
                continue
            frames = sorted(
                p.stem for p in img_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg")
            )
            if not frames:
                continue
            self._frame_lists[vid] = frames

            # Build non-overlapping clips
            n_frames = len(frames)
            clip_len  = self.max_frames * self.stride
            n_clips   = max(1, n_frames // clip_len)
            for c in range(n_clips):
                start = c * clip_len
                if start + self.max_frames * self.stride > n_frames:
                    break
                self.clips.append((vid, start))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        """
        Returns:
            frames: [T, 3, H, W]  float32 [0, 1]
            masks:  [T, 1, H, W]  float32 binary (0/1)
            meta:   dict {'video', 'frame_indices', 'orig_hw'}
        """
        vid, start = self.clips[idx]
        frame_stems = self._frame_lists[vid]
        img_dir  = self.img_root  / vid
        anno_dir = self.anno_root / vid

        frames_list, masks_list, frame_ids = [], [], []

        for i in range(self.max_frames):
            fi = start + i * self.stride
            if fi >= len(frame_stems):
                fi = len(frame_stems) - 1
            stem = frame_stems[fi]

            # Load image
            img_path = img_dir / f"{stem}.jpg"
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            H, W = img.shape[:2]

            # Load mask
            anno_path = anno_dir / f"{stem}.png"
            if anno_path.exists():
                anno = np.array(Image.open(anno_path))
                mask = (anno == self.obj_id).astype(np.float32)
            else:
                mask = np.zeros((H, W), dtype=np.float32)

            frames_list.append(
                torch.from_numpy(img).permute(2, 0, 1)        # [3, H, W]
            )
            masks_list.append(
                torch.from_numpy(mask).unsqueeze(0)            # [1, H, W]
            )
            frame_ids.append(fi)

        frames = torch.stack(frames_list)   # [T, 3, H, W]
        masks  = torch.stack(masks_list)    # [T, 1, H, W]
        H, W   = frames.shape[-2:]

        meta = {
            "video":         vid,
            "frame_indices": frame_ids,
            "orig_hw":       (H, W),
            "start_frame":   start,
        }
        return frames, masks, meta

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_prompt_point(self, mask: torch.Tensor) -> Optional[np.ndarray]:
        """
        Return the centroid of the GT mask as a SAM2 point prompt.
        mask: [1, H, W] binary float tensor.
        Returns: np.ndarray [1, 2] in (x, y) pixel coords, or None.
        """
        m = mask[0].numpy().astype(bool)
        ys, xs = np.where(m)
        if len(ys) == 0:
            return None
        cx, cy = int(xs.mean()), int(ys.mean())
        return np.array([[cx, cy]], dtype=np.float32)


def load_single_video(
    davis_root: str,
    video_name: str,
    resolution: str = "480p",
    obj_id: int = 1,
    max_frames: int = -1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Utility: load all frames and masks for a single DAVIS video as numpy arrays.

    Returns:
        frames: list of [H, W, 3] uint8 RGB arrays
        masks:  list of [H, W]   binary uint8 arrays
        stems:  list of frame stem strings (e.g. '00000')
    """
    img_dir  = Path(davis_root) / "JPEGImages"  / resolution / video_name
    anno_dir = Path(davis_root) / "Annotations" / resolution / video_name

    stems = sorted(
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if max_frames > 0:
        stems = stems[:max_frames]

    frames, masks = [], []
    for stem in stems:
        frame = np.array(Image.open(img_dir / f"{stem}.jpg").convert("RGB"))
        frames.append(frame)

        anno_path = anno_dir / f"{stem}.png"
        if anno_path.exists():
            anno = np.array(Image.open(anno_path))
            mask = (anno == obj_id).astype(np.uint8)
        else:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        masks.append(mask)

    return frames, masks, stems
