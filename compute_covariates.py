from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-video covariates for DAVIS and YouTube-VOS results."
    )
    parser.add_argument(
        "--results-json",
        nargs="+",
        default=[],
        help="One or more results JSON files. Supports raw lists or {'results': [...]} format.",
    )
    parser.add_argument(
        "--davis-results-json",
        default="",
        help="DAVIS results JSON (forced dataset=DAVIS regardless of path).",
    )
    parser.add_argument(
        "--ytvos-results-json",
        default="",
        help="YouTube-VOS results JSON (forced dataset=YTVOS regardless of path).",
    )
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--davis-anno-root", default="data/DAVIS/Annotations/480p")
    parser.add_argument("--davis-frame-root", default="data/DAVIS/JPEGImages/480p")
    parser.add_argument("--ytvos-anno-root", default="data/ytbvos/train/Annotations")
    parser.add_argument("--ytvos-frame-root", default="data/ytbvos/train/JPEGImages")
    parser.add_argument(
        "--ring-width",
        type=int,
        default=24,
        help="Ring width used for covariate geometry. Must match the attack geometry.",
    )
    parser.add_argument(
        "--proxy-blur-sigma",
        type=float,
        default=24.0,
        help="Gaussian sigma for proxy_err. Use the same sigma as the background proxy.",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=1,
        help="Object ID for indexed PNG masks. Ignored for binary masks.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap on frames per video. -1 means all frames.",
    )
    parser.add_argument(
        "--min-annulus-pixels",
        type=int,
        default=256,
        help="Skip frames with too few outer-annulus pixels.",
    )
    return parser.parse_args()


def canonical_dataset(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "davis" in text:
        return "DAVIS"
    if "yt" in text or "youtube" in text or "vos" in text:
        return "YTVOS"
    raise ValueError(f"Unrecognized dataset label: {value!r}")


def pick_first(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def load_results_records(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            results = payload
            dataset_hint = None
        elif isinstance(payload, dict):
            results = payload.get("results", payload.get("videos", []))
            dataset_hint = payload.get("dataset")
        else:
            raise TypeError(f"Unsupported JSON structure in {path}")

        # Infer dataset from file path if not present in JSON
        if not dataset_hint:
            path_lower = str(path).lower()
            if "davis" in path_lower:
                dataset_hint = "DAVIS"
            elif "ytb" in path_lower or "youtube" in path_lower or "ytvos" in path_lower:
                dataset_hint = "YTVOS"

        if not isinstance(results, list):
            raise TypeError(f"Expected list-like results in {path}")

        for item in results:
            if not isinstance(item, dict):
                continue
            video_id = pick_first(item, ("video_id", "video", "name", "id"))
            jf_clean = pick_first(item, ("JF_clean", "jf_clean"))
            delta_jf = pick_first(item, ("\u0394JF_codec", "delta_jf_codec"))
            dataset = pick_first(item, ("dataset",)) or dataset_hint
            if video_id is None or jf_clean is None or delta_jf is None or dataset is None:
                continue
            rows.append(
                {
                    "video_id": str(video_id),
                    "dataset": canonical_dataset(dataset),
                    "JF_clean": float(jf_clean),
                    "delta_jf_codec": float(delta_jf),
                }
            )
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        deduped[(row["dataset"], row["video_id"])] = row
    return list(deduped.values())


def read_mask(mask_path: Path, obj_id: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    # Collapse channels: for DAVIS palette PNGs, object colors span channels
    # (object 1 = BGR(128,0,0), object 2 = BGR(0,0,128), etc.)
    # Taking the max across channels gives a binary-like mask for any single object
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    unique = np.unique(mask)
    nonbg = unique[unique > 0]
    if len(nonbg) == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    # If only one non-background value OR binary/255-binary, treat as binary
    if len(nonbg) == 1 or np.all(np.isin(unique, [0, 1])) or np.all(np.isin(unique, [0, 255])):
        binary = mask > 0
    else:
        # Multi-object: use the max (all non-zero == some object present)
        binary = mask > 0
    return binary.astype(np.uint8)


def read_rgb(frame_path: Path) -> np.ndarray:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise FileNotFoundError(frame_path)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def find_frame_path(frame_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = frame_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def list_mask_paths(anno_dir: Path, max_frames: int) -> list[Path]:
    paths = sorted(p for p in anno_dir.iterdir() if p.suffix.lower() == ".png")
    return paths if max_frames <= 0 else paths[:max_frames]


def build_ring_and_annulus(mask: np.ndarray, ring_width: int) -> tuple[np.ndarray, np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel_w = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), dtype=np.uint8)
    kernel_2w = np.ones((ring_width * 4 + 1, ring_width * 4 + 1), dtype=np.uint8)
    dil_w = cv2.dilate(mask_u8, kernel_w)
    ero_w = cv2.erode(mask_u8, kernel_w)
    dil_2w = cv2.dilate(mask_u8, kernel_2w)
    ring = (dil_w > 0) & (ero_w == 0)
    annulus = (dil_2w > 0) & (dil_w == 0)
    return ring, annulus


def frame_covariates(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    ring_width: int,
    proxy_blur_sigma: float,
    min_annulus_pixels: int,
) -> tuple[float, float, float] | None:
    if mask.sum() == 0:
        return None
    ring, annulus = build_ring_and_annulus(mask, ring_width)
    if annulus.sum() < min_annulus_pixels or ring.sum() == 0:
        return None

    ring_burden = float(ring.sum()) / float(max(mask.sum(), 1))

    y = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) / 255.0
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    boundary_dominance = float(grad_mag[ring].mean()) / float(grad_mag[annulus].mean() + 1e-6)

    y_blur = cv2.GaussianBlur(y, (0, 0), proxy_blur_sigma)
    proxy_err = float(np.abs(y[annulus] - y_blur[annulus]).mean())
    return ring_burden, boundary_dominance, proxy_err


def dataset_roots(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    if dataset == "DAVIS":
        return Path(args.davis_anno_root), Path(args.davis_frame_root)
    if dataset == "YTVOS":
        return Path(args.ytvos_anno_root), Path(args.ytvos_frame_root)
    raise ValueError(dataset)


def compute_video_covariates(
    dataset: str,
    video_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    anno_root, frame_root = dataset_roots(args, dataset)
    anno_dir = anno_root / video_id
    frame_dir = frame_root / video_id
    if not anno_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    ring_burdens: list[float] = []
    boundary_dominances: list[float] = []
    proxy_errs: list[float] = []

    for mask_path in list_mask_paths(anno_dir, args.max_frames):
        frame_path = find_frame_path(frame_dir, mask_path.stem)
        if frame_path is None:
            continue
        mask = read_mask(mask_path, args.obj_id)
        covs = frame_covariates(
            frame_rgb=read_rgb(frame_path),
            mask=mask,
            ring_width=args.ring_width,
            proxy_blur_sigma=args.proxy_blur_sigma,
            min_annulus_pixels=args.min_annulus_pixels,
        )
        if covs is None:
            continue
        ring_burden, boundary_dominance, proxy_err = covs
        ring_burdens.append(ring_burden)
        boundary_dominances.append(boundary_dominance)
        proxy_errs.append(proxy_err)

    if not ring_burdens:
        raise RuntimeError(f"No valid frames for {dataset}:{video_id}")

    return {
        "ring_burden": float(np.median(ring_burdens)),
        "boundary_dominance": float(np.median(boundary_dominances)),
        "proxy_err": float(np.median(proxy_errs)),
        "n_valid_frames": int(len(ring_burdens)),
    }


def load_results_with_forced_dataset(path: str, forced_dataset: str) -> list[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, dict):
        results = payload.get("results", payload.get("videos", []))
    else:
        raise TypeError(f"Unsupported JSON structure in {p}")
    rows = []
    for item in results:
        if not isinstance(item, dict):
            continue
        video_id = pick_first(item, ("video_id", "video", "name", "id"))
        jf_clean = pick_first(item, ("JF_clean", "jf_clean"))
        delta_jf = pick_first(item, ("\u0394JF_codec", "delta_jf_codec"))
        if video_id is None or jf_clean is None or delta_jf is None:
            continue
        rows.append({
            "video_id": str(video_id),
            "dataset": forced_dataset,
            "JF_clean": float(jf_clean),
            "delta_jf_codec": float(delta_jf),
        })
    return rows


def main() -> None:
    args = parse_args()
    records = load_results_records(args.results_json)
    if args.davis_results_json:
        records += load_results_with_forced_dataset(args.davis_results_json, "DAVIS")
    if args.ytvos_results_json:
        records += load_results_with_forced_dataset(args.ytvos_results_json, "YTVOS")
    # Deduplicate
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        deduped[(r["dataset"], r["video_id"])] = r
    records = list(deduped.values())
    print(f"[covariates] loaded {len(records)} unique video records")

    output_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        dataset = record["dataset"]
        video_id = record["video_id"]
        print(f"[covariates] {idx:04d}/{len(records):04d} {dataset}:{video_id}")
        try:
            covs = compute_video_covariates(dataset, video_id, args)
        except Exception as exc:
            print(f"  [skip] {dataset}:{video_id} -> {exc}")
            continue
        output_rows.append(
            {
                "video_id": video_id,
                "dataset": dataset,
                "JF_clean": record["JF_clean"],
                "\u0394JF_codec": record["delta_jf_codec"],
                "delta_jf_codec": record["delta_jf_codec"],
                "ring_burden": covs["ring_burden"],
                "boundary_dominance": covs["boundary_dominance"],
                "proxy_err": covs["proxy_err"],
                "n_valid_frames": covs["n_valid_frames"],
            }
        )

    if not output_rows:
        raise SystemExit("No rows were computed. Check JSON fields and dataset roots.")
    output_rows.sort(key=lambda r: (r["dataset"], r["video_id"]))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "dataset", "JF_clean", "\u0394JF_codec", "delta_jf_codec",
                  "ring_burden", "boundary_dominance", "proxy_err", "n_valid_frames"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[covariates] wrote {len(output_rows)} rows -> {out_path}")
    by_dataset: dict[str, int] = {}
    for r in output_rows:
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
    for ds, cnt in sorted(by_dataset.items()):
        print(f"  {ds}: {cnt}")


if __name__ == "__main__":
    main()
