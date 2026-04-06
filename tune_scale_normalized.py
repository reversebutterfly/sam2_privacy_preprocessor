from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally execute a scale-normalized rho/alpha sweep manifest."
    )
    parser.add_argument("--results-json", nargs="+", required=True)
    parser.add_argument("--preprocessor-script", default="pilot_mask_guided.py")
    parser.add_argument("--save-dir", default="results_v100/scale_norm_sweep")
    parser.add_argument("--prompt", default="mask", choices=["mask", "point"])
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--min-jf-clean", type=float, default=0.9)
    parser.add_argument("--rhos", default="0.06,0.10,0.14")
    parser.add_argument("--alphas", default="0.6,0.8,0.9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extra-args", default="", help="Raw extra args appended to every preprocessor command.")
    return parser.parse_args()


def pick_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def canonical_dataset(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "davis" in text:
        return "DAVIS"
    if "yt" in text or "youtube" in text or "vos" in text:
        return "YTVOS"
    raise ValueError(f"Unknown dataset label: {value!r}")


def load_rows(paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload if isinstance(payload, list) else payload.get("results", [])
        dataset_hint = payload.get("dataset") if isinstance(payload, dict) else None
        for item in results:
            rows.append(
                {
                    "video_id": str(pick_first(item, ("video_id", "video", "name", "id"))),
                    "dataset": canonical_dataset(pick_first(item, ("dataset",)) or dataset_hint),
                    "JF_clean": float(pick_first(item, ("JF_clean", "jf_clean"))),
                }
            )
    df = pd.DataFrame(rows).dropna().drop_duplicates(subset=["dataset", "video_id"])
    return df


def make_split(df: pd.DataFrame, dataset: str, seed: int, min_jf_clean: float | None = None) -> tuple[list[str], list[str]]:
    sub = df[df["dataset"] == dataset].copy()
    if min_jf_clean is not None:
        sub = sub[sub["JF_clean"] >= min_jf_clean].copy()
    if sub.empty:
        raise ValueError(f"No rows available for dataset={dataset} min_jf_clean={min_jf_clean}")

    stratify = None
    if len(sub) >= 8:
        stratify = pd.qcut(sub["JF_clean"], q=min(4, len(sub)), duplicates="drop")
    train_df, test_df = train_test_split(
        sub,
        test_size=0.5,
        random_state=seed,
        stratify=stratify,
    )
    return train_df["video_id"].tolist(), test_df["video_id"].tolist()


def build_command(
    args: argparse.Namespace,
    dataset: str,
    videos: list[str],
    rho: float,
    alpha: float,
    tag: str,
) -> list[str]:
    # TODO: wire these flags into the actual preprocessor once scale-normalized width is implemented there.
    cmd = [
        sys.executable,
        args.preprocessor_script,
        "--prompt", args.prompt,
        "--crf", str(args.crf),
        "--blend_alpha", str(alpha),
        "--device", args.device,
        "--tag", tag,
        "--save_dir", args.save_dir,
        "--videos", ",".join(videos),
        "--ring-width-mode", "scale_norm",
        "--ring-width-rho", str(rho),
    ]
    if dataset == "YTVOS":
        cmd.extend(["--dataset", "ytvos"])
    else:
        cmd.extend(["--dataset", "davis"])
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def main() -> None:
    args = parse_args()
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.results_json)
    davis_train, davis_test = make_split(df, dataset="DAVIS", seed=args.seed, min_jf_clean=None)
    yt_train, yt_test = make_split(df, dataset="YTVOS", seed=args.seed, min_jf_clean=args.min_jf_clean)

    split_json = {
        "DAVIS": {"train": davis_train, "test": davis_test},
        "YTVOS_JF>=0.9": {"train": yt_train, "test": yt_test},
    }
    (out_dir / "splits.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")

    rhos = [float(x) for x in args.rhos.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    manifest_rows: list[dict[str, Any]] = []
    for dataset_name, split_name, videos in (
        ("DAVIS", "train", davis_train),
        ("DAVIS", "test", davis_test),
        ("YTVOS", "train", yt_train),
        ("YTVOS", "test", yt_test),
    ):
        for rho in rhos:
            for alpha in alphas:
                tag = f"{dataset_name.lower()}_{split_name}_rho{rho:.2f}_a{alpha:.2f}"
                cmd = build_command(args, dataset_name, videos, rho, alpha, tag)
                manifest_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "rho": rho,
                        "alpha": alpha,
                        "tag": tag,
                        "videos": ",".join(videos),
                        "command": " ".join(shlex.quote(part) for part in cmd),
                    }
                )
                if args.execute:
                    subprocess.run(cmd, check=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "job_manifest.csv", index=False)
    print(f"[scale_norm] wrote manifest -> {out_dir / 'job_manifest.csv'}")
    print(
        "[scale_norm] next step: parse each output results.json, select the best train config "
        "per dataset under SSIM>=0.92, then evaluate that frozen config on both held-out test splits."
    )


if __name__ == "__main__":
    main()
