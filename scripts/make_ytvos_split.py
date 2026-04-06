"""Generate a stratified 50/50 train/test split for YT-VOS transfer experiment.

Uses existing ytbvos_combo_strong_v1 results to get video IDs,
stratifies by jf_clean (4 quartile bins), saves split JSON.

Usage:
  python scripts/make_ytvos_split.py \
    --results-json results_v100/ytbvos/ytbvos_combo_strong_v1/results.json \
    --out-json results_v100/transfer/ytvos_split_seed0.json \
    --test-size 0.5 \
    --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--test-size", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-jf-clean", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.results_json) as f:
        data = json.load(f)

    records = [r for r in data["results"] if r.get("jf_clean", 0) >= args.min_jf_clean]
    print(f"Total valid videos: {len(records)}")

    # Stratify by jf_clean quartiles
    jf_values = np.array([r["jf_clean"] for r in records])
    quartiles = np.percentile(jf_values, [25, 50, 75])
    strata = np.digitize(jf_values, quartiles)  # 0,1,2,3

    train_ids, test_ids = [], []
    for s in range(4):
        indices = np.where(strata == s)[0]
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * args.test_size))
        test_ids.extend([records[i]["video"] for i in indices[:n_test]])
        train_ids.extend([records[i]["video"] for i in indices[n_test:]])

    print(f"Split: train={len(train_ids)}, test={len(test_ids)}")

    # Verify no overlap
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap!"

    out = {
        "seed": args.seed,
        "test_size": args.test_size,
        "min_jf_clean": args.min_jf_clean,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train": sorted(train_ids),
        "test": sorted(test_ids),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved split -> {out_path}")

    # Print comma-separated IDs for use in pilot_ytbvos.py --videos arg
    print(f"\nTRAIN IDs (comma-separated):")
    print(",".join(train_ids[:5]), "... [truncated]")


if __name__ == "__main__":
    main()
