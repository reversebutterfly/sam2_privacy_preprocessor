from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a threshold or logistic gate and evaluate on held-out data.")
    parser.add_argument("--csv", required=True, help="CSV from compute_covariates.py")
    parser.add_argument("--split-json", default="", help="Optional JSON with {'train': [...], 'test': [...]} video IDs.")
    parser.add_argument("--dataset", default="", help="Optional dataset filter, e.g. YTVOS or DAVIS")
    parser.add_argument("--min-jf-clean", type=float, default=None, help="Optional JF_clean lower bound.")
    parser.add_argument("--mode", choices=["threshold", "logistic"], default="threshold")
    parser.add_argument("--gate-feature", default="boundary_dominance", help="Feature for threshold mode.")
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-neg-rate-target", type=float, default=0.10)
    parser.add_argument("--min-mean-delta", type=float, default=5.0)
    parser.add_argument("--out-json", default="", help="Optional JSON path to save gate summary.")
    return parser.parse_args()


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "delta_jf_codec" not in df.columns and "\u0394JF_codec" in df.columns:
        df["delta_jf_codec"] = df["\u0394JF_codec"]
    required = ["video_id", "dataset", "JF_clean", "delta_jf_codec", "ring_burden", "boundary_dominance", "proxy_err"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.dropna(subset=required).copy()


def apply_filters(df: pd.DataFrame, dataset: str, min_jf_clean: float | None) -> pd.DataFrame:
    if dataset:
        df = df[df["dataset"].astype(str).str.upper() == dataset.upper()].copy()
    if min_jf_clean is not None:
        df = df[df["JF_clean"] >= min_jf_clean].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")
    return df


def load_or_make_split(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.split_json:
        with open(args.split_json, "r", encoding="utf-8") as handle:
            split = json.load(handle)
        train_ids = set(split["train"])
        test_ids = set(split["test"])
        train_df = df[df["video_id"].isin(train_ids)].copy()
        test_df = df[df["video_id"].isin(test_ids)].copy()
    else:
        stratify = None
        if df["dataset"].nunique() > 1:
            stratify = df["dataset"]
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def summarize_partition(df: pd.DataFrame, accepted: pd.Series) -> dict[str, Any]:
    if accepted.sum() == 0:
        return {
            "accepted_videos": 0,
            "coverage": 0.0,
            "mean_delta_jf": float("nan"),
            "negative_rate": float("nan"),
        }
    accepted_df = df.loc[accepted].copy()
    return {
        "accepted_videos": int(len(accepted_df)),
        "coverage": float(len(accepted_df) / len(df)),
        "mean_delta_jf": float(accepted_df["delta_jf_codec"].mean()),
        "negative_rate": float((accepted_df["delta_jf_codec"] < 0).mean()),
    }


def choose_threshold(train_df: pd.DataFrame, feature: str, neg_rate_target: float, min_mean_delta: float) -> tuple[float, dict[str, Any]]:
    candidates = sorted(train_df[feature].dropna().unique().tolist())
    best_tau = None
    best_stats = None
    best_ok_key = None
    best_fallback_key = None
    for tau in candidates:
        accepted = train_df[feature] >= tau
        stats = summarize_partition(train_df, accepted)
        is_valid = stats["negative_rate"] <= neg_rate_target and stats["mean_delta_jf"] >= min_mean_delta
        if is_valid:
            key = (stats["coverage"], stats["mean_delta_jf"], -stats["negative_rate"])
            if best_ok_key is None or key > best_ok_key:
                best_ok_key = key
                best_tau = float(tau)
                best_stats = stats
        else:
            key = (-stats["negative_rate"], stats["coverage"], stats["mean_delta_jf"])
            if best_ok_key is None and (best_fallback_key is None or key > best_fallback_key):
                best_fallback_key = key
                best_tau = float(tau)
                best_stats = stats
    if best_tau is None or best_stats is None:
        raise RuntimeError("No threshold candidate evaluated.")
    return best_tau, best_stats


def choose_logistic_threshold(train_df: pd.DataFrame, neg_rate_target: float, min_mean_delta: float) -> tuple[LogisticRegression, float, dict[str, Any]]:
    features = ["ring_burden", "boundary_dominance", "proxy_err"]
    X = train_df[features].to_numpy(dtype=float)
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True)
    stds[stds < 1e-12] = 1.0
    Xz = (X - means) / stds
    y = (train_df["delta_jf_codec"] >= 0).astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("Logistic gate requires both success and failure cases in the training split.")

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(Xz, y)
    scores = model.predict_proba(Xz)[:, 1]

    best_tau = None
    best_stats = None
    best_ok_key = None
    best_fallback_key = None
    for tau in sorted(np.unique(scores).tolist()):
        accepted = scores >= tau
        stats = summarize_partition(train_df, pd.Series(accepted, index=train_df.index))
        is_valid = stats["negative_rate"] <= neg_rate_target and stats["mean_delta_jf"] >= min_mean_delta
        if is_valid:
            key = (stats["coverage"], stats["mean_delta_jf"], -stats["negative_rate"])
            if best_ok_key is None or key > best_ok_key:
                best_ok_key = key
                best_tau = float(tau)
                best_stats = stats
        else:
            key = (-stats["negative_rate"], stats["coverage"], stats["mean_delta_jf"])
            if best_ok_key is None and (best_fallback_key is None or key > best_fallback_key):
                best_fallback_key = key
                best_tau = float(tau)
                best_stats = stats
    model._feature_means = means.squeeze(0)  # type: ignore[attr-defined]
    model._feature_stds = stds.squeeze(0)    # type: ignore[attr-defined]
    if best_tau is None or best_stats is None:
        raise RuntimeError("No logistic threshold candidate evaluated.")
    return model, best_tau, best_stats


def score_logistic(model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
    features = ["ring_burden", "boundary_dominance", "proxy_err"]
    X = df[features].to_numpy(dtype=float)
    means = model._feature_means.reshape(1, -1)  # type: ignore[attr-defined]
    stds = model._feature_stds.reshape(1, -1)    # type: ignore[attr-defined]
    Xz = (X - means) / stds
    return model.predict_proba(Xz)[:, 1]


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.csv)
    df = apply_filters(df, args.dataset, args.min_jf_clean)
    train_df, test_df = load_or_make_split(df, args)

    print(f"[gate] train={len(train_df)} test={len(test_df)}")
    print(train_df.groupby("dataset")["video_id"].count())

    summary: dict[str, Any] = {
        "mode": args.mode,
        "dataset_filter": args.dataset,
        "min_jf_clean": args.min_jf_clean,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    if args.mode == "threshold":
        tau, train_stats = choose_threshold(
            train_df=train_df,
            feature=args.gate_feature,
            neg_rate_target=args.train_neg_rate_target,
            min_mean_delta=args.min_mean_delta,
        )
        test_stats = summarize_partition(test_df, test_df[args.gate_feature] >= tau)
        summary.update(
            {
                "gate_feature": args.gate_feature,
                "threshold": tau,
                "train": train_stats,
                "test": test_stats,
            }
        )
        print(f"[gate] feature={args.gate_feature} threshold={tau:.6f}")
    else:
        model, tau, train_stats = choose_logistic_threshold(
            train_df=train_df,
            neg_rate_target=args.train_neg_rate_target,
            min_mean_delta=args.min_mean_delta,
        )
        test_scores = score_logistic(model, test_df)
        test_stats = summarize_partition(test_df, pd.Series(test_scores >= tau, index=test_df.index))
        summary.update(
            {
                "features": ["ring_burden", "boundary_dominance", "proxy_err"],
                "threshold": tau,
                "coefficients": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "train": train_stats,
                "test": test_stats,
            }
        )
        print(f"[gate] logistic threshold={tau:.6f}")

    print("[gate] train:", json.dumps(summary["train"], indent=2))
    print("[gate] test:", json.dumps(summary["test"], indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[gate] wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
