from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OLS and logistic regressions on covariate CSV.")
    parser.add_argument("--csv", required=True, help="CSV from compute_covariates.py")
    parser.add_argument("--out-dir", default="", help="Optional directory to save coefficient tables.")
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def coefficient_table(result, index) -> pd.DataFrame:
    params = np.asarray(result.params)
    bse    = np.asarray(result.bse)
    tvals  = np.asarray(result.tvalues)
    pvals  = np.asarray(result.pvalues)
    conf   = result.conf_int()
    if isinstance(conf, np.ndarray):
        ci_low, ci_high = conf[:, 0], conf[:, 1]
    else:
        ci_low  = conf.iloc[:, 0].to_numpy()
        ci_high = conf.iloc[:, 1].to_numpy()
    return pd.DataFrame(
        {"coef": params, "std_err": bse, "z_or_t": tvals,
         "p_value": pvals, "ci_low": ci_low, "ci_high": ci_high},
        index=index,
    )


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "delta_jf_codec" not in df.columns and "\u0394JF_codec" in df.columns:
        df["delta_jf_codec"] = df["\u0394JF_codec"]
    required = [
        "dataset",
        "JF_clean",
        "delta_jf_codec",
        "ring_burden",
        "boundary_dominance",
        "proxy_err",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["dataset_binary"] = (df["dataset"].astype(str).str.upper() == "YTVOS").astype(int)
    df["neg"] = (df["delta_jf_codec"] < 0).astype(int)
    df["log1p_ring_burden"] = np.log1p(df["ring_burden"].clip(lower=0))
    df["log_boundary_dominance"] = np.log(df["boundary_dominance"].clip(lower=1e-8))

    for source, target in (
        ("JF_clean", "z_JF_clean"),
        ("log1p_ring_burden", "z_log1p_ring_burden"),
        ("log_boundary_dominance", "z_log_boundary_dominance"),
        ("proxy_err", "z_proxy_err"),
    ):
        df[target] = zscore(df[source])

    return df


def fit_ols(df: pd.DataFrame, predictors: list[str], name: str) -> tuple[object, pd.DataFrame]:
    X = sm.add_constant(df[predictors], has_constant="add")
    y = df["delta_jf_codec"]
    base = sm.OLS(y, X).fit()
    robust = base.get_robustcov_results(cov_type="HC3")
    table = coefficient_table(robust, index=X.columns)
    print(f"\n=== {name} OLS ===")
    print(table.round(4))
    print(f"R^2={robust.rsquared:.4f}  adj_R^2={robust.rsquared_adj:.4f}  n={int(robust.nobs)}")
    return robust, table


def fit_logistic(df: pd.DataFrame, predictors: list[str], name: str) -> tuple[object, pd.DataFrame, float]:
    X = sm.add_constant(df[predictors], has_constant="add")
    y = df["neg"]
    if y.nunique() < 2:
        raise ValueError(f"{name} logistic requires both classes in neg.")
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit(cov_type="HC3")
    probs = result.predict(X)
    auc = roc_auc_score(y, probs)
    table = coefficient_table(result, index=X.columns)
    print(f"\n=== {name} Logistic (neg = 1[dJF<0]) ===")
    print(table.round(4))
    print(f"AUC={auc:.4f}  n={int(result.nobs)}")
    return result, table, auc


def maybe_save(out_dir: str, filename: str, table: pd.DataFrame) -> None:
    if not out_dir:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path / filename)


def main() -> None:
    args = parse_args()
    df = prepare_dataframe(args.csv)
    print(f"[regression] loaded {len(df)} rows from {args.csv}")
    print(df.groupby("dataset")["delta_jf_codec"].agg(["count", "mean"]).round(4))

    base_predictors = ["z_JF_clean", "dataset_binary"]
    full_predictors = [
        "z_JF_clean",
        "z_log1p_ring_burden",
        "z_log_boundary_dominance",
        "z_proxy_err",
        "dataset_binary",
    ]

    _, ols_base_table = fit_ols(df, base_predictors, "Base")
    _, ols_full_table = fit_ols(df, full_predictors, "Full")
    _, logit_base_table, _ = fit_logistic(df, base_predictors, "Base")
    _, logit_full_table, _ = fit_logistic(df, full_predictors, "Full")

    maybe_save(args.out_dir, "ols_base.csv", ols_base_table)
    maybe_save(args.out_dir, "ols_full.csv", ols_full_table)
    maybe_save(args.out_dir, "logit_base.csv", logit_base_table)
    maybe_save(args.out_dir, "logit_full.csv", logit_full_table)


if __name__ == "__main__":
    main()
