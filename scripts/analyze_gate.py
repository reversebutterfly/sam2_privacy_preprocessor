"""Analyze ring_burden gate performance on YT-VOS."""
import pandas as pd
import numpy as np
import json
import pathlib

df = pd.read_csv("results_v100/covariates.csv")
df = df[df["JF_clean"] >= 0.3]
yt = df[df["dataset"] == "YTVOS"].copy()

print(f"Total YT-VOS: {len(yt)}")
print(f"Overall: mean_delta={yt['delta_jf_codec'].mean():.4f} (+{yt['delta_jf_codec'].mean()*100:.1f}pp)  neg_rate={(yt['delta_jf_codec']<0).mean():.2%}")

print("\nring_burden threshold sweep:")
print(f"{'pct':>5}  {'tau':>8}  {'cov':>8}  {'n':>5}  {'mean_delta_pp':>15}  {'neg_rate':>10}")
rows = []
for pct in [50, 60, 70, 75, 80, 85, 90]:
    tau = float(yt["ring_burden"].quantile(pct / 100))
    subset = yt[yt["ring_burden"] >= tau]
    n = len(subset)
    md = float(subset["delta_jf_codec"].mean())
    nr = float((subset["delta_jf_codec"] < 0).mean())
    cov = n / len(yt)
    print(f"  p{pct:02d}  {tau:8.3f}  {cov:8.1%}  {n:5}  {md*100:+15.1f}pp  {nr:10.2%}")
    rows.append({"pct": pct, "tau": tau, "coverage": cov, "n": n, "mean_delta_pp": md*100, "neg_rate": nr})

# Select best: highest coverage that keeps neg_rate <= 0.15
best = None
for row in rows:
    if row["neg_rate"] <= 0.15:
        if best is None or row["coverage"] > best["coverage"]:
            best = row

if best is None:
    best = min(rows, key=lambda r: r["neg_rate"])

p80 = float(yt["ring_burden"].quantile(0.80))
gate_set = yt[yt["ring_burden"] >= p80]
print(f"\nSelected gate (p80): ring_burden >= {p80:.3f}")
print(f"  n={len(gate_set)}, coverage={len(gate_set)/len(yt):.1%}")
print(f"  mean_delta = +{gate_set['delta_jf_codec'].mean()*100:.1f}pp")
print(f"  neg_rate = {(gate_set['delta_jf_codec']<0).mean():.2%}")

result = {
    "gate_feature": "ring_burden",
    "threshold_p80": p80,
    "sweep": rows,
    "gated_p80": {
        "n": len(gate_set),
        "mean_delta_pp": float(gate_set["delta_jf_codec"].mean() * 100),
        "neg_rate": float((gate_set["delta_jf_codec"] < 0).mean()),
        "coverage": float(len(gate_set) / len(yt)),
    },
    "all_ytvos": {
        "n": len(yt),
        "mean_delta_pp": float(yt["delta_jf_codec"].mean() * 100),
        "neg_rate": float((yt["delta_jf_codec"] < 0).mean()),
    },
}
out = pathlib.Path("results_v100/gate/ytvos_rb_gate_final.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(f"Saved -> {out}")
