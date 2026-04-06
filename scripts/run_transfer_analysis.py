"""
MG-TRANS: Analyse train split results, pick best config, run on test split.
Run: python scripts/run_transfer_analysis.py
"""
import json
import os
import pathlib
import subprocess
import sys

BASE = pathlib.Path("results_v100/transfer")
SPLIT = BASE / "ytvos_split_seed0.json"
DEFAULT_ALL = pathlib.Path("results_v100/ytbvos/ytbvos_combo_strong_v1/results.json")

split = json.loads(SPLIT.read_text())
train_ids = set(split["train"])
test_ids  = set(split["test"])

default_all = json.loads(DEFAULT_ALL.read_text())["results"]
default_train = [x for x in default_all if x["video"] in train_ids and x.get("jf_clean", 0) >= 0.3]
default_test  = [x for x in default_all if x["video"] in test_ids  and x.get("jf_clean", 0) >= 0.3]


def stats(records):
    vals  = [x["delta_jf_codec"] for x in records
             if x.get("delta_jf_codec") not in (None, -1.0)]
    ssims = [x.get("mean_ssim", 0) for x in records
             if x.get("delta_jf_codec") not in (None, -1.0)]
    neg   = sum(1 for v in vals if v < 0)
    n = len(vals)
    return dict(
        mean     = sum(vals)  / n if n else float("nan"),
        ssim     = sum(ssims) / n if n else float("nan"),
        n        = n,
        neg_rate = neg / n if n else float("nan"),
    )


configs = {"default_rw24_a08": stats(default_train)}
for tag in ["yt_transfer_rw16_a08_train",
            "yt_transfer_rw32_a08_train",
            "yt_transfer_rw24_a06_train"]:
    path = BASE / tag / "results.json"
    if path.exists():
        data = json.loads(path.read_text())
        records = data.get("results", data if isinstance(data, list) else [])
        configs[tag] = stats(records)
    else:
        print(f"[warn] missing: {path}")

print("\n=== TRAIN SPLIT COMPARISON ===")
print(f"{'Config':<45}  {'ΔJF_codec':>10}  {'SSIM':>6}  {'n':>4}  neg")
print("-" * 80)
best_tag   = "default_rw24_a08"
best_delta = configs["default_rw24_a08"]["mean"]
for tag, s in configs.items():
    marker = " ← BEST" if tag == best_tag else ""
    print(f"{tag:<45}  {s['mean']*100:>+8.2f}pp  {s['ssim']:>6.3f}  {s['n']:>4}  {s['neg_rate']:.2f}{marker}")
    if s["mean"] > best_delta and s["ssim"] >= 0.91:
        best_delta = s["mean"]
        best_tag   = tag

# Re-mark best
print()
print(f"Best on train: {best_tag}  (ΔJF={best_delta*100:.2f}pp)")

d_test = stats(default_test)
print(f"\n=== DEFAULT (rw=24,a=0.8) ON TEST SPLIT ===")
print(f"  ΔJF_codec={d_test['mean']*100:.2f}pp  SSIM={d_test['ssim']:.3f}"
      f"  n={d_test['n']}  neg_rate={d_test['neg_rate']:.2f}")

result = {
    "best_train_config":   best_tag,
    "best_train_delta_pp": round(best_delta * 100, 3),
    "default_test": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in d_test.items()},
    "all_train": {
        k: {kk: round(vv, 4) if isinstance(vv, float) else vv
            for kk, vv in v.items()}
        for k, v in configs.items()
    },
}
(BASE / "train_analysis.json").write_text(json.dumps(result, indent=2))
print(f"\nSaved -> {BASE}/train_analysis.json")

# ── Now run best config on test split ───────────────────────────────────────
param_map = {
    "yt_transfer_rw16_a08_train": (16, 0.8),
    "yt_transfer_rw32_a08_train": (32, 0.8),
    "yt_transfer_rw24_a06_train": (24, 0.6),
    "default_rw24_a08":           (24, 0.8),
}
rw, alpha = param_map.get(best_tag, (24, 0.8))
test_tag  = f"yt_transfer_best_test_rw{rw}_a{str(alpha).replace('.','')}"

TEST_IDS = ",".join(split["test"])
FFMPEG   = "/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/envs/sam2_privacy_preprocessor/bin/ffmpeg"
PYTHON   = sys.executable

print(f"\n=== RUNNING BEST CONFIG ON TEST SPLIT ===")
print(f"  Config: rw={rw}, alpha={alpha}  ->  {test_tag}")
print(f"  n_test = {len(split['test'])} videos")

cmd = [
    PYTHON, "pilot_ytbvos.py",
    "--ytvos_root", "data/youtube_vos",
    "--prompt", "mask",
    "--ring_width", str(rw),
    "--blend_alpha", str(alpha),
    "--ffmpeg_path", FFMPEG,
    "--videos", TEST_IDS,
    "--tag", test_tag,
    "--save_dir", str(BASE),
]
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ret = subprocess.run(cmd)
print(f"exit_code={ret.returncode}")

# ── Final comparison table ───────────────────────────────────────────────────
test_path = BASE / test_tag / "results.json"
if test_path.exists():
    best_test_data = json.loads(test_path.read_text())
    best_test_recs = best_test_data.get("results",
                         best_test_data if isinstance(best_test_data, list) else [])
    b_test = stats(best_test_recs)

    gap = b_test["mean"] - d_test["mean"]
    print("\n=== FINAL TRANSFER TABLE ===")
    print(f"{'Config':<45}  {'Dataset':<12}  {'ΔJF_codec':>10}  {'SSIM':>6}  {'n':>4}  neg")
    print("-" * 90)
    print(f"{'DAVIS-default (rw=24,a=0.8)':<45}  {'YT-VOS all':<12}  {4.0:>+8.1f}pp  {'0.940':>6}  497")
    print(f"{'DAVIS-default (rw=24,a=0.8)':<45}  {'YT-VOS test':<12}  {d_test['mean']*100:>+8.2f}pp"
          f"  {d_test['ssim']:>6.3f}  {d_test['n']:>4}  {d_test['neg_rate']:.2f}")
    print(f"{best_tag+' → test':<45}  {'YT-VOS test':<12}  {b_test['mean']*100:>+8.2f}pp"
          f"  {b_test['ssim']:>6.3f}  {b_test['n']:>4}  {b_test['neg_rate']:.2f}")
    print(f"\n  Gap (tuned vs default on test): {gap*100:+.2f}pp")
    if abs(gap * 100) < 2:
        interp = "GAP IS CONTENT-CONDITIONED — tuning does not help"
    elif gap * 100 >= 5:
        interp = f"TUNING PARTIALLY RECOVERED {gap*100:.1f}pp — some parameter bias"
    else:
        interp = "MARGINAL TUNING IMPROVEMENT"
    print(f"  Interpretation: {interp}")

    final = {
        "default_ytvos_all":  {"mean_delta_pp": 4.0, "ssim": 0.940, "n": 497},
        "default_ytvos_test": {k: round(v,4) if isinstance(v,float) else v
                               for k,v in d_test.items()},
        "best_config":        best_tag,
        "best_test": {k: round(v,4) if isinstance(v,float) else v
                      for k,v in b_test.items()},
        "gap_pp":             round(gap * 100, 3),
        "interpretation":     interp,
    }
    (BASE / "final_comparison.json").write_text(json.dumps(final, indent=2))
    print(f"\nSaved -> {BASE}/final_comparison.json")
