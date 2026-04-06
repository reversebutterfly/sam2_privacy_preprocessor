#!/bin/bash
# YT-VOS Tuning/Transfer Experiment
# Runs 3 alternative configs on the TRAIN split to find the best,
# then evaluates on the TEST split for the definitive comparison.
#
# Usage: bash scripts/run_ytvos_transfer.sh
#
# Results saved under results_v100/transfer/

set -e

PYTHON=/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/envs/sam2_privacy_preprocessor/bin/python
FFMPEG=/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/envs/sam2_privacy_preprocessor/bin/ffmpeg
SAVE_DIR=results_v100/transfer
SPLIT_JSON=results_v100/transfer/ytvos_split_seed0.json

cd ~/sam2_privacy_preprocessor

# ─── Step 1: Generate train/test split ───────────────────────────────────────
echo "[step 1] Generating train/test split..."
mkdir -p $SAVE_DIR
$PYTHON scripts/make_ytvos_split.py \
  --results-json results_v100/ytbvos/ytbvos_combo_strong_v1/results.json \
  --out-json $SPLIT_JSON \
  --test-size 0.5 --seed 0 --min-jf-clean 0.3

# Extract train IDs as comma-separated string
TRAIN_IDS=$($PYTHON -c "import json; d=json.load(open('$SPLIT_JSON')); print(','.join(d['train']))")
TEST_IDS=$($PYTHON -c "import json; d=json.load(open('$SPLIT_JSON')); print(','.join(d['test']))")
N_TRAIN=$($PYTHON -c "import json; d=json.load(open('$SPLIT_JSON')); print(d['n_train'])")
N_TEST=$($PYTHON -c "import json; d=json.load(open('$SPLIT_JSON')); print(d['n_test'])")
echo "Train: $N_TRAIN videos, Test: $N_TEST videos"

# ─── Step 2: Run Config B (rw=16, alpha=0.8) on TRAIN ────────────────────────
echo "[step 2a] Config B: rw=16, alpha=0.8 on TRAIN split..."
$PYTHON pilot_ytbvos.py \
  --ytvos_root data/youtube_vos \
  --prompt mask \
  --ring_width 16 --blend_alpha 0.8 \
  --ffmpeg_path $FFMPEG \
  --videos "$TRAIN_IDS" \
  --tag yt_transfer_rw16_a08_train \
  --save_dir $SAVE_DIR \
  2>&1 | tee $SAVE_DIR/yt_transfer_rw16_a08_train.log

# ─── Step 3: Run Config C (rw=32, alpha=0.8) on TRAIN ────────────────────────
echo "[step 2b] Config C: rw=32, alpha=0.8 on TRAIN split..."
$PYTHON pilot_ytbvos.py \
  --ytvos_root data/youtube_vos \
  --prompt mask \
  --ring_width 32 --blend_alpha 0.8 \
  --ffmpeg_path $FFMPEG \
  --videos "$TRAIN_IDS" \
  --tag yt_transfer_rw32_a08_train \
  --save_dir $SAVE_DIR \
  2>&1 | tee $SAVE_DIR/yt_transfer_rw32_a08_train.log

# ─── Step 4: Run Config D (rw=24, alpha=0.6) on TRAIN ────────────────────────
echo "[step 2c] Config D: rw=24, alpha=0.6 on TRAIN split..."
$PYTHON pilot_ytbvos.py \
  --ytvos_root data/youtube_vos \
  --prompt mask \
  --ring_width 24 --blend_alpha 0.6 \
  --ffmpeg_path $FFMPEG \
  --videos "$TRAIN_IDS" \
  --tag yt_transfer_rw24_a06_train \
  --save_dir $SAVE_DIR \
  2>&1 | tee $SAVE_DIR/yt_transfer_rw24_a06_train.log

# ─── Step 5: Analyze train results to find best config ───────────────────────
echo "[step 3] Analyzing train results..."
$PYTHON - << 'PYEOF'
import json, os

split = json.load(open("results_v100/transfer/ytvos_split_seed0.json"))
train_ids = set(split["train"])
test_ids = set(split["test"])

# Default config: slice existing results to train split
default_all = json.load(open("results_v100/ytbvos/ytbvos_combo_strong_v1/results.json"))["results"]
default_train = [x for x in default_all if x["video"] in train_ids and x.get("jf_clean", 0) >= 0.3]
default_test  = [x for x in default_all if x["video"] in test_ids  and x.get("jf_clean", 0) >= 0.3]

def mean_delta(records):
    vals = [x["delta_jf_codec"] for x in records if x.get("delta_jf_codec") is not None and x.get("delta_jf_codec") != -1.0]
    ssims = [x.get("mean_ssim", 0) for x in records if x.get("delta_jf_codec") is not None and x.get("delta_jf_codec") != -1.0]
    return sum(vals)/len(vals) if vals else float("nan"), sum(ssims)/len(ssims) if ssims else float("nan"), len(vals)

configs = {
    "default_rw24_a08": (default_train, default_test),
}

for tag in ["yt_transfer_rw16_a08_train", "yt_transfer_rw32_a08_train", "yt_transfer_rw24_a06_train"]:
    path = f"results_v100/transfer/{tag}/results.json"
    if os.path.exists(path):
        data = json.load(open(path))["results"]
        configs[tag] = (data, None)

print("\n=== TRAIN SPLIT COMPARISON ===")
best_tag = None; best_delta = -999
for tag, (train_recs, _) in configs.items():
    delta, ssim, n = mean_delta(train_recs)
    print(f"{tag:40s}  ΔJF_codec={delta:.4f}  SSIM={ssim:.3f}  n={n}")
    if ssim >= 0.91 and delta > best_delta:
        best_delta = delta; best_tag = tag

print(f"\nBest config on train: {best_tag} (ΔJF_codec={best_delta:.4f})")

# Also print test split default
d, s, n = mean_delta(default_test)
print(f"\n=== DEFAULT ON TEST SPLIT ===")
print(f"default_rw24_a08  ΔJF_codec={d:.4f}  SSIM={s:.3f}  n={n}")

# Save analysis
result = {
    "split_json": "results_v100/transfer/ytvos_split_seed0.json",
    "best_train_config": best_tag,
    "best_train_delta": best_delta,
    "default_test_delta": d,
    "default_test_ssim": s,
    "default_test_n": n,
}
import pathlib; pathlib.Path("results_v100/transfer/train_analysis.json").write_text(json.dumps(result, indent=2))
print("\nSaved train analysis -> results_v100/transfer/train_analysis.json")
PYEOF

# ─── Step 6: Run best config on TEST split ───────────────────────────────────
BEST_TAG=$($PYTHON -c "import json; d=json.load(open('results_v100/transfer/train_analysis.json')); print(d['best_train_config'])")
echo "[step 4] Best config: $BEST_TAG — running on TEST split..."

# Parse best config params from tag name
if [[ "$BEST_TAG" == "yt_transfer_rw16_a08_train" ]]; then
    RW=16; ALPHA=0.8
elif [[ "$BEST_TAG" == "yt_transfer_rw32_a08_train" ]]; then
    RW=32; ALPHA=0.8
elif [[ "$BEST_TAG" == "yt_transfer_rw24_a06_train" ]]; then
    RW=24; ALPHA=0.6
else
    # Default config is best — run default on test for completeness
    RW=24; ALPHA=0.8
fi

TEST_TAG="yt_transfer_best_test_rw${RW}_a$(echo $ALPHA | tr -d '.')"
echo "Running rw=$RW alpha=$ALPHA on test split -> $TEST_TAG"
$PYTHON pilot_ytbvos.py \
  --ytvos_root data/youtube_vos \
  --prompt mask \
  --ring_width $RW --blend_alpha $ALPHA \
  --ffmpeg_path $FFMPEG \
  --videos "$TEST_IDS" \
  --tag $TEST_TAG \
  --save_dir $SAVE_DIR \
  2>&1 | tee $SAVE_DIR/${TEST_TAG}.log

# ─── Step 7: Final comparison table ─────────────────────────────────────────
echo "[step 5] Final comparison table..."
$PYTHON - << 'PYEOF'
import json, os, pathlib

split = json.load(open("results_v100/transfer/ytvos_split_seed0.json"))
test_ids = set(split["test"])

# Default on test (sliced from existing)
default_all = json.load(open("results_v100/ytbvos/ytbvos_combo_strong_v1/results.json"))["results"]
default_test = [x for x in default_all if x["video"] in test_ids and x.get("jf_clean",0)>=0.3]

def stats(records):
    vals = [x["delta_jf_codec"] for x in records if x.get("delta_jf_codec") not in (None, -1.0)]
    ssims = [x.get("mean_ssim",0) for x in records if x.get("delta_jf_codec") not in (None, -1.0)]
    neg = sum(1 for v in vals if v < 0)
    return {
        "mean_delta": sum(vals)/len(vals) if vals else float("nan"),
        "mean_ssim": sum(ssims)/len(ssims) if ssims else float("nan"),
        "n": len(vals),
        "neg_rate": neg/len(vals) if vals else float("nan"),
    }

d_test = stats(default_test)
print("\n=== FINAL TRANSFER EXPERIMENT TABLE ===")
print(f"{'Config':40s}  {'Dataset':12s}  ΔJF_codec   SSIM    n    neg_rate")
print("-"*90)
print(f"{'DAVIS-default (rw=24,a=0.8)':40s}  {'YT-VOS all':12s}  +{4.0:.3f}pp   0.940   497  (from paper)")
print(f"{'DAVIS-default (rw=24,a=0.8)':40s}  {'YT-VOS test':12s}  +{d_test['mean_delta']*100:.1f}pp   {d_test['mean_ssim']:.3f}   {d_test['n']}    {d_test['neg_rate']:.2f}")

# Best config on test
analysis = json.load(open("results_v100/transfer/train_analysis.json"))
best_tag = analysis["best_train_config"]
test_tag_files = [f for f in os.listdir("results_v100/transfer") if "best_test" in f and os.path.isdir(f"results_v100/transfer/{f}")]
if test_tag_files:
    best_test_data = json.load(open(f"results_v100/transfer/{test_tag_files[0]}/results.json"))["results"]
    b_test = stats(best_test_data)
    print(f"{'YT-VOS-tuned best ('+best_tag+')':40s}  {'YT-VOS test':12s}  +{b_test['mean_delta']*100:.1f}pp   {b_test['mean_ssim']:.3f}   {b_test['n']}    {b_test['neg_rate']:.2f}")

# Interpretation
gap = b_test['mean_delta']*100 - d_test['mean_delta']*100 if test_tag_files else 0
print(f"\nGap (tuned vs default on test): {gap:+.1f}pp")
if abs(gap) < 3:
    print("Interpretation: GAP IS CONTENT-CONDITIONED (tuning doesn't help significantly)")
elif gap >= 5:
    print("Interpretation: GAP WAS PARTLY A TUNING ISSUE (tuning recovered {gap:.1f}pp)")
else:
    print("Interpretation: MARGINAL TUNING IMPROVEMENT")

# Save final table
out = {
    "default_ytvos_all": {"mean_delta": 4.0, "ssim": 0.940, "n": 497, "source": "ytbvos_combo_strong_v1"},
    "default_ytvos_test": d_test,
    "best_config": best_tag,
    "best_test": b_test if test_tag_files else None,
    "gap_pp": gap,
}
pathlib.Path("results_v100/transfer/final_comparison.json").write_text(json.dumps(out, indent=2))
print("Saved -> results_v100/transfer/final_comparison.json")
PYEOF

echo ""
echo "=== Transfer experiment COMPLETE ==="
echo "Results: results_v100/transfer/"
echo "Key file: results_v100/transfer/final_comparison.json"
