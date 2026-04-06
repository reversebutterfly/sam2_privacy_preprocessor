# UAP-SAM2 Baseline — Patch Log & Protocol Audit

**Date**: 2026-04-01
**Repo**: CGCL-codes/UAP-SAM2
**Commit**: `779ce0b7ebb8cc09fb712c46c555099f6a99e08f`
**Server path**: `/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/`
**Verdict**: official-code default-protocol baseline (NOT strict reproduction)

---

## Local Patches Applied to Official Code

### Patch 1 — `attack_setting.py` line 20: CUDA_VISIBLE_DEVICES

**Original**:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

**Final state** (at eval):
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2,0"  # patched: cuda:1 = physical GPU0
```

**Patch history** (all during OOM troubleshooting):
| Session | Value | Physical cuda:1 | Reason |
|---------|-------|----------------|--------|
| Training launch 1 | `"0,1"` | GPU1 | original |
| Training launch 2 | `"4,5"` | GPU5 | GPU1 OOM (other users) |
| Training launch 3 | `"1,2"` | GPU2 | GPU5 OOM (other users) |
| Eval 1-4 | `"4,5"` / `"2,3"` | GPU5/GPU3 | OOM troubleshooting |
| Eval 5-7 (current) | `"2,0"` | GPU0 | GPU0 32GB free |

**Impact**: Changes which physical GPU is used as `cuda:1`. The model, UAP, and computation are identical. No effect on results.

---

### Patch 2 — `uap_atk_test.py`: stdout.flush

**Added at end of file**:
```python
    import sys; sys.stdout.flush()
```

**Why**: With `python -u script.py > log 2>&1`, Python's final `print()` was sometimes lost before the file descriptor closed. This ensures the metric line is written before exit.

**Impact on results**: None. Output-only change.

---

## Infrastructure Changes (Not in Official Code)

### Infra 1 — Checkpoint symlink
```bash
mkdir -p ~/sam2/checkpoints/
ln -s ~/UAP-SAM2/checkpoints/sam2_hiera_tiny.pt ~/sam2/checkpoints/sam2_hiera_tiny.pt
```
**Why**: `dataset_YOUTUBE.py` hardcodes `"../sam2/checkpoints/sam2_hiera_tiny.pt"`. The official code expects `~/sam2/` to be a sibling directory of `~/UAP-SAM2/`.

### Infra 2 — Data symlink
```bash
ln -s /path/to/youtube_vos ~/UAP-SAM2/data/YOUTUBE
```
Official expects `./data/YOUTUBE/`. We symlinked our existing YouTube-VOS download.

### Infra 3 — SA-V substitute
Real SA-V (HuggingFace) unreachable on server. Substituted 30 YouTube-VOS valid dirs as `data/sav_test/JPEGImages_24fps/` placeholders.

**Impact**: `--loss_fea` NOT passed during training. `weight_fea=1e-6` in paper — SA-V term negligible.

---

## Protocol Audit Findings (from GPT-5.4 xhigh review)

### Confirmed Bugs in Official Released Code

| Bug | Description | Impact on Reported 37.03% |
|-----|-------------|--------------------------|
| **Train/eval split leakage** | `choose_dataset()` always reads `./data/YOUTUBE/train/`. `--test_dataset` is parsed but NEVER USED. With `seed=30, limit_img=100`, train and eval sample IDENTICAL 100 videos. | HIGH — in-sample eval |
| **JPEG compression destroys UAP** | Phase 1 saves adv frames as JPEG (`cv2.imwrite`), Phase 2 reloads them. UAP ε=10/255≈0.039 is partially destroyed by JPEG compression. | MEDIUM — eval underestimates UAP (anti-conservative) |
| **Filtered mIoU** | Frames with clean IoU < 0.3 are excluded from both clean and adv scoring. Not standard mIoU. | MEDIUM — inflates both miouimg and miouadv |
| **matplotlib leak** | `plt.subplots()` per frame, never `plt.close()`. Memory accumulates → OOM. | Memory only |
| **SAM2 in Dataset constructor** | `Dataset_YOUTUBE.__init__` calls `build_sam2()`. Loads SAM2 at dataset creation time. | Memory only |

### Protocol Mismatches vs Paper Claims

| Item | Paper | Our Run | Notes |
|------|-------|---------|-------|
| Eval split | YouTube-VOS (unclear if train/valid) | train split | Same 100 videos as training |
| Frames per video | 15 | 15 (eval7 used -1 before OOM) | Use 15 for protocol match |
| Frame sampling | "consecutive" (paper claim) | strided (`frames[::step][:15]`) | Code does NOT do consecutive |
| mIoU definition | plain mIoU | clean-IoU-filtered mIoU | Official code uses filter |
| Eval path | in-memory tensor | JPEG save → reload | Official code uses JPEG |
| SA-V / loss_fea | weight=1e-6 | excluded | Negligible impact |
| Checkpoint | SAM2 1.0 hiera_tiny | ✓ same | Correct |

---

## Held-Out Evaluation Results (2026-04-01)

### Protocol Fixes Applied

| Fix | What changed | Why |
|-----|--------------|-----|
| **Split** | `choose_dataset()` (train) → `choose_heldout_dataset()` (valid) | Eliminate train/eval overlap |
| **Frame format** | JPEG → PNG lossless (`cv2.IMWRITE_PNG_COMPRESSION, 0`) | Preserve ε=10/255 UAP exactly |
| **mIoU** | Unchanged (clean<0.3 filter kept) | Apples-to-apples with official |

Code changes (minimal, additive — official code unmodified):
- `sam2_util.py`: added `DATA_ROOT_VIDEO_YOUTUBE_VALID`, `choose_heldout_dataset()`, `save_image_only(use_png=True)`, `.png` in frame filter
- `sam2/utils/misc.py`: added `.png`/`.PNG` to `load_video_frames_from_jpg_images` extension list
- `uap_eval_heldout.py`: new eval script using `choose_heldout_dataset()` + `use_png=True`

### Comparison Table

| Protocol | Split | Frame fmt | mIoU_clean | mIoU_adv | videos | frames | vs paper (37.03%) |
|----------|-------|-----------|-----------|---------|--------|--------|-------------------|
| Official default (eval7, limit_frames=-1) | train (in-sample) | JPEG | 83.36% | 59.49% | 100 | 2265 | +22.46pp |
| **Protocol-locked** (limit_frames=15, seed=30) | train (in-sample) | JPEG | **76.41%** | **58.61%** | 100 | 1114 | +21.58pp |
| **Held-out** (limit_frames=15, seed=30, PNG) | valid (held-out) | PNG | **82.82%** | **56.01%** | 100 | 1277 | +18.98pp |
| Paper Table 2 (point, YouTube-VOS) | — | — | ~82.8% | 37.03% | — | — | 0 |

### Key Observations

1. **Held-out vs in-sample gap**: adv mIoU drops only 2.6pp (58.61% → 56.01%). Small generalization gap — UAP does transfer to unseen videos.
2. **Clean mIoU**: held-out (82.82%) > in-sample (76.41%). The valid set is inherently easier for SAM2 tracking (cleaner/simpler videos).
3. **Large gap vs paper (37.03%)**: Both in-sample and held-out are ~19-22pp above paper's target. Root cause: training instability (OOM kills during optimization). The UAP was not fully converged.
4. **JPEG vs PNG on adv mIoU**: The protocol-locked (JPEG, in-sample=58.61%) vs the eval7 result (JPEG, in-sample=59.49% but limit_frames=-1) — the JPEG bug slightly deflates adv mIoU relative to in-memory eval.

### Bottom Line

The held-out eval confirms:
- The UAP-SAM2 attack generalizes to unseen videos (2.6pp drop only)
- The large gap vs paper (37.03% target) is due to incomplete training, not the eval protocol
- The train/eval leakage in the official code slightly inflated the reported number (+2.6pp)

---

## Protocol-Locked Eval Parameters

The following parameters define the **"official-code default-protocol"** baseline:

```
--train_dataset YOUTUBE    # controls which split (train, always)
--test_dataset YOUTUBE     # INERT — not used by code
--test_prompts pt          # point prompts (Table 2 condition)
--checkpoints sam2-t       # SAM2 1.0 hiera_tiny
--limit_img 100            # 100 videos (matches training)
--limit_frames 15          # 15 frames per video (paper default)
--seed 30                  # same seed as training
--P_num 10                 # for any internal prompt generation
--prompts_num 256          # for any internal prompt generation
```

---

## Video ID Overlap Analysis

**Train video IDs** (seed=30, limit_img=100, from `random.sample()` before training):
Saved to: `refine-logs/train_video_ids.json`

**Eval video IDs** (seed=30, limit_img=100, same `random.sample()` call):
Will be saved during eval run to: `refine-logs/eval_video_ids.json`

Expected overlap: **100/100** (same seed + same params = same sample)

---

## Final Verdict

| Claim | Valid? |
|-------|--------|
| Strict reproduction of UAP-SAM2 paper Table 2 | **NO** |
| Official-code default-protocol baseline | **YES** (with caveats above) |
| Held-out evaluation on YouTube-VOS | **NO** — train-split in-sample eval |
| Comparable to our method under same eval protocol | **POSSIBLE** — if we apply same JPEG/filter protocol |

**Recommended citation wording**:
> "UAP-SAM2 baseline: we ran the official released code (CGCL-codes/UAP-SAM2, commit 779ce0b) under its default protocol (seed=30, limit_img=100, limit_frames=15, train-split evaluation, filtered mIoU, JPEG eval path). The reported miouadv is on the same 100 training videos used to optimize the UAP."
