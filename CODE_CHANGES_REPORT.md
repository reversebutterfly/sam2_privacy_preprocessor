# Code Changes Report — SAM2 Privacy Preprocessor

Auto-review loop driven improvements. Two review rounds completed.

---

## Round 1 (2026-03-25)

**Reviewer score: 2/10 — Not ready**

Root causes identified:
1. Training/eval task mismatch (image predictor vs. video predictor)
2. Evaluation bias (positive-only averaging)
3. Missing SSIM constraint
4. Mislabelled dataset (not actually "person only")
5. EOT proxy too weak (no YUV420p simulation)

---

## Round 2 Fixes (2026-03-25)

### Fix 1 — `eval_codec.py` lines 311–321: Remove positive-only eval bias

**Before:**
```python
for key in ["jf_clean", "jf_adv", ...]:
    vals = [r[key] for r in results if r.get(key, -1) >= 0]  # ← drops negative deltas
```

**After:**
```python
for key in ["jf_clean", "jf_adv", "delta_jf_adv", "mean_ssim", "mean_psnr"]:
    vals = [r[key] for r in results if key in r]  # ← all videos included
    if vals:
        print(f"  mean_{key}: {np.mean(vals):.4f}")
# codec metrics: only skip hard ffmpeg failures (jf_codec == -1.0)
codec_vals = [r["jf_codec"] for r in results if r.get("jf_codec", -1) != -1.0]
djf_codec_vals = [r["delta_jf_codec"] for r in results if r.get("delta_jf_codec", -999) != -1.0]
if codec_vals:
    print(f"  mean_jf_codec: {np.mean(codec_vals):.4f} (n={len(codec_vals)})")
    print(f"  mean_delta_jf_codec: {np.mean(djf_codec_vals):.4f} (includes negatives)")
```

**Why:** Previous code silently excluded videos where attack was ineffective (negative delta_jf), creating an upward-biased mean that didn't represent true average effectiveness.

---

### Fix 2 — `src/losses.py`: Add differentiable SSIM constraint

**Added (lines 99–160):**
```python
def compute_ssim(x, y, window_size=11):
    """Differentiable SSIM via 11×11 Gaussian sliding window."""
    C1, C2 = 0.01**2, 0.03**2
    # Build Gaussian window, apply per-channel blur
    # Compute mean, variance, covariance maps
    # Return mean SSIM scalar
    ...

class SSIMConstraint(nn.Module):
    """Hinge: max(0, (1 - SSIM) - threshold)"""
    def __init__(self, threshold=0.10):  # SSIM ≥ 0.90
        ...
    def forward(self, x_orig, x_adv):
        ssim_val  = compute_ssim(x_orig, x_adv)
        ssim_loss = 1.0 - ssim_val
        return F.relu(ssim_loss - self.threshold)
```

**Why:** Paper claimed SSIM constraint was enforced, but only LPIPS hinge was implemented. Adding SSIMConstraint makes the perceptual budget claim honest and provides a second complementary constraint.

---

### Fix 3 (Critical) — `train.py`: Clip-level cascaded training

**Root cause of generalization failure:** Training used `build_sam2` (image predictor) with a fresh GT centroid prompt on every frame. Evaluation uses `build_sam2_video_predictor` with first-frame prompting and memory propagation across frames. g_θ learned frame-local artifacts that don't transfer to video tracking.

**Three additions:**

#### 3a. `SAM2Attacker.forward_with_prior()` (lines 161–222)
```python
def forward_with_prior(self, x01, point_coords_np, point_labels_np, prior_mask):
    """SAM2 forward with optional dense mask prompt.

    prior_mask: [1,1,H,W] soft mask from previous frame — fed as dense prompt
    to simulate how VideoPredictor propagates first-frame annotation.
    """
    ...
    if prior_mask is not None:
        mask_input = F.interpolate(prior_mask, size=(256, 256), ...)
        mask_input = mask_input * 20.0 - 10.0  # approx logit scaling
    sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
        points=points_arg, boxes=None, masks=mask_input,
    )
```

#### 3b. `build_clip_pool()` (lines 224–248)
```python
def build_clip_pool(video_names, davis_root, max_frames=30, clip_len=4):
    """Returns overlapping clips of clip_len consecutive frames with stride=clip_len//2."""
```

#### 3c. `train_ours_clip()` (lines ~580–660)
```python
# Frame 0: GT centroid point prompt (matches eval first-frame prompting)
logits = attacker.forward_with_prior(x_adv_0, coords, labels, prior_mask=None)
l_attack += soft_iou_loss(logits, gt[0])
prior_mask = torch.sigmoid(logits).detach()

# Frames 1..N: propagate via prior mask (matches eval memory propagation)
for i in range(1, clip_len):
    logits = attacker.forward_with_prior(x_adv_i, None, None, prior_mask=prior_mask)
    l_attack += soft_iou_loss(logits, gt[i])
    prior_mask = torch.sigmoid(logits).detach()

l_attack = l_attack / clip_len  # normalize
```

**New CLI flags:** `--train_mode clip`, `--clip_len 4`

**Result at step 5000 (C1_clip_s1 on V100):**
| Step | JF_clean | JF_adv | ΔJF |
|------|----------|--------|-----|
| 1000 | 0.591 | 0.283 | 0.307 |
| 2000 | 0.591 | 0.182 | 0.408 |
| 3000 | 0.591 | 0.186 | 0.405 |
| 4000 | 0.591 | 0.181 | 0.409 |
| 5000 | 0.591 | 0.173 | **0.417** |

(Note: quick eval uses image predictor; post-codec video predictor eval pending)

---

### Fix 4 — `config.py`: Correct dataset labelling

**Before:**
```python
# DAVIS person-category train videos (DAVIS 2017 semi-supervised, person only)
DAVIS_TRAIN_VIDEOS_ALL = [
    "bear", ..., "elephant", "flamingo",  # ← NOT person only
]
```

**After:**
```python
# DAVIS 2017 mixed-category videos (generic VOS target suppression benchmark).
# NOTE: despite the earlier "person only" label, the actual split includes animals,
# vehicles, and other objects. Method reframed as generic VOS target suppression.
DAVIS_TRAIN_VIDEOS_ALL = [...]

# Human/person-centric subset (for privacy-specific evaluation)
DAVIS_HUMAN_TRAIN = ["breakdance", "breakdance-flare", "dance-jump",
                     "dance-twirl", "color-run", "bike-packing"]
DAVIS_HUMAN_VAL   = ["bmx-bumps", "bmx-trees", "cat-girl"]
```

---

### Fix 5 — `src/codec_eot.py`: Stronger H.264 proxy (YUV 4:2:0)

**Added `simulate_yuv420p()` (lines 28–58):**
```python
def simulate_yuv420p(x):
    """Differentiable YUV 4:2:0 chroma subsampling — dominant H.264 artifact."""
    # BT.601 RGB → YCbCr
    # Cb/Cr: avg_pool2d (2×2) then bilinear upsample → 4:2:0 effect
    # YCbCr → RGB
    cb_sub = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cb_up  = F.interpolate(cb_sub, size=(H, W), mode="bilinear", ...)
    ...
```

**Updated `codec_proxy_transform()` signature:**
```python
def codec_proxy_transform(x, ..., p_yuv420=0.8):
    # 1. Chroma subsampling (NEW — dominant H.264 artifact)
    if random.random() < p_yuv420:
        out = simulate_yuv420p(out)
    # 2. Low-pass blur  (existing)
    # 3. Quantisation noise  (existing)
    # 4. Spatial scale rounding  (existing)
```

**Why:** H.264 always encodes in YUV 4:2:0, halving chroma resolution. The old proxy (blur+noise+resize) did not capture this, so g_θ learned perturbations that survived the proxy but not real codec compression.

---

## Summary Table

| File | Change | Status |
|------|--------|--------|
| `eval_codec.py` | Remove positive-only filter | ✅ Done |
| `src/losses.py` | Add `SSIMConstraint` + `compute_ssim` | ✅ Done |
| `train.py` | Add `forward_with_prior`, `build_clip_pool`, `train_ours_clip` | ✅ Done |
| `config.py` | Fix dataset labels, add human-only splits | ✅ Done |
| `src/codec_eot.py` | Add `simulate_yuv420p`, update `codec_proxy_transform` | ✅ Done |

## Experiment Status

| Run | Description | Status |
|-----|-------------|--------|
| C1_clip_s1 | Clip-level Stage 1, 5000 steps, 20 videos | ✅ Complete (ΔJF=0.417) |
| C1_clip_eval | Post-codec eval on 9 held-out videos (CRF 18/23/28) | 🔄 Running |
| C2_clip_s3 | Clip-level Stage 3 + codec EOT from C1 checkpoint | ⏳ Pending C1_eval |
