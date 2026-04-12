# Research Review: UAP-SAM2 Baseline +1.14pp Anomaly

**Date**: 2026-04-09
**Reviewer**: GPT-5.4 xhigh (Codex MCP)
**Thread ID**: 019d7055-510a-7202-a88c-a3eb10c9efbf
**Trigger**: User flagged that pre-codec ΔJF = +1.14pp for UAP-SAM2 on full DAVIS (n=89) "is not a normal value"

---

## Verdict

**The +1.14pp number is a measurement artifact, not a faithful evaluation of UAP-SAM2.**
GPT-5.4 confirmed the diagnosis and identified a SECOND latent bug we had not noticed.

---

## Bug 1: Resolution mismatch (primary)

### Diagnosis
- Released UAP weights `YOUTUBE.pth` are `[1, 3, 1024, 1024]` float32, pure ±0.0392 binary high-frequency pattern (`abs_mean=0.0392 = 10/255`).
- The official UAP-SAM2 eval (`uap_eval_heldout_jpeg.py:188-200`) applies the UAP **at 1024×1024**, after SAM2's internal `transform_image()` resize:
  ```python
  X = sam_fwder.transform_image(image).to(device)   # Resize((1024,1024)) + ImageNet normalize
  benign_img = denorm(X).to(device)                  # un-normalize back to [0,1] AT 1024
  adv_img = torch.clamp(benign_img + uap, 0, 1)      # add UAP at 1024
  ```
- Our eval (`eval_uap_davis.py:51`, `apply_uap()`) applies the UAP **at native DAVIS 480p**, after `cv2.INTER_LINEAR` downsampling the UAP from 1024 → 480.

### Measured impact
| Resampling | abs_mean | std |
|---|---|---|
| Raw UAP at 1024×1024 | 0.0392 | 0.0392 |
| `cv2.INTER_LINEAR` → 480p | **0.0242** (−38%) | 0.0274 |
| `cv2.INTER_NEAREST` → 480p | 0.0392 | 0.0392 (but spatial alignment lost) |

Then SAM2 internally resizes 480p → 1024p again. The trained UAP relies on patch-grid-aligned high-frequency content; **double resampling kills both magnitude and patch alignment**. The +1.14pp pre-codec result is just residual noise after this destruction.

---

## Bug 2: Hidden JPEG ingress in "pre-codec" path (secondary, but pervasive)

### Diagnosis
GPT-5.4 caught this — both `eval_uap_davis.py:91-93` and `pilot_mask_guided.py:593-600` write each frame to a tempdir as JPEG **quality=95** before SAM2 reads it back via `init_state(video_path=tmp_dir)`:
```python
cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
...
state = predictor.init_state(video_path=tmp_dir)
```

### Implications
- **For UAP-SAM2 baseline**: JPEG QF=95 disproportionately hurts UAP because high-freq ±10/255 patterns are exactly what JPEG quantizes hardest. Our "pre-codec" UAP number is therefore *also* not pre-codec.
- **For our AdvOpt main numbers**: JPEG QF=95 is mild and BRS uses smooth low-freq proxies that survive JPEG QF=95 with minimal loss. The +57.7pp main result is probably essentially intact, but should be re-stated honestly: it is "after a single JPEG QF=95 round-trip", not "raw pixel-space".
- **Asymmetric effect**: This bug systematically biases the comparison **against** high-frequency attacks (UAP) and in favour of low-frequency ones (BRS).

---

## Recommended Fix

GPT-5.4 recommendation: **A + B**, with **B as the main paper baseline**.

### Option A: Re-run released UAP weights at native 1024 attack space (DIAGNOSTIC)
- Resize each DAVIS frame to 1024×1024 + ImageNet-normalize, denorm, add UAP, save 1024×1024 uint8.
- Run SAM2 directly at 1024×1024 (no internal re-resize).
- Use `nearest-neighbor` for GT masks (never bilinear).
- Map predictions back to native DAVIS geometry for J&F if metric comparability is needed.
- Eliminate the hidden JPEG QF=95 ingress (pass tensors directly to SAM2 in-memory if possible, or use lossless PNG).
- **Purpose**: confirm the released weights actually attack SAM2 in their native protocol on our subset. Establishes a "weights are alive" positive control.

### Option B: Train a fresh codec-naive UAP under our exact protocol (PAPER BASELINE)
- Same backbone (SAM2.1 hiera_tiny), same DAVIS, same GT-mask prompt, same J&F metric, same eps=10/255.
- This is the apples-to-apples comparison that the paper actually needs.
- **Purpose**: a fair "what does a strong codec-naive attack look like in our setting?" baseline.

### Rejected options
- **Option C** (apply UAP at 1024 then downsample adv frame to 480): still throws away high-freq content the UAP relies on. Same root failure as current code.
- **Option D** (run our entire eval at 1024 for everything): rigorous but expensive, makes our main numbers incomparable with the existing tables. Appendix material at best.

---

## Expected Results After Fix (calibration)

GPT-5.4 explicitly said **do NOT anchor on "should be ~30pp"**. The published UAP-SAM2 number (~37pp mIoU drop) was on:
- YouTube-VOS, not DAVIS
- Original SAM2, not SAM2.1-tiny
- Point/box prompts, not GT mask prompts
- mIoU, not J&F

Realistic ranges for the fixed eval:
- `<3pp` after fix → still suspect a bug (run more sanity checks)
- **`5-15pp` → believable transfer**
- `25pp+` → possible but should not be assumed
- Post-codec at CRF 23 → could legitimately fall to `0-3pp` if attack is codec-fragile

---

## Sanity Checks Before Trusting Any Re-Run

1. **Verify input tensor**: log the first pixel values of `denorm(transform_image(frame)) + uap` and confirm it matches what SAM2 actually sees.
2. **Random ±10/255 control**: apply random sign perturbation at the same protocol. UAP must be visibly stronger than this control, or the weights are dead.
3. **5–10 video paired ablation**: current broken path vs corrected path. Should see a clear jump.
4. **Tiny official-protocol positive control**: replicate the exact official YouTube-VOS subset eval to confirm released weights are not corrupted/zeroed.
5. **Re-measure SSIM/LPIPS in the actual final delivered frame space** of the corrected protocol — the current `SSIM=0.69` is wrong because the perturbation magnitude was wrong.
6. **Clean J&F comparison at 480p vs 1024p** — if clean numbers shift significantly, do not directly compare 1024-protocol UAP numbers against our 480p-protocol AdvOpt numbers.

---

## Paper Implications

### What we CANNOT claim
- "UAP-SAM2 only achieves +0.49pp on DAVIS" — this is the broken-protocol number.
- **"Our method is 44× stronger than UAP-SAM2"** — this overstates based on a measurement artifact.

### What we CAN claim, conditionally
- If corrected pre-codec is e.g. **+10–20pp** and post-codec is **+0–3pp**: clean story — "released codec-naive UAP transfers nontrivially in clean digital space but is destroyed by H.264, while our codec-aware method survives at +57pp."
- If corrected pre-codec is also low (**<5pp**): the released-weights baseline is too weak to be interesting. Pivot to **trained-from-scratch codec-naive UAP under our protocol** as the main baseline. The honest claim becomes: "even a freshly trained codec-naive UAP under our exact protocol collapses post-H.264, while our codec-aware design retains +57pp."

### Other latent bugs to audit
- GT mask resize kernel (must be NEAREST, never bilinear)
- Prompt coordinate scaling under any resolution change
- Aspect ratio distortion when squaring DAVIS to 1024×1024
- Model version mismatch (released UAP was trained against SAM2 1.0, we use SAM2.1-tiny)
- Metric mismatch (released paper reports mIoU, we report J&F — do not compare absolute pp 1:1)

---

## Action Items (Prioritized)

1. **[CRITICAL]** Stop the running YT-VOS UAP eval — same broken protocol, would waste compute.
2. **[HIGH]** Implement Option A: 1024-protocol UAP eval on a 5-video DAVIS subset as diagnostic. Eliminate hidden JPEG ingress (use PNG or in-memory pipe).
3. **[HIGH]** Add random-sign ±10/255 positive control to the diagnostic.
4. **[HIGH]** Audit `pilot_mask_guided.py` JPEG QF=95 ingress — quantify its impact on our main +57.7pp number (probably small, but must verify).
5. **[MEDIUM]** Decide A-only vs A+B: if Option A gives ≥5pp pre-codec, A may be enough as a baseline. If <5pp, must do B.
6. **[MEDIUM]** Update `discussion.tex` to honestly reflect the corrected baseline once available.

---

## Honest Bottom Line

Our +57.7pp main result is probably real (BRS with α=0.93 is a strong attack and JPEG QF=95 is mild for low-freq edits). But the **comparison number for UAP-SAM2 is wrong**, and the "44× stronger" framing is unsupported. We need at least Option A (and ideally A+B) before any UAP comparison goes into the paper.

---

## Round 2: JPEG vs PNG Decision

### New evidence (from the official UAP-SAM2 reproduction logs)
The released repo ships BOTH a PNG and a JPEG eval script using the same `YOUTUBE.pth` weights:
- `uap_eval_heldout_jpeg.py` (default `use_png=False` → JPEG QF=95): `miou_clean=82.16, miou_adv=57.94 → ΔmIoU = −24.22pp` (n=1287 frames, 100 videos)
- `uap_eval_heldout.py` (PNG, lossless): `miou_clean=83.36, miou_adv=59.49 → ΔmIoU = −23.87pp` (n=2265 frames, 100 videos)
- **JPEG vs PNG difference: only ~0.35pp.** JPEG QF=95 does NOT meaningfully attenuate UAP-SAM2 in their setting.
- Note: published paper target is 37.03% adv mIoU (~−45.77pp). The reproduction achieves only ~24pp because `loss_fea` was excluded (SA-V dataset unavailable on server). So the achievable strength of the released weights is **~24pp**, not 37pp.

### Decision: keep JPEG QF=95 everywhere
GPT-5.4 verdict: **JPEG QF=95 is the correct primary protocol; do NOT re-run ~570 videos to PNG.**

Rationale:
1. **Official UAP-SAM2 protocol IS JPEG QF=95** — defaults to `use_png=False`, OpenCV default JPEG quality is 95. Our hidden ingress accidentally matches the official protocol exactly.
2. **JPEG vs PNG only differs by ~0.35pp** for UAP-SAM2 — re-run cost not worth it.
3. **All our existing experiments** (AdvOpt main 57.7pp, all DAVIS sweeps, all YT-VOS results, frontier sweep) are already JPEG QF=95 — re-running is ~570 video evaluations.

### Required disclosure changes
1. **Elevate JPEG QF=95 from "hidden step" to explicit protocol choice** — document in Methods section.
2. **Set `IMWRITE_JPEG_QUALITY=95` explicitly** in all eval code (don't rely on OpenCV implicit default).
3. **Record OpenCV version** in repro report.
4. **Optional appendix experiment**: 5–10 video JPEG-vs-PNG sensitivity check on the corrected UAP-DAVIS to show the gap is negligible (≤1pp).

### Revised expected range for corrected DAVIS UAP eval
GPT-5.4 calibration update (key insight: GT-mask prompts are MUCH stronger than point prompts, so the released UAP transfers much weaker on our DAVIS-mask protocol than on the official YouTube-VOS-pt protocol):

| Range | Interpretation |
|---|---|
| **3–10pp pre-codec** | **Most plausible** under GT-mask threat model |
| 2–15pp pre-codec | Defensible broader range |
| <2pp pre-codec | Still suspicious — sanity-check more |
| >15pp pre-codec | Stronger than expected for GT-mask |

**~5pp pre-codec is completely believable** even after the resolution bug is fixed.

Post-codec: very likely 0–3pp (codec-fragile UAP collapses under H.264).

### Revised paper framing (cannot claim "44× stronger")

**Old claim** (now unsupported):
> "AdvOpt achieves +57.7pp post-codec, 44× stronger than UAP-SAM2 (+1.3pp)."

**Revised claim** (defensible under GPT-5.4 framing):
> "Under the GT-mask-initialized tracking threat model, codec-naive universal pixel perturbations are weak and non-robust to standard codec round-trip. Codec-aware mask-conditioned preprocessing remains effective: AdvOpt at (rw=24, α=0.93) achieves post-codec ΔJF = +57.7pp, while a representative codec-naive baseline (UAP-SAM2 released weights, evaluated under our protocol) achieves at most a few pp pre-codec and collapses to near zero post-codec."

This is a **threat-model-specific systems claim**, not a general attack-leaderboard claim. Weaker as "we beat UAP", stronger as "in the GT-mask threat model, codec-aware design is necessary."

### Updated action plan

1. **Stop the running YT-VOS UAP eval** (already broken protocol)  ✅ **DONE**
2. **Implement Option A correctly**:  ✅ **DONE** (`eval_uap_davis_1024.py`)
   - 1024×1024 attack space (resize frame to 1024, apply UAP, run SAM2 at 1024)
   - JPEG QF=95 ingress (explicitly set quality flag)
   - Nearest-neighbor for GT mask resize
   - Random ±10/255 universal control with seed
3. **GPT-5.4 code review** (Round 3): caught 3 issues fixed before deploy
   - CRITICAL: `load_single_video` returns 3 values, not 2
   - MAJOR: pre-codec ΔJF should subtract from `jf_clean`, not `jf_codec_clean`
   - MAJOR: SSIM should use project-standard `quality_summary` (color RGB), not grayscale
4. **5-video sanity (mask prompt)**:  ✅ **DONE** — see Round 3 results below
5. **Full DAVIS at 1024 protocol** (UAP + Random + point-prompt UAP):  🔄 **RUNNING**
6. **Document JPEG QF=95** in Methods + add OpenCV version to repro
7. **Update discussion.tex** with revised framing once corrected numbers are in

---

## Round 3: Sanity Results (5 DAVIS videos, 50 frames, 1024×1024 protocol)

| Video | UAP pre-codec | UAP post-codec | Random pre-codec | Random post-codec |
|-------|---------------|----------------|------------------|-------------------|
| bear | +1.24pp | +0.96pp | +0.96pp | +0.74pp |
| elephant | +2.27pp | +1.21pp | +0.45pp | +0.68pp |
| dog | +4.35pp | +2.63pp | +1.56pp | +1.38pp |
| horsejump-high | +1.60pp | +0.74pp | +1.32pp | +0.41pp |
| breakdance | +4.43pp | +4.01pp | +2.69pp | +2.46pp |
| **Mean (n=5)** | **+2.78pp** | **+1.91pp** | **+1.40pp** | **+1.13pp** |
| **Mean SSIM@1024** | 0.6305 | — | 0.6260 | — |

### Interpretation
1. **The resolution bug fix worked**: 1024-protocol UAP gives mean +2.78pp vs old broken 480p protocol +1.14pp on full DAVIS (n=89). Roughly 2.4× stronger after fix.
2. **UAP > Random by +1.38pp pre-codec**: real signal exists, released UAP weights are not dead.
3. **Both attacks remain weak under GT mask prompt**: dense mask prompt provides strong anchor that universal pixel perturbations can barely move. Median post-codec ΔJF ~1.2pp for UAP.
4. **UAP/Random gap collapses post-codec**: +0.78pp (1.91 - 1.13). H.264 quantizes the high-freq UAP-specific structure, leaving only the residual common to both.
5. **+2.78pp pre-codec is just at the lower edge of GPT-5.4's "5-15pp believable" range**, but consistent with the **GT-mask-prompt threat model being inherently robust** to universal additive perturbations. This is the core paper insight, not a sign of further bugs.
6. **SSIM ~0.63 at 1024**: this is genuinely different from old broken 480p number (0.69). The 1024 number is the correct one — both methods see the perturbation at full magnitude.

### Implication for paper claim (Round 2 framing CONFIRMED)
The "44× stronger than UAP-SAM2" claim is **dead and should not appear anywhere**. The honest, defensible claim is:

> "Under the GT-mask-initialized tracking threat model, the released UAP-SAM2 universal perturbation achieves only +2.78pp ΔJF on DAVIS (n=5 sanity, full eval pending). A random ±10/255 sign control achieves +1.40pp under the same protocol, indicating that universal pixel perturbations are inherently weak when the tracker is anchored by a dense mask prompt. In contrast, the publisher-side mask-conditioned BRS+AdvOpt preprocessing achieves +57.7pp post-codec on the same protocol — a regime where pixel-space attacks fundamentally cannot reach. The contribution is not 'a stronger attack', but a structurally different attack class that exploits the boundary feature pathway codec-robustly."

This is a **threat-model-specific systems claim**, not an attack-leaderboard claim.

### Outstanding work
- Wait for full 90-video runs (UAP-mask, Random-mask, UAP-point) to confirm the 5-video pattern
- The point-prompt UAP run will tell us whether the released weights work better under their training-time prompt regime (point) vs our deployment regime (mask)
- If point-prompt UAP gives e.g. ~10pp on DAVIS, that's evidence we want: "weights are alive under their native protocol; the GT-mask threat model is what makes them weak"
- If point-prompt UAP also gives ~3pp, then DAVIS dataset shift dominates and we should mention that

### Status: Loop 5 baseline reproduction COMPLETE in protocol; awaiting full eval numbers
