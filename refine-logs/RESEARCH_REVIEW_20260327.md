# Research Review — 2026-03-27
**Reviewer**: GPT (xhigh reasoning) via Codex MCP
**Thread**: `019d2f86-d992-7fa2-8cdc-d9040b98c3f4`
**Context**: Publisher-side video preprocessing against SAM2 tracking; all single-frame approaches exhausted; user says full-video processing is allowed

---

## Overall Verdict

> "This is not an ICCV/ECCV/NeurIPS main-track attack paper yet. Right now you have a strong negative result and a useful realism benchmark."

The project has 3 rounds of experiments confirming robustness, not attack viability. The negative result is scientifically valuable but not yet packaged. A positive attack paper remains possible if the **object-aware camouflage** approach works by April 7.

---

## Critical Findings from Review

### 1. Why Full-Frame Global Edits Will Also Fail
A consistent all-frame appearance shift (color, style, DDIM reconstruction) creates a new stable visual domain for SAM2 — it does NOT break tracking. SAM2 tracks **figure-ground separability over time**, not absolute RGB identity. A consistent global edit simply redefines the "normal appearance" that SAM2 tracks.

**Therefore the only plausible full-video attack mechanism is:**
- Separability collapse: make object-background boundary ambiguous
- Assignment confusion: add competing boundaries / decoy structure near the object
- Temporal inconsistency: subtle object-local warps that destabilize memory while staying watchable

### 2. The Right Attack Family: Object-Aware Boundary Suppression
YCbCr-space, object-aware, low-frequency edit:
```
I_t  = eroded interior of object (from GT mask or SAM2 auto-mask)
B_t  = boundary ring = dilate(M_t, r_out) - erode(M_t, r_in)
O_t  = outer context = background near object
G_t  = background proxy from O_t (masked inpaint or Gaussian fill)

Y'  = Y + aY_int * S_I * (Y_G - Y)    # interior harmonization
    + aY_ring * S_B * (Y_G - Y)         # boundary suppression
    + a_halo * S_H * blur(Y_G - Y)      # competing external contour
Cb/Cr' analogous but smaller weights
```

**8-10D parameter vector** (shared across ALL frames of a video):
- `aY_int`, `aC_int` — interior blend weights (Y and Chroma)
- `aY_ring`, `aC_ring` — boundary ring blend weights
- `r_in`, `r_out` — mask erosion/dilation radii
- `feather_sigma` — boundary feathering
- `a_halo`, `halo_width`, `halo_offset` — external competing contour

**Optimization**: Black-box (CMA-ES / random search) directly against real VideoPredictor + H.264 CRF23. No differentiable path needed.

### 3. GT Mask Dependency — Three Tiers
Using GT masks is oracle upper bound, not "weaker" threat model. Paper should present three tiers:
- **Oracle**: GT masks from DAVIS annotations
- **Practical**: SAM2 auto-mask on frame 0 (publisher uses public SAM2 to localize target)
- **Robustness**: Noisy-mask ablation (erosion, dilation, shifts, partial masks)

This converts a potential weakness into a measured axis of realism.

### 4. Eval Protocol Issue
Current `eval_codec.py` and `pilot_semantic_sweep.py` use **centroid point prompt** (valid, but non-standard for DAVIS). Standard DAVIS semi-supervised uses **first-frame mask prompt**.

Action: Add mask-prompt evaluation immediately. If both point and mask show null results, the robustness claim becomes much stronger.

### 5. Conference Timing Correction
- ECCV 2026: **closed March 5, 2026** (missed)
- ICCV 2026: **does not exist** (ICCV is odd years only; next ICCV = 2027)
- **NeurIPS 2026**: abstract/full deadlines **May 4/6, 2026** (≈5 weeks away)
- NeurIPS E&D track: **best venue for negative paper**

---

## Priority Experiment Sequence

**Hard kill date: April 7, 2026**

| Date | Task | GPU-hours | Deliverable |
|------|------|-----------|-------------|
| Mar 27-28 | Eval calibration: point vs mask prompt on 9-video mini-val, clean + one null attack | 1.5-2 | Table: prompt sensitivity + clean ceilings |
| Mar 28-29 | Global all-frame baseline (1 DDIM strength or color shift, 9 videos) | 2-2.5 | Go/no-go on global domain shift hypothesis |
| Mar 29-31 | Implement 3 hand-crafted object-aware families (boundary harm, ring blur, halo) | 4-5 | Best family + default parameter range |
| Apr 1-3 | Black-box tune best family on 3 anchor videos (random search → CMA-ES if signal) | 6-8 | Best param vector + mean post-codec drop |
| Apr 4-6 | Light per-video tuning on 9-video mini-val | 5-7 | Real decision metric |
| **Apr 7** | **HARD DECISION** | — | Continue or pivot to negative paper |

**Total pilot budget: 19-24 GPU-hours. 2 V100s → feasible.**

### Kill criteria (Apr 7)
Continue positive paper ONLY if at least one of:
- Mean post-codec J&F drop ≥ 5pp on 9-video mini-val
- Median drop ≥ 4pp AND ≥6/9 videos with ≥5pp drop
- Clearly reproducible effect under BOTH point and mask prompts

### Global DDIM go/no-go thresholds
- Kill if mean post-codec drop < 1pp OR < 2/9 videos exceed 3pp
- Keep as baseline only if mean 1-3pp
- Real signal only if mean ≥ 3pp AND ≥3/9 videos exceed 5pp

---

## Paper Structures

### IF Object-Aware Camouflage Works (≥5pp post-codec drop)

1. Introduction
2. Related Work
3. Threat Model and Evaluation Protocol
4. Object-Aware Codec-Robust Camouflage (method section)
5. Black-Box Optimization Through Real VideoPredictor + H.264
6. Experiments (main results table)
7. Ablations and Mechanistic Analysis
8. Limitations and Ethical Discussion

**Essential figures (5):**
1. Pipeline: target-local edit → H.264 → downstream SAM2 failure
2. Main results: clean vs pre-codec vs post-codec across baselines + method
3. Quality-attack tradeoff: J&F drop vs LPIPS/SSIM/VMAF
4. Qualitative sequence: edited frames, predicted masks, failure over time
5. Ablation: boundary-only vs interior-only vs halo-only; Y-only vs chroma-only

**Essential ablations:**
- Point vs mask prompt
- CRF18/23/28 sweep
- Boundary-only / interior-only / halo-only
- Luma-only / chroma-only / full YCbCr
- GT mask vs approximate SAM2 mask

**Optional (if time allows):**
- Second SAM2 backbone (hiera_small)
- Second tracker
- Temporal flicker metric / user study
- Universal parameters vs per-video tuning

### IF Pilot Fails (→ Negative Paper)

Strengthen `paper/main.pdf` with additional experiments:

1. Introduction
2. Why Publisher-Side Preprocessing Is Hard for SAM2
3. Threat Model, Prompt Settings, and Evaluation Pitfalls
4. Attack Families Tested (pixel L∞, feature-space, single-frame semantic, all-frame global)
5. Codec Purification of Pixel/Feature Attacks (existing content)
6. Temporal Recovery from Single-Frame Semantic Edits (new: semantic sweep results)
7. All-Frame Global Appearance Shifts Also Fail (new: DDIM baseline)
8. Mechanistic Analysis (DCT spectral analysis + surrogate gap analysis)
9. Recommendations for Robust Evaluation (point vs mask, surrogate calibration)
10. Limitations and Outlook

**Additional experiments to add:**
- Point vs mask prompt sweep (both show null → stronger claim)
- CRF18/23/28 sweep for completeness
- Spectral/DCT analysis before and after H.264
- One all-frame global appearance-shift null result (DDIM or color grading)
- Surrogate-vs-real predictor gap as a first-class quantitative result (12× JF discrepancy)
- At least one additional SAM2 size or dataset for breadth

**Best venue for negative paper:**
- NeurIPS 2026 Evaluations & Datasets track (primary target)
- Vision/Security workshop if E&D too competitive
- Consider broadening benchmark for CVPR 2027 main track

---

## Key Insight: Two Independent Robustness Mechanisms

1. **H.264 codec (compression layer)**: Zeroes all high-frequency (ε=8/255) perturbations
2. **SAM2 temporal memory (model architecture)**: Single-frame edits get diluted by K−1 clean memory entries within 1-2 frames

Together these make ALL single-frame attack paths infeasible. This joint characterization is the core contribution of the negative paper.

---

## Action Items (Immediate)

1. **TODAY**: Add mask-prompt support to eval_codec.py → run calibration eval
2. **TODAY/TOMORROW**: Implement object-aware boundary edit (3 families) — no diffusion needed
3. **Mar 29**: Deploy all-frame global DDIM baseline as quick go/no-go
4. **Apr 1**: Begin CMA-ES black-box optimization on best edit family
5. **Apr 7**: Hard decision checkpoint
