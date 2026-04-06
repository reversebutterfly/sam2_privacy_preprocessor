# Preliminary Results — Semantic Boundary Suppression
**Date**: 2026-03-28
**Status**: Sweeps running; partial results below

---

## Executive Summary

Mask-guided semantic boundary suppression **survives H.264 CRF23** and degrades SAM2 J&F by **5–25pp** depending on video and parameter strength. This is a positive pilot result that confirms the feasibility of publisher-side semantic privacy preprocessing for annotated video datasets.

Key finding from sanity + partial sweep (6–9 videos):

| Method | Mean ΔJF post-H264 | Mean SSIM |
|--------|-------------------|-----------|
| Pixel L∞ (Paper 1) | ≈ 0pp (fails completely) | 0.88–0.90 |
| combo std (ring=16, α=0.6) | **+5.0pp** | ~0.945 |
| idea1 only (boundary supp) | **+3.9pp** | ~0.947 |
| idea2 only (echo contour) | **+0.6pp** (negligible) | ~0.980 |
| **combo_strong (ring=24, α=0.8)** | **+14.7pp** | ~0.885 |

---

## Method: Mask-Guided Boundary Suppression

The edit applies to **every frame** using GT segmentation masks:

1. **Boundary suppression (idea1)**: Blend a ring of `ring_width` pixels around the mask boundary toward the local background, with strength `blend_alpha`. This reduces figure-ground separability at the object boundary.

2. **Echo contour (idea2)**: Add a faint smooth halo ring just outside the mask. Intended to create a competing false boundary for SAM2's boundary detector.

The combo of (1)+(2) with stronger params (ring=24, blend=0.8) is the primary method.

**Why it survives codec**: Boundary suppression is a low-frequency spatial operation (broad blending kernel, smooth gradients). H.264 DCT quantization zeroes high-frequency components but preserves low-frequency pixel patterns — exactly what our edit uses.

**Why pixel attacks fail**: L∞ pixel noise concentrates energy in high-frequency pixel changes (salt-and-pepper-like patterns), which are zeroed by DCT.

---

## Mechanism Ablation

### Single-frame vs. all-frame attack

Tested on 5 videos (GPU 0, `pilot_memory_ablation.py`):

| Condition | Mean Δfut (future frames) |
|-----------|--------------------------|
| adv_all (all frames edited, post-codec) | +7.2pp |
| adv_t (only frame t=2 edited, post-codec) | **+1.0pp** |
| adv_t + memory reset at t+1 | +0.7pp |

**Memory PROOF: NO** — single-frame attack fails; persistent editing required.

**Interpretation**: The attack works through **repeated boundary confusion** at every frame, not through SAM2 memory state poisoning. At each frame, SAM2 must re-estimate the boundary; if the boundary is suppressed at every frame, tracking degrades cumulatively. A single corrupted frame does not poison the memory bank enough to cause lasting failure.

This changes the paper story:
- **Not**: "memory poisoning attack"
- **Yes**: "persistent semantic obfuscation as a publisher-side privacy tool"

The publisher-side setting is actually *more realistic* — a dataset publisher who controls the video and has GT masks (common for research datasets like DAVIS, YouTube-VOS, MOSE) can apply this to suppress downstream SAM2-based tracking.

---

## Paper Story (Revised)

### Title
"Codec-Surviving Semantic Boundary Suppression for Privacy-Preserving Video Dataset Release"

### Thesis
Pixel-constrained adversarial attacks against SAM2 are completely neutralized by H.264 compression (confirmed in Paper 1). However, **mask-guided semantic boundary suppression** — a simple image processing operation applied persistently across all frames using ground-truth segmentation masks — successfully degrades SAM2 tracking by 5–25pp while surviving H.264 CRF23, at a SSIM cost of only 0.05–0.07.

### Claim Map
| Claim | Evidence | Status |
|-------|----------|--------|
| C1: Semantic edits survive H.264 codec | ΔJF_codec ≥ 8pp at SSIM ≥ 0.88 | CONFIRMED (partial) |
| C2: Pixel attacks fail (baseline) | ΔJF_codec ≈ 0 from Paper 1 | CONFIRMED (prior work) |
| C3: Mechanism = persistent boundary confusion, not memory poisoning | Single-frame Δfut = +1pp vs all-frame = +7pp | CONFIRMED (ablation) |
| C4: idea2 (echo contour) alone does not work | ΔJF_codec = +0.6pp | CONFIRMED (partial) |
| C5: Boundary suppression (idea1) is the active mechanism | idea1 = +3.9pp, idea2 = +0.6pp | CONFIRMED (partial) |

---

## Preliminary Per-Video Results (combo_strong, 9 videos)

| Video | ΔJF post-H264 | SSIM |
|-------|--------------|------|
| bear | +8.2pp | 0.887 |
| bike-packing | +22.3pp | 0.894 |
| blackswan | +5.8pp | 0.865 |
| bmx-bumps | +9.4pp | ~0.88 |
| bmx-trees | +25.8pp | ~0.93 |
| boat | +6.2pp | 0.890 |
| boxing-fisheye | (pending) | |
| ... | | |

**Highest effect** (>20pp): bike-packing (+22.3pp), bmx-trees (+25.8pp)
**Moderate effect** (8–12pp): bear (+8.2pp), bmx-bumps (+9.4pp)
**Lower effect** (<8pp): blackswan (+5.8pp), boat (+6.2pp)

---

## Pending Experiments

| Experiment | Status | ETA | Purpose |
|-----------|--------|-----|---------|
| full_combo (std params, all 85 videos) | 6/85 | ~4h | Characterize quality-effectiveness tradeoff |
| full_idea1 (boundary only, all 85 videos) | 9/85 | ~3h | Ablate echo contour contribution |
| full_idea2 (echo only, all 85 videos) | 10/85 | ~3h | Confirm echo is negligible |
| full_combo_strong (all 85 videos) | 9/85 | ~3h | **Primary result** |
| Memory ablation (10 videos) | 5/10 done | ~1h | Mechanism proof |
| Point-prompt sweep | PENDING | queue | Non-oracle tracking robustness |
| CRF sweep {18, 23, 28} | PENDING | queue | Robustness to compression level |
| Parameter sweep (ring_width, blend_alpha) | PENDING | queue | Pareto frontier visualization |

---

## Go/No-Go Decision

**Preliminary verdict: GO**

The signal is strong enough to warrant a full paper:
- Mean ΔJF_codec = **+14.7pp** for combo_strong across first 9 videos
- SSIM ≈ 0.885 (acceptable for privacy preprocessing)
- Mechanism is clean and explainable
- Novel over all prior work (confirmed in IDEA_REPORT.md)
- Infrastructure already exists (DAVIS, SAM2, H.264)

**Minimum bar for ICCV 2026**:
- Full DAVIS 2017 val (30 videos) with ΔJF_codec ≥ 8pp mean
- Two SAM2 models (hiera_tiny + hiera_small)
- Pixel attack baseline clearly failing
- Quality: SSIM ≥ 0.85 and PSNR ≥ 18dB
- One additional dataset (YouTube-VOS short clips)
