# Research Review Session — 2026-03-30
**Thread**: `019d416c-4bdb-7912-afd4-afd5f0167321`  
**Reviewer**: GPT-5.4 (xhigh reasoning)  
**Questions**: What experiments done? Current claims? What's needed? What's abandoned?

---

## TL;DR

**Paper plan**: Body 2 ONLY (boundary suppression). Body 1 (negative result) → motivation section.  
**Current score if submitted today**: CVPR 2.5/5 (weak reject)  
**Score after 3 must-do items**: CVPR 6/10 (weak accept to borderline)

---

## Experiment Inventory

### Essential for paper (keep)
| ID | Description |
|----|-------------|
| MG-04 | combo_strong mask prompt, DAVIS (n=88, +16.4pp) |
| MG-05 | combo_strong point prompt, DAVIS (n=73, +40.5pp) |
| MG-07 | hiera_small backbone (n=88, +17.8pp) |
| MG-02 | idea1 (boundary only, n=88, +4.8pp) |
| MG-03 | idea2 (echo only, n=88, +0.5pp) |
| MG-FB1 | boundary_blur fair baseline (n=89, +10.8pp) |
| MG-FB2 | interior_feather fair baseline (n=89, +0.9pp) |
| MG-06 | global_blur (n=88, +9.4pp) |
| MG-A1 | single-frame ablation (+0.9pp vs +6.5pp) |
| MG-C1/C2 | CRF sweep (18/23/28), mask + point |
| MG-D1 | YouTube-VOS (n=497, +4.0pp) |
| MG-G1 | DAVIS-27 gap control (n=89, +12.8pp) |
| MG-M1 | Codec amplification mechanism |
| MG-R1/R2 | Mask robustness dilate/erode |
| MG-COV | Covariate regression (R²=0.28, 585 videos) |

### Supportive / nice-to-have
| ID | Description |
|----|-------------|
| MG-01 | combo std (rw=16, α=0.6) |
| MG-08 | Param sweep grid |
| MG-MINI | Scale-norm vs fixed, n=24 YT-VOS |
| MG-R3 | Mask robustness noise (note: SSIM=0.581 is bad) |
| Body 1 | Pixel/feature attacks fail (motivation section) |
| B1a | Simplified UAP baseline (if reframed honestly) |

### Wasted / abandoned
| ID | Description |
|----|-------------|
| C1/C2 generator evals | g_theta_size=256/0 mismatch → invalid |
| UAP reproduction claim | B1a ≠ "Vanish into Thin Air" |
| VMAF pipeline | No libvmaf, low priority |
| Semantic sweep | Incomplete, not needed |
| idea2 as main claim | +0.5pp, negligible |

---

## Current Claims Assessment

### Supported claims ✓
- On DAVIS, combo_strong → +16.4pp post-codec (mask), +40.5pp (point)
- Point prompt 2.5× more vulnerable than mask prompt
- Transfers from hiera_tiny → hiera_small (+17.8pp)
- Better than global blur on DAVIS (+16.4pp@SSIM=0.921 vs +9.4pp@SSIM=0.687)
- Single-frame ablation: persistent editing needed (+6.5pp vs +0.9pp)
- DAVIS/YT-VOS gap is content-conditioned (regression supports, not just frame count)

### Claims to narrow ⚠️
| Claim | Problem | Fix |
|-------|---------|-----|
| "Codec amplification phenomenon" | Only -58.5%→-61.0% (2.5pp extra, not headline) | "edit survives, often modestly strengthened" |
| "Persistent editing, not memory poisoning" | Ablation shows it; mechanism not fully isolated | Keep ablation claim, drop "mechanism" |
| "Robust to mask imperfection" | Only dilate/erode, not realistic auto-masks | "tolerant to mild morphological errors" |
| "Publisher-side privacy defense" | Content-conditioned, not general | "content-conditioned failure mode of SAM2" |
| "Pareto dominance" | YOLO utility is too weak | Change to "distortion-privacy Pareto" |
| "UAP baseline fails" | Only simplified B1a baseline | "simplified codec-unaware UAP baseline fails" |

### Overclaiming (must fix before submission)
- Do NOT say: "we solved publisher-side privacy for SAM2"
- DO say: "low-frequency boundary suppression is a codec-compatible, content-conditioned failure mode of SAM2"

---

## Three Must-Do Experiments Before Submission

### 1. Held-Out Gate Experiment (compute_gate.py) — ~5min CPU
**Goal**: Identify reliable operating regime on YT-VOS; convert "unsafe average" into scoped claim.

```bash
python compute_gate.py \
  --csv results_v100/covariates.csv \
  --dataset YTVOS \
  --min-jf-clean 0.3 \
  --mode threshold \
  --gate-feature boundary_dominance \
  --test-size 0.5 --seed 0 \
  --train-neg-rate-target 0.10 \
  --min-mean-delta 5.0 \
  --out-json results_v100/gate/ytvos_bd_gate_seed0.json
```

**Targets**: negative_rate ≤10%, coverage ≥20%, mean_delta_jf ≥5pp  
**Report**: tau, accepted_videos, coverage%, mean_delta_jf, negative_rate, success_rate_5pp

### 2. YT-VOS Tuning/Transfer (B-lite) — ~6h GPU
**Goal**: Decide if gap is "bad hyperparameters" or genuine content limitation.

**Protocol**:
- 50/50 stratified (by JF_clean) train/test split of YT-VOS
- Tune on YT-VOS train: rw∈{16,24,32} × rho∈{0.06,0.10,0.14} × alpha∈{0.6,0.8,0.9}
- Constraint: SSIM ≥ 0.92
- Report 2×2 matrix:

| Config | DAVIS test | YT-VOS test |
|--------|-----------|-------------|
| DAVIS default (rw=24, α=0.8) | +16.4pp | +4.0pp |
| YT-VOS-tuned best | ? | ? |

**Interpretation**:
- If YT-tuned YT-VOS test ≤6pp or neg_rate ≥15% → "content-conditioned" claim is strong
- If YT-tuned YT-VOS test ≥8-10pp and neg_rate ≤10% → gap was tuning issue (bad for story)

### 3. Standardize Validity Filter — 0 extra compute
- Main tables: JF_clean ≥ 0.3 (consistent everywhere)
- Appendix: same numbers at ≥0.5
- Reviewer-safe justification: "We predefine JF_clean≥0.3 to exclude degenerate baseline failures, report retained counts for every condition, and show threshold sensitivity in appendix."

---

## Framing Corrections for Paper

**New title direction**: "A Codec-Compatible, Content-Conditioned Failure Mode of SAM2 Video Tracking under Mask-Guided Boundary Suppression"

**Paper structure** (one paper, Body 2 only):
1. Motivation: pixel/UAP attacks fail under H.264 → need fundamentally different approach (cites Body 1 or brief Sec 2 negative)
2. Method: low-frequency mask-guided boundary suppression
3. Main DAVIS results + matched baselines
4. Robustness and mechanism (CRF, masks, backbones, single-frame ablation, gradient analysis)
5. Cross-dataset scope: YT-VOS gap, regression, gate/abstention
6. Limitations (oracle masks, SSIM cost, content-conditioned)
7. Conclusion

**Utility section**:
- Do NOT say "utility preserved"
- DO say "lower collateral distortion than global blur"
- YOLO recall = secondary proxy, NOT main utility claim
- Reframe Pareto as: "distortion-privacy Pareto" (SSIM vs ΔJF_codec)

---

## Mock Review Score (After 3 Must-Do Items)

| Aspect | Score |
|--------|-------|
| CVPR 2026 | 6/10 (weak accept to borderline) |
| NeurIPS | 5/10 (borderline reject — empirical) |
| ACM MM | Likely accept |

**Strengths (reviewer)**:
- Realistic setting, clean method, large DAVIS effect
- Proper failure mode disclosure (YT-VOS weakness, prompt dependence)
- Regression + gate + transfer experiment strengthen scope claim
- Training-free, reproducible

**Weaknesses (reviewer)**:
- Oracle mask requirement
- Content-conditioned, gated subset may cover only modest fraction
- Utility evidence limited
- Mechanism is suggestive, not closed-form

**What moves to accept**:
- Auto-mask experiment (realistic masks)
- Small human perceptual study (30-50 clip pairs)
- Gate stability across multiple splits

---

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| Body 1 → motivation only | Too narrow, n=9 videos, not enough for standalone main-track paper |
| Keep "codec amplification" as secondary claim only | -58.5%→-61.0% is real but small; not the headline |
| Drop "utility preserved" claim | YOLO proxy insufficient; change to "distortion-privacy Pareto" |
| Keep MG-R3 (noise) but note SSIM=0.581 | Important honest disclosure, not a selling point |
| Relabel B1a everywhere | Must say "simplified codec-unaware UAP baseline inspired by..." |
| Gate on YT-VOS is must-have | Converts unsafe mean to scoped operational claim |
