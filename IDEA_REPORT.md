# Idea Discovery Report

**Direction**: Diffusion models × SAM2 security — adversarial robustness, defense, detection, red-teaming
**Date**: 2026-03-27
**Pipeline**: research-lit → idea-creator (GPT-5.4 xhigh) → novelty-check → research-review (GPT-5.4 xhigh)
**Ideas evaluated**: 10 generated → 5 deep-validated → 1 recommended + 2 backup

---

## Executive Summary

The existing project paper establishes a clean negative result: all pixel-constrained adversarial attacks (ε=8/255, L∞) against SAM2 video tracking are destroyed by H.264 CRF23 codec. Section 8.1 of that paper explicitly names **semantic-level perturbations** as the most promising open direction. The present pipeline confirms this is an unstudied gap and sharpens the research claim: the real contribution is not "diffusion vs pixel attacks" but rather **causal SAM2 memory state poisoning via codec-surviving semantic edits**, where diffusion img2img is the best candidate method. A preliminary (1-2 hour) pilot on the existing 9 DAVIS videos with the existing SAM2 evaluation pipeline can produce publishable signal.

**Recommended next step**: Run the 2-stage pilot experiment described below. If ΔJF_auc ≥ 0.05 on ≥3 videos at any diffusion strength, implement the full paper with causal memory ablations.

---

## Literature Landscape

**SAM/SAM2 adversarial attacks** are well-studied for image-domain attacks (DarkSAM NeurIPS'24, Robust SAM AAAI'25, UAP-SAM2 NeurIPS'25 arXiv:2510.24195, Attack-SAM, UAP-SAM, RGA) but every published attack uses pixel-space L∞ perturbations and none test survival under real video codec compression. The existing project paper confirms that UAP-SAM2-style FPN feature attacks achieve ΔJF_adv = +0.121 without codec but ΔJF_auc = −0.004 under CRF23.

**Diffusion-based adversarial defenses** (DiffPure ICML'22, DIFFender ECCV'24, Robust-DiffPure) are applied to image classification — not SAM2 video, not with codec robustness analysis.

**Diffusion for privacy** (DiffProtect, Diff-Privacy, CoPSAM) targets facial recognition via static image perturbations. No video instance segmentation, no codec testing.

**Red-teaming VFMs with diffusion** (RedDiffuser arXiv:2503.06223) targets VLMs, not SAM2.

**Key structural gaps confirmed novel** (exhaustive arXiv/web search):
1. ✅ No paper combines (a) diffusion img2img editing + (b) H.264 codec-surviving + (c) SAM2 video tracking attack
2. ✅ No paper studies causal SAM2 memory poisoning via semantic edits (only pixel-space memory targeting exists)
3. ✅ No paper tests simple semantic edits (color jitter, style transfer, low-freq Fourier) against H.264 + SAM2

---

## Ranked Ideas

### 🏆 Idea 1: "Codec-Surviving Semantic Memory Poisoning of SAM2" — **RECOMMENDED**

**Refined title** (per GPT reviewer): *Semantic State Poisoning of SAM2: Codec-Surviving Low-Frequency Edits That Break What Pixel Attacks Cannot*

**Hypothesis**: A semantically-aware edit to a single memory-write frame (t ≥ 1, not frame 0) — generated via diffusion img2img — can:
1. Survive H.264 CRF23 (low-frequency semantic content preserved by DCT)
2. Keep the attacked frame's segmentation quality intact (attacked-frame J&F within 2-3 pp of clean)
3. Corrupt SAM2's maskmem/obj_ptr memory state → persistent tracking failure on subsequent clean frames
4. Causally implicate the memory bank (shown via memory reset / clean memory transplant ablation)
5. Exceed simple non-diffusion low-frequency baselines (color jitter, histogram shift, blur, style transfer)

**Why the existing paper sets this up perfectly**:
- Section 7.2: "L∞ perturbations concentrate energy in high-frequency patterns, zeroed by H.264 DCT"
- Section 8.1 explicit open question: "Semantic-level perturbations that directly modify low-frequency video statistics may survive codec"
- Section 8.3 most important limitation: "Adaptive codec-aware optimization or semantic-level manipulation might succeed"

The narrative arc is clean: Paper 1 = "pixel attacks fail against H.264" → Paper 2 = "semantic-level diffusion edits break through the codec barrier — and the attack works through SAM2's memory bank specifically."

**Method**:
1. **Stage 1 (sweep pilot)**: For each DAVIS video, apply DDIM inversion (SDv1.5/SDXL) at noise strengths {0.2, 0.3, 0.4, 0.5, 0.6} to frame t=2 (second memory write). No targeted guidance. Encode through real H.264 CRF23. Measure ΔJF_auc of future frames. This identifies whether ANY diffusion edit level survives AND hurts SAM2 — provides go/no-go signal.
2. **Stage 2 (targeted optimization)**: If Stage 1 shows signal, optimize the diffusion latent (via DDIM inversion z* → perturbed z*) to maximize J_mem_write loss while constraining attacked-frame J&F ≥ JF_clean − 3pp and codec encode/decode in the loop. Constraint: edit only memory-write frame t, leave all other frames clean. Objective: `max E_future[JF_drop] subject to JF_t_adv ≥ JF_t_clean - 3pp`
3. **Stage 3 (mechanism)**: Memory transplant: take poisoned memory from attacked video, inject into clean video → failure transfers. Memory reset at t+1: performance recovers. This proves causal memory involvement.

**Minimum viable experiment for a credible ICCV 2026 paper**:
- Full DAVIS 2017 val (30 videos, not 9), validity filter JF_clean ≥ 0.5
- One second dataset: MOSE or YouTube-VOS short clips
- Two SAM2 variants: hiera_tiny + hiera_small
- Two prompt types: first-frame point + box
- Real codec sweep: H.264 CRF18/23/28 + HEVC CRF23
- Baselines: pixel L∞ attack (from Paper 1), color jitter, histogram shift, blur, low-freq Fourier, style transfer (neural), untargeted diffusion
- Mechanism ablations: attacked-frame J&F control, memory reset, pointer transplant, maskmem vs obj_ptr
- Metrics: ΔJF_auc, attacked-frame J&F, future-only J&F, recovery horizon, SSIM/LPIPS of edited frame

**Pilot design** (1-2 hours on 1 V100):
```
Stage 1 pilot:
- 9 DAVIS videos × 5 noise levels × ~60 DDIM steps @512px
- Estimated: 9 × 5 × 60 × 0.3s ≈ 810 seconds (~14 min DDIM)
- Plus H.264 + SAM2 evaluation: ~5 min/video × 9 ≈ 45 min
- Total: ~1 hour
- Success criterion: ΔJF_auc ≥ 0.05 on ≥3/9 videos at any noise level AND attacked-frame JF drop < 3pp
```

**Novelty**: CONFIRMED (deep novelty check completed). Full results below.

**Deep Novelty Check Results (from parallel search agents):**

**On Idea 1 (Memory Poisoning):**
- UAP-SAM2 (arXiv:2510.24195, NeurIPS'25) is closest: attacks SAM2 memory but uses gradient-based UAP across ALL frames; explicitly states **"attacking only the first frame has limited effectiveness"**
- No paper does single-frame diffusion-editing targeting maskmem/obj_ptr specifically
- ACM MM 2023 (arXiv:2309.13857) does first-frame attacks on older VOS models (pre-SAM2), not maskmem/obj_ptr
- **Key differentiator**: UAP-SAM2 claims single-frame attacks fail → if diffusion semantic edits prove them wrong (by using low-frequency semantic content vs. pixel noise), that is the central empirical claim

**On Idea 3 (Track-Me-Not):**
- Fully novel: no paper combines diffusion editing + first-frame SAM2 memory exploitation + privacy goal
- UAP-SAM2 has the memory-exploitation angle but no diffusion, no privacy, no early-frame-only strategy
- TUEs for VOT (arXiv:2507.07483, ICCV 2025) does video privacy via unlearnable examples but no diffusion, no SAM2

**On DiffProtect/Diff-Privacy**: facial recognition only, static images, no video/codec.

**Contribution type**: New method + empirical finding + mechanism/diagnostic
**Risk**: MEDIUM (diffusion edits likely survive H.264; the question is whether they hurt SAM2 enough)
**Estimated effort**: 1-2 weeks for full paper (infrastructure mostly exists from Paper 1)

**GPT reviewer score**: 5/10 as originally framed → **8/10 with the sharper framing** (non-frame-0 attack, delayed failure proof, causal memory ablation, simple baseline comparison)

**The UAP-SAM2 counter-claim is your strongest asset**: UAP-SAM2 (NeurIPS'25) explicitly claims "single-frame attacks fail, cross-frame attack is necessary." Your paper directly falsifies this claim — but only if using diffusion semantic edits specifically. Pixel noise attacks (our Paper 1) confirm UAP-SAM2's claim. Diffusion edits will contradict it. This is a perfect "prior work says X → we show X is wrong under condition Y" structure.

**Key reviewer objections and mitigations**:
| Objection | Mitigation |
|-----------|-----------|
| "This is just a visible semantic edit / domain shift" | Constrain attacked-frame J&F within 3pp of clean; report SSIM/LPIPS; natural DDIM reconstruction looks realistic |
| "Frame 0 confounds: initialization failure, not memory poisoning" | **Attack frame t≥2**, not frame 0; memory reset/transplant ablation proves causality |
| "Diffusion is unnecessary; simple color jitter would work too" | Include all simple baselines; if diffusion beats them, that's the claim; if not, reframe as "semantic edits in general survive H.264" |

**Why we should do this**: This is the minimal, clean follow-up to an existing published negative result. The infrastructure (DAVIS dataset, SAM2 evaluation, H.264 codec, ΔJF_auc metric) all exists. The pilot takes 1-2 hours. If signal appears, this is a paper-in-a-week.

---

### Idea 2: "Ghost Trigger — Transient Diffusion Patch Persisting via SAM2 Memory" — BACKUP

**Hypothesis**: A diffusion-synthesized natural-looking patch present only on frame t, removed in all subsequent frames, causes persistent SAM2 failure because SAM2 caches the patch-conditioned object pointer, which then drives erroneous cross-attention in future frames.

**Novelty**: PARTIALLY CONFIRMED. UAP-SAM2 (NeurIPS'25) attacks SAM2 video but tests persistent perturbations, not transient. The "transient trigger persisting via memory state" mechanism is novel.

**Risk**: MEDIUM. Requires the diffusion patch to be (a) recognizable to SAM2's object pointer and (b) semantically consistent enough to corrupt it without destroying the prompt-based initialization.

**Relationship to Idea 1**: Can be added as a second experiment in the same paper (extend from "memory-write frame attack" to "transient patch attack").

**Effort**: 1-2 additional days on top of Idea 1 infrastructure.

---

### Idea 3: "Track-Me-Not — Diffusion Privacy Preprocessor for H.264 Distribution Pipelines" — BACKUP (application)

**Hypothesis**: For the publisher-side privacy protection goal: diffusion img2img editing on the first 2-3 frames of a person-containing video creates semantic appearance changes that (a) survive H.264 CRF23, (b) suppress SAM2-based person tracking in downstream analytics, while (c) preserving human-perceptible action semantics (video content still watchable).

**Relationship to existing project**: This is the ORIGINAL project goal (publisher-side privacy preprocessor) now powered by diffusion instead of pixel attacks. The negative result paper showed pixel attacks fail. Diffusion edits would succeed.

**Novelty**: CONFIRMED. No paper does diffusion-based video privacy against codec-compressed video distribution + SAM2 tracking.

**Risk**: MEDIUM-HIGH. Requires balancing privacy (large enough edit to hurt SAM2) vs. utility (small enough edit to preserve video watchability). Threat model is less "adversarial" and more "privacy tool."

**Framing**: This can be a direct positive counterpart to Paper 1 specifically for the privacy preprocessor application, framed as: "Pixel attacks fail for publisher-side privacy (Paper 1). Here's what works instead."

---

## Eliminated Ideas

| Idea | Reason eliminated |
|------|-------------------|
| Idea 6: Diffusion Stress Curves | Clean diagnostic, low ceiling; becomes supporting figure in Idea 1 paper |
| Idea 7: Which Frame Matters | Ablation table, not a standalone paper; supporting analysis |
| Idea 5: Purify the Cache | Defense side; adaptive attacker likely routes through unpurified frames; too weak as standalone |
| Idea 4: Pointer Collision | MEDIUM-HIGH risk, unclear whether SAM2 obj_ptr has the required identity margin properties |
| Idea 10: Delayed-Reward RedDiffuser | HIGH risk, 1-2 months effort; scope too large for current timeline |
| Idea 8: Memory Leaks Appearance | 2-4 week effort; interesting but different paper direction (privacy audit, not attack/defense) |

---

## Pilot Experiment Results

| Idea | Status | Estimated GPU-hours | Priority |
|------|--------|---------------------|----------|
| Idea 1 Stage 1 (sweep) | READY TO RUN | ~1-1.5 hrs on 1 V100 | **IMMEDIATE** |
| Idea 1 Stage 2 (targeted) | Conditional on Stage 1 | ~4-8 hrs on 1 V100 | After Stage 1 signal |
| Idea 2 (Ghost Trigger) | After Idea 1 infra | ~4 hrs | Secondary |
| Idea 3 (Track-Me-Not) | After Idea 1 confirmed | ~8 hrs | Tertiary |

---

## Suggested Execution Order

1. **TODAY**: Run Stage 1 pilot (DDIM noise sweep on frame t=2, 9 DAVIS videos, 5 noise levels)
   - Success: ΔJF_auc ≥ 0.05 on ≥3 videos → proceed to full paper
   - Failure: ΔJF_auc ≈ 0 at all strengths → pivot to larger edit / different frame timing

2. **If Stage 1 positive** (expected 1-2 days after running):
   - Implement targeted diffusion latent optimization (Stage 2)
   - Add memory transplant ablation (Stage 3)
   - Expand to full DAVIS val (30 videos)

3. **Paper writing** (~2 weeks after successful pilot):
   - Framing: "Codec-Surviving Semantic State Poisoning of SAM2"
   - Venue target: ICCV 2026 or ECCV 2026 (deadline check needed)
   - Use `/auto-review-loop` after draft

---

## Key Open Questions Before Starting

1. **Which diffusion model for editing?** Recommended: SDv1.5 with DDIM inversion (fast, controllable). Alternative: InstructPix2Pix (instruction-guided, more semantic). Avoid SDXL for pilot (too slow on V100).
2. **Which frame to attack?** Frame t=2 (second memory write, after clean initialization). Rationale: avoids frame-0 initialization confound raised by reviewer.
3. **What edit guidance?** For Stage 1: none (random DDIM reconstruction at varying strengths). For Stage 2: optimize latent direction to maximize maskmem cosine distance after codec.
4. **Codec in optimization loop?** Stage 1: evaluate with codec post-hoc. Stage 2: use differentiable proxy (YUV 4:2:0 + blur + noise from existing code) with periodic real H.264 validation.

---

## Connection to Existing Work

This research directly follows from the existing paper:
- **Uses same evaluation pipeline**: 9 DAVIS 2017 val videos, SAM2.1 hiera_tiny, ΔJF_auc metric, real H.264 CRF23
- **Uses same SAM2 loss functions**: Mode C+D (J_mem_write, J_mem_match, J_ptr) from existing eval_codec.py
- **Answers the open question from Section 8.1**: "semantic-level perturbations may survive codec"
- **Directly comparable**: ΔJF_auc from diffusion attack vs. ΔJF_auc ≈ 0 from all pixel attacks

Implementation delta needed:
- Add DDIM inversion wrapper (SDv1.5, ~50 lines)
- Add img2img sweep loop around existing evaluation
- Rest of pipeline (H.264, SAM2, ΔJF_auc) reused unchanged
