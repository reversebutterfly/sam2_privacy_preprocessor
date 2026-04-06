# Feature-Space Attack Plan — SAM2 Privacy Preprocessor
**Date:** 2026-03-26
**Status:** FROZEN — ready for implementation
**Based on:** Literature review (UAP-SAM2 NeurIPS 2025, 10 papers) + 2-round external review (xhigh reasoning)

---

## Context

4-round auto-review completed. Pixel-level attack (g_θ residual CNN, ε=8/255, mask-logit loss) confirmed negative result:
- Image predictor ΔJF ≈ 0.44 (strong)
- Video predictor post-codec CRF23 mean ΔJF_codec = **+0.004** (near-zero)

Key literature finding: **UAP-SAM2** (NeurIPS 2025, arXiv:2510.24195) achieves −45.8pp mIoU on DAVIS with pixel-constrained attack using **feature-space objectives** on FPN encoder output.
Critical gap: UAP-SAM2 does NOT test under H.264 codec. This is our thesis differentiator.

---

## Candidate Locations — Summary Verdict

| Location | Causal leverage | Codec survival | Verdict |
|----------|----------------|----------------|---------|
| **A** Hiera Stage 3 [B,768,8×8] | Medium | Medium-high | **Discard** — poor gradient, no direct memory path |
| **B** FPN finest [B,256,64×64] | Medium | Medium-low | **Backup** / falsification baseline |
| **C** maskmem_features [B,64,8×8] | **High** | **Highest** | **Primary** |
| **D** obj_ptr [B,256] | Medium alone | Medium-high | **Auxiliary with C** |

**Do not invest in A.** It is semantically attractive but optimization-poor, and going further upstream is more likely to vanish inside the neck than strengthen the attack.

---

## 1. PRIMARY PLAN: C + D (maskmem_features + obj_ptr)

### Rationale
- Attacks the **actual propagated state** used by `SAM2VideoPredictor`, not a pre-memory surrogate
- Mechanistic story: "codec-surviving perturbation poisons the memory write; later frames attend to wrong state"
- 8×8 spatial resolution → lower high-frequency content → better codec survival
- Distinct from UAP-SAM2: we supervise the **written causal state** (maskmem_features + key-query compatibility), not just encoder output divergence

### Loss Formulas

Let `T` = sampled codec/EOT transform (same settings applied to both branches).
`x_t^c = T(x_t)`, `x_t^a = T(x_t + δ_t)`

**J_mem_write** (cosine distance on written memory tokens):
```
J_mem_write(t) = (1/N_m) · Σ_i [ 1 - cos(M_t^a_i, M_t^c_i) ]
```
- M_t^a, M_t^c ∈ ℝ^{N_m × d_m} = flattened maskmem_features
- Use cosine (not L2): L2 too sensitive to mask-area / codec amplitude artifacts
- **Log a frozen-mask diagnostic**: if M_t^a moves only when mask changes, it is a mask attack, not memory-write attack

**J_mem_match** (key-query compatibility reduction):
```
C_{u←t}(Q, K) = (1/N_q) · Σ_j τ · log( (1/N_m) · Σ_i exp(q̂_j^T k̂_i / τ) )

J_mem_match(t) = (1/k) · Σ_{u=t+1}^{t+k} [ C_{u←t}(Q_u^c, K_t^c) − C_{u←t}(Q_u^c, K_t^a) ]
```
- Q_u^c = clean future-frame query tokens; K_t^a = poisoned memory keys
- τ = temperature (use 0.1 as starting point)
- Alternative if you can hook full attention output:
```
J_attn_out = (1/k) · Σ_{u=t+1}^{t+k} (1/N_q) · ||A(Q_u^c, K_t^a, V_t^a) - A(Q_u^c, K_t^c, V_t^c)||_F^2
```

**J_ptr** (object pointer drift):
```
J_ptr(t) = (1 - cos(p_t^a, p_t^c)) + η · (1/k) · Σ_{u=t+1}^{t+k} (1 - cos(p_t^a, p_u^c))
```
- η = 0.5
- Direction: **away from clean pointer**, NOT toward zero (zero is not mechanistically grounded)

**J_mask** (weak auxiliary, mask output BCE):
```
J_mask = weak auxiliary only, never headline loss
```

**Total loss** (minimization form):
```
L_total = −(α·J_mem_write + β·J_mem_match + γ·J_ptr + δ·J_mask)
          + λ_LPIPS · L_LPIPS + λ_SSIM · L_SSIM
```

**Starting coefficients:**
```
(α, β, γ, δ) = (1.0, 2.0, 0.25, 0.05)
λ_LPIPS = 1.0,  λ_SSIM = 1.0
```
- Increase β to 3.0 for frame-0-only decisive experiment
- L_mask must remain weak (δ ≤ 0.05); if it dominates, reverts to discredited anti-pattern

### Where to apply losses
- **J_mem_write**: on the attacked frame (local)
- **J_mem_match, J_ptr (future term), J_mask**: on **downstream clean frames** that attend to poisoned memory
  → This is the causal test: clean later frames fail because of poisoned initial state

---

## 2. BACKUP PLAN: B (FPN finest level only)

### When to activate
Only if C+D fails kill criteria (see §5).

### Method
- Replicate UAP-SAM2 feature-shift loss on FPN finest [B,256,64×64]
- Add inter-frame misalignment (maximize feature discrepancy between consecutive frames)
- Test under real CRF23 with official `SAM2VideoPredictor`

### Purpose
This is the minimum check that **any** feature-space attack survives codec. If B also fails, the positive thesis is almost certainly wrong and we pivot to negative-result paper.

**Differentiator from UAP-SAM2**: we test post-codec with official predictor. If B pre-codec works but post-codec collapses, that IS the paper story (Claim 3 validated as a negative result on B, and C+D as the fix).

---

## 3. LOW-COST BASELINE (single GPU, ≤8 hours)

**Setup**: Location B, first conditioning frame only, real CRF23, propagate clean later frames with official predictor.

**Minimal engineering**: No multi-frame unroll. No g_θ training. Direct per-video PGD optimization.

**Purpose**: Fast answer to "can any feature-space perturbation survive codec strongly enough to poison downstream tracking?"

**Pass threshold**: mean dJF_codec > +0.05 on ≥6 valid videos (JF_clean ≥ 0.5)
**Fail threshold**: mean dJF_codec < +0.03 → codec kills even FPN features → move directly to B kill + negative result

---

## 4. SMALLEST DECISIVE EXPERIMENT

**Attack frame 0 only. All later frames clean. Target C+D.**

**Exact setup**:
```
x̃_0 = T(x_0 + δ_0)      # attacked + codec-transformed frame 0
x̃_u = T(x_u)              # clean + codec-transformed frames u ≥ 1

Optimize δ_0:
  - Losses: J_mem_write + J_mem_match + J_ptr (β=3.0)
  - Constraint: ||δ_0||_∞ ≤ 8/255, SSIM ≥ 0.95, LPIPS ≤ 0.10
  - Optimizer: projected Adam, 300 steps, step size ~1/255, 2-3 restarts
  - Per-video direct optimization (NO amortized g_θ)

Evaluate with: official SAM2VideoPredictor.init_state + propagate_in_video
  - Inject adversarial x̃_0 at pixel level before init_state
  - Report: mean dJF_codec on valid subset (JF_clean ≥ 0.5), n ≥ 6 videos
```

**Why frame-0-only is decisive**:
- Frame 0 seeds the entire memory bank and pointer state
- If we cannot poison the initial memory write after CRF23, a multi-frame preprocessor will not rescue the claim
- Cleanest mechanistic diagnostic: either post-codec memory/pointer state moves enough to derail propagation, or it doesn't

**Why projected Adam over PGD**:
- More stable with compound loss (cosine + log-sum-exp + LPIPS + SSIM)
- PGD brittle with EOT on top of relational losses

---

## 5. KILL CRITERIA

### Early kill (primary C+D experiment)
Stop and move to Backup B if:
- Frame-0-only C+D gives **mean dJF_codec < +0.03** on n ≥ 6 valid videos at CRF23
- OR: post-codec memory-token shift (cosine distance) is NOT larger than clean temporal variability by ≥ 1× average
  → memory state is not meaningfully moving under codec

### Full kill (positive-result pivot)
After one C+D run + one B fallback run, pivot to negative-result paper if:
- Best official-predictor result: **mean dJF_codec < +0.08**
- OR: **median dJF_codec < +0.05**
- OR: fewer than **70% of valid videos** show positive dJF_codec

**Negative-result paper requirements** (if kill triggered):
1. Multi-seed, multi-backbone (SAM2-T/S/L) confirmation
2. Feature-space diagnostic: representation-survival ratio r_loc (see §6)
3. Reframe H.264 proxy as "heuristic EOT augmentation" (not differentiable codec)
4. Claim: SAM2 memory path is a natural defense against both pixel-level and feature-space perturbations under real deployment conditions

---

## 6. MECHANISTIC INTERPRETABILITY EXPERIMENTS

For publication-level codec story:

**Representation-survival ratio**:
```
r_loc = ||Δz_loc^{post-codec}|| / ||Δz_loc^{pre-codec}||
```
Compute for B, C, D. Correlate r_loc with dJF_codec.
**Target result**: r_B collapses (≈ 0.1), r_C survives (≈ 0.5+), and only r_C predicts tracking failure.

**DCT-band ablation** (support evidence, not main claim):
- Decompose δ into: low-frequency only / mid-frequency only / high-frequency only
- Re-measure dJF_codec and ΔM for each band
- Expected: low-frequency δ survives codec AND moves maskmem_features; high-frequency δ destroyed by codec

**Frozen-mask diagnostic**:
- Log whether J_mem_write changes are driven by mask changes vs. feature changes
- Required to distinguish "memory-write attack" from "mask attack disguised as memory attack"

---

## 7. DISTINCTNESS FROM UAP-SAM2

**Two-sentence defensible distinction**:
> "UAP-SAM2 shows that pixel-constrained perturbations can distort SAM2 semantics under standard evaluation, but it does not establish that the perturbation survives the mandatory H.264/H.265 release channel or that it corrupts the specific state written into SAM2's streaming memory bank. Our contribution is to attack the memory-write/pointer interface itself under the codec-compressed official `SAM2VideoPredictor`, validating the only deployment-relevant pipeline for publisher-side privacy protection."

**Strong only if**:
1. We actually supervise maskmem_features and key-query compatibility (not just encoder output)
2. We show post-codec results with official VideoPredictor
3. We confirm UAP-SAM2 collapses after CRF23 (must run this baseline explicitly)

**Weak distinctions reviewers will see through**:
- "Our feature loss is different" (not sufficient)
- "We also have a memory misalignment term" (UAP-SAM2 claims this too)

**Risk**: If UAP-SAM2 full paper already supervises written memory tokens, mechanism distinction weakens significantly. Codec/deployment story must carry the paper.

---

## 8. EXPERIMENT EXECUTION ORDER

```
Step 1 (≤8h, 1 GPU):  Low-cost baseline — B, frame-0, direct Adam, CRF23
  → If dJF_codec < 0.03: skip to negative-result paper path
  → If dJF_codec > 0.05: proceed

Step 2 (≤8h, 1 GPU):  Decisive experiment — C+D, frame-0, direct Adam, CRF23
  → If dJF_codec < 0.03: trigger early kill
  → If dJF_codec > 0.05: proceed to full training

Step 3 (≤24h, 1-2 GPU): Full g_θ training with C+D losses, multi-frame clips, CRF sweep (18/23/28)
  + Representation-survival ratio analysis (r_B, r_C, r_D)
  + DCT-band ablation

Step 4 (≤8h):          Baseline: UAP-SAM2 reimplementation, pre-codec vs. post-codec comparison
  → Must show UAP-SAM2 collapses at CRF23 (Claim 3 foundation)

Step 5 (if positive):  Multi-backbone (SAM2-S, SAM2-L) + additional dataset (YT-VOS mini)
```

---

## 9. WHAT THIS PLAN CANNOT CLAIM (HONEST LIMITS)

1. **Not a real-time defense** — offline preprocessor only, ~0.5s/frame
2. **Not universal anonymization** — disrupts SAM2 family specifically; other trackers not tested in round 1
3. **Codec proxy ≠ differentiable H.264** — heuristic EOT (YUV 4:2:0 + blur + noise + resize); must label it as such
4. **If kill criteria triggered**: pivot to negative result, NOT to "our approach works with more compute"
5. **Training surrogate gap remains**: logits.detach() and single-frame memory bank in current train.py mean we haven't ruled out exact unrolled gradients — but this is NOT our approach (we're doing direct δ_0 optimization)
