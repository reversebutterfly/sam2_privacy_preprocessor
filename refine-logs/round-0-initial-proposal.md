# Research Proposal: OMBS — Operator-Mix Boundary Scheduler for Codec-Amplified Privacy Preprocessing

## Problem Anchor
- **Bottom-line problem**: Design a publisher-side video preprocessor that degrades SAM2 tracking after H.264 compression, while bounding visual quality loss.
- **Must-solve bottleneck**: Current methods use a single edit operator (BRS or pixelation) uniformly across the entire boundary. But different boundary regions have different texture, curvature, motion, and codec-block alignment — a single operator cannot be optimal everywhere. The oracle gap (+31pp for BRS, +12pp for pixelation) proves significant room for improvement through spatially non-uniform editing.
- **Non-goals**: (1) Not an adversarial attack in the UAP sense — no L_inf constraint, we do structural editing. (2) Not a general VOS attack framework — scoped to SAM2 + H.264. (3) Not end-to-end differentiable through SAM2.
- **Constraints**: 6× V100, ~8 GPU-hours remaining. Publisher-side: GT mask available. Post-H.264 CRF=23. SSIM ≥ 0.90.
- **Success condition**: A learned boundary scheduler that (1) chooses per-sector operator AND strength, (2) outperforms best uniform operator by ≥5pp on held-out videos, (3) survives H.264.

## Technical Gap

Current approaches treat boundary preprocessing as a single-operator parameter tuning problem: "pick one edit type (BRS or pixelation), pick one strength (α), apply uniformly." Our experiments show:

1. **Operator choice matters enormously**: pixelation (+64pp) vs BRS (+38pp) vs inpainting (+43pp) at matched quality — 26pp gap between operators.
2. **Spatial allocation matters**: anisotropic allocation gives +31pp over uniform BRS, +12pp over uniform pixelation.
3. **But operator choice and allocation are studied separately**: no method jointly selects "which operator" and "how strong" per boundary region.

The gap: **There is no method that jointly optimizes per-region operator selection and strength allocation under a quality budget.**

## Method Thesis

**One-sentence thesis**: At fixed visual quality budget, jointly selecting the edit operator and strength per boundary sector yields significantly more tracking degradation than any single-operator method, because different boundary regions have different codec-amplification profiles.

**Why smallest adequate intervention**: We don't need a large generator or end-to-end training. We just need a structured decision over a small discrete-continuous space (8 sectors × 4 operators × continuous strength), guided by oracle search + distillation.

## Contribution Focus
- **Dominant contribution**: OMBS — a spatiotemporal boundary editing framework that selects per-sector (operator, strength) from a codec-surviving operator library, trained via oracle distillation.
- **Supporting contribution**: Comprehensive matched-quality benchmark showing operator choice × allocation direction are two orthogonal axes of privacy effectiveness.
- **Explicit non-contributions**: Not a new operator (we reuse existing ones). Not end-to-end adversarial training.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: SAM2 (frozen, eval only), H.264 codec (frozen), DINOv2 (not used), all base operators (BRS, pixelation, inpainting, morph_fill)
- **New trainable**: Sector policy network (tiny MLP/transformer, ~50K params)
- **Intentionally excluded**: Dense per-pixel prediction, codec-in-loop differentiable training, SAM2 gradient

### System Overview

```
Input: frame_0, GT mask
  ↓
Feature extraction: per-sector (texture, gradient, curvature, motion, ring area)
  ↓
Sector Policy Network (learned):
  For each of 8 sectors:
    → operator_k ∈ {BRS, pixelation, inpainting, flat_codebook}
    → strength_k ∈ [0, 1]
  ↓
Quality-budget projection: ensure total edit ≤ B (SSIM ≥ 0.90)
  ↓
Apply: each sector edited with its chosen (operator, strength)
  ↓
H.264 CRF=23 encode/decode → SAM2 tracking evaluation
```

### Core Mechanism: Operator Library + Sector Scheduler

**Operator Library** (4 codec-surviving primitives):
1. `flat_mean_brs`: Gaussian-blurred background mean in ring (proven: +38pp)
2. `pixelation`: Downsample/upsample in ring (proven: +64pp, strongest)
3. `inpainting`: cv2.inpaint TELEA on ring boundary (proven: +43pp)
4. `flat_codebook`: 3-level DC codebook per 8×8 block in ring (NEW — hypothesis: richer structure breaks than single flat-mean)

**Sector Scheduler**:
- Input: 60-dim per-video features (same as existing predictor: 4 global + 7×8 per-sector)
- Output: 8 × (operator_logits[4], strength[1]) = 8 × 5 = 40 outputs
- Architecture: MLP (60 → 128 → 128 → 40) with:
  - Gumbel-softmax for discrete operator selection (differentiable training)
  - Sigmoid for continuous strength
  - Budget projection layer (water-filling on strength, given operator-specific quality costs)

### Training Plan

**Stage 1: Oracle Search (offline)**
For 15 DAVIS videos, search over (operator, strength) per sector:
- 8 sectors × 4 operators × 3 strength levels = 96 combinations per sector
- Greedy/beam search: fix sectors 0-7 sequentially, try all options, keep best
- Evaluate with real H.264 + SAM2 (post-codec ΔJF)
- Quality constraint: per-video SSIM ≥ 0.90
- Output: per-video oracle (operator_k, strength_k) labels

**Stage 2: Distillation (fast, ~0.5 GPU-hours)**
Train sector policy network on oracle labels:
- Loss: cross-entropy for operator + MSE for strength
- Pairwise ranking loss: predicted combo must beat uniform pixelation
- Gate: low confidence → fallback to uniform pixelation (safe default)

### Quality Budget Projection

Each operator has a pre-calibrated quality cost function:
- `cost(operator, strength, sector_area)` → expected SSIM reduction
- Calibrated offline on 10 videos

Budget constraint: `Σ_k cost(op_k, s_k, area_k) ≤ B`
Enforced via: strength clipping + iterative projection (same as existing iso-budget framework)

### Inference Path
1. Extract per-sector features from frame 0 + mask (no SAM2 needed)
2. Forward through policy network → (op_k, s_k) for each sector
3. Apply gate: if confidence < threshold → uniform pixelation fallback
4. Apply per-sector edits with chosen operators
5. H.264 compress → release video

### Failure Modes and Diagnostics
- **Oracle search too noisy**: Greedy search may miss good combos → add random restarts
- **Operator costs miscalibrated**: Different videos have different cost curves → validate on held-out
- **Policy overfits to training videos**: n=20-30 may not be enough → use leave-one-out CV + gate
- **Flat codebook operator doesn't help**: May collapse to "always pixelation" → ablation will tell

### Novelty and Elegance Argument

**Closest work**: Our own anisotropic allocation (8-sector alpha predictor).
**Key difference**: That predicted only strength; OMBS predicts both operator AND strength — a joint discrete-continuous decision.
**Why not trivial**: (1) Different operators have different quality-cost curves → budget projection is operator-aware. (2) The oracle search space is combinatorial (4^8 = 65536 operator combos × continuous strength). (3) The policy must learn "this sector has high texture → pixelation; that sector has smooth boundary → flat_mean."

## Claim-Driven Validation Sketch

### Claim 1: Joint operator+strength selection outperforms best uniform operator
- Experiment: OMBS oracle vs uniform pixelation vs uniform BRS (matched SSIM)
- Metric: ΔJF_codec, win-rate
- Baseline: best uniform operator at matched quality
- Expected: OMBS oracle ≥ +5pp over uniform pixelation

### Claim 2: Learned policy recovers significant portion of oracle gap
- Experiment: 5-fold CV, policy network vs oracle vs uniform
- Metric: oracle gap closure, win-rate
- Expected: ≥30% closure, ≥80% win-rate (matching prior predictor results)

### Claim 3: Operator selection is content-dependent (not always pixelation)
- Experiment: Analyze oracle operator distribution across videos/sectors
- Metric: operator diversity, correlation with boundary features
- Expected: ≥2 operators selected in ≥60% of videos

## Experiment Handoff Inputs
- Must-prove: Claim 1 (oracle gap exists for joint selection)
- Must-run ablation: single-operator oracle vs multi-operator oracle
- Critical dataset: DAVIS (n=30+), matched-quality protocol with SSIM+LPIPS
- Highest-risk assumption: operator diversity (may collapse to "always pixelation")

## Compute & Timeline Estimate
- Oracle search: ~12 GPU-hours (15 videos × 96 evals × 30s, parallelizable to ~2h wall on 6 GPUs)
- Policy training: ~0.5 GPU-hours
- Evaluation (CV + ablations): ~3 GPU-hours
- Total: ~15.5 GPU-hours (~3h wall time on 6 GPUs)
