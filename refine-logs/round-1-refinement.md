# Round 1 Refinement

## Problem Anchor
[Verbatim from round 0]
- **Bottom-line problem**: Publisher-side video preprocessor that degrades SAM2 tracking after H.264, while bounding quality loss.
- **Must-solve bottleneck**: Single operator applied uniformly is suboptimal. Different boundary regions need different operators.
- **Non-goals**: Not adversarial attack, not general VOS, not end-to-end differentiable through SAM2.
- **Constraints**: 6× V100 (no hard time limit). GT mask. Post-H.264 CRF=23. SSIM ≥ 0.90.
- **Success condition**: Learned scheduler choosing per-sector operator + strength, ≥5pp over best uniform.

## Anchor Check
- Original bottleneck: single operator uniformly applied is suboptimal
- Revised method still addresses it: yes — OMBS routes between operators per sector
- Reviewer suggestions rejected as drift: none

## Simplicity Check
- Dominant contribution after revision: **mixed-operator routing over 2 proven operators under exact post-codec quality constraint**
- Components removed:
  - `flat_codebook` operator → deleted entirely (contradicts "not a new operator")
  - `inpainting` → removed from initial version (add later if 2-op routing works)
  - Continuous strength optimization → deferred (use fixed calibrated strengths first)
  - "Spatiotemporal" framing → removed (system uses frame_0 only)
  - Operator diversity as success criterion → dropped
  - Pairwise ranking loss → dropped (CE + MSE sufficient)
  - Gumbel-softmax → replaced with hard assignment at inference, CE at train
  - Benchmark as co-equal contribution → demoted to supporting evidence
- Why smallest adequate: just routing between 2 operators per sector with fixed strengths

## Changes Made

### 1. Reduced operator library to 2
- Reviewer said: 4 operators is too many, flat_codebook contradicts scope, start with 2
- Action: library = {BRS, pixelation} only. These are the two most different operators (flat fill vs mosaic). Inpainting is close to BRS and can be added as ablation later.
- Impact: search space drops from 4^8 to 2^8 = 256 total operator combos

### 2. Fixed calibrated strengths (no continuous optimization)
- Reviewer said: continuous strength adds complexity without proving routing works first
- Action: each operator has ONE pre-calibrated strength matched to SSIM≈0.93. BRS: α=0.80. Pixelation: block_size=16. The decision is purely "which operator per sector."
- Impact: search space = 2^8 = 256, can be exhaustively evaluated on a few videos

### 3. Exact post-codec quality constraint
- Reviewer said: additive sector costs are wrong due to codec block interactions
- Action: no proxy budget. After composing all sector edits, compute actual post-H.264 SSIM on the full frame. Reject candidates with SSIM < 0.90.
- Impact: slower oracle search but bulletproof fairness

### 4. Four-row ablation table isolating operator routing
- Reviewer said: must separate routing gain from strength gain
- Action: the ONE decisive table is:
  1. Uniform pixelation (α matched) — current best single-op
  2. Anisotropic pixelation oracle (strength-only routing) — already have data
  3. Uniform operator-mix (random 50/50 BRS/pixel per sector, matched quality)
  4. **OMBS oracle** (per-sector operator choice) — new
- Impact: row 4 > row 2 proves operator routing adds value beyond strength routing

### 5. Locked claims table
- Reviewer said: internal contradictions need resolution
- Action:
  - Input: frame_0 RGB + GT mask (NOT "spatiotemporal")
  - Operators: {BRS, pixelation} only (NOT new operators)
  - Quality: exact post-H.264 SSIM ≥ 0.90 (NOT additive proxy)
  - Claim: "operator routing is an additional degree of freedom beyond strength allocation"
  - Non-claim: "we invent new operators" / "we solve general VOS attack"

## Revised Proposal

# OMBS-Lite: Operator Routing for Codec-Amplified Privacy Preprocessing

## Problem Anchor
[Same as above]

## Technical Gap
Operator choice (pixelation vs BRS: 26pp gap) and spatial allocation (+31pp gap) are studied separately. No method jointly selects which operator to use per boundary region.

## Method Thesis
At fixed post-codec quality, per-sector operator routing between two proven codec-surviving operators yields more tracking degradation than either operator alone, because different boundary textures have different codec-amplification profiles.

## Contribution Focus
- **Dominant**: Per-sector operator routing under exact post-codec quality constraint
- **Supporting**: Matched-quality operator benchmark (pixelation > BRS > inpainting > blur)
- **Non-contributions**: No new operators. No continuous strength optimization (deferred). No temporal strategy.

## Proposed Method

### Operator Library (2 frozen operators, fixed strengths)
1. `BRS` (flat-mean background fill, α=0.80): low-entropy plateau → DCT ringing
2. `Pixelation` (block_size=16, downsample/upsample): mosaic → high-frequency destruction

Strengths pre-calibrated so each operator individually achieves SSIM ≈ 0.93.

### 8-Sector Boundary Partition
Same angular partition as prior anisotropic work. Binary decision per sector: BRS or pixelation.

### Oracle Search (Training Data)
For each of 15 DAVIS training videos:
1. Enumerate all 2^8 = 256 operator assignments
2. For each assignment:
   a. Apply: sector k gets operator[assignment[k]] at fixed strength
   b. H.264 CRF=23 encode/decode
   c. SAM2 tracking → ΔJF_codec
   d. Compute actual full-frame SSIM
   e. Reject if SSIM < 0.90
3. Pick assignment with max ΔJF among quality-valid candidates
4. Cost: 256 SAM2 evals × 15 videos = 3840 evals × ~30s = ~32 GPU-hours
   Parallelized on 6 GPUs: ~5.5h wall time

Output: per-video oracle binary vector b* ∈ {0,1}^8 (0=BRS, 1=pixelation)

### Sector Scheduler Network
- Input per sector: 7-dim local features (ring area, mean RGB×3, gradient, curvature, motion) + 4-dim global (mask area, aspect, compactness, boundary length) = 11-dim
- Architecture: shared MLP (11 → 64 → 64 → 1 logit), applied independently to each sector
- Output: sigmoid → p(pixelation). Threshold at 0.5.
- Gate: if max(ΔJF across sectors) < 2pp → fallback to uniform pixelation
- Parameters: ~4.5K (tiny)
- Training: BCE loss on oracle binary labels, 300 epochs, Adam lr=1e-3

### Quality Enforcement
No proxy. After applying the full 8-sector edit:
1. Compute actual SSIM on full edited frame vs original
2. If SSIM < 0.90: reduce strength of the most aggressive sectors until constraint met
3. This is exact and codec-interaction-aware

### Inference Path
1. Extract per-sector features from frame 0 + mask (~10ms)
2. Forward scheduler → 8 binary decisions (~1ms)
3. Apply BRS/pixelation per sector (~50ms/frame)
4. Verify SSIM ≥ 0.90
5. H.264 compress → release

## Claim-Driven Validation

### The ONE decisive table:
| Row | System | Quality Control | What It Tests |
|-----|--------|----------------|---------------|
| 1 | Uniform pixelation (bs=16) | SSIM matched | Best single-op baseline |
| 2 | Anisotropic pixelation oracle (strength routing) | Iso-budget | Strength allocation only |
| 3 | Uniform random operator-mix (50/50) | SSIM matched | Naive mixing |
| 4 | **OMBS oracle** (per-sector BRS/pixel routing) | SSIM ≥ 0.90 | **Operator routing** |

**Success**: Row 4 > Row 2 by ≥5pp proves operator routing adds value beyond strength routing.

### Learned scheduler CV:
5-fold on 15 videos (12 train / 3 test). Metric: gap closure ≥ 30%, win-rate ≥ 80%.

### Ablation:
- What % of oracle assignments are pure-pixelation vs mixed? (diagnostic, not success criterion)
- Per-sector: which features predict operator choice?

## Compute
- Oracle search: ~32 GPU-hours (5.5h wall on 6 GPUs)
- Scheduler training: ~0.5 GPU-hours
- CV evaluation: ~4 GPU-hours
- Total: ~36.5 GPU-hours (~7h wall time)
