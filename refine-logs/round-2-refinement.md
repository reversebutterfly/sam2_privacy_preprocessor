# Round 2 Refinement

## Problem Anchor
[Verbatim]
- Bottom-line problem: Publisher-side video preprocessor degrades SAM2 tracking after H.264, bounding quality loss.
- Must-solve bottleneck: Single operator uniformly applied is suboptimal.
- Constraints: 6× V100 (no hard time limit). GT mask. Post-H.264 CRF=23. SSIM ≥ 0.90.
- Success condition: Per-sector operator routing ≥5pp over best single-operator method.

## Anchor Check
- Still solving the right problem: yes
- Drift: none. Reviewer explicitly confirmed "scope narrowing, not drift"

## Simplicity Check
- Dominant contribution: mixed binary operator routing under exact post-codec quality
- Nothing to remove — already at minimum
- Fix: sharpen the ablation definitions and elevate learned scheduler

## Changes Made

### 1. Redefined ablation Row 2
- Reviewer said: "Row 2 is not well-defined under binary BRS/pixel setup"
- Action: Row 2 is now "Best single-operator anisotropic oracle" — for EACH of BRS and pixelation separately, run the existing anisotropic strength allocation (8-sector α optimization), then take the better one. This is the strongest possible single-operator baseline.
- Impact: clean comparison — Row 4 (mixed routing) vs Row 2 (best strength-only routing within one operator)

### 2. Replaced random mix baseline with deterministic alternating
- Reviewer said: "random 50/50 is a strawman"
- Action: Row 3 is now "alternating operator pattern" — sectors 0,2,4,6 get pixelation, sectors 1,3,5,7 get BRS (or vice versa, take the better one). This is a deterministic, non-learned mixing baseline.
- Impact: harder baseline — any pattern advantage without learning

### 3. Elevated learned scheduler to first-class claim
- Reviewer said: "reads as oracle discovery + tiny distillation appendix"
- Action: restructured claims:
  - Claim 1 (discovery): OMBS oracle > best single-op anisotropic oracle
  - Claim 2 (method): learned scheduler recovers the gap on held-out videos
  - These are equal-weight claims, not primary+appendix
- Impact: the paper now has "discovery + usable method" structure (like AdvOpt → global-fixed was "discovery + deployment")

## Revised Proposal

# OMBS-Lite: Binary Operator Routing for Codec-Amplified Privacy Preprocessing

## Problem Anchor
[Same]

## Method Thesis
Under exact post-codec quality matching, per-sector binary operator routing between BRS and pixelation yields more tracking degradation than the best single-operator anisotropic allocation, because different boundary regions have fundamentally different codec-amplification profiles.

## Contribution Focus
- **Claim 1 (Discovery)**: Mixed operator routing is a new degree of freedom beyond strength allocation — oracle evidence on n=15 DAVIS videos
- **Claim 2 (Method)**: A tiny learned per-sector router (4.5K params) recovers the routing gap on held-out videos with ≥80% win-rate
- **Non-claims**: No new operators. No continuous strength optimization. No temporal strategy.

## Method

### Operators (2, frozen, fixed strengths)
1. `BRS` (flat-mean, α=0.80) — low-entropy plateau → DCT ringing
2. `Pixelation` (bs=16) — mosaic → high-freq boundary destruction

Both pre-calibrated to SSIM ≈ 0.93 individually.

### Decision Space
Per sector k ∈ {0..7}: binary choice b_k ∈ {BRS, pixelation}.
Total: 2^8 = 256 possible assignments.

### Oracle Search
For each of 15 DAVIS videos:
1. Enumerate all 256 assignments
2. For each: apply per-sector edits → H.264 CRF=23 → decode → SAM2 tracking → ΔJF
3. Compute actual post-H.264 full-frame SSIM
4. Reject if SSIM < 0.90
5. Pick max ΔJF among valid candidates
6. Cost: 256 × 15 = 3840 SAM2 evals × ~30s ≈ 32 GPU-hours (5.5h wall on 6 GPUs)

### Learned Sector Router
- **Input**: 11-dim per sector (7 local: ring area, RGB×3, gradient, curvature, motion + 4 global: mask area, aspect, compactness, boundary length)
- **Architecture**: shared MLP (11 → 64 → 64 → 1), sigmoid output = p(pixelation)
- **Training**: BCE on oracle binary labels, 300 epochs, Adam 1e-3
- **Gate**: max confidence < threshold → fallback uniform pixelation
- **Parameters**: ~4.5K

### Quality Enforcement
Exact: compute actual SSIM on fully-edited frame after composing all 8 sectors. No proxy.

## The Decisive Ablation Table

| Row | System | Operator(s) | Per-Sector? | Quality | Tests |
|-----|--------|------------|-------------|---------|-------|
| 1 | Uniform pixelation | pixel only | No (uniform) | SSIM matched | Single-op baseline |
| 2 | Best aniso single-op oracle | pixel OR brs (better one) | Yes (strength) | Iso-budget | Strength routing ceiling |
| 3 | Alternating pattern | pixel+brs (0,2,4,6 vs 1,3,5,7) | Yes (fixed pattern) | SSIM ≥ 0.90 | Deterministic mixing |
| 4 | **OMBS oracle** | pixel+brs (optimized) | Yes (operator choice) | SSIM ≥ 0.90 | **Operator routing** |
| 5 | **OMBS learned** | pixel+brs (predicted) | Yes (learned) | SSIM ≥ 0.90 | **Usable method** |

**Decisive comparisons**:
- Row 4 > Row 2 proves operator routing adds value beyond strength routing
- Row 5 > Row 1 proves learned router is practically useful
- Row 5 closure toward Row 4 quantifies learnability

## Compute
- Oracle search: ~32 GPU-hours (5.5h wall)
- Aniso single-op oracle (Row 2): already have data for pixelation (+12pp gap); BRS (+31pp gap)
- Alternating pattern (Row 3): ~1 GPU-hour (15 videos × 2 patterns × 30s)
- Router training + CV: ~1 GPU-hour
- Total: ~34 GPU-hours (~6h wall)
