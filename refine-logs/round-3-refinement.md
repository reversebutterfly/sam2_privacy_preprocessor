# Round 3 Refinement — CIGR Upgrade

## Problem Anchor
[Verbatim]
- Bottom-line problem: Publisher-side video preprocessor degrades SAM2 tracking after H.264, bounding quality loss.
- Must-solve bottleneck: Single operator uniformly applied is suboptimal. Different boundary regions need different operators.
- Constraints: 6× V100 (no hard time limit). GT mask. Post-H.264 CRF=23. SSIM ≥ 0.90.
- Success condition: Per-sector operator routing ≥5pp over best single-operator method.

## Anchor Check
- Still solving same problem: yes
- Key upgrade: from "oracle search + distill" to "structured optimization method"

## Change: OMBS-Lite → CIGR

User feedback: 7.8/10 ceiling is too low. The issue is not experiment volume but **method novelty**. "oracle search + tiny MLP distill" is the same pattern as prior anisotropic allocation work. Need a principled method, not just a larger search space.

## Revised Proposal

# CIGR: Codec-Interaction Graph Router for Privacy Preprocessing

## Problem Anchor
[Same]

## Technical Gap

Current operator routing treats each boundary sector independently: "sector k should use BRS or pixelation?" But adjacent sectors **interact** through H.264's block-based compression:

1. **Seam artifacts**: When sector k uses BRS (flat fill) and sector k+1 uses pixelation (mosaic), the transition between them creates an additional DCT block mismatch — potentially STRONGER than either operator alone.
2. **Quality interference**: Two adjacent high-strength sectors may jointly exceed the quality budget even though each individually passes.
3. **Codec block sharing**: A single 8×8 DCT block may span two sectors — the operator choice for both sectors jointly determines the block's encoding.

These are **pairwise interactions** that an independent per-sector classifier cannot capture. This is the missing mechanism.

## Method Thesis

Per-sector operator routing is a **structured prediction problem with pairwise interactions**, not an independent classification problem. By modeling sector-sector codec interactions as edges in a graph energy model, we can find provably better operator assignments than any factored (independent) predictor.

## Contribution Focus
- **Dominant contribution**: CIGR — a graph-structured energy model for operator routing, with codec-aware pairwise terms and exact inference via circular graph DP
- **Supporting**: empirical evidence that pairwise interactions improve over independent routing
- **Non-contributions**: no new operators, no continuous strength, no temporal strategy

## Proposed Method

### Energy Model

For 8 sectors arranged in a circular graph (sector 0 adjacent to sector 7):

```
E(z) = Σ_i  ψ_i(z_i)  +  λ Σ_(i,i+1)  φ_{i,i+1}(z_i, z_{i+1})
```

where:
- z_i ∈ {BRS, pixelation} — operator choice for sector i
- ψ_i(z_i) — **unary potential**: how effective is operator z_i at sector i?
- φ_{i,i+1}(z_i, z_{i+1}) — **pairwise potential**: does the (z_i, z_{i+1}) combination create extra codec seam artifacts or quality interference?
- λ — interaction strength (learned or cross-validated)

**Inference**: find z* = argmax E(z). For a circular graph with binary nodes, this is solvable in O(2^2 × 8) = O(32) via circular DP — exact, no approximation needed.

### Unary Potentials ψ_i(z_i)

Computed from per-sector features via a tiny shared MLP:

```
ψ_i(z_i) = MLP_unary([local_features_i; global_features])[z_i]
```

- Input: 11-dim (7 local + 4 global), same features as OMBS-Lite
- Output: 2 scores (one per operator)
- Architecture: 11 → 64 → 2
- Parameters: ~1K

### Pairwise Potentials φ_{i,i+1}(z_i, z_{i+1})

This is the **core novelty**. The pairwise term captures codec-mediated interactions:

```
φ_{i,i+1}(z_i, z_{i+1}) = MLP_pair([feat_i; feat_{i+1}; codec_boundary_feat])[z_i * 2 + z_{i+1}]
```

- Input: local features of both sectors + codec boundary features at their shared edge
- **Codec boundary features** (new, per sector-pair):
  - Number of 8×8 DCT blocks spanning the sector boundary
  - Mean texture contrast across the boundary
  - Boundary curvature at the transition point
- Output: 4 scores (2×2 operator combinations: BRS-BRS, BRS-PIX, PIX-BRS, PIX-PIX)
- Architecture: 25 → 64 → 4
- Parameters: ~2K

Total model parameters: ~3K (unary) + ~2K (pairwise) = **~5K** (tiny)

### Training

**Stage 1: Data Collection**

For 15 DAVIS videos, exhaustively evaluate all 2^8 = 256 operator assignments (same as OMBS-Lite):
- Apply per-sector edits → H.264 CRF=23 → SAM2 tracking → ΔJF
- Compute actual post-H.264 SSIM
- This gives 256 labeled (z, ΔJF, SSIM) tuples per video

**Stage 2: Energy Model Fitting**

From the 256 evaluations per video, fit the energy model parameters:

Option A — **Structured SVM / Ranking**:
- For each pair (z_good, z_bad) where ΔJF(z_good) > ΔJF(z_bad) + margin:
  - Loss: max(0, margin - E(z_good) + E(z_bad))
- This learns potentials so that high-ΔJF assignments have high energy

Option B — **Direct regression**:
- Learn E(z) ≈ ΔJF(z) directly as a structured regression
- Loss: MSE(E(z), ΔJF(z)) over all 256 assignments per video

Option C — **Log-linear energy + conditional likelihood**:
- P(z|x) ∝ exp(E(z|x))
- Loss: -log P(z*|x) where z* is the oracle assignment
- Standard structured prediction

Recommend **Option A (Structured Ranking)**: most robust to noise, doesn't require modeling absolute ΔJF values.

### Inference

At test time:
1. Extract per-sector features + codec boundary features (~10ms)
2. Compute all unary potentials: 8 × MLP_unary (~1ms)
3. Compute all pairwise potentials: 8 × MLP_pair (~1ms)
4. Circular DP to find z* = argmax E(z) (~0.01ms, exact)
5. Apply per-sector edits with z* (~50ms/frame)
6. Verify SSIM ≥ 0.90

Total inference: ~62ms per frame — real-time capable.

### Quality Enforcement

Same as OMBS-Lite: exact post-H.264 SSIM on full frame. If z* violates SSIM < 0.90:
- Fallback: flip the highest-energy sector to the lower-strength operator
- Repeat until constraint met (at most 8 flips)

### Why This Is a Method Contribution (Not Just Bigger Search)

| | OMBS-Lite (before) | CIGR (now) |
|---|---|---|
| Decision model | Independent per-sector | **Structured graph model with pairwise terms** |
| Training | Label imitation (CE+MSE) | **Structured ranking / conditional likelihood** |
| Inference | Independent sigmoid → threshold | **Exact circular DP** |
| Modeling | Treats sectors as independent | **Models codec-mediated sector interactions** |
| Paper framing | "Oracle discovery + distill" | **"Structured prediction for codec-aware routing"** |

The novelty is NOT "search a bigger space." It is: **formalize operator routing as a structured prediction problem, where the graph structure captures codec block interactions between adjacent sectors.**

### Novelty and Elegance Argument

**Closest work**: Our own OMBS-Lite (independent per-sector routing).
**Key difference**: CIGR models pairwise codec interactions — adjacent sectors jointly determine codec artifacts at their shared boundary.
**Why elegant**: The circular graph structure is natural (sectors form a ring around the object boundary), the energy model is clean (unary + pairwise), inference is exact (DP on a cycle), and the whole thing has ~5K parameters.
**Why it's not just MRF-as-buzzword**: The pairwise term has a specific physical meaning (codec block seam at sector boundary), not a generic smoothness prior.

## Claim-Driven Validation

### Claim 1: Structured routing > Independent routing
- CIGR oracle vs OMBS-Lite oracle (same search data, different model)
- If CIGR finds better assignments from the same 256 candidates → pairwise terms help
- Expected: CIGR extracts ≥2pp more from the same data

### Claim 2: Learned CIGR > Best single-operator
- 5-fold CV: CIGR router vs uniform pixelation
- Expected: ≥5pp gain, ≥80% win-rate

### Ablation: Pairwise terms necessary?
- CIGR (unary+pairwise) vs CIGR-unary-only (= OMBS-Lite with graph notation)
- If pairwise helps → validates the codec interaction hypothesis
- If not → reduces to OMBS-Lite (still useful, just less novel)

## Compute
- Oracle search: ~32 GPU-hours (same as OMBS-Lite, exhaustive 256 per video)
- Energy model fitting: ~0.5 GPU-hours
- CV evaluation: ~4 GPU-hours
- Total: ~36.5 GPU-hours (~7h wall on 6 GPUs)
