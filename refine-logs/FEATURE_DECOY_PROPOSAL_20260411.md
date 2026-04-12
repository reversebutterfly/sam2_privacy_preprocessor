# CoDecoy: Codec-Aware Feature Decoy Tube Generation for SAM2 Misdirection

## 1. One-Line Thesis

CoDecoy learns a target-conditioned background decoy tube whose low-frequency appearance is aligned with the true object in frozen DINOv2 and SAM2 feature space after codec compression, so that SAM2 drifts to the decoy and then rewrites its own memory around the wrong region.

## 2. Problem Anchor

- Bottom-line problem: publisher-side privacy preprocessing for SAM2 video tracking should remain effective after real H.264 compression while preserving visual quality.
- Must-solve bottleneck: boundary suppression only removes evidence around the ground-truth object and tends to saturate under SSIM / LPIPS constraints; pixel-level feature attacks collapse after codec.
- Non-goals: universal attack without masks, real-time on-device inference, cross-tracker universality in the first paper.
- Constraints: GT mask available; frozen DINOv2 / SAM2 allowed; 6 x V100; about 15 aggregate GPU-hours; target quality SSIM >= 0.90 and LPIPS <= 0.10.
- Success condition: at matched quality, the learned decoy method beats uniform BRS and UAP-style baselines under real H.264, with a clean mechanism story and ablations showing why decoy generation matters.

## 3. Why Decoy Induction Instead of Boundary Breaking

Boundary breaking is subtractive. It weakens local evidence near the true target, but SAM2 still has a prompt-conditioned memory trace and temporal continuity to recover from partial boundary degradation. A feature decoy is additive: it creates a competing region whose representation looks target-like to the tracker. For memory-based tracking, this is usually the more catastrophic failure mode because once the predictor locks onto the wrong region, later memory writes reinforce the error.

This is especially attractive in the publisher-side setting:

1. We know the real target appearance from the GT mask, so we can synthesize a targeted false attractor instead of blindly perturbing pixels.
2. Low-frequency structured appearance transfer survives H.264 far better than high-frequency adversarial noise.
3. A false target can be placed in clean background where small edits are harder to notice than directly corrupting the object contour.

## 4. Method Overview

### 4.1 Dominant Contribution

A target-conditioned, codec-aware decoy generation pipeline with two trainable modules:

1. **Decoy Tube Selector**: choose a temporally stable background tube where a false target is most likely to attract SAM2.
2. **Low-Frequency Decoy Renderer**: synthesize a subtle pseudo-object inside that tube so that its frozen DINOv2 and SAM2 features match the true target after codec compression.

### 4.2 Frozen Components

- DINOv2 backbone for semantic feature alignment.
- SAM2 image encoder and video predictor for tracker-specific feature alignment and attack supervision.
- Optical flow or point tracking for candidate background tubes.
- Differentiable codec proxy during training; real H.264 in validation and final evaluation.

### 4.3 Data Flow

Given video frames `x_1:T` and GT masks `m_1:T`:

1. Extract target prototype from frame 1 using the GT mask.
2. Enumerate background candidate tubes across frames 2:K.
3. Score each candidate tube with the selector and pick the best one.
4. Render a low-frequency pseudo-object into the selected tube across frames.
5. Pass edited frames through codec-EOT and frozen SAM2 propagation.
6. Optimize the selector and renderer so that:
   - the decoy region becomes feature-similar to the real target,
   - SAM2 shifts prediction and attention toward the decoy,
   - edits remain codec-stable and perceptually acceptable.

## 5. Architecture

### 5.1 Target Prototype Encoder

Use frozen DINOv2 and SAM2 encoders to build a target representation from frame 1:

- `q_d`: masked average pooled DINOv2 tokens inside `m_1`
- `q_s`: masked average pooled SAM2 encoder / FPN tokens inside `m_1`
- `q = P([q_d ; q_s])`: a learned 256-d joint prototype used by both trainable modules

Optional context: average the same pooled features over the first 2 to 3 clean frames warped by GT masks to reduce single-frame noise.

### 5.2 Decoy Tube Selector

Generate `N` candidate background tracklets `tau_j` from background points or patches:

- start from a coarse background grid or superpixels,
- remove any location overlapping GT or within a safety margin around GT,
- propagate each point by optical flow over frames 2:K,
- retain only tracklets with stable visibility and sufficient area.

For each candidate tube, pool frozen DINOv2 and SAM2 features along the tracklet and feed them to a lightweight scorer:

- input: `q`, pooled candidate features, motion stability, distance to GT, local entropy, local texture risk
- network: 2-layer transformer or MLP with temporal pooling
- output: decoyability score `s_j` and an initial ellipse parameterization `(cx_t, cy_t, sx_t, sy_t, theta_t)`

The selector is trained with pseudo-labels from an offline oracle search and later fine-tuned end-to-end with the full attack loss.

### 5.3 Low-Frequency Decoy Renderer

The renderer operates on cropped tubes rather than full frames.

Inputs per frame:

- target low-frequency appearance template from the true object crop,
- target prototype `q`,
- current background crop on the selected tube,
- previous renderer hidden state for temporal coherence.

Network:

- a small U-Net or ConvGRU decoder at `64 x 64`,
- cross-attention from target prototype / target crop tokens to background crop tokens,
- two output heads:
  - `A_t`: soft alpha mask for the decoy shape
  - `R_t`: low-resolution YUV residual / template coefficients
- optional warp head `W_t` to deform the target appearance slightly toward background geometry

Codec-aware parameterization:

- Y channel generated at `32 x 32`
- U and V generated at `16 x 16`
- all outputs are bilinearly upsampled and low-pass filtered before compositing

Rendering:

`D_t = LP( Warp(T_low, W_t) + R_t )`

`x'_t = x_t + M_t * (D_t - crop(x_t))`

where `M_t = upsample(A_t)` and `LP` is a fixed low-pass filter. The edited crop is pasted back into the full frame.

This forces the edit to live in the frequency band most likely to survive H.264.

## 6. Training Objective

Let `C_psi(.)` denote differentiable codec-EOT and `SAM2(.)` denote frozen video propagation with a clean frame-1 prompt. Training uses clips of length `K = 4` or `6`.

### 6.1 Feature Alignment Loss

Make the decoy region feature-similar to the true target:

- `z_d^t = pool(E_d(C_psi(x'_t)), M_t)`
- `z_s^t = pool(E_s(C_psi(x'_t)), M_t)`

`L_align = sum_t [1 - cos(z_d^t, q_d)] + lambda_s * sum_t [1 - cos(z_s^t, q_s)]`

### 6.2 Contrastive Hard-Negative Loss

Ensure the decoy is closer to the target than nearby background regions:

`L_nce = sum_t InfoNCE(q_d, z_d^t, negatives = hard background candidates)`

Hard negatives are taken from top-scoring background patches before editing.

### 6.3 SAM2 Diversion Loss

Run SAM2 on the codec-transformed edited clip. Let `y_t` be the predicted soft mask.

`L_divert = sum_t [ softIoU(y_t, m_t) - alpha * softIoU(y_t, M_t) ]`

Minimizing `L_divert` reduces GT overlap and increases decoy overlap.

### 6.4 Memory / Attention Competition Loss

If memory attention maps are accessible, collapse attention over heads and encourage higher mass on the decoy than on GT:

`L_attn = sum_t [ AttnMass(A_t, m_t) - AttnMass(A_t, M_t) ]`

If direct attention readout is not available, replace this with the same mass difference computed on decoder logits or on query-decoy similarity in the SAM2 feature space.

### 6.5 Temporal and Geometric Regularization

- `L_temp`: temporal consistency of `A_t`, `R_t`, and warp parameters under optical-flow warping
- `L_area`: keep decoy area within a ratio band of the true object area
- `L_sep`: barrier term to keep the decoy outside GT and outside a fixed exclusion ring

### 6.6 Perceptual and Codec Regularization

- `L_ssim = max(0, 0.90 - SSIM(x'_t, x_t))`
- `L_lpips = max(0, LPIPS(x'_t, x_t) - 0.10)`
- `L_hf`: penalize high-frequency energy of the residual, especially on chroma channels
- `L_tv`: total variation on `A_t` to avoid noisy edges

### 6.7 Total Loss

`L_total =`

`lambda_1 * L_align + lambda_2 * L_nce + lambda_3 * L_divert + lambda_4 * L_attn +`

`lambda_5 * L_temp + lambda_6 * L_area + lambda_7 * L_sep +`

`lambda_8 * L_ssim + lambda_9 * L_lpips + lambda_10 * L_hf + lambda_11 * L_tv`

Recommended training schedule:

1. warm up selector with oracle pseudo-labels,
2. pretrain renderer on `L_align + L_nce + quality`,
3. joint train with `L_divert + L_attn` under codec-EOT.

## 7. Training Data and Procedure

### 7.1 Data

- DAVIS-2017 train split for fast iteration and clean masks
- optional YT-VOS mini subset for diversity
- use only short clips for training, not full-video unrolls

### 7.2 Oracle Pseudo-Labels for Selector

Before training the selector, run a small offline search over candidate tubes:

1. sample 16 to 32 candidate background tracklets per clip,
2. apply a cheap feature-only decoy synthesis objective,
3. re-score top candidates with a small number of frozen SAM2 evaluations,
4. keep the best candidate as the selector target.

This gives the selector a real learning problem instead of relying on hand-crafted heuristics.

### 7.3 Three-Stage Optimization

**Stage A: selector pretraining**

- input: candidate tubes plus target prototype
- target: oracle best tube
- loss: cross-entropy or pairwise ranking

**Stage B: renderer pretraining**

- target: maximize post-codec DINOv2 / SAM2 feature similarity
- no SAM2 unroll yet
- purpose: learn codec-stable target-like decoys before full attack training

**Stage C: joint attack training**

- freeze DINOv2 and SAM2
- unroll SAM2 for 4 to 6 frames
- optimize full loss with codec-EOT

## 8. Inference Pipeline

At deployment time:

1. compute target prototype from the first frame and GT mask,
2. build candidate background tubes on the clip,
3. select the best tube with the selector,
4. render the decoy along that tube for frames 2:T,
5. encode the edited video with real H.264.

No test-time optimization is required.

## 9. Module-by-Module Design Choices

### 9.1 Decoy Location Selection

The selector should favor tubes that are:

- outside GT with a large spatial margin,
- temporally stable under flow,
- large enough to host an object-sized pseudo-target,
- not already highly salient,
- feature-compatible with the target after light editing.

This is exactly where a learned scorer is stronger than a centroid-distance heuristic. The scorer can learn that some apparently easy locations are visually risky or poor after codec, while some textured but stable surfaces are much better hosts for false objects.

### 9.2 Decoy Content Generation

The renderer should not hallucinate arbitrary high-frequency patterns. It should synthesize a coarse pseudo-object by transporting low-frequency target appearance and blending it into the selected background patch. The decoy does not need pixel-level resemblance; it needs representation-level resemblance under frozen DINOv2 and SAM2.

The most important design principle is that the decoy should look like a plausible low-contrast object or contour fragment in the scene, not like adversarial noise.

### 9.3 Quality Control

Quality is enforced by:

- hinge constraints on SSIM and LPIPS,
- compact support via `L_area`,
- soft shape via `L_tv`,
- low-frequency-only residual parameterization,
- temporal smoothness across frames.

### 9.4 Codec Survival

Codec survival is enforced at four levels:

1. YUV low-resolution output parameterization
2. explicit low-pass rendering
3. differentiable codec-EOT during training
4. spectral penalty against high-frequency chroma residuals

This is the core distinction from UAP-style feature attacks whose success is mostly pre-codec and high-frequency.

## 10. How DINOv2 and SAM2 Are Used Together

The cleanest role split is:

- **DINOv2**: semantic attractor. It tells the renderer what background patch should become more object-like in a generic feature space.
- **SAM2 encoder / predictor**: tracker-specific attractor. It tells the renderer and the attack objective what actually competes with SAM2's readout and propagation mechanism.

Why both matter:

- DINO-only may produce semantically target-like patches that do not actually hijack SAM2.
- SAM2-only may overfit a fragile tracker-specific signal and become less stable after codec or scene changes.
- The combination gives semantic similarity plus tracker compatibility.

## 11. Claim-Driven Experiments

### Claim 1: Feature decoy generation beats boundary-only suppression at matched quality

- compare against best uniform BRS, anisotropic BRS, and UAP-style baseline
- metrics: post-codec `Delta JF`, attack success rate, SSIM, LPIPS
- decisive test: matched SSIM `>= 0.90`, LPIPS `<= 0.10`

### Claim 2: The gain comes from feature-aligned decoys, not just extra parameters

- ablate the renderer into:
  - random background patch editor
  - unconstrained low-frequency editor
  - target-conditioned editor without feature losses
- show that only target-conditioned feature alignment gives strong drift

### Claim 3: Codec-aware low-frequency rendering is necessary

- compare:
  - full YUV low-frequency renderer
  - RGB residual U-Net without low-pass constraint
  - pixel-space UAP or FPN feature attack
- evaluate pre-codec and post-codec to show the survival gap

### Claim 4: Tube selection matters

- compare learned selector vs random tube vs farthest-background heuristic vs oracle tube
- report oracle-gap closure and negative-tail reduction

### Claim 5: The mechanism is real

- log SAM2 attention or mask mass on GT vs decoy over time
- visualize first hijack frame and subsequent memory rewrite
- measure decoy-feature similarity and correlate it with tracking drift

## 12. Minimum Ablations

At least the following:

1. **DINOv2 only vs SAM2 only vs DINOv2 + SAM2**
2. **Learned selector vs heuristic selector vs random selector**
3. **Low-frequency YUV renderer vs unconstrained RGB renderer**
4. **Without `L_divert` / `L_attn`**
5. **Frame-wise independent decoy vs temporally coherent tube**

## 13. Expected AAAI Strength

If executed cleanly, this is materially stronger than sector allocation or a small regression head because it provides:

- a real trainable method,
- a mechanism aligned with SAM2 memory tracking,
- a codec-aware renderer rather than a generic perturbation,
- a clear oracle-to-learned story for tube selection.

Expected paper score:

- **Proposal-only score now**: about `7.9 to 8.2`
- **If main results land** with clear matched-quality wins and solid ablations: about `8.3 to 8.6`, which is realistic AAAI borderline-to-positive territory

It is still not a NeurIPS-level universal attack story unless the decoy mechanism transfers much more broadly, but it is substantially more method-heavy and defensible than the current BRS allocation story.

## 14. 15 GPU-Hour Implementation Plan

Target budget: about 15 aggregate GPU-hours, parallelized across 6 V100 if available.

### Step 0: Data and feature cache

- precompute DINOv2 and SAM2 encoder tokens for training clips
- cache candidate background tracklets
- budget: `1.5 GPUh`

### Step 1: Oracle tube search on a small training subset

- 16 to 32 candidates per clip
- cheap feature-screening plus limited SAM2 reranking
- output pseudo-labels for selector
- budget: `2.5 GPUh`

### Step 2: Train Decoy Tube Selector

- small network, ranking or classification objective
- budget: `1.0 GPUh`

### Step 3: Pretrain Low-Frequency Decoy Renderer

- optimize feature alignment plus quality and codec losses
- no full SAM2 unroll yet
- budget: `4.0 GPUh`

### Step 4: Joint training with frozen SAM2

- 4-frame unroll
- codec-EOT
- full attack loss
- budget: `4.5 GPUh`

### Step 5: Evaluation and ablations

- real H.264 main table
- three must-run ablations
- budget: `1.5 GPUh`

Total: about `15.0 GPUh`

## 15. Main Risks and Fallbacks

### Risk 1: Feature similarity does not translate to SAM2 drift

- symptom: high DINO similarity, weak tracking failure
- fallback: increase SAM2-space supervision, keep DINO only for selector / pretraining

### Risk 2: Decoys become visible before they become effective

- symptom: LPIPS or SSIM fails first
- fallback: shrink support, strengthen low-frequency prior, use luma-dominant rendering and more conservative tube selection

### Risk 3: Codec destroys the decoy

- symptom: strong pre-codec, weak post-codec
- fallback: tighter YUV parameterization, stronger spectral penalty, joint training across CRF 18/23/28

### Risk 4: Learned selector is unstable

- symptom: large negative tail from bad placements
- fallback: top-K heuristic shortlist plus learned reranker, or oracle-assisted evaluation first

### Risk 5: SAM2 memory is too sticky after the first frame

- symptom: the decoy never wins early competition
- fallback: apply stronger edits on frames 2 to 4 only, optimize the first hijack event, and explicitly supervise attention / logit mass shift before later memory rewrite

## 16. Final Positioning

The sharpest paper thesis is not "we found another perturbation family." It is:

> Under publisher-side masks and realistic H.264 compression, the right attack surface is not boundary erasure but codec-stable false-target induction. A target-conditioned low-frequency decoy can hijack SAM2 more effectively than boundary-only editing because it wins the feature-matching competition that drives memory-based propagation.
