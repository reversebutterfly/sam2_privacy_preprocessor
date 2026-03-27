# Experiment Plan — SAM2 Privacy Preprocessor

**Problem**: Third parties can use SAM2 to extract person trajectories from publicly released video datasets, violating privacy.
**Method Thesis**: A learnable residual preprocessor `g_θ` injects visually imperceptible perturbations that survive H.264/H.265 re-encoding and disrupt SAM2 tracking, via perceptual-budget training + codec-aware EOT + decoy memory competition.
**Date**: 2026-03-19
**Single-GPU target**: RTX 3090 / A100 × 1, ≤48 h total for must-run experiments.

---

## Claim Map

| ID | Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|----|-------|----------------|-----------------------------|---------------|
| C1 | Codec-aware EOT preserves attack effectiveness after H.264 re-encoding | Core differentiator from UAP-SAM2; reviewers will ask "does the attack survive real deployment?" | J&F drop (UAP-SAM2 + codec) ≈ 0%; J&F drop (ours + codec) ≥ 15% on DAVIS-mini | B2, B3 |
| C2 | Decoy branch induces memory-slot competition → additional failure at frame 60+ | Novelty-strongest component; no prior work directly targets SAM2 memory bank eviction | Stage 3 vs. Stage 4 (decoy on/off) J&F curve diverges at ≥frame 40; memory entropy increases in Stage 4 | B4, B5 |
| Anti-C | Gain is NOT from simply adding an LPIPS penalty to UAP-SAM2 | Prevent reviewer: "just a constrained version of UAP-SAM2" | Fair-budget baseline (UAP-SAM2 + LPIPS constraint, equal steps) still fails post-codec | B3 |

---

## Paper Storyline

**Main paper must prove:**
- C1: codec gap is real and quantitative (Table 1)
- Perceptual quality is maintained within budget (Table 1 quality columns)
- Anti-C: the fair baseline cannot bridge the gap (Table 1 ablation row)

**Appendix can support:**
- C2: decoy branch long-term curve (Figure A1) — strong if positive, demoted if weak
- Downstream utility (YOLOv8 / MMPose) (Table A1)
- Prompt robustness stress test (Table A2)

**Experiments intentionally cut (for now):**
- Multi-dataset scaling (YT-VOS full / MOSE full) — run only if DAVIS-mini results are strong
- Crowdsourced perceptual study — expensive; VMAF ≥ 90 serves as proxy
- Surrogate ensemble ablation — run only after Stage 3 is validated

---

## Experiment Blocks

### Block 0: Sanity Check (M0)
- **Claim tested**: Pipeline correctness — no claim
- **Why this block exists**: Verify data loading, SAM2 inference, J&F metric computation, and one-step gradient flow through `g_θ` before any training
- **Dataset / split / task**: DAVIS-mini — first 5 videos (≤ 30 frames each), person category only
- **Compared systems**: No comparison; overfit `g_θ` on 1 video (Stage 1 loss only), check J&F drops
- **Metrics**: J&F on that 1 video (should drop sharply), LPIPS (should be ≤ 0.10), no codec step yet
- **Setup details**: Freeze SAM2-T, train `g_θ` (lightweight UNet or residual conv) for 500 steps on 1 video, lr=1e-3
- **Success criterion**: J&F drops ≥ 30% on the overfit video; LPIPS ≤ 0.15; no NaN/inf in gradients
- **Failure interpretation**: If J&F does not drop, the attack loss is not backpropagating through SAM2; check gradient tape and SAM2 frozen/unfrozen layers
- **Table / figure target**: Not in paper; internal gate only
- **Priority**: **MUST-RUN** (gate for all downstream runs)
- **Estimated time**: 0.5 h

---

### Block 1: Baseline Reproduction (M1)
- **Claim tested**: Establishes the codec vulnerability of lp-norm baselines (supports C1 and Anti-C)
- **Why this block exists**: Must prove UAP-SAM2-style attack collapses after H.264 before claiming our method solves it
- **Dataset / split / task**: DAVIS-mini (5 videos, person), DAVIS-val subset (20 videos) for final numbers
- **Compared systems**:
  - UAP-SAM2 reimplemented: lp-norm ε=8/255 universal perturbation, no codec EOT
  - UAP-SAM2 + LPIPS (fair budget baseline): same as above but with hinge LPIPS ≤ 0.10
- **Metrics**: J&F pre-codec (↓), J&F post-H264-CRF23 (↓), LPIPS, SSIM
- **Setup details**: Train both baselines on DAVIS-train-mini (10 videos), 2000 steps, SAM2-T surrogate; evaluate with FFmpeg H.264 CRF 23
- **Success criterion**: UAP-SAM2 pre-codec J&F drop ≥ 10%; post-codec J&F drop < 3% (confirms codec kills it). Fair-budget baseline also fails post-codec.
- **Failure interpretation**: If UAP-SAM2 post-codec gap is < 3 pp, the story is weaker; increase CRF to 28 or test CRF 18 to find the breakpoint
- **Table / figure target**: Table 1 (baseline rows), Figure 1 (pre/post codec bar chart)
- **Priority**: **MUST-RUN**
- **Estimated time**: 4 h

---

### Block 2: Main Method — Stage 3 (Codec-Aware EOT, No Decoy) (M2)
- **Claim tested**: C1 — codec-aware EOT preserves attack after H.264
- **Why this block exists**: Core paper result; isolates the codec contribution without the decoy branch
- **Dataset / split / task**: DAVIS-mini train (10 videos), eval on DAVIS-val-20
- **Compared systems**: vs. UAP-SAM2 (B1), vs. UAP-SAM2+LPIPS (B1)
- **Metrics**: J&F pre-codec (↓), J&F post-H264 CRF {18, 23, 28} (↓), LPIPS ≤ 0.10 (↑), SSIM ≥ 0.95 (↑), VMAF ≥ 90 (↑)
- **Setup details**: Stage 1 (residual + perceptual) → Stage 2 (+ temporal consistency) → Stage 3 (+ codec EOT). Differentiable H.264 proxy via DCT quantization simulation; also add resize × {0.9, 1.0, 1.1} and Gaussian blur σ ∈ {0, 0.5, 1}. SAM2-T surrogate. Train 3000 steps (Stage 1: 1000, Stage 2: 1000, Stage 3: 1000), lr=5e-4, AdamW.
- **Success criterion**: Post-H264-CRF23 J&F drop ≥ 12% and LPIPS ≤ 0.10 simultaneously. The codec gap vs. UAP-SAM2 (B1) is ≥ 10 pp.
- **Failure interpretation**: If post-codec J&F drop < 5%, the differentiable proxy is not well-calibrated to real FFmpeg; try adding direct FFmpeg augmentation (non-differentiable) at eval time and report the gap separately
- **Table / figure target**: Table 1 (Stage 3 row), Figure 2 (J&F vs. CRF curve)
- **Priority**: **MUST-RUN**
- **Estimated time**: 6 h

---

### Block 3: Novelty Isolation — Codec EOT Ablation (M3-A)
- **Claim tested**: Anti-C — codec EOT is the mechanism, not just the perceptual budget
- **Why this block exists**: Reviewer will ask: "is Stage 3 just Stage 1 + harder training?" Need one row that removes only the codec EOT
- **Dataset / split / task**: DAVIS-val-20
- **Compared systems**: Stage 2 (perceptual + temporal, no codec EOT) vs. Stage 3
- **Metrics**: Post-H264 J&F drop (↓); pre-codec J&F drop (to check pre-codec is not hurt)
- **Setup details**: Reuse Stage 2 checkpoint from B2; just evaluate post-codec without any codec EOT training
- **Success criterion**: Stage 2 post-codec J&F drop < Stage 3 post-codec J&F drop by ≥ 8 pp
- **Failure interpretation**: If Stage 2 is already codec-robust, the EOT contribution is marginal; in that case, the perceptual constraint itself may be indirectly suppressing high-frequency fragile components (still a valid finding, but reframe accordingly)
- **Table / figure target**: Table 2 (ablation)
- **Priority**: **MUST-RUN**
- **Estimated time**: 1 h (reuse B2 checkpoints, no retraining)

---

### Block 4: Supporting Claim — Stage 4 Decoy Branch (M3-B)
- **Claim tested**: C2 — decoy-induced memory competition causes additional long-term failure
- **Why this block exists**: The decoy branch is the strongest novelty claim; must show it adds value beyond Stage 3
- **Dataset / split / task**: DAVIS-val-20, video clips of 60–80 frames (filter for longer clips from DAVIS)
- **Compared systems**: Stage 3 (no decoy) vs. Stage 4 (with decoy); on long clips only
- **Metrics**: J&F curve at frames {10, 20, 30, 40, 50, 60+}; memory cross-attention entropy (diagnostic, not primary)
- **Setup details**: Add decoy branch to Stage 3: a lightweight head predicting background adversarial patches at ≤ 5% image area; add L_decoy loss term (maximize SAM2 memory attention on non-target regions). Train 1000 additional steps on top of Stage 3. λ₃ = 0.1 initially.
- **Success criterion**: At frame ≥ 40, Stage 4 J&F < Stage 3 J&F by ≥ 5 pp. Memory entropy is measurably higher in Stage 4.
- **Failure interpretation**: If no gap appears, demote decoy to "minor contribution" or appendix. Do not remove; document as a negative finding. Main story still holds via C1 (B2).
- **Table / figure target**: Figure 3 (J&F temporal curves), Appendix Table A1 if weak
- **Priority**: **MUST-RUN** (but demotion path is clear if negative)
- **Estimated time**: 4 h

---

### Block 5: Memory Mechanism Diagnostic (M3-C)
- **Claim tested**: C2 mechanism — memory slot eviction, not just noise accumulation
- **Why this block exists**: Reviewers will ask for mechanistic evidence, not just J&F numbers
- **Dataset / split / task**: DAVIS-val-20 (subset: 5 videos with > 60 frames)
- **Compared systems**: Stage 3 vs. Stage 4, visualize SAM2 memory bank cross-attention maps
- **Metrics**: Qualitative attention heatmaps (frames 1, 20, 40, 60); object pointer token cosine similarity decay curve
- **Setup details**: Hook into SAM2 memory attention layers; log `[B, heads, T_mem, H*W]` attention tensors at inference time; no training changes
- **Success criterion**: Attention maps in Stage 4 show non-target regions receiving higher cross-attention weight; pointer token similarity to ground-truth object decays faster in Stage 4
- **Failure interpretation**: If no visual difference, the decoy is not mechanistically engaging the memory bank; revisit decoy placement strategy
- **Table / figure target**: Figure 3 inset or Appendix Figure A2
- **Priority**: **NICE-TO-HAVE** (run only if B4 shows positive J&F gap)
- **Estimated time**: 1 h

---

### Block 6: Downstream Utility Test (M4-A)
- **Claim tested**: C5 — preprocessed video retains utility for legitimate research tasks
- **Why this block exists**: Without this, paper cannot claim "privacy-preserving" — it must show privacy–utility tradeoff is favorable
- **Dataset / split / task**: DAVIS-val-20 (same clips used in B2)
- **Compared systems**: Original video vs. Stage 3 preprocessed video vs. Stage 4 preprocessed video
- **Metrics**: YOLOv8n person mAP (target: drop < 5 pp), MMPose ViTPose-S PCKh (target: drop < 5 pp)
- **Setup details**: Run off-the-shelf YOLOv8n and ViTPose-S on original vs. preprocessed frames; no retraining; compare per-video metrics
- **Success criterion**: mAP drop < 5 pp AND PCKh drop < 5 pp for both Stage 3 and Stage 4
- **Failure interpretation**: If utility drop > 10 pp, the perceptual budget (LPIPS ≤ 0.10) is insufficient for utility preservation; consider tightening to LPIPS ≤ 0.07 and rerunning B2
- **Table / figure target**: Table 1 (utility columns) or Appendix Table A1
- **Priority**: **MUST-RUN**
- **Estimated time**: 2 h

---

### Block 7: Prompt Robustness Stress Test (M4-B)
- **Claim tested**: Threat model robustness — attack holds against stronger SAM2 prompts
- **Why this block exists**: Reviewers will ask: "what if the adversary uses more clicks or a box?"
- **Dataset / split / task**: DAVIS-val-20
- **Compared systems**: Stage 3 under: (a) 1-point prompt (standard), (b) 5-point prompt, (c) bounding box prompt, (d) 3 conditioning frames
- **Metrics**: J&F drop post-H264 for each prompt type
- **Setup details**: Vary SAM2 prompting protocol; no retraining; Stage 3 checkpoint from B2
- **Success criterion**: J&F drop remains ≥ 8 pp across all prompt types
- **Failure interpretation**: If box prompt fully recovers tracking, the attack is prompt-sensitive; add this as a known limitation and propose prompt-agnostic training as future work
- **Table / figure target**: Appendix Table A2
- **Priority**: **NICE-TO-HAVE**
- **Estimated time**: 2 h

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Est. Time | Risk |
|-----------|------|------|---------------|-----------|------|
| **M0 Sanity** | Validate pipeline end-to-end | B0 only | J&F drops on overfit video; no gradient errors → proceed | 0.5 h | High (first run) |
| **M1 Baseline** | Confirm codec kills UAP-SAM2-style attack | B1 | UAP-SAM2 pre-codec drop ≥ 10%, post-codec drop < 3% → codec gap exists → proceed | 4 h | Medium |
| **M2 Main** | Prove Stage 3 survives codec | B2 | Post-H264 J&F drop ≥ 12% AND LPIPS ≤ 0.10 → core claim proven → proceed | 6 h | High (core result) |
| **M3 Decision** | Isolate contributions; test decoy | B3, B4, B5 | B3 ablation confirms EOT is the mechanism; B4 either adds to story or is demoted | 6 h | Medium |
| **M4 Polish** | Utility, robustness, figures | B6, B7 | Utility drop < 5 pp → paper story complete | 4 h | Low |

**Total must-run estimate**: ~21 h (M0 + M1 + M2 + M3-core + M4-A)
**Total with nice-to-have**: ~24 h

---

## Compute and Data Budget

- **Total estimated GPU-hours (must-run)**: ~21 h on 1× GPU (RTX 3090 / A100)
- **DAVIS-mini**: 5 training videos, 20 eval videos — download from DAVIS 2017 (person category); ~2 GB
- **DAVIS-val full (optional)**: 30 videos for extended eval — same download
- **YT-VOS / MOSE**: Skip for initial runs; add only if DAVIS results are strong
- **Human evaluation**: Replaced by VMAF ≥ 90 proxy; skip unless targeting top-4 security venues
- **Biggest bottleneck**: Differentiable H.264 proxy fidelity (codec EOT training); validate early in M2

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Differentiable H.264 proxy does not correlate with real FFmpeg | Always evaluate with real FFmpeg; report proxy-vs-real gap as a finding; consider non-differentiable data augmentation with pre-encoded clips |
| Stage 3 post-codec J&F drop < 12% (core claim weak) | Increase codec EOT batch size; try CRF sweep to find best operating point; if still weak, reframe as "partial robustness" and fall back to USENIX workshop |
| Decoy branch has no measurable J&F benefit (B4 negative) | Demote to appendix; main story is C1; the decoy becomes "a preliminary exploration" |
| UAP-SAM2 baseline is not reproducible (no official code) | Implement minimal UAP-SAM2 equivalent: universal additive δ (lp-norm ε=8/255), maximize -J&F directly; document reimplementation details |
| LPIPS + J&F tradeoff is not Pareto-dominant over baselines | Show Pareto frontier (LPIPS vs. J&F drop) across λ values; identify operating points where our method dominates |
| SAM2 version change makes results incomparable | Lock to SAM2-T v1.0 for all runs; test SAM2.1 in appendix |

---

## Final Checklist

- [ ] Main paper table (Table 1) is covered by B1 + B2 + B3 + B6
- [ ] Novelty (codec EOT) is isolated by B3 ablation
- [ ] Simplicity is defended (no unnecessary components added beyond Stage 3 for core claim)
- [ ] Frontier contribution: codec-aware EOT is compared against plain lp-norm baselines (Anti-C)
- [ ] Decoy branch is positioned as supporting/novel, with clear demotion path if negative
- [ ] Nice-to-have runs (B5, B7) are separated from must-run runs
- [ ] All runs use DAVIS-mini first before expanding to full datasets
