# AAAI 2026 Mock Review — SAM2 Privacy Preprocessor
**Date**: 2026-04-09
**Reviewer**: GPT-5.4 (xhigh reasoning, via Codex MCP)
**ThreadId**: 019d72af-3348-7f33-9a19-000d000b2735

---

## Score: 5/10 (Borderline, leaning reject)
## Confidence: 4/5

---

## Summary

This paper studies a practical and nontrivial problem: publisher-side privacy preprocessing that must remain effective after standard video compression, and it shows a surprisingly strong empirical signal for boundary ring suppression (BRS). However, in its current form I view the paper as **below the AAAI accept bar**: the main operator is quite simple, the claimed optimization contribution largely collapses to discovering a near-universal constant setting, and the evaluation does not yet rule out simpler matched-quality alternatives. The unpublished anisotropic-allocation result sounds materially more novel than the current AdvOpt-centered story.

## Strengths

1. The problem formulation is practically relevant and meaningfully different from standard codec-free adversarial attack papers.
2. The reported effect sizes are large on the stated DAVIS protocol, and the paper does make a credible case that codec-aware structured edits can survive where classic additive perturbations often do not.
3. The proxy-validation story is stronger than I expected for a two-parameter method: within-video correlation and the oracle-gap result are both useful evidence.
4. The paper is relatively honest about tradeoffs and limitations: surrogate brittleness, partial cross-tracker transfer, non-imperceptibility, and downstream utility loss are all acknowledged.
5. The abstract and core method description are clear and easy to follow.

## Weaknesses (ranked by severity)

1. **Novelty is borderline for AAAI.** BRS is a simple structured boundary edit, and AdvOpt appears to mostly discover that `(r, α) ≈ (24, 0.93)` is a good global setting. Once global-fixed matches AdvOpt on DAVIS and slightly edges it on the YT-VOS fair comparison, the algorithmic novelty of AdvOpt becomes thin.
2. **The baseline suite is not strong enough.** Comparing mainly against a weaker fixed version of the same operator (α=0.80) is not sufficient. I would expect matched-quality comparisons against blur, pixelation, direct inpainting, boundary erosion/dilation, and naive fixed α=0.93 or similar hand-tuned settings.
3. **The threat model is privileged, making comparison to UAP-SAM2 / DarkSAM tricky.** Those works are universal additive attacks without dense first-frame GT masks and without deliberate content removal. Your method is closer to annotation-assisted publisher-side editing. That is a valid setting, but only if framed explicitly as a different problem, not as a stronger attack in the same regime.
4. **Experimental completeness is still limited.** Main DAVIS uses n=36/90; global-fixed uses n=27; utility uses n=20; proxy validation is only on DAVIS; only SAM2.1 hiera_tiny is tested; XMem transfer is partial. This is promising evidence, not fully locked-down evidence.
5. **The constrained optimization story is brittle.** A 6% SSIM-floor violation rate and the YT-VOS surrogate failure cases matter because AdvOpt's main claim is precisely constrained operating-point selection.
6. **The manuscript story appears fragile.** Based on the provided material, the core sections are understandable, but the paper still sounds like it may contain mixed old/new narrative and sectioning. AAAI reviewers penalize this hard.
7. **Mechanistic understanding is still shallow.** "Codec amplifies boundary feature destruction" is plausible and empirically supported, but the paper does not yet deeply explain why this specific operator is right, when it fails, or what aspect of SAM2 it exploits beyond broad intuition.
8. **Generalization claims should be narrower.** This is not tracker-agnostic, and probably not yet SAM2-family-agnostic.

## Questions for Authors

1. Why is BRS superior to simpler publisher-side transforms such as Gaussian blur, pixelation, direct inpainting, or morphological boundary removal at matched post-codec SSIM/LPIPS?
2. If global-fixed (24, 0.93) matches AdvOpt on DAVIS and fixed 0.93 is slightly better on YT-VOS fair comparison, what is the scientific contribution of AdvOpt beyond discovering one constant?
3. How sensitive are results to imperfect first-frame masks, automatic masks, or mask noise? This seems central for real deployment.
4. Why are only 36/90 DAVIS videos used in the main paired result? Please provide a full exclusion breakdown and show that selection is not biasing the mean gain.
5. Why is the proxy validated only on DAVIS? Can you provide the same within-video correlation and oracle-gap analysis on YT-VOS or HEVC?
6. How should readers compare this work fairly to UAP-SAM2 and DarkSAM given the much stronger information assumption here?
7. Does the discovered operating point remain stable for larger SAM2 backbones and for point/box prompts rather than GT-mask initialization?
8. Why use the linear MSE-to-SSIM surrogate instead of differentiating SSIM directly or using a codec-aware learned quality predictor?

## Missing Experiments

1. Matched-quality baselines against blur, pixelation, direct inpainting, boundary erosion/dilation, and fixed α=0.93.
2. Robustness to imperfect masks or automatic annotation noise.
3. Full DAVIS sweep with confidence intervals, per-video distributions, and nonparametric paired tests.
4. Validation on additional SAM2 backbones.
5. Stronger task-matched comparison to codec-naive UAP baselines under the same GT-mask, post-codec protocol.
6. If the anisotropic iso-budget result is real, full evaluation of that result on DAVIS/YT-VOS with a clear learned-vs-oracle story.

## What Would Move Toward Accept

1. Reframe the paper much more explicitly as **publisher-side, mask-conditioned, codec-aware privacy preprocessing**, not as a direct "better attack" than UAP-SAM2/DarkSAM.
2. Add strong matched-utility baselines against simple non-adversarial transforms. This is the biggest missing piece.
3. Complete the evidence package: full DAVIS, more backbone coverage, imperfect-mask robustness, and direct proxy validation beyond DAVIS.
4. Demote AdvOpt's claim from "main method" to "offline calibration tool" unless you can show real advantages beyond discovering a constant.
5. **Yes, the anisotropic allocation result is probably the part that should become the core contribution** if it holds up under full evaluation. At fixed distortion budget, a large gain from directional budget allocation is a much stronger AAAI-level scientific story than uniform BRS plus a two-parameter optimizer.
