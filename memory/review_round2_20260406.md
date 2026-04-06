---
name: Review Round 2
description: Updated research-review conclusions after preliminary AdvOpt results
type: project
date: 2026-04-06
---

## Key Conclusion

AdvOpt materially improves the paper relative to Round 1 and likely pushes the work just above the AAAI viability threshold, but only narrowly.

- Current internal score: `AAAI 6.3/10`, `ICCV/CVPR 5.2/10`
- Verdict: `Almost`

## Why the Score Improved

- Real paired evidence now exists: `AdvOpt +53.5pp` vs `idea1 +23.3pp` on the first 15 DAVIS videos, with `AdvOpt` better on 14/15.
- Within-video proxy validation is strong: mean Pearson `r=+0.811`, mean Spearman `rho=+0.878`.
- `MSB` is now best used as a negative ablation showing alpha is the dominant control variable.

## Remaining Must-Fix Items

1. Finish the full 88-video sweep and report CI plus paired significance.
2. Add oracle-gap analysis: proxy-chosen alpha vs oracle alpha under the same SSIM constraint.
3. Replace the current "matched SSIM 2.3x" headline with a stricter iso-quality or Pareto-style comparison.
4. Keep only `AdvOpt` as the main method; demote `MSB`; drop `LFNet` unless it clearly wins.
5. Handle the `SSIM < 0.90` failure cases explicitly.

## Submission Guidance

- AAAI: plausible if the current trend holds on the full sweep and the proxy-selection validation is tightened.
- ICCV/CVPR: still below bar due to limited novelty and generalization evidence.
