---
name: Review Round 3
description: Updated research-review conclusions after paired stats and oracle-gap validation
type: project
date: 2026-04-06
---

## Key Conclusion

AdvOpt now has a real scientific core: large paired gains over `idea1`, strong within-video proxy validation, and an unusually strong oracle-gap result. This is materially better than Round 2.

- Current internal score: `AAAI 6.7/10`, `ICCV/CVPR 5.6/10`
- Verdict: `Almost`

## Why the Score Improved

- Main paired evidence is now strong at `n=28`: `AdvOpt +57.3pp` vs `idea1 +24.4pp`, paired gain `+32.9pp`, win-rate `25/28`, Wilcoxon and paired t-test both effectively `p≈0`.
- Oracle-gap validation is excellent: proxy-chosen alpha matches oracle on `98%` of tested `(video, ring_width)` cases under `SSIM≥0.90`, with mean gap `+0.0pp`.
- The stronger SSIM penalty appears to have fixed the earlier quality-control weakness.
- `MSB` now serves a useful role as an informative negative ablation rather than a competing main method.

## Remaining Critical Weaknesses

1. The main paper is still anchored on an incomplete sweep.
   - The current `n=28/88` result is strong, but the headline contribution is experimental. Reviewers will still ask why the paper is being written before the full DAVIS run is finished.
   - Minimum fix: finish the full `88`-video sweep and make the final table use only completed-run numbers, with mean, median, 95% CI, paired gain, win-rate, and significance.

2. Generalization evidence is still too small to carry any broad claim.
   - `HEVC n=3` and mask-prompt `n=2` are fine as pilots but not as evidence for a serious generalization section.
   - Minimum fix: complete at least one extra axis to about `20` videos and demote the other to a pilot or appendix if it is still tiny at submission time.

3. The paper can still be attacked as "first-frame tuning of a hand-designed blur" rather than a genuinely new method.
   - The oracle-gap result helps a lot, but the narrative must stay narrow and precise.
   - Minimum fix: frame the contribution explicitly as `proxy-validated first-frame parameter adaptation` for a fixed boundary-suppression operator; keep `AdvOpt` as the only primary method and use `MSB` only as a negative ablation.

4. Utility evidence is still weaker than the privacy claim strength.
   - `SSIM` is useful, but for a privacy preprocessor paper many reviewers will want at least one indication that the edit does not simply destroy all downstream value.
   - Minimum fix: add one compact non-tracking utility check on the main setting, or narrow the claim to `visual quality preservation` instead of broader `utility preservation`.

## Submission Guidance

- AAAI: plausible if the full `88`-video run lands near the current trend and one compact generalization axis is completed.
- ICCV/CVPR: still below bar because the method novelty is too limited and the scope is too narrow.

## Practical Recommendation

If the ongoing experiments finish with roughly similar effect sizes, this becomes a reasonable AAAI submission. In its current state, it is not yet clean enough to submit because the two most reviewable objections are still "main sweep incomplete" and "generalization too small."
