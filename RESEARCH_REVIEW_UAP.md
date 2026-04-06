# UAP Reproduction Review — Research Review Session
**Date**: 2026-03-30  
**Reviewer**: GPT-5.4 (xhigh reasoning)  
**Thread**: `019d3ec0-b94b-7d73-858c-90505d0615ae`  
**Topic**: Is our reproduction of "Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2" complete?

---

## TL;DR

**Short answer**: B1a is NOT a faithful reproduction of the paper. It is a simplified UAP baseline. Our main results (combo_strong +16.4pp) are clean and unaffected. Reframe accordingly before submission.

---

## Key Finding: Paper Mismatch

The "Vanish into Thin Air" paper (NeurIPS 2025):
- Uses **mIoU**, NOT J&F
- Does NOT claim H.264 robustness
- Uses random prompt scanning + dual semantic-deviation losses + video-aware memory misalignment attacks
- Evaluates on YouTube-VOS, DAVIS, MOSE under input corruption defenses

Our B1a baseline:
- Simple additive pixel-space UAP, single-frame centroid point prompt
- Evaluated under post-H.264 J&F protocol
- This is an **out-of-paper stress test**, not a reproduction endpoint

---

## Current Reproduction Status

### What's Clean
- **Main results** (combo_strong +16.4pp ΔJF_codec, SSIM=0.921): Clean — from signal-processing path, no g_theta
- **B1a raw UAP path**: Internally consistent. uap_delta_final.pt stores raw perturbation, not generator weights.

### What's Broken
1. **B1a server eval**: 5/9 videos show -1.0 sentinel (FFmpeg not in PATH) → invalid
2. **C1/C2 generator evals**: g_theta_size=256 in training, g_theta_size=0 in eval → broken (but these are not main results)
3. **Train/eval task mismatch**: UAP trains on single frames with centroid prompt, eval uses full SAM2 video tracker

### Near-Zero UAP Test Result
- 4/9 valid server eval videos: delta_jf_adv ≈ 0.002–0.035pp (near zero)
- This is likely genuine overfitting + task mismatch, NOT a code bug
- Honest interpretation: UAP doesn't generalize from 20 train videos to unseen test videos under codec eval

---

## Reviewer Assessment

**Most dangerous reviewer comment**:
> "The comparison is not convincing. The UAP-SAM2 paper studies a stronger video-aware cross-prompt attack under mIoU; authors evaluate a simplified single-perturbation baseline under post-H.264 J&F. Near-zero UAP effect may reflect under-tuned implementation. Further weakened by privileged mask access assumption."

**Ideal rebuttal**: B1a is explicitly labeled as a simplified codec-unaware UAP baseline; we do not claim to reproduce the NeurIPS 2025 method. Comparison is a systems tradeoff: annotation-free+training-heavy vs training-free+annotation-assisted.

---

## Minimum Checklist Before Submission

### Strictly Necessary (desk-rejection risk if missing)
- [ ] Rename B1a as "simplified codec-unaware UAP baseline inspired by UAP-SAM2" everywhere
- [ ] Fix FFmpeg path on server, rerun B1a eval with zero sentinel failures
- [ ] Report both pre-codec ΔJF_adv AND post-codec ΔJF_codec for B1a
- [ ] Use prompt-matched comparison: point-prompt for primary direct comparison
- [ ] One fixed validity filter across all methods (e.g., JF_clean ≥ 0.5)
- [ ] Add explicit limitations paragraph: B1a ≠ faithful NeurIPS 2025 UAP-SAM2 reimplementation

### Significantly Improves Acceptance
- [ ] Epsilon sweep for B1a: 4/255, 8/255, 16/255
- [ ] 2-3 seeds for B1a at main epsilon
- [ ] Mask robustness table for our method (dilate/erode/noise — ALREADY DONE: +12.0/+8.0/+10.0pp)
- [ ] Comparison table with columns: Requires masks? / Needs training? / Universal? / Codec-tested? / Metric / Prompt type
- [ ] 3-4 visual examples (clean → edited pre-codec → edited post-codec → SAM2 output)

### Nice-to-Have (optional)
- [ ] One appendix cross-prompt result for B1a
- [ ] Appendix note comparing J&F protocol vs paper's mIoU setup
- [ ] Second SAM2 backbone

---

## Revised Baseline Text for Paper

> We compare against B1a, a simplified universal adversarial perturbation baseline inspired by recent SAM2 UAP work, but we do not present it as a faithful reproduction of UAP-SAM2. B1a learns a single additive pixel-space perturbation on training videos using a point-prompted segmentation surrogate, evaluated under our deployment protocol using SAM2 video tracking after H.264 compression with J&F. This baseline answers a narrower practical question: whether a codec-unaware universal pixel perturbation remains effective under our release-time video pipeline. It is not meant to reproduce the original paper's video-aware attack design, prompt-scanning strategy, dual semantic-deviation losses, or mIoU-based evaluation.
>
> Our method targets a different operating regime. It is a publisher-side preprocessing method that uses per-frame masks already available in annotation pipelines to apply structured boundary-ring suppression before release, requiring no adversarial training. We frame the comparison as a systems tradeoff: B1a represents a training-heavy, annotation-free universal perturbation baseline; our method represents a training-free, annotation-assisted preprocessing baseline for codec-compatible privacy protection.

---

## What To Run With Remaining ~1-2 GPU Days

1. **Full held-out B1a eval (point-prompt, all DAVIS test videos, zero FFmpeg failures)** — primary
2. **Epsilon sweep B1a**: 3 values × ~1h each = ~3h
3. **One extra seed B1a**: ~5h
4. Already have mask-noise robustness results (dilate/erode/noise) — just add to table

---

## Paper Framing Confirmed

- Do NOT claim: "We reproduced Vanish into Thin Air and it fails under codec"
- DO claim: "Under our codec-constrained publisher-side evaluation, structured mask-guided preprocessing is more reliable than a simplified codec-unaware universal pixel perturbation baseline"
- Frame as systems tradeoff, not attack leaderboard
- Note: UAP baseline is mask-free (stronger information constraint) vs our method (oracle masks)
