# Paper Improvement Log

**Paper:** H.264 Codec as an Adversarial Purifier: A Systematic Negative Result on Pixel-Constrained Attacks Against SAM2 Video Tracking
**Date:** 2026-03-26
**Venue:** ICLR

---

## Round 0 (Baseline)

**Score:** 4/10 (Codex GPT review)
**PDF:** `main_round0_original.pdf` (11 pages)

### Issues Found

| Severity | Issue |
|----------|-------|
| CRITICAL | Experiment count says "six" but paper has 7 (4 pixel rounds + 3 feature modes) |
| CRITICAL | "H.264 CRF23 is an effective adversarial purifier" — overclaim, scope too broad |
| CRITICAL | No adaptive codec-aware attack (BPDA/EOT through real H.264) |
| MAJOR | No discussion of adaptive evaluation (Athalye et al.) |
| MAJOR | Root cause (frequency concentration) stated as fact, no empirical evidence |
| MAJOR | "Insurmountable barrier" — too strong |
| MAJOR | Contribution 4 stated as absolute claim |
| MINOR | Sign convention for dJF metrics not explicit |
| MINOR | 3 overfull hboxes (39pt, 31pt, 38pt) |

---

## Round 1

**Score:** 5/10 (Codex GPT review)
**PDF:** `main_round1.pdf` (11 pages)

### Fixes Applied

1. **Experiment count**: "six" → "seven" in abstract, introduction, conclusion
2. **Softened overclaims**: Added "within the tested budget and attack families" to abstract, introduction, conclusion
3. **Insurmountable barrier**: Changed to "appears to be an insurmountable barrier for gradient-based optimization without true end-to-end differentiability through the codec"
4. **Added Athalye et al. (ICML 2018) + Guo et al. (ICLR 2018)** to references.bib
5. **New Related Work paragraph**: "Input-transformation defenses and adaptive evaluation" discussing BPDA/EOT and why adaptive evaluation is relevant even though we are the attacker
6. **New Limitations paragraph**: "Adaptive codec-aware optimization (most important limitation)" — explicitly acknowledges BPDA-style gaps as the key open question
7. **Contribution 4**: Softened to "for the tested direct-optimization attacks" and "evidence against" rather than absolute claim

### Issues Remaining

| Severity | Issue |
|----------|-------|
| CRITICAL | Still no adaptive codec-aware attack |
| MAJOR | Root cause still stated too definitively |
| MAJOR | Sign convention not explicit |
| MAJOR | JF_clean ≥ 0.5 selection bias not acknowledged |
| MAJOR | Single CRF23 point — no sweep |
| MINOR | "Overfull hboxes" — 1 remaining (31pt) |

---

## Round 2

**Score:** ~6/10 (estimated improvement)
**PDF:** `main_round2.pdf` (12 pages)

### Fixes Applied

1. **Sign convention**: Added explicit "(positive = attack degrades tracking; larger is better for attacker)" to both dJF and dJF_auc definitions in background.tex
2. **JF filter bias**: New limitations paragraph "Dataset selection bias" acknowledging the JF_clean ≥ 0.5 filter may favor videos where attacks matter (privacy) but notes this is appropriate for the privacy protection use case
3. **CRF limitation expanded**: Single CRF setting limitation now explicitly calls for a CRF sweep (CRF18/23/28) as future work
4. **Frequency mechanism demoted to hypothesis**: Section renamed "Hypothesized Root Cause: Frequency Concentration"; added explicit "This remains a hypothesis — we do not provide direct spectral measurements"; noted predictions (CRF18 would attenuate less, DCT-domain optimization might survive better) and that future work should validate via frequency-domain analysis
5. **Contribution 3 updated**: Now says "We hypothesize and discuss the root cause... and identify the key open question: whether BPDA/EOT-style codec-aware optimization can overcome this barrier"
6. **Six → seven** in analysis.tex (one remaining instance fixed)
7. **Overfull hboxes**: Resolved by shortening SAM2VideoPredictor.init_state() reference; emergencystretch=2em in main.tex

### Remaining Open Issues (Not Fixable Without New Experiments)

| Issue | Status |
|-------|--------|
| No adaptive BPDA/EOT attack | Acknowledged in limitations and related work; requires new experiment |
| Single CRF23 only | Acknowledged; requires new experiments |
| 9 DAVIS videos, 1 model | Acknowledged; requires new experiments |
| No spectral/frequency analysis | Demoted to hypothesis; requires measurement |

---

## Summary

| Round | Score | Key Changes |
|-------|-------|-------------|
| Round 0 | 4/10 | Baseline |
| Round 1 | 5/10 | Fixed count, softened claims, added adaptive attack context |
| Round 2 | ~6/10 | Explicit metrics, demoted mechanism to hypothesis, dataset bias, CRF limitation |

The paper is honest about its scope and limitations. The main remaining gap (adaptive codec-aware attacks) is explicitly acknowledged as the most important open question. The empirical result itself — that all 7 tested attack configurations fail under H.264 CRF23 — is reproducible and clearly documented.
