# Research Review — 2026-04-02
**Reviewer**: GPT-5.4 xhigh (via Codex MCP)
**Thread ID**: `019d4e0f-2d14-7fd1-8147-1b893720dbeb`
**Project**: Codec-Amplified Semantic Boundary Suppression

---

## Round 1: Overall Assessment

### Current Score: 3/6 (Weak Reject) for ICCV/ECCV

Realistic reviewer spread: 2 / 3 / 4 (average on reject side).

**Why**: The core idea (low-frequency boundary edits survive H.264) is genuinely interesting, but:
1. Threat model too weak for "privacy" claim — no adaptive attacker evaluation
2. Baseline/novelty risk — method is close to mask-guided boundary blur; UAP baseline too weak to help
3. Scope/utility mismatch — +4.0pp YT-VOS is a major generalization warning; SSIM+YOLO too thin for utility claim

---

## Round 1: Top 3 Gaps

### Gap 1: Threat Model (CRITICAL)
- Current: only evaluate against off-the-shelf SAM2 with no adaptation
- Reviewer concern: once published, attacker can simulate preprocessing offline
- **Fix**: Add "adaptive-lite" attacker — sharpen/deblur/edge-enhance before SAM2. Cheap, high credibility gain.
- Note: Full defender-aware retraining NOT required if claim is narrowed to closed-release black-box setting

### Gap 2: Baseline Weakness
- UAP ΔJF_codec ≈ 0.85pp is too weak — comparison looks trivial
- "Boundary blur" (our fair baseline) at +10.8pp is the real competitor; need matched-utility comparison
- **Fix**: Report matched-utility comparison (same SSIM range) vs boundary blur + global blur; add one additional VOS tracker (XMem/Cutie) as non-adaptive baseline

### Gap 3: Scope/Utility Mismatch
- +4.0pp on YT-VOS vs +16.4pp on DAVIS is suspicious without clear explanation
- "Privacy for people" claim not backed by person-centric evaluation
- **Fix**: Filter DAVIS + YT-VOS to person-only categories, re-report

---

## Round 2: Revised Assessment

### Revised Score (conditional)
- If person-only results STRONG (+12pp) AND HEVC confirmed AND adaptive-lite still degrades: **4/6** (borderline accept at TCSVT)
- If person-only results WEAK (+4pp): still 3/6

### Top 3 TODOs (by score-per-GPU-hour)
1. **H.265/HEVC eval** — same pipeline, just codec flag change. Very cheap, very high reviewer value. Directly tests whether claim is H.264-specific or codec-general.
2. **Person-only subset** — filter DAVIS + YT-VOS to person categories, re-report ΔJF_codec. High risk but highest value if strong.
3. **Adaptive-lite attacker** — unsharp mask / deblur / edge enhancement before SAM2. Buys more credibility per GPU-hour than a second non-adaptive tracker.

---

## Round 2: YT-VOS Gap Assessment

**+4.0pp is survivable ONLY IF** you:
- Explicitly narrow claim to "geometry-conditioned defense"
- Say works best for "boundary-dominant" targets
- Show person-centric benchmark where performance is materially better than +4pp
- Frame YT-VOS as scope limitation, not failure

**If broad framing is kept**: reviewers will read DAVIS as cherry-picked and YT-VOS as the truth.

---

## Round 2: Venue Recommendation

**TCSVT (IEEE Transactions on Circuits and Systems for Video Technology)** — NOT giving up, genuinely best fit.

Rationale:
- Work is about video preprocessing + compression interaction + downstream video analysis
- TCSVT explicitly covers video processing, filtering, analysis, and compression
- Narrowed threat model is acceptable at TCSVT; ICCV/ECCV would require stronger attacker analysis

**Timeline**:
- April 2026: Run HEVC + person-only + adaptive-lite experiments
- May 2026: Rewrite with narrowed threat model → submit to TCSVT
- June 2026: If relevant ECCV 2026 workshop appears, use for visibility/feedback
- 2027: Re-aim at ICCV only if attacker story hardened and person-centric story holds

**Note**: ECCV 2026 main closed March 5; ACM MM 2026 closed April 1; ECCV workshops TBD after April 12.

---

## Round 3: Paper Outline (TCSVT)

| Section | Key Claim | Main Evidence | Figure |
|---------|-----------|---------------|--------|
| 1. Introduction | Publisher-side codec-aware privacy is distinct from adversarial attack | UAP failure after H.264, DAVIS example | Teaser: original/defended/H.264/SAM2 outputs |
| 2. Problem Setting & Threat Model | Closed-release, black-box, off-the-shelf tracker setting | Formal notation, pipeline description | Pipeline diagram |
| 3. Related Work | This work differs from pixel-space attacks, anonymization, compression-robust perturbations | Taxonomy table | Comparison table |
| 4. Method | Low-frequency boundary suppression removes cues while surviving codec | Method equations, design rationale | Method figure: mask/ring/proxy/weight/output |
| 5. Why Codec Helps | Low-frequency edits persist through DCT; UAP high-freq destroyed | Sobel gradient drop, DCT energy, UAP collapse | Mechanism 3-panel figure |
| 6. Experimental Protocol | Evaluation is controlled and deployment-relevant | Datasets, metrics, codec settings, tracker list | Protocol table |
| 7. Main Results (H.264) | Method degrades off-the-shelf VOS; beats baselines at matched utility | Main DAVIS table + XMem/Cutie | **Pareto plot: privacy vs utility** |
| 8. Generalization & Scope | Effect is robust to nuisances but content-conditioned | Robustness table, cross-dataset, YT-VOS diagnosis, person-only | Scatter: gain vs ring-burden; subgroup bars |
| 9. Codec & Counter-Processing | Not H.264 overfitting; not trivially removed | HEVC table, adaptive-lite attacker table | Bar: H.264 vs HEVC vs adaptive-lite |
| 10. Discussion & Limitations | Practical black-box defense, not white-box guarantee; works best for boundary-dominant targets | Gap diagnosis, ablation | Limitations box |
| 11. Conclusion | Publisher can use mask-guided edits to raise cost of automated tracking after compression | — | — |

**TCSVT emphasis**: Sections 5, 8, 9 more important than ICCV would require. Center of gravity = "video preprocessing + codec + downstream impact."

---

## Round 3: Claims Matrix

| HEVC result | Person-only | Adaptive-lite | Allowed Claim |
|-------------|-------------|---------------|---------------|
| Strong (+12pp) | Strong (+12pp) | Still degrades | "Codec-robust publisher-side preprocessor degrades off-the-shelf person/object tracking across common codecs" — **STRONGEST** |
| Strong (+12pp) | Weak (+4pp) | Still degrades | "Codec-robust, content-conditioned method; strongest for boundary-dominant objects, not general people" |
| Weak (+3pp) | Strong (+12pp) | Still degrades | "H.264-targeted preprocessor degrades person tracking; effect is codec-specific" |
| Strong (+12pp) | Strong (+12pp) | Broken | "Degrades default pipelines; not robust against simple counter-processing" — avoid "privacy defense" without qualification |
| Any | Any | Broken | "Empirical phenomenon about codec interaction" — not a strong privacy paper |

**Key**: Exp3 (adaptive-lite) sets the ceiling. If it breaks the method, paper is a codec-interaction study, not a privacy defense.

---

## Round 3: Threat Model Text (paper-ready)

> We consider a closed-release publisher-side setting in which the data owner has access to the original video and ground-truth object masks at curation time, and releases only a compressed video stream to downstream users. The downstream user is modeled as a black-box consumer of the released asset: they receive no clean reference video, no masks, no side information about the edited regions, and no interaction channel with the publisher, and they apply off-the-shelf segmentation or tracking pipelines directly to the released compressed video. The goal is not to make objects invisible to human viewers, but to reduce the effectiveness of automated mask propagation and tracking while maintaining usable visual quality in the released media.
>
> Our scope is deliberately practical rather than white-box adversarial. We therefore do not claim robustness against an informed attacker who knows the exact preprocessing function, synthesizes defended training data offline, or jointly designs restoration and tracking for this specific defense. Such stronger guarantees define a different problem setting with different assumptions and evaluation methodology. The contribution here is that, within a realistic media-distribution regime where third parties operate on released compressed videos using standard tools, simple low-frequency boundary edits can survive modern video coding and materially degrade downstream automated tracking performance.

---

## Action Items (Priority Order)

| # | Experiment | GPU cost | Score impact | Status |
|---|-----------|----------|--------------|--------|
| 1 | **H.265/HEVC eval** — same DAVIS+YT-VOS pipeline, just add `--codec hevc` flag to ffmpeg | ~2h (CPU mostly) | HIGH — directly tests core claim generality | TODO |
| 2 | **Person-only subset** — filter DAVIS (person categories) + YT-VOS (person categories), re-report ΔJF_codec | ~1h (GPU) | HIGH if strong, risky if weak | TODO |
| 3 | **Adaptive-lite attacker** — unsharp mask + deblur + edge enhance as pre-SAM2 pipeline | ~1h (CPU+GPU) | HIGH — determines "privacy" vs "phenomenon" framing | TODO |
| 4 | **XMem/Cutie as additional tracker** | ~3h (GPU) | MEDIUM — strengthens generality | OPTIONAL |

---

## Summary Verdict

**Current state**: Experiments complete, results solid for DAVIS. YT-VOS gap diagnosed and explained. UAP baseline complete.

**Gap to TCSVT acceptance**: 3 lightweight experiments (HEVC, person-only, adaptive-lite). Each <2h. Together they could move score from 3/6 → 4/6.

**Paper writing**: Start now (use outline above). TCSVT submission May 2026.
