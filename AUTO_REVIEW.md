# Auto Review Log — SAM2 Privacy Preprocessor
**Goal**: 将工作推进到 CCF-A（至少 AAAI）水平
**Started**: 2026-04-02
**MAX_ROUNDS**: 4
**Reviewer**: GPT-5.4 xhigh (via Codex MCP)

---

## Round 1 (2026-04-02)

### Assessment (Summary)
- **AAAI Score**: 4/10 (Realistic spread: 3/4/5, outside 6 possible)
- **ICCV/ECCV Score**: 3/10 (Spread: 2/3/4)
- **Verdict**: Almost — not CCF-A main track ready. AAAI reachable without fundamental redesign; ICCV/ECCV probably needs stronger method.
- **Concrete blocking item**: Adaptive-lite attacker result — determines "privacy defense" vs "codec artifact" framing.

### Key Criticisms
1. **Privacy claim ahead of evidence**: No adaptive attacker, no person-centric eval, single tracker
2. **Method novelty fragile**: Fixed ring+alpha reads as "boundary blur with tuning" — not strong enough for CCF-A
3. **External validity weak**: +16.4pp DAVIS vs +4.0pp YT-VOS, H.264-only scope, thin utility evidence

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Verdict**

Current work is interesting and publishable, but not CCF-A main-track ready as-is. Bluntly: AAAI is still reachable with targeted upgrades; ICCV/ECCV probably are not without a stronger method story or much broader validation.

1. **Current score**
AAAI: `4/10` current. Realistic spread: `3 / 4 / 5`, with an outside `6` only if a reviewer really buys the codec-amplification insight.

ICCV: `3/10` current. Realistic spread: `2 / 3 / 4`.

ECCV: `3/10` current. Realistic spread: `2 / 3 / 4`.

If the next experiments are positive, I could see AAAI moving to `6/10` borderline. ICCV/ECCV would still likely sit around `4.5-5/10` unless the method itself becomes more principled.

2. **Top 3 critical weaknesses**
`1)` The privacy claim is ahead of the evidence. No adaptive-lite attacker, no person-centric evaluation, and only one default tracker means reviewers can say this is a brittle black-box artifact, not a privacy defense.

`2)` The method novelty is fragile. A fixed ring plus alpha mask-guided blend will be read as "boundary blur with tuning." The phenomenon is interesting; the method, by itself, is not yet strong enough for CCF-A.

`3)` External validity is weak. `+16.4pp` on DAVIS but `+4.0pp` on YT-VOS is a major warning sign. Add single-tracker and H.264-only scope, and the broad claim becomes hard to defend.

3. **Minimum fix for each**
For `1)`, run a cheap adaptive-lite attacker: unsharp mask, deblur, edge enhancement before SAM2.
For `2)`, matched-utility comparison against boundary blur at same SSIM/VMAF range.
For `3)`, add at least two of: person-only subset, HEVC/H.265, second tracker.

4. **Highest-ROI experiments**
`1)` Adaptive-lite attacker — determines defense vs artifact framing
`2)` HEVC/H.265 rerun — cheap, strengthens codec story
`3)` Matched-utility Pareto sweep vs baselines
`4)` XMem/Cutie second tracker
`5)` Person-only subset — high upside for privacy story

5. **Method upgrade path**
Content-adaptive parameter selection is enough only for AAAI if it materially reduces YT-VOS gap. For ICCV/ECCV: need stronger adaptive formulation, or lightweight learned controller, or broader benchmark/analysis paper.

6. **AAAI vs ICCV/ECCV**: AAAI is more reachable. AAAI needs: honest threat-model narrowing + adaptive-lite + stronger matched baselines + one generalization axis beyond DAVIS/H.264.

7. **Ready?** Almost. Blocking item: decisive robustness section against simple counter-processing. Manuscript also not aligned with current contribution (old paper in paper/).

</details>

### Actions Taken
- Experiments to implement: adaptive-lite attacker, HEVC eval, person-only subset

### Status
- Continuing to Round 2 after experiments complete

---

## Round 2 (2026-04-02, preliminary results)

### Assessment (Summary)
- **AAAI Score**: 5.5/10 (conditional on full results; could reach 6/10)
- **Realistic spread**: 4/5/6
- **Verdict**: Close to AAAI-borderline — preliminary results are strong, but work needs to look "scientifically complete"
- **Concrete blocking item**: still single tracker; YT-VOS gap needs reframing; method consistency (idea1 vs combo)

### Key Improvements Since Round 1
- HEVC: +17.46pp (1.79x stronger than H.264!) → removes "H.264-only artifact" criticism
- Person-only: +10.48pp (n=8, 0 negatives) → makes privacy framing credible
- Adaptive-lite: +6.57pp residual after combined counter → removes "trivially undone" criticism

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Updated Verdict**

These new results move the paper materially. The strongest criticism from Round 1 was "weak threat model + likely brittle artifact." HEVC generalization plus partial survival under adaptive-lite counter-processing directly improve that.

My updated **AAAI score is `5.5/10`**, conditional on the full runs holding. If the full `n` confirms the same direction with similar magnitude, this becomes **`6/10` borderline**. Realistic reviewer spread is now **`4 / 5 / 6`**, with one reviewer still likely unconvinced by heuristic novelty and YT-VOS fragility.

1. **Updated AAAI score**: Score moves up by about 1.5 points from Round 1.
   Why: HEVC removes "H.264-only artifact", adaptive-lite removes "trivially undone by sharpening", person-only makes privacy framing credible.
   Caps: single tracker, heuristic method, large YT-VOS gap, matched-utility fairness against boundary blur (5.6pp at lower SSIM).

2. **Person-only +10.5pp sufficient?** Directionally yes, publication-grade no. Full 50-video + mean/CI/neg-rate needed. Method variant must be identical to main paper.

3. **Adaptive-lite residual +6.6pp enough?** Yes for "not trivially removable." Supports: "not trivially removed by standard counter-processing", "raises cost of off-the-shelf tracking." Does NOT support: robustness to adaptive attackers, strong privacy guarantees.

4. **Minimum work for 6/10**: Full HEVC+person+adaptive with CI; H.264 vs HEVC quality-matched (PSNR/VMAF not raw CRF); one more tracker (XMem/Cutie — highest value); true Pareto vs boundary blur; narrow claim to publisher-side black-box.

5. **YT-VOS gap**: Scale-norm +6pp would add only 0.2-0.3 to score. Gap is fundamental — must reframe as content-conditioned effectiveness, not rescue with tuning.

Bottom line: Close to AAAI-borderline if preliminary numbers survive. Remaining blocker is "scientifically complete" appearance: still one tracker, one favorable dataset, heuristic method.

</details>

### Actions Taken (Phase C Round 2)
1. Experiments running (person_only, HEVC, adaptive_lite, scale_norm_ytvos)
2. New priority: install second tracker (XMem/Cutie) on server
3. Fix method consistency: person_only should use combo not idea1
4. Implement PSNR/bitrate quality-matched HEVC comparison

### Status
- Continuing to Round 3 (after full experiment results + second tracker)

---

## Round 3 (2026-04-03, full results n=89-90)

### Assessment (Summary)
- **AAAI Score**: 5.8/10 (→ 6.1-6.3 after full XMem)
- **ICCV/ECCV Score**: 4.4/10 (not moving without stronger method)
- **Verdict**: AAAI borderline-reachable; method-consistency fix required
- **Concrete blocking item**: idea1 vs combo inconsistency; YT-VOS gap (content-conditioned framing required)

### Full Experiment Results
| Experiment | n | Result |
|-----------|---|--------|
| HEVC eval (idea1+H264 vs HEVC) | 89/90 | H264=+10.69pp, HEVC=+16.46pp (1.54×) |
| Person-only (idea1+H264) | 50/51 | +12.13pp, 0/50 negatives |
| Adaptive-lite (idea1+H264) | 89/90 | base=+10.69pp, max recovery=13.4% (CLAHE) |
| XMem (preliminary n≈10) | ~10/90 | ~+9.32pp, 0 negatives |

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**

**AAAI:** `5.8/10` now. If full XMem lands around the current `+9pp` with low negative rate, this becomes `6.1-6.3/10`, i.e. real borderline-accept territory. Realistic spread: `5 / 6 / 6`, or `5 / 6 / 7` if XMem finishes cleanly.

**ICCV/ECCV:** `4.4/10` now. Even with a positive full XMem, I only move this to about `4.8-5.0/10`. Realistic spread: `3 / 4 / 6`.

**Verdict**

You have mostly fixed the "brittle artifact / weak threat model" problem. HEVC, person-only, and adaptive-lite are all genuinely strong. The paper is now much closer to **AAAI-borderline**.

You have **not** fixed the top-CV problem: the contribution still reads more like a strong empirical codec-interaction study plus a heuristic preprocessor than a top-tier vision method paper.

**Remaining Weaknesses**

`1)` **Method inconsistency is now the biggest blocker.**
Your headline result is on `combo`, but the strongest new evidence is on `idea1`. A harsh reviewer will say: "the robustness/generalization evidence is for the ablation, not the claimed method."

`2)` **Novelty ceiling is still real.**
`idea1` looks close to a well-tuned boundary-aware blur/blend family. The interesting part is now the **codec-amplification phenomenon**, not the sophistication of the method itself.

`3)` **Cross-dataset generality is still weak.**
The YT-VOS gap remains the main external-validity warning sign.

`4)` **Second-tracker evidence is still incomplete.**
Preliminary XMem is good news, but until the full run is in, a reviewer can still say "single tracker in substance."

**Minimum Fixes To Reach 6/10**

For **AAAI**, the minimum package is now small and clear:

`1)` Finish full XMem and report mean, CI, and negative rate.

`2)` **Resolve the `idea1` vs `combo` mismatch.**
Best fix: rerun at least the key supplementary experiments on `combo`.

`3)` **Tighten the claim language.**
Claim: "publisher-side, black-box, off-the-shelf tracking degradation across H.264/H.265, not trivially removed by standard counter-processing."

`4)` **Be explicit that the method is content-conditioned.**
Do not try to "explain away" YT-VOS. Frame it as a real scope limit.

For **ICCV/ECCV**, these fixes are still not enough for `6/10`. To get there, you likely need a more principled method contribution.

**Bottom line**: you have largely solved the robustness credibility problem, but not the "is this enough of a method paper?" problem.

</details>

### Actions Taken (Phase C Round 3)
1. Launched HEVC rerun with `--edit_type combo` (GPU 2, screen hevc_combo)
2. Launched adaptive-lite rerun with `--edit_type combo` (GPU 3, screen adaptive_combo)
3. XMem full DAVIS evaluation running (GPU 0, screen xmem_eval)
4. Added `--edit_type` argument to pilot_hevc_eval.py and pilot_adaptive_lite.py

### Status
- Continuing to Round 4 (after combo reruns + XMem complete, ~2h)

---

## Round 4 (2026-04-03, partial combo reruns + XMem n=22 combo / n=88 idea1)

### Assessment (Summary)
- **AAAI Score**: 5.9/10 (idea1-first paper: ~6.1/10; combo-first paper: ~5.6-5.9/10)
- **ICCV/ECCV Score**: 4.6/10
- **Verdict**: Almost ready — main blocker is now **method identity**, not missing experiments
- **Key pivot**: reviewer strongly recommends making `idea1` the primary method, `combo` as SAM2-enhanced variant

### Updated Experiment Results (Round 4)
| Experiment | Method | n | Result |
|-----------|--------|---|--------|
| Main SAM2 (DAVIS) | combo_strong | 88 | +16.36pp, 0/88 neg |
| HEVC generalization | combo (partial) | 26 | H264=+11.15pp, HEVC=+14.98pp (1.34×) |
| Adaptive-lite robustness | combo (partial) | 20 | base=+10.74pp, combined recovery=38.5%, residual=+7.4pp |
| Person-only subset | combo (partial) | 19 | +12.81pp, 0 neg |
| XMem second tracker | idea1 (complete) | 88 | +11.67pp, 2 neg (2.3%) |
| XMem second tracker | combo (preliminary) | 22 | +4.39pp, 1 neg |

**Key new finding**: idea1 generalizes better to XMem than combo (+11.67pp vs +4.39pp). SAM2 is boundary-sensitive (combo's halo disrupts it); XMem uses different cues (idea1's suppression is more effective). Both variants show positive XMem degradation.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**

**AAAI:** `5.9/10` as the package stands now. Realistic spread: `4 / 6 / 6`.
If full `combo`-XMem finishes clearly positive, this can move to `6.1-6.3/10`.
If `combo`-XMem stays near `+4pp`, I would keep it below a clean `6`.

**ICCV/ECCV:** `4.6/10`. Realistic spread: `4 / 5 / 6`.
Still not main-track competitive there.

A more useful way to say it is this:

- **`idea1`-first paper:** about `6.1/10` for AAAI.
- **`combo`-first paper:** about `5.6-5.9/10` for AAAI.

That difference now matters.

**Verdict**

This is now **almost ready for AAAI**, but the final blocker is no longer "missing experiments." It is **method identity**.

You now have enough evidence for a solid paper if you present it as:

- core method = **boundary suppression (`idea1`)**
- `combo` = **SAM2-enhanced variant** that boosts peak suppression on SAM2 but is less robust to counter-processing and transfers worse to XMem

That is scientifically coherent. If you instead insist that `combo` is the single headline method that broadly generalizes, the new XMem and adaptive-lite results weaken the story.

**Remaining Weaknesses**

`1)` **`combo` is not the clean general method.**
Its partial adaptive-lite recovery is much worse than `idea1`, and its preliminary XMem transfer is much weaker than `idea1`. That is not fatal, but it changes the story.

`2)` **Cross-dataset scope is still narrow.**
The YT-VOS gap is still a real external-validity limitation. Your new framing is correct; do not try to hide it.

`3)` **Novelty ceiling remains.**
The strongest contribution is now the empirical/analytic finding about codec-amplified boundary disruption, not algorithmic sophistication.

**Q2. Do the combo consistency fix + XMem evidence satisfy the Round 3 blockers?**

**Mostly yes, but only partially for `combo`.**

- The **single-tracker blocker** is basically addressed at the paper level because XMem is now in the picture.
- The **method-consistency blocker** is only partly addressed, because the strongest cross-tracker result is still on `idea1`, not `combo`.
- So:
  - if the paper claims **boundary suppression** is the method, blocker closed;
  - if the paper claims **`combo` specifically** is the general method, blocker not fully closed yet.

**Q3. Is `combo` adaptive-lite recovery of `38.5%` a material weakening?**

**Yes, but not a fatal one.**

It weakens the robustness story from "strong" to "moderate."
It does **not** break the paper, because the residual effect is still `+7.4pp`, which is enough for:

- "not trivially removable"
- "still degrades off-the-shelf tracking after simple counter-processing"

It is **not** enough for any stronger robustness language.

Relative to my `5.8/10` score from Round 3, this is a small negative, about `-0.2` in spirit, but it is offset by the XMem evidence and the method-consistency reruns.

**Q4. Is the `idea1` vs `combo` XMem differential a concern or a contribution?**

**Both.**

It is a **concern** if you hide it, because reviewers will conclude `combo` is overfit to SAM2.

It is an **interesting contribution** if you foreground it as an architecture-dependent finding:

- `idea1` is the cross-tracker core
- the halo in `combo` seems to exploit SAM2's boundary sensitivity
- that halo transfers less well to XMem and is more recoverable by counter-processing

That is actually a better paper than pretending `combo` is uniformly best.

**Minimum Fixes To Reach A Clean 6/10**

`1)` **Choose the main method now.**
My recommendation is: make `idea1` the primary method, and present `combo` as a SAM2-optimized enhancement.

`2)` **Finish full `combo`-XMem if time permits.**
This is the one remaining experiment that can still move the score.

`3)` **Make the paper's tables variant-consistent.**
For every claim, be explicit whether it is `idea1` or `combo`. Do not mix them casually.

`4)` **Keep the claim narrow and honest.**
The current editorial narrowing is correct. Do not expand it again.

**Ready for AAAI submission?**

**Almost.**

The minimum remaining work is not a new big experiment set. It is:

- **freeze the method story**
- **finish `combo`-XMem if possible**
- **rewrite the paper around the actual evidence hierarchy**

If the deadline were very close, I would rather submit a **clean `idea1`-centered AAAI paper** than a muddled `combo`-centered one.

</details>

### Actions Taken (Phase C Round 4)
1. Fixed `pilot_xmem_eval.py` to add `--edit_type` argument (was hardcoded idea1)
2. Restarted XMem combo run on GPU0 (currently n=22, in progress)
3. Combo reruns running: hevc_combo_v2 (n=26), adaptive_combo_v2 (n=20), person_combo_v2 (n=19)
4. Identified key pivot: idea1-first paper scores ~6.1/10, combo-first ~5.6-5.9/10

### Status
- **MAX_ROUNDS (4/4) reached — loop terminating**
- Final score: 5.9/10 (AAAI "Almost")
- Next step: Pivot paper narrative to idea1-primary framing; finish combo-XMem run; rewrite tables for consistency

---

## Final Summary

**Score progression**: 4.0 → 5.5 → 5.8 → 5.9/10
**Verdict progression**: Not ready → Close → Borderline → Almost

**Key blocker resolved**: Robustness/validity (HEVC, person-only, adaptive-lite, XMem all added)
**Final remaining blocker**: Method identity — pivot from combo-first to idea1-primary narrative

**Recommended paper framing**:
- **Primary method**: `idea1` (boundary suppression, rw=24, α=0.8)
  - +10.69pp DAVIS H264, +11.67pp XMem, robust to adaptive-lite (9.4% recovery)
  - Generalizes to HEVC (1.34-1.54×), person-only (+12.13pp, 0 neg)
- **Enhanced variant**: `combo` (idea1 + halo)
  - +16.36pp DAVIS (clean→defended), SAM2-specific gain
  - Less robust to adaptive-lite (38.5% combined recovery)
  - Weaker XMem transfer (~+4.4pp vs +11.67pp for idea1)
- **Core claim**: "Codec-amplified boundary disruption for publisher-side VOS degradation; effectiveness is architecture-conditioned"
- **Scope**: DAVIS-style videos with clear boundaries; not universal; content-conditioned

