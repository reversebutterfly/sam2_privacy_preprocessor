# Auto Review Log — SAM2 Privacy Preprocessor (Fancy Methods Round)
**Goal**: 引入 MSB / AdvOpt / LFNet 三种 fancy 方法，将 AAAI 评分从 5.9 推向 ≥6.5
**Started**: 2026-04-06
**MAX_ROUNDS**: 4
**Reviewer**: GPT-5.4 xhigh (via Codex MCP)
**Prior loop**: 4 rounds, final score 5.9/10 ("Almost AAAI")

---

## Round 1 (2026-04-06)

### Assessment (Summary)
- **AAAI Score**: 5.8/10 (not yet moved above 6/10)
- **ICCV/CVPR Score**: 4.7/10
- **Verdict**: Not ready — methods proposed but no results yet; proxy unvalidated

### Key Criticisms
1. No actual fancy-eval results (just code) — can't score a hypothetical method
2. Proxy loss (pre-codec boundary gradient) not validated against post-codec ΔJF
3. Still reads as "boundary blur + more machinery" without one clear principled method
4. Per-video adaptation overstated (only first-frame, should be labelled accurately)
5. Utility evaluated on first 5 frames only (weak)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Scores**

AAAI 2026: `5.8/10`
ICCV/CVPR level: `4.7/10`

My answer to your core question is: `no`, the addition of `MSB` / `AdvOpt` / `LFNet` does not yet clearly move this above `6/10`.

**Critical Weaknesses**

1. The new contribution is still mostly hypothetical, not demonstrated. I found the code and pilot script, but no saved fancy-method results under `results_v100/fancy_eval`. Minimum fix: run full matched-utility results for `idea1`, `msb`, `adv_opt`, and one chosen advanced method on DAVIS.

2. The proxy story is the main scientific gap. `AdvOpt` optimizes pre-codec boundary gradient suppression, but your headline phenomenon is post-codec degradation. The code never models codec in the loop. Minimum fix: show correlation between proxy value and actual post-codec `ΔJF`, or add a codec-aware surrogate.

3. The work still reads as "boundary blur with more machinery." `MSB` is multi-scale ring blending, `AdvOpt` tunes the same family, `LFNet` learns weights for the same operation. Minimum fix: pick one main advanced method and demote the others to ablations.

4. "Per-video" adaptation is overstated. `AdvOpt` and `LFNet` are fit on the first annotated frame. Minimum fix: rename honestly as first-frame adaptation.

5. Utility control weaker than framing suggests. Optimizer uses rough `SSIM ~= 1 - 20*MSE` approximation; eval measures quality on only first 5 frames.

**Verdict**: READY for AAAI submission: `No`. Minimum package to reach 6/10: choose one method, verify actual gains, validate proxy correlation, run compact robustness package.

</details>

### Actions Taken (Phase C Round 1)
1. Fixed AdvOpt clamp bug (`torch.clamp(x, float, Tensor)` → `(x).clamp(min=0)`)
2. Fixed quality eval: all frames capped at 20 (was first 5 only)
3. Renamed "per-video" → "first-frame adaptation" in pilot script
4. Added `validate_proxy_correlation.py`: Pearson/Spearman proxy vs ΔJF
5. **Proxy validation result (n=88)**: r=-0.097, p=0.37 — weak cross-video correlation
   - Finding: suppression ratio varies little (~0.63-0.77); ΔJF variance driven by content/prompt
   - Conclusion: WITHIN-video correlation is the right test for AdvOpt (not cross-video)
6. Fixed MSB: additive/incremental fusion replaces max-weight (bear: msb was +9.7pp < idea1's +37.4pp)
7. Launched experiments on remote GPU 2 (fancy_eval: idea1/msb/adv_opt on all DAVIS, point prompt)
8. **Early results (bear, point prompt)**: idea1=+37.4pp, msb=+9.7pp (pre-fix), adv_opt=+72.9pp (α=0.928)

### Status
- Continuing to Round 2

---

## Round 2 (2026-04-06, n=15 preliminary)

### Assessment (Summary)
- **AAAI Score**: 6.3/10 — crossed 6/10 threshold!
- **ICCV/CVPR Score**: 5.2/10
- **Verdict**: Almost ready for AAAI

### Key Progress Since Round 1
- AdvOpt: 2.3× better than idea1 (53.5pp vs 23.3pp, n=15, point prompt)
- At matched SSIM ≥ 0.90: 1.77× better (53.1pp vs 30.0pp, 11/15 videos)
- Within-video proxy r=+0.811 validates AdvOpt foundation
- MSB informative negative: alpha level > spatial coverage

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Scores**

AAAI: `6.3/10`
ICCV/CVPR: `5.2/10`

AdvOpt materially changes the picture. You now have the two things Round 1 was missing: real paired gains and a proxy validation in the regime that actually matters. That pushes this above the AAAI 6 line, but not by a comfortable margin.

**Remaining Weaknesses**

1. Main result is still only 15/88 videos. Minimum fix: finish full DAVIS sweep, report mean/median/95%CI/paired win-rate/paired significance test for AdvOpt vs idea1.

2. Proxy validation shows monotonicity, not optimal selection. Minimum fix: oracle-gap experiment comparing proxy-chosen alpha to oracle alpha that maximizes post-codec ΔJF under same SSIM constraint.

3. "2.3× at matched SSIM" wording is vulnerable. Minimum fix: use Pareto plot or iso-SSIM matching; otherwise use paired gain: "+30.2pp paired gain over idea1, 14/15 wins."

4. Method can be dismissed as per-video alpha tuning. Minimum fix: make AdvOpt the only primary method, use MSB as negative ablation, drop LFNet unless it clearly beats AdvOpt.

5. Quality constraint not fully under control. Minimum fix: enforce SSIM bound harder or explicitly report constrained success rate + failure cases.

6. Generalization still thin. Minimum fix: one compact extra axis for AdvOpt (HEVC, adaptive-lite, imperfect masks, or second backbone).

**Direct Answers**: AAAI 6.3/10, crosses 6/10? Yes narrowly, "2.3×" genuine signal yes / phrasing vulnerable. Ready: Almost.

</details>

### Actions Taken (Phase C Round 2)
1. Stronger SSIM constraint: penalty 5.0 → 20.0 in optimize_adv_params
2. Launched HEVC + adaptive-lite generalization for AdvOpt (GPU 5)
3. Added oracle-gap analysis script + Pareto plot to validate_proxy_correlation.py
4. Waiting for full 88-video results; will add paired stats analysis

### Status
- Continuing to Round 3 after full results + generalization package complete

---

## Round 3 (2026-04-06, n=28 paired)

### Assessment (Summary)
- **AAAI Score**: 6.7/10
- **ICCV/CVPR Score**: 5.6/10
- **Verdict**: Almost ("close but not ready")

### Key Progress Since Round 2
- AdvOpt: n=28, mean +57.3pp, median +68.4pp, SSIM 0.938
- idea1: mean +24.4pp, median +12.7pp, SSIM 0.953
- Paired gain: +32.9pp; win-rate: 25/28 (89%); Wilcoxon/t-test p≈0
- Oracle gap: proxy-chosen alpha matches oracle in 98% of tested cases (SSIM≥0.90)
- SSIM penalty 5.0→20.0 materially reduced quality failures

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Scores**

AAAI: `6.7/10`
ICCV/CVPR: `5.6/10`

AdvOpt materially changes the picture. The oracle-gap result (98% proxy-oracle match) is the strongest single scientific argument for the method.

**Remaining Weaknesses**

1. Main sweep still 28/88 — finish full DAVIS sweep before submission
2. Generalization thin: HEVC n=3, mask-prompt n=2; need one axis at ~20 videos
3. Novelty narrow: "optimize two parameters of handcrafted suppression on first frame"; reframe as proxy-validated parameter selection
4. Utility: only SSIM — add one non-tracking utility check or narrow claim

**Verdict**: Almost. Full sweep + one generalization axis → reasonable AAAI submission.

</details>

### Actions Taken (Phase C Round 3)
1. Launched full remaining 62-video sweep: `screen fancy_r4` (idea1+adv_opt, point prompt, all remaining DAVIS)
2. Launched HEVC generalization sweep: `screen hevc_r4` (20 videos, idea1+adv_opt, HEVC codec)
3. Pulled `eval_utility_adv.py` (LPIPS + YOLO utility) to remote server via `git pull`
4. Note: previous fancy_v1/hevc/mask experiment directories were empty (crashed on missing `paragliding-lift` video); re-launched with `--videos` list filtered to available videos

### Status
- Continuing to Round 4 after experiments complete

---

## Round 4 (2026-04-06, n=36 paired — FINAL ROUND)

### Assessment (Summary)
- **AAAI Score**: 6.9/10
- **ICCV/CVPR Score**: ~5.8/10
- **Verdict**: Almost — "No today, but close"

### Key Progress Since Round 3
- Combined n=36 paired (28 orig + 8 new fancy_r4), trend stable: adv_opt mean=57.7pp vs idea1=24.7pp
- Win-rate: 33/36 = 91.7%; t-stat=7.68 (p<<0.001); SSIM=0.936
- HEVC generalization (n=8 of 20): adv_opt mean=58.5pp vs idea1=31.7pp — trend holds
- **NEW Utility results (n=20)**: SSIM=0.928, LPIPS=0.096, YOLO recall=55.2% vs idea1 77.5%
  - Honest tradeoff: stronger privacy at measurable downstream utility cost
- Reframing: AdvOpt as "constrained-optimal parameter selection" (converges to rw≈24, α≈0.93)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**AAAI Score**: `6.9/10`

**Verdict**: `Almost` — submitted today = borderline reject; after rewrite + full sweep = 7.2–7.5

**Key Strengths**:
- Paired comparison strong: n=36, 91.7% win-rate, t=7.68, p<<0.001
- Oracle gap 98% is the strongest scientific argument
- Proxy-validated constrained parameter selection is a legitimate contribution

**Critical Weaknesses**:
1. Paper manuscript still describes old negative-result story (not AdvOpt) — submission artifact not ready
2. Main sweep 36/90 — incomplete; simple methods demand full validation
3. Utility tradeoff real: YOLO recall 55% vs 78% — cannot claim broad utility preservation
4. Optimizer converges to universal params (rw≈24, α≈0.93) — need comparison vs global fixed setting

**Minimum to reach ≥7/10**:
1. Full DAVIS sweep → mean/median/CI/win-rate/significance
2. HEVC n=20
3. Global-fixed-vs-AdvOpt comparison (shows adaptation = validation of universal operating point)
4. Complete manuscript rewrite: narrow to "proxy-validated constrained selection," acknowledge utility cost

</details>

### Actions Taken (Phase C Round 4 — MAX_ROUNDS reached)
1. Round 4 is the final round; loop terminates
2. Experiments still running on GPU: fancy_r4 (36/90 done), hevc_r4 (8/20 done) — continue to completion independently
3. Fixed eval_utility_adv.py bug (optimize_adv_params call); committed and pushed
4. **Remaining blockers identified** (see Final Status below)

### Final Status — LOOP COMPLETE (MAX_ROUNDS=4)

**Score progression**: 5.8 → 6.3 → 6.7 → **6.9/10**

**Remaining blockers before AAAI submission**:
1. **[Critical]** Finish main DAVIS sweep to n=90; run paired stats (2–3h on GPU, fancy_r4 already running)
2. **[Critical]** Finish HEVC n=20 (hevc_r4 running, ~45 min remaining)
3. **[Critical]** Rewrite paper manuscript (LaTeX) around AdvOpt narrative — current draft still describes old paper
4. **[High]** Add global-fixed-setting comparison (single tuning on held-out, apply universally)
5. **[Medium]** Narrow all utility claims to "visual quality preservation (SSIM≥0.90)"

**Recommended next steps**:
- Wait for fancy_r4 and hevc_r4 to complete → run analyze_paired_stats.py on full results
- Do global-fixed comparison: run idea1 with rw=24, α=0.93 on full DAVIS
- Start LaTeX rewrite (paper/sections/abstract.tex, discussion.tex, conclusion.tex)
- Venue recommendation: AAAI 2026 if deadline allows full sweep + rewrite; else TCSVT (May 2026)

---

---

# New Auto-Review Loop — 2026-04-06 (AdvOpt Continuation)
**Goal**: Full sweep + generalization: YT-VOS full, global-fixed comparison, cross-tracker
**MAX_ROUNDS**: 4  
**Prior loop score**: 6.9/10 (MAX_ROUNDS reached, status=completed)
**New threadId**: 019d632e-19b0-77b0-80e5-d75bfd69d191

## Round 1 (2026-04-06, new loop — full results at start)

### Assessment (Summary)
- **AAAI Score**: 6.6/10
- **Verdict**: No today, but close — "proxy-validated constrained selection" framing viable
- **Key finding**: paper viable only if framed honestly as collapsing to global point (DAVIS) + heterogeneous adaptation (YT-VOS)

### Key Criticisms
1. Main novelty unstable — global-fixed comparison not yet run; adaptation claim may be oversold
2. Paper tells wrong story — still structured around gradient optimization, not proxy validation
3. Utility tradeoff substantial — YOLO recall 55% is meaningful degradation
4. Generalization incomplete — HEVC n=17, YT-VOS full pending, no second tracker
5. Manuscript hard blocker — local draft still describes old negative-result paper

### Reviewer Raw Response
See: RESEARCH_REVIEW_ADVOPT_ROUND1_20260406.md

### Actions Taken (Phase C Round 1 — new loop)
1. Launched global-fixed comparison (DAVIS): idea1(rw=24,α=0.93) vs adv_opt — `screen global_fixed` GPU 4
2. Launched XMem cross-tracker experiment: idea1(rw=24,α=0.93) through XMem — `screen xmem_adv` GPU 2
3. Launched full YT-VOS sweep (507 videos): `screen ytvos_full` GPU 0
4. Launched YT-VOS global-fixed comparison (50 videos, idea1_alpha=0.93): `screen ytvos_gfixed` GPU 3
5. Committed and pushed analyze_proxy_table.py + review docs

### Early Results (Phase D)
- Global-fixed DAVIS (n=7 preliminary): idea1(α=0.93)=58.6pp vs adv_opt=59.7pp, +1.0pp gain (t=0.98, not sig)
  → DAVIS collapses — per-video adaptation gives negligible additional benefit
- XMem cross-tracker (n=8): mean ΔJF_codec=13.1pp (vs SAM2 ~58pp), SSIM not saved in schema
  → Boundary suppression generalizes to XMem; smaller effect consistent with memory-based tracking
- YT-VOS global-fixed: just launched, results pending

### Status
- Continuing to Round 2

---

## Round 2 (2026-04-06, new loop)

### Assessment (Summary)
- **AAAI Score**: 7.1/10 — crossed 7/10 threshold!
- **Verdict**: Almost — "in plausible AAAI accept territory" after YT-VOS fixed comparison + manuscript rewrite
- **Key finding**: Global-fixed DAVIS definitively resolves adaptation claim; paper narrative now correctly framed as "operating-point discovery"

### Key Progress Since Round 1
- Global-fixed DAVIS (n=10): +0.1pp, t=0.14, win-rate 50% — DAVIS collapses ✓
- XMem cross-tracker (n=18): 18.1pp — generalization verified ✓
- HEVC confirmed (n=17): +28.5pp, t=4.20, wins=88.2% ✓
- Score: 6.6 → 7.1/10

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 7.1/10, Verdict: Almost

Q1: Global-fixed result materially resolves adaptation claim (but n=10 still thin for central headline result).
Q2: YT-VOS theoretical prediction not sufficient — need quantitative fixed-vs-opt with SSIM-violation counts.
Q3: XMem 18.1pp meets "one generalization axis" requirement with caveat (boundary transfer, not full proxy-selection story).
Q4: No blocker worse than rewrite; YT-VOS fixed-vs-opt quantitative confirmation is second priority.

Revised claim: "AdvOpt is a proxy-validated operating-point discovery and constrained selection method. On homogeneous data (DAVIS), collapses to near-universal setting. On heterogeneous data (YT-VOS), constrained per-video selection may remain useful."

Remaining Weaknesses:
1. YT-VOS heterogeneity claim one experiment short (need fixed-vs-opt with SSIM violations)
2. Proxy not validated on YT-VOS (where adaptation supposedly matters most)
3. DAVIS fixed-vs-opt n=10 too small for central claim
4. Utility tradeoff framing still weak (needs explicit privacy-utility tradeoff)
5. XMem limited transfer evidence — do not oversell
6. Manuscript rewrite still hard blocker

Minimum Fixes:
1. Finish YT-VOS fixed alpha=0.93 vs adv_opt with SSIM-bound violation counts
2. YT-VOS proxy validation table (oracle gap or rank consistency)
3. Expand DAVIS fixed-vs-opt to n≥20 or present collapsed parameter distribution
4. Rewrite paper around constrained operating-point discovery
5. Keep XMem as short generalization section only

</details>

### Actions Taken (Phase C Round 2)
1. YT-VOS global-fixed experiment running (ytvos_gfixed, GPU 3): idea1(alpha=0.93) on 50 videos
2. Monitoring global_fixed DAVIS (GPU 4): accumulating toward n≥20
3. Will compare YT-VOS idea1(alpha=0.93) vs adv_opt from ytvos_adv_n50 cross-run
4. Will compute YT-VOS SSIM violation rate for fixed alpha=0.93

### Status
- Continuing to Round 3

---

## Round 3 (2026-04-06, new loop)

### Assessment (Summary)
- **AAAI Score**: 7.2/10 (+0.1 from Round 2)
- **Verdict**: Almost — "coherent AAAI paper" but not "submit today Yes"
- **Key finding**: n=27 definitively closes DAVIS; 88% YT-VOS converges to global-fixed; proxy still only validated on DAVIS

### Key Progress Since Round 2
- Global-fixed DAVIS (n=27): +0.3pp, t=0.31, win-rate=48.1% — definitively zero ✓
- YT-VOS alpha distribution: 44/50 = 88% converges to [0.92,0.93), 6 outliers need lower alpha ✓
- SSIM violation accounting: adv_opt SSIM<0.90 for 3/50 = 6% (soft constraint) ✓
- XMem (n=19): 17.6pp mean — cross-tracker verified ✓

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 7.2/10, Verdict: Almost

Global-fixed n=27 closes DAVIS definitively. 88% YT-VOS convergence weakens heterogeneous adaptation but strengthens "near-universal operating point" claim. XMem meets generalization requirement.

Key remaining weaknesses:
1. SSIM constraint is soft (3/50 = 6% violations with adv_opt). Must report explicitly.
2. YT-VOS global-fixed comparison still pending (theoretical argument + 1 result insufficient)
3. Proxy validation only on DAVIS — where YT-VOS edge-case adaptation lives, proxy not validated
4. Novelty narrower than before — requires disciplined writing
5. Utility loss must be framed as deliberate tradeoff

Minimum fixes for Round 4:
1. Finish YT-VOS fixed(0.93) vs adv_opt with SSIM violation counts
2. Clarify SSIM soft vs hard constraint explicitly
3. Rewrite title/abstract/intro/contributions around one thesis
4. Demote optimization mechanics to implementation detail

</details>

### Actions Taken (Phase C Round 3)
1. Computed SSIM violation rates: adv_opt SSIM<0.90 → 3/50=6% (all high-alpha videos, approximation error)
2. YT-VOS global-fixed experiment still running (720p frames, slower than expected ~50-110s/video)
3. Waiting for 10+ results before final Round 4 review

### Status
- Continuing to Round 4 (FINAL) after ytvos_global_fixed accumulates ≥10 results

---

## Round 4 (2026-04-07, new loop — FINAL ROUND)

### Assessment (Summary)
- **AAAI Score**: 7.1/10 (slight drop from 7.2 due to YT-VOS ssim_floor artifact finding)
- **Verdict**: Almost
- **Key finding**: YT-VOS global-fixed experiment revealed ssim_floor mismatch (0.92 vs paper's 0.90); one clean paired comparison missing

### Key Progress Since Round 3
- YT-VOS global-fixed (n=50, ssim_floor=0.92) COMPLETED:
  - idea1(α=0.93): 17.27pp, SSIM=0.9584, 5/50 SSIM<0.92 violations
  - adv_opt(ssim_floor=0.92): 2.69pp, SSIM=0.9801, α≈0.61 for ALL videos
  - Finding: MSE proxy over-penalizes on YT-VOS at strict floor → optimizer artifact, not true constraint
- Manuscript fully rewritten: abstract, discussion, conclusion all updated
- SSIM soft constraint explicitly documented

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: 7.1/10
**Verdict**: Almost

Findings:
1. Biggest remaining problem is claim mismatch on YT-VOS. YT-VOS evidence under ssim_floor=0.90 does not directly compare to fixed-vs-opt under same protocol. Convergence to alpha≈0.93 is supportive, but not a matched paired comparison.
2. The 0.92 result exposes real fragility in the method. Not the operating-point story itself, but weakens "generally reliable constrained optimizer" claim. Stronger as "discovers good operating point under paper's soft-constraint regime."
3. Proxy validation still incomplete on YT-VOS — asymmetric: validated where it works, not where it fails.
4. Paper is now much more honest — rewrite sounds materially better. Utility drop disclosed, soft-constraint explicit, discovery separated from deployment. Right direction.
5. Practical scope narrow. XMem much smaller than SAM2, YOLO recall drops hard. Should be sold as SAM2-targeted privacy preprocessing study.

**YT-VOS Framing**: Do not omit ssim_floor=0.92 result; frame as "failure-mode ablation" showing surrogate brittleness, not direct evidence against operating point existence.

**Minimum Fix**: Run fixed(α=0.93) vs adv_opt on same YT-VOS n=50 under ssim_floor=0.90. This would likely push to 7.6-7.8/10 → Ready.

</details>

### Actions Taken (Phase C Round 4)
1. Launched `ytvos_fair_fixed_n50` (GPU 5, screen ytvos_fair): idea1(α=0.93) vs adv_opt, ssim_floor=0.90, n=50 YT-VOS
   - This is the "minimum fix" experiment; expected to show adv_opt converging to α≈0.93 matching idea1
2. Round 4 is MAX_ROUNDS — loop terminates; experiment continues independently

### Final Status — LOOP 2 COMPLETE (MAX_ROUNDS=4)

**Score progression (2nd loop)**: 6.6 → 7.1 → 7.2 → **7.1/10** (terminal, slight regression due to honest ssim artifact)

**One experiment to reach Ready (7.6-7.8/10)**:
- `ytvos_fair_fixed_n50` (running): idea1(α=0.93) vs adv_opt(ssim_floor=0.90) on 50 YT-VOS videos
- Expected: adv_opt converges to α≈0.93, paired gain ≈ 0 (equivalence), confirming operating point on YT-VOS

**Remaining manuscript notes**:
- conclusion.tex already mentions 88% convergence (consistent with ssim_floor=0.90 finding)
- Once ytvos_fair completes, update abstract/conclusion with paired gain ≈ 0 on YT-VOS
- Frame ssim_floor=0.92 result in Limitations as surrogate brittleness evidence

---

---

# New Auto-Review Loop — 2026-04-08 (Loop 3: Full evidence + manuscript)
**Goal**: Present n=497 YT-VOS + manuscript rewrites; fix YT-VOS fair JF comparison (codec bug)
**MAX_ROUNDS**: 4
**Prior loop score**: 7.1/10 (Loop 2, status=completed)
**New threadId**: 019d6b07-dffc-7243-8a59-74eafaeb15fe

## Round 1 (2026-04-08)

### Assessment (Summary)
- **AAAI Score**: 7.8/10 — new highest!
- **Verdict**: Almost
- **Key finding**: n=497 YT-VOS sweep resolves generalization concern "mostly yes"; fair JF comparison still critical blocker

### Key Progress Since Last Loop
- YT-VOS full sweep COMPLETED: n=497, idea1=7.4pp, adv=16.5pp, gain=+9.2pp, **91.9% win-rate**
- YT-VOS alpha convergence (ssim_floor=0.90, n=50): 94% at α≈0.93 confirmed
- BUT: YT-VOS fair JF comparison (codec evaluation) still NaN (codec_round_trip failed — ffmpeg not in PATH)
- Paper manuscript: abstract, discussion, conclusion previously rewritten

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: 7.8/10 — Verdict: Almost

**Remaining Blockers**:
1. Critical: YT-VOS codec-roundtrip JF comparison at matched quality still missing. n=497 proves cross-dataset gain over α=0.80 baseline, not the stronger "universal operating point" claim. Alpha convergence is strong indirect evidence, not outcome evidence. Minimum fix: run n=50 YT-VOS fair comparison at ssim_floor=0.90 and report paired ΔJF, win-rate, violation rate.
2. High: Proxy validation DAVIS-only. Minimum fix: small YT-VOS proxy check (within-video correlation, 30-50 videos), or narrow claim to "validated on DAVIS, transferred elsewhere."
3. High: Contribution identity vulnerable — "if global-fixed ≈ AdvOpt, what's the contribution?" Must frame explicitly: AdvOpt is offline discovery/calibration tool → deployable global point.
4. Medium-High: "constrained" overstated → "soft surrogate-constrained" everywhere. Keep violation rates in main results.
5. Medium: Baseline narrow (only vs α=0.80); utility sample small (n=20, one detector).
6. Medium: Sample accounting unclear — protocol table needed.

**Direct Answers**:
- n=497 resolves generalization: "Mostly yes" for weak claim, "Not fully" for universal-point claim.
- Fair JF comparison still a blocker for claims 2 and 3.
- Score: 7.8/10.

</details>

### Actions Taken (Phase C Round 1)
1. **Re-launched ytvos_fair_v2** (GPU 4, screen ytvos_fair_v2): fixed ffmpeg path bug (was using "ffmpeg" in PATH; now uses explicit conda env path). idea1(α=0.93) vs adv_opt(ssim_floor=0.90), n=50.
2. **Rewrote introduction.tex**: complete rewrite for AdvOpt paper with proper contribution bullets; AdvOpt framed as "offline calibration tool → deployable global point."
3. **Fixed abstract.tex**: "soft surrogate-constrained," added "offline calibration tool" framing.
4. **Fixed discussion.tex**: renamed SSIM section to "Soft Surrogate Constraint," added Protocol Accounting Table (Table 1), documented ssim_floor brittleness.
5. **Fixed conclusion.tex**: added "offline calibration tool" framing.
6. Committed and pushed all changes (commit de501ee).

### Status
- Continuing to Round 2 after ytvos_fair_v2 completes

---

## Strategy Review (2026-04-06, via GPT-5.4 xhigh)

**Questions**: idea1/combo 地位？是否跑 YT-VOS adv_opt？下一步实验？

### Q1: idea1 是基线还是共同主方法？
**→ 基线**。idea1 锚定 privacy-utility tradeoff 的低扰动端，不作为主要贡献。

### Q2: combo 如何处理？
**→ 降为 ablation / appendix 负面对照**。有用价值：证明"更激进的扰动"在跨追踪器上反而退化。

### Q3: 是否在 YT-VOS 上跑 adv_opt？
**→ 不要**。原因：optimizer 已收敛到近通用参数（rw=24, α=0.93），YT-VOS 失败是内容分布问题，不是参数问题。跑出来是弱/中性结果，对论文有害无益。

### Q4: 下一步最高价值实验（优先级）
1. **[最高] B: Global-fixed (rw=24, α=0.93) vs per-video AdvOpt on DAVIS**
   - 解决最大战略风险："adaptation 是否必要？还是 optimizer 只是发现了一个通用最优点？"
   - 若二者相近 → claim 变为"发现稳定的 codec-robust 工作点"；若 AdvOpt 更好 → adaptation 是真贡献
2. **[高] D: XMem + adv_opt on DAVIS**
   - 跨追踪器迁移验证，允许 claim 不只是 SAM2-specific
3. **[中] C: Mask robustness with adv_opt**
   - 解释 YT-VOS gap 的机制：瓶颈是掩膜/内容质量，不是超参
4. **[低] A: adv_opt on YT-VOS** — 当前最低优先级

---

## Round 2 (2026-04-08) — LOOP 3 FINAL

### Assessment (Summary)
- **Score**: 8.0/10
- **Verdict**: Ready after minor revisions
- **Key finding**: YT-VOS fair comparison closes main empirical blocker; AdvOpt story confirmed as offline calibration tool; manuscript method section added

### Key Progress Since Round 1
- **ytvos_fair_v4 COMPLETED** (n=50 YT-VOS, H.264 CRF=23):
  - idea1(α=0.93): 17.27pp | AdvOpt(ssim_floor=0.90): 16.45pp
  - Paired gain: −0.82pp, t=−2.08, n=50
  - **94% of gains within ±5pp** (near-equivalence for vast majority)
  - 88% of AdvOpt videos: alpha>0.92 (converges to universal point)
  - 12% surrogate brittleness: 6/50 backed-off videos (adv_alpha as low as 0.335), all with idea1 SSIM≥0.96
  - Both methods: identical SSIM violation rate (3/50 = 6%)
- **manuscript method.tex ADDED**: BRS definition (Eq.1), AdvOpt objective, Algorithm 1 box, evaluation protocol
- **main.tex**: corrected title + method section included + old attack sections commented out
- **abstract.tex**: fair comparison result + global-fixed deployment recommendation
- **discussion.tex**: protocol table updated (fair comparison row added) + operating-point section rewritten with honest surrogate brittleness analysis

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: 8.0/10

**Verdict**: `Ready after minor revisions` on the science. The new YT-VOS fair comparison closes the main empirical blocker from the last round. It also settles the correct story: AdvOpt is not the best deployment policy; it is an offline calibration/discovery procedure that finds the global-fixed operating point. If you submit with that framing, this is now in the accept-range for an AAAI-level review. If the manuscript still truly lacks a proper method section, then the paper package is still practically `Almost`.

**Remaining Critical Weaknesses**:
1. The contribution framing is still fragile. On the fairest YT-VOS comparison, AdvOpt is slightly worse than fixed α=0.93, so any wording that presents AdvOpt as the deployed method will get attacked. Minimum fix: make global-fixed BRS at (r=24, alpha=0.93) the deployment recommendation everywhere, and describe AdvOpt as the offline calibration tool that discovered it.
2. The manuscript structure sounds incomplete. A top-tier reviewer will not forgive "old sections commented out" plus "new methodology section not yet written." Minimum fix: add a real Method section with BRS definition, AdvOpt objective, first-frame-only protocol, codec pipeline, and one clean algorithm box or pseudocode block.
3. Sample accounting still needs to be airtight. Minimum fix: add a single evaluation-accounting table that explains exactly what each subset is, why it differs, and whether failures/exclusions occurred.
4. The proxy claim is now narrower than the title-level emphasis suggests. It is strong on DAVIS, but YT-VOS shows a real 12% brittle regime. Minimum fix: explicitly say "proxy-validated on DAVIS; observed brittle on 12% of YT-VOS cases," and do not imply broader proxy validity than you measured.
5. Utility evidence is still relatively thin. Minimum fix: if you have time, enlarge the utility set or add one more detector/metric; if not, keep utility claims modest.

**Bottom Line**: Yes, these findings are sufficient to move the work to about 8/10. The main remaining risk is no longer missing experiments; it is whether the paper is written to match the evidence you now have.

</details>

### Actions Taken (Phase C Round 2)
1. **ytvos_fair_v4**: completed (n=50), results incorporated into discussion.tex and abstract.tex
2. **paper/sections/method.tex**: NEW — BRS definition, AdvOpt algorithm box, codec pipeline, evaluation protocol
3. **paper/main.tex**: updated title, included method.tex, commented out old attack sections
4. **paper/sections/abstract.tex**: added fair comparison result (−0.82pp, 94% within ±5pp) + global-fixed deployment recommendation
5. **paper/sections/discussion.tex**: added fair comparison row to protocol table; rewrote operating-point section with surrogate brittleness analysis
6. Committed all changes (commit e4ece51)

### Final Status — LOOP 3 COMPLETE (score ≥ 8, "Ready after minor revisions")

**Score progression (Loop 3)**: 7.8 → **8.0/10** ✓

**Score progression (all loops)**: 5.8→6.3→6.7→6.9 | 6.6→7.1→7.2→7.1 | 7.8→**8.0** ✓

**Remaining for full submission** (per reviewer):
1. Ensure all sections frame AdvOpt as offline calibration tool (not deployed method)
2. Expand or explicitly narrow utility claims (n=20, one detector)
3. Write results/experiments section (currently inline in discussion)
4. Compile full paper LaTeX and verify it builds correctly

**Venue recommendation**: AAAI 2026 or TCSVT (May 2026) — science is submission-ready; manuscript needs final polish.

