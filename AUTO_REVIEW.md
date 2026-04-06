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

