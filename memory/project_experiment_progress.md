---
name: Experiment Progress — Semantic Boundary Suppression (2026-03-28)
description: Current experimental status and key findings for semantic video privacy paper
type: project
---

**Current phase**: Full sweep running (5 GPUs, ~2h remaining as of 10:05 server time 2026-03-28)

**Why**: Pivot from negative-result Paper 1 (pixel attacks fail under H.264) to positive Paper 2 (semantic boundary suppression survives H.264 and suppresses SAM2 tracking).

**How to apply**: Use these results to inform paper writing and next experiment decisions.

## Key Results So Far

- **combo_strong (mask prompt)**: mean ΔJF_codec ≈ +13.7pp at SSIM ≈ 0.88
- **combo_strong (point prompt)**: mean ΔJF_codec ≈ +38.1pp (3× stronger!) at SSIM ≈ 0.88
- **idea2 (echo contour) alone**: +0.7pp — negligible, not needed
- **Single-frame attack**: +0.9pp — memory poisoning does NOT drive the effect; persistent editing required

## Running Experiments (2026-03-28)

| Tag | Status | ETA |
|-----|--------|-----|
| full_combo (std, mask) | 13/85 | ~12:30 |
| full_idea1 (boundary only) | 21/85 | ~12:00 |
| full_idea2 (echo only) | 22/85 | ~12:00 |
| full_combo_strong (mask) | 21/85 | ~12:00 |
| full_combo_strong_point (point) | 6/85 | ~12:30 |
| full_global_blur (queued, GPU 4) | queued | ~12:30 |
| full_combo_strong_small (queued, GPU 2) | queued | ~13:00 |

## Scripts

- `pilot_mask_guided.py` — main sweep script (edit_type: idea1, idea2, combo, global_blur, global_bright)
- `pilot_memory_ablation.py` — single-frame vs all-frame ablation
- `analyze_mask_guided.py` — cross-run analysis once results complete
- Results: `results_v100/mask_guided/<tag>/results.json`

## Paper Story

Publisher-side privacy preprocessor: GT-mask-guided boundary suppression (combo_strong params: ring=24, alpha=0.8) applied to all frames survives H.264 CRF23 and reduces SAM2 J&F by 14-38pp depending on prompt type. No ML training required (pure image processing). Works best with point prompts (~38pp drop) since boundary ambiguity amplifies the edit.
