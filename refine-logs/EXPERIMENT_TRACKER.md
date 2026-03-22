# Experiment Tracker — SAM2 Privacy Preprocessor

_Last updated: 2026-03-19_

| Run ID | Milestone | Block | Purpose | System / Variant | Dataset | Key Metrics | Priority | Status | Notes |
|--------|-----------|-------|---------|------------------|---------|-------------|----------|--------|-------|
| R001 | M0 | B0 | Sanity: overfit 1 video, check gradient flow | Stage 1 (residual + perceptual), 500 steps, 1 video | DAVIS-mini (1 video, ≤30 frames) | J&F drop on overfit video (target ≥30%), LPIPS (target ≤0.15) | MUST | TODO | Gate for all downstream runs |
| R002 | M1 | B1 | Baseline: UAP-SAM2 pre/post codec | UAP-SAM2 reimpl (lp-norm ε=8/255, 2000 steps, SAM2-T) | DAVIS-train-mini (10v) → DAVIS-val-20 | J&F pre-codec ↓, J&F post-H264-CRF23 ↓, LPIPS, SSIM | MUST | TODO | Confirm codec kills lp-norm attack |
| R003 | M1 | B1 | Baseline: fair-budget UAP-SAM2 + LPIPS | UAP-SAM2 + hinge LPIPS ≤0.10, same budget as R002 | DAVIS-train-mini → DAVIS-val-20 | J&F post-H264-CRF23 ↓ | MUST | TODO | Anti-C: show budget alone doesn't help |
| R004 | M2 | B2 | Main: Stage 1 training (residual + perceptual) | g_θ Stage 1 only, 1000 steps, SAM2-T | DAVIS-train-mini (10v) → DAVIS-val-20 | J&F pre-codec ↓, LPIPS ≤0.10, SSIM ≥0.95 | MUST | TODO | Checkpoint saved for Stage 2 init |
| R005 | M2 | B2 | Main: Stage 2 (+ temporal consistency) | g_θ Stage 1+2, 2000 steps total | DAVIS-train-mini (10v) → DAVIS-val-20 | J&F pre-codec ↓, J&F post-H264 ↓, temporal smoothness | MUST | TODO | Checkpoint saved for Stage 3 init |
| R006 | M2 | B2 | Main: Stage 3 (+ codec-aware EOT) | g_θ Stage 1+2+3, 3000 steps total, CRF∈{18,23,28} EOT | DAVIS-train-mini (10v) → DAVIS-val-20 | J&F post-H264 CRF{18,23,28} ↓, LPIPS ≤0.10, SSIM ≥0.95, VMAF ≥90 | MUST | TODO | **Core result** — target: post-H264 J&F drop ≥12% |
| R007 | M3 | B3 | Ablation: Stage 2 post-codec (no EOT) | g_θ Stage 2 checkpoint from R005, no retraining | DAVIS-val-20 | J&F post-H264-CRF23 ↓ vs R006 | MUST | TODO | No training; reuse R005 ckpt; expect gap ≥8 pp vs R006 |
| R008 | M3 | B4 | Supporting: Stage 4 (+ decoy branch) | g_θ Stage 3+4, 1000 additional steps, λ₃=0.1 | DAVIS-val-20 long clips (≥60 frames) | J&F curve at frames {10,20,30,40,50,60+} vs R006 | MUST | TODO | Target: ≥5 pp gap at frame≥40; demote if negative |
| R009 | M3 | B5 | Diagnostic: memory attention entropy | Stage 3 vs Stage 4 inference, hook attention layers | DAVIS-val-20 (5 videos >60 frames) | Attention heatmaps, pointer token cosine similarity decay | NICE | TODO | Run only if R008 shows positive J&F gap |
| R010 | M4 | B6 | Utility: YOLOv8n person mAP | Stage 3 (R006) vs original video | DAVIS-val-20 | YOLOv8n mAP drop (target <5 pp) | MUST | TODO | Off-the-shelf inference, no retraining |
| R011 | M4 | B6 | Utility: MMPose ViTPose-S PCKh | Stage 3 (R006) vs original video | DAVIS-val-20 | ViTPose-S PCKh drop (target <5 pp) | MUST | TODO | Off-the-shelf inference, no retraining |
| R012 | M4 | B6 | Utility: Stage 4 utility check | Stage 4 (R008) vs original video | DAVIS-val-20 | YOLOv8n mAP drop, PCKh drop | MUST | TODO | Run after R008; reuse YOLOv8/MMPose from R010/R011 |
| R013 | M4 | B7 | Robustness: prompt stress test | Stage 3 (R006), vary prompts {1-pt, 5-pt, box, 3-cond-frames} | DAVIS-val-20 | J&F drop post-H264 per prompt type | NICE | TODO | No retraining; reveals threat model limits |

---

## Summary: Must-Run vs. Nice-to-Have

| Priority | Run IDs | Total Est. Time |
|----------|---------|-----------------|
| MUST | R001–R008, R010–R012 | ~21 h |
| NICE | R009, R013 | ~3 h |

---

## Stop/Go Decision Gates

| After | Decision Gate |
|-------|--------------|
| R001 | J&F drop ≥30% overfit AND no gradient errors → proceed to R002 |
| R003 | UAP-SAM2 post-codec gap ≤3 pp → codec vulnerability confirmed → proceed to R004 |
| R006 | Post-H264 J&F drop ≥12% AND LPIPS ≤0.10 → **core claim proven** → proceed to R007 |
| R006 | Post-H264 J&F drop <8% → recheck EOT proxy calibration; do NOT proceed to paper writing |
| R007 | Stage 2 post-codec < Stage 3 post-codec by ≥8 pp → codec EOT isolated → proceed to R008 |
| R008 | J&F gap at frame≥40 ≥5 pp → decoy claim supported → run R009 |
| R008 | No gap → demote decoy to appendix; continue to R010 |

---

## Result Log (fill in as runs complete)

| Run ID | Date | J&F pre-codec | J&F post-H264-CRF23 | LPIPS | SSIM | VMAF | Notes |
|--------|------|--------------|---------------------|-------|------|------|-------|
| R001 | — | — | — | — | — | — | |
| R002 | — | — | — | — | — | — | |
| R003 | — | — | — | — | — | — | |
| R004 | — | — | — | — | — | — | |
| R005 | — | — | — | — | — | — | |
| R006 | — | — | — | — | — | — | |
| R007 | — | — | — | — | — | — | |
| R008 | — | — | — | — | — | — | |
| R010 | — | YOLOv8 mAP (orig): | YOLOv8 mAP (prep): | — | — | — | |
| R011 | — | PCKh (orig): | PCKh (prep): | — | — | — | |
| R012 | — | — | — | — | — | — | |
