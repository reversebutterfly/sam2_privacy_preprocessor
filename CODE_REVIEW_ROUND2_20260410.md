# Code Review Round 2 — 2026-04-10
**Reviewer**: GPT-5.4 xhigh
**ThreadId**: 019d75a7-5908-7fc2-a719-345ebc382a72
**Context**: Verification of 9 bug fixes + new issues search

## New Issues Found

| # | 严重度 | 问题 | 文件 |
|---|--------|------|------|
| 1 | **严重** | distill closure 仍可能 apples-to-oranges（max_frames/crf 不一致） | oracle_distill.py |
| 2 | **严重** | eval_baselines SSIM 匹配不成立（5帧选参，不回算全视频） | eval_baselines.py |
| 3 | **高** | compute_ssim 对全黑图返回 0.9 而非 1.0（den+1e-8 偏置） | src/losses.py |
| 4 | **高** | Bug #4 只修了一半（clean codec 失败时退回 jf_clean，pipeline 不一致） | eval_baselines.py |
| 5 | **中** | codec 帧数不一致时静默截断不报错 | src/codec_eot.py |
| 6 | **中** | pilot_mask_guided 仍只记录前5帧 SSIM | pilot_mask_guided.py |
| 7 | **低** | project_to_budget 是迭代近似，残余误差 1e-3~4.7e-3 | oracle_mask_search.py, oracle_distill.py |

## Fix Verification Results

| 旧Bug | 修复状态 |
|-------|----------|
| #1 per-frame iso-budget | ✅ 修复正确（uniform sector ≈ archived BRS，差异 1 pixel） |
| #3 train/test dedup | ⚠️ 部分修复（只删 overlap，不 fail-fast，不去重内部） |
| #4 codec fallback | ⚠️ 部分修复（adv 侧修了，clean 侧仍 fallback） |
| #5 SSIM 全帧 | ⚠️ 部分修复（oracle 修了，pilot_mask_guided 和 eval_baselines 没修） |
| #7 奇数 n_angular | ✅ |
| #8 torch total_area | ✅ |
| #9 未使用形参 | ✅ |

## Key Confirmations

- `_apply_old_brs_proxy` 的 mask 内部置均值 + blur 是有意设计，不是 bug
- sector 和 uniform BRS 的 ring 构造一致（同一套 dilate/erode + blur）
- PNG 中间步骤无额外量化（无损）
- 跨脚本默认 prompt 不一致（pilot=mask, baselines=point, oracle=point）— 需检查实际实验命令
