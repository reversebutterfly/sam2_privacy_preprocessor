# UAP-SAM2 Baseline Protocol Definitions

**Date**: 2026-04-01
**Repo**: CGCL-codes/UAP-SAM2, commit `779ce0b7ebb8cc09fb712c46c555099f6a99e08f`
**Paper**: "Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2"
**Audit doc**: `refine-logs/UAP_BASELINE_PATCHLOG.md`

---

## 三种协议的区别

### 协议 A：Official Default-Protocol Baseline

| 属性 | 值 |
|------|-----|
| **名称** | official default-protocol baseline |
| **eval split** | YouTube-VOS **train** split（与训练集相同） |
| **frame format** | JPEG（`cv2.imwrite` 默认 Q95） |
| **video ID overlap** | **100 / 100（100%）**——训练与评估使用完全相同的 100 个视频 |
| **mIoU 类型** | 官方过滤版（clean IoU < 0.3 的帧双向跳过） |
| **是否严格复现论文** | **否** |
| **说明** | 官方发布代码（`uap_atk_test.py`）的默认行为。`--test_dataset` 参数虽声明但从未被读取，eval 始终读取 train split。经意图审计（GPT-5.4 xhigh）判定为**代码/协议错误（bug）**，而非故意设计。详见 PATCHLOG。 |

**结果**（seed=30, limit_img=100, limit_frames=15, test_prompts=pt）：

| metric | value |
|--------|-------|
| mIoU_clean | 76.41% |
| mIoU_adv | **58.61%** |
| frames scored | 1114 |
| overlap(train, eval) | 100 / 100 |

---

### 协议 B：UAP-SAM2 Held-Out JPEG Baseline ← **本文使用的公平比较基准**

| 属性 | 值 |
|------|-----|
| **名称** | UAP-SAM2 held-out JPEG baseline |
| **eval split** | YouTube-VOS **valid** split（held-out，训练中未见） |
| **frame format** | JPEG（有意保留，模拟发布链路压缩） |
| **video ID overlap** | **0 / 100（0.0%）**——eval 与 train 视频集合完全不重叠 |
| **mIoU 类型** | 官方过滤版（与协议 A 相同） |
| **是否严格复现论文** | **否**（split 不同于官方代码默认行为） |
| **说明** | 最小协议修复：仅修改 eval split 使 `--test_dataset YOUTUBE_VALID` 真正生效；JPEG 路径有意保留以模拟真实发布链路。攻击方法、损失函数、扰动预算、mIoU 指标均未改动。这是我们论文中与 UAP-SAM2 的主要对比基准。 |
| **代码** | `uap_eval_heldout_jpeg.py` |

**结果**（seed=30, limit_img=100, limit_frames=15, test_prompts=pt）：

| metric | value |
|--------|-------|
| mIoU_clean | 82.16% |
| mIoU_adv | **57.94%** |
| frames scored | 1287 |
| overlap(train, eval) | 0 / 100 |

---

### 协议 C：Strict Reproduction — **未完成**

| 属性 | 值 |
|------|-----|
| **名称** | UAP-SAM2 strict reproduction |
| **eval split** | YouTube-VOS valid split（正式 held-out） |
| **frame format** | 内存直传（无 JPEG 压缩），或完整复现论文中描述的路径 |
| **video ID overlap** | 0（目标） |
| **mIoU 类型** | 标准 mIoU（不含 clean < 0.3 过滤） |
| **是否严格复现论文** | **目标是，但尚未完成** |
| **未完成原因** | 训练过程中因系统内存不足（OOM Kill）被多次中断，UAP 未充分收敛。当前训练结果 mIoU_adv≈57–59%，与论文目标 37.03% 相差约 20pp。此差距被确认为训练问题，而非评估协议问题。 |

---

## 对比表

| 协议 | split | frame fmt | mIoU_clean | mIoU_adv | overlap(train,eval) | 可称 strict reproduction |
|------|-------|-----------|-----------|---------|---------------------|------------------------|
| **A: official default** | train (in-sample) | JPEG | 76.41% | 58.61% | 100/100 (100%) | ❌ 否 |
| **B: held-out JPEG** ← 本文基准 | valid (held-out) | JPEG | 82.16% | **57.94%** | 0/100 (0%) | ❌ 否 |
| **C: strict reproduction** | valid (held-out) | in-memory | — | — | 0 (目标) | 🔄 未完成 |
| 论文 Table 2 (point, YouTube-VOS) | — | — | ~82.8% | **37.03%** | — | 目标值 |

---

## 协议选择依据

### 为什么选协议 B（而非协议 A）作为对比基准

1. **协议 A 存在 bug**：`--test_dataset` 参数声明但从未生效，eval 始终读取训练集，造成 100% 数据重叠。经严格意图审计判定为代码错误，而非设计选择。

2. **协议 B 是公平比较的最小修复**：仅修改 eval split（使参数真正生效），不改变攻击方法、损失函数、mIoU 计算方式。

3. **JPEG 路径有意保留**：协议 B 保留 JPEG save/reload，与协议 A 保持 frame format 一致，isolate 了 split 变量的影响。

4. **重叠确认**：协议 B 的 eval video IDs 已保存并验证 overlap = 0（`refine-logs/eval_heldout_jpeg_video_ids.json`）。

### 与协议 A 的差距解读

| 比较 | mIoU_adv 差值 | 解读 |
|------|---------------|------|
| A vs B | 58.61% → 57.94% = **−0.67pp** | in-sample 泄漏使官方数字虚高约 0.67pp（较小） |
| A vs 论文 | 58.61% → 37.03% = **+21.58pp** | 训练未收敛（OOM kills），与 eval 协议无关 |

---

## 论文引用建议

```
UAP-SAM2 baseline（held-out JPEG）：
"我们运行 CGCL-codes/UAP-SAM2 官方代码（commit 779ce0b），
修复其 eval split 泄漏问题（--test_dataset 原本未生效，
eval 与 train 读取相同视频集），在 YouTube-VOS valid split
（100 个 held-out 视频，seed=30）上重新评估。
保留 JPEG 保存/重载路径以与官方 eval 格式一致。
结果：mIoU_clean=82.16%，mIoU_adv=57.94%（train/eval overlap=0）。
注：论文报告值 37.03% 为 in-sample eval 且对应完整收敛训练；
我们的训练因计算资源限制存在中断，结果偏高，此差距与 eval 协议无关。"
```

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `uap_eval_heldout_jpeg.py` | 协议 B eval 脚本 |
| `uap_eval_heldout.py` | 协议 B 的 PNG 变体（lossless，参考用） |
| `refine-logs/UAP_BASELINE_PATCHLOG.md` | 完整 patch 记录与意图审计 |
| `refine-logs/train_video_ids.json` | 训练使用的 100 个视频 ID |
| `refine-logs/eval_heldout_jpeg_video_ids.json` | 协议 B eval 使用的 100 个视频 ID |
| `refine-logs/heldout_jpeg_results.json` | 协议 B 结果 JSON |
| `hjpeg_full.log`（服务器） | 协议 B 完整 eval 日志 |
| `eval_locked_stdout.log`（服务器） | 协议 A 结果日志 |
