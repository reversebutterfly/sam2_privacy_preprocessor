# Idea Discovery Report — SAM2-Guided Preprocessing

**Direction**: 预处理时让SAM2分割一次，用分割结果指导处理，使别人无法再次通过SAM2分割
**Date**: 2026-03-27
**Pipeline**: research-lit → idea-creator (GPT xhigh) → novelty-check → research-review (GPT xhigh)
**Ideas evaluated**: 10 generated → 4 deep-validated → 1 recommended combo + 1 backup

---

## Executive Summary

**可行性：是，但有条件。**

核心机制是：发布者用SAM2对frame 0做一次分割，得到对象掩码M，然后对视频所有帧的M区域进行
"边界抑制 + 外部伪边界"处理（低频语义编辑）。该编辑存活H.264 CRF23压缩，并通过持续污染
SAM2内存库（maskmem_features）导致下游视频追踪失败。

**最小验证实验**：在6-8个DAVIS视频上，使用GT掩码，测试Ideas 1+2组合的post-codec J&F下降。
**成功标准**：mean post-codec J&F下降 ≥ 5pp（用GT掩码）。
**终止标准**：If 1+2 with GT masks < 5pp → 方向太弱，停止。

**论文创新点（一句话）**：
"Unlike prior gradient-based SAM/SAM2 UAP attacks, we study a publisher-side, mask-guided,
low-frequency video edit that uses one-shot SAM2 localization to induce downstream SAM2
tracking failure after H.264 compression, without adversarial optimization."

---

## Literature Landscape

### 已有工作（均有局限）

| 论文 | 目标 | 与本方向的差距 |
|------|------|--------------|
| DarkSAM (NeurIPS 2024) | SAM图像分割 | 无视频/时序记忆，无编码测试，基于SAM非SAM2 |
| Vanish/UAP-SAM2 (NeurIPS 2025) | SAM2视频追踪 | 梯度UAP (L∞约束)，无H.264测试，非发布者侧 |
| UnSeg (NeurIPS 2024) | 分割不可学习性 | 训练时保护，非推理时预处理 |
| Multi-Scale Privacy Video (2021) | 视频隐私 | 语义分割+模糊，目标是人眼可识别，不针对SAM2 |
| Ilic et al. CVPR 2024 | 动作识别隐私 | 时序一致局部混淆，但不针对SAM2/视频追踪 |

### 结构性空缺（确认新颖）

1. ✅ 无论文将 SAM2自身输出 + H.264存活 + SAM2视频追踪失败 三者结合
2. ✅ 无论文研究低频语义编辑（非L∞像素攻击）对SAM2时序记忆的影响
3. ✅ 无论文将发布者侧单次分割→持续语义编辑→下游追踪失败作为主题

---

## Ranked Ideas（来自GPT xhigh idea-creator）

### 🏆 Idea 1+2 Combination — **RECOMMENDED**

**名称**：Alpha-Matte Boundary Suppression + Exterior Echo Contour

**机制**：
- **Idea 1（边界抑制）**：将掩码M周围8-24px的环形区域（边界环）与邻近背景做低频alpha混合，
  使真实边界变得模糊。SAM2的边界token变成FG/BG混合状态 → maskmem_features写入偏小/渗漏的掩码
  → 记忆污染随帧数累积。
- **Idea 2（外部伪边界）**：在真实轮廓外几像素处添加一圈平滑的光影/明暗变化，制造竞争性假边界。
  SAM2的patch token同时看到两条可信边界 → 预测可能跳到外层假边界 → 引入背景区域到记忆中。
- **组合1+2**：同时抑制真边界+创建假边界，双向攻击SAM2的figure-ground separability判断。

**为什么存活H.264**：边界alpha混合和平滑光影变化都是低频内容，H.264 DCT保留低频 → 编码后编辑仍在。

**为什么破坏SAM2时序记忆**：
每帧写入maskmem的掩码都略有偏差（真边界模糊/假边界干扰）→ K=6帧记忆条目全部是偏差版本
→ 当记忆库全部被污染条目填满后（~6帧后），追踪完全失败。
这与单帧攻击（单帧损坏被K-1个干净条目"稀释"）有根本区别：**全帧持续编辑让整个记忆库持续污染**。

**视觉质量**：若环宽8-16px、混合强度0.3-0.6，看起来像轻微散焦/光晕，在SSIM≥0.85 LPIPS≤0.15预算内。

**可行性**：2天实现（纯numpy/cv2），~4-6 GPU-hours评估。

**评审评分（GPT xhigh）**：
- 若5pp post-codec drop：4/10（弱拒绝，需要更强结果或更深机制分析）
- 若8-12pp + 自动掩码鲁棒性 + 记忆污染证据：6-7/10（接近接受）
- 对应NeurIPS 2026 accept的目标：8-12pp均值，auto-mask有效，记忆路径因果证明

---

### Idea 3: Patch-Scale Boundary Superpixelization — BACKUP

**机制**：在掩码边界环中，将像素按16-32px大小的superpixel展平，匹配SAM2 Hiera ViT的patch尺度。
这使边界处的每个patch token包含混合FG/BG统计，在记忆编码之前就污染了特征提取。

**架构特定性**：直接针对SAM2的patch token大小（Hiera ViT，patch=16），是最有理论依据的攻击。

**与Idea 1+2的关系**：可作为第3个实验变体，验证"攻击特定patch scale"的假设。

---

### Idea 4: Local Clone Decoy — BACKUP（高风险/高回报）

**机制**：将对象的一个粗粒度复制片段粘贴/warping到附近背景中，与真实对象运动同步。
SAM2的obj_ptr（对象指针）对应两个候选，attention扩散/跳转 → 记忆污染。

**优势**：最直接针对SAM2的obj_ptr（对象指针），DarkSAM和Vanish都没有专门针对obj_ptr的设计。
**风险**：视觉质量难以控制；若decoy不够自然，video明显被篡改。

---

## Eliminated Ideas

| 想法 | 淘汰原因 |
|------|---------|
| 全局颜色/风格变换（全帧） | 评审Round 1已证实：一致全局变换只是给SAM2定义新跟踪域，不会破坏追踪 |
| 单帧语义编辑（任何类型） | 实验已证实：SAM2时序鲁棒性，单帧在1-2帧内恢复 |
| 像素级L∞攻击 | H.264编码完全消除高频扰动（4轮实验结果） |
| Background co-motion sheath (Idea 5) | 需要光流，实现复杂，且"运动同步"比"边界操作"更容易被SAM2忽略 |
| Temporal bimodal appearance cycling (Idea 8) | 双稳态外观交替可能看起来像闪烁，视觉质量难以控制 |

---

## 关键警告（来自评审）

### 1. 效果量 vs. 创新性
- 5pp post-codec drop → 弱拒绝（"tuned boundary artifact that mildly hurts SAM2"）
- 需要 8-12pp + 复合证据（auto-mask, memory contamination, temporal compounding）才能进入接受区间

### 2. 循环依赖问题
攻击使用SAM2自己的掩码来指导处理。如果SAM2掩码不准（遮挡、快速运动），攻击也失败。
解决方案：设计三档掩码精度实验：
- Oracle（GT掩码）
- Practical（SAM2 frame-0 auto-mask + 光流传播）
- Noisy（人工劣化的掩码，不同IoU级别）
如果三档性能平滑下降 → reviewers接受；如果突然崩溃 → 被批"brittle and model-coupled"

### 3. 会议目标
- ECCV 2026：已关闭（截止2026-03-05）
- NeurIPS 2026：deadline 2026-05-04/06（约5周）
  - 需要在约2026-04-07前有可信正结果才有时间准备投稿
- 如果效果弱：NeurIPS E&D track更合适
- CVPR 2027：最充裕，内容最匹配（视觉会议）

---

## 最小验证实验设计

```
实验名: pilot_mask_guided
数据: DAVIS 2017 val，6-8个视频（dog-agility, elephant, flamingo, drift-straight, drift-turn + 2-3个）
掩码: GT掩码（oracle上界）
方法: Idea 1, Idea 2, Idea 1+2组合
参数: 边界环宽度 {8, 16, 24}px，混合强度 {0.3, 0.5, 0.7}
评估: VideoPredictor + H.264 CRF23，point AND mask prompt
指标: mean J&F, mean F (boundary), SSIM, LPIPS
成本: ~4-6 GPU-hours on 1 V100
```

**成功标准**：Idea 1+2 with GT masks → mean post-codec J&F drop ≥ 5pp
**终止标准**：< 5pp even with GT masks → direction too weak，转负面结果论文

---

## 实施路线图（对应research-review给出的实验序列）

| 日期 | 任务 | GPU-h |
|------|------|-------|
| 2026-03-27/28 | 实现Ideas 1, 2, 1+2（纯numpy/cv2）+ mask-prompt eval | 2-3 |
| 2026-03-29 | 在6-8视频运行pilot（GT掩码，point+mask prompt） | 4-6 |
| 2026-03-30/31 | 若有信号：优化参数，测试SAM2 auto-mask | 4-6 |
| 2026-04-01/03 | 若有信号：扩展到全DAVIS val，mask精度消融 | 8-10 |
| **2026-04-07** | **Hard decision checkpoint** | — |

---

## 与现有论文（paper/main.pdf）的关系

- **现有论文**：negative result，"H.264 Codec as Adversarial Purifier"（像素攻击失败）
- **本新方向**：positive result，"Publisher-Side SAM2-Guided Semantic Obfuscation"
- **论文关系**：清晰的Paper 2 → Paper 1说"所有现有L∞攻击失败"，Paper 2说"我们的语义方法成功"
- **共同narrative**：H.264编码是一道屏障，像素攻击无法通过，但结构性语义编辑可以通过

---

## 下一步

1. **立即**：实现 Idea 1（alpha-matte boundary suppression）+ Idea 2（exterior echo contour）
2. **今天/明天**：在6-8个DAVIS视频上运行pilot（GT掩码，H.264 CRF23）
3. **2026-04-07**：根据结果决定：正面论文 or 强化负面论文
