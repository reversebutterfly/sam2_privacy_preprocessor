# SAM2 Privacy Preprocessor — Idea Summary

**生成日期**: 2026-03-18
**状态**: 设计完成，待实验验证（auto-review 评分 6/10 ALMOST）

---

## 一句话描述

一个**发布端批量视频预处理器** `g_θ`，对视频帧施加视觉不可察觉的残差扰动，使 SAM2 及其变体无法稳定分割/追踪视频中的人物，且扰动在 H.264/H.265 重编码后仍然有效。

---

## 问题定位

**受众**：公开视频数据集发布者（研究机构、竞赛数据团队）
**威胁**：第三方下载数据集后，用 SAM2 标准提示（点击/框）自动提取人物轨迹，构成隐私风险
**现有方案的缺陷**：人脸打码/身体模糊显眼、损害数据效用、可被部分还原
**本方案的边界**：不声明对所有追踪系统的全面匿名化；仅针对 SAM2 家族 + 被动对手

---

## 方法：7 个模块化组件

| 模块 | 功能 |
|------|------|
| **Learnable residual preprocessor** | 主干网络，输出低幅残差，最小化视觉差异 |
| **Decoy / competition branch** | 在背景非目标区域注入竞争性线索，劫持 SAM2 streaming memory bank 的写入，增加长时追踪失败概率 |
| **Temporal consistency module** | 约束相邻帧残差不产生可见闪烁；可选结构化跨帧 drift 增加长时失效 |
| **Perceptual constraint** | LPIPS≤0.10, SSIM≥0.95, VMAF≥90 感知预算（hinge loss，非 lp-norm） |
| **Codec-aware EOT** | 训练时在 H.264/H.265 模拟变换分布上做期望优化，使扰动在真实发布链路（FFmpeg 重编码）后仍有效 |
| **Surrogate ensemble** | 联合在 SAM2-T/SAM2-S/SAM2.1 上训练，提升黑盒迁移性 |
| **Evaluation harness** | 长时失效曲线 + 编码后鲁棒性 + 下游效用测试 |

### 训练阶段
- **Stage 1（MVP）**：residual + 感知预算 + 单 surrogate → 验证训练闭环
- **Stage 2**：加入 temporal consistency
- **Stage 3**：加入 codec-aware EOT
- **Stage 4（完整系统）**：加入 decoy 分支 + surrogate ensemble

### 优化目标
```
L = L_attack + λ₁·L_perceptual + λ₂·L_temporal + λ₃·L_decoy

L_attack_codec = E_{T~P_codec}[ -mIoU_drop(f_SAM2(T(g_θ(V))), P) ]
P_codec = { H264(CRF∈{18,23,28}), resize(0.9×–1.1×), gaussian_blur(σ∈{0,0.5,1}) }
```

---

## 5 个核心 Claim

### Claim 1：发布端优先（Deployment-first）
目标是数据集发布前的**离线批量处理**，而非实时防御。
约束：视觉不可察觉 + 编码鲁棒 + 下游效用保留。

### Claim 2：感知预算约束训练
以 LPIPS/SSIM/VMAF 为硬预算（hinge loss），而非 UAP-SAM2 仅用的 lp-norm ε。
**关键实验**：证明满足感知预算的同时，分割指标显著下降。

### Claim 3：Codec-aware EOT 使扰动在重编码后仍有效
UAP-SAM2 的 lp-norm 扰动能量集中于高频分量，H.264 量化会将其摧毁。
**关键实验**：UAP-SAM2 重编码后 J&F 跌幅接近 0；本方法重编码后 J&F 跌幅保持。
这是本论文与 UAP-SAM2 的**最核心分水岭**。

### Claim 4：诱饵竞争分支（Decoy-induced memory competition）
SAM2 memory bank 默认存储 ≤7 个 object pointer token。
背景区域注入的诱饵线索与目标竞争 memory slot → 目标 pointer token 被驱逐/稀释 → 长时追踪失败。
**关键实验**：Stage 3 vs. Stage 4（decoy on/off）在 60+ 帧处的 J&F 曲线；memory cross-attention 熵变化诊断。
这是**新颖性最强的子组件**（文献中无直接先验工作）。

### Claim 5：下游效用保留
预处理后视频仍可用于合法研究任务：
- YOLOv8n 行人检测 mAP 下降 < 5%
- MMPose ViTPose-S 骨骼估计 PCKh 下降 < 5%
- 众包标注者接受率 > 90%

---

## 最接近先验工作

| 论文 | 与本工作的重叠 | 关键差异 |
|------|---------------|---------|
| **UAP-SAM2** (NeurIPS 2025 Spotlight) | 攻击 SAM2 视频跨帧传播，含 memory misalignment 组件 | 无感知预算、无 codec 鲁棒、非发布端场景、无诱饵机制 |
| **DarkSAM** (NeurIPS 2024) | 针对 SAM 的 prompt-free universal attack | 图像模式，无视频/时序，无编码鲁棒 |
| **RoVISQ** (NDSS 2023) | Codec-surviving 对抗扰动 | 针对视频分类器，非分割/追踪，非隐私场景 |
| **AdvCloak/Fawkes** | 面部隐私对抗扰动 | 静态图像，人脸识别目标，无视频/编码鲁棒 |

---

## 实验方案（1× A100，~42h）

| 实验 | 数据集 | 目的 | 工时 |
|------|--------|------|------|
| UAP-SAM2 编码前后对比（核心差距证明） | DAVIS, YT-VOS | Claim 3 | 10h |
| Stage 3 vs. Stage 4 长时 J&F 曲线 | DAVIS, MOSE | Claim 4 | 8h |
| 诱饵因果消融（随机放置/结构化/memory 诊断） | DAVIS | Claim 4 机制 | 6h |
| 下游效用测试（YOLOv8, MMPose） | DAVIS | Claim 5 | 5h |
| Baseline 等价对比（UAP-SAM2+LPIPS，相同训练时长） | DAVIS, YT-VOS | 公平对比 | 8h |
| 提示鲁棒性压测（多点/box/多 conditioning 帧） | DAVIS | 威胁模型扩展 | 5h |

**主结果表指标**：J&F↓（DAVIS, YT-VOS, MOSE），Post-H264 J&F↓，LPIPS/VMAF（质量），mAP/PCKh（效用）

---

## 潜在分数预测

| 实验结果场景 | 预期分数 | 建议投稿目标 |
|------------|---------|------------|
| 三项全中（codec gap + decoy 增量 + utility 保留） | **7/10** | CVPR/NeurIPS 边缘可投 |
| Codec gap 实证 + utility，decoy 增量弱 | 6/10 | CVPR workshop / 安全会议 |
| 全部实验结果弱 | 3–4/10 | 需重大重写 |
| 三项全中，投安全会议 | **8/10** | **USENIX Security / IEEE S&P（推荐）** |

---

## 伦理声明要点

- **不开放权重**，不提供 turnkey 训练脚本（仅发布评测代码）
- 明确声明：不等同于匿名化，不适用于实时系统规避
- 处理速度约 0.5s/帧，不适合实时对抗应用
- 推荐投稿目标：**USENIX Security / IEEE S&P / ACM CCS**（安全顶会对防御性隐私工具框架更成熟）

---

## 主要风险

| 风险 | 缓解方案 |
|------|---------|
| 编码鲁棒性验证失败（proxy ≠ 真实 FFmpeg） | 同时用可微 proxy 训练 + 真实 FFmpeg 测试，报告 gap |
| 诱饵分支无增量价值 | 降调为"minor contribution"，靠 codec gap 撑主故事 |
| CVPR/NeurIPS 伦理阻断 | 转投 USENIX Security，同等质量可能更高评分 |
| UAP-SAM2 版本更新使 codec 鲁棒性差距缩小 | 在提交时加入最新 SAM2 版本测试 |
