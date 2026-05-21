# AAAI 论文初稿大纲

> **目标会议**：AAAI（8 内容页 + 无限引用，双栏 letter，~7000–8000 词）
> **状态**：方法 / 故事 / Related Work 已可写；实验仅占位（待数据回来填）
> **本文件**：按 AAAI 结构组织的可写内容。中文起草，提交前译英。

---

## Title（候选）

- **HTCS: Hierarchical Task-Conditioned Saliency for Vision-Language-Action History Compression**
- **Closing the Loop: Language-Conditioned Codec Compression for Vision-Language-Action Models**
- **Task-Conditioned Bi-Granularity History Compression for Vision-Language-Action Policies**

> 倾向第一个——含方法名 + 关键技术词 + 任务。

---

## Abstract（~180 词，已可写）

Vision-language-action (VLA) models for long-horizon manipulation require history compression to handle partial observability without inflating token budgets. However, all existing VLA history modules—both compression-based (HiF-VLA, OneVision-Encoder, CoPE-VideoLM) and memory-based (MEM, π0.7, MemoryVLA, ReMem-VLA)—treat the short-term visual encoder as **task-agnostic**. Even in MEM, whose chain-of-thought reasoner continuously emits task-relevant language signals (sub-instructions and rolling summaries), those signals never flow back to condition visual perception. We propose **HTCS** (Hierarchical Task-Conditioned Saliency), the first framework that injects task language into VLA history compression itself. HTCS introduces **bi-granularity language conditioning**: (i) at the parameter level, a direction-aware codec saliency function whose weights and target direction vector are derived from language, recovering manipulation-critical motion semantics ("push left" vs. "push right") that prior codec-based VLAs uniformly discard; (ii) at the structure level, K compression slots derived from the language sequence that select patches via competitive cross-attention. The two granularities are demonstrably non-redundant. HTCS exposes a planner-compatible language-only interface, validated as a plug-in low-level module for modern dual-system VLA stacks via a sub-instruction switching probe and adapter integration with a public VLA checkpoint, with main results on LIBERO and CALVIN benchmarks.

---

## 1. Introduction（已可写）

### 1.1 问题动机
真实世界机器人操作普遍违反马尔可夫假设——遮挡导致任务相关状态部分可观测、多步流程携带隐式进度、自然语言指令本身就常引用过去事件。近一年 VLA 工作已形成共识：单帧观测不足以支撑长程操作（HiF-VLA, MEM, MemoryVLA, π0.7）。然而朴素堆叠多帧会损害而非帮助性能——它膨胀 VLM token 预算、稀释 attention、放大对历史无关线索的过拟合。**问题不在"要不要用历史"，而在"如何压缩历史"**。

### 1.2 现有路线及共同盲点
近一年压缩 / 记忆方向同时涌现两条路线：

- **压缩路线**：HiF-VLA、OneVision-Encoder、CoPE-VideoLM 利用 codec 副产物（MV、Residual、I/P 分布）做稀疏化
- **记忆路线**：MEM、π0.7、MemoryVLA、ReMem-VLA 分尺度处理短期密集编码 + 长期摘要

两条路线**共同盲点**：短期视觉编码器全部任务无关——无论指令是"抓黑碗"还是"放酒瓶"，同一段历史观测都用同一套压缩规则。任务条件化要么完全没有，要么只发生在压缩之后的末端融合层。

MEM 是这一盲点最具揭示性的反例：其 π_HL 在 chain-of-thought 解码中持续产出任务相关语言信号（子指令 l_t + 滚动摘要 m_t），但这些信号**只往动作端流动，从未回流条件化视觉感知**。架构上语言信号唾手可得，整条路线却全员错过。

### 1.3 核心洞察
任务条件化不是单一操作，而是**有不同 granularity 的层级**：
- **参数级**（决定"看哪种运动"）：codec 信号本身有多个分量，不同任务对它们的相对重要性不同
- **结构级**（决定"看哪些 patch"）：在每帧的具体 patch 上做语言驱动的选择

两个 granularity 互补、不冗余、且仅需语言信号（不依赖 proprioception）。

### 1.4 Contributions（标准 AAAI 列表，3 条 mid-granularity）

We make three contributions:

1. **HTCS framework**: We propose HTCS, the first task-conditioned history compression framework for VLA. We identify a perception-conditioning gap left open by all prior compression- and memory-based VLAs, whose short-term visual encoders remain task-agnostic even when rich task language flows elsewhere in the system. HTCS closes this loop, making the compression result a function of the current instruction.
2. **Bi-granularity language conditioning**: We propose cascading parameter-level direction-aware codec saliency (α, β, γ, v_tgt all derived from language) with structure-level slot cross-attention (K language-derived queries). The two granularities are demonstrably non-redundant, and the direction-aware MV·v_tgt term recovers motion semantics ("push left" vs. "push right") that all codec-based VLAs uniformly discard.
3. **Validated plug-in low-level module**: We validate HTCS as a low-level perception module for modern dual-system VLA stacks (Hi Robot, π0.5/π0.6, Gemini Robotics 1.5, MEM π_HL) through (i) a sub-instruction switching probe simulating planner outputs and (ii) adapter integration with a public VLA checkpoint via frozen-backbone LoRA fine-tuning. Main results on LIBERO and CALVIN.

---

## 2. Related Work（已可写）

### 2.1 Codec-based Visual Compression
- **CoViAR (CVPR'18)**：原创用 MV + Residual 做视频识别；取 MV 模长
- **OneVision-Encoder (2026)**：HEVC 启发的 ViT patch 选择，多模态架构基础原则；任务无关
- **CoPE-VideoLM (2026.02)**：codec I/P 帧二分作 VideoLM token 预算；任务无关
- **HiF-VLA (CVPR 2026)**：首次将 MPEG-4 MV 作 VLA 历史输入，AdaLN 末端融合任务条件
- 共同特征：取 MV 模长丢方向；任务条件化（若有）仅在压缩之后

### 2.2 Memory Architectures for VLA
- **MEM (PI 2026.03)**：双尺度——短期 ViT 时空注意力 + 长期 chain-of-thought 滚动语言摘要（π_HL 自回归生成，训练用教师 LLM 监督）
- **π0.7 (PI 2026.04)**：沿用 MEM 编码器 + multi-modal prompt
- **MemoryVLA (2025.08)**：Perceptual-Cognitive Memory Bank + working memory token
- **ReMem-VLA (2026.03)**：帧级 + 块级双层循环 query
- **MemER / EchoVLA**：外置记忆库 / 检索式
- 共同特征：短期层为相对朴素的密集编码，且任务无关

### 2.3 Dual-System VLA Architectures（高层 planner + 低层 policy）
- **Hi Robot (PI 2025)**：高层 VLM 输出自然语言子指令，低层 π0 执行
- **π0.5 / π0.6**：同一模型内 hierarchical inference，离散解码出子任务，flow-matching 执行
- **Gemini Robotics 1.5 (DeepMind 2025.10)**：显式 orchestrator + Action Model 分工
- **Helix-02 (Figure)**：三层 S2/S1/S0，接口为 latent embedding
- **HiRT, Fast-in-Slow, RoboBrain 2.0**：均落入慢 VLM (1–10 Hz) + 快 VLA (50–200 Hz) 模板
- 共同特征：**高层 planner 不负责低层视觉历史记忆**——这是 HTCS 的设计 niche

### 2.4 Chain-of-Thought VLA
- **ECoT (Embodied Chain-of-Thought, 2024)**：CoT 嵌入 VLA 内部，planner/policy 合一
- **CoT-VLA**：类似思路
- 与 HTCS 关系：HTCS 是 perception-side conditioning；ECoT 是 decoding-side reasoning，两者正交

### 2.5 本工作定位
HTCS 是**第一个**把任务语言信号注入短期视觉历史压缩本身的工作。与上述工作的差异：
- 与 §2.1 不同：任务条件化在**压缩阶段内部**而非末端融合
- 与 §2.2 不同：HTCS 替换其朴素短期层，与其长期层正交可组合
- 与 §2.3 不同：HTCS 是低层 policy 内部的感知模块，消费 planner 输出但不承担规划
- 与 §2.4 不同：HTCS 作用于 perception 而非 decoding

---

## 3. Method（已可写）

### 3.1 Overview
给定 T 帧历史 \(\{I_{t-T},\dots,I_{t-1}\}\) 与语言指令 \(\ell\)，HTCS 输出 K 个任务相关摘要词元 \(H \in \mathbb{R}^{K\times d_{\text{vlm}}}\) 喂入下游融合模块。流水线分两个 granularity 级联条件化：

```
T 帧 ──HEVC──► I-frame RGB + P-frame MV(二维) + Residual
                      ↓
        Stage 1：参数级语言条件化 codec saliency → 稀疏 patch 集 P
                      ↓
        Stage 2：结构级语言条件化 patch 压缩 → K 个 summary token H
                      ↓
        M4 cross-attention 注入动作查询 → 动作头 → a_t
```

### 3.2 Stage 1: Direction-Aware Language-Modulated Saliency
对每个 P-frame patch 计算 saliency
$$
s = \alpha(\ell)\|MV\| + \gamma(\ell)(MV\cdot v_{tgt}(\ell)) + \beta(\ell)|Residual|
$$
其中：
- \((\alpha, \beta, \gamma) = \text{softplus}(W_3 \cdot \text{pool}(\ell_{\text{emb}}))\)
- \(v_{tgt} = W_2 \cdot \text{pool}(\ell_{\text{emb}}) / \|\cdot\|\) 单位向量
- 三路信号经 per-instance percentile 归一化后加权融合
- I-frame 全部保留；P-frame 按 \(s\) 排序取 top ρ%（ρ≈20%）

**关键创新**：方向项 \(\gamma(MV\cdot v_{tgt})\) 让"推左 vs 推右"的相同模长在不同指令下产生**不同**的 patch 集——填补 codec-based VLA 系列"取模长丢方向"的盲点。

### 3.3 Stage 2: Language-Conditioned Patch Compression
K 个 slot 由语言序列经 MHA 派生（**不池化**）：
$$
\text{slots} = \text{MHA}(Q=Q_{\text{learn}}^{K}, K=V=\ell_{\text{emb}}) \in \mathbb{R}^{K\times d_p}
$$
与 Stage 1 输出 P 做 competitive softmax cross-attention（沿 K 维归一）+ Stage 1 saliency 作 attention bias：
$$
A_{logits} = (\text{slots}\cdot P^T)/\sqrt{d} + \lambda \cdot s_{\text{stage1}}, \quad A = \text{softmax}_K(A_{logits})
$$
$$
H = \text{Linear}(A\cdot P) \in \mathbb{R}^{K\times d_{\text{vlm}}}
$$

### 3.4 训练
- M4 cross-attention 输出端加 LayerScale (γ_ls = 0 初始化) 平滑接入
- text_embeds 初期 detach 保持 VLM 预训练对齐稳定
- 单阶段端到端训练；动作头 L1 / flow-matching 可切换

### 3.5 Design Properties
- **纯语言驱动**：所有任务条件来自 \(\ell\)，不需要 proprioception
- **跨形态可迁移**：图像平面 v_tgt 与机械臂坐标解耦，人类示范视频亦可用
- **Plug-in adapter**：新增参数 ~15–25M（与 LoRA rank=8 同档），可冻结 backbone 微调集成进现成 VLA

---

## 4. Experimental Setup（结构可写，数字待填）

### 4.1 Benchmarks
- **LIBERO**：4 个套件（Spatial / Object / Goal / Long），均联合训练 30K 步
- **CALVIN ABC-D**：长程多任务，第二基准
- （RoboTwin 作为可选第三基准，时间允许时加）

### 4.2 Baselines
- **Single-frame** VLA：无历史
- **Naive stacking**：朴素堆叠 N 帧
- **OneVision-Encoder 风格**：固定权重 codec saliency（任务无关）
- **Late-fusion-only**：HTCS 压缩任务无关，语言仅在末端融合处看到
- **HiF-VLA 风格 baseline**（如可复现）

### 4.3 Metrics
- 主指标：任务成功率（按套件分 + 平均）
- 次指标：token 预算 / 推理延迟 / FLOPs
- 跨任务一致性指标（标准差或最差套件成功率）

### 4.4 关键消融
- **E1（命门）**：反事实指令替换（同段历史，替换无关指令）测成功率下降
- **E8**：codec-aware vs 同 token 预算 RGB 选取
- **E9（非冗余证据）**：Stage 1 / Stage 2 patch 集合 IoU
- **E10**：去 Stage 2 退化为 OneVision + 池化
- **E13**：saliency 三档（OneVision 等权 / 仅 αβ / 完整含 γv_tgt）
- **E16**：方向反事实（"推左 vs 推右"产生不同 patch 集）

### 4.5 Plug-in 评测（C3 配套）
- **Sub-instruction switching probe**：LLM 离线分解 LIBERO-Long → 时间窗喂入子指令；度量 (α, β, γ, v_tgt) 边界跳变 + slot 注意力转移 + 成功率
- **Adapter 集成**：冻结公开 VLA ckpt 的 backbone + LoRA HTCS 模块（候选 GR00T-2B / π0-3B），5–10K 步微调

### 4.6 实现细节
- VLM：Qwen3-VL-0.8B
- Codec：HEVC，GOP=8，preset=medium
- 图像分辨率 224²，SigLIP-Large patch 14×14
- T = 16，K = 8，ρ = 20%
- 硬件 / 训练时长（待填）

---

## 5. Results（占位，等数据）

> 待 §9.2 第一里程碑完成后填充：主表 + 全部消融 + 可视化。
>
> 待 §9.3 第二里程碑完成后填充：sub-instruction probe + adapter 集成 + CALVIN 主表。

---

## 6. Discussion（部分可写）

### 6.1 Why bi-granularity matters
（基于 E9 IoU + E13 跨任务参数分布讨论非冗余性——等数字回来填）

### 6.2 Direction conditioning recovers manipulation-specific semantics
（基于 E16 方向反事实讨论 v_tgt 的可解释性——等数字回来填）

### 6.3 HTCS as a low-level module for dual-system VLA
讨论 HTCS 与 Hi Robot / π0.5 / Gemini Robotics 1.5 / MEM π_HL 的天然接口对齐——这部分**现在就可以全写**，纯定位讨论：

> HTCS 通过纯语言接口与现代分层 VLA 架构兼容。两条独立的正交维度：(A) 与高层任务规划器正交——HTCS 消费 planner 输出的子指令作 ℓ；(B) 与长期记忆抽象层正交——HTCS 替换 MEM-style 多尺度系统的短期密集编码层。这两个正交是独立的：A 沿任务向前分解，B 沿时间向后回溯。

### 6.4 Limitations
- 只在 1 个公开 VLA ckpt 上做 adapter 集成
- Sub-instruction probe 用 LLM 伪造而非真实 planner 端到端联动
- 双尺度（HTCS + 长期摘要层）联合训练留 future work
- v_tgt 当前是全局 2D 单位向量，per-frame / spatial 扩展未做

---

## 7. Conclusion（占位）

总结 HTCS 的三条贡献（C1/C2/C3），重申"闭合 perception-conditioning 回路"的 framing，指出 future work（多尺度 HTCS+MEM 组合 / 多 ckpt adapter / 真 planner 联动）。

---

## 当前可立刻动笔的章节

| 章节 | 状态 | 现在可写比例 |
|---|---|---|
| Abstract | ✅ | 100% |
| 1. Introduction | ✅ | 100% |
| 2. Related Work | ✅ | 100%（等小调整） |
| 3. Method | ✅ | 95%（细节张量形状已在工作文档） |
| 4. Experimental Setup | ✅ | 90%（实现细节硬件待填） |
| 5. Results | ⬜ | 0%（等实验） |
| 6. Discussion §6.3 | ✅ | 100% |
| 6. Discussion §6.1/6.2/6.4 | ⬜ | 仅大纲 |
| 7. Conclusion | ⬜ | 仅大纲 |

**建议动笔顺序**（不必等实验）：Abstract → Related Work → Method → Introduction → Discussion §6.3 → 4.1/4.2/4.3 → 6.4 Limitations
