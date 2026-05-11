# VLA 历史/记忆压缩相关工作调研

> 主线：**多尺度记忆与历史压缩（Multi-Scale Memory）**。其他相关方向仅做一句话定位。

> **本次更新（2026-05）**：在 §1.1 多尺度记忆之外，新增 §1.2（外置记忆库 / 检索式）、§1.3（隐式历史编码与 token 压缩）、§1.4（系统级记忆 / 反馈式）三类近期工作（2025.08–2026.02），覆盖 MemER / HAMLET / SD-VLA / EchoVLA / MAP-VLA / EvoVLA / MindExplore / Long-VLA。详见 §1.5 横向对比。

---

## 1.1 多尺度记忆与历史压缩（Multi-Scale Memory）

这一方向的共同假设：机器人决策需要的"历史"是**异质**的——既有秒级的视觉细节（被遮挡物体的位置、刚刚发生的动作轨迹），也有分钟级的语义状态（哪些子任务已完成、当前在菜谱的哪一步）。把这两类信息塞进同一个上下文窗口既费算力又拟合不好，因此最近的工作开始**显式拆分尺度**，分别用不同机制承载。

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **MEM: Multi-Scale Embodied Memory for VLA**<br>*Physical Intelligence, 2026.03 (arXiv:2603.03596)* | 把记忆按时间尺度拆为两路：（a）**视频编码器**承载短时密集观测（秒级），（b）**LLM 摘要式语言记忆**承载长时语义（分钟级）。基于 π₀.₆ + Gemma3-4B 实现，可完成做三明治、清理厨房等多分钟任务。 | VLA 仅消费当前观测，难以处理需要跨多个抽象层级保留信息的长时程任务；朴素拼接所有历史会突破实时性约束，且训练—推理分布偏移严重。 | **(1) 短时视频编码（零新增参数）**：每隔几层 ViT 插入时间注意力，空间/时间注意力分离；**只把当前时间步的 patch 表征传递下去，丢弃所有过去时间步的 patch**——保持与单帧 VLA 相同的词元预算。<br>**(2) 长时语言记忆（LLM 摘要）**：高层策略同时预测下一个子任务和更新后的语言记忆串；训练数据由预训练 LLM 总结历史子任务并显式压缩冗余（如把多条物体描述合并成一句），从而避免训练—推理分布偏移。 |
| **π0.7: a Steerable Generalist Robotic Foundation Model**<br>*Physical Intelligence, 2026.04 (arXiv:2604.15483)* | π0 系列最新模型（5B 总参数）。**直接继承 MEM 的视频历史编码器**，并扩展为多模态 prompt 条件化（subtask 指令 + subgoal 图像 + episode metadata）。**每相机 6 帧 × 1s 间隔**的短时窗口，压缩到与单帧相同的 token 预算。展示零样本跨形态泛化和组合泛化能力。 | π0.6 之上要进一步增强 steerability——使模型可被多种上下文信号（不只是语言指令）驱动。 | **(1) 历史编码**：沿用 MEM 的视频编码器（每 4 层 ViT 插入时间注意力 + 丢弃过去时间步 patch）。**关键决策**：历史 token **直接进 VLM backbone**（block-causal masking，观测 token 内部双向注意力），不走旁路。<br>**(2) 历史预算**：6 帧 × 1s = 6 秒，最终 token 数 = 单帧 token 数。<br>**(3) 多模态 prompt 条件化**：subtask instruction + 最多 3 张 subgoal 图（多视角）+ episode metadata（speed / quality / mistakes / control mode）。subgoal 图 token 可以双向 attend observation。<br>**(4) 历史编码本身依然任务无关**：语言在 prompt 层面参与，但**不进入 history encoder 内部**。<br>**(5) 参数**：4B Gemma3 VLM（含 400M SigLIP）+ 860M action expert。 |
| **MemoryVLA: Perceptual-Cognitive Memory**<br>*Shi et al., 2025.08 (arXiv:2508.19236)* | 受认知科学启发的 **Cognition–Memory–Action** 框架。当前观测进 VLM 形成"工作记忆"，外置一个**Perceptual-Cognitive Memory Bank** 同时保留低层视觉细节与高层语义。 | 操控本质非马尔可夫，主流 VLA 默认 Markov 假设，长时程任务上严重受限；单层记忆既要记细节又要记语义，难以兼顾。 | 工作记忆从 Bank 中**按需检索**决策相关条目，与当前词元自适应融合后送入动作头；写入时**合并冗余条目**控制 Bank 规模。低层与高层条目分开存储，对应"细节—语义"两个尺度。Bank 整体端到端训练，无需外部 LLM。 |
| **ReMem-VLA: Memory via Dual-Level Recurrent Queries**<br>*2026.03 (arXiv:2603.12942)* | 在模型**内部**用两组循环查询承载短时与长时记忆：**帧级**循环查询逐帧传递（短期），**块级**循环查询跨时间分块传递（长期）。 | 已有两条路径都不理想：外置记忆库容易被分心物误导；扩展滑窗的固定窗口仍限制长期保留。需要"模型内部学到记忆"且不增推理开销。 | 双层循环 query 端到端训练，**无额外推理代价**；辅助引入 **Past Observation Prediction (POP)**：解码时同时预测过去观测，强迫记忆查询保留可追溯的历史信号。 |

### 1.1 内部四条路线的核心差异

| 维度 | MEM | π0.7 | MemoryVLA | ReMem-VLA |
|---|---|---|---|---|
| 短时尺度承载 | ViT 内时空注意力 + 丢弃过去 patch | **直接沿用 MEM** | VLM 输出的工作记忆词元 | 帧级循环 query |
| 短时窗口 | 最长 54s / 18 帧 | **6 秒 / 6 帧** | 中期 | 短期 |
| 长时尺度承载 | **外置 LLM 摘要的自然语言串** | 多模态 prompt（subgoal img + metadata） | 外置可训练记忆库 | 块级循环 query |
| 历史 token 进 VLM 吗 | 是 | **是** | 是（融合后） | 内部循环 |
| 历史是否任务条件化 | ✗ | **✗（任务无关）** | ✗ | ✗ |
| 时间尺度 | 秒级 + 分钟级 | 6s + prompt 多模态 | 中长期 | 短期 + 长期 |

> **共同启示**：单一压缩机制无法同时胜任秒级细节与分钟级语义，必须显式分尺度。**分歧点**在于长时记忆要不要符号化（自然语言 vs 隐向量）——这是一个直接决定可解释性、可审计性和工程复杂度的设计选择。

### 与 MoSA-VLA 的关系与可探索方向

MoSA-VLA 当前只覆盖**秒级短时记忆**这一尺度（16 帧 / 1.6s）。在多尺度视角下，新工作可考虑：

1. **保留 MoSA-VLA 的短时优势**：MPEG MV / DCT 结构化动态先验 + 摘要槽汇聚，是**比 MEM 的 ViT 内 patch 丢弃更轻量、且不依赖大规模视频—语言预训练**的短时压缩路径。
2. **补一条长时尺度**：可借鉴 MEM 的语言摘要路径（可解释、可审计），或 MemoryVLA / ReMem-VLA 的隐向量路径（无外部 LLM、纯端到端）。
3. **关键决策点**：长时记忆**是否符号化**——若要支持失败重试与人类干预，语言串路径更友好；若以纯部署效率为先，循环 query 路径更轻量。

---

## 1.2 外置记忆库 / 检索式记忆

> 共同假设：与其在网络内部"压缩"历史，不如**显式地把历史存到一个外部容器里**（keyframe / memory bank / soft prompt 库），在每步推理时从中**按需检索**少量相关条目。优势是可扩展到分钟级以上、与策略网络解耦；代价是引入了"检索器"这一新部件及其训练问题。

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **MemER: Scaling Up Memory for Robot Control via Experience Retrieval**<br>*Sridhar, Pan, Sharma, Finn — Stanford, 2025.10 (arXiv:2510.20328)* | **分层 VLA + keyframe 检索**：高层策略（Qwen2.5-VL-7B）从流式观测中**挑选关键历史帧**，与近期帧拼接后生成文本指令；低层策略（π₀.₅）执行。把"分钟级历史"压缩为"几个 keyframe + 文本指令"。 | 朴素地把所有历史塞进上下文，既计算昂贵又在分布偏移下脆弱；但下采样又会丢任务关键信息。需要**有选择性**的历史保留。 | **(1) 分层结构**：高层 VLM 负责检索+规划，低层 VLA 负责动作；**关键帧不进入低层**。<br>**(2) Online consolidation**：把每个时间步候选 keyframe 在线整合成紧凑稳定的 episodic memory（避免无脑全量保留）。<br>**(3) 训练数据**：用人类示范+最小语言标注；与现有 VLA backbone 兼容。<br>**(4) 评估**：三个需要"分钟级记忆"的真实长时程任务上显著优于 baseline。 |
| **MAP-VLA: Memory-Augmented Prompting for VLA**<br>*Li et al., 2025.11 (arXiv:2511.09516)* | 把示范数据中的子阶段信息蒸馏成一组**可学习软提示**，作为"记忆库"挂在冻结的 VLA 模型外侧，推理时按轨迹相似度检索相关 prompt 注入。 | 已有 VLA 多为冻结大模型，重训成本高；但长时程任务又需要任务相关的历史先验。如何在**不动模型权重**的前提下注入记忆？ | **(1) 即插即用**：底层 VLA 完全冻结，只训提示库；**与 MoSA / HiF / MemoryVLA 等"动权重"方案正交**。<br>**(2) 记忆形式**：演示衍生的**软提示（learnable soft prompts）**，每条对应任务的一个阶段。<br>**(3) 检索机制**：基于当前轨迹与历史轨迹相似度匹配。<br>**(4) 增益**：模拟 +7.0%、真机 +25.0%（长时程任务）。 |
| **EchoVLA: Synergistic Declarative Memory for Mobile Manipulation**<br>*2025.11 (arXiv:2511.18112)* | **双重声明性记忆**：**Scene Memory**（空间—语义地图集合）+ **Episodic Memory**（多模态上下文特征的任务级体验）。两条记忆独立读写，再通过**粗—细粒度注意力融合**指导扩散策略。 | 主流 VLA 缺乏"环境层"的持久表示；只在向量层做记忆很难刻画"我现在在哪、做过什么"这种地图—事件混合信息。 | **(1) Scene Memory**：以空间—语义图（spatial-semantic map）形式存储环境结构；典型用于移动操控的导航—操作切换。<br>**(2) Episodic Memory**：存储任务级多模态上下文（视觉 + 语言 + 状态）。<br>**(3) 读出方式**：两条记忆均按当前观察 + 任务历史 + 指令独立检索，融合后输出 cross-attention 条件给底盘+臂的联合扩散策略。<br>**(4) MoMani 基准**：作者提出的 MLLM-guided 移动操作基准，含真机轨迹。 |
| **ExpReS-VLA: Experience Replay and Retrieval Specialization**<br>*2025.11 (arXiv:2511.06202)* | 首次把**检索机制引入 VLA 微调**：在新任务上微调时，**从历史经验库中检索相似轨迹**作为辅助上下文，加快适应速度。 | 通用 VLA 在新任务上的少样本适应仍然慢；纯参数更新没充分利用已有经验。 | LIBERO 平均成功率 88.7%；纯 RAG 检索贡献 +6.6pp（单项最大增益）。属于**训练范式层面**的记忆使用——记忆不是给推理时用，而是给微调时用。 |

### 1.2 节核心观察
- **"记忆"在这里从"在线状态"变成"离线检索器"**：MemER 的 keyframe、MAP-VLA 的 prompt 库、EchoVLA 的 map 都是**离线/半离线**容器，与 MemoryVLA 的端到端可训练 bank 形成对比。
- **MemER 与 MAP-VLA 互为镜像**：前者是"检索原始观察 keyframe"（信息密度高、检索器要强），后者是"检索蒸馏的 prompt"（信息密度低、检索成本低、解释性差）。
- **EchoVLA 的 scene memory 是新维度**：把"环境地图"作为一个独立的记忆条件——对**移动操作**特别有意义；纯桌面操作（LIBERO / CALVIN）可能用不上。

---

## 1.3 隐式历史编码 / Token 级压缩

> 共同假设：历史不需要"存"在外面，只要在**网络内部**把多帧观测压缩成少量 token 即可。这条路与多尺度记忆（§1.1）的差异在于：多尺度强调"不同时间尺度用不同机制"，而这里更关心"如何用最少的 token 表达过去几秒到几十秒"。

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **HAMLET: History-Aware Policy**<br>*Anonymous (ICLR 2026 / arXiv:2510.00695)* | 用**moment tokens**在每个时间步紧凑编码感知信息，先通过**时序对比学习预训练**moment tokens，再用轻量记忆模块整合历史 token 做动作预测。**把现成 VLA 一键升级为 history-aware policy**。 | 主流 VLA 是单帧消费的，朴素地塞历史会破坏预训练对齐、训不动。如何在**最小改动**预训练 VLA 的前提下注入历史？ | **(1) Moment tokens**：每步生成一组小尺寸"瞬时 token"，靠**时序对比学习**初始化以捕获时间结构。<br>**(2) 记忆模块**：轻量，跨时间整合 moment tokens 后注入动作头。<br>**(3) 训练范式**：**作为现有 VLA 的适配层**——保持原 backbone 几乎不变。<br>**(4) 关键数字**：在历史依赖任务上比朴素微调的 VLA **+47.2%**；GR00T N1.5 上 76.4%、RoboCasa 100-demo 64.1→66.4、LIBERO 95.6→97.7。 |
| **SD-VLA: Static-Dynamic Decomposition for Efficient Long-Horizon VLA**<br>*Qiu, Huang, Ying, 2026.02 (arXiv:2602.03983)* | 观察到 VLA 历史里"**背景等 static 信息跨帧重复**"，提出**只保留 static token 的单一副本** + **轻量 recache gate**按需更新 KV cache。**实现 2.26× 推理加速 + 长时依赖基准 +39.8%**。 | 朴素地把多帧历史拼上下文 → context 长度 × 帧数，计算成本爆炸；但其实大部分 token 是冗余的。 | **(1) Token 分类**：把每帧的视觉 token 自动判别为 static / dynamic 两类。<br>**(2) 单副本保留**：static token 跨帧只保留一份，dynamic token 逐帧更新——本质是**显式的时间冗余消除**。<br>**(3) KV cache 复用**：static token 的 KV 不重算；recache gate 决定何时刷新（轻量门控）。<br>**(4) 关键 ablation**：相比 baseline，长时依赖基准 +39.8%、SimplerEnv +3.9%、推理 2.26× 加速。 |

### 1.3 节核心观察
- **HAMLET 与 SD-VLA 都是"VLA backbone 不动 / 微动"的适配层**：前者加 moment tokens + memory module，后者加 static-dynamic decomposition + recache gate。**这条路线对工程友好、容易复现**。
- **SD-VLA 的 static / dynamic 二分与 codec 的 I-frame / P-frame 二分在哲学上是同一思想**（§2.3 的 CoPE-VideoLM 也是 I/P 二分），但 SD-VLA 在 **token 层学习**判别，而 codec 路线在**像素层用编解码器**判别。两者可叠加。
- **HAMLET 的 +47.2% 是个强信号**：说明在"历史依赖任务"这个口径下，朴素 VLA 的天花板远没到，简单加点历史就有大幅提升。

---

## 1.4 系统级记忆 / 反馈式记忆

> 这条线不强调"压缩"，而强调**记忆作为高层规划 / 推理的状态**——更接近"agent-style memory"。

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **MindExplore (Towards Long-Horizon VLA System: Reasoning, Acting, Memory)**<br>*Li et al., ICCV 2025 (Xidian + AgileX)* | 三层系统：**Reasoning**（任务专属 CoT 生成元动作信号）+ **Acting**（Mixture of Policy Experts + MMDP 多模态扩散）+ **Memory**（reasoning–acting 双层之间的反馈循环）。配套 SandGo-1k / SandThink-21k 数据集。 | 端到端 VLA 在分钟级任务上规划失败时**无法回退、无法重规划**；需要一个能"出错—重想—再试"的系统。 | **(1) Reasoning 层**：CoT 规划子任务序列。<br>**(2) Acting 层**：MoE 技能专家 + MMDP 空间多模态扩散，30 FPS 高频执行。<br>**(3) Memory 不是向量库**，而是**两层之间的反馈信号流**——更接近 ReAct 风格的中间状态共享。<br>**(4) 数据集**：作者贡献的 21K CoT 标注。 |
| **EvoVLA: Self-Evolving VLA**<br>*AIGeeksGroup, 2025.11 (arXiv:2511.16166)* | 三件套自监督框架：**Stage-Aligned Reward**（Gemini 生成 hard negative + 三元对比）+ **Pose-Based Object Exploration**（基于物体—夹爪相对位姿的好奇心）+ **Long-Horizon Memory**（**selective context retention + gated fusion**）。 | 长时程 rollout 中"内在奖励 shaping"容易漂移；需要一个稳定的长期上下文。 | **(1) Long-Horizon Memory**：从历史 rollout 中**选择性保留上下文片段**，通过**门控融合**注入到内在奖励 shaping，避免分布漂移。<br>**(2) Discoverse-L 基准**：作者提出的三阶段长时程基准；仿真 69.2%、真机 54.6%。<br>**(3) 与 §1.1–§1.3 的差异**：这里的记忆**主要服务 RL 内在奖励**而不是动作策略本身——属于"训练过程的记忆"。 |
| **Long-VLA**<br>*CoRL 2025 (arXiv:2508.19958)* | 端到端长时程 VLA，**Phase-Aware Input Masking**：把每个子任务分为"移动 / 交互"两个阶段，自适应 mask 输入感官特征。配套 L-CALVIN 基准。 | 长时程任务里"移动"与"交互"两阶段的感知需求不同，固定输入混了。 | 严格说不是"记忆压缩"，更像**输入侧的阶段先验**——但因为论文标题主打 long-horizon，常被一并讨论。**对本工作启发有限**。 |

---

## 1.5 全部 11 篇工作的横向对比

| 工作 | 类别 | 记忆形式 | 短时尺度 | 长时尺度 | 任务条件化? | backbone 改动 | 时间 |
|---|---|---|---|---|---|---|---|
| **MEM** | 多尺度 | 视频编码 + LLM 自然语言摘要 | 秒级 (ViT 内时空注意力) | 分钟级 (LLM 摘要字符串) | ✗ | 中（ViT 内插时间注意力） | 2026.03 |
| **π0.7** | 多尺度 | 同 MEM + multi-modal prompt | 6 帧/6s | prompt 多模态 (subgoal img + metadata) | ✗ | 中（沿用 MEM 编码器） | 2026.04 |
| **MemoryVLA** | 多尺度 | 工作记忆 + Perceptual-Cognitive Memory Bank | 当前 VLM tokens | bank 中合并冗余的低/高层条目 | ✗ | 大（新 bank 模块） | 2025.08 |
| **ReMem-VLA** | 多尺度 | 双层循环 query（帧级 + 块级） | 帧级循环 | 块级循环 | ✗ | 中（双层 RNN-like queries） | 2026.03 |
| **MemER** | 外置/检索 | Keyframe + 高层文本指令 | 近期帧 | 检索出的 keyframes（分钟级） | ✓（高层 VLM 检索） | 大（分层架构） | 2025.10 |
| **MAP-VLA** | 外置/检索 | 可学习软提示库 | — | 演示衍生 prompt 库 | ✓（按轨迹相似度检索） | **无（冻结 VLA）** | 2025.11 |
| **EchoVLA** | 外置/检索 | Scene map + episodic 多模态特征 | — | 空间—语义地图 + 任务级体验 | ✓（按指令检索） | 大（双记忆模块） | 2025.11 |
| **ExpReS-VLA** | 外置/检索 | 经验回放库（训练期） | — | 历史轨迹 | ✓（按任务检索） | 小（训练范式） | 2025.11 |
| **HAMLET** | 隐式编码 | Moment tokens + 轻量 memory module | 时序对比学习的 moment tokens | 同上累积 | ✗ | **极小（适配层）** | 2025.10 (ICLR'26) |
| **SD-VLA** | 隐式编码 | Static (单副本) + Dynamic (逐帧) tokens | — | 长上下文（含 KV cache 复用） | ✗ | 小（recache gate） | 2026.02 |
| **MindExplore** | 系统级 | Reasoning–Acting 反馈信号流 | Acting 层 30 FPS | Reasoning 层 CoT 状态 | ✓（CoT 显式任务相关） | 大（双层系统） | ICCV 2025 |
| **EvoVLA** | 系统级 | Selective context retention + gated fusion | — | RL rollout 历史片段 | ✓（用于奖励 shaping） | 中 | 2025.11 |
| **Long-VLA** | （阶段先验） | 阶段相关输入 mask | — | — | ✓（按阶段） | 中 | CoRL 2025 |

### 1.5 节核心观察 — **设计空间的四个轴**

1. **记忆容器**：内部隐状态 / 外置可训练 bank / 外置离线检索器（keyframe / prompt / map）
2. **时间尺度**：单尺度（HAMLET / SD-VLA / MAP-VLA） vs 多尺度（MEM / π0.7 / MemoryVLA / ReMem-VLA / EchoVLA）
3. **任务条件化**：检索式天然条件化（MemER / MAP-VLA / EchoVLA / ExpReS-VLA / MindExplore / EvoVLA）；隐式编码式与多尺度式**绝大多数任务无关**（**这正是 §2 codec 路线的同一盲点**——见后文）
4. **backbone 改动量**：零（MAP-VLA 冻结）→ 适配层（HAMLET）→ 模块化（MEM、MemoryVLA、MemER）→ 重构（EchoVLA、MindExplore）

### 1.5 节对 MoSA-VLA / 新工作的启示更新

- **任务条件化已经在"检索式记忆"那一支落地**（MemER / MAP-VLA / EchoVLA），但**在 codec 路线、在"短时密集 token"那一支还是空白**——这正是新工作的真正空白格子。
- **HAMLET 的 moment tokens + 时序对比学习预训练**是个值得借鉴的训练范式：在不大改 backbone 的前提下，先用自监督任务给"历史 token"一个良好初始化。MoSA-VLA 可以考虑把"运动先验"作为一个对比目标（同一轨迹的运动 vs 不同轨迹的运动）。
- **SD-VLA 的 static/dynamic token 二分**与 codec 的 I/P-frame 二分本质同源——这给"为什么用 codec"这一论证又加了一层证据：连**不使用 codec 的工作都自发地走到了相同二分**。
- **MAP-VLA 的"冻结 backbone + 加 prompt 库"**为新工作提供了一个零参数对比基线：如果新工作要主打"任务条件化"，应该与 MAP-VLA 做直接对比，证明 prompt 检索 ≠ 真正的任务条件化压缩。

---

## 2. 视频编码先验（Codec-Based）⭐ 与本工作最相关

> **重点关注方向**。Codec-based 思路在 VLA 里目前由 HiF-VLA 占据；但更有意思的是 **OneVision-Encoder**（同 LLaVA-OneVision-1.5 团队）把 codec 思想正式引入主流 VLM 编码器。两者代表了两条不同的路线：**actual codec output**（HiF-VLA / MoSA-VLA） vs **codec-inspired learned sparsity**（OneVision-Encoder）。

### 2.1 直接用 codec 输出（Actual Codec Output）

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for VLA**<br>*Lin et al., Westlake Univ. 等, CVPR 2026 (arXiv:2512.09928)* | 基于运动表示的双向时间推理 VLA：（a）**Hindsight**：MPEG-4 16×16 macroblock MV 作为历史紧凑先验；（b）**Foresight**：模型自回归预测**未来运动向量**；（c）用历史 MV 通过 AdaLN 调制 foresight + 动作流。LIBERO-Long **96.4%**，CALVIN ABC-D 平均链长 **4.35**。 | 标准 VLA 只看当前帧（temporal myopia）；多帧堆叠又算力暴增。需要一种紧凑且能预测未来的运动表示。 | **(1) Hindsight Encoder**：MV 张量 \(h \times \tfrac{H}{16} \times \tfrac{W}{16} \times 2\)，先用 3D 卷积分块再过 4 层 ViT 得到 1024 维 hindsight tokens。**作者明确不把 MV 注入 VLM，理由是"会破坏 VLM 视觉—语言对齐"，改为通过 AdaLN 调制 decoder。**<br>**(2) Foresight**：VLM 末端加 K_f 个 foresight query + K_a 个 action query，并联预测 motion + action。<br>**(3) 训练**：\(\mathcal{L} = \mathcal{L}_A + 0.01\,\mathcal{L}_{MV}\)，OpenVLA 初始化，LIBERO 跑 150K 步。<br>**(4) 关键限制（自承）**：**hindsight 编码任务无关**——只有 foresight 路径用语言；历史 MV 不被语言条件化。 |
| **FAST: Frequency-space Action Sequence Tokenization for VLA**<br>*2025.01 (arXiv:2501.09747)* | 用 DCT 对**动作信号**做频域压缩 → 量化 → BPE 编码，得到紧凑的 action token 序列，让自回归 VLA 能处理灵巧/高频任务。 | 标准动作离散化在高频任务上失败；动作序列时域冗余高。 | DCT 应用于**输出端动作**，不是视觉历史输入。低频系数保留动作主体，高频反映尖锐变化。**与本工作的 DCT 用法（视觉频域差分）问题域完全不同**，只是共享了"DCT"这个工具，不冲突。 |

### 2.2 ⭐ Codec-启发的 patch 选择（OneVision-Encoder 路线）

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence**<br>*EvolvingLMMs-Lab（LLaVA-OneVision-1.5 同团队）, arXiv:2602.08683, 2026* | 把视频 codec（HEVC）的"能量集中、信号稀疏"原则**正式提升为多模态架构的基础原则**。**真的用 HEVC 编码视频**取出 MV/Residual energy，但 codec 的角色不是"输入特征"——而是**决定哪些 RGB patch 留下来**（patch selection）。最终 ViT 处理的是**经 codec 筛选过的稀疏 RGB patch**，不是 MV 本身。Diving-48 上比 SigLIP2 +17.1%、DINOv3 +8.1%。 | "视觉信号高度冗余，但有判别力的信息是稀疏的"——传统 ViT 对所有 patch 一视同仁，浪费在静态背景上。 | **(1) HEVC GOP**：每 32 帧设 I-frame；I-frame 保留所有 patch（完整空间锚），P-frame 用 MV+Residual energy 计算 patch saliency，只保留 3.1–25% 的 patch。<br>**(2) Codec Patchification**：颠倒了主流 video LLM 的"few frames × all patches"惯例，改为 **"many frames × only important patches"**。<br>**(3) Zigzag 抽取顺序**：直接来自 HEVC/JPEG 的系数扫描顺序。<br>**(4) 3D RoPE**：跨 T/H/W 三维位置编码。<br>**(5) 训练**：cluster discrimination over 1M+ semantic concepts，2M concept bank（vs CLIP 32K-64K batch negatives）。 |

### 2.3 ⭐ Codec primitives 作为 P-frame token 压缩器（CoPE-VideoLM 路线）

| 标题 | 总结 | 研究问题 | 解决方案 |
|---|---|---|---|
| **CoPE-VideoLM: Leveraging Codec Primitives For Efficient Video Language Modeling**<br>*Sarkar, Pautrat, Miksik, Pollefeys, Armeni, Rad, Dusmanu — Stanford / Microsoft Spatial AI / ETH Zurich, arXiv:2602.13191, 2026.02* | **直接照搬 codec 的 I/P-frame 二分结构作为 token 分配规则**：I-frame 全编码（210 tokens via SigLIP），P-frame 仅用 MV+残差压成 **8 个 Δ-tokens**。LLaVA-Video-7B 上 PerceptionTest 70.5% (token 仅 19.5%, +6.9pp)，TTFT 减少 86.2%（0.33s vs 2.39s），1M token 预算下可处理 **88 小时 1FPS 视频**。 | Video LLM 处理长视频时 token 暴增；但视频 codec 已经天然按"重要程度"分了 I-frame（关键）和 P-frame（变化）——为什么不直接照搬这个结构来分配 token 预算？ | **(1) 输入**：H.264/HEVC GOP 结构；I-frame 是完整 RGB（过 frozen SigLIP 得 M=210 tokens），P-frame 只用 MV \(\tau(t) \in \mathbb{Z}^{H\times W\times 2}\) 和残差 \(\delta(t)\)。<br>**(2) Δ-Encoder**：双分支处理 P-frame——MV 分支（min-max 归一化 → 16×16 patch → MLP → transformer with \(K_\tau=4\) queries），残差分支（截断 ResNet-18 → transformer with \(K_\delta=4\) queries），合并得到 **8 Δ-tokens / P-frame**。<br>**(3) 序列**：I-frame tokens 与 P-frame Δ-tokens 时序交错拼接，直接送 LLM，无需投影层（d=1152）。<br>**(4) 两阶段训练**：① 预训练 Δ-Encoder：用 patch-wise MSE 让 Δ-tokens 重建对应 SigLIP RGB tokens（关键！没这步性能从 67.33% 掉到 63.45%）；② 接入 LLaVA-Video-7B 端到端 next-token 微调。<br>**(5) 关键 ablation**：codec-aware 训练 vs 同等 token 数的 RGB → **+5.2pp**——硬证据证明 codec primitives 比单纯减帧/减 patch 更有信息量。 |

### 2.4 三条 codec 路线对比（Actual feature / Patch selection / Token compression）

四种 codec 工作的机制可以用一句话区分清楚：

| 工作 | codec 在做什么 | 进 ViT/LLM 的实际内容 |
|---|---|---|
| **HiF-VLA / MoSA-VLA** | codec 输出**作为输入特征**（替代或补充 RGB） | MV 张量本身 |
| **OneVision-Encoder** | codec 输出**作为 patch 门控**（决定哪些 RGB patch 留下） | 经 codec 筛过的稀疏 RGB patch |
| **CoPE-VideoLM** | codec 的 I/P-frame **结构作为 token 分配规则** | I-frame 全 RGB tokens + P-frame 的 Δ-tokens（从 MV+残差压来） |

详细对比：

| 维度 | HiF-VLA / MoSA-VLA | OneVision-Encoder | CoPE-VideoLM |
|---|---|---|---|
| 是否用真实 codec 输出 | ✓ MV | ✓ MV + Residual | ✓ MV + Residual + I/P 结构 |
| codec 的角色 | 输入特征 | patch 门控 | token 预算分配 |
| RGB 是否保留 | 否（只看 MV） | 是（保留稀疏子集） | 部分（仅 I-frame 全 RGB） |
| I/P 结构是否显式利用 | 否 | 部分（每 32 帧 I-frame） | **全显式**（不同 token 数） |
| 是否端到端可微 | 否（前端 ffmpeg 不可微） | 否（codec 端不可微，但 ViT 可微） | 否（codec 端不可微，但 Δ-Encoder 可微） |
| 主要场景 | 机器人策略 | 通用长视频理解 | 通用长视频理解（极致效率） |
| 任务条件化 | ✗ | ✗ | ✗ |

> **三条线的核心共识**："视觉信号稀疏 + codec 已经把稀疏性显式编码出来"——codec 是 **免费的、人类工程精炼了几十年的 video tokenizer**。
> **三条线的共同盲点**：**全部任务无关**——不管语言指令是什么，都用同一套 codec 原则压缩历史。这就是新工作的空白格子。

### 2.5 对新工作的具体启示（必读）

CoPE-VideoLM 的存在加强了你新工作的合法性，也提供了几个直接可借鉴的工程点：

1. **两阶段训练范式可借鉴**：CoPE-VideoLM 先预训练 Δ-Encoder 让 codec tokens 对齐 SigLIP RGB token 空间（MSE 重建），再做下游 SFT。**没这步性能掉 4pp**——说明把 codec 输出"翻译"到 RGB token 空间是必要的预处理。你的新工作如果想把语言条件化做强，可以在第一阶段就把 codec 输入对齐到 VLM 理解的语义空间，再在第二阶段引入语言条件。
2. **"codec-aware vs 同等 token RGB" 是必做的硬 ablation**：CoPE-VideoLM 用这一点拿到 +5.2pp 的硬证据，证明 codec primitives 比单纯减帧/减 patch 更有信息量。HiF-VLA 没做这个；MoSA-VLA 也没做；如果你新工作要说服 reviewer，**这一项 ablation 必须做**，且最好直接复用 CoPE-VideoLM 的设置以保持可比性。
3. **它再次确认了"任务无关"是普遍盲点**：三条 codec 路线都是任务无关的；CoPE-VideoLM 在 limitations 里提的是"操作直接在量化 DCT 系数上"等效率方向，**完全没碰任务条件化**——这进一步印证你新工作的空白判断。
4. **CoPE-VideoLM 是 video LLM 工作，HiF-VLA 是 VLA 工作，两者独立但同时走到 codec 路线**——这给你"两路汇流"的开篇 framing 又加了一笔证据。可以改成"**三路汇流**"：actual feature input（HiF-VLA / MoSA-VLA）、patch selection（OneVision-Encoder）、token allocation（CoPE-VideoLM）——VLA + video LLM 两个社区，三种独立技术路线，全部任务无关。本工作填这个共同空白。

### 2.4 主流 video LLM 的 token 压缩全景（仅简要，主要做背景）

为完整性记录主流 video LLM 在帧/词元压缩上的做法。**这些方法绝大多数与 codec 无关**，但用于说明你工作的差异化（"为什么我们用 codec？"——因为可解释、零训练、机器人侧实时性需求）。

| 模型 / 方法 | 压缩思路 | 是否涉及 codec |
|---|---|---|
| **LLaVA-OneVision-1.5** ([arXiv:2509.23661](https://arxiv.org/abs/2509.23661)) | bilinear pool 到 **196 tokens / frame**；帧间无显式时序建模 | ✗ |
| **OneVision-Encoder** ([GitHub](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder)) | Codec-aligned sparsity（zigzag + 3.1–25% 区域选择） | ⭐⭐⭐ 显式 codec 启发 |
| **Qwen2.5-VL** ([HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)) | 动态 FPS 采样 + temporal mRoPE（含绝对时间对齐） | △ 动态采样有 codec rate-control 影子 |
| **InternVL-X** ([arXiv:2503.21307](https://arxiv.org/abs/2503.21307)) | PVTC（adjacent embedding 融合）+ LVTC（层级压缩 + upsample） | ✗ |
| **VideoLLaMA 2** | 3D 卷积时空下采样（实验证明优于 2D） | ✗ |
| **STORM** ([arXiv:2503.04130](https://arxiv.org/abs/2503.04130)) | 在 image encoder 与 LLM 间插一个 Mamba 时序编码器 | ✗ |
| **LLaVA-Mini** ([arXiv:2501.03895](https://arxiv.org/abs/2501.03895)) | **极致：每帧 1 个 vision token**，靠 LLM 端学好 | ✗ |
| **LLaVA-Scissor** ([arXiv:2506.21862](https://arxiv.org/abs/2506.21862)) | 语义联通分量做 token 聚类 | ✗ |
| **Less Is More** ([arXiv:2512.06866](https://arxiv.org/abs/2512.06866)) | LLM-guided keyframe prior 动态选关键帧 | ✗ |

> **核心观察**：在 video LLM 一线，**除了 OneVision-Encoder，没有任何主流模型用 codec 思想**——绝大多数走"学一个池化器"或"选关键帧"的路。这说明：(a) codec 思路在 video LLM 社区刚刚被合法化（OneVision-Encoder 是新声音）；(b) 你和 HiF-VLA 的"actual codec output"路线在更窄的 VLA 赛道上是合理的差异化点；(c) 主流 video LLM 的所有压缩方法**全部是任务无关**——这进一步印证你"任务条件化压缩"的空白判断。

### 与 MoSA-VLA / 新工作的关系（关键定位）

**坏消息**：
- 你之前以为"用 MPEG MV"是没人占的格子——HiF-VLA 已经占了，且发在 CVPR 2026。
- 你的"结构化运动先验"作为输入这个角度，**已经不是论文级差异化**。
- LIBERO-Long 数字非常接近：HiF-VLA 96.4% vs MoSA-VLA 96.2%——简单复刻"我也用 MV"无法压过 HiF-VLA。

**好消息**：
- HiF-VLA 的 hindsight 编码**自承是任务无关的**（用 3D conv + ViT 直接卷过去，不接语言）。
- 这正是你新工作的核心命题——**让历史压缩任务条件化**。HiF-VLA 没做这件事，等于亲手画了你新工作的可行性边界。
- 注入机制不同：HiF-VLA 用 **AdaLN 调制 decoder**；你可以用**语言派生的 cross-attention 槽**，是另一个机制方向。
- 你不需要也不应该再卷 foresight——HiF-VLA 已经在那条线上拿了分；你应该全力卷 hindsight 的"任务条件化"这一条线。

### Codec-based 工作的全景定位

| 类别 | 代表工作 | 与本工作的关系 |
|---|---|---|
| **VLA + actual codec MV 输入** | **HiF-VLA**（CVPR 2026） | 唯一直接撞线（VLA 内） |
| **Video VLM + codec-inspired patch 选择** | **OneVision-Encoder**（LLaVA-OV-1.5 团队, 2026） | codec 思想在主流 VLM 编码器侧的代表作 |
| **Video VLM + codec primitives 作 token 压缩** | **CoPE-VideoLM**（Stanford/Microsoft/ETH, 2026.02） | I/P-frame 二分结构作 token 分配规则，长视频效率王 |
| **VLA + DCT，但作用在动作端** | [FAST](https://arxiv.org/abs/2501.09747) | 共享 DCT 工具，问题域不同 |
| **压缩域动作识别**（pre-VLA 源头） | CoViAR (CVPR 2018) 及其后续：[Refined MV](https://arxiv.org/abs/1910.02533)、Lightweight Action Recognition in Compressed Videos | 用 MV + I-frame + residual 做动作**分类**，是 codec 在视觉理解领域的源头 |
| **VLA + 学习到的运动表示**（不是 codec） | Motion Tracks、Amplify、CoMo、Vid2Robot、mimic-video、EgoVLA | 用 keypoint / motion latent / 视频条件，非 codec |
| **压缩潜在空间** | VAE 压轨迹、LightDP | "压缩"指模型/潜空间嵌入，与 codec 无关 |

**对新工作叙事的三点提示**：
1. **MV 输入本身已不是创新点**——CoViAR 起步 8 年了，HiF-VLA 已把它带进 VLA。新颖性必须出在 MV 之外的环节（即语言条件化压缩）。
2. **codec 思维已在三条独立路线被同时合法化**——actual feature input（HiF-VLA）、patch selection（OneVision-Encoder）、token allocation（CoPE-VideoLM），跨 VLA 和 video LLM 两个社区。论文叙事可以"**三路汇流**"而不是孤例。
3. **三条路线的共同盲点都是任务无关**——这才是真正的空白格子。新工作的核心创新空间在此。

### 新工作 vs HiF-VLA 的差异化定位

| 维度 | HiF-VLA | 新工作（建议） |
|---|---|---|
| 历史 MV 编码 | 任务无关（3D conv + ViT） | **任务条件化**（语言派生槽 + cross-attention） |
| 注入 VLA 的位置 | decoder via AdaLN | 动作查询 cross-attention（沿用 MoSA-VLA） |
| 是否预测未来 | **是**（foresight MV） | **否**（专注 hindsight 任务条件化） |
| 训练步数 | 150K | 30K（沿用 MoSA-VLA 协议） |
| 卖点 | 双向时间推理 + 效率 | **任务—运动模式对齐 + 可解释性** |

**关键消息**：HiF-VLA 的存在反而**强化**了你新工作的合法性——它证明 codec MV 是可行方向，但留下了"任务条件化"这个明确空白让你去填。

---

## 3. 其他相关方向（简要）

仅作背景定位，不作为新工作的主线。

- **推理时词元裁剪（Training-free Token Pruning）**：在不重训练的前提下减少视觉词元数量。代表：[TTF-VLA](https://arxiv.org/abs/2508.19257)（双维启发式时序融合 + 关键帧锚定）、[EfficientVLA](https://arxiv.org/abs/2506.10100)（语言层 + 视觉词元 + 扩散步三件套）、[VLA-Pruner](https://arxiv.org/abs/2511.16449)（语义层 + 动作层双重要性）。和多尺度记忆**正交**，可叠加使用。
- **训练时特征压缩（多帧 → 单一向量）**：[CronusVLA](https://arxiv.org/abs/2506.19816) 把单帧 VLA 经两阶段后训练扩展为多帧，用 feature chunking 聚合历史。属于"压缩短时历史到一个特征"，仍是单尺度。
- **状态空间建模（Recurrent Memory）**：[MTIL](https://arxiv.org/abs/2505.12410) 用 Mamba/SSM 把整段轨迹递归压成固定维度隐状态，推理对历史长度近似 O(1)；优点是简单，缺点是隐状态不可审计、长时尺度信息易被覆盖。

---

## 4. 引用

**多尺度记忆（§1.1）**
- [MEM — arXiv:2603.03596](https://arxiv.org/abs/2603.03596) ｜ [PDF](https://www.pi.website/download/Mem.pdf) ｜ [项目页](https://www.pi.website/research/memory)
- ⭐ **[π0.7 — arXiv:2604.15483](https://arxiv.org/abs/2604.15483)** ｜ [PDF](https://www.pi.website/download/pi07.pdf) — Physical Intelligence 2026.04，沿用 MEM 历史编码器
- [MemoryVLA — arXiv:2508.19236](https://arxiv.org/abs/2508.19236) ｜ [项目页](https://shihao1895.github.io/MemoryVLA/)
- [ReMem-VLA — arXiv:2603.12942](https://arxiv.org/abs/2603.12942)

**外置 / 检索式记忆（§1.2）**
- [MemER — arXiv:2510.20328](https://arxiv.org/abs/2510.20328) — Stanford (Chelsea Finn lab)，分层 keyframe 检索
- [MAP-VLA — arXiv:2511.09516](https://arxiv.org/abs/2511.09516) — 冻结 VLA + 可学习软提示库
- [EchoVLA — arXiv:2511.18112](https://arxiv.org/abs/2511.18112) — 场景记忆 + 情节记忆（移动操作）
- [ExpReS-VLA — arXiv:2511.06202](https://arxiv.org/abs/2511.06202) — 检索增强微调

**隐式历史编码 / Token 压缩（§1.3）**
- ⭐ **[HAMLET — arXiv:2510.00695](https://arxiv.org/abs/2510.00695)** — Moment tokens + 时序对比学习预训练（ICLR 2026）
- ⭐ **[SD-VLA — arXiv:2602.03983](https://arxiv.org/abs/2602.03983)** — Static / Dynamic token 二分 + KV cache 复用

**系统级记忆 / 长时程框架（§1.4）**
- [MindExplore (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Towards_Long-Horizon_Vision-Language-Action_System_Reasoning_Acting_and_Memory_ICCV_2025_paper.pdf) — Xidian + AgileX，三层系统
- [EvoVLA — arXiv:2511.16166](https://arxiv.org/abs/2511.16166) — RL 自监督 + Long-Horizon Memory
- [Long-VLA — arXiv:2508.19958](https://arxiv.org/abs/2508.19958) — CoRL 2025，phase-aware masking

**Codec-based（重点）**
- ⭐ **[HiF-VLA — arXiv:2512.09928](https://arxiv.org/abs/2512.09928)（CVPR 2026）｜ [项目页](https://hifvla.github.io/)** — actual codec MV 作输入，与本工作直接撞线
- ⭐ **[OneVision-Encoder — arXiv:2602.08683](https://arxiv.org/abs/2602.08683)** ｜ [GitHub](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder) — Codec-Aligned Sparsity（patch selection 路线）
- ⭐ **[CoPE-VideoLM — arXiv:2602.13191](https://arxiv.org/abs/2602.13191)** — Stanford/Microsoft/ETH，I/P-frame 作 token 分配规则
- [FAST — arXiv:2501.09747](https://arxiv.org/abs/2501.09747) — DCT 用于动作端
- CoViAR 系列：[CoViAR (CVPR'18)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)、[DMC-Net (CVPR'19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shou_DMC-Net_Generating_Discriminative_Motion_Cues_for_Fast_Compressed_Video_Action_CVPR_2019_paper.pdf)、[Refined MV (2019)](https://arxiv.org/abs/1910.02533)、[MM-ViT (WACV'22)](https://openaccess.thecvf.com/content/WACV2022/papers/Chen_MM-ViT_Multi-Modal_Video_Transformer_for_Compressed_Video_Action_Recognition_WACV_2022_paper.pdf)
- DCT 频域学习：[Learning in the Frequency Domain (CVPR'20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Learning_in_the_Frequency_Domain_CVPR_2020_paper.pdf)、[DCFormer (WACV'23)](https://openaccess.thecvf.com/content/WACV2023/papers/Li_Discrete_Cosin_TransFormer_Image_Modeling_From_Frequency_Domain_WACV_2023_paper.pdf)
- 学习式视频压缩：[DVC (CVPR'19)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.pdf)、[FVC (CVPR'21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_FVC_A_New_Framework_Towards_Deep_Video_Compression_in_Feature_Space_CVPR_2021_paper.pdf)

**Video LLM token 压缩（背景）**
- [LLaVA-OneVision-1.5 — arXiv:2509.23661](https://arxiv.org/abs/2509.23661)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [InternVL-X — arXiv:2503.21307](https://arxiv.org/abs/2503.21307)
- [STORM — arXiv:2503.04130](https://arxiv.org/abs/2503.04130)
- [LLaVA-Mini — arXiv:2501.03895](https://arxiv.org/abs/2501.03895)
- [LLaVA-Scissor — arXiv:2506.21862](https://arxiv.org/abs/2506.21862)
- [Less Is More — arXiv:2512.06866](https://arxiv.org/abs/2512.06866)

**其他相关方向**
- [TTF-VLA — arXiv:2508.19257](https://arxiv.org/abs/2508.19257)
- [EfficientVLA — arXiv:2506.10100](https://arxiv.org/abs/2506.10100)
- [VLA-Pruner — arXiv:2511.16449](https://arxiv.org/abs/2511.16449)
- [CronusVLA — arXiv:2506.19816](https://arxiv.org/abs/2506.19816)
- [MTIL — arXiv:2505.12410](https://arxiv.org/abs/2505.12410)
