# HTCS Fork — 开发指引

> 这份 README 用来快速带新协作者 / 新 Claude session 进入本 fork 的工作上下文。读完这份就足以开始动手写代码。
>
> **本 fork 用途**：基于 [starVLA](https://github.com/starVLA/starVLA) 实现 **HTCS (Hierarchical Task-Conditioned Saliency)** —— 一种基于 codec 的双 granularity 语言条件化历史压缩 VLA 方法。
>
> **代码主体仍是 starVLA**，HTCS 只是其上多注册的一个 framework，参考 starVLA `examples/LIBERO/` 的现有结构对应改造即可。

---

## 0. 把本 fork 与 starVLA 的关系搞清楚

```
   GitHub: starVLA/starVLA  ─────► (upstream remote, 只读 pull)
                    │
                    │ fork
                    ▼
   GitHub: <your-org>/starVLA-htcs ─────► (origin remote, 推 HTCS 代码)
                    │
                    │ clone
                    ▼
   本机工作目录 (本 repo)
```

- `origin` = 你的 fork —— **所有 HTCS 提交推到这里**
- `upstream` = starVLA 官方 —— **只 fetch，不 push**
- 工作分支 `htcs-dev`（基于 `upstream/starVLA_dev`，跟最新功能）
- 投稿/复现分支 `htcs-stable`（基于 `upstream/starVLA`，稳定）

> 第一次 setup 命令在 [§8 工作流参考](#8-工作流参考) 节，**首次 clone 之后必须先做**。

---

## 1. HTCS 是什么（一分钟概览）

VLA 历史压缩必须经过"任务相关"的滤波。本工作主张语言条件化在 codec 压缩流水线中应分成两层 granularity：

1. **Stage 1（参数级）**：语言派生标量权重 \(\alpha(\ell), \beta(\ell)\) 调制 codec saliency 公式
   \[
     s = \alpha(\ell)\cdot \|MV\| + \beta(\ell) \cdot |Residual|
   \]
   决定"看哪种运动"

2. **Stage 2（结构级）**：语言派生 K 个 query，对 Stage 1 输出做 cross-attention（含竞争性 softmax + Stage 1 saliency bias），决定"看哪些 patch"

输出 K=8 个摘要词元 H → cross-attention 注入 action queries → action head。**整个流水线仅依赖语言信号，不需要 proprioception**。

VLM backbone：Qwen3-VL-0.8B（starVLA 称作 Qwen3.5-0.8B）。
基准：LIBERO 4 套件联合训练。

完整故事 / 决策 / 风险 见 [设计文档](#3-文档导航)。

---

## 2. 上下文文档（必读 → 选读 顺序）

| 优先级 | 文档 | 用途 | 何时读 |
|---|---|---|---|
| ⭐⭐⭐ | **`新工作-HTCS-实现文档.md`** | 文件清单、张量形状、starVLA 集成方式、checklist、工程陷阱 | **开工前必读** |
| ⭐⭐⭐ | **`新工作-语言条件化历史压缩-工作文档.md`** | 故事、决策、claims、消融实验、风险 | 开工前必读 |
| ⭐⭐ | **`相关工作调研-VLA历史压缩.md`** | MEM / π0.7 / MemoryVLA / HiF-VLA / OneVision-Encoder / CoPE-VideoLM 等 | 写论文 / 答辩前读 |
| ⭐ | [`code/starVLA/docs/starVLA_guideline.md`](code/starVLA/docs/starVLA_guideline.md) | starVLA 库的 install / train / eval 流程 | 第一次配环境时 |
| ⭐ | [`code/starVLA/docs/integrate_your_dataset.md`](code/starVLA/docs/integrate_your_dataset.md) | LeRobot 数据格式与自定义数据集集成 | 需要改 dataloader 时 |
| ⭐ | [`code/starVLA/examples/LIBERO/README.md`](code/starVLA/examples/LIBERO/README.md) | LIBERO 训练 / 评测的完整 example | 调通 baseline 时 |

> **这三份 `.md` 文档当前在 fork 之外的工作目录** (`c:\work\memvla\`)。建议将它们复制 / 软链到本 fork 根目录下的 `docs/htcs/` 子目录，方便新 session 直接读取。

---

## 3. 文档导航

```
c:\work\memvla\
├── 新工作-HTCS-fork-README.md              ← 本文档（开发指引）
├── 新工作-HTCS-实现文档.md                 ← 实现规范（与 starVLA 对齐）
├── 新工作-语言条件化历史压缩-工作文档.md   ← 设计 / 决策 / 风险
├── 相关工作调研-VLA历史压缩.md             ← 文献调研
└── code/
    ├── starVLA/                            ← 上游 starVLA 副本（仅供查阅）
    └── starVLA-htcs/  (待 fork)            ← 本 fork 的工作目录
```

新 session 打开本 fork 时，请先按以下顺序读：
1. 本 README（5 分钟）
2. `新工作-HTCS-实现文档.md`（20 分钟）
3. `新工作-语言条件化历史压缩-工作文档.md` 的 §0–§3（10 分钟，看故事 + Claims + 模块清单即可，§5 决策细节按需查）

---

## 4. 文件 Map —— HTCS 在 starVLA 树内的位置

> 详细每个文件的内容、类签名、张量形状 → 见 `新工作-HTCS-实现文档.md` §1–§7。

### 🆕 新增（13 个文件）

| 路径 | 用途 |
|---|---|
| `starVLA/model/framework/VLM4A/HTCS.py` | 顶层 framework 类，注册 `HTCS` |
| `starVLA/model/modules/htcs/__init__.py` | export |
| `starVLA/model/modules/htcs/saliency_mlp.py` | α(ℓ), β(ℓ) MLP |
| `starVLA/model/modules/htcs/stage1_codec_selector.py` | M2 |
| `starVLA/model/modules/htcs/stage2_lang_compressor.py` | M3 |
| `starVLA/model/modules/htcs/slot_attention.py` | 14.2 竞争性 softmax |
| `starVLA/dataloader/gr00t_lerobot/htcs_codec_transform.py` | 数据 transform：读取离线 codec 包 |
| `examples/LIBERO/train_files/codec_preprocess.py` | 离线 codec 提取 |
| `examples/LIBERO/train_files/starvla_htcs_libero.yaml` | HTCS 训练 config |
| `examples/LIBERO/train_files/run_htcs_libero_train.sh` | 训练启动 |
| `examples/LIBERO/eval_files/e1_counterfactual.py` | E1 反事实评测 |
| `examples/LIBERO/eval_files/htcs_ablations/*.yaml` | E2–E15 各自一个 config |
| `examples/LIBERO/train_files/modality_htcs.json` | 含 codec 字段的 modality 声明 |

### ✏️ 最小修改（1 处）

| 路径 | 改动 |
|---|---|
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | `LeRobotSingleDataset.__getitem__` 末尾加 3 行 hook，触发 HTCSCodecLoader |

### ❌ 绝不要碰

- 任何 `starVLA/model/framework/VLM4A/Qwen*.py`（不要直接改现有 framework，照抄一份重命名为 HTCS.py）
- `starVLA/training/train_starvla.py`（用 YAML 配置，不改训练循环）
- `starVLA/model/modules/action_model/`（action head 通过 YAML 切换，不要新写）
- `starVLA/model/modules/vlm/`（VLM backbone 由 starVLA 维护）
- `starVLA/config/deepseeds/`（DeepSpeed 配置不要改）

> 改这些会导致与 `upstream` rebase 时产生大量冲突，违背 [§7 工作纪律](#7-工作纪律) 的第二条。

---

## 5. 实现 Roadmap（高层）

| Phase | 内容 | 工时 | 完成标准 |
|---|---|---|---|
| 0 | 环境 + 数据 + 复现 starVLA QwenPI baseline | 1–2 天 | QwenPI 在 LIBERO-Long 跑出 ≥ 报告数字 |
| 1 | Codec 离线预处理 + dataloader hook | 1–2 天 | DataLoader 输出 example 含 codec 字段 |
| 2 | HTCS 模块编码（Stage1/Stage2/Framework） | 3–5 天 | `python HTCS.py` 假 batch forward 通过 |
| 3 | 首版 30K 步训练 + LIBERO eval | 1 周 | LIBERO-Long ≥ 96.2%（MoSA-VLA 持平） |
| 4 | 硬实验：E1 / E8 / E9 / E10 | 2 周 | E1 drop > 0.15；E8 codec 比 RGB +pp |
| 5 | 完整消融：E2–E7 / E11–E15 | 2–3 周 | 全 ablation 表填完 |
| 6 | 升级：DiT action head + 两阶段训练 | 1–2 周 | 第二里程碑数字 |

**关键决策点**：Phase 3 完成后看 LIBERO 数字 + E1 反事实。任一不通过 → 回设计阶段，**不要往下做完整消融**。

详见实现文档 §9 + 工作文档 §9。

---

## 6. starVLA 关键约定（动手前必懂）

### 6.1 Framework 注册

每个 VLA 是一个继承 `baseframework` 的类，挂在 `starVLA/model/framework/VLM4A/<Name>.py`，用装饰器注册：

```python
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("HTCS")
class HTCS(baseframework):
    def __init__(self, config): ...
    def forward(self, examples: List[dict]) -> dict: ...
    def predict_action(self, examples) -> dict: ...
```

`build_framework(cfg)` 通过 `cfg.framework.name` 路由到对应类。

### 6.2 Forward 签名（强约定）

`forward` 接收 `examples: List[dict]`，每个 dict 至少含：

```python
{
    "image": List[PIL.Image],       # 多视角，T 帧
    "lang": str,                     # 指令
    "action": np.ndarray,            # [T, action_dim]
    "state": np.ndarray,             # 可选
    "codec": {                       # 【HTCS 新增字段】
        "mv": np.ndarray,            # (T, 14, 14, 2) int8
        "residual": np.ndarray,      # (T, 14, 14) float16
        "is_i_frame": np.ndarray,    # (T,) bool
    },
    "task_id": int,                  # 用于 aux loss
}
```

返回 dict 必含 `action_loss`，可选 `loss_breakdown` 用于 logging。

### 6.3 Config-driven 一切

所有超参走 YAML（`examples/LIBERO/train_files/starvla_htcs_libero.yaml`），命令行覆盖用 `--framework.stage1.keep_ratio 0.2`。**任何 magic number 不要硬编码在代码里**。

### 6.4 模块组合

| starVLA 已实现 | HTCS 直接复用方式 |
|---|---|
| `starVLA/model/modules/vlm/QWen3_5.py` | `get_vlm_model(config)` |
| `starVLA/model/modules/action_model/MLP_ActionHeader.py` | `action_model_type: MLP` |
| `starVLA/model/modules/action_model/DiTActionHeader.py` | `action_model_type: DiT` |
| `starVLA/dataloader/gr00t_lerobot/` | `dataset_py: lerobot_datasets` |
| `starVLA/training/train_starvla.py` | 直接调用，不改 |
| `examples/LIBERO/eval_files/eval_libero.py` | 直接调用，不改 |

---

## 7. 工作纪律（减少与 upstream 的冲突）

三条规则严格执行，rebase 时几乎零冲突：

1. **新代码进新文件**（HTCS.py、htcs/ 目录、starvla_htcs_libero.yaml 都是全新文件——永远不冲突）
2. **对现有文件改动最小化**——只加 hook，不改逻辑。例如 `LeRobotSingleDataset.__getitem__` 只加 3 行
3. **配置走 YAML 不改 default**——所有 HTCS 参数从 YAML 注入，**不修改 starVLA 任何模块的默认值**

如果发现要"为了 HTCS 改某个 starVLA 文件的逻辑"——先想想能不能通过：
- 在新的 HTCS 文件里包一层（wrapper）
- 通过 YAML 配置开关
- 写一个 hook / callback

只有这三个都不行才修改 starVLA 文件，**并且修改提交单独打 tag**（commit message 加 `[upstream-touch]` 前缀），方便 rebase 时定位。

---

## 8. 工作流参考

> 第一次 setup 完之后这一节按需查。

### 8.1 首次 clone + 设置 upstream

```bash
# 1. 在 GitHub 上把 starVLA/starVLA fork 到 <your-org>/starVLA-htcs
# 2. clone 你的 fork（不是官方）
git clone git@github.com:<your-org>/starVLA-htcs.git
cd starVLA-htcs

# 3. 加 upstream remote
git remote add upstream https://github.com/starVLA/starVLA.git
git remote -v   # 验证

# 4. 拉一份 upstream 最新（不 merge 进当前分支）
git fetch upstream

# 5. 开 HTCS 工作分支
git checkout -b htcs-dev upstream/starVLA_dev
git push -u origin htcs-dev
```

### 8.2 日常开发

```bash
git checkout htcs-dev
# 改代码 / 提交
git add starVLA/model/framework/VLM4A/HTCS.py ...
git commit -m "[htcs] add Stage1 codec selector"
git push origin htcs-dev
```

### 8.3 拉 upstream 更新

```bash
git fetch upstream
git checkout htcs-dev
git rebase upstream/starVLA_dev           # 推荐
# 如果冲突, 解决后:
#   git add <文件>
#   git rebase --continue
git push --force-with-lease origin htcs-dev
```

> `--force-with-lease` 比 `--force` 安全：如果别人在 origin/htcs-dev 上推过你不知道的提交，会拒绝推送。

### 8.4 出实验结果 / 投稿时切到 stable

```bash
# 基于稳定分支建一个发布分支
git checkout -b htcs-stable upstream/starVLA
git cherry-pick <htcs-dev 上的 HTCS commits>
# 或者：
git rebase --onto upstream/starVLA upstream/starVLA_dev htcs-dev
```

---

## 9. 给新 session 的开工指引

如果你（新的 Claude session / 协作者）刚被指派进入本 fork 干活，请按这个顺序：

1. **读这份 README 到 §6**（10 分钟）
2. **读 `新工作-HTCS-实现文档.md` §0 + §1 + §4**（15 分钟）—— 理解 starVLA 约定 + 文件 map + Framework 模板
3. **`git log --oneline -20`** 看 HTCS 已有提交，确认当前 Phase 进度
4. **`cat playground/Checkpoints/*/run_*.sh | head`** 看上次训练用的 config 是哪个
5. 根据当前 Phase 找对应任务：
   - Phase 0 没过 → `bash examples/LIBERO/train_files/run_libero_train.sh`（默认 QwenPI baseline）
   - Phase 1 没过 → 写 `codec_preprocess.py`
   - Phase 2 没过 → 写 `HTCS.py` 或 stage1/2 模块
   - Phase 3+ → 看 wandb dashboard 当前训练进度

**新 session 不要做的事**：
- 不要修改 `新工作-语言条件化历史压缩-工作文档.md` 的故事 / Claims（除非用户明确要求）
- 不要新增 framework 文件（HTCS 只有一个 framework 类）
- 不要"为了让代码更优雅"重构 starVLA 的现有模块
- 不要做实现文档之外的 ablation（实验是有计划的，E1–E15 已枚举）

**新 session 应该做的事**：
- 严格按实现文档的张量形状 / 类签名写
- 提交 message 用 `[htcs]` 前缀
- 训练前先 `python starVLA/model/framework/VLM4A/HTCS.py` 自测一遍
- 每完成一个 Phase 在本 README §5 的 roadmap 表标记 ✅

---

## 10. 已知风险 / 注意点

### 10.1 不要 push 到 starVLA 官方
原则上不会发生（你没有 push 权限），但**永远不要 `git push upstream`**。如果未来你做了好东西想贡献给 starVLA，发 PR 而不是直接 push。

### 10.2 训练 checkpoint 不入 git
所有 `playground/Checkpoints/` 应在 `.gitignore` 里（starVLA 已默认配好）。**绝不 commit 二进制 ckpt**。

### 10.3 wandb credentials 不入 git
`wandb_entity` 用环境变量或本地 YAML override，**不写进入库的 YAML**。

### 10.4 LIBERO 数据集不入 git
`playground/Datasets/` 也在 `.gitignore`。codec 预提取产物 `codec.parquet` 也不入 git——重新跑 `codec_preprocess.py` 即可重建。

### 10.5 starVLA `starVLA_dev` 分支可能 break
README 自己说了"may be temporarily unstable"。如果 rebase 后训练 break，先 `git log upstream/starVLA_dev --oneline -10` 看最近 starVLA 有没有大改动，必要时 rebase 到上一个稳定 commit。

### 10.6 Phase 0 不能跳
"先跑通 starVLA 默认 QwenPI" 这一步看起来浪费时间，**实际上能帮你 debug 70% 的环境 / 数据 / flash-attn 问题**。HTCS 自己跑出 NaN 时你不会想再 debug 这些。

---

## 11. 引用 / Credit

本 fork 基于 starVLA：

```
@misc{starvla2026,
  title={StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing},
  author={...},
  year={2026},
  eprint={2604.05014},
  archivePrefix={arXiv}
}
```

发表论文时请引用 starVLA 同时声明本 fork 的修改。

HTCS 方法本身的引用待论文发表后补充。

---

## 12. 联系

- HTCS 方法 / 实验设计：xieji.li@monash.edu
- starVLA 库问题：[starVLA issue tracker](https://github.com/starVLA/starVLA/issues)
