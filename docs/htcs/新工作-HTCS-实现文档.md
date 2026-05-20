# HTCS 实现规范（基于 starVLA）

> **目的**：把 HTCS 设计映射到 starVLA 库的代码骨架。每节给出 **starVLA 树内的具体文件位置、类签名、张量形状、YAML 配置**。设计动机/故事/风险见 `新工作-语言条件化历史压缩-工作文档.md`。
>
> **基线库**：`code/starVLA`（lego 式模块化 VLA 库，Apache + MIT）
> **不重写的部分**：trainer、optimizer、dataloader、eval scripts、action heads、VLM backbone wrapper
> **新增的部分**：HTCS framework 类（一份）、Stage1/Stage2 子模块（三份）、codec 数据预处理脚本（一份）、HTCS YAML config（一份）
> **总训练步数预算**：30K–80K（参考 starVLA LIBERO 默认 80K），8×L20 (48GB)

---

## 0. starVLA 关键约定（必读）

阅读 `code/starVLA/docs/starVLA_guideline.md` 和 `code/starVLA/examples/LIBERO/README.md` 前先理解三点：

### 0.1 Framework 注册模式

每个 VLA framework 都是 `baseframework` 子类，挂在 `starVLA/model/framework/VLM4A/<Name>.py`，用 `@FRAMEWORK_REGISTRY.register("Name")` 暴露给 build 系统。**HTCS 走这条路**：

```python
# starVLA/model/framework/VLM4A/HTCS.py
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("HTCS")
class HTCS(baseframework):
    ...
```

### 0.2 模块来源

- **VLM**：从 `starVLA/model/modules/vlm/` 取——HTCS 用 `QWen3_5.py`（0.8B/2B/4B/9B 都支持），通过 `get_vlm_model(config)` 拿到
- **Action Head**：从 `starVLA/model/modules/action_model/` 取——第一阶段 `MLP_ActionHeader`，第二阶段 `DiTActionHeader` 或 `LayerwiseFM_ActionHeader`，**不要自己写**
- **HTCS 新模块**：只增加 `starVLA/model/modules/htcs/`（新目录），包含 Stage1/Stage2/SaliencyMLP/CompetitiveSlotAttention 四个文件

### 0.3 数据与训练

- 数据走 **LeRobot 格式**，已有 `examples/LIBERO/data_preparation.sh` 一键下载 4 套件
- 训练走 `starVLA/training/train_starvla.py` + Accelerate + DeepSpeed ZeRO-2
- 多帧 video 已支持（`trainer.vla_data.video_backend: torchvision_av`），HTCS 只需要在数据预处理时把 codec 副产物**写回 LeRobot 数据集的 meta 字段**

### 0.4 Forward 签名（必须对齐）

参考 `QwenPI.py` 的 `forward(self, examples: List[dict], **kwargs)`，每个 example 是：

```python
{
    "image": List[PIL.Image],   # 多视角，T 帧历史
    "lang": str,
    "action": np.ndarray,       # [T, action_dim]
    "state": np.ndarray,        # 可选，[T, state_dim]——HTCS 不用
    "codec": dict,              # 【HTCS 新增】MV / Residual / I-frame indices
}
```

---

## 1. HTCS 在 starVLA 内的文件清单

| 新增/修改 | 路径 | 行数预估 | 说明 |
|---|---|---|---|
| 🆕 | `starVLA/model/framework/VLM4A/HTCS.py` | ~250 | 顶层 framework 类，注册 `HTCS` |
| 🆕 | `starVLA/model/modules/htcs/__init__.py` | ~5 | export |
| 🆕 | `starVLA/model/modules/htcs/saliency_mlp.py` | ~30 | α(ℓ), β(ℓ) MLP |
| 🆕 | `starVLA/model/modules/htcs/stage1_codec_selector.py` | ~120 | M2 |
| 🆕 | `starVLA/model/modules/htcs/stage2_lang_compressor.py` | ~80 | M3 |
| 🆕 | `starVLA/model/modules/htcs/slot_attention.py` | ~50 | 14.2 竞争性 softmax |
| 🆕 | `starVLA/dataloader/gr00t_lerobot/htcs_codec_transform.py` | ~80 | 数据 transform：读取离线 codec 包 |
| 🆕 | `examples/LIBERO/train_files/codec_preprocess.py` | ~150 | 离线 codec 提取脚本 |
| 🆕 | `examples/LIBERO/train_files/starvla_htcs_libero.yaml` | ~80 | HTCS 训练 config |
| 🆕 | `examples/LIBERO/train_files/run_htcs_libero_train.sh` | ~40 | 训练启动 |
| 🆕 | `examples/LIBERO/eval_files/e1_counterfactual.py` | ~100 | E1 反事实指令评测 |
| 🆕 | `examples/LIBERO/eval_files/htcs_ablations/*.yaml` | ~10×多 | E2–E15 各自一个 config |
| ✏️ | `examples/LIBERO/train_files/modality.json` | +5 | 增加 codec 字段声明 |

**总工作量估计**：~900 行原创 + 12 行修改。

---

## 2. M1 — Codec 提取（离线预处理）

> **选择离线而不是 online**：starVLA 用 LeRobot + torchvision_av 已经做了高效的多帧 video 解码；如果在 dataloader 里临时跑 ffmpeg/PyAV 会撞 IO 瓶颈。**正确做法是把 codec 副产物预先提取，存进 LeRobot 数据集的 metadata 字段**。

### 2.1 离线脚本：`examples/LIBERO/train_files/codec_preprocess.py`

```python
"""
对 LIBERO LeRobot 数据集做一次性 codec 提取：
- 读取每个 episode 的 video.mp4
- 用 PyAV 重新编码为 HEVC（GOP=8, preset=medium）
- 提取 I-frame indices / P-frame MV / P-frame residual energy
- 写回 episode 的 meta/codec.parquet
"""

import av
from pathlib import Path
import numpy as np
import pandas as pd

def extract_codec_for_episode(video_path: Path, gop_size: int = 8):
    container_out_path = video_path.with_suffix(".hevc.mp4")
    # 1. 重编码为 HEVC
    in_ctx = av.open(str(video_path))
    out_ctx = av.open(str(container_out_path), mode='w')
    stream = out_ctx.add_stream('libx265', rate=30)
    stream.options = {'preset': 'medium', 'g': str(gop_size), 'x265-params': 'log-level=error'}

    frames = list(in_ctx.decode(video=0))
    for f in frames:
        out_ctx.mux(stream.encode(f))
    out_ctx.mux(stream.encode())
    out_ctx.close()

    # 2. 解码 HEVC 取 MV / residual
    in_ctx2 = av.open(str(container_out_path))
    in_ctx2.streams.video[0].codec_context.export_mvs = True

    i_frame_idx, mv_list, res_list = [], [], []
    for i, packet in enumerate(in_ctx2.demux(video=0)):
        for frame in packet.decode():
            if frame.key_frame:
                i_frame_idx.append(i)
                mv_list.append(np.zeros((14, 14, 2), dtype=np.int8))
                res_list.append(np.zeros((14, 14), dtype=np.float16))
            else:
                mv = extract_mv_grid(frame, grid_size=14)         # (14,14,2) int8
                res = extract_residual_energy(frame, grid=14)     # (14,14) float16
                mv_list.append(mv)
                res_list.append(res)

    # 3. 写 parquet
    df = pd.DataFrame({
        'frame_idx': list(range(len(mv_list))),
        'is_i_frame': [j in i_frame_idx for j in range(len(mv_list))],
        'mv': [m.tobytes() for m in mv_list],
        'residual': [r.tobytes() for r in res_list],
    })
    df.to_parquet(video_path.parent / 'codec.parquet')


def main():
    data_root = Path('playground/Datasets/LEROBOT_LIBERO_DATA')
    for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
        for episode_dir in (data_root / f'{suite}_no_noops_1.0.0_lerobot' / 'videos').rglob('*.mp4'):
            extract_codec_for_episode(episode_dir)


if __name__ == '__main__':
    main()
```

**运行**：
```bash
python examples/LIBERO/train_files/codec_preprocess.py
# 预计 4 套件总耗时 1–2 小时（CPU multi-process 可加速）
```

### 2.2 数据 transform：`starVLA/dataloader/gr00t_lerobot/htcs_codec_transform.py`

```python
"""
在 LeRobot dataloader 输出 example 之前，读取离线提取的 codec.parquet,
把 MV/Residual/I-frame mask 加入 example['codec']。
"""

import numpy as np
import pandas as pd
from pathlib import Path


class HTCSCodecLoader:
    def __init__(self, history_len: int = 16):
        self.T = history_len
        self._cache = {}    # episode_path -> dataframe

    def _load_codec(self, episode_dir: Path):
        if episode_dir not in self._cache:
            self._cache[episode_dir] = pd.read_parquet(episode_dir / 'codec.parquet')
        return self._cache[episode_dir]

    def __call__(self, example: dict, episode_dir: Path, frame_indices: list[int]) -> dict:
        df = self._load_codec(episode_dir).iloc[frame_indices]
        mv = np.stack([np.frombuffer(b, dtype=np.int8).reshape(14, 14, 2) for b in df['mv']])
        res = np.stack([np.frombuffer(b, dtype=np.float16).reshape(14, 14) for b in df['residual']])
        is_i = df['is_i_frame'].values
        example['codec'] = {
            'mv': mv,                # (T, 14, 14, 2) int8
            'residual': res,         # (T, 14, 14)   float16
            'is_i_frame': is_i,      # (T,)          bool
        }
        return example
```

### 2.3 注册到现有 dataloader

修改 `starVLA/dataloader/gr00t_lerobot/datasets.py` 的 `LeRobotSingleDataset.__getitem__`（**只加一行**）：

```python
# 在返回 example 前
if getattr(self, '_htcs_codec_loader', None):
    example = self._htcs_codec_loader(example, episode_dir, frame_indices)
return example
```

在 `modality.json` 中声明 codec 字段（让 starVLA 不会过滤掉）：

```json
{
    "video": {...},
    "codec": {
        "mv": {"shape": [16, 14, 14, 2], "dtype": "int8"},
        "residual": {"shape": [16, 14, 14], "dtype": "float16"},
        "is_i_frame": {"shape": [16], "dtype": "bool"}
    }
}
```

---

## 3. M2 / M3 — HTCS 子模块

### 3.1 `starVLA/model/modules/htcs/saliency_mlp.py`

```python
import torch
import torch.nn as nn


class SaliencyMLP(nn.Module):
    """语言 embedding → (α, β)，调制 codec saliency 公式."""

    def __init__(self, d_text: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text, hidden), nn.GELU(),
            nn.Linear(hidden, 2), nn.Softplus(),
        )

    def forward(self, lang_emb: torch.Tensor):
        """
        lang_emb: (B, L_text, d_text)
        returns: alpha, beta — both (B, 1, 1, 1) broadcast-ready
        """
        ab = self.net(lang_emb.mean(dim=1))             # (B, 2)
        return ab[:, 0:1, None, None], ab[:, 1:2, None, None]
```

### 3.2 `starVLA/model/modules/htcs/slot_attention.py`

```python
import torch
import torch.nn as nn


class CompetitiveSlotAttention(nn.Module):
    """14.2: softmax 沿 slot 维归一; 14.3: 支持 Stage1 saliency bias."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.scale = (d_model // n_heads) ** -0.5
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, slots, patches, saliency_bias=None, bias_scale=1.0):
        B, K, d = slots.shape
        N = patches.shape[1]
        H, dh = self.n_heads, d // self.n_heads

        Q = self.q_proj(slots).view(B, K, H, dh).transpose(1, 2)
        Kk = self.k_proj(patches).view(B, N, H, dh).transpose(1, 2)
        V = self.v_proj(patches).view(B, N, H, dh).transpose(1, 2)

        logits = torch.einsum('bhkd,bhnd->bhkn', Q, Kk) * self.scale
        if saliency_bias is not None:
            logits = logits + bias_scale * saliency_bias[:, None, None, :]

        # 14.2 必须在 fp32 下避免 bf16 NaN
        with torch.cuda.amp.autocast(enabled=False):
            attn = logits.float().softmax(dim=2)              # 沿 K 归一
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8) # 沿 N 二次 normalize

        out = torch.einsum('bhkn,bhnd->bhkd', attn.to(V.dtype), V)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, K, d))
```

### 3.3 `starVLA/model/modules/htcs/stage1_codec_selector.py`

```python
import torch
import torch.nn as nn
from .saliency_mlp import SaliencyMLP


class Stage1CodecSelector(nn.Module):
    """
    输入: codec dict + 语言 embedding + ViT (借自 VLM)
    输出: 稀疏 RGB patch tokens (B, N_kept, d_p) + saliency 分数 + coords
    """

    def __init__(self,
                 vit_encoder: nn.Module,           # 借自 qwen_vl_interface.model.visual
                 d_text: int,
                 keep_ratio: float = 0.20,
                 grid_size: int = 14):
        super().__init__()
        self.vit = vit_encoder                     # 已 freeze in framework
        self.saliency_mlp = SaliencyMLP(d_text=d_text)
        self.keep_ratio = keep_ratio
        self.G = grid_size                         # 14

    def forward(self, codec: dict, lang_emb: torch.Tensor,
                frames: torch.Tensor):
        """
        codec['mv']: (B, T, G, G, 2) int8
        codec['residual']: (B, T, G, G) float16
        codec['is_i_frame']: (B, T) bool
        lang_emb: (B, L_text, d_text)
        frames: (B, T, 3, H, W) — 已 resize 到 ViT 输入
        """
        B, T = codec['mv'].shape[:2]
        device = frames.device

        # 1. saliency 评分
        alpha, beta = self.saliency_mlp(lang_emb)              # (B,1,1,1) 各
        mv_norm = codec['mv'].float().norm(dim=-1)             # (B,T,G,G)
        residual = codec['residual'].float()
        s = alpha * mv_norm + beta * residual                  # (B,T,G,G)

        # 2. 全帧过 ViT (复用 VLM 的 vision encoder)
        with torch.no_grad():
            patches = self.vit(frames.flatten(0, 1))           # (B*T, G*G, d_p)
        d_p = patches.shape[-1]
        patches = patches.view(B, T, self.G * self.G, d_p)     # (B,T,196,d_p)

        # 3. I-frame: 全保留; P-frame: top-ρ
        is_i = codec['is_i_frame']                             # (B,T) bool
        s_flat = s.view(B, T * self.G * self.G)                # (B, T*196)
        patches_flat = patches.view(B, T * self.G * self.G, d_p)

        # 给 I-frame 的 patch 一个 s = +inf 保证全选
        i_mask = is_i.view(B, T, 1, 1).expand(-1, -1, self.G, self.G).reshape(B, -1)
        s_for_topk = torch.where(i_mask, torch.full_like(s_flat, float('inf')), s_flat)

        k_total = int((self.keep_ratio * T + (~is_i[0]).sum() * 0) * self.G * self.G)  # 近似
        # 实际: I-frame 全保留 + P-frame top-ρ%
        n_p_patches = ((~is_i[0]).sum().item()) * self.G * self.G
        n_i_patches = (is_i[0].sum().item()) * self.G * self.G
        N_kept = n_i_patches + int(self.keep_ratio * n_p_patches)

        topk_vals, topk_idx = s_for_topk.topk(N_kept, dim=1)
        P = torch.gather(patches_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d_p))

        # 4. saliency 分数 (minmax 归一到 [0,1], I-frame 设 1)
        s_kept = torch.where(torch.isinf(topk_vals),
                              torch.ones_like(topk_vals),
                              (topk_vals - topk_vals.min(dim=1, keepdim=True).values) /
                              (topk_vals.max(dim=1, keepdim=True).values + 1e-6))

        # 5. coords for 3D RoPE: (t, h, w)
        t_idx = topk_idx // (self.G * self.G)
        hw = topk_idx % (self.G * self.G)
        h_idx, w_idx = hw // self.G, hw % self.G
        coords = torch.stack([t_idx, h_idx, w_idx], dim=-1)    # (B, N_kept, 3)

        return P, s_kept, coords
```

### 3.4 `starVLA/model/modules/htcs/stage2_lang_compressor.py`

```python
import torch
import torch.nn as nn
from .slot_attention import CompetitiveSlotAttention


class Stage2LangCompressor(nn.Module):
    """14.1 + 14.2 + 14.3 三件套."""

    def __init__(self, d_text: int, d_patch: int, d_vlm: int,
                 K: int = 8, n_heads: int = 4):
        super().__init__()
        self.K = K
        self.learnable_q = nn.Parameter(torch.randn(K, d_text) * 0.02)
        self.q_from_lang = nn.MultiheadAttention(d_text, n_heads, batch_first=True)
        self.proj_to_patch = nn.Linear(d_text, d_patch)
        self.cross_attn = CompetitiveSlotAttention(d_patch, n_heads)
        self.proj_to_vlm = nn.Linear(d_patch, d_vlm)
        self.norm = nn.LayerNorm(d_vlm)

    def forward(self, P, s_stage1, lang_emb):
        B = P.shape[0]
        q = self.learnable_q.unsqueeze(0).expand(B, -1, -1)
        slots, _ = self.q_from_lang(q, lang_emb, lang_emb)       # 14.1
        slots = self.proj_to_patch(slots)
        H_raw = self.cross_attn(slots, P, saliency_bias=s_stage1) # 14.2+14.3
        return self.norm(self.proj_to_vlm(H_raw))
```

---

## 4. 顶层 Framework：`starVLA/model/framework/VLM4A/HTCS.py`

模仿 `QwenPI.py` 的写法。**核心要点**：
1. 继承 `baseframework`
2. 用 `@FRAMEWORK_REGISTRY.register("HTCS")` 注册
3. 在 `__init__` 里组装 VLM (via `get_vlm_model`) + Stage1 + Stage2 + ActionHead
4. `forward(examples: List[dict])` 对齐 starVLA 约定

```python
"""HTCS Framework — Hierarchical Task-Conditioned Saliency."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model as get_mlp_head
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model as get_dit_head
from starVLA.model.modules.htcs import (
    Stage1CodecSelector, Stage2LangCompressor,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class HTCSDefaultConfig:
    name: str = "HTCS"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3.5-0.8B",
        "attn_implementation": "flash_attention_2",
        "vl_hidden_dim": 1536,
        "num_vl_layers": 24,
    })
    stage1: dict = field(default_factory=lambda: {
        "keep_ratio": 0.20,
        "grid_size": 14,
    })
    stage2: dict = field(default_factory=lambda: {
        "K": 8,
        "n_heads": 4,
        "d_patch": 1152,                          # SigLIP-Large 输出维度
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",               # 第一阶段; 后续改 "DiT"
        "action_dim": 7,
        "state_dim": 7,
        "action_horizon": 8,
    })
    aux_loss: dict = field(default_factory=lambda: {
        "lambda_aux": 0.1,
        "n_tasks": 40,                            # LIBERO 4 套件共 40 任务
    })


@FRAMEWORK_REGISTRY.register("HTCS")
class HTCS(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(HTCSDefaultConfig, config)

        # ---------- VLM (Qwen3-VL-0.8B) ----------
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        d_vlm = int(vlm_hf_cfg.hidden_size)
        d_text = d_vlm
        self.config.framework.qwenvl.vl_hidden_dim = d_vlm
        self.config.framework.qwenvl.num_vl_layers = int(text_cfg.num_hidden_layers)

        # ---------- Stage 1 ----------
        # ViT encoder 借自 VLM 的 visual tower; 强制 freeze
        vit = self.qwen_vl_interface.model.visual
        for p in vit.parameters():
            p.requires_grad = False
        self.stage1 = Stage1CodecSelector(
            vit_encoder=vit,
            d_text=d_text,
            keep_ratio=self.config.framework.stage1.keep_ratio,
            grid_size=self.config.framework.stage1.grid_size,
        )

        # ---------- Stage 2 ----------
        self.stage2 = Stage2LangCompressor(
            d_text=d_text,
            d_patch=self.config.framework.stage2.d_patch,
            d_vlm=d_vlm,
            K=self.config.framework.stage2.K,
            n_heads=self.config.framework.stage2.n_heads,
        )

        # ---------- Fusion (M4) ----------
        self.fusion = nn.MultiheadAttention(d_vlm, num_heads=8, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_vlm)

        # ---------- Action Head (M5) ----------
        # 第一阶段: MLP_ActionHeader + L1; 后续切换 DiT 只改 config
        head_type = self.config.framework.action_model.action_model_type
        if head_type == "MLP":
            self.action_model = get_mlp_head(config=self.config)
        elif head_type == "DiT":
            self.action_model = get_dit_head(config=self.config)
        else:
            raise ValueError(f"Unknown action head: {head_type}")

        # ---------- Aux task classifier (D4) ----------
        self.task_head = nn.Linear(d_vlm, self.config.framework.aux_loss.n_tasks)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """
        examples[i]:
          'image': List[PIL.Image] (T 帧历史 × n_view)
          'lang': str
          'action': np.ndarray [T_full, action_dim]
          'codec': {'mv', 'residual', 'is_i_frame'} (numpy)
          'task_id': int (用于 aux loss)
        """
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        task_ids = torch.tensor([ex.get("task_id", 0) for ex in examples],
                                device=next(self.parameters()).device)

        # ---- 1. 语言 embedding (复用 VLM text embedder) ----
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[imgs[-1] for imgs in batch_images],  # 当前帧给 VLM
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_out = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True,
            )
            # 取最后一层 hidden state 作为 Q (action queries)
            Q = vlm_out.hidden_states[-1]                           # (B, L_seq, d_vlm)
            # 抽语言 token embedding 给 Stage 1/2
            lang_emb = vlm_out.hidden_states[0]                     # (B, L_seq, d_vlm)

        # ---- 2. 准备 codec 与历史帧 tensors ----
        codec = self._collate_codec([ex["codec"] for ex in examples])
        history_frames = self._collate_history(batch_images)         # (B,T,3,H,W)

        # ---- 3. Stage 1 ----
        P, s_stage1, coords = self.stage1(codec, lang_emb, history_frames)
        # P: (B, N_kept, d_p)

        # 3D RoPE 加到 P 上 (借助 share_tools 的工具函数)
        from starVLA.model.framework.share_tools import apply_3d_rope
        P = apply_3d_rope(P, coords)

        # ---- 4. Stage 2 ----
        H = self.stage2(P, s_stage1, lang_emb)                      # (B, K, d_vlm)

        # ---- 5. Fusion (M4) ----
        action_q = Q[:, -self.action_horizon:, :]                   # 取末尾 C 个 token 作 action query
        fused, _ = self.fusion(action_q, H, H)
        Q_prime = self.fusion_norm(action_q + fused)                # (B, C, d_vlm)

        # ---- 6. Action loss ----
        actions = torch.tensor(np.array(actions),
                               device=Q_prime.device, dtype=Q_prime.dtype)
        actions_target = actions[:, -self.action_horizon:, :]

        action_loss = self.action_model(Q_prime, actions_target, None)

        # ---- 7. Aux task classifier loss ----
        task_logits = self.task_head(H.mean(dim=1))
        aux_loss = nn.functional.cross_entropy(task_logits, task_ids)
        lambda_aux = self.config.framework.aux_loss.lambda_aux

        return {
            "action_loss": action_loss + lambda_aux * aux_loss,
            "loss_breakdown": {
                "action": action_loss.detach(),
                "aux": aux_loss.detach(),
            },
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference path — 与 forward 几乎一致, 但不算 loss."""
        # ... (略, 见 QwenPI.predict_action 模板)
        pass

    # ---------- private helpers ----------
    def _collate_codec(self, codec_list: List[dict]) -> dict:
        device = next(self.parameters()).device
        return {
            'mv': torch.from_numpy(np.stack([c['mv'] for c in codec_list])).to(device),
            'residual': torch.from_numpy(np.stack([c['residual'] for c in codec_list])).to(device),
            'is_i_frame': torch.from_numpy(np.stack([c['is_i_frame'] for c in codec_list])).to(device),
        }

    def _collate_history(self, batch_images: List[List]) -> torch.Tensor:
        # 把 List[PIL] 转成 (B,T,3,H,W) tensor; resize 到 ViT 输入
        # 参考 starVLA 现有 resize_images 工具
        from starVLA.training.trainer_utils.trainer_tools import resize_images
        # ... (略)
        pass
```

---

## 5. YAML Config：`examples/LIBERO/train_files/starvla_htcs_libero.yaml`

照抄 `starvla_cotrain_libero.yaml`，只改 `framework.name` 和加 HTCS 字段：

```yaml
run_id: htcs_libero_4suite
run_root_dir: playground/Checkpoints
seed: 42
wandb_project: starVLA_HTCS
version_id: "0.1"

framework:
  name: HTCS                                         # ← 关键
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3.5-0.8B
    attn_implementation: flash_attention_2
  stage1:
    keep_ratio: 0.20
    grid_size: 14
  stage2:
    K: 8
    n_heads: 4
    d_patch: 1152
  action_model:
    action_model_type: MLP                           # 第一阶段; 后续切 "DiT"
    action_dim: 7
    state_dim: 7
    action_horizon: 8
  aux_loss:
    lambda_aux: 0.1
    n_tasks: 40

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA
    data_mix: libero_all                             # 4 套件联合 (D5)
    action_type: delta_qpos
    sequential_step_sampling: False
    history_len: 16                                  # ← HTCS T=16
    per_device_batch_size: 8                         # 8 GPU × 8 = 64 全局
    load_all_data_for_training: true
    video_backend: torchvision_av
    enable_htcs_codec: true                          # ← 新增标志, 触发 HTCSCodecLoader

trainer:
  max_train_steps: 30000                             # 沿用 MoSA-VLA 协议
  num_warmup_steps: 1000
  save_interval: 5000
  eval_interval: 5000
  learning_rate:
    base: 3.0e-04
    qwen_vl_interface: 1.0e-05                       # VLM 解冻部分小 lr
    stage1: 3.0e-04
    stage2: 3.0e-04
    action_model: 3.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 3.0e-05
  freeze_modules: 'qwen_vl_interface.model.visual'   # SigLIP ViT 冻死
  max_grad_norm: 1.0
  weight_decay: 0.05
  logging_frequency: 100
  gradient_accumulation_steps: 1
  gradient_checkpointing: true

  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
```

---

## 6. 启动脚本：`examples/LIBERO/train_files/run_htcs_libero_train.sh`

```bash
#!/bin/bash
Framework_name=HTCS
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_htcs_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=$(date +%m%d)_htcs_libero4in1

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir} && cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_HTCS
```

---

## 7. 评测

starVLA 已经有 LIBERO 完整评测 pipeline（policy server + `eval_libero.py`），**直接复用**：

```bash
CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_30000_pytorch_model.pt

# 1. 起 policy server (HTCS 自动从 ckpt 加载, framework registry 路由)
bash examples/LIBERO/eval_files/run_policy_server.sh ${CKPT}

# 2. 跑 4 套件评测
for suite in libero_spatial libero_object libero_goal libero_10; do
    bash examples/LIBERO/eval_files/eval_libero.sh ${CKPT} ${suite}
done
```

### 7.1 E1 反事实指令（新增脚本）

`examples/LIBERO/eval_files/e1_counterfactual.py`：

```python
"""
E1: 反事实指令替换. 推理时把每个 episode 的 instruction 换成另一个任务的指令,
对比成功率, drop > 0.15 视为语言被真用上.
"""
import random
from examples.LIBERO.eval_files.eval_libero import LiberoRunner

def main():
    runner_orig = LiberoRunner(ckpt=..., counterfactual=False)
    runner_cf = LiberoRunner(ckpt=..., counterfactual=True,
                              instruction_pool=load_all_libero_instructions())
    sr_orig = runner_orig.run(n_episodes=500)
    sr_cf = runner_cf.run(n_episodes=500)
    print(f"E1: orig {sr_orig:.3f}, counterfactual {sr_cf:.3f}, drop {sr_orig-sr_cf:.3f}")
```

`LiberoRunner` 加个 `counterfactual` flag——当 True 时，每 episode 重置时随机抽**其他任务**的指令传给 model server。

---

## 8. 超参数总表

| 类别 | 参数 | 值 | 备注 |
|---|---|---|---|
| **架构** | T (历史帧数) | 16 | YAML `history_len` |
|  | GOP | 8 |  |
|  | ρ (Stage 1 keep ratio) | 0.20 | YAML `framework.stage1.keep_ratio` |
|  | K (slot 数) | 8 | YAML `framework.stage2.K` |
|  | C (chunk size) | 8 | YAML `framework.action_model.action_horizon` |
|  | d_p (SigLIP) | 1152 | 自动从 VLM 取 |
|  | d_vlm (Qwen3.5-0.8B) | 1536 | 自动从 VLM 取 |
| **训练** | Optimizer | AdamW (β1=0.9, β2=0.95) | YAML |
|  | LR (新模块) | 3e-4 | 分组 |
|  | LR (VLM 解冻部分) | 1e-5 |  |
|  | LR (action_model) | 3e-4 |  |
|  | Weight decay | 0.05 |  |
|  | Grad clip | 1.0 |  |
|  | Per-device batch | 8 (8 GPU = 64 global) | L20 48GB |
|  | 训练步数 | 30K |  |
|  | Warmup | 1K linear |  |
|  | LR Schedule | cosine_with_min_lr (3e-5) |  |
|  | bf16 | ✓ (autocast) |  |
|  | Grad checkpointing | ✓ |  |
|  | λ_aux | 0.1 |  |
|  | DeepSpeed Stage | ZeRO-2 |  |
| **Eval** | 每套件 rollouts | 500 | 沿用 starVLA 协议 |
|  | Eval 频率 | 每 5K 步 |  |

---

## 9. 实现 Checklist

### Phase 0：环境与数据（1–2 天）
- [ ] git clone starVLA 并按 `docs/starVLA_guideline.md` 装好 conda env + flash-attn
- [ ] `bash examples/LIBERO/data_preparation.sh` 下载 4 套件
- [ ] 下载 Qwen3.5-0.8B 到 `playground/Pretrained_models/Qwen3.5-0.8B`
- [ ] 跑通 `examples/LIBERO/train_files/run_libero_train.sh`（用默认 QwenPI），验证 starVLA 基本可用
- [ ] **关键里程碑**：QwenPI 在 LIBERO-Long 上跑出 ≥ MoSA-VLA 同档数字（说明库+硬件 OK）

### Phase 1：Codec 预处理（1–2 天）
- [ ] 写 `examples/LIBERO/train_files/codec_preprocess.py`
- [ ] 跑一遍生成 4 套件的 `codec.parquet`
- [ ] 写 `starVLA/dataloader/gr00t_lerobot/htcs_codec_transform.py`
- [ ] 在 `LeRobotSingleDataset.__getitem__` 加注入逻辑
- [ ] 更新 `modality.json` 加 codec 字段
- [ ] **单测**：DataLoader 输出的 example 含正确 shape 的 `codec` 字段

### Phase 2：HTCS 模块（3–5 天）
- [ ] 写 `starVLA/model/modules/htcs/` 下 4 个文件
- [ ] 写 `starVLA/model/framework/VLM4A/HTCS.py`
- [ ] 跑一遍 `python starVLA/model/framework/VLM4A/HTCS.py` 自测 (模仿 QwenGR00T 的 main block)
- [ ] **单测**：HTCS forward 跑通假 batch，loss 不 NaN

### Phase 3：首版训练（1 周）
- [ ] 写 YAML + run script
- [ ] 跑 30K 步训练（4 套件联合）
- [ ] 跑 starVLA 自带 LIBERO eval，记录 4 套件成功率 + 任务间方差
- [ ] **决策点**：若 LIBERO-Long ≥ 96.2%（MoSA-VLA 持平）→ 继续；否则回调（检查 codec 解析、Stage 1/2 shape）

### Phase 4：硬实验（2 周）
- [ ] **E1 反事实指令** ← 立刻跑！drop > 0.15 才能继续
- [ ] **E8** codec-aware vs 同 token RGB（写一个 ablation YAML 关掉 Stage 1，改用 mean-pool）
- [ ] **E9** Stage 1/2 patch IoU（写 hook 在 model 上抓 topk_idx 与 stage2 attn）
- [ ] **E10** 去 Stage 2

### Phase 5：完整消融（2–3 周）
- [ ] E2 / E3 / E4 / E5 / E11 / E12 / E13 / E14 / E15
- [ ] 每个消融一个 YAML（覆盖 framework 字段），共享同一 codec 预处理

### Phase 6：升级（1–2 周，第二里程碑）
- [ ] action head 从 MLP 切到 DiT（改 YAML `action_model_type: DiT`）
- [ ] 两阶段训练（E6）
- [ ] codec residual 消融（E7）

---

## 10. 已知工程陷阱（基于 starVLA 实际行为）

### 10.1 starVLA 的 `freeze_modules` 字段是 dotted prefix
- `freeze_modules: 'qwen_vl_interface.model.visual'` 会冻所有以这个前缀开头的参数
- **建议确认方式**：训练启动后看 `auto_get_trainable_modules` 的 log 输出，确认 visual 不在列表里

### 10.2 LeRobot 多帧采样的 history_len
- starVLA 现有 LeRobot loader 的 `history_len` 字段控制采样的过去帧数
- **HTCS 需要 T=16**——必须与 codec.parquet 里的帧数严格对齐
- **陷阱**：LeRobot 的 frame_indices 可能含负数（episode 边界），HTCSCodecLoader 要 clamp

### 10.3 PyAV `export_mvs` 在 libx265 路径
- ffmpeg 命令行的 `-flags2 +export_mvs` 对 libx265 部分支持，**正确方式是 PyAV 设 `codec_context.export_mvs = True`** + 解码 packet 后从 `frame.side_data` 取
- HEVC 的 MV 在 16×16 子块上插值——保证输出 14×14 网格与 SigLIP patch grid 对齐

### 10.4 bf16 + CompetitiveSlotAttention NaN
- DeepSpeed ZeRO-2 + bf16 下，两次 softmax-normalize 容易 NaN
- 已在 `slot_attention.py` 中用 `autocast(enabled=False)` 强制 fp32 attention
- **症状**：loss 训练 ~500 步突然变 NaN——多半是这里

### 10.5 SigLIP ViT 不能用 `gradient_checkpointing` 包
- starVLA 默认 `gradient_checkpointing: true`，但 ViT 已 freeze 时打开 checkpoint 反而**慢**（多次 forward 重算）
- **建议**：在 framework 里手动对 frozen modules 关掉 checkpointing：
  ```python
  if hasattr(self.qwen_vl_interface.model.visual, 'gradient_checkpointing_disable'):
      self.qwen_vl_interface.model.visual.gradient_checkpointing_disable()
  ```

### 10.6 DeepSpeed ZeRO-2 与 `learning_rate` 分组
- starVLA 用 `trainer.learning_rate.<module_name>` 做分组（见 `trainer_tools.build_param_lr_groups`）
- 新增的 `stage1` / `stage2` 必须在 YAML 的 `learning_rate` 下声明，**否则会落入 base lr**

### 10.7 LIBERO eval 是 CPU-bound
- starVLA 已经处理：`run_policy_server.sh` 起 server，`eval_libero.py` 走 socket 调用
- **不要自己写 rollout 循环**，改 model2libero_interface.py 来接 HTCS

### 10.8 Topk 反传到 SaliencyMLP
- `torch.topk` 不可微，但 saliency_kept 还是有梯度（gather 路径可微）→ SaliencyMLP 能学
- **诊断**：训练 1K 步后查 `saliency_mlp.net[0].weight.grad.norm()`，应该 > 0

---

## 11. 与原计划的差异（备忘）

| 原计划（独立 repo） | starVLA 内 |
|---|---|
| 自己写 `data/codec_extractor.py` 在线提取 | 离线预处理 → LeRobot meta 字段 |
| 自己写 `training/trainer.py` + AdamW | 用 `train_starvla.py` + YAML 配 |
| 自己写 4 套件采样器 | LeRobot `data_mix: libero_all` |
| 自己写 `models/action_head.py` MLP + DiT | `MLP_ActionHeader` + `DiTActionHeader` 现成 |
| 自己写 rollout 脚本 | `eval_libero.sh` + policy server |
| 自己加 wandb 集成 | starVLA 已集成 |
| 单卡训练脚本 | DeepSpeed ZeRO-2 多卡现成 |

**净结果**：原计划估计 900 行原创代码 → starVLA 内 **~600 行原创** + 12 行修改现有文件。**省下来的工程预算全部投到消融实验上**。

---

## 12. 下一步建议

1. **Phase 0 必须先跑通 QwenPI 默认 LIBERO 训练**——这是验证库 + 硬件的最便宜方式，**不要跳过**
2. **Phase 1 写 codec_preprocess.py 时务必 dump 一个 episode 的 MV 可视化图**——确认 MV 与机械臂运动方向一致，不然后面 Stage 1 saliency 没意义
3. **Phase 3 第一次跑训练只用 1 套件 (libero_spatial)**——快速验证 forward / loss / grad flow，**不要直接上 4 套件**
4. 进入 Phase 4 前先看 `auto_get_trainable_modules` 输出，确认 stage1/stage2 的参数都进了 optimizer
