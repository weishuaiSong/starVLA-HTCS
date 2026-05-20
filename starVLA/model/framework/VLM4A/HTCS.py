"""HTCS Framework — Hierarchical Task-Conditioned Saliency.

Top-level VLA framework that wires Qwen3-VL backbone + Stage1 codec selector +
Stage2 language compressor + fusion + (MLP or DiT) action head + auxiliary
task classifier.

Pattern follows ``QwenPI.py`` — see that file as a reference for ``forward``
and ``predict_action`` conventions. Reference: HTCS impl doc §4.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model as get_dit_head
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model as get_mlp_head
from starVLA.model.modules.htcs import (
    Stage1CodecSelector,
    Stage2LangCompressor,
    apply_3d_rope,
)
from starVLA.model.modules.vlm import get_vlm_model
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
        "d_patch": 1152,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",
        "action_dim": 7,
        "state_dim": 7,
        "action_horizon": 8,
        # MLP head specific (forwarded to MLP_ActionHeader.get_action_model).
        "action_hidden_dim": 1536,
    })
    aux_loss: dict = field(default_factory=lambda: {
        "lambda_aux": 0.1,
        "n_tasks": 40,
    })


@FRAMEWORK_REGISTRY.register("HTCS")
class HTCS(baseframework):
    """Hierarchical Task-Conditioned Saliency VLA framework."""

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
        # Make sure the MLP head sees the right input dim once VLM is loaded.
        if hasattr(self.config.framework.action_model, "action_hidden_dim"):
            self.config.framework.action_model.action_hidden_dim = d_vlm

        # ---------- Stage 1 ----------
        # Borrow the visual tower and freeze it. Stage1CodecSelector stores it
        # as a plain attribute (not a submodule) to avoid double registration.
        vit = self.qwen_vl_interface.model.visual
        for p in vit.parameters():
            p.requires_grad = False
        if hasattr(vit, "gradient_checkpointing_disable"):
            # Frozen ViT shouldn't pay re-compute cost (impl doc §10.5).
            try:
                vit.gradient_checkpointing_disable()
            except Exception:
                pass

        self.stage1 = Stage1CodecSelector(
            vit_encoder=vit,
            d_text=d_text,
            keep_ratio=self.config.framework.stage1.keep_ratio,
            grid_size=self.config.framework.stage1.grid_size,
        )

        # ---------- Stage 2 ----------
        self.stage2 = Stage2LangCompressor(
            d_text=d_text,
            d_patch=int(self.config.framework.stage2.d_patch),
            d_vlm=d_vlm,
            K=int(self.config.framework.stage2.K),
            n_heads=int(self.config.framework.stage2.n_heads),
        )

        # ---------- Fusion (M4) ----------
        self.fusion = nn.MultiheadAttention(d_vlm, num_heads=8, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_vlm)

        # ---------- Action head (M5) ----------
        head_type = self.config.framework.action_model.action_model_type
        if head_type == "MLP":
            self.action_model = get_mlp_head(config=self.config)
        elif head_type == "DiT":
            self.action_model = get_dit_head(config=self.config)
        else:
            raise ValueError(f"Unknown action head: {head_type}")

        # ---------- Aux task classifier (D4) ----------
        self.task_head = nn.Linear(d_vlm, int(self.config.framework.aux_loss.n_tasks))

        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    # ------------------------------------------------------------------ #
    #  shared encoder path
    # ------------------------------------------------------------------ #
    def _encode_step(self, examples: List[dict]):
        """Run VLM + Stage1 + Stage2 + Fusion. Returns (Q_prime, H, lang_emb)."""
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]

        # 1. Language + current-frame embeddings via VLM.
        # build_qwenvl_inputs expects List[List[PIL]] — one image list per
        # sample. We pass just the newest frame (imgs[-1]) of the history
        # window since the VLM only needs current visual grounding; the full
        # T-frame history goes to Stage1 / Stage2 via _collate_history.
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[[imgs[-1]] for imgs in batch_images],
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_out = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            Q = vlm_out.hidden_states[-1]            # (B, L_seq, d_vlm)
            lang_emb = vlm_out.hidden_states[0]      # (B, L_seq, d_vlm)

        # 2. Codec + history.
        codec = self._collate_codec([ex["codec"] for ex in examples])
        history_frames = self._collate_history(batch_images)                 # (B, T, 3, H, W)

        # 3. Stage 1 → sparse patches + saliency + coords.
        P, s_stage1, coords = self.stage1(codec, lang_emb, history_frames)
        P = apply_3d_rope(P, coords)

        # 4. Stage 2 → K language-conditioned summary slots.
        H = self.stage2(P, s_stage1, lang_emb)                               # (B, K, d_vlm)

        # 5. Fusion: action queries cross-attend to summary slots.
        action_q = Q[:, -self.action_horizon:, :]
        fused, _ = self.fusion(action_q, H, H)
        Q_prime = self.fusion_norm(action_q + fused)                         # (B, C, d_vlm)
        return Q_prime, H, lang_emb

    # ------------------------------------------------------------------ #
    #  training / inference entry points
    # ------------------------------------------------------------------ #
    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Train-time forward — returns ``action_loss`` + breakdown."""
        actions = [ex["action"] for ex in examples]
        task_ids = torch.tensor(
            [ex.get("task_id", 0) for ex in examples],
            device=next(self.parameters()).device,
            dtype=torch.long,
        )

        Q_prime, H, _ = self._encode_step(examples)

        # Action loss — MLP head: simple L1 between predicted and target chunk.
        actions = torch.tensor(
            np.array(actions),
            device=Q_prime.device,
            dtype=Q_prime.dtype,
        )                                                                     # (B, T_full, action_dim)
        actions_target = actions[:, -self.action_horizon:, :]                 # (B, C, action_dim)

        head_type = self.config.framework.action_model.action_model_type
        if head_type == "MLP":
            pred = self.action_model(Q_prime)                                 # (B, C, action_dim)
            action_loss = F.l1_loss(pred, actions_target)
        else:
            # DiT-style heads return loss when given a target.
            action_loss = self.action_model(Q_prime, actions_target, None)

        # Auxiliary task classifier (D4).
        task_logits = self.task_head(H.mean(dim=1))                           # (B, n_tasks)
        aux_loss = F.cross_entropy(task_logits, task_ids)
        lambda_aux = float(self.config.framework.aux_loss.lambda_aux)

        return {
            "action_loss": action_loss + lambda_aux * aux_loss,
            "loss_breakdown": {
                "action": action_loss.detach(),
                "aux":    aux_loss.detach(),
            },
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """Inference path — mirrors ``forward`` but skips loss computation.

        Returns:
            {'normalized_actions': np.ndarray of shape (B, C, action_dim)}
        """
        if type(examples) is not list:
            examples = [examples]

        Q_prime, _, _ = self._encode_step(examples)
        # Both MLP and DiT action heads expose ``predict_action``.
        pred = self.action_model.predict_action(Q_prime)                      # (B, C, action_dim)
        return {"normalized_actions": pred.detach().float().cpu().numpy()}

    # ------------------------------------------------------------------ #
    #  private helpers
    # ------------------------------------------------------------------ #
    def _collate_codec(self, codec_list: List[dict]) -> dict:
        """Stack per-example numpy codec dicts → batched torch tensors on device."""
        device = next(self.parameters()).device
        return {
            "mv":         torch.from_numpy(np.stack([c["mv"] for c in codec_list])).to(device),
            "residual":   torch.from_numpy(np.stack([c["residual"] for c in codec_list])).to(device),
            "is_i_frame": torch.from_numpy(np.stack([c["is_i_frame"] for c in codec_list])).to(device),
        }

    def _collate_history(self, batch_images: List[List]) -> torch.Tensor:
        """Convert List[List[PIL.Image]] → (B, T, 3, 224, 224) bf16 tensor.

        Uses SigLIP / CLIP normalisation to match the frozen ViT's input
        distribution; falls back to ImageFromArray when given numpy arrays.
        """
        from PIL import Image as _PIL
        from torchvision import transforms

        device = next(self.parameters()).device

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        B = len(batch_images)
        T = len(batch_images[0])
        out = torch.zeros(B, T, 3, 224, 224, dtype=torch.float32)
        for b, frames in enumerate(batch_images):
            for t, img in enumerate(frames):
                if not isinstance(img, _PIL.Image.Image):
                    img = _PIL.fromarray(np.asarray(img, dtype=np.uint8))
                out[b, t] = preprocess(img)

        return out.to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------- #
#  Smoke test (impl doc §9 Phase 2 checklist):
#    python -m starVLA.model.framework.VLM4A.HTCS --config_yaml ...
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_htcs_libero.yaml",
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    model = HTCS(cfg)
    print(model)

    T = 16
    G = 14
    pil_imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                for _ in range(T)]
    sample = {
        "image":   pil_imgs,
        "lang":    "pick up the red block",
        "action":  np.random.uniform(-1, 1, size=(T, 7)).astype(np.float32),
        "task_id": 0,
        "codec": {
            "mv":         np.zeros((T, G, G, 2), dtype=np.int8),
            "residual":   np.zeros((T, G, G), dtype=np.float16),
            "is_i_frame": np.array([i % 8 == 0 for i in range(T)], dtype=bool),
        },
    }
    out = model([sample])
    print("loss:", out["action_loss"].item(), "breakdown:", {k: v.item() for k, v in out["loss_breakdown"].items()})
