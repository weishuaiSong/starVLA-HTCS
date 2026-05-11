"""HTCS Framework — Hierarchical Task-Conditioned Selection.

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
from torch import nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model as get_dit_head
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model as get_mlp_head
from starVLA.model.modules.htcs import Stage1CodecSelector, Stage2LangCompressor
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
    })
    aux_loss: dict = field(default_factory=lambda: {
        "lambda_aux": 0.1,
        "n_tasks": 40,
    })


@FRAMEWORK_REGISTRY.register("HTCS")
class HTCS(baseframework):
    """Hierarchical Task-Conditioned Selection VLA framework."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(HTCSDefaultConfig, config)

        # TODO(htcs):
        #   1. self.qwen_vl_interface = get_vlm_model(config=self.config)
        #   2. Pull d_vlm / d_text / num_layers from VLM hf config; write back
        #      into self.config.framework.qwenvl for downstream YAML logging.
        #   3. Freeze visual tower:
        #        vit = self.qwen_vl_interface.model.visual
        #        for p in vit.parameters(): p.requires_grad = False
        #   4. self.stage1 = Stage1CodecSelector(vit, d_text, keep_ratio, grid_size)
        #   5. self.stage2 = Stage2LangCompressor(d_text, d_patch, d_vlm, K, n_heads)
        #   6. self.fusion = MultiheadAttention(d_vlm, num_heads=8, batch_first=True)
        #      self.fusion_norm = LayerNorm(d_vlm)
        #   7. self.action_model = MLP head or DiT head based on action_model_type.
        #   8. self.task_head = Linear(d_vlm, n_tasks)  — aux classifier (D4).
        #   9. self.action_horizon = action_model.action_horizon (C, default 8).
        raise NotImplementedError

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """
        Each example is a dict with at least:
            'image':   List[PIL.Image]    — multi-view, T history frames
            'lang':    str
            'action':  np.ndarray         — (T_full, action_dim)
            'codec':   dict               — {'mv', 'residual', 'is_i_frame'}
            'task_id': int                — for auxiliary loss (D4)

        Returns:
            {
              'action_loss':    scalar tensor (action_loss + lambda * aux_loss),
              'loss_breakdown': {'action': ..., 'aux': ...},
            }
        """
        # TODO(htcs):
        #   1. Extract batched images, instructions, actions, task_ids.
        #   2. Build VLM inputs via self.qwen_vl_interface.build_qwenvl_inputs(...)
        #      using only the current frame (imgs[-1]); run VLM with
        #      output_hidden_states=True under autocast(bfloat16).
        #        Q        = vlm_out.hidden_states[-1]   # (B, L_seq, d_vlm)
        #        lang_emb = vlm_out.hidden_states[0]    # (B, L_seq, d_vlm)
        #   3. Collate codec + history frames (see _collate_* helpers below).
        #   4. P, s_stage1, coords = self.stage1(codec, lang_emb, history_frames).
        #      Apply 3D RoPE via share_tools.apply_3d_rope(P, coords).
        #   5. H = self.stage2(P, s_stage1, lang_emb).        # (B, K, d_vlm)
        #   6. action_q = Q[:, -action_horizon:, :].
        #      fused, _ = self.fusion(action_q, H, H).
        #      Q_prime  = self.fusion_norm(action_q + fused).
        #   7. action_loss = self.action_model(Q_prime, actions[:, -horizon:, :], None).
        #   8. task_logits = self.task_head(H.mean(dim=1)).
        #      aux_loss   = F.cross_entropy(task_logits, task_ids).
        #   9. Return dict with total loss + breakdown.
        raise NotImplementedError

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference path — mirrors forward but skips loss computation.

        See QwenPI.predict_action for the canonical template.
        """
        # TODO(htcs): replicate forward up through Q_prime; call
        # self.action_model.predict_action (or analogous) to roll out actions.
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _collate_codec(self, codec_list: List[dict]) -> dict:
        """Stack per-example numpy codec dicts → batched torch tensors on device."""
        # TODO(htcs):
        #   device = next(self.parameters()).device
        #   return {k: torch.from_numpy(np.stack([c[k] for c in codec_list])).to(device)
        #           for k in ('mv', 'residual', 'is_i_frame')}
        raise NotImplementedError

    def _collate_history(self, batch_images: List[List]) -> torch.Tensor:
        """Convert List[List[PIL]] → (B, T, 3, H, W) tensor, resized for ViT.

        Use ``starVLA.training.trainer_utils.trainer_tools.resize_images``.
        """
        # TODO(htcs).
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Smoke test (impl doc §9 Phase 2 checklist):
#   python starVLA/model/framework/VLM4A/HTCS.py
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # TODO(htcs): mirror QwenGR00T's main block — build a fake batch with
    # T=16, 14×14 codec, run forward once, assert loss is finite.
    raise NotImplementedError
