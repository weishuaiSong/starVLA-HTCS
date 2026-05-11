"""Stage-2 language-conditioned slot compressor (M3).

Three-step pipeline (impl doc §3.4):
    14.1  K learnable queries cross-attend to language tokens → language-conditioned slots.
    14.2  Competitive slot attention compresses Stage-1 patches to K summary tokens.
    14.3  Stage-1 saliency is injected as an additive bias in the attention logits.

Output H ∈ (B, K, d_vlm) is fused with VLM action queries downstream (M4).
"""

import torch
import torch.nn as nn

from .slot_attention import CompetitiveSlotAttention


class Stage2LangCompressor(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_patch: int,
        d_vlm: int,
        K: int = 8,
        n_heads: int = 4,
    ):
        super().__init__()
        # TODO(htcs):
        #   - self.K = K
        #   - learnable_q: nn.Parameter(randn(K, d_text) * 0.02)
        #   - q_from_lang: nn.MultiheadAttention(d_text, n_heads, batch_first=True) [14.1]
        #   - proj_to_patch: Linear(d_text, d_patch)
        #   - cross_attn: CompetitiveSlotAttention(d_patch, n_heads)        [14.2 + 14.3]
        #   - proj_to_vlm: Linear(d_patch, d_vlm)
        #   - norm: LayerNorm(d_vlm)
        raise NotImplementedError

    def forward(
        self,
        P: torch.Tensor,           # (B, N_kept, d_patch)  — Stage-1 output
        s_stage1: torch.Tensor,    # (B, N_kept)           — Stage-1 saliency
        lang_emb: torch.Tensor,    # (B, L_text, d_text)
    ) -> torch.Tensor:
        """Returns H: (B, K, d_vlm)."""
        # TODO(htcs):
        #   1. expand learnable_q to (B, K, d_text).
        #   2. slots, _ = q_from_lang(q, lang_emb, lang_emb).
        #   3. slots = proj_to_patch(slots).
        #   4. H_raw = cross_attn(slots, P, saliency_bias=s_stage1).
        #   5. return norm(proj_to_vlm(H_raw)).
        raise NotImplementedError
