"""Stage-2 language-conditioned slot compressor (M3).

Three-step pipeline (impl doc §3.4):
    14.1  K learnable queries cross-attend to language tokens →
          language-conditioned slots.
    14.2  Competitive slot attention compresses Stage-1 patches to K
          summary tokens (softmax along slot axis, then re-normalised along
          patch axis).
    14.3  Stage-1 saliency is injected as an additive bias in the attention
          logits.

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
        self.K = int(K)
        # Learnable queries, initialised small so q_from_lang dominates early.
        self.learnable_q = nn.Parameter(torch.randn(self.K, d_text) * 0.02)
        # 14.1 — pull language context into slot queries.
        self.q_from_lang = nn.MultiheadAttention(
            d_text, n_heads, batch_first=True,
        )
        self.proj_to_patch = nn.Linear(d_text, d_patch)
        # 14.2 + 14.3 — competitive softmax with saliency bias.
        self.cross_attn = CompetitiveSlotAttention(d_patch, n_heads)
        self.proj_to_vlm = nn.Linear(d_patch, d_vlm)
        self.norm = nn.LayerNorm(d_vlm)

    def forward(
        self,
        P: torch.Tensor,           # (B, N_kept, d_patch) — Stage-1 output
        s_stage1: torch.Tensor,    # (B, N_kept)          — Stage-1 saliency
        lang_emb: torch.Tensor,    # (B, L_text, d_text)
    ) -> torch.Tensor:
        """Returns H: (B, K, d_vlm)."""
        B = P.shape[0]
        q = self.learnable_q.unsqueeze(0).expand(B, -1, -1)                  # (B, K, d_text)
        slots, _ = self.q_from_lang(q, lang_emb, lang_emb)                   # 14.1
        slots = self.proj_to_patch(slots)                                    # (B, K, d_patch)
        H_raw = self.cross_attn(slots, P, saliency_bias=s_stage1)            # 14.2 + 14.3
        return self.norm(self.proj_to_vlm(H_raw))                            # (B, K, d_vlm)
