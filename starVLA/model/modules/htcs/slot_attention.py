"""Competitive slot attention.

Cross-attention from K learnable + language-conditioned slots to N patch
tokens, with two HTCS-specific tweaks (impl doc §3.2):

* 14.2  softmax normalised along the *slot* axis first, then re-normalised
        along the patch axis so each slot's weights sum to 1 — slots compete
        for patches.
* 14.3  optional saliency bias from Stage 1, additive to the attention
        logits, shaped (B, N).

The softmax is forced to fp32 under bf16/DeepSpeed ZeRO-2 to avoid the
NaN failure mode documented in impl doc §10.4.
"""

from typing import Optional

import torch
import torch.nn as nn


class CompetitiveSlotAttention(nn.Module):
    """14.2 competitive softmax + 14.3 Stage1 saliency bias."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        slots: torch.Tensor,                          # (B, K, d_model)
        patches: torch.Tensor,                        # (B, N, d_model)
        saliency_bias: Optional[torch.Tensor] = None, # (B, N), optional
        bias_scale: float = 1.0,
    ) -> torch.Tensor:
        """Returns updated slots: (B, K, d_model)."""
        B, K, d = slots.shape
        N = patches.shape[1]
        H, dh = self.n_heads, self.head_dim

        Q = self.q_proj(slots).view(B, K, H, dh).transpose(1, 2)        # (B, H, K, dh)
        Kk = self.k_proj(patches).view(B, N, H, dh).transpose(1, 2)     # (B, H, N, dh)
        V = self.v_proj(patches).view(B, N, H, dh).transpose(1, 2)      # (B, H, N, dh)

        logits = torch.einsum('bhkd,bhnd->bhkn', Q, Kk) * self.scale    # (B, H, K, N)
        if saliency_bias is not None:
            logits = logits + bias_scale * saliency_bias[:, None, None, :]

        # 14.2 — must run in fp32 to avoid bf16 NaN under DeepSpeed ZeRO-2 (§10.4).
        with torch.cuda.amp.autocast(enabled=False):
            attn = logits.float().softmax(dim=2)                        # competitive — along K
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)           # re-normalise along N

        out = torch.einsum('bhkn,bhnd->bhkd', attn.to(V.dtype), V)      # (B, H, K, dh)
        out = out.transpose(1, 2).contiguous().view(B, K, d)
        return self.out_proj(out)
