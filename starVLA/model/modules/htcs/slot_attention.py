"""Competitive slot attention.

Cross-attention from K learnable + language-conditioned slots to N patch
tokens, with two HTCS-specific tweaks (impl doc §3.2):

* 14.2  softmax normalised along the *slot* axis first, then re-normalised
        along the patch axis so each slot's weights sum to 1 — slots compete
        for patches.
* 14.3  optional saliency bias from Stage 1, additive to the attention
        logits, shaped (B, N).

The softmax must be forced to fp32 under bf16/DeepSpeed ZeRO-2 to avoid the
NaN failure mode documented in impl doc §10.4.
"""

import torch
import torch.nn as nn


class CompetitiveSlotAttention(nn.Module):
    """14.2 competitive softmax + 14.3 Stage1 saliency bias."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        # TODO(htcs): q_proj / k_proj / v_proj / out_proj as nn.Linear(d_model, d_model).
        # Store n_heads, head_dim, scale = head_dim ** -0.5.
        raise NotImplementedError

    def forward(
        self,
        slots: torch.Tensor,           # (B, K, d_model)
        patches: torch.Tensor,         # (B, N, d_model)
        saliency_bias: torch.Tensor | None = None,  # (B, N), optional
        bias_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns:
            updated slots: (B, K, d_model).
        """
        # TODO(htcs):
        #   1. project Q, K, V; reshape to (B, H, *, head_dim).
        #   2. logits = einsum('bhkd,bhnd->bhkn', Q, K) * scale.
        #   3. if saliency_bias is not None: logits += bias_scale * bias[:, None, None, :].
        #   4. with autocast(enabled=False):
        #        attn = softmax(logits.float(), dim=2)  # along K — competitive
        #        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # re-normalise along N
        #   5. out = einsum('bhkn,bhnd->bhkd', attn.to(V.dtype), V).
        #   6. reshape, out_proj.
        raise NotImplementedError
