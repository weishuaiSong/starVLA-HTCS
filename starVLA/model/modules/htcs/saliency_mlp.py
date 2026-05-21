"""Stage-1 saliency MLP (direction-aware, 5-output).

Maps a pooled language embedding to FIVE outputs that modulate the codec
saliency formula (impl doc §3.1, D12)::

    s = alpha(l) * ||MV||_n + gamma(l) * (MV . v_tgt(l))_n + beta(l) * |Y-128|_n

Outputs:
    alpha, beta, gamma : (B,)   non-negative scalars (Softplus)
    v_tgt              : (B, 2) unit vector — task-expected motion direction

This is the *parameter-level* leg of HTCS hierarchical language conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyMLP(nn.Module):
    """语言 embedding → (alpha, beta, gamma, v_tgt), 调制 codec saliency 公式."""

    def __init__(self, d_text: int, hidden: int = 256):
        super().__init__()
        # 5 outputs: alpha, beta, gamma, v_x, v_y
        self.net = nn.Sequential(
            nn.Linear(d_text, hidden),
            nn.GELU(),
            nn.Linear(hidden, 5),
        )

    def forward(self, lang_emb: torch.Tensor):
        """
        Args:
            lang_emb: (B, L_text, d_text) — token-level language embeddings.

        Returns:
            alpha: (B,)   non-negative — weights ||MV||
            beta:  (B,)   non-negative — weights |Y-128|
            gamma: (B,)   non-negative — weights (MV . v_tgt)
            v_tgt: (B, 2) unit vector — task-expected motion direction
        """
        out = self.net(lang_emb.mean(dim=1))            # (B, 5)
        abg = F.softplus(out[..., 0:3])                 # (B, 3) non-negative
        alpha = abg[..., 0]                             # (B,)
        beta  = abg[..., 1]                             # (B,)
        gamma = abg[..., 2]                             # (B,)
        v_raw = out[..., 3:5]                           # (B, 2)
        # Unit-normalise. Avoid ``v_raw.norm()`` whose gradient is
        # ``v_raw / ||v_raw||`` and evaluates to NaN at zero. Use
        # ``sqrt(sum + eps)`` so the gradient is well-defined everywhere.
        v_norm = (v_raw.pow(2).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
        v_tgt = v_raw / v_norm.clamp_min(1e-6)
        return alpha, beta, gamma, v_tgt
