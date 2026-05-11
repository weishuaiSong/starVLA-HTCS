"""Stage-1 saliency MLP.

Maps a pooled language embedding to two positive scalar weights (alpha, beta)
that modulate the codec saliency formula::

    s = alpha(l) * ||MV|| + beta(l) * |Residual|

This is the *parameter-level* leg of HTCS hierarchical language conditioning.

Reference: HTCS impl doc §3.1.
"""

import torch
import torch.nn as nn


class SaliencyMLP(nn.Module):
    """语言 embedding → (alpha, beta), 调制 codec saliency 公式."""

    def __init__(self, d_text: int, hidden: int = 256):
        super().__init__()
        # TODO(htcs): two-layer MLP, output dim 2, Softplus to guarantee positivity.
        #   Linear(d_text, hidden) -> GELU -> Linear(hidden, 2) -> Softplus
        raise NotImplementedError

    def forward(self, lang_emb: torch.Tensor):
        """
        Args:
            lang_emb: (B, L_text, d_text) — token-level language embeddings.

        Returns:
            alpha, beta: each shaped (B, 1, 1, 1) for broadcasting against
            saliency maps of shape (B, T, G, G).
        """
        # TODO(htcs): pool over L_text (mean), apply MLP, split last dim into 2,
        # then reshape both to (B, 1, 1, 1) for broadcasting.
        raise NotImplementedError
