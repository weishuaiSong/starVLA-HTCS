"""Stage-1 codec-based patch selector (M2).

Combines codec saliency (motion-vector norm + residual energy, modulated by
SaliencyMLP(lang)) with top-rho selection over a 14x14 patch grid. I-frames
are always retained; P-frames keep the top keep_ratio fraction of patches.

Outputs:
    P:          (B, N_kept, d_p)   sparse patch tokens (sourced from frozen ViT)
    s_kept:     (B, N_kept)        saliency scores, min-max normalised to [0,1]
    coords:     (B, N_kept, 3)     (t, h, w) for downstream 3D RoPE

Reference: HTCS impl doc §3.3.
"""

import torch
import torch.nn as nn

from .saliency_mlp import SaliencyMLP


class Stage1CodecSelector(nn.Module):
    def __init__(
        self,
        vit_encoder: nn.Module,
        d_text: int,
        keep_ratio: float = 0.20,
        grid_size: int = 14,
    ):
        """
        Args:
            vit_encoder: frozen ViT, borrowed from
                ``qwen_vl_interface.model.visual``.
            d_text: language embedding dimension.
            keep_ratio: rho — fraction of P-frame patches to retain.
            grid_size: G (=14 for SigLIP-Large).
        """
        super().__init__()
        # TODO(htcs): store vit (do NOT register as submodule that will be
        # double-frozen elsewhere — caller already froze it), saliency_mlp,
        # keep_ratio, grid_size.
        raise NotImplementedError

    def forward(
        self,
        codec: dict,                       # 'mv', 'residual', 'is_i_frame'
        lang_emb: torch.Tensor,            # (B, L_text, d_text)
        frames: torch.Tensor,              # (B, T, 3, H, W) — ViT-resized
    ):
        """
        codec['mv']:         (B, T, G, G, 2) int8
        codec['residual']:   (B, T, G, G)    float16
        codec['is_i_frame']: (B, T)          bool

        Returns:
            P:      (B, N_kept, d_p)
            s_kept: (B, N_kept)
            coords: (B, N_kept, 3)  — (t, h, w) ints
        """
        # TODO(htcs):
        #   1. alpha, beta = self.saliency_mlp(lang_emb).
        #   2. s = alpha * ||MV|| + beta * residual  → (B, T, G, G).
        #   3. with no_grad: patches = self.vit(frames.flatten(0, 1)) →
        #      reshape to (B, T, G*G, d_p).
        #   4. Force I-frame patch saliency to +inf so top-k always keeps them;
        #      compute N_kept = (#I-frame patches) + keep_ratio * (#P-frame patches).
        #   5. topk along T*G*G to get indices; gather P from patches_flat.
        #   6. s_kept = min-max normalise (I-frame entries → 1.0).
        #   7. derive coords = (t_idx, h_idx, w_idx) from flat topk indices.
        raise NotImplementedError
