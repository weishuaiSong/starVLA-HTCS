"""Stage-1 codec-based patch selector (M2, direction-aware).

Combines codec saliency with top-rho selection over a 14×14 patch grid:

    s = alpha(l) * ||MV||_n + gamma(l) * (MV . v_tgt(l))_n + beta(l) * |Y-128|_n

I-frames are always retained; P-frames keep the top keep_ratio fraction of
patches. Three signals are independently percentile-95 normalised per
instance, so the language-derived weights (alpha, beta, gamma) live on a
comparable scale across batches.

Outputs:
    P:          (B, N_kept, d_p)   sparse patch tokens (from frozen ViT)
    s_kept:     (B, N_kept)        saliency scores, min-max normalised to [0,1]
    coords:     (B, N_kept, 3)     (t, h, w) for downstream 3D RoPE

Reference: HTCS impl doc §3.3.
"""

import torch
import torch.nn as nn

from .saliency_mlp import SaliencyMLP


def _percentile_normalize(
    x: torch.Tensor,
    p: float = 0.95,
    signed: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-instance percentile-p normalize (OneVision-Encoder style).

    x:        (B, ...)         batch dim first.
    signed:   False → input is non-negative (||MV||, |Y-128|); output in [0, 1].
              True  → input may be signed (MV·v_tgt); divide by percentile of
                      |x| but keep sign; output in [-1, 1].
    p:        quantile in (0, 1), default 0.95.

    Wrapped in no_grad — torch.quantile() runs a sort and is not differentiable.
    Gradients still flow through the numerator (i.e. back to v_tgt for the
    signed direction term, which is the whole point of D12).
    """
    B = x.shape[0]
    with torch.no_grad():
        # quantile in fp32 to avoid bf16 instability (impl doc §10.11).
        flat = x.float().abs().flatten(1) if signed else x.float().flatten(1)
        denom = torch.quantile(flat, p, dim=1).clamp(min=eps)
        view_shape = (B,) + (1,) * (x.ndim - 1)
        denom = denom.view(view_shape)
    out = x / denom
    return out.clamp(-1.0, 1.0) if signed else out.clamp(0.0, 1.0)


class Stage1CodecSelector(nn.Module):
    """Direction-aware codec saliency + top-ρ patch selection."""

    def __init__(
        self,
        vit_encoder: nn.Module,
        d_text: int,
        keep_ratio: float = 0.20,
        grid_size: int = 14,
        percentile: float = 0.95,
    ):
        """
        Args:
            vit_encoder: frozen ViT, borrowed from
                ``qwen_vl_interface.model.visual``. Stored as plain attribute
                (NOT a submodule) so it does not double-register with the
                outer framework's parameter list.
            d_text:      language embedding dimension.
            keep_ratio:  rho — fraction of P-frame patches to retain.
            grid_size:   G (=14 for SigLIP-Large patch grid on 224 input).
            percentile:  p for OneVision-style per-instance normalisation.
        """
        super().__init__()
        # Plain attribute → does NOT appear under self.parameters().
        # The outer framework already froze vit's parameters.
        object.__setattr__(self, "vit", vit_encoder)
        self.saliency_mlp = SaliencyMLP(d_text=d_text)
        self.keep_ratio = float(keep_ratio)
        self.G = int(grid_size)
        self.percentile = float(percentile)

        # telemetry, set in forward() — for E13 / E16 / wandb logging.
        self._last_saliency_params: dict = {}

    def _compute_saliency(
        self,
        mv: torch.Tensor,         # (B, T, G, G, 2) int8/float, 2D vector
        residual: torch.Tensor,   # (B, T, G, G)    float16/float
        alpha: torch.Tensor,      # (B,) non-negative
        beta: torch.Tensor,       # (B,) non-negative
        gamma: torch.Tensor,      # (B,) non-negative
        v_tgt: torch.Tensor,      # (B, 2) unit vector
    ) -> torch.Tensor:
        """Fuse three codec signals; returns (B, T, G, G) signed saliency."""
        mv_f       = mv.float()
        mv_norm    = mv_f.norm(dim=-1)                                       # (B,T,G,G) non-neg
        mv_aligned = (mv_f * v_tgt[:, None, None, None, :]).sum(-1)          # (B,T,G,G) signed
        res        = residual.float()                                        # (B,T,G,G) non-neg

        mv_norm_n     = _percentile_normalize(mv_norm,    self.percentile, signed=False)
        mv_aligned_n  = _percentile_normalize(mv_aligned, self.percentile, signed=True)
        res_n         = _percentile_normalize(res,        self.percentile, signed=False)

        # broadcast (B,) → (B, 1, 1, 1)
        a = alpha[:, None, None, None]
        b = beta [:, None, None, None]
        g = gamma[:, None, None, None]

        return a * mv_norm_n + g * mv_aligned_n + b * res_n                  # (B, T, G, G)

    def forward(
        self,
        codec: dict,                       # 'mv', 'residual', 'is_i_frame'
        lang_emb: torch.Tensor,            # (B, L_text, d_text)
        frames: torch.Tensor,              # (B, T, 3, H, W) — ViT-resized
    ):
        """
        codec['mv']:         (B, T, G, G, 2) int8  — 2D motion vector, NOT normed
        codec['residual']:   (B, T, G, G)    float16
        codec['is_i_frame']: (B, T)          bool

        Returns:
            P:      (B, N_kept, d_p)
            s_kept: (B, N_kept)
            coords: (B, N_kept, 3) long  — (t, h, w)
        """
        B, T = codec['mv'].shape[:2]
        G = self.G

        # 1. Language-derived parameters (D12, direction-aware).
        alpha, beta, gamma, v_tgt = self.saliency_mlp(lang_emb)

        # 2. Fuse three codec signals → (B, T, G, G).
        s = self._compute_saliency(
            codec['mv'], codec['residual'], alpha, beta, gamma, v_tgt,
        )

        # 3. Run frozen ViT on all frames once → (B, T, G*G, d_p).
        with torch.no_grad():
            patches = self.vit(frames.flatten(0, 1))                         # (B*T, G*G, d_p)
        d_p = patches.shape[-1]
        patches = patches.view(B, T, G * G, d_p)

        # 4. Top-ρ selection: I-frames always kept (+inf), then P-frame top-ρ.
        is_i = codec['is_i_frame']
        s_flat = s.view(B, T * G * G)
        patches_flat = patches.view(B, T * G * G, d_p)

        i_mask_flat = is_i.view(B, T, 1, 1).expand(-1, -1, G, G).reshape(B, -1)  # bool
        s_for_topk = torch.where(
            i_mask_flat,
            torch.full_like(s_flat, float('inf')),
            s_flat,
        )

        # We assume codec['is_i_frame'] follows the same fixed GOP pattern
        # across every example in the batch → N_kept identical per row.
        n_i_patches = int(is_i[0].sum().item()) * G * G
        n_p_patches = int((~is_i[0]).sum().item()) * G * G
        N_kept = n_i_patches + max(1, int(self.keep_ratio * n_p_patches))

        topk_vals, topk_idx = s_for_topk.topk(N_kept, dim=1)                 # (B, N_kept)
        P = torch.gather(
            patches_flat, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, d_p),
        )                                                                     # (B, N_kept, d_p)

        # 5. Min-max normalise s_kept to [0, 1]; I-frame entries (was +inf) → 1.
        is_inf = torch.isinf(topk_vals)
        # Replace inf with a finite sentinel (max of finite) before min-max.
        finite_mask = ~is_inf
        if finite_mask.any():
            # Per-row min/max over finite entries; rows with no finite values use 0/1.
            safe_vals = topk_vals.masked_fill(is_inf, float('-inf'))
            max_v = safe_vals.amax(dim=1, keepdim=True)
            safe_vals2 = topk_vals.masked_fill(is_inf, float('+inf'))
            min_v = safe_vals2.amin(dim=1, keepdim=True)
            denom = (max_v - min_v).clamp(min=1e-6)
            s_kept = (topk_vals - min_v) / denom
            s_kept = torch.where(is_inf, torch.ones_like(s_kept), s_kept)
        else:
            s_kept = torch.ones_like(topk_vals)

        # 6. coords for 3D RoPE: (t, h, w) from flat top-k indices.
        t_idx = topk_idx // (G * G)
        hw    = topk_idx %  (G * G)
        h_idx = hw // G
        w_idx = hw %  G
        coords = torch.stack([t_idx, h_idx, w_idx], dim=-1).long()           # (B, N_kept, 3)

        # 7. Telemetry — picked up by upper-framework logging (E13 / E16).
        self._last_saliency_params = {
            'alpha': alpha.detach(),
            'beta':  beta.detach(),
            'gamma': gamma.detach(),
            'v_tgt': v_tgt.detach(),
        }

        return P, s_kept, coords
