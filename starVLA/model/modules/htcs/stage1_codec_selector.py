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
        vision_adapter,
        d_text: int,
        keep_ratio: float = 0.20,
        percentile: float = 0.95,
    ):
        """
        Args:
            vision_adapter: A ``BaseVisionAdapter`` instance wrapping the
                frozen ViT. Stage1 reads ``grid_size`` / ``d_patch`` off
                it and calls ``.encode(frames)`` to get per-patch features
                in a row-major ``(B, T, G*G, d_patch)`` layout. The
                adapter is stored as a plain attribute (NOT registered as
                a child module) so the underlying ViT — which lives
                inside ``qwen_vl_interface`` too — is not double-counted
                in ``self.parameters()`` / DeepSpeed bucketing.
            d_text:      language embedding dimension.
            keep_ratio:  rho — fraction of P-frame patches to retain.
            percentile:  p for OneVision-style per-instance normalisation.
        """
        super().__init__()
        # Plain attribute → does NOT appear under self.parameters().
        # The adapter must already have frozen its inner ViT.
        object.__setattr__(self, "vision_adapter", vision_adapter)
        self.saliency_mlp = SaliencyMLP(d_text=d_text)
        self.keep_ratio = float(keep_ratio)
        # Pull G off the adapter so codec / Stage1 / ViT stay aligned by
        # construction; no chance of a yaml-wired ``grid_size: 14`` clashing
        # with a backbone that natively emits a different grid.
        self.G = int(vision_adapter.grid_size)
        self.d_patch = int(vision_adapter.d_patch)
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
        mv_f = mv.float()
        # ``mv_f.norm(dim=-1)`` is NOT safe at zero: its gradient is
        # ``mv_f / ||mv_f||`` which evaluates to ``0/0 = NaN`` whenever a
        # cell has zero motion. I-frames are always all-zero MV (see
        # codec_preprocess.py:141), so this fires on every step and
        # poisons SaliencyMLP grads. Use the (squared-sum + eps).sqrt()
        # form, which is bit-identical away from zero and well-defined
        # there.
        mv_norm = (mv_f.pow(2).sum(-1) + 1e-12).sqrt()                       # (B,T,G,G) non-neg
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

        # Codec parquet's grid (from offline preprocess) must match the
        # adapter's patch grid (and therefore Stage1's). A mismatch means
        # the YAML's ``stage1.grid_size`` was changed without re-running
        # ``codec_preprocess.py`` — fail loudly here instead of letting
        # the saliency multiply blow up with a less informative shape error.
        codec_g = codec['mv'].shape[2]
        if codec_g != G:
            raise ValueError(
                f"codec grid_size ({codec_g}) != Stage1 / adapter grid_size "
                f"({G}). Re-run codec_preprocess.py with grid_size={G}."
            )

        # 1. Language-derived parameters (D12, direction-aware).
        alpha, beta, gamma, v_tgt = self.saliency_mlp(lang_emb)

        # 2. Fuse three codec signals → (B, T, G, G).
        s = self._compute_saliency(
            codec['mv'], codec['residual'], alpha, beta, gamma, v_tgt,
        )

        # 3. Run frozen ViT via the adapter → (B, T, G*G, d_p).
        # Adapter normalises + reshapes per its backbone's API; Stage1
        # never touches model-specific details. ``encode`` is expected
        # to be no-grad-friendly (every released adapter wraps the call
        # in ``torch.no_grad()`` because the ViT is frozen).
        patches = self.vision_adapter.encode(frames)                         # (B, T, G*G, d_p)
        d_p = patches.shape[-1]

        # 4. Top-ρ selection: I-frames always kept, then P-frame top-ρ.
        #
        # An earlier version used ``float('inf')`` as the I-frame sentinel
        # to force I-frames into top-k. Forward worked, but **backward
        # poisoned the SaliencyMLP gradients**: the min-max normalisation
        # downstream computes ``d/d(denom) ∝ -(topk_vals - min_v) / denom²``
        # which evaluates to ``-inf`` at I-frame entries; multiplied by the
        # downstream ``where(is_inf, ones, …)``-masked grad of ``0`` it
        # produces ``-inf × 0 = NaN`` (PyTorch autograd does not short-circuit
        # multiplications). The NaN then floods into SaliencyMLP via the
        # saliency map and via ``clip_grad_norm_``'s global norm.
        #
        # Fix: use a *finite* sentinel above the realistic ``s`` range.
        # Forward semantics are identical (I-frames still get the topmost
        # rank); backward stays finite because no inf enters the graph.
        is_i = codec['is_i_frame']
        s_flat = s.view(B, T * G * G)
        patches_flat = patches.view(B, T * G * G, d_p)

        i_mask_flat = is_i.view(B, T, 1, 1).expand(-1, -1, G, G).reshape(B, -1)  # bool

        # Finite I-frame sentinel: strictly larger than the largest realistic
        # |s| in the batch. Detached so its grad never flows.
        with torch.no_grad():
            sentinel_scalar = s_flat.detach().abs().max() + 1.0
        s_for_topk = torch.where(
            i_mask_flat,
            sentinel_scalar.expand_as(s_flat),
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

        # Recover which of the picked positions were I-frame (robust to the
        # sentinel value choice — works for any finite sentinel above the
        # realistic saliency range).
        is_i_at_topk = torch.gather(i_mask_flat, 1, topk_idx)                # (B, N_kept) bool

        # 5. Min-max normalise s_kept to [0, 1] over **P-frame entries only**;
        # I-frame entries are pinned to exactly 1.0 below.
        # Compute bounds in no_grad — they're statistics, not signal —
        # so a degenerate batch (no P-frames, or constant saliency) cannot
        # backprop a div-by-zero or 0×∞ path.
        with torch.no_grad():
            tv_d = topk_vals.detach()
            max_p = tv_d.masked_fill(is_i_at_topk, float('-inf')).amax(dim=1, keepdim=True)
            min_p = tv_d.masked_fill(is_i_at_topk, float('+inf')).amin(dim=1, keepdim=True)
            # If a row has no P-frame entries, both ±inf survive → guard.
            no_p = torch.isinf(max_p) | torch.isinf(min_p)
            max_p = torch.where(no_p, torch.zeros_like(max_p), max_p)
            min_p = torch.where(no_p, torch.zeros_like(min_p), min_p)
            denom = (max_p - min_p).clamp(min=1e-6)

        # Only ``topk_vals`` has grad here; ``min_p`` / ``denom`` are detached.
        s_kept = ((topk_vals - min_p) / denom).clamp(0.0, 1.0)
        # Pin I-frame entries to exactly 1.0.
        s_kept = torch.where(is_i_at_topk, torch.ones_like(s_kept), s_kept)

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
