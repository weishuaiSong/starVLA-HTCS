"""Vision adapter for Qwen3.5-* models.

What Qwen3.5's vision tower actually wants (vs. what Stage1 used to assume):

* input: a flat ``(N, in_ch * tps * ps * ps)`` patch matrix plus a
  ``grid_thw`` table, not raw ``(B*T, 3, H, W)`` pixels;
* normalisation: ``mean = std = 0.5`` (per ``preprocessor_config.json``),
  NOT CLIP's [0.48..,0.45..,0.40..] / [0.27..,0.26..,0.28..];
* the spatial *merger* (2×2 → 1) and the temporal patch-embed conv
  (stride=2 on T) compress the output for the LLM. Stage1 wants the
  *pre-merge* 14×14 grid, which lives in ``out.last_hidden_state`` at
  ``hidden_size`` channels (e.g. 768 for Qwen3.5-0.8B), not the
  ``pooler_output`` that's already merged to 7×7 at ``out_hidden_size``.

To keep T-resolution at 16 instead of 16 / temporal_patch_size = 8, each
history frame is treated as its **own** image item (``grid_thw = (1, G, G)``),
with the frame duplicated to fill the temporal_patch_size=2 window. This
costs nothing — the 3D conv on the duplicate just halves its own
temporal stride.

See ``transformers/models/qwen3_5/modeling_qwen3_5.py::Qwen3_5VisionModel``
for ground-truth source. Key file lines (transformers 4.x):
* ``PatchEmbed``: Conv3d kernel/stride = (tps, ps, ps)
* ``forward``: returns ``last_hidden_state`` (pre-merge) and
  ``pooler_output`` (post-merge, post-projection)
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

from . import register_vision_adapter
from .base import BaseVisionAdapter


@register_vision_adapter("qwen3_5")
class Qwen3_5VisionAdapter(BaseVisionAdapter):
    """Wraps Qwen3.5's nested ``hf_model.model.visual`` ViT for Stage1."""

    # ----- detection ------------------------------------------------- #

    @classmethod
    def can_handle(cls, hf_model) -> bool:
        # Defensive: never raise on unrelated models. The factory swallows
        # exceptions from us regardless, but keep it cheap.
        cfg = getattr(hf_model, "config", None)
        if cfg is None:
            return False

        # model_type is the most stable signal across class renames.
        if str(getattr(cfg, "model_type", "")).startswith("qwen3_5"):
            return True

        # Wrapper / fine-tune class names sometimes drift; fall back to
        # the structural pattern Qwen3.5 specifically exhibits:
        #   hf_model.model.visual exists  AND
        #   cfg.vision_config.spatial_merge_size is defined AND
        #   cfg.vision_config.temporal_patch_size is defined
        nested_visual = (
            hasattr(hf_model, "model")
            and hasattr(getattr(hf_model, "model", None), "visual")
        )
        vc = getattr(cfg, "vision_config", None)
        return (
            nested_visual
            and vc is not None
            and getattr(vc, "spatial_merge_size", None) is not None
            and getattr(vc, "temporal_patch_size", None) is not None
            and getattr(vc, "patch_size", None) is not None
            and "qwen" in str(type(hf_model).__name__).lower()
        )

    # ----- ctor ------------------------------------------------------ #

    def __init__(self, hf_model, grid_size: int = 14, freeze: bool = True):
        super().__init__()
        cfg = hf_model.config
        vc = cfg.vision_config

        # Held as a *plain* attribute, not an ``nn.Module`` child — the
        # outer framework still owns ``qwen_vl_interface.model.model.visual``
        # via the language backbone, and we must NOT register the ViT a
        # second time (DeepSpeed bucketing + optimizer param count would
        # both go wrong).
        object.__setattr__(self, "_vit", hf_model.model.visual)

        self._patch_size: int = int(vc.patch_size)                       # 16
        self._tps: int = int(vc.temporal_patch_size)                     # 2
        self._merge_size: int = int(vc.spatial_merge_size)               # 2
        self._d_patch: int = int(vc.hidden_size)                         # 768 pre-merge
        self._grid_size: int = int(grid_size)                            # 14 by default
        # Input image side that produces a (grid_size × grid_size) grid
        # at this ViT's patch size. 14 × 16 = 224 for the default config.
        self._image_size: int = self._grid_size * self._patch_size

        if self._grid_size % self._merge_size != 0:
            raise ValueError(
                f"grid_size={self._grid_size} must be divisible by "
                f"spatial_merge_size={self._merge_size} — Qwen3.5 packs "
                "patches in (merge_block × intra_block) order, which "
                "requires a clean split."
            )

        # Mean/std live on the *image processor*, not on the main config.
        # Default to (0.5, 0.5, 0.5) which is what Qwen2VLImageProcessor's
        # preprocessor_config.json sets in every released Qwen3.5 / Qwen3-VL
        # checkpoint we've seen — and what the ViT was trained with.
        self._mean = (0.5, 0.5, 0.5)
        self._std = (0.5, 0.5, 0.5)

        if freeze:
            self.freeze()

    # ----- properties ------------------------------------------------ #

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def d_patch(self) -> int:
        return self._d_patch

    # ----- preprocess ------------------------------------------------ #

    def preprocess(
        self,
        images: Union[List[List], np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Normalise frames to Qwen3.5's training distribution.

        Returns ``(B, T, 3, image_size, image_size)`` float32 on CPU.
        Stage1 will move it to the ViT's device + dtype right before
        encoding.
        """
        # 1. Lift everything to a (B, T, 3, H, W) float32 CPU tensor.
        if isinstance(images, torch.Tensor):
            if images.ndim != 5:
                raise ValueError(
                    f"Tensor input must be 5-D; got shape {tuple(images.shape)}."
                )
            if images.shape[-1] == 3 and images.shape[2] != 3:
                # (B, T, H, W, 3) -> (B, T, 3, H, W)
                images = images.permute(0, 1, 4, 2, 3).contiguous()
            x = images.float()
        elif isinstance(images, np.ndarray):
            if images.ndim != 5 or images.shape[-1] != 3:
                raise ValueError(
                    f"ndarray input must be (B, T, H, W, 3); "
                    f"got shape {images.shape}."
                )
            x = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float()
        else:
            # List[List[PIL.Image | ndarray]] — Stage1's normal path.
            x = _stack_pil_history(images)                            # (B,T,3,H,W) float32 0-255

        # 2. Resize to (image_size, image_size) per frame if needed.
        if x.shape[-1] != self._image_size or x.shape[-2] != self._image_size:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)                                          # (B*T, 3, H, W)
            x = F.interpolate(
                x, size=(self._image_size, self._image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
            x = x.view(B, T, 3, self._image_size, self._image_size)

        # 3. uint8-style scale → 0..1 if it looks like uint8.
        if x.max() > 1.5:
            x = x / 255.0

        # 4. Qwen3.5 normalisation: (x - 0.5) / 0.5 in default config.
        mean = torch.tensor(self._mean, dtype=x.dtype).view(1, 1, 3, 1, 1)
        std  = torch.tensor(self._std,  dtype=x.dtype).view(1, 1, 3, 1, 1)
        x = (x - mean) / std

        return x

    # ----- encode ---------------------------------------------------- #

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Run Qwen3.5's frozen vision tower on per-frame items.

        Treats every (B, T) frame as its own image item with
        ``grid_thw = (1, G, G)`` so the temporal_patch_size=2 conv only
        compresses the duplicated frame — T resolution stays at T.

        Args:
            frames: ``(B, T, 3, image_size, image_size)`` float, on any
                device. We move it to the ViT's device + dtype.

        Returns:
            ``(B, T, G * G, d_patch)``.
        """
        if frames.ndim != 5:
            raise ValueError(
                f"encode() expects (B, T, 3, H, W); got {tuple(frames.shape)}."
            )

        vit = self._vit
        device = next(vit.parameters()).device
        dtype = next(vit.parameters()).dtype

        B, T, C, H, W = frames.shape
        G = self._grid_size
        ps = self._patch_size
        tps = self._tps

        if H != G * ps or W != G * ps:
            raise ValueError(
                f"Frame size {H}x{W} does not match adapter "
                f"image_size={self._image_size}={G}x{ps} on each side. "
                f"Did you skip preprocess()?"
            )

        x = frames.to(device=device, dtype=dtype, non_blocking=True)
        ms = self._merge_size
        Gb = G // ms                                                     # blocks per side

        # ---- pack to Qwen-VL's "(sum_N, in_ch * tps * ps * ps)" layout ----
        #
        # Qwen3.5's vision tower stores per-patch positional information
        # in *block-major* order:
        #   token i ↔ full position (bh*ms+ir, bw*ms+ic) with
        #     i = bh * (Gb * ms * ms) + bw * (ms * ms) + ir * ms + ic
        # See ``modeling_qwen3_5.py::fast_pos_embed_interpolate`` (the
        # ``.view(t, h/ms, ms, w/ms, ms, -1).permute(0,1,3,2,4,5)`` step)
        # and ``rot_pos_emb`` (``.expand(merged_h, merged_w, ms, ms)``).
        #
        # PatchEmbed itself just does ``view(-1, C, tps, ps, ps)`` so the
        # per-patch flat layout is (C, tps, ps, ps) in C-major order.
        #
        # So we have to:
        #   1. split (H, W) into (Gb, ms, Gb, ms) patches in (block_row,
        #      intra_row, block_col, intra_col) order,
        #   2. permute to block-major (Gb, Gb, ms, ms) order,
        #   3. duplicate the still frame across the temporal axis,
        #   4. reshape to (B*T*G*G, C * tps * ps * ps).

        # (B, T, C, Gb, ms, ps, Gb, ms, ps)
        x = x.view(B, T, C, Gb, ms, ps, Gb, ms, ps)
        # Permute to: (B, T, block_row, block_col, intra_row, intra_col, C, ps, ps)
        # source dims:        0  1  2   3   4   5   6   7   8
        #                     B  T  C   bh  ir  ph  bw  ic  pw
        # target dims order:  0  1  3   6   4   7   2   5   8
        x = x.permute(0, 1, 3, 6, 4, 7, 2, 5, 8).contiguous()
        # → (B, T, Gb, Gb, ms, ms, C, ps, ps)
        # Insert temporal axis (duplicate still frame to fill tps):
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, tps, -1, -1, -1).contiguous()
        # → (B, T, Gb, Gb, ms, ms, tps, C, ps, ps)
        # Re-order per-patch content to PatchEmbed's expected (C, tps, ps, ps):
        # source dims:        0  1  2   3   4   5   6   7  8   9
        #                     B  T  bh  bw  ir  ic  tps C  ps  ps
        # target dims order:  0  1  2   3   4   5   7   6  8   9
        x = x.permute(0, 1, 2, 3, 4, 5, 7, 6, 8, 9).contiguous()
        # Flatten: (B*T*G*G, C * tps * ps * ps)
        pixel_values = x.view(B * T * G * G, C * tps * ps * ps)

        # ---- grid_thw: one item per frame ----
        grid_thw = torch.tensor(
            [[1, G, G]] * (B * T), device=device, dtype=torch.long,
        )

        # ---- forward (no grad — adapter is frozen) ----
        with torch.no_grad():
            out = vit(pixel_values, grid_thw)
        hidden = out.last_hidden_state                                   # (B*T*G*G, d_patch)
        if hidden.shape != (B * T * G * G, self._d_patch):
            raise RuntimeError(
                f"unexpected Qwen3.5 ViT output shape {tuple(hidden.shape)}; "
                f"expected ({B * T * G * G}, {self._d_patch})."
            )

        # ---- unscramble block-major → row-major (Stage1 / codec aligned) ----
        # hidden[i] for item k is the token at block-major position. We
        # reverse the same (Gb, Gb, ms, ms) permute used during input
        # packing so Stage1 sees patches in raw (h, w) row-major order
        # — that's what the codec saliency map at (T, G, G) expects.
        hidden = hidden.view(B, T, Gb, Gb, ms, ms, self._d_patch)
        # block-major (bh, bw, ir, ic) -> row-major (bh, ir, bw, ic)
        hidden = hidden.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return hidden.view(B, T, G * G, self._d_patch)


# ---------------------------------------------------------------------- #
#  helpers
# ---------------------------------------------------------------------- #
def _stack_pil_history(images: List[List]) -> torch.Tensor:
    """``List[List[PIL.Image | ndarray]]`` → ``(B, T, 3, H, W)`` float32 0..255."""
    from PIL import Image as _PIL

    B = len(images)
    if B == 0:
        raise ValueError("preprocess() got an empty batch.")
    T = len(images[0])

    arrs = []
    for b, history in enumerate(images):
        if len(history) != T:
            raise ValueError(
                f"All samples must share the same T; sample {b} has "
                f"{len(history)} frames vs. {T} in sample 0."
            )
        frame_arrs = []
        for img in history:
            if isinstance(img, _PIL.Image):
                a = np.asarray(img.convert("RGB"), dtype=np.uint8)
            elif isinstance(img, np.ndarray):
                a = img.astype(np.uint8, copy=False)
            else:
                raise TypeError(
                    f"Unsupported frame type {type(img).__name__}; "
                    "expected PIL.Image or np.ndarray."
                )
            frame_arrs.append(a)                                          # (H, W, 3)
        arrs.append(np.stack(frame_arrs, axis=0))                         # (T, H, W, 3)
    arr = np.stack(arrs, axis=0)                                          # (B, T, H, W, 3)
    return torch.from_numpy(arr).permute(0, 1, 4, 2, 3).float()           # (B, T, 3, H, W)
