"""Base contract for HTCS vision-tower adapters.

Stage1 sees only this interface — it doesn't know whether the underlying
ViT is SigLIP, Qwen3-VL, Qwen3.5, InternVL, or anything else. Each
backbone-specific subclass handles:

* what input layout the ViT wants (raw pixels vs. packed patches, with or
  without ``grid_thw``);
* which mean / std the ViT was trained with;
* how to recover **pre-merge** patch features (HTCS Stage1 needs them at
  the codec grid resolution, not whatever the VLM emits to the LLM);
* whether the model exposes a 14×14 grid natively, or has to be coaxed
  into one.

Adapters are picked by ``build_vision_adapter`` (see ``__init__.py``).
The registry uses each adapter's ``can_handle(hf_model)`` classmethod
rather than a hard ``type().__name__`` switch, so transformers class
renames and PEFT / DDP wrappers don't silently break detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


class BaseVisionAdapter(nn.Module, ABC):
    """Backbone-agnostic vision-tower wrapper for HTCS Stage1.

    Subclasses implement four things:

    1. ``can_handle(cls, hf_model) -> bool``: structural-feature detector
       used by the factory. Must not raise on unrelated models.
    2. ``grid_size`` property: per-frame patch grid side (e.g. 14).
       Stage1 + codec preprocessing both consume this — the codec parquet
       MV/residual maps are sized to (T, grid_size, grid_size, …).
    3. ``d_patch`` property: per-patch feature dimension Stage1/Stage2
       will consume downstream.
    4. ``preprocess(images)`` + ``encode(frames)``: two-step pipeline so
       online callers (e.g. RollingCodecEncoder) can pre-build the
       float tensor once and reuse it across forwards.

    All adapters expect frozen-by-default visual towers; turn off
    ``freeze`` only when you know what you're doing.
    """

    # ----- contract ---------------------------------------------------- #

    @classmethod
    @abstractmethod
    def can_handle(cls, hf_model) -> bool:
        """Return True iff this adapter knows how to drive ``hf_model``.

        Implementations should be **defensive**: never raise on models
        they don't recognise — the factory calls every registered adapter
        in turn and picks the unique match.

        Recommended pattern: combine multiple cheap structural checks
        (config.model_type, presence of nested ``visual`` module, vision
        config attributes, …) with ``or``, so renaming the HF wrapper
        class or wrapping it in PEFT/DDP doesn't kill detection.
        """

    @property
    @abstractmethod
    def grid_size(self) -> int:
        """Per-frame patch-grid side (Stage1 + codec parquet use this)."""

    @property
    @abstractmethod
    def d_patch(self) -> int:
        """Per-patch feature dim emitted by ``encode``."""

    @abstractmethod
    def preprocess(
        self,
        images: Union[List[List], np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Convert raw frames to a model-specific float tensor.

        Accepted inputs (Stage1 / RollingCodecEncoder both pass one of
        these):

        * ``List[List[PIL.Image | np.ndarray]]`` — batched temporal history,
          outer = batch, inner = T frames per sample;
        * ``np.ndarray`` of shape (B, T, H, W, 3) uint8;
        * ``torch.Tensor`` of shape (B, T, H, W, 3) uint8 or
          (B, T, 3, H, W) float — passed through as appropriate.

        Output: ``(B, T, 3, H_in, W_in)`` float tensor normalised the way
        this adapter's ViT was trained. ``H_in``/``W_in`` are whatever
        the adapter wants — Stage1 doesn't care.
        """

    @abstractmethod
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Run the (frozen) ViT and return per-patch features.

        Args:
            frames: ``(B, T, 3, H_in, W_in)`` as returned by ``preprocess``.

        Returns:
            ``(B, T, grid_size * grid_size, d_patch)``.
        """

    # ----- conveniences -------------------------------------------------- #

    def forward(
        self,
        images: Union[List[List], np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """One-shot ``preprocess + encode``. Convenient for one-off calls."""
        return self.encode(self.preprocess(images))

    def freeze(self) -> "BaseVisionAdapter":
        """Set ``requires_grad = False`` on every parameter and switch to eval.

        Subclasses commonly hold the underlying ViT as a *plain Python
        attribute* (via ``object.__setattr__``) to avoid double-registering
        it with the outer framework — that means ``self.parameters()``
        does **not** see those weights, and a naive ``for p in
        self.parameters(): p.requires_grad = False`` silently freezes
        nothing.

        We walk both ``self.parameters()`` and any plain-attribute
        ``nn.Module`` named with conventional prefixes (``_vit``,
        ``_backbone``, ``_encoder``) so subclass authors only need to
        store the ViT under one of those names to get correct freezing
        for free. If subclasses use a different attribute name they can
        override ``freeze()`` directly.
        """
        for p in self.parameters():
            p.requires_grad = False
        for attr in ('_vit', '_backbone', '_encoder'):
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = False
                mod.eval()
        self.eval()
        return self
