# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


class _HTCSStatefulAdapter:
    """Server-side state for HTCS inference.

    The eval client (``model2libero_interface.py``) sends one fresh observation
    per step with no temporal context, but HTCS needs:

    * a T=16-frame primary-view image history (oldest -> newest), and
    * a per-step codec window from a rolling HEVC encoder/decoder pair.

    This adapter maintains both, resetting whenever ``example['lang']`` changes
    -- which the eval client uses as the implicit episode boundary signal
    (see ``ModelClient.step``: ``if task_description != self.task_description:
    self.reset(...)``).

    Single-environment assumption: each policy_server process handles one
    rollout at a time (LIBERO eval launches one server per CUDA_VISIBLE_DEVICES).
    Batched inference with mixed episodes would need one adapter per slot.

    The adapter exposes only ``predict_action`` -- ``PolicyServerWrapper`` is
    the only caller and treats the wrapped object as a duck-typed framework.
    """

    def __init__(
        self,
        framework,
        history_len: int = 16,
        grid_size: int = 14,
        frame_h: int = 224,
        frame_w: int = 224,
    ) -> None:
        from starVLA.model.modules.htcs.rolling_codec import RollingCodecEncoder

        self._framework = framework
        self.T = int(history_len)
        self.G = int(grid_size)
        self.H = int(frame_h)
        self.W = int(frame_w)

        self._image_buf: deque = deque(maxlen=self.T)
        self._codec = RollingCodecEncoder(
            history_len=self.T,
            grid_size=self.G,
            frame_h=self.H,
            frame_w=self.W,
        )
        self._codec.reset()
        self._last_lang: Optional[str] = None

    def _reset(self, lang: Optional[str]) -> None:
        self._image_buf.clear()
        self._codec.reset()
        self._last_lang = lang

    def _build_t_frame_history(self) -> List[Image.Image]:
        """Pad the image deque to T frames by repeating the oldest entry.

        Mirrors training-side ``np.maximum(step_indices, 0)`` clamp in
        ``LeRobotSingleDataset.get_video``: pre-episode-start indices reuse
        frame 0. The codec side gets zero MV/residual + ``frame_valid=False``
        at those slots, which is intentionally asymmetric -- visuals look like
        a static first frame, codec saliency is dead.
        """
        actual = list(self._image_buf)
        if len(actual) == 0:
            raise RuntimeError("HTCS adapter: image buffer empty at predict_action")
        if len(actual) < self.T:
            actual = [actual[0]] * (self.T - len(actual)) + actual
        return actual

    def predict_action(self, examples: List[dict], **kwargs) -> Dict[str, Any]:
        """Augment each example with T-frame history + codec window, forward.

        ``examples`` come from the eval client and contain
        ``{"image": [primary_uint8_HWC, wrist_uint8_HWC], "lang": str, ...}``.
        We only consume the primary view (index 0) -- wrist is discarded for
        HTCS because Stage1's frozen ViT + codec saliency are wired to the
        single view that ``codec_preprocess.py`` produced parquets for.
        """
        augmented: List[dict] = []
        for ex in examples:
            lang = ex.get("lang", None)
            if lang != self._last_lang:
                self._reset(lang)

            raw_img = ex.get("image", None)
            if raw_img is None:
                raise KeyError("HTCS adapter: example missing 'image' field")
            primary_np = np.asarray(raw_img[0] if isinstance(raw_img, (list, tuple)) else raw_img)
            if primary_np.dtype != np.uint8:
                primary_np = primary_np.astype(np.uint8)
            # Eval client resizes to image_size (default 224x224); defend against
            # callers that skip that step or use a different resolution.
            if primary_np.shape[:2] != (self.H, self.W):
                primary_pil_tmp = Image.fromarray(primary_np).resize((self.W, self.H))
                primary_np = np.asarray(primary_pil_tmp, dtype=np.uint8)

            self._codec.push(primary_np)
            self._image_buf.append(Image.fromarray(primary_np))

            new_ex = dict(ex)
            new_ex["image"] = self._build_t_frame_history()
            new_ex["codec"] = self._codec.get_window()
            augmented.append(new_ex)

        return self._framework.predict_action(examples=augmented, **kwargs)


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg

        # HTCS needs server-side rolling state (image history + codec encoder)
        # because the eval client only sends a single current frame per step.
        # We wrap the framework in a stateful adapter so PolicyServerWrapper's
        # predict_action path stays model-agnostic.
        framework_name = model_cfg.get("framework", {}).get("name", "")
        if framework_name == "HTCS":
            history_len = int(
                model_cfg.get("datasets", {}).get("vla_data", {}).get("history_len", 16)
            )
            grid_size = int(
                model_cfg.get("framework", {}).get("stage1", {}).get("grid_size", 14)
            )
            logging.info(
                "PolicyServerWrapper: detected HTCS ckpt; wrapping framework "
                "with _HTCSStatefulAdapter (T=%d, G=%d)",
                history_len, grid_size,
            )
            framework = _HTCSStatefulAdapter(
                framework, history_len=history_len, grid_size=grid_size,
            )
        self._framework = framework

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        return {"actions": unnorm}
