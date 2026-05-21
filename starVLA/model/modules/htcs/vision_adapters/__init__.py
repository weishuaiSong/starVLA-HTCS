"""Registry + factory for HTCS vision adapters.

Why a registry, not ``type().__name__`` dispatch:

* transformers occasionally renames model classes between versions;
* PEFT / DDP / accelerate wrap the HF model, hiding the original class;
* fine-tuned checkpoints sometimes ship custom wrapper classes.

Each adapter answers ``can_handle(hf_model)`` via cheap structural checks
(``config.model_type``, presence of ``.visual``, vision config attrs, …),
so detection survives the cases above.

Usage:
    from starVLA.model.modules.htcs.vision_adapters import build_vision_adapter
    adapter = build_vision_adapter(hf_model, grid_size=14)             # auto
    adapter = build_vision_adapter(hf_model, override="qwen3_5")       # explicit
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import BaseVisionAdapter

# ---------------------------------------------------------------------- #
#  registry
# ---------------------------------------------------------------------- #
_REGISTRY: Dict[str, Type[BaseVisionAdapter]] = {}


def register_vision_adapter(name: str):
    """Decorator: register an adapter subclass under ``name``.

    ``name`` is the short string users put into the YAML
    (``framework.vision_adapter: <name>``) to force a specific adapter.
    It should be unique and stable across releases.
    """
    def deco(cls: Type[BaseVisionAdapter]) -> Type[BaseVisionAdapter]:
        if not issubclass(cls, BaseVisionAdapter):
            raise TypeError(
                f"{cls.__name__} must subclass BaseVisionAdapter to be registered."
            )
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Duplicate vision-adapter name {name!r}: "
                f"{_REGISTRY[name].__name__} vs {cls.__name__}."
            )
        _REGISTRY[name] = cls
        return cls
    return deco


def list_vision_adapters() -> List[str]:
    """Return the registered short names (for diagnostics / yaml hints)."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------- #
#  factory
# ---------------------------------------------------------------------- #
def build_vision_adapter(
    hf_model,
    override: Optional[str] = None,
    **kwargs,
) -> BaseVisionAdapter:
    """Pick the right adapter for ``hf_model`` and instantiate it.

    Resolution order:

    1. If ``override`` is given, look it up in the registry and use it.
       Errors loudly if the name is unknown.
    2. Otherwise, ask every registered adapter via ``can_handle``;
       expect exactly one match. Zero matches → ``NotImplementedError``
       (with the registered names so the user can add or override).
       Multiple matches → ``RuntimeError`` (so silent mis-routing is
       impossible).

    ``kwargs`` are forwarded to the adapter constructor (e.g. ``grid_size``).
    """
    if override:
        if override not in _REGISTRY:
            raise KeyError(
                f"Unknown vision_adapter override {override!r}. "
                f"Registered: {list_vision_adapters()}."
            )
        return _REGISTRY[override](hf_model, **kwargs)

    candidates = [
        (name, cls) for name, cls in _REGISTRY.items()
        if _safe_can_handle(cls, hf_model)
    ]
    if len(candidates) == 1:
        return candidates[0][1](hf_model, **kwargs)
    if len(candidates) == 0:
        raise NotImplementedError(
            f"No vision adapter can_handle {type(hf_model).__name__}. "
            f"Registered: {list_vision_adapters()}. "
            f"Add @register_vision_adapter for it, or set "
            f"framework.vision_adapter in the YAML to override."
        )
    names = [n for n, _ in candidates]
    raise RuntimeError(
        f"Ambiguous vision adapter for {type(hf_model).__name__}: "
        f"{names} all claim to handle it. "
        f"Set framework.vision_adapter in the YAML to disambiguate."
    )


def _safe_can_handle(cls: Type[BaseVisionAdapter], hf_model) -> bool:
    """Swallow exceptions from ``can_handle`` so one buggy detector
    can't break detection of every other adapter."""
    try:
        return bool(cls.can_handle(hf_model))
    except Exception:
        return False


# ---------------------------------------------------------------------- #
#  side-effect imports: registering concrete adapters
# ---------------------------------------------------------------------- #
# Keep imports at the bottom so the registry exists before adapters
# decorate themselves into it. Each concrete adapter module is expected
# to call ``@register_vision_adapter("<name>")`` at import time.
from . import qwen3_5  # noqa: E402,F401

__all__ = [
    "BaseVisionAdapter",
    "register_vision_adapter",
    "list_vision_adapters",
    "build_vision_adapter",
]
