"""HTCS modules — Hierarchical Task-Conditioned Saliency.

Stage 1: codec-based saliency selection (parameter-level language conditioning).
Stage 2: language-derived slot compression (structure-level conditioning).
"""

from .codec_config import HTCS_CODEC_CONFIG
from .rolling_codec import RollingCodecEncoder
from .rope_utils import apply_3d_rope
from .saliency_mlp import SaliencyMLP
from .slot_attention import CompetitiveSlotAttention
from .stage1_codec_selector import Stage1CodecSelector
from .stage2_lang_compressor import Stage2LangCompressor
from .vision_adapters import (
    BaseVisionAdapter,
    build_vision_adapter,
    list_vision_adapters,
    register_vision_adapter,
)

__all__ = [
    "HTCS_CODEC_CONFIG",
    "RollingCodecEncoder",
    "apply_3d_rope",
    "SaliencyMLP",
    "CompetitiveSlotAttention",
    "Stage1CodecSelector",
    "Stage2LangCompressor",
    "BaseVisionAdapter",
    "build_vision_adapter",
    "list_vision_adapters",
    "register_vision_adapter",
]
