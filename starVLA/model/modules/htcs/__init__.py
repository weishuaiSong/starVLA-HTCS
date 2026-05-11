"""HTCS modules — Hierarchical Task-Conditioned Selection.

Stage 1: codec-based saliency selection (parameter-level language conditioning).
Stage 2: language-derived slot compression (structure-level conditioning).
"""

from .saliency_mlp import SaliencyMLP
from .slot_attention import CompetitiveSlotAttention
from .stage1_codec_selector import Stage1CodecSelector
from .stage2_lang_compressor import Stage2LangCompressor

__all__ = [
    "SaliencyMLP",
    "CompetitiveSlotAttention",
    "Stage1CodecSelector",
    "Stage2LangCompressor",
]
