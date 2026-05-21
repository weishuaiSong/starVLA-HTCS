"""Shared codec configuration — train/eval symmetry.

Both ``codec_preprocess.py`` (offline, training data) and
``RollingCodecEncoder`` (online, inference) MUST consume the same options
here. Any divergence makes the (MV, residual) distribution at inference
mismatch what SaliencyMLP saw during training, and the model breaks.

See impl doc §7.2.1 for the rationale behind each setting.

Codec choice — libx264 (H.264):
FFmpeg's ``export_mvs`` side-data is implemented for the H.264, MPEG-2,
MPEG-4, VP8 and VP9 decoders only. The HEVC decoder does NOT emit
``AV_FRAME_DATA_MOTION_VECTORS``, so re-encoding to HEVC produced
all-zero MV grids and broke the Stage-1 saliency signal. Switching to
H.264 restores per-frame MVs while keeping every other knob identical.
"""

HTCS_CODEC_CONFIG = {
    # libx264 — H.264 encoder paired with the 'h264' decoder, which is
    # the codec FFmpeg's MV export actually supports.
    'codec':        'libx264',
    # ultrafast — single-frame latency budget at inference.
    'preset':       'ultrafast',
    # zerolatency — force each input frame to emit a packet immediately
    # (no lookahead buffer). Mandatory for online streaming use; training
    # side adopts the same setting so distributions match.
    'tune':         'zerolatency',
    # GOP = 8 frames. With T=16 history we always see ≥1 I-frame in the
    # window; pairs cleanly with the I-frame-always-kept top-ρ rule.
    'gop':          8,
    # bframes=0:   drop B-frames → MVs are purely forward, simple causality.
    # scenecut=0:  disable scene-cut detection → GOP boundaries are fixed.
    # log_level=error: silence x264 banner spam.
    'x264_params':  'log_level=error:bframes=0:scenecut=0',
    # Decoder name passed to PyAV / av.codec.CodecContext.create.
    'decoder':      'h264',
}
