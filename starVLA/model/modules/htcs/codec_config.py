"""Shared HEVC codec configuration — train/eval symmetry.

Both ``codec_preprocess.py`` (offline, training data) and
``RollingCodecEncoder`` (online, inference) MUST consume the same options
here. Any divergence makes the (MV, residual) distribution at inference
mismatch what SaliencyMLP saw during training, and the model breaks.

See impl doc §7.2.1 for the rationale behind each setting.

D12 lock: codec is fixed to HEVC (libx265). No other codec switch.
"""

HTCS_CODEC_CONFIG = {
    # libx265 — HEVC encoder. PyAV exposes MV side-data most reliably for this.
    'codec':        'libx265',
    # ultrafast — single-frame latency budget at inference.
    'preset':       'ultrafast',
    # zerolatency — force each input frame to emit a packet immediately
    # (no lookahead buffer). Mandatory for online streaming use; training
    # side adopts the same setting so distributions match.
    'tune':         'zerolatency',
    # GOP = 8 frames. With T=16 history we always see ≥1 I-frame in the
    # window; pairs cleanly with the I-frame-always-kept top-ρ rule.
    'gop':          8,
    # bframes=0:    drop B-frames → MVs are purely forward, simple causality.
    # no-scenecut:  disable scene-cut detection → GOP boundaries are fixed.
    # log-level:    silence x265 banner spam.
    'x265_params':  'log-level=error:bframes=0:no-scenecut=1',
}
