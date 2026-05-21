"""Streaming H.264 encoder/decoder for inference (M-online).

Maintains a persistent PyAV encoder + decoder pair so each rollout step
pushes one RGB frame and gets that frame's (MV, residual, is_i_frame)
back as the new tail of a sliding window. The window always has length T
— earlier slots are zero-padded with frame_valid=False when the episode
has not yet supplied T frames.

The codec configuration is shared with offline preprocessing (see
``codec_config.py``); divergence here breaks train/eval symmetry — refer
to impl doc §7.2 for the rationale.

Multi-view eval: instantiate one ``RollingCodecEncoder`` per camera view.
"""

import io
from collections import deque
from typing import Optional

import av
import numpy as np

from .codec_config import HTCS_CODEC_CONFIG


# Enum key for MV side-data on a decoded PyAV frame. Looking up the
# string ``'MOTION_VECTORS'`` silently returns None on PyAV ≥10.
_MV_SIDE_DATA_TYPE = av.sidedata.sidedata.Type.MOTION_VECTORS


# ---------------------------------------------------------------------- #
#  Per-frame extraction helpers (shared with codec_preprocess.py — both
#  paths MUST stay identical or train/eval symmetry breaks).
# ---------------------------------------------------------------------- #
def extract_mv_grid(frame, grid_size: int = 14) -> np.ndarray:
    """Aggregate decoded H.264 motion vectors onto a (grid_size, grid_size, 2) int8 grid."""
    W, H = frame.width, frame.height
    cell_w = max(W // grid_size, 1)
    cell_h = max(H // grid_size, 1)

    acc = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    cnt = np.zeros((grid_size, grid_size), dtype=np.int32)

    mv_block = frame.side_data.get(_MV_SIDE_DATA_TYPE)
    if mv_block is not None:
        for mv in mv_block:
            gx = min(int(mv.dst_x) // cell_w, grid_size - 1)
            gy = min(int(mv.dst_y) // cell_h, grid_size - 1)
            scale = max(int(mv.motion_scale), 1)
            acc[gy, gx, 0] += int(mv.motion_x) / scale
            acc[gy, gx, 1] += int(mv.motion_y) / scale
            cnt[gy, gx] += 1

    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return np.clip(acc, -127, 127).astype(np.int8)


def extract_luma_residual_energy(frame, grid: int = 14) -> np.ndarray:
    """Aggregate luma |Y-128| residual onto a (grid, grid) float16 map (block sum)."""
    y = frame.reformat(format='gray8').to_ndarray()        # (H, W) uint8
    H, W = y.shape
    block_h = H // grid
    block_w = W // grid
    res = np.abs(y[:block_h * grid, :block_w * grid].astype(np.float32) - 128.0)
    out = res.reshape(grid, block_h, grid, block_w).sum(axis=(1, 3))
    return out.astype(np.float16)


# ---------------------------------------------------------------------- #
#  Streaming encoder
# ---------------------------------------------------------------------- #
class RollingCodecEncoder:
    """Online H.264 encoder + decoder pair for one camera view.

    Usage:
        enc = RollingCodecEncoder(history_len=16, grid_size=14)
        enc.reset()                            # at episode start
        for rgb in episode_rgb_stream:
            enc.push(rgb)                      # one frame at a time
        codec_window = enc.get_window()        # latest T frames, D16-padded
    """

    def __init__(
        self,
        history_len: int = 16,
        grid_size: int = 14,
        frame_h: int = 224,
        frame_w: int = 224,
        fps: int = 30,
    ):
        self.T = int(history_len)
        self.G = int(grid_size)
        self.H = int(frame_h)
        self.W = int(frame_w)
        self.fps = int(fps)

        self.mv_buf:  deque = deque(maxlen=self.T)
        self.res_buf: deque = deque(maxlen=self.T)
        self.is_i_buf: deque = deque(maxlen=self.T)
        self.step_count = 0

        self._enc_buf: Optional[io.BytesIO] = None
        self._enc_ctn = None
        self._enc_stream = None
        self._dec_ctx = None
        self._dec_initialised = False

    # ------------------------------------------------------------------ #
    #  internal: codec lifecycle
    # ------------------------------------------------------------------ #
    def _init_codec_pair(self):
        cfg = HTCS_CODEC_CONFIG
        # In-memory mp4 muxer for the encoder side.
        self._enc_buf = io.BytesIO()
        self._enc_ctn = av.open(self._enc_buf, mode='w', format='mp4')
        self._enc_stream = self._enc_ctn.add_stream(cfg['codec'], rate=self.fps)
        self._enc_stream.width = self.W
        self._enc_stream.height = self.H
        self._enc_stream.pix_fmt = 'yuv420p'
        self._enc_stream.options = {
            'preset':       cfg['preset'],
            'tune':         cfg['tune'],
            'g':            str(cfg['gop']),
            'x264-params':  cfg['x264_params'],
        }
        # Stand-alone decoder so we can flip export_mvs on without touching
        # the encoder. extradata is copied lazily once available.
        self._dec_ctx = av.codec.CodecContext.create(cfg['decoder'], 'r')
        try:
            self._dec_ctx.export_mvs = True
        except AttributeError:
            # Older PyAV versions: set the raw flag.
            self._dec_ctx.flags2 |= getattr(
                av.codec.context.Flags2, 'export_mvs', 0,
            )
        self._dec_initialised = False

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #
    def reset(self):
        """Call at episode start — clears buffers and rebuilds the codec pair."""
        self.mv_buf.clear()
        self.res_buf.clear()
        self.is_i_buf.clear()
        self.step_count = 0
        try:
            if self._enc_ctn is not None:
                self._enc_ctn.close()
        except Exception:
            pass
        self._init_codec_pair()

    def push(self, rgb_frame: np.ndarray):
        """Push one (H, W, 3) uint8 RGB frame; updates internal buffers in place."""
        assert rgb_frame.dtype == np.uint8 and rgb_frame.shape == (self.H, self.W, 3), \
            f"expected ({self.H}, {self.W}, 3) uint8, got {rgb_frame.shape} {rgb_frame.dtype}"
        if self._enc_ctn is None:
            self._init_codec_pair()

        av_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
        av_frame.pts = self.step_count

        # 1. Encode → packets (zerolatency: 1 input → 1 packet).
        packets = list(self._enc_stream.encode(av_frame))

        # 2. Once we have the first packet, the encoder has produced extradata;
        #    propagate to the decoder so it can parse subsequent packets.
        if not self._dec_initialised and packets:
            extradata = self._enc_stream.codec_context.extradata
            if extradata is not None:
                self._dec_ctx.extradata = extradata
                self._dec_ctx.open()
                self._dec_initialised = True

        # 3. Decode packets → frames with MV side-data.
        decoded = []
        if self._dec_initialised:
            for pkt in packets:
                try:
                    decoded.extend(self._dec_ctx.decode(pkt))
                except av.AVError:
                    # First packet sometimes carries SPS/PPS only — skip.
                    pass

        # 4. Extract MV / residual / I-frame flag; append to sliding buffer.
        for df in decoded:
            mv = extract_mv_grid(df, grid_size=self.G)
            res = extract_luma_residual_energy(df, grid=self.G)
            self.mv_buf.append(mv)
            self.res_buf.append(res)
            self.is_i_buf.append(bool(df.key_frame))

        self.step_count += 1

    def get_window(self) -> dict:
        """Return the most recent T frames as a codec dict, D16-padded if needed.

        Returns:
            {
              'mv':          (T, G, G, 2) int8     — padded slots filled with 0
              'residual':    (T, G, G)    float16
              'is_i_frame':  (T,)         bool     — False for padded slots
              'frame_valid': (T,)         bool     — True iff a real frame
            }
        """
        actual = len(self.mv_buf)
        mv = np.zeros((self.T, self.G, self.G, 2), dtype=np.int8)
        res = np.zeros((self.T, self.G, self.G), dtype=np.float16)
        is_i = np.zeros((self.T,), dtype=bool)
        valid = np.zeros((self.T,), dtype=bool)

        # Oldest real frame goes at index (T - actual); newest at T - 1.
        for i in range(actual):
            pos = self.T - actual + i
            mv[pos] = self.mv_buf[i]
            res[pos] = self.res_buf[i]
            is_i[pos] = self.is_i_buf[i]
            valid[pos] = True

        return {
            'mv':          mv,
            'residual':    res,
            'is_i_frame':  is_i,
            'frame_valid': valid,
        }
