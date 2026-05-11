"""Offline codec extraction for LIBERO LeRobot datasets.

Re-encodes each episode video to HEVC with a fixed GOP, then walks the
decoded stream to dump per-frame motion-vector grids, residual energy maps
and I-frame masks into ``codec.parquet`` next to the original video.

This is the data side of HTCS Phase 1 (impl doc §2.1). Choosing offline
preprocessing avoids the IO bottleneck of running ffmpeg/PyAV inside the
dataloader (see impl doc §2 preamble).

Engineering notes:
* ``codec_context.export_mvs = True`` is required to surface MV side-data
  on the decoded frames (impl doc §10.3).
* HEVC MVs live on 16x16 sub-blocks; aggregate / interpolate them onto a
  14x14 grid so they line up with the SigLIP patch grid.
* Dump a per-episode visualisation before kicking off full batches —
  confirm MV direction matches arm motion (impl doc §12 step 2).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# import av  # PyAV — uncomment once installed in the runtime env.


# ---------------------------------------------------------------------- #
# Per-frame extraction helpers
# ---------------------------------------------------------------------- #
def extract_mv_grid(frame, grid_size: int = 14) -> np.ndarray:
    """Aggregate HEVC 16x16 motion vectors onto a (grid_size, grid_size, 2) int8 grid.

    Returns zeros for I-frames.
    """
    # TODO(htcs): read frame.side_data['MOTION_VECTORS'], bin by spatial
    # location into the target grid, average dx/dy per bin, cast to int8.
    raise NotImplementedError


def extract_residual_energy(frame, grid: int = 14) -> np.ndarray:
    """Approximate per-block residual energy on a (grid, grid) float16 map.

    For HEVC, derive from QP-weighted coded block flag energy or, when
    side-data is unavailable, fall back to luma high-pass magnitude.
    """
    # TODO(htcs): implement residual energy aggregation; return float16.
    raise NotImplementedError


# ---------------------------------------------------------------------- #
# Episode-level driver
# ---------------------------------------------------------------------- #
def extract_codec_for_episode(video_path: Path, gop_size: int = 8) -> None:
    """Re-encode video to HEVC and dump codec.parquet alongside it."""
    # TODO(htcs):
    #   1. Re-encode video_path → video_path.with_suffix('.hevc.mp4') using
    #      libx265 with options preset=medium, g=gop_size.
    #   2. Re-open the HEVC file with export_mvs=True; iterate demux/decode.
    #   3. For each frame: record is_i_frame, mv grid (zeros if I), residual.
    #   4. Write a parquet with columns:
    #        frame_idx (int), is_i_frame (bool),
    #        mv (bytes; int8 (G,G,2)),
    #        residual (bytes; float16 (G,G))
    #      Path: video_path.parent / 'codec.parquet'.
    raise NotImplementedError


def main() -> None:
    """Iterate all LIBERO suite videos and extract codec artifacts."""
    data_root = Path("playground/Datasets/LEROBOT_LIBERO_DATA")
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    for suite in suites:
        videos_root = data_root / f"{suite}_no_noops_1.0.0_lerobot" / "videos"
        for episode_video in videos_root.rglob("*.mp4"):
            # TODO(htcs): skip if codec.parquet already present (idempotent rerun).
            extract_codec_for_episode(episode_video)


if __name__ == "__main__":
    main()
