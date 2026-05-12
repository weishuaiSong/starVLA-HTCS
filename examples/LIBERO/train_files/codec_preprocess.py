"""Offline codec extraction for LIBERO LeRobot datasets.

Re-encodes each episode's **primary view** video to HEVC with a fixed GOP,
then walks the decoded stream to dump per-frame motion-vector grids, residual
energy maps and I-frame masks into a per-episode codec parquet.

Path scheme (matches ``htcs_codec_transform.HTCSCodecLoader``):
    <dataset_path>/htcs_codec/episode_{trajectory_id:06d}.parquet

This is the data side of HTCS Phase 1 (impl doc §2.1). Choosing offline
preprocessing avoids the IO bottleneck of running ffmpeg/PyAV inside the
dataloader (impl doc §2 preamble).

Engineering notes:
* ``codec_context.export_mvs = True`` is required to surface MV side-data
  on the decoded frames (impl doc §10.3).
* HEVC MVs live on 16×16 sub-blocks; aggregate them onto a 14×14 grid so
  they line up with the SigLIP patch grid.
* Idempotent: skips episodes whose codec parquet already exists.
* Dump a per-episode MV visualisation before the full run — confirm
  direction matches arm motion (impl doc §12 step 2).

Codec settings come from ``starVLA.model.modules.htcs.codec_config`` so
that offline preprocessing and the online ``RollingCodecEncoder`` stay
bit-identical — train/eval symmetry (impl doc §7.2.1).
"""

import re
from pathlib import Path

import av
import numpy as np
import pandas as pd

from starVLA.model.modules.htcs.codec_config import HTCS_CODEC_CONFIG

# LIBERO primary view directory name in LeRobot layout.
PRIMARY_VIEW_DIR = "observation.images.image"


# ---------------------------------------------------------------------- #
#  Per-frame extraction helpers
# ---------------------------------------------------------------------- #
def extract_mv_grid(frame, grid_size: int = 14) -> np.ndarray:
    """Aggregate HEVC motion vectors onto a (grid_size, grid_size, 2) int8 grid.

    MV stays as a 2D vector (Δx, Δy) — Stage1's direction term ``MV · v_tgt``
    needs the components, not the norm. Returns zeros for frames with no MV
    side-data (I-frames).
    """
    W, H = frame.width, frame.height
    cell_w = max(W // grid_size, 1)
    cell_h = max(H // grid_size, 1)

    acc = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    cnt = np.zeros((grid_size, grid_size), dtype=np.int32)

    if 'MOTION_VECTORS' in frame.side_data:
        for mv in frame.side_data['MOTION_VECTORS']:
            gx = min(int(mv.dst_x) // cell_w, grid_size - 1)
            gy = min(int(mv.dst_y) // cell_h, grid_size - 1)
            scale = max(int(mv.motion_scale), 1)
            acc[gy, gx, 0] += int(mv.motion_x) / scale
            acc[gy, gx, 1] += int(mv.motion_y) / scale
            cnt[gy, gx] += 1

    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return np.clip(acc, -127, 127).astype(np.int8)


def extract_residual_energy(frame, grid: int = 14) -> np.ndarray:
    """Aggregate luma ``|Y - 128|`` energy onto a (grid, grid) float16 map.

    Locked to single-channel luma (D12 in impl doc).
    """
    y = frame.reformat(format='gray8').to_ndarray()        # (H, W) uint8
    H, W = y.shape
    block_h = H // grid
    block_w = W // grid
    res = np.abs(y[:block_h * grid, :block_w * grid].astype(np.float32) - 128.0)
    out = res.reshape(grid, block_h, grid, block_w).sum(axis=(1, 3))
    return out.astype(np.float16)


# ---------------------------------------------------------------------- #
#  Episode-level driver
# ---------------------------------------------------------------------- #
def extract_codec_for_episode(
    video_path: Path,
    out_parquet_path: Path,
    gop_size: int = None,
    grid_size: int = 14,
) -> None:
    """Re-encode video to HEVC and dump per-episode codec parquet."""
    cfg = HTCS_CODEC_CONFIG
    if gop_size is None:
        gop_size = cfg['gop']

    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    hevc_path = video_path.with_suffix('.hevc.mp4')

    # 1. Re-encode to HEVC with shared codec config (train/eval symmetry).
    in_ctx = av.open(str(video_path))
    in_stream = in_ctx.streams.video[0]
    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0

    out_ctx = av.open(str(hevc_path), mode='w')
    out_stream = out_ctx.add_stream(cfg['codec'], rate=fps)
    out_stream.width = in_stream.codec_context.width
    out_stream.height = in_stream.codec_context.height
    out_stream.pix_fmt = 'yuv420p'
    out_stream.options = {
        'preset':       cfg['preset'],
        'tune':         cfg['tune'],
        'g':            str(gop_size),
        'x265-params':  cfg['x265_params'],
    }
    for frame in in_ctx.decode(video=0):
        for pkt in out_stream.encode(frame):
            out_ctx.mux(pkt)
    for pkt in out_stream.encode():
        out_ctx.mux(pkt)
    out_ctx.close()
    in_ctx.close()

    # 2. Decode HEVC with MV side-data exposed.
    in_ctx2 = av.open(str(hevc_path))
    in_ctx2.streams.video[0].codec_context.export_mvs = True

    mv_list:  list[np.ndarray] = []
    res_list: list[np.ndarray] = []
    is_i_flags: list[bool] = []
    for frame in in_ctx2.decode(video=0):
        if frame.key_frame:
            mv_list.append(np.zeros((grid_size, grid_size, 2), dtype=np.int8))
            res_list.append(extract_residual_energy(frame, grid=grid_size))
            is_i_flags.append(True)
        else:
            mv_list.append(extract_mv_grid(frame, grid_size=grid_size))
            res_list.append(extract_residual_energy(frame, grid=grid_size))
            is_i_flags.append(False)
    in_ctx2.close()

    # 3. Dump parquet.
    df = pd.DataFrame({
        'frame_idx':  list(range(len(mv_list))),
        'is_i_frame': is_i_flags,
        'mv':         [m.tobytes() for m in mv_list],
        'residual':   [r.tobytes() for r in res_list],
    })
    df.to_parquet(out_parquet_path)

    # Optional: drop the intermediate hevc.mp4 to save disk.
    try:
        hevc_path.unlink()
    except OSError:
        pass


def _episode_id_from_video_name(video_name: str) -> int:
    m = re.search(r'episode_(\d+)', video_name)
    if not m:
        raise ValueError(f"Unrecognised video filename: {video_name!r}")
    return int(m.group(1))


def process_suite(suite_root: Path) -> None:
    """Walk ``<suite_root>/videos`` for the primary view and extract codec parquets."""
    videos_root = suite_root / "videos"
    if not videos_root.exists():
        print(f"[codec] skip — no videos dir at {videos_root}")
        return
    out_root = suite_root / "htcs_codec"
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skip = 0
    for video in videos_root.rglob("*.mp4"):
        # Primary view only — wrist cam is not needed for HTCS Stage1.
        if PRIMARY_VIEW_DIR not in video.parts:
            continue
        ep_id = _episode_id_from_video_name(video.stem)
        out_parquet = out_root / f"episode_{ep_id:06d}.parquet"
        if out_parquet.exists():
            n_skip += 1
            continue
        print(f"[codec] {video} -> {out_parquet.name}")
        extract_codec_for_episode(video, out_parquet)
        n_done += 1
    print(f"[codec] {suite_root.name}: done {n_done}, skipped {n_skip}")


def main() -> None:
    """Iterate all LIBERO suite videos and extract codec artifacts."""
    data_root = Path("playground/Datasets/LEROBOT_LIBERO_DATA")
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    for suite in suites:
        process_suite(data_root / f"{suite}_no_noops_1.0.0_lerobot")


if __name__ == "__main__":
    main()
