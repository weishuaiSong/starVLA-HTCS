"""Offline codec extraction for LIBERO LeRobot datasets.

Re-encodes each episode's **primary view** video to H.264 with a fixed GOP,
then walks the decoded stream to dump per-frame motion-vector grids, residual
energy maps and I-frame masks into a per-episode codec parquet.

Path scheme (matches ``htcs_codec_transform.HTCSCodecLoader``):
    <dataset_path>/htcs_codec/episode_{trajectory_id:06d}.parquet

This is the data side of HTCS Phase 1 (impl doc §2.1). Choosing offline
preprocessing avoids the IO bottleneck of running ffmpeg/PyAV inside the
dataloader (impl doc §2 preamble).

Engineering notes:
* ``codec_context.export_mvs = True`` surfaces MV side-data on decoded
  frames. FFmpeg implements this for H.264/MPEG-2/MPEG-4/VP8/VP9 only —
  not HEVC — which is why we re-encode to libx264 (see codec_config.py).
* We re-encode at **224×224**, matching both (a) the online
  ``RollingCodecEncoder`` resolution and (b) the ViT patch grid at
  ``patch_size=16``: 224 / 14 = 16, so each 14×14 codec cell covers
  exactly one H.264 macroblock = one ViT patch. This buys two things at
  once — pixel-aligned codec↔patch correspondence, and train/eval
  symmetry of the (MV, residual) distribution. LIBERO ships 256×256, so
  the re-encode step downsamples; this matches what the eval client
  feeds the model at runtime, so the SaliencyMLP sees the same MV
  distribution train and test.
* MV side-data lives under the ``Type.MOTION_VECTORS`` enum key, not the
  string ``'MOTION_VECTORS'`` — looking up by string silently returns
  None and yields all-zero MV grids.
* Idempotent: skips episodes whose codec parquet already exists.
* Dump a per-episode MV visualisation before the full run — confirm
  direction matches arm motion (impl doc §12 step 2).

Codec settings come from ``starVLA.model.modules.htcs.codec_config`` so
that offline preprocessing and the online ``RollingCodecEncoder`` stay
bit-identical — train/eval symmetry (impl doc §7.2.1).
"""

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm

from starVLA.model.modules.htcs.codec_config import HTCS_CODEC_CONFIG

# LIBERO primary view directory name in LeRobot layout.
PRIMARY_VIEW_DIR = "observation.images.image"

# Enum key used to fetch MV side-data off a decoded PyAV frame.
_MV_SIDE_DATA_TYPE = av.sidedata.sidedata.Type.MOTION_VECTORS


# ---------------------------------------------------------------------- #
#  Per-frame extraction helpers
# ---------------------------------------------------------------------- #
def extract_mv_grid(frame, grid_size: int = 14) -> np.ndarray:
    """Aggregate H.264 motion vectors onto a (grid_size, grid_size, 2) int8 grid.

    MV stays as a 2D vector (Δx, Δy) — Stage1's direction term ``MV · v_tgt``
    needs the components, not the norm. Returns zeros for frames with no MV
    side-data (I-frames).
    """
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
    target_size: int = 224,
) -> None:
    """Re-encode video to H.264 at ``target_size`` and dump per-episode codec parquet.

    The re-encode is **deliberately** done at the same spatial resolution
    the online ``RollingCodecEncoder`` uses (default 224×224) so that
    the (MV, residual) distribution seen at train time matches what the
    SaliencyMLP sees at eval. This also makes the 14×14 codec grid
    align exactly with the ViT's 14×14 patch grid at patch_size=16,
    since 224 / 14 = 16 / cell with no remainder — every codec cell
    covers exactly one H.264 macroblock = one ViT patch.
    """
    cfg = HTCS_CODEC_CONFIG
    if gop_size is None:
        gop_size = cfg['gop']

    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    reenc_path = video_path.with_suffix('.htcs.mp4')

    # 1. Re-encode with shared codec config (train/eval symmetry).
    #
    # NB: feeding the source AV1 frame's pts (time_base=1/10240) straight
    # through to libx264 makes the encoder treat nearly every frame as an
    # IDR — observed 55/110 keyframes regardless of g / keyint settings.
    # Rebuild each frame with a clean monotonic pts at the stream rate so
    # GOP=8 is honoured exactly.
    in_ctx = av.open(str(video_path))
    in_stream = in_ctx.streams.video[0]
    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0

    out_ctx = av.open(str(reenc_path), mode='w')
    out_stream = out_ctx.add_stream(cfg['codec'], rate=fps)
    # Force output dims to target_size so codec macroblock grid matches
    # the ViT patch grid (e.g. 224 / patch=16 → 14×14 cleanly).
    out_stream.width = int(target_size)
    out_stream.height = int(target_size)
    out_stream.pix_fmt = 'yuv420p'
    out_stream.options = {
        'preset':       cfg['preset'],
        'tune':         cfg['tune'],
        'g':            str(gop_size),
        'x264-params':  cfg['x264_params'],
    }
    pts = 0
    for frame in in_ctx.decode(video=0):
        # PyAV's frame.reformat keeps the call inside the C codec path,
        # avoiding a Python-side PIL roundtrip per frame (preprocess
        # dominates wall-time when serial-debugging).
        if frame.width != target_size or frame.height != target_size:
            frame = frame.reformat(
                width=target_size, height=target_size, format='rgb24',
            )
            rgb = frame.to_ndarray()
        else:
            rgb = frame.to_ndarray(format='rgb24')
        new_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        new_frame.pts = pts
        pts += 1
        for pkt in out_stream.encode(new_frame):
            out_ctx.mux(pkt)
    for pkt in out_stream.encode():
        out_ctx.mux(pkt)
    out_ctx.close()
    in_ctx.close()

    # 2. Decode with MV side-data exposed.
    in_ctx2 = av.open(str(reenc_path))
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

    # Optional: drop the intermediate re-encoded mp4 to save disk.
    try:
        reenc_path.unlink()
    except OSError:
        pass


def _episode_id_from_video_name(video_name: str) -> int:
    m = re.search(r'episode_(\d+)', video_name)
    if not m:
        raise ValueError(f"Unrecognised video filename: {video_name!r}")
    return int(m.group(1))


def _process_one(args: tuple) -> tuple:
    """Pickle-friendly worker — extracts one episode. Returns (status, video_str)."""
    video, out_parquet = args
    try:
        extract_codec_for_episode(Path(video), Path(out_parquet))
        return ("done", str(video))
    except Exception as e:
        return (f"error: {e}", str(video))


def process_suite(suite_root: Path, num_workers: int = 1) -> None:
    """Walk ``<suite_root>/videos`` for the primary view and extract codec parquets."""
    videos_root = suite_root / "videos"
    if not videos_root.exists():
        print(f"[codec] skip — no videos dir at {videos_root}")
        return
    out_root = suite_root / "htcs_codec"
    out_root.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[Path, Path]] = []
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
        todo.append((video, out_parquet))

    if not todo:
        print(f"[codec] {suite_root.name}: nothing to do (skipped {n_skip})")
        return

    print(f"[codec] {suite_root.name}: {len(todo)} episodes to process "
          f"(skipped {n_skip}, workers={num_workers})")

    t0 = time.time()
    if num_workers <= 1:
        # Serial path — easier to debug (full traceback on hangs).
        for video, out_parquet in tqdm(todo, desc=suite_root.name):
            extract_codec_for_episode(video, out_parquet)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_process_one, (str(v), str(o))) for v, o in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=suite_root.name):
                status, video = fut.result()
                if status != "done":
                    print(f"[codec] FAILED {video}: {status}")

    print(f"[codec] {suite_root.name}: done in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="playground/Datasets/LEROBOT_LIBERO_DATA",
        help="LeRobot LIBERO root containing libero_*_no_noops_1.0.0_lerobot/.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel episodes. Default: half of os.cpu_count(). Set to 1 to "
             "debug a hang (serial run prints tqdm progress + full tracebacks).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    for suite in args.suites:
        process_suite(data_root / f"{suite}_no_noops_1.0.0_lerobot", num_workers=args.num_workers)


if __name__ == "__main__":
    main()
