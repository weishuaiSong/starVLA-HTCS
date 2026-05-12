"""HTCS codec loader for the LeRobot dataloader.

Reads the offline per-episode codec parquet produced by
``examples/LIBERO/train_files/codec_preprocess.py`` and injects the
``{'mv', 'residual', 'is_i_frame', 'frame_valid'}`` quadruple into each
example dict.

Path scheme (matches codec_preprocess.py):
    <dataset_path>/htcs_codec/episode_{trajectory_id:06d}.parquet

Triggered from ``LeRobotSingleDataset.__getitem__`` when the
``enable_htcs_codec: true`` YAML flag is set; see impl doc §2.2 / §2.3.

Engineering note (impl doc §10.9 / D16): LeRobot frame indices may go
negative at episode boundaries. HTCSCodecLoader needs the **raw** indices
so it can produce a ``frame_valid`` mask — padded slots get zero MV /
residual / is_i_frame to keep training and inference symmetric.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class HTCSCodecLoader:
    """Per-episode lazy loader for codec parquet artifacts."""

    def __init__(self, history_len: int = 16, grid_size: int = 14):
        self.T = int(history_len)
        self.G = int(grid_size)
        self._cache: Dict[Path, pd.DataFrame] = {}

    def _load_codec(self, parquet_path: Path) -> pd.DataFrame:
        parquet_path = Path(parquet_path)
        if parquet_path not in self._cache:
            self._cache[parquet_path] = pd.read_parquet(parquet_path)
        return self._cache[parquet_path]

    def __call__(
        self,
        example: dict,
        codec_parquet_path: Union[Path, str],
        frame_indices: List[int],
    ) -> dict:
        """Inject ``example['codec']`` with mv / residual / is_i_frame / frame_valid.

        Args:
            example:            sample dict produced by ``_pack_sample``.
            codec_parquet_path: full path to ``episode_XXXXXX.parquet``.
            frame_indices:      raw (possibly negative / out-of-range) frame
                                indices spanning the T-length history window.
        """
        df = self._load_codec(Path(codec_parquet_path))
        n_total = len(df)

        # D16 layer 1: identify padded positions (frame index out of range).
        raw = np.asarray(frame_indices, dtype=np.int64)
        valid_mask = (raw >= 0) & (raw < n_total)
        clamped = np.clip(raw, 0, max(n_total - 1, 0))

        rows = df.iloc[clamped]
        mv = np.stack([
            np.frombuffer(b, dtype=np.int8).reshape(self.G, self.G, 2)
            for b in rows['mv']
        ]).astype(np.int8)
        res = np.stack([
            np.frombuffer(b, dtype=np.float16).reshape(self.G, self.G)
            for b in rows['residual']
        ]).astype(np.float16)
        is_i = rows['is_i_frame'].to_numpy().astype(bool).copy()

        # D16 layer 2: padded slots are forced to zero codec + False I-frame.
        invalid = ~valid_mask
        if invalid.any():
            mv[invalid] = 0
            res[invalid] = 0
            is_i[invalid] = False

        example['codec'] = {
            'mv':          mv,                                              # (T, G, G, 2) int8
            'residual':    res,                                             # (T, G, G)    float16
            'is_i_frame':  is_i,                                            # (T,)         bool
            'frame_valid': valid_mask,                                      # (T,)         bool
        }
        return example
