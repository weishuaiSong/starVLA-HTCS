"""HTCS codec loader for the LeRobot dataloader.

Reads the offline ``codec.parquet`` produced by
``examples/LIBERO/train_files/codec_preprocess.py`` and injects the
``{'mv', 'residual', 'is_i_frame'}`` triple into each example dict.

Triggered from ``LeRobotSingleDataset.__getitem__`` when the
``enable_htcs_codec: true`` YAML flag is set; see impl doc §2.2 / §2.3.

Engineering note (impl doc §10.2): LeRobot frame indices may go negative at
episode boundaries — clamp before indexing the parquet rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class HTCSCodecLoader:
    """Per-episode lazy loader for codec parquet artifacts."""

    def __init__(self, history_len: int = 16, grid_size: int = 14):
        self.T = history_len
        self.G = grid_size
        self._cache: dict[Path, pd.DataFrame] = {}

    def _load_codec(self, episode_dir: Path) -> pd.DataFrame:
        # TODO(htcs): cache parquet by episode_dir to avoid repeated disk IO.
        #   if episode_dir not in self._cache:
        #       self._cache[episode_dir] = pd.read_parquet(episode_dir / 'codec.parquet')
        #   return self._cache[episode_dir]
        raise NotImplementedError

    def __call__(
        self,
        example: dict,
        episode_dir: Path,
        frame_indices: list[int],
    ) -> dict:
        """
        Inject ``example['codec']`` with:
            'mv':         (T, G, G, 2) int8
            'residual':   (T, G, G)    float16
            'is_i_frame': (T,)         bool
        """
        # TODO(htcs):
        #   1. df = self._load_codec(episode_dir).
        #   2. Clamp frame_indices to [0, len(df) - 1] (boundary case).
        #   3. rows = df.iloc[frame_indices].
        #   4. mv  = stack(frombuffer(b, int8).reshape(G, G, 2) for b in rows['mv']).
        #   5. res = stack(frombuffer(b, float16).reshape(G, G) for b in rows['residual']).
        #   6. is_i = rows['is_i_frame'].to_numpy().
        #   7. example['codec'] = {'mv': mv, 'residual': res, 'is_i_frame': is_i}.
        raise NotImplementedError
