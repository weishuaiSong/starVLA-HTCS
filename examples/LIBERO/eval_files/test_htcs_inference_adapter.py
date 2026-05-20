"""Smoke test for _HTCSStatefulAdapter — verifies that the server-side
rolling codec + image-history adapter assembles correctly-shaped examples
without needing real Qwen3.5-VL weights or a websocket server.

Usage:
    PYTHONPATH=. python examples/LIBERO/eval_files/test_htcs_inference_adapter.py

Requires PyAV with HEVC (x265) — same dependency as codec_preprocess.py.
"""

import numpy as np

from deployment.model_server.policy_wrapper import _HTCSStatefulAdapter


class _RecordingFramework:
    """Stub that records the augmented examples instead of running a model."""

    def __init__(self):
        self.calls: list = []

    def predict_action(self, examples, **kwargs):
        self.calls.append({"examples": examples, "kwargs": kwargs})
        # Pretend to emit an 8-step, 7-dim action chunk per example.
        B = len(examples)
        return {"normalized_actions": np.zeros((B, 8, 7), dtype=np.float32)}


def _fake_obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    return {
        "image": [img, img],          # primary + wrist as eval client emits
        "lang":  "pick up the red block",
    }


def main() -> None:
    fw = _RecordingFramework()
    adapter = _HTCSStatefulAdapter(fw, history_len=16, grid_size=14)

    # ---- 1. First N steps of an episode: deque pads with oldest frame -------
    for t in range(20):
        adapter.predict_action(examples=[_fake_obs(t)])
        ex = fw.calls[-1]["examples"][0]
        assert len(ex["image"]) == 16, f"step {t}: image history len = {len(ex['image'])}, want 16"
        # Codec window shapes.
        codec = ex["codec"]
        assert codec["mv"].shape         == (16, 14, 14, 2)
        assert codec["residual"].shape   == (16, 14, 14)
        assert codec["is_i_frame"].shape == (16,)
        assert codec["frame_valid"].shape == (16,)
        # During the first T steps the buffer should grow; once filled the
        # oldest frame is evicted by deque(maxlen).
        n_valid_expected = min(t + 1, 16)
        assert int(codec["frame_valid"].sum()) <= n_valid_expected, \
            f"step {t}: frame_valid={int(codec['frame_valid'].sum())} > expected {n_valid_expected}"

    # ---- 2. Language change triggers reset ----------------------------------
    new_obs = _fake_obs(99)
    new_obs["lang"] = "open the top drawer"
    adapter.predict_action(examples=[new_obs])
    ex = fw.calls[-1]["examples"][0]
    codec = ex["codec"]
    # After reset + 1 push, only the newest slot should be valid.
    assert int(codec["frame_valid"].sum()) == 1, \
        f"reset: frame_valid sum = {int(codec['frame_valid'].sum())}, want 1"

    # ---- 3. Same lang again continues the (new) history ---------------------
    for t in range(5):
        adapter.predict_action(examples=[new_obs])
    ex = fw.calls[-1]["examples"][0]
    codec = ex["codec"]
    assert int(codec["frame_valid"].sum()) >= 2, \
        "lang stable: codec should keep accumulating frames"

    print("OK — _HTCSStatefulAdapter smoke test passed "
          f"({len(fw.calls)} predict_action calls, last codec valid="
          f"{int(codec['frame_valid'].sum())}/16)")


if __name__ == "__main__":
    main()
