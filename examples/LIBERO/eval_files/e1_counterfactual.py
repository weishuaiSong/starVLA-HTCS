"""E1 — Counterfactual instruction evaluation.

For each episode, replace the ground-truth instruction with a randomly
sampled instruction from a *different* task and re-run rollout. The drop
in success rate measures how much the policy actually conditions on
language. drop > 0.15 is the green-light criterion for HTCS Phase 4 (impl
doc §9 / §12).

Usage:
    python examples/LIBERO/eval_files/e1_counterfactual.py \\
        --ckpt playground/Checkpoints/<run_id>/checkpoints/steps_30000_pytorch_model.pt
"""

import argparse
import random
from typing import List


# from examples.LIBERO.eval_files.eval_libero import LiberoRunner  # noqa: E501
# Real import deferred — keep this script importable on machines without
# the LIBERO sim installed (impl doc §10.7).


def load_all_libero_instructions() -> List[str]:
    """Collect every unique task instruction across the 4 LIBERO suites."""
    # TODO(htcs): scan task metadata for libero_spatial / object / goal / 10
    # and return the deduplicated instruction list.
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # TODO(htcs):
    #   1. instr_pool = load_all_libero_instructions().
    #   2. runner_orig = LiberoRunner(ckpt=args.ckpt, counterfactual=False).
    #   3. runner_cf   = LiberoRunner(ckpt=args.ckpt, counterfactual=True,
    #                                  instruction_pool=instr_pool).
    #   4. sr_orig = runner_orig.run(n_episodes=args.n_episodes).
    #   5. sr_cf   = runner_cf.run(n_episodes=args.n_episodes).
    #   6. Print + log "E1: orig {sr_orig:.3f}, counterfactual {sr_cf:.3f},
    #                   drop {sr_orig - sr_cf:.3f}".
    #   7. Threshold check: assert sr_orig - sr_cf > 0.15 else warn.
    raise NotImplementedError


if __name__ == "__main__":
    main()
