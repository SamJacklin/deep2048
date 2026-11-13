"""Give a trained n-tuple network a second stage for the late game, starting as an exact copy of the first.

Boards whose real largest tile is STAGE_TILE or more then read their own copy of the weights, so the late game
can learn values of its own instead of sharing the earlier stages' compromise. Both stages read boards the same
way (including any downgrading), and the copy is exact, so the network plays identically until it trains. The
checkpoint is saved without temporal coherence sums, so training resumes with every weight at the full rate,
unless --keep-coherence copies them into the second stage too.

Usage:
    uv run python scripts/add_stage.py checkpoints/ntuple/long-8x6/games_1500000.npz checkpoints/ntuple/two-stage-8x6/games_1500000.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import CoherenceTraces, load_traces


def add_stage(network: NTupleNetwork, stage_tile: int) -> NTupleNetwork:
    """Return the network with a second stage from stage_tile, copied from its single stage."""
    if network.stage_tile is not None:
        raise ValueError(f"The network already has a second stage from {network.stage_tile}")
    tables = sum(16 ** len(t) for t in network.tuples)
    weights = np.concatenate([network.weights[:tables], network.weights[:tables], network.weights[tables:]])
    return NTupleNetwork(
        network.tuples,
        weights,
        downgrade_tile=network.downgrade_tile,
        scale_feature=network.scale_feature,
        scale_when_downgraded=network.scale_when_downgraded,
        stage_tile=stage_tile,
    )


def add_stage_to_traces(traces: CoherenceTraces, network: NTupleNetwork) -> CoherenceTraces:
    """Return the temporal coherence sums laid out for the network with a second stage, copied from the first."""
    tables = sum(16 ** len(t) for t in network.tuples)
    return CoherenceTraces(*(np.concatenate([sums[:tables], sums[:tables], sums[tables:]]) for sums in (traces.errors, traces.abs_errors)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Give an n-tuple network a second stage, copied from the first")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage-tile", type=int, default=16384, help="Boards with this tile or larger use the second stage (default: 16384)")
    parser.add_argument("--keep-coherence", action="store_true", help="Copy the temporal coherence sums into the second stage instead of dropping them")
    args = parser.parse_args()

    network, metadata = NTupleNetwork.load(args.checkpoint)
    traces = load_traces(args.checkpoint) if args.keep_coherence else None
    if args.keep_coherence and traces is None:
        raise SystemExit(f"{args.checkpoint} has no temporal coherence sums to keep")
    add_stage(network, args.stage_tile).save(args.output, metadata, None if traces is None else add_stage_to_traces(traces, network))
    print(f"Saved {args.output}: a second stage from {args.stage_tile}, copied from {args.checkpoint}")


if __name__ == "__main__":
    main()
