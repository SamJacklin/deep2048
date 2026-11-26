"""Add tuples to a trained 8x6 network, starting them at zero: Jaśkowski's redundant small tuples by default, or
with --network 12x6 the four extra 6-cell shapes.

The network then reads the 8x6 tuples plus the added ones (NETWORKS[--network]). New weights are zero,
so it plays identically until it trains. Temporal coherence sums, if the checkpoint has them, are kept for the
existing weights and start at zero for the new ones, so the new weights learn at the full rate.

Usage:
    uv run python scripts/add_tuples.py checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz checkpoints/ntuple/redundant-8x6/games_2000000.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import NETWORKS, CoherenceTraces, load_traces, network_name


def add_tuples(network: NTupleNetwork, target: str = "8x6-redundant") -> NTupleNetwork:
    """Return the 8x6 network with the rest of NETWORKS[target]'s tuples added after its own, their weights at zero."""
    if network_name(network.tuples) != "8x6" or network.stage_tile is not None:
        raise ValueError("Only a single-stage 8x6 network can be given more tuples")
    tables = sum(16 ** len(t) for t in network.tuples)
    extra = sum(16 ** len(t) for t in NETWORKS[target][len(network.tuples) :])
    weights = np.concatenate([network.weights[:tables], np.zeros(extra, dtype=network.weights.dtype), network.weights[tables:]])
    return NTupleNetwork(
        NETWORKS[target],
        weights,
        downgrade_tile=network.downgrade_tile,
        scale_feature=network.scale_feature,
        scale_when_downgraded=network.scale_when_downgraded,
    )


def add_tuples_to_traces(traces: CoherenceTraces, network: NTupleNetwork, target: str = "8x6-redundant") -> CoherenceTraces:
    """Return the temporal coherence sums laid out for the network with the added tuples, zero for the new weights."""
    tables = sum(16 ** len(t) for t in network.tuples)
    extra = sum(16 ** len(t) for t in NETWORKS[target][len(network.tuples) :])
    return CoherenceTraces(*(np.concatenate([sums[:tables], np.zeros(extra, dtype=sums.dtype), sums[tables:]]) for sums in (traces.errors, traces.abs_errors)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add tuples to a trained 8x6 network")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--network", choices=("8x6-redundant", "12x6"), default="8x6-redundant", help="the network to grow into")
    args = parser.parse_args()

    network, metadata = NTupleNetwork.load(args.checkpoint)
    traces = load_traces(args.checkpoint)
    add_tuples(network, args.network).save(args.output, metadata, None if traces is None else add_tuples_to_traces(traces, network, args.network))
    print(f"Saved {args.output}: {len(NETWORKS[args.network])} tuples, the added ones starting at zero")


if __name__ == "__main__":
    main()
