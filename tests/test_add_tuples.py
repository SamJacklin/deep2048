import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import NETWORKS, CoherenceTraces
from deep_2048.game import bitboard as bb
from scripts.add_tuples import add_tuples, add_tuples_to_traces

from tests.helpers import random_boards


def test_added_tuples_start_at_zero_so_values_are_unchanged() -> None:
    network = NTupleNetwork(NETWORKS["8x6"], downgrade_tile=16384)
    rng = np.random.default_rng(0)
    for board in random_boards(50, seed=1):
        network.update(bb.from_array(board), float(rng.normal()))  # some non-zero weights where boards land
    redundant = add_tuples(network)
    assert redundant.tuples == NETWORKS["8x6-redundant"]
    for board in random_boards(50, seed=1):
        packed = bb.from_array(board)
        assert redundant.value(packed) == network.value(packed)

    traces = CoherenceTraces(np.ones(network.weights.size, np.float32), np.ones(network.weights.size, np.float32))
    widened = add_tuples_to_traces(traces, network)
    assert widened.errors.size == redundant.weights.size
    assert widened.abs_errors[: network.weights.size].all() and not widened.abs_errors[network.weights.size :].any()
