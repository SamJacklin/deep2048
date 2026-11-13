import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import NETWORKS, CoherenceTraces, evaluate, update_coherent, weight_indices
from deep_2048.game import bitboard as bb
from scripts.update_gain import update_gain


def board_with(*cells: tuple[int, int]) -> np.uint64:
    board = np.uint64(0)
    for cell, exponent in cells:
        board |= np.uint64(exponent) << np.uint64(4 * cell)
    return board


def test_the_gain_predicts_a_real_update_including_repeated_weights() -> None:
    network = NTupleNetwork(NETWORKS["8x6"], downgrade_tile=16384)
    layout = network.layout
    # One tile in a corner: its symmetries make many tuples read the same weight several times
    board = board_with((0, 3))
    indices = weight_indices(layout, board, board)
    assert len(indices) == 64 and len(np.unique(indices)) < 64
    for traces, tolerance in ((CoherenceTraces.zeros(layout.n_weights), 1e-6), (CoherenceTraces.zeros(layout.n_weights), 1e-3)):
        if tolerance > 1e-6:
            # Large sums, so the update's own error barely moves each weight's coherence of 0.3
            traces.abs_errors[indices] = 1e7
            traces.errors[indices] = 3e6
        network.weights[:] = 0
        gain, repeats, coherence = update_gain(indices, traces.errors, traces.abs_errors, 0.02)
        before = evaluate(network.weights, layout, board, board)
        update_coherent(network.weights, traces, layout, board, board, 0.02, 10.0)
        after = evaluate(network.weights, layout, board, board)
        assert abs((after - before) / 10.0 - gain) <= tolerance * gain
        assert repeats > 1
    assert abs(coherence - 0.3) < 1e-9


def test_indices_follow_downgrading() -> None:
    layout = NTupleNetwork(NETWORKS["8x6"], downgrade_tile=16384).layout
    # 16384 next to 4096 with no 8192: read as 8192 next to 4096
    real = board_with((0, 14), (1, 12), (2, 1))
    read = board_with((0, 13), (1, 12), (2, 1))
    assert bb.downgrade(real, 14) == read
    assert np.array_equal(weight_indices(layout, real, real), weight_indices(layout, read, read))
