import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple import arrival as ar
from deep_2048.agents.ntuple.search import _play_from, play_seeded
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES

TOP, LENGTH = 10, 4  # a chain of 1024, 512, 256 and 128, so small networks can play the test's games
FLOOR = TOP - LENGTH + 1


def board(rows: list[list[int]]) -> np.uint64:
    return bb.from_array(np.array(rows))


def noisy_network(seed: int) -> NTupleNetwork:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(seed).normal(scale=10, size=network.weights.shape)
    network.weights[0] += 200
    return network


# Up, left and right change nothing, so the only legal move, down, takes the chain off its edge
ONLY_DOWN = [[1024, 512, 256, 128], [2, 4, 2, 8], [4, 2, 4, 2], [0, 0, 0, 0]]


def test_chain_reach_counts_tiles_from_the_largest_down() -> None:
    assert ar.chain_reach(board([[1024, 512, 256, 128], [0, 0, 2, 4], [0] * 4, [0] * 4])) == 4
    assert ar.chain_reach(board([[1024, 512, 0, 128], [0, 0, 2, 4], [0] * 4, [0] * 4])) == 2
    assert ar.chain_reach(board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])) == 10
    assert ar.holds_chain(board(ONLY_DOWN), TOP, LENGTH)
    assert not ar.holds_chain(board([[2048, 1024, 512, 256], [128, 0, 0, 0], [0] * 4, [0] * 4]), TOP, LENGTH)  # a larger tile
    assert not ar.holds_chain(board([[1024, 512, 256, 64], [0] * 4, [0] * 4, [0] * 4]), TOP, LENGTH)


def test_formation_is_big_tiles_in_order_from_a_corner_along_an_edge() -> None:
    assert ar.in_formation(board(ONLY_DOWN), FLOOR, LENGTH)
    # every rotation and reflection of a formation is one
    for symmetric in bb.symmetries(board(ONLY_DOWN)):
        assert ar.in_formation(np.uint64(symmetric), FLOOR, LENGTH)
    # a fifth big tile elsewhere doesn't matter, and a cascade stays in formation as it merges
    assert ar.in_formation(board([[1024, 512, 256, 128], [0, 0, 0, 128], [0] * 4, [0] * 4]), FLOOR, LENGTH)
    assert ar.in_formation(board([[1024, 512, 256, 256], [0] * 4, [0] * 4, [0] * 4]), FLOOR, LENGTH)
    assert ar.in_formation(board([[1024, 512, 512, 2], [0] * 4, [0] * 4, [0] * 4]), FLOOR, LENGTH)
    assert ar.in_formation(board([[1024, 1024, 4, 2], [0] * 4, [0] * 4, [0] * 4]), FLOOR, LENGTH)
    assert ar.in_formation(board([[2048, 4, 4, 2], [0] * 4, [0] * 4, [0] * 4]), FLOOR, LENGTH)
    # off the edge, out of order, or with a gap: not a formation
    assert not ar.in_formation(board([[0] * 4, [1024, 512, 256, 128], [0] * 4, [0] * 4]), FLOOR, LENGTH)
    assert not ar.in_formation(board([[1024, 256, 512, 128], [0] * 4, [0] * 4, [0] * 4]), FLOOR, LENGTH)
    assert not ar.in_formation(board([[1024, 512, 256, 2], [0, 0, 0, 128], [0] * 4, [0] * 4]), FLOOR, LENGTH)


def test_placement_classes() -> None:
    assert ar.placement(board(ONLY_DOWN), FLOOR, LENGTH) == ar.IN_ORDER
    assert ar.placement(board([[1024, 256, 512, 128], [0] * 4, [0] * 4, [0] * 4]), FLOOR, LENGTH) == ar.CORNER_ONLY
    assert ar.placement(board([[2, 1024, 512, 256], [0, 0, 0, 128], [0] * 4, [0] * 4]), FLOOR, LENGTH) == ar.OFF_CORNER


def test_a_forced_break_is_recorded_as_forced() -> None:
    network = noisy_network(0)
    fields, break_board, _ = ar.trace_from(network.weights, network.layout, board(ONLY_DOWN), 1, 0.0, TOP, LENGTH, np.random.default_rng(1))
    assert fields[ar.START_PLACEMENT] == ar.IN_ORDER
    assert fields[ar.BREAK_MOVE] == 0 and fields[ar.BREAK_FORCED] == 1
    assert fields[ar.BREAK_EMPTY] == 4 and fields[ar.BREAK_REACH] == 4
    assert break_board == board(ONLY_DOWN)
    assert fields[ar.MOVES_AFTER_BREAK] == fields[ar.STAGE_MOVES]


def test_a_break_with_a_safe_alternative_is_recorded_as_chosen() -> None:
    # With every weight 0, play takes the first legal move with the highest reward. Up changes nothing here, down
    # (action 1) breaks the formation, and right keeps it, so play breaks it by choice.
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    start = board([[1024, 512, 256, 128], [2, 4, 0, 0], [0] * 4, [0] * 4])
    assert bb.move(start, bb.UP)[0] == start and bb.move(start, bb.DOWN)[0] != start
    fields, break_board, _ = ar.trace_from(network.weights, network.layout, start, 1, 0.0, TOP, LENGTH, np.random.default_rng(2))
    assert fields[ar.BREAK_MOVE] == 0 and fields[ar.BREAK_FORCED] == 0
    assert break_board == start


def test_positions_that_start_out_of_formation_record_no_break() -> None:
    network = noisy_network(3)
    start = board([[2, 1024, 512, 256], [0, 0, 0, 128], [0] * 4, [0] * 4])
    fields, break_board, _ = ar.trace_from(network.weights, network.layout, start, 1, 0.0, TOP, LENGTH, np.random.default_rng(4))
    assert fields[ar.START_PLACEMENT] == ar.OFF_CORNER
    assert fields[ar.BREAK_MOVE] == -1 and fields[ar.BREAK_FORCED] == -1 and break_board == 0


def test_a_trace_plays_the_game_play_from_plays() -> None:
    """Until the larger tile is made, a trace is play_from's game: same moves, same end."""
    start = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])
    for seed in range(20):
        network = noisy_network(seed)
        for depth in (1, 2):
            fields, _, last = ar.trace_from(network.weights, network.layout, start, depth, 0.0, TOP, LENGTH, np.random.default_rng(seed))
            _, max_tile, moves = _play_from(network.weights, network.layout, start, depth, 0.0, np.random.default_rng(seed))
            assert fields[ar.CONVERTED] == (max_tile > 1 << TOP)
            assert fields[ar.STAGE_MOVES] <= moves
            if not fields[ar.CONVERTED]:
                assert fields[ar.STAGE_MOVES] == moves and bb.max_exponent(last) == TOP
            assert fields[ar.REACH] >= 10 or fields[ar.CONVERTED] or fields[ar.REACH] >= ar.chain_reach(start)


def test_arrival_board_is_the_first_board_holding_the_chain() -> None:
    network = noisy_network(5)
    # A chain of two tiles from 16 is easy for a random network to reach
    for seed in range(10):
        found, moves, score = ar.arrival_board(network.weights, network.layout, 1, 0.0, 4, 2, np.random.default_rng(seed))
        _, max_tile = play_seeded(network.weights, network.layout, 1, 0.0, np.random.default_rng(seed))
        if found != 0:
            assert ar.holds_chain(found, 4, 2) and moves > 0 and score > 0 and max_tile >= 16


def test_tail_boards_are_the_boards_of_the_traced_game() -> None:
    start = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])
    offsets = np.array([1, 3, 10_000], dtype=np.int64)
    for seed in range(10):
        network = noisy_network(seed)
        fields, _, _ = ar.trace_from(network.weights, network.layout, start, 1, 0.0, TOP, LENGTH, np.random.default_rng(seed))
        converted, moves, boards = ar.tail_boards(network.weights, network.layout, start, 1, 0.0, TOP, offsets, np.random.default_rng(seed))
        assert (converted, moves) == (fields[ar.CONVERTED], fields[ar.STAGE_MOVES])
        assert boards[2] == 0  # the stage was shorter than that
        if moves >= 3:
            assert boards[0] != 0 and boards[1] != 0
            # playing on from the board three moves before the end with nothing left to chance can't be checked here,
            # but the board one move before the end must have a legal move
            assert not bb.is_game_over(boards[0])
        if moves == 1:
            assert boards[0] == start


def test_converts_agrees_with_a_trace() -> None:
    start = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])
    for seed in range(10):
        network = noisy_network(seed)
        fields, _, _ = ar.trace_from(network.weights, network.layout, start, 1, 0.0, TOP, LENGTH, np.random.default_rng(seed))
        assert ar.converts(network.weights, network.layout, start, 1, 0.0, TOP, np.random.default_rng(seed)) == fields[ar.CONVERTED]


def test_ending_at_once_counts_the_spawns_that_leave_no_move() -> None:
    # One empty cell: a 2 there leaves no move, a 4 merges with the 4 beside it
    after = board([[0, 4, 8, 16], [16, 8, 4, 32], [8, 4, 16, 64], [1024, 512, 256, 128]])
    assert abs(ar.ends_at_once(after) - 0.9) < 1e-9
    assert ar.ends_at_once(board([[2, 4, 8, 16], [16, 8, 4, 32], [8, 4, 16, 64], [1024, 512, 256, 128]])) == 0.0  # no spawn to come
    assert ar.ends_at_once(board([[0, 0, 8, 16], [16, 8, 4, 32], [8, 4, 16, 64], [1024, 512, 256, 128]])) == 0.0
    # From the full board below, up and down merge the 2s: down keeps the formation, and a 2 spawned after it ends the game
    full = board([[2, 4, 8, 16], [2, 8, 4, 32], [8, 4, 16, 64], [1024, 512, 256, 128]])
    assert abs(ar.safest_keeping_move(full, FLOOR, LENGTH) - 0.9) < 1e-9
    assert ar.safest_keeping_move(board(ONLY_DOWN), FLOOR, LENGTH) == -1.0


def test_finish_board_is_the_first_board_near_the_full_sum() -> None:
    from deep_2048.agents.ntuple.search import tile_sum

    start = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])  # sums to 2046
    for seed in range(10):
        network = noisy_network(seed)
        # Measured against a 4096: a mark the start has already reached returns the start itself
        assert ar.finish_board(network.weights, network.layout, start, 1, 0.0, TOP + 1, 4096 - 2046, np.random.default_rng(seed)) == start
        # A mark 14 further on is reached by the first board at or past it, since spawns add 2 or 4
        found = ar.finish_board(network.weights, network.layout, start, 1, 0.0, TOP + 1, 4096 - 2060, np.random.default_rng(seed))
        if found != 0:
            assert 2060 <= tile_sum(found) < 2064 and bb.max_exponent(found) <= TOP + 1


def test_play_with_finish_is_play_from_when_the_last_stretch_is_searched_no_deeper() -> None:
    start = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])
    for seed in range(8):
        network = noisy_network(seed)
        for depth in (1, 2):
            expected = _play_from(network.weights, network.layout, start, depth, 0.0, np.random.default_rng(seed))
            score, tile, moves, deep = ar.play_with_finish(network.weights, network.layout, start, depth, depth, 0.0, TOP, 64, False, np.random.default_rng(seed))
            assert (score, tile, moves) == expected
            assert deep >= 1  # the start is already within 64 of a 2048's sum
        # and a whole seeded game matches play_seeded
        full = ar.play_seeded_with_finish(network.weights, network.layout, 1, 1, 0.0, TOP, 64, np.random.default_rng(seed))
        assert full[:2] == play_seeded(network.weights, network.layout, 1, 0.0, np.random.default_rng(seed))


def test_play_with_finish_searches_deep_only_in_the_last_stretch() -> None:
    network = noisy_network(1)
    far = board([[1024, 512, 0, 0], [0] * 4, [0] * 4, [0] * 4])  # 512 short of a 2048's sum
    _, _, moves, deep = ar.play_with_finish(network.weights, network.layout, far, 1, 3, 0.0, TOP, 64, True, np.random.default_rng(0))
    assert deep < moves
    near = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])
    _, tile, moves, deep = ar.play_with_finish(network.weights, network.layout, near, 1, 3, 0.0, TOP, 64, True, np.random.default_rng(0))
    assert deep == moves or tile > 1 << TOP  # every move is in the stretch until the 2048 is made
