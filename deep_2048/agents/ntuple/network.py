"""N-tuple network from Szubert & Jaśkowski (2014).

A board's value is a sum of weights from lookup tables. Each tuple is a group of cells whose
exponents index into that tuple's table, and each tuple is read in all 8 rotations and
reflections of the board using the same table. Cells are numbered 4 * row + col.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
from numba import njit, types
from numba.core.errors import NumbaIRAssumptionWarning
from numba.extending import overload

from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.game import bitboard as bb

# numba's pedantic IR check flags a variable from the inlined stage-downgrading rule; the compiled code is correct
# (tests/test_stage_downgrade.py), so the warning is only noise in every training log
warnings.filterwarnings("ignore", category=NumbaIRAssumptionWarning)

TUPLES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)

# Weights in the scale table: counts of five tile sizes, each capped at 3, in 2 bits each
SCALE_WEIGHTS = 1 << 10

NetworkName = Literal["2x4-2x6", "4x6", "8x6", "8x6-redundant"]
DEFAULT_NETWORK: NetworkName = "2x4-2x6"

# Layouts as given in Guei's thesis (arXiv 2212.11087): 4x6 is Yeh et al.'s (2016), also Jaśkowski's 42-33
# (2018), and 8x6 is Matsuzaki's, used by the state of the art.
NETWORKS: dict[NetworkName, tuple[tuple[int, ...], ...]] = {
    "2x4-2x6": TUPLES,
    "4x6": ((0, 1, 2, 3, 4, 5), (4, 5, 6, 7, 8, 9), (0, 1, 2, 4, 5, 6), (4, 5, 6, 8, 9, 10)),
    "8x6": (
        (0, 1, 2, 4, 5, 6),
        (4, 5, 6, 7, 8, 9),
        (0, 1, 2, 3, 4, 5),
        (2, 3, 4, 5, 6, 9),
        (0, 1, 2, 5, 9, 10),
        (3, 4, 5, 6, 7, 8),
        (1, 3, 4, 5, 6, 7),
        (0, 1, 4, 8, 9, 10),
    ),
}
# Jaśkowski's (2018) redundant encoding: small tuples read alongside the large ones (straight 4s, 2x2 squares and
# straight 3s), which generalise across positions the large tuples see rarely. They add about 340k weights.
REDUNDANT_TUPLES: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 4, 5), (1, 2, 5, 6), (5, 6, 9, 10), (0, 1, 2), (4, 5, 6))
NETWORKS["8x6-redundant"] = NETWORKS["8x6"] + REDUNDANT_TUPLES
assert set(NETWORKS) == set(get_args(NetworkName))


def network_name(tuples: tuple[tuple[int, ...], ...]) -> NetworkName | None:
    """Return the name of a known network layout, or None for a custom one."""
    return next((name for name, layout in NETWORKS.items() if layout == tuples), None)


def _split_into_runs(cells: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Split a tuple's cells into runs of adjacent cells in the same row, as (first cell, length, position in tuple)."""
    runs: list[tuple[int, int, int]] = []
    start = 0
    for k in range(1, len(cells) + 1):
        run_ends = k == len(cells) or cells[k] != cells[k - 1] + 1 or cells[k] // 4 != cells[start] // 4
        if run_ends:
            runs.append((cells[start], k - start, start))
            start = k
    return runs


class TupleLayout(NamedTuple):
    """How to read each tuple's weight index from a board.

    Each tuple's cells are split into runs of adjacent cells in the same row, so a run is read
    in one step: run j of tuple t is (board >> shifts[t, j]) & masks[t, j], placed at bit
    dests[t, j] of the index. Tuple t's weights start at offsets[t] in the flat weight array.
    """

    offsets: np.ndarray
    n_runs: np.ndarray
    shifts: np.ndarray
    masks: np.ndarray
    dests: np.ndarray
    n_weights: int
    sharing: "ScaleSharing | None" = None

    @classmethod
    def build(
        cls,
        tuples: tuple[tuple[int, ...], ...],
        downgrade_tile: int | None = None,
        scale_feature: bool = False,
        scale_when_downgraded: bool = False,
        stage_tile: int | None = None,
        stage_downgrade_tile: int | None = None,
        scale_rate: float = 1.0,
        scale_skip_empty: bool = False,
        scale_below_tile: int | None = None,
        scale_by_state: bool = False,
    ) -> "TupleLayout":
        max_runs = max(len(t) for t in tuples)
        offsets = np.zeros(len(tuples), dtype=np.int64)
        n_runs = np.zeros(len(tuples), dtype=np.int64)
        shifts = np.zeros((len(tuples), max_runs), dtype=np.uint64)
        masks = np.zeros((len(tuples), max_runs), dtype=np.uint64)
        dests = np.zeros((len(tuples), max_runs), dtype=np.uint64)

        table_start = 0
        for t, cells in enumerate(tuples):
            runs = _split_into_runs(cells)
            n_runs[t] = len(runs)
            for j, (first_cell, length, position) in enumerate(runs):
                shifts[t, j] = 4 * first_cell
                masks[t, j] = (1 << (4 * length)) - 1
                dests[t, j] = 4 * position
            offsets[t] = table_start
            table_start += 16 ** len(cells)

        # A second stage is a second copy of every tuple's table, after the first
        stage_size = table_start if stage_tile is not None else 0
        table_start += stage_size
        sharing = None
        if stage_downgrade_tile is not None and stage_tile is None:
            raise ValueError("stage_downgrade_tile needs a second stage (stage_tile)")
        if (scale_rate != 1.0 or scale_skip_empty or scale_below_tile is not None or scale_by_state) and not scale_feature:
            raise ValueError("scale_rate, scale_skip_empty, scale_below_tile and scale_by_state need the scale feature")
        if scale_below_tile is not None and (scale_below_tile & (scale_below_tile - 1) or scale_below_tile < 2):
            raise ValueError(f"scale_below_tile must be a power of two, not {scale_below_tile}")
        if downgrade_tile is not None or scale_feature or stage_tile is not None:
            downgrade_exponent = 0 if downgrade_tile is None else downgrade_tile.bit_length() - 1
            sharing = ScaleSharing(
                downgrade_exponent=downgrade_exponent,
                scale_offset=table_start if scale_feature else -1,
                scale_when_downgraded=scale_when_downgraded,
                stage_exponent=0 if stage_tile is None else stage_tile.bit_length() - 1,
                stage_size=stage_size,
                # 0 turns downgrading off in the second stage; None (-1) reads it as the first stage does
                stage_downgrade_exponent=-1 if stage_downgrade_tile is None else max(stage_downgrade_tile.bit_length() - 1, 0),
                scale_rate=scale_rate,
                scale_skip_empty=scale_skip_empty,
                scale_below_exponent=0 if scale_below_tile is None else scale_below_tile.bit_length() - 1,
                scale_by_state=scale_by_state,
            )
        return cls(
            offsets=offsets,
            n_runs=n_runs,
            shifts=shifts,
            masks=masks,
            dests=dests,
            n_weights=table_start + (SCALE_WEIGHTS if scale_feature else 0),
            sharing=sharing,
        )


class ScaleSharing(NamedTuple):
    """How a layout shares late-game knowledge across tile scales. A NamedTuple so numba can use it.

    Boards with a 2**downgrade_exponent tile or larger are read by the tuples with their tiles above
    the largest gap halved (0 turns this off). If scale_offset isn't -1, one more weight is added
    from a table there, indexed by how many tiles of each size from 2048 to 32768 the real board has:
    for every board, or only for boards that were downgraded if scale_when_downgraded is set.
    """

    downgrade_exponent: int
    scale_offset: int
    scale_when_downgraded: bool = False
    # Boards whose real largest tile is 2**stage_exponent or more read the second stage's tables, stage_size
    # weights further on (0 means a single stage)
    stage_exponent: int = 0
    stage_size: int = 0
    # Above 0, the second stage plays only a window: boards whose largest tile is exactly 2**stage_exponent and whose
    # tiles sum to at least stage_sum, the last stretch before the next tile (the finisher's stage). Tiles only ever sum
    # to more, so a game enters the window once and leaves it when the next tile is made.
    stage_sum: int = 0
    # The second stage's own downgrading: boards read in it are downgraded from 2**stage_downgrade_exponent (0: not
    # at all) instead of from 2**downgrade_exponent; -1 reads them as the first stage does
    stage_downgrade_exponent: int = -1
    # The scale weight's own step, as a share of the tuples' rate: a board reads 64 tuple weights but one scale weight,
    # so at 1 that single, heavily shared weight takes as big a step as each tuple weight and can soak up the error
    scale_rate: float = 1.0
    # Boards with no tile of 2048 or more read no scale weight: their entry would be one bias shared by every early board
    scale_skip_empty: bool = False
    # Above 0, boards whose real largest tile is 2**scale_below_exponent or more read no scale weight: the table only
    # gives context below that tile, so it can't learn a value offset for the boards that downgrading reads at a smaller scale
    scale_below_exponent: int = 0
    # Read the scale weight from the board the move is made from (as the stage is chosen) instead of the afterstate:
    # every move of a decision then reads the same entry, so the table can't make one move look better than another,
    # only take up the part of the value that the large tiles already on the board explain
    scale_by_state: bool = False
    # Above 0, a second window with tables of its own, 2 * stage_size weights on: boards whose largest tile is exactly
    # 2**window2_exponent and whose tiles sum to at least window2_sum (a second finisher, below the first's tile)
    window2_exponent: int = 0
    window2_sum: int = 0


@njit
def _board_sum(board: np.uint64) -> int:
    """The sum of the board's tiles. A call rather than inlined, which numba's inliner mishandles inside the stage rule."""
    total = 0
    for i in range(bb.NUM_CELLS):
        exponent = bb.get_cell(board, i)
        if exponent > 0:
            total += 1 << exponent
    return total


@njit(inline="always")
def _weight_index(layout: TupleLayout, t: int, board: np.uint64) -> int:
    index = np.uint64(0)
    for j in range(layout.n_runs[t]):
        index |= ((board >> layout.shifts[t, j]) & layout.masks[t, j]) << layout.dests[t, j]
    return int(layout.offsets[t] + np.int64(index))


def _tuple_board(layout: TupleLayout, board: np.uint64) -> np.uint64:
    """Return the board the tuples read: downgraded if the layout's sharing asks for it. Numba only."""
    raise NotImplementedError


def _scale_index(layout: TupleLayout, board: np.uint64, tuple_board: np.uint64, state: np.uint64) -> int:
    """Return the board's scale weight index, or -1 if it gets no scale weight. Numba only."""
    raise NotImplementedError


def _stage_tuple_board(layout: TupleLayout, board: np.uint64, stage: int) -> np.uint64:
    """Return the board the tuples read in the given stage (its tables' offset): downgraded as that stage asks. Numba only."""
    raise NotImplementedError


def _stage_offset(layout: TupleLayout, board: np.uint64) -> int:
    """Return where the board's stage's tables start, from its real (not downgraded) largest tile. Numba only."""
    raise NotImplementedError


def _has_sharing(layout_type: types.NamedTuple) -> bool:
    return not isinstance(layout_type.types[layout_type.fields.index("sharing")], types.NoneType)


# These dispatch on the layout's type rather than branching on its values. A runtime check here cost
# about 15% even with sharing off, whereas this compiles plain layouts exactly as before.
@overload(_tuple_board, inline="always")
def _tuple_board_overload(layout, board):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout, board: board

    def downgraded(layout, board):  # type: ignore[no-untyped-def]
        if layout.sharing.downgrade_exponent == 0:
            return board
        return bb.downgrade(board, layout.sharing.downgrade_exponent)

    return downgraded


@njit(inline="always")
def _downgrade_from(board: np.uint64, exponent: int) -> np.uint64:
    """The board downgraded from 2**exponent, or unchanged for 0."""
    return board if exponent == 0 else bb.downgrade(board, exponent)


@overload(_stage_tuple_board, inline="always")
def _stage_tuple_board_overload(layout, board, stage):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout, board, stage: board

    def downgraded(layout, board, stage):  # type: ignore[no-untyped-def]
        # Chosen without a branch: an if here trips a pedantic numba warning once inlined into the training loop
        own = int(stage != 0) * int(layout.sharing.stage_downgrade_exponent >= 0)
        return _downgrade_from(board, own * layout.sharing.stage_downgrade_exponent + (1 - own) * layout.sharing.downgrade_exponent)

    return downgraded


@overload(_scale_index, inline="always")
def _scale_index_overload(layout, board, tuple_board, state):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout, board, tuple_board, state: -1

    def scale_index(layout, board, tuple_board, state):  # type: ignore[no-untyped-def]
        if layout.sharing.scale_offset < 0:
            return -1
        if layout.sharing.scale_when_downgraded and tuple_board == board:
            return -1
        counted = state if layout.sharing.scale_by_state else board
        if layout.sharing.scale_below_exponent > 0 and bb.max_exponent(counted) >= layout.sharing.scale_below_exponent:
            return -1
        counts = bb.large_tile_counts(counted)
        if layout.sharing.scale_skip_empty and counts == 0:
            return -1
        return layout.sharing.scale_offset + counts

    return scale_index


def _scale_rate(layout: TupleLayout) -> float:
    """The scale weight's share of the tuples' rate (1 without scale sharing). Numba only."""
    raise NotImplementedError


@overload(_scale_rate, inline="always")
def _scale_rate_overload(layout):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout: 1.0
    return lambda layout: layout.sharing.scale_rate


@overload(_stage_offset, inline="always")
def _stage_offset_overload(layout, board):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout, board: 0

    def stage_offset(layout, board):  # type: ignore[no-untyped-def]
        if layout.sharing.window2_exponent > 0 and bb.max_exponent(board) == layout.sharing.window2_exponent and _board_sum(board) >= layout.sharing.window2_sum:
            return 2 * layout.sharing.stage_size
        if layout.sharing.stage_exponent == 0 or bb.max_exponent(board) < layout.sharing.stage_exponent:
            return 0
        if layout.sharing.stage_sum > 0 and (bb.max_exponent(board) > layout.sharing.stage_exponent or _board_sum(board) < layout.sharing.stage_sum):
            return 0
        return layout.sharing.stage_size

    return stage_offset


class ResidualLayout(NamedTuple):
    """A frozen network's layout plus a late-game correction, for weights holding the network's then the correction's.

    Once the real afterstate's largest tile reaches 2**gate_exponent, its value adds `scale` times the correction:
    weights[constant_index] plus the residual tuples' weights, which read the real board's tile exponents (no
    downgrading, unlike the base). The gate is decided by the afterstate being valued, wherever it's valued (greedy
    play, training or search), or with `gate_on_state` by the board the move is made from, so every move of a
    decision is judged alike. With `clamp`, a corrected value below 0 counts as 0, as the end of a game does.
    """

    base: TupleLayout
    residual: TupleLayout  # offsets already past the base's weights
    gate_exponent: int
    constant_index: int
    scale: float = 1.0
    gate_on_state: bool = False
    clamp: bool = False


# Weights in a work-area table: 8 cells of 3 bits each (empty, or a tile up to 128)
AREA_WEIGHTS = 1 << 24


class AreaLayout(NamedTuple):
    """A two-stage layout plus a work-area table for its second stage, in weights[area_offset:].

    In the second stage (chosen by the board a move is made from, as the stage's tables are), an afterstate's value
    adds one weight for each symmetry of the real board whose top two rows hold nothing above 128: those 8 cells read
    as one pattern. Late in a build the large tiles fill one half of the board, so the other half is everything that
    changes, and each 6-cell tuple sees only part of it.
    """

    base: TupleLayout
    area_offset: int


@njit
def area_index(board: np.uint64) -> int:
    """The board's top two rows as one index, 3 bits a cell, or -1 if any of them holds a tile above 128."""
    half = board & np.uint64(0xFFFFFFFF)
    if half & np.uint64(0x88888888):
        return -1
    index = 0
    for i in range(8):
        index |= int((half >> np.uint64(4 * i)) & np.uint64(7)) << (3 * i)
    return index


@njit(inline="always")
def evaluate(weights: np.ndarray, layout: TupleLayout, board: np.uint64, state: np.uint64) -> float:
    """The afterstate's value: the network's sum of weights, plus the late-game correction for a ResidualLayout."""
    return _value(weights, layout, board, state)


def _value(weights: np.ndarray, layout: TupleLayout, board: np.uint64, state: np.uint64) -> float:
    """Dispatch evaluate() on the layout's type. Numba only."""
    raise NotImplementedError


def _is_residual(layout_type: types.Type) -> bool:
    return isinstance(layout_type, types.BaseNamedTuple) and "residual" in layout_type.fields


def _is_area(layout_type: types.Type) -> bool:
    return isinstance(layout_type, types.BaseNamedTuple) and "area_offset" in layout_type.fields


@overload(_value, inline="always")
def _value_overload(weights, layout, board, state):  # type: ignore[no-untyped-def]
    if _is_area(layout):
        return lambda weights, layout, board, state: _area_value(weights, layout, board, state)
    if not _is_residual(layout):
        return lambda weights, layout, board, state: _tuple_value(weights, layout, board, state)

    return lambda weights, layout, board, state: _corrected_value(weights, layout, board, state)


@njit
def _area_value(weights: np.ndarray, layout: AreaLayout, board: np.uint64, state: np.uint64) -> float:
    """The two-stage value, plus the work-area weights in the second stage. A call rather than inlined, as
    _corrected_value is."""
    total = _tuple_value(weights, layout.base, board, state)
    if _stage_offset(layout.base, state) != 0:
        for symmetric_board in bb.symmetries(board):
            index = area_index(symmetric_board)
            if index >= 0:
                total += weights[layout.area_offset + index]
    return total


def fill_indices(layout: "TupleLayout | AreaLayout", board: np.uint64, state: np.uint64, indices: np.ndarray) -> int:
    """fill_weight_indices() for either layout: an AreaLayout's work-area weights follow its tuples'. Numba only."""
    raise NotImplementedError


@overload(fill_indices)
def _fill_indices_overload(layout, board, state, indices):  # type: ignore[no-untyped-def]
    if not _is_area(layout):
        return lambda layout, board, state, indices: fill_weight_indices(layout, board, state, indices)

    def with_area(layout, board, state, indices):  # type: ignore[no-untyped-def]
        k = fill_weight_indices(layout.base, board, state, indices)
        if _stage_offset(layout.base, state) != 0:
            for symmetric_board in bb.symmetries(board):
                index = area_index(symmetric_board)
                if index >= 0:
                    indices[k] = layout.area_offset + index
                    k += 1
        return k

    return with_area


@njit
def _corrected_value(weights: np.ndarray, layout: ResidualLayout, board: np.uint64, state: np.uint64) -> float:
    """The base's value, plus the constant and residual past the gate. A call rather than inlined, which numba's
    inliner mishandles when one function values several boards."""
    total = _tuple_value(weights, layout.base, board, state)
    if bb.max_exponent(state if layout.gate_on_state else board) >= layout.gate_exponent:
        total += layout.scale * (weights[layout.constant_index] + _residual_value(weights, layout.residual, board))
        if layout.clamp and total < 0.0:
            total = 0.0
    return total


@njit
def _residual_value(weights: np.ndarray, layout: TupleLayout, board: np.uint64) -> float:
    """Sum the residual tuples' weights over the real board's symmetries. A separate call, so numba doesn't inline the
    same loop twice into one function."""
    total = 0.0
    for symmetric_board in bb.symmetries(board):
        for t in range(len(layout.offsets)):
            total += weights[_weight_index(layout, t, symmetric_board)]
    return total


@njit(inline="always")
def _tuple_value(weights: np.ndarray, layout: TupleLayout, board: np.uint64, state: np.uint64) -> float:
    """Sum the weights each tuple reads from each symmetry of the board, plus the scale weight if any."""
    stage = _stage_offset(layout, state)
    tuple_board = _stage_tuple_board(layout, board, stage)
    boards = bb.symmetries(tuple_board)
    total = 0.0
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            total += weights[stage + _weight_index(layout, t, symmetric_board)]
    scale = _scale_index(layout, board, tuple_board, state)
    if scale >= 0:
        total += weights[scale]
    return total


@njit
def weight_indices(layout: TupleLayout, board: np.uint64, state: np.uint64) -> np.ndarray:
    """Return the index of every weight evaluate() reads for this board, repeats included, as an update touches them."""
    indices = np.empty(len(layout.offsets) * bb.NUM_SYMMETRIES + 1, dtype=np.int64)
    return indices[: fill_weight_indices(layout, board, state, indices)]


@njit(inline="always")
def fill_weight_indices(layout: TupleLayout, board: np.uint64, state: np.uint64, indices: np.ndarray) -> int:
    """Write weight_indices() into `indices` (room for one per tuple and symmetry, plus one) and return how many there are."""
    stage = _stage_offset(layout, state)
    tuple_board = _stage_tuple_board(layout, board, stage)
    boards = bb.symmetries(tuple_board)
    k = 0
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            indices[k] = stage + _weight_index(layout, t, symmetric_board)
            k += 1
    scale = _scale_index(layout, board, tuple_board, state)
    if scale >= 0:
        indices[k] = scale
        k += 1
    return k


@njit(inline="always")
def update(weights: np.ndarray, layout: TupleLayout, board: np.uint64, state: np.uint64, delta: float) -> None:
    """Add delta to every weight read for this board."""
    stage = _stage_offset(layout, state)
    tuple_board = _stage_tuple_board(layout, board, stage)
    boards = bb.symmetries(tuple_board)
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            weights[stage + _weight_index(layout, t, symmetric_board)] += delta
    scale = _scale_index(layout, board, tuple_board, state)
    if scale >= 0:
        weights[scale] += delta * _scale_rate(layout)


class CoherenceTraces(NamedTuple):
    """Running sums of each weight's errors and their sizes, for temporal coherence learning. A NamedTuple so numba can use it.

    Built by zeros() or from_arrays(), the sums are float64 and each weight's two sit side by side in one array:
    - float32 sums of a much-used weight grow so large that a typical error rounds to nothing when added,
      freezing its rate (25,000 of the long run's weights had reached that by 1.5M games);
    - side by side, an update reads both sums from one cache line instead of two.
    """

    errors: np.ndarray
    abs_errors: np.ndarray
    # Each update first shrinks the weight's sums by this factor, so its rate follows its recent errors rather than
    # its whole history (1 keeps everything, as in Beal & Smith)
    decay: float = 1.0

    @classmethod
    def zeros(cls, n_weights: int) -> "CoherenceTraces":
        return cls.from_arrays(np.zeros(n_weights), np.zeros(n_weights))

    @classmethod
    def from_arrays(cls, errors: np.ndarray, abs_errors: np.ndarray, decay: float = 1.0) -> "CoherenceTraces":
        """Return traces holding these sums as float64, side by side per weight."""
        sums = np.empty((len(errors), 2))
        sums[:, 0], sums[:, 1] = errors, abs_errors
        return cls(sums[:, 0], sums[:, 1], decay)


@njit(inline="always")
def update_coherent(weights: np.ndarray, traces: CoherenceTraces, layout: TupleLayout, board: np.uint64, state: np.uint64, learning_rate: float, error: float) -> None:
    """Move each weight read for this board by learning_rate * coherence * error.

    A weight's coherence is |sum of its errors| / sum of their sizes: near 1 while its errors keep
    pointing the same way, and near 0 once they cancel out. A weight with no history learns at the full rate.
    """
    stage = _stage_offset(layout, state)
    tuple_board = _stage_tuple_board(layout, board, stage)
    boards = bb.symmetries(tuple_board)
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            _update_coherent_weight(weights, traces, stage + _weight_index(layout, t, symmetric_board), learning_rate, error)
    scale = _scale_index(layout, board, tuple_board, state)
    if scale >= 0:
        _update_coherent_weight(weights, traces, scale, learning_rate * _scale_rate(layout), error)


@njit(inline="always")
def _update_coherent_weight(weights: np.ndarray, traces: CoherenceTraces, i: int, learning_rate: float, error: float) -> None:
    coherence = abs(traces.errors[i]) / traces.abs_errors[i] if traces.abs_errors[i] > 0 else 1.0
    weights[i] += learning_rate * coherence * error
    traces.errors[i] = traces.decay * traces.errors[i] + error
    traces.abs_errors[i] = traces.decay * traces.abs_errors[i] + abs(error)


class NTupleNetwork:
    def __init__(
        self,
        tuples: tuple[tuple[int, ...], ...] = TUPLES,
        weights: np.ndarray | None = None,
        downgrade_tile: int | None = None,
        scale_feature: bool = False,
        scale_when_downgraded: bool = False,
        stage_tile: int | None = None,
        stage_downgrade_tile: int | None = None,
        scale_rate: float = 1.0,
        scale_skip_empty: bool = False,
        scale_below_tile: int | None = None,
        scale_by_state: bool = False,
    ) -> None:
        self.tuples = tuples
        self.downgrade_tile = downgrade_tile
        self.scale_feature = scale_feature
        self.scale_when_downgraded = scale_when_downgraded
        self.stage_tile = stage_tile
        self.stage_downgrade_tile = stage_downgrade_tile
        self.scale_rate = scale_rate
        self.scale_skip_empty = scale_skip_empty
        self.scale_below_tile = scale_below_tile
        self.scale_by_state = scale_by_state
        self.layout = TupleLayout.build(tuples, downgrade_tile, scale_feature, scale_when_downgraded, stage_tile, stage_downgrade_tile, scale_rate, scale_skip_empty,
                                        scale_below_tile, scale_by_state)
        self.weights = np.zeros(self.layout.n_weights, dtype=np.float32) if weights is None else weights

    @property
    def features(self) -> dict[str, Any]:
        """The scale-sharing options, as saved with a checkpoint."""
        return {
            "downgrade_tile": self.downgrade_tile,
            "scale_feature": self.scale_feature,
            "scale_when_downgraded": self.scale_when_downgraded,
            "stage_tile": self.stage_tile,
        } | ({} if self.stage_downgrade_tile is None else {"stage_downgrade_tile": self.stage_downgrade_tile}) \
          | ({} if self.scale_rate == 1.0 else {"scale_rate": self.scale_rate}) | ({"scale_skip_empty": True} if self.scale_skip_empty else {}) \
          | ({} if self.scale_below_tile is None else {"scale_below_tile": self.scale_below_tile}) | ({"scale_by_state": True} if self.scale_by_state else {})

    @property
    def n_features(self) -> int:
        return len(self.tuples) * bb.NUM_SYMMETRIES

    def value(self, board: np.uint64, state: np.uint64 | None = None) -> float:
        """The afterstate's value, read in the stage of `state`, the board the move was made from (default: the board itself)."""
        return float(evaluate(self.weights, self.layout, np.uint64(board), np.uint64(board if state is None else state)))

    def update(self, board: np.uint64, delta: float) -> None:
        update(self.weights, self.layout, np.uint64(board), np.uint64(board), delta)

    def save(self, path: str | Path, metadata: CheckpointMetadata, traces: CoherenceTraces | None = None, extra: dict[str, Any] | None = None, compress: bool = False) -> None:
        """Save the network, plus its temporal coherence traces if it's being trained with them and any `extra` arrays
        (another optimiser's state). `compress` trades a slower save for a smaller file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features = json.dumps(self.features)
        arrays: dict[str, Any] = {"weights": self.weights, "tuples": json.dumps(self.tuples), "features": features, "metadata": metadata.model_dump_json()}
        if traces is not None:
            # Stored as float32 to keep checkpoints small: rounding a sum once costs nothing, only accumulating in float32 does
            arrays |= {"coherence_errors": traces.errors.astype(np.float32), "coherence_abs_errors": traces.abs_errors.astype(np.float32)}
        arrays |= extra or {}
        # Write then rename so a Ctrl-C mid-save can't leave a broken checkpoint
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            (np.savez_compressed if compress else np.savez)(f, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["NTupleNetwork", CheckpointMetadata]:
        """Load a checkpoint and return (network, metadata)."""
        with np.load(path) as data:
            tuples = tuple(tuple(t) for t in json.loads(str(data["tuples"])))
            # Checkpoints from before scale sharing have no features entry
            features = json.loads(str(data["features"])) if "features" in data else {}
            network = cls(tuples=tuples, weights=data["weights"], **features)
            metadata = CheckpointMetadata.model_validate_json(str(data["metadata"]))
        return network, metadata


def load_traces(path: str | Path) -> CoherenceTraces | None:
    """Load a checkpoint's temporal coherence traces, or None if it was trained without them."""
    with np.load(path) as data:
        if "coherence_errors" not in data:
            return None
        return CoherenceTraces.from_arrays(data["coherence_errors"], data["coherence_abs_errors"])
