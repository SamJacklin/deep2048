"""N-tuple network from Szubert & Jaśkowski (2014).

A board's value is a sum of weights from lookup tables. Each tuple is a group of cells whose
exponents index into that tuple's table, and each tuple is read in all 8 rotations and
reflections of the board using the same table. Cells are numbered 4 * row + col.
"""

import json
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
from numba import njit, types
from numba.extending import overload

from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.game import bitboard as bb

TUPLES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)

# Weights in the scale table: counts of five tile sizes, each capped at 3, in 2 bits each
SCALE_WEIGHTS = 1 << 10

NetworkName = Literal["2x4-2x6", "4x6", "8x6"]
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
        if downgrade_tile is not None or scale_feature or stage_tile is not None:
            downgrade_exponent = 0 if downgrade_tile is None else downgrade_tile.bit_length() - 1
            sharing = ScaleSharing(
                downgrade_exponent=downgrade_exponent,
                scale_offset=table_start if scale_feature else -1,
                scale_when_downgraded=scale_when_downgraded,
                stage_exponent=0 if stage_tile is None else stage_tile.bit_length() - 1,
                stage_size=stage_size,
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


@njit(inline="always")
def _weight_index(layout: TupleLayout, t: int, board: np.uint64) -> int:
    index = np.uint64(0)
    for j in range(layout.n_runs[t]):
        index |= ((board >> layout.shifts[t, j]) & layout.masks[t, j]) << layout.dests[t, j]
    return int(layout.offsets[t] + np.int64(index))


def _tuple_board(layout: TupleLayout, board: np.uint64) -> np.uint64:
    """Return the board the tuples read: downgraded if the layout's sharing asks for it. Numba only."""
    raise NotImplementedError


def _scale_index(layout: TupleLayout, board: np.uint64, tuple_board: np.uint64) -> int:
    """Return the board's scale weight index, or -1 if it gets no scale weight. Numba only."""
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


@overload(_scale_index, inline="always")
def _scale_index_overload(layout, board, tuple_board):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout, board, tuple_board: -1

    def scale_index(layout, board, tuple_board):  # type: ignore[no-untyped-def]
        if layout.sharing.scale_offset < 0:
            return -1
        if layout.sharing.scale_when_downgraded and tuple_board == board:
            return -1
        return layout.sharing.scale_offset + bb.large_tile_counts(board)

    return scale_index


@overload(_stage_offset, inline="always")
def _stage_offset_overload(layout, board):  # type: ignore[no-untyped-def]
    if not _has_sharing(layout):
        return lambda layout, board: 0

    def stage_offset(layout, board):  # type: ignore[no-untyped-def]
        if layout.sharing.stage_exponent == 0 or bb.max_exponent(board) < layout.sharing.stage_exponent:
            return 0
        return layout.sharing.stage_size

    return stage_offset


@njit(inline="always")
def evaluate(weights: np.ndarray, layout: TupleLayout, board: np.uint64, state: np.uint64) -> float:
    """Sum the weights each tuple reads from each symmetry of the board, plus the scale weight if any."""
    tuple_board = _tuple_board(layout, board)
    stage = _stage_offset(layout, state)
    boards = bb.symmetries(tuple_board)
    total = 0.0
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            total += weights[stage + _weight_index(layout, t, symmetric_board)]
    scale = _scale_index(layout, board, tuple_board)
    if scale >= 0:
        total += weights[scale]
    return total


@njit(inline="always")
def update(weights: np.ndarray, layout: TupleLayout, board: np.uint64, state: np.uint64, delta: float) -> None:
    """Add delta to every weight read for this board."""
    tuple_board = _tuple_board(layout, board)
    stage = _stage_offset(layout, state)
    boards = bb.symmetries(tuple_board)
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            weights[stage + _weight_index(layout, t, symmetric_board)] += delta
    scale = _scale_index(layout, board, tuple_board)
    if scale >= 0:
        weights[scale] += delta


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
    tuple_board = _tuple_board(layout, board)
    stage = _stage_offset(layout, state)
    boards = bb.symmetries(tuple_board)
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            _update_coherent_weight(weights, traces, stage + _weight_index(layout, t, symmetric_board), learning_rate, error)
    scale = _scale_index(layout, board, tuple_board)
    if scale >= 0:
        _update_coherent_weight(weights, traces, scale, learning_rate, error)


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
    ) -> None:
        self.tuples = tuples
        self.downgrade_tile = downgrade_tile
        self.scale_feature = scale_feature
        self.scale_when_downgraded = scale_when_downgraded
        self.stage_tile = stage_tile
        self.layout = TupleLayout.build(tuples, downgrade_tile, scale_feature, scale_when_downgraded, stage_tile)
        self.weights = np.zeros(self.layout.n_weights, dtype=np.float32) if weights is None else weights

    @property
    def features(self) -> dict[str, Any]:
        """The scale-sharing options, as saved with a checkpoint."""
        return {
            "downgrade_tile": self.downgrade_tile,
            "scale_feature": self.scale_feature,
            "scale_when_downgraded": self.scale_when_downgraded,
            "stage_tile": self.stage_tile,
        }

    @property
    def n_features(self) -> int:
        return len(self.tuples) * bb.NUM_SYMMETRIES

    def value(self, board: np.uint64, state: np.uint64 | None = None) -> float:
        """The afterstate's value, read in the stage of `state`, the board the move was made from (default: the board itself)."""
        return float(evaluate(self.weights, self.layout, np.uint64(board), np.uint64(board if state is None else state)))

    def update(self, board: np.uint64, delta: float) -> None:
        update(self.weights, self.layout, np.uint64(board), np.uint64(board), delta)

    def save(self, path: str | Path, metadata: CheckpointMetadata, traces: CoherenceTraces | None = None) -> None:
        """Save the network, plus its temporal coherence traces if it's being trained with them."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features = json.dumps(self.features)
        arrays: dict[str, Any] = {"weights": self.weights, "tuples": json.dumps(self.tuples), "features": features, "metadata": metadata.model_dump_json()}
        if traces is not None:
            # Stored as float32 to keep checkpoints small: rounding a sum once costs nothing, only accumulating in float32 does
            arrays |= {"coherence_errors": traces.errors.astype(np.float32), "coherence_abs_errors": traces.abs_errors.astype(np.float32)}
        # Write then rename so a Ctrl-C mid-save can't leave a broken checkpoint
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
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
