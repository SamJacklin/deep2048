"""N-tuple network from Szubert & Jaśkowski (2014).

A board's value is a sum of weights from lookup tables. Each tuple is a group of cells whose
exponents index into that tuple's table, and each tuple is read in all 8 rotations and
reflections of the board using the same table. Cells are numbered 4 * row + col.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.game import bitboard as bb

TUPLES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)


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

    @classmethod
    def build(cls, tuples: tuple[tuple[int, ...], ...]) -> "TupleLayout":
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

        return cls(offsets=offsets, n_runs=n_runs, shifts=shifts, masks=masks, dests=dests, n_weights=table_start)


@njit(inline="always")
def _weight_index(layout: TupleLayout, t: int, board: np.uint64) -> int:
    index = np.uint64(0)
    for j in range(layout.n_runs[t]):
        index |= ((board >> layout.shifts[t, j]) & layout.masks[t, j]) << layout.dests[t, j]
    return int(layout.offsets[t] + np.int64(index))


@njit(inline="always")
def evaluate(weights: np.ndarray, layout: TupleLayout, board: np.uint64) -> float:
    """Sum the weights each tuple reads from each symmetry of the board."""
    boards = bb.symmetries(board)
    total = 0.0
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            total += weights[_weight_index(layout, t, symmetric_board)]
    return total


@njit(inline="always")
def update(weights: np.ndarray, layout: TupleLayout, board: np.uint64, delta: float) -> None:
    """Add delta to every weight read for this board."""
    boards = bb.symmetries(board)
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            weights[_weight_index(layout, t, symmetric_board)] += delta


class CoherenceTraces(NamedTuple):
    """Running sums of each weight's errors and their sizes, for temporal coherence learning. A NamedTuple so numba can use it."""

    errors: np.ndarray
    abs_errors: np.ndarray

    @classmethod
    def zeros(cls, n_weights: int) -> "CoherenceTraces":
        return cls(np.zeros(n_weights, dtype=np.float32), np.zeros(n_weights, dtype=np.float32))


@njit(inline="always")
def update_coherent(weights: np.ndarray, traces: CoherenceTraces, layout: TupleLayout, board: np.uint64, learning_rate: float, error: float) -> None:
    """Move each weight read for this board by learning_rate * coherence * error.

    A weight's coherence is |sum of its errors| / sum of their sizes: near 1 while its errors keep
    pointing the same way, and near 0 once they cancel out. A weight with no history learns at the full rate.
    """
    boards = bb.symmetries(board)
    for t in range(len(layout.offsets)):
        for symmetric_board in boards:
            i = _weight_index(layout, t, symmetric_board)
            coherence = abs(traces.errors[i]) / traces.abs_errors[i] if traces.abs_errors[i] > 0 else 1.0
            weights[i] += learning_rate * coherence * error
            traces.errors[i] += error
            traces.abs_errors[i] += abs(error)


class NTupleNetwork:
    def __init__(self, tuples: tuple[tuple[int, ...], ...] = TUPLES, weights: np.ndarray | None = None) -> None:
        self.tuples = tuples
        self.layout = TupleLayout.build(tuples)
        self.weights = np.zeros(self.layout.n_weights, dtype=np.float32) if weights is None else weights

    @property
    def n_features(self) -> int:
        return len(self.tuples) * bb.NUM_SYMMETRIES

    def value(self, board: np.uint64) -> float:
        return float(evaluate(self.weights, self.layout, np.uint64(board)))

    def update(self, board: np.uint64, delta: float) -> None:
        update(self.weights, self.layout, np.uint64(board), delta)

    def save(self, path: str | Path, metadata: CheckpointMetadata, traces: CoherenceTraces | None = None) -> None:
        """Save the network, plus its temporal coherence traces if it's being trained with them."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"weights": self.weights, "tuples": json.dumps(self.tuples), "metadata": metadata.model_dump_json()}
        if traces is not None:
            arrays |= {"coherence_errors": traces.errors, "coherence_abs_errors": traces.abs_errors}
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
            network = cls(tuples=tuples, weights=data["weights"])
            metadata = CheckpointMetadata.model_validate_json(str(data["metadata"]))
        return network, metadata


def load_traces(path: str | Path) -> CoherenceTraces | None:
    """Load a checkpoint's temporal coherence traces, or None if it was trained without them."""
    with np.load(path) as data:
        if "coherence_errors" not in data:
            return None
        return CoherenceTraces(data["coherence_errors"], data["coherence_abs_errors"])
