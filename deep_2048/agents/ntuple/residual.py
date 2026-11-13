"""A frozen n-tuple network with a late-game correction: a constant, or a residual network trained on its own.

Stage 1 found downgrading makes the network value a first 16384 like a first 8192, about 50% too high. The
correction applies once the real afterstate's largest tile reaches the gate (16384), and reads the real board:
- a constant added to every gated afterstate's value, fitted beforehand;
- a residual: four 5-tuples read from the real tile exponents in all 8 symmetries (32 weights per board, 4.2M in all),
  zero to start and trained by TD(0) on the combined value, with the base network's weights left untouched.

A corrected network is saved apart from its base: the correction's weights and settings, with the base checkpoint's
path and SHA-256, which loading checks so the base can't change underneath it.
"""

from pathlib import Path

import numpy as np
from numba import njit
from pydantic import BaseModel

from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import NTupleNetwork, ResidualLayout, TupleLayout, _weight_index, evaluate
from deep_2048.agents.ntuple.search import has_two_top_tiles
from deep_2048.bench import file_sha256
from deep_2048.game import bitboard as bb

# Two rows with the cell below their end, and two 2x2 squares with a cell below: shapes that hold a corner chain
RESIDUAL_TUPLES: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3, 4), (4, 5, 6, 7, 8), (0, 1, 4, 5, 8), (1, 2, 5, 6, 9))
GATE_TILE = 16384


class CorrectionMetadata(BaseModel):
    """How a correction was made: the constant's fit, or the residual's training."""

    base_checkpoint: str
    base_sha256: str
    base_games_played: int
    gate_tile: int = GATE_TILE
    constant: float = 0.0
    residual_tuples: list[list[int]] | None = None
    residual_rate: float | None = None
    residual_games: int = 0
    residual_moves: int = 0
    notes: str = ""


class CorrectedNetwork:
    """A base NTupleNetwork plus a gated late-game constant and residual, used wherever an NTupleNetwork is."""

    def __init__(self, base: NTupleNetwork, base_metadata: CheckpointMetadata, metadata: CorrectionMetadata, residual_weights: np.ndarray | None = None) -> None:
        self.base = base
        self.base_metadata = base_metadata
        self.correction = metadata
        tuples = tuple(tuple(t) for t in (metadata.residual_tuples or RESIDUAL_TUPLES))
        residual = TupleLayout.build(tuples)
        residual = residual._replace(offsets=residual.offsets + base.weights.size)
        n_residual = residual.n_weights
        self.weights = np.zeros(base.weights.size + n_residual + 1, dtype=np.float32)
        self.weights[: base.weights.size] = base.weights
        if residual_weights is not None:
            self.weights[base.weights.size : -1] = residual_weights
        self.weights[-1] = metadata.constant
        self.layout = ResidualLayout(base=base.layout, residual=residual, gate_exponent=metadata.gate_tile.bit_length() - 1, constant_index=self.weights.size - 1)
        self.residual_tuples = tuples
        # What NTupleAgent reads to describe the network
        self.tuples, self.downgrade_tile, self.stage_tile = base.tuples, base.downgrade_tile, base.stage_tile
        self.scale_feature, self.scale_when_downgraded = base.scale_feature, base.scale_when_downgraded

    @property
    def residual_weights(self) -> np.ndarray:
        return self.weights[self.base.weights.size : -1]

    def value(self, board: np.uint64, state: np.uint64 | None = None) -> float:
        return float(evaluate(self.weights, self.layout, np.uint64(board), np.uint64(board if state is None else state)))

    def base_unchanged(self) -> bool:
        return bool(np.array_equal(self.weights[: self.base.weights.size], self.base.weights))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, correction=self.correction.model_dump_json(), residual_weights=self.residual_weights)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "CorrectedNetwork":
        with np.load(path) as data:
            metadata = CorrectionMetadata.model_validate_json(str(data["correction"]))
            residual_weights = data["residual_weights"]
        if file_sha256(Path(metadata.base_checkpoint)) != metadata.base_sha256:
            raise ValueError(f"{metadata.base_checkpoint} has changed since {path} was made from it")
        base, base_metadata = NTupleNetwork.load(metadata.base_checkpoint)
        return cls(base, base_metadata, metadata, residual_weights)

    @classmethod
    def from_base(cls, base_checkpoint: str | Path, constant: float = 0.0, residual: bool = False, notes: str = "") -> "CorrectedNetwork":
        base, metadata = NTupleNetwork.load(base_checkpoint)
        correction = CorrectionMetadata(base_checkpoint=str(base_checkpoint), base_sha256=file_sha256(Path(base_checkpoint)),
                                        base_games_played=metadata.games_played, constant=constant,
                                        residual_tuples=[list(t) for t in RESIDUAL_TUPLES] if residual else None, notes=notes)
        return cls(base, metadata, correction)


def is_corrected_checkpoint(path: str | Path) -> bool:
    with np.load(path) as data:
        return "correction" in data


def correction_label(metadata: CorrectionMetadata) -> str:
    parts = []
    if metadata.constant:
        parts.append(f"late constant {metadata.constant:+,.0f}")
    if metadata.residual_tuples:
        parts.append(f"residual {len(metadata.residual_tuples)}x{len(metadata.residual_tuples[0])} after {metadata.residual_games:,} late games")
    return f", from {metadata.gate_tile}: " + " + ".join(parts) if parts else ""


@njit(inline="always")
def _greedy_move(weights: np.ndarray, layout: ResidualLayout, board: np.uint64) -> tuple[np.uint64, int, float]:
    """Return the best move's afterstate, reward and reward + corrected value (-inf if no move is legal)."""
    best, best_reward, best_value = board, 0, -np.inf
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            value = reward + evaluate(weights, layout, afterstate, board)
            if value > best_value:
                best, best_reward, best_value = afterstate, reward, value
    return best, best_reward, best_value


@njit(inline="always")
def _update_residual(weights: np.ndarray, layout: ResidualLayout, afterstate: np.uint64, delta: float) -> None:
    """Add delta to every residual weight the afterstate reads, if it's past the gate. Nothing else changes."""
    if bb.max_exponent(afterstate) < layout.gate_exponent:
        return
    for symmetric_board in bb.symmetries(afterstate):
        for t in range(len(layout.residual.offsets)):
            weights[_weight_index(layout.residual, t, symmetric_board)] += delta


@njit(nogil=True)
def train_residual(weights: np.ndarray, layout: ResidualLayout, starts: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Play a greedy game on from each start with the corrected value, learning the residual by TD(0) as it goes.

    Each afterstate's combined value moves towards the next move's reward plus combined afterstate value, or 0 when
    the game ends. Only residual weights change, and only for gated afterstates: the base and the constant are frozen.
    Spawns come from numba's RNG, seeded here. A game that comes to hold two 32768s stops without a final update,
    since the bitboard can't play the merge that would follow. Returns each game's (score added, max exponent,
    moves, mean absolute TD error).
    """
    np.random.seed(seed)
    scores = np.zeros(starts.size, dtype=np.int64)
    tops = np.zeros(starts.size, dtype=np.int64)
    moves = np.zeros(starts.size, dtype=np.int64)
    errors = np.zeros(starts.size)
    for g in range(starts.size):
        board = starts[g]
        previous = np.uint64(0)
        score = 0
        n = 0
        total_error = 0.0
        while True:
            afterstate, reward, value = _greedy_move(weights, layout, board)
            game_over = value == -np.inf
            if n > 0:
                target = 0.0 if game_over else value
                error = target - evaluate(weights, layout, previous, previous)
                total_error += abs(error)
                _update_residual(weights, layout, previous, rate * error)
            if game_over:
                break
            score += reward
            previous = afterstate
            n += 1
            board = bb.spawn_tile(afterstate)
            if has_two_top_tiles(board):
                break
        scores[g], tops[g], moves[g], errors[g] = score, bb.max_exponent(board), n, total_error / max(n, 1)
    return scores, tops, moves, errors

