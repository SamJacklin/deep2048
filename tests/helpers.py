import numpy as np

# Keeps the weight tables tiny so tests run fast
SMALL_TUPLES = ((0, 1, 2), (4, 5, 6), (0, 1, 4))


def symmetry_grids() -> list[np.ndarray]:
    """Return the 8 rotations and reflections of the board as grids of source cell indices."""
    grid = np.arange(16).reshape(4, 4)
    return [np.rot90(g, k) for g in (grid, grid.T) for k in range(4)]


def random_boards(n: int, seed: int) -> list[np.ndarray]:
    """Generate boards with plenty of merges and empty cells.

    Tiles stop at 16384 because two 32768s merge in game/moves.py but not in the bitboard.
    """
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        highest = int(rng.integers(1, 15))
        exponents = rng.integers(0, highest + 1, size=(4, 4))
        boards.append(np.where(exponents == 0, 0, 1 << exponents).astype(np.int32))
    return boards
