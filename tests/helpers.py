import numpy as np

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
