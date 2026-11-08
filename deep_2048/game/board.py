import numpy as np

from deep_2048.game.models import BoardStats, RenderedBoard, SpawnResult


def create_empty_board(board_size: int) -> np.ndarray:
    return np.zeros((board_size, board_size), dtype=np.int32)


def copy_board(board: np.ndarray) -> np.ndarray:
    return board.copy()


def get_empty_positions(board: np.ndarray) -> list[tuple[int, int]]:
    positions = np.argwhere(board == 0)
    return [(int(row), int_col) for row, int_col in positions.tolist()]


def count_empty_tiles(board: np.ndarray) -> int:
    return int(np.sum(board == 0))


def get_max_tile(board: np.ndarray) -> int:
    return int(np.max(board))


def has_empty_tiles(board: np.ndarray) -> bool:
    return bool(np.any(board == 0))


def get_board_stats(board: np.ndarray) -> BoardStats:
    empty_tiles = count_empty_tiles(board)
    return BoardStats(
        empty_tiles=empty_tiles,
        max_tile=get_max_tile(board),
        has_empty_tiles=empty_tiles > 0,
    )


def spawn_random_tile(
    board: np.ndarray,
    rng: np.random.Generator,
    tile_weights: dict[int, float],
) -> SpawnResult:
    new_board = board.copy()
    empty_positions = get_empty_positions(new_board)

    if not empty_positions:
        return SpawnResult(board=new_board)

    spawn_index = int(rng.integers(0, len(empty_positions)))
    row, col = empty_positions[spawn_index]

    tile_values = np.array(list(tile_weights.keys()), dtype=np.int32)
    probabilities = np.array(list(tile_weights.values()), dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    spawned_value = int(rng.choice(tile_values, p=probabilities))
    new_board[row, col] = spawned_value

    return SpawnResult(
        board=new_board,
        spawned_value=spawned_value,
        spawned_position=(row, col),
    )


def initialise_board(
    board_size: int,
    rng: np.random.Generator,
    tile_weights: dict[int, float],
) -> np.ndarray:
    board = create_empty_board(board_size)
    board = spawn_random_tile(board, rng, tile_weights).board
    board = spawn_random_tile(board, rng, tile_weights).board
    return board


def boards_equal(board_a: np.ndarray, board_b: np.ndarray) -> bool:
    return bool(np.array_equal(board_a, board_b))


def render_board(board: np.ndarray, score: int | None = None) -> RenderedBoard:
    cell_width = max(4, max(len(str(int(value))) for value in board.flatten()))
    divider = "+" + "+".join(["-" * (cell_width + 2)] * board.shape[1]) + "+"

    lines: list[str] = []
    if score is not None:
        lines.append(f"Score: {score}")

    lines.append(divider)

    for row in board:
        rendered_cells = []
        for value in row:
            cell = "" if int(value) == 0 else str(int(value))
            rendered_cells.append(f" {cell:>{cell_width}} ")
        lines.append("|" + "|".join(rendered_cells) + "|")
        lines.append(divider)

    return RenderedBoard(lines=lines)
