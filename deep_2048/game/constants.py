from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTIONS: list[Action] = list(Action)

SPAWN_TILE_WEIGHTS: dict[int, float] = {
    2: 0.9,
    4: 0.1,
}
