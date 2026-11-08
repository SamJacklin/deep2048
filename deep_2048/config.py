from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Game-level settings shared by every agent."""

    BOARD_SIZE: int = 4
    NUM_TILE_LEVELS: int = 16  # covers empty (0) + tiles 2^1 through 2^15
    SEED: int = 42


def get_settings() -> Settings:
    return Settings()
