from pydantic_settings import SettingsConfigDict

from deep_2048.config import Settings


class NTupleSettings(Settings):
    # Without the prefix these would pick up the DQN's LEARNING_RATE etc. from .env
    model_config = SettingsConfigDict(env_prefix="NTUPLE_")

    LEARNING_RATE: float = 0.0025
    TRAINING_GAMES: int = 100_000
    LOG_EVERY: int = 1_000
    CHECKPOINT_EVERY: int = 10_000

    CHECKPOINT_DIR: str = "checkpoints/ntuple"
    OUTPUT_DIR: str = "output/ntuple"


def get_ntuple_settings() -> NTupleSettings:
    return NTupleSettings()
