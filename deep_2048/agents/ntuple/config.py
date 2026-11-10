from pydantic_settings import SettingsConfigDict

from deep_2048.agents.ntuple.network import DEFAULT_NETWORK, NetworkName
from deep_2048.config import Settings


class NTupleSettings(Settings):
    # Without the prefix these would pick up the DQN's LEARNING_RATE etc. from .env
    model_config = SettingsConfigDict(env_prefix="NTUPLE_")

    NETWORK: NetworkName = DEFAULT_NETWORK
    LEARNING_RATE: float = 0.0025
    # If set, the rate decays geometrically from LEARNING_RATE to this over each training session
    LEARNING_RATE_END: float | None = None
    # Scale each weight's rate by how consistently its errors point the same way
    TEMPORAL_COHERENCE: bool = False
    # Share of training games started from synthetic late-game positions: normal games donate the
    # position where they first make LATE_START_TILE, and its tiles above the gap are doubled
    LATE_START_FRACTION: float = 0.0
    LATE_START_TILE: int = 8192
    LATE_START_POOL: int = 10_000
    # Share late-game knowledge across tile scales: the tuples read boards with this tile or larger
    # with their tiles above the largest gap halved, and SCALE_FEATURE adds a large-tile count term
    DOWNGRADE_TILE: int | None = None
    SCALE_FEATURE: bool = False
    TRAINING_GAMES: int = 100_000
    LOG_EVERY: int = 1_000
    CHECKPOINT_EVERY: int = 10_000
    SNAPSHOT_AT: list[int] = []

    CHECKPOINT_DIR: str = "checkpoints/ntuple"
    OUTPUT_DIR: str = "output/ntuple"


def get_ntuple_settings() -> NTupleSettings:
    return NTupleSettings()
