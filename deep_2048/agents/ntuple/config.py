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
    # With temporal coherence, shrink each weight's error sums by this factor before every update, so a weight whose
    # errors have cancelled out over its history can speed up again once they start agreeing (1 never forgets)
    TC_DECAY: float = 1.0
    # TD(λ): if LAMBDA > 0, each afterstate learns LAMBDA_HORIZON moves later towards its λ-return.
    # If LAMBDA_END is set, λ moves linearly from LAMBDA to it over each training session.
    LAMBDA: float = 0.0
    LAMBDA_HORIZON: int = 3
    LAMBDA_END: float | None = None
    # Share of training games started from synthetic late-game positions: normal games donate the
    # position where they first make LATE_START_TILE, and its tiles above the gap are doubled
    LATE_START_FRACTION: float = 0.0
    LATE_START_TILE: int = 8192
    LATE_START_POOL: int = 10_000
    # Share late-game knowledge across tile scales: the tuples read boards with this tile or larger
    # with their tiles above the largest gap halved, and SCALE_FEATURE adds a large-tile count term,
    # for every board or, with SCALE_WHEN_DOWNGRADED, only for boards that were downgraded
    DOWNGRADE_TILE: int | None = None
    SCALE_FEATURE: bool = False
    SCALE_WHEN_DOWNGRADED: bool = False
    # A second copy of the weights for boards whose real largest tile is STAGE_TILE or more, so the late
    # game learns values of its own (scripts/add_stage.py starts it as a copy of a trained network)
    STAGE_TILE: int | None = None
    # Search during training (stage 5): with GUIDE_DEPTH > 0, decisions from boards holding GUIDE_TILE or more whose greedy
    # relative margin between the top two moves lies in [GUIDE_MARGIN_LOW, GUIDE_MARGIN_HIGH) are searched GUIDE_DEPTH
    # moves ahead and the searched move is played. With RANK_STEP > 0, where search prefers another move by at least
    # RANK_MIN_GAP of its value, a ranking update closes RANK_STEP of the two moves' gap (plus RANK_MARGIN) directly.
    GUIDE_DEPTH: int = 0
    GUIDE_TILE: int = 16384
    GUIDE_MARGIN_LOW: float = 0.0
    GUIDE_MARGIN_HIGH: float = float("inf")
    RANK_STEP: float = 0.0
    RANK_MARGIN: float = 1.0
    RANK_MIN_GAP: float = 0.0
    # Threads playing and learning in parallel without locks: faster, but not bit-reproducible.
    # The first THREADS_AFTER games are single-threaded, since a fresh network collides often.
    THREADS: int = 1
    THREADS_AFTER: int = 10_000
    TRAINING_GAMES: int = 100_000
    LOG_EVERY: int = 1_000
    CHECKPOINT_EVERY: int = 10_000
    SNAPSHOT_AT: list[int] = []

    CHECKPOINT_DIR: str = "checkpoints/ntuple"
    OUTPUT_DIR: str = "output/ntuple"


def get_ntuple_settings() -> NTupleSettings:
    return NTupleSettings()
