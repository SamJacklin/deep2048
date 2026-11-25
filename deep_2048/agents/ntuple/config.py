from typing import Literal

from pydantic import model_validator
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
    # The learning rule, if named outright: "td" (constant rate), "tc" (the same as TEMPORAL_COHERENCE) or "autotidbd"
    # (AutoStep TIDBD(0), stage 8: each weight starts at TIDBD_INITIAL_ALPHA and learns its own rate by meta-descent
    # at rate TIDBD_META_RATE, with AutoStep's normaliser decaying over TIDBD_TAU). Unset, TEMPORAL_COHERENCE decides.
    OPTIMIZER: Literal["td", "tc", "autotidbd"] | None = None
    TIDBD_INITIAL_ALPHA: float = 0.004
    TIDBD_META_RATE: float = 0.01
    TIDBD_TAU: float = 10_000.0
    # "global" is the published rule: an update whose effective step is too large divides every weight's step size.
    # "local" (stage 8A, experimental) divides only those of the weights in the trace the step was measured with.
    # "proportional" (also experimental) shares the cut among the weights that made up the step, by how much each did.
    TIDBD_NORMALISATION: Literal["global", "local", "proportional"] = "global"
    # 0 is the published per-weight normaliser, in effect the largest meta-gradient the weight has ever seen. Above 0
    # (experimental) it's a running mean instead, moving this fraction towards each new value.
    TIDBD_NORMALISER_DECAY: float = 0.0
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
    # A .npy file of boards to start every training game from, drawn uniformly: a fixed pool of genuine positions that
    # games never add to or upgrade (stage 6 trains its late-game stage only from first-16384 positions)
    START_POOL: str | None = None
    # Share late-game knowledge across tile scales: the tuples read boards with this tile or larger
    # with their tiles above the largest gap halved, and SCALE_FEATURE adds a large-tile count term,
    # for every board or, with SCALE_WHEN_DOWNGRADED, only for boards that were downgraded
    DOWNGRADE_TILE: int | None = None
    SCALE_FEATURE: bool = False
    SCALE_WHEN_DOWNGRADED: bool = False
    # The scale weight's step as a share of the tuples' rate, and whether boards with no tile of 2048 or more skip it
    # (plain TD(0) or TC only: TD(lambda) and autotidbd update every weight they read alike)
    SCALE_RATE: float = 1.0
    SCALE_SKIP_EMPTY: bool = False
    # Boards whose largest tile is SCALE_BELOW_TILE or more read no scale weight
    SCALE_BELOW_TILE: int | None = None
    # Read the scale weight from the board the move is made from, not the afterstate, so it can't separate one move from another
    SCALE_BY_STATE: bool = False
    # A second copy of the weights for boards whose real largest tile is STAGE_TILE or more, so the late
    # game learns values of its own (scripts/add_stage.py starts it as a copy of a trained network)
    STAGE_TILE: int | None = None
    # The second stage's own downgrading tile (0: none), instead of DOWNGRADE_TILE: the first stage can share across
    # scales from 16384 while the late game reads real 16384s and shares only from 32768
    STAGE_DOWNGRADE_TILE: int | None = None
    # A tailored first stage: every training game ends on the move that makes GOAL_TILE, whose afterstate is worth
    # GOAL_VALUE (a fixed bonus for reaching it, standing in for the later stages). Plain TD(0) or TC only, no late starts,
    # and no downgrading at or below GOAL_TILE, which would share the goal's afterstates with boards still short of it.
    GOAL_TILE: int | None = None
    GOAL_VALUE: float = 0.0
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

    @model_validator(mode="after")
    def _optimizer_decides_coherence(self) -> "NTupleSettings":
        if self.OPTIMIZER is not None:
            self.TEMPORAL_COHERENCE = self.OPTIMIZER == "tc"
        return self

    @model_validator(mode="after")
    def _scale_options_fit(self) -> "NTupleSettings":
        if (self.SCALE_RATE != 1.0 or self.SCALE_SKIP_EMPTY or self.SCALE_BELOW_TILE is not None or self.SCALE_BY_STATE) and not self.SCALE_FEATURE:
            raise ValueError("SCALE_RATE, SCALE_SKIP_EMPTY, SCALE_BELOW_TILE and SCALE_BY_STATE need SCALE_FEATURE")
        if self.SCALE_RATE != 1.0 and (self.OPTIMIZER == "autotidbd" or self.LAMBDA > 0):
            raise ValueError("SCALE_RATE works with plain TD(0) or TC only")
        if not self.SCALE_RATE > 0:
            raise ValueError(f"SCALE_RATE must be positive, not {self.SCALE_RATE}")
        return self

    @model_validator(mode="after")
    def _goal_fits(self) -> "NTupleSettings":
        if self.GOAL_TILE is None:
            return self
        if self.GOAL_TILE & (self.GOAL_TILE - 1) or self.GOAL_TILE < 4:
            raise ValueError(f"GOAL_TILE must be a power of two, not {self.GOAL_TILE}")
        if self.OPTIMIZER == "autotidbd" or self.LAMBDA > 0 or self.GUIDE_DEPTH > 0:
            raise ValueError("GOAL_TILE works with plain TD(0) or TC only: no autotidbd, LAMBDA or GUIDE_DEPTH")
        if self.LATE_START_FRACTION > 0 or self.START_POOL is not None:
            raise ValueError("GOAL_TILE trains whole games from an empty board: LATE_START_FRACTION must be 0 and START_POOL unset")
        if self.DOWNGRADE_TILE is not None and self.DOWNGRADE_TILE <= self.GOAL_TILE:
            raise ValueError(f"DOWNGRADE_TILE {self.DOWNGRADE_TILE} would read the goal's afterstates as boards short of {self.GOAL_TILE}: unset it")
        if self.STAGE_TILE is not None:
            raise ValueError("GOAL_TILE trains a single-stage network: STAGE_TILE must be unset")
        return self


def get_ntuple_settings() -> NTupleSettings:
    return NTupleSettings()
