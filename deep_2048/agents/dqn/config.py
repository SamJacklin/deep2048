from deep_2048.config import Settings


class DQNSettings(Settings):
    """DQN hyperparameters. Inherits the shared game settings (board size, tile levels, seed)."""

    GAMMA: float = 0.99
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 64
    REPLAY_CAPACITY: int = 100_000
    MIN_REPLAY_SIZE: int = 2_000

    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: int = 50_000

    N_STEPS: int = 5
    MILESTONE_BONUS_SCALE: float = 3.0
    TAU: float = 1.0
    TARGET_UPDATE_FREQUENCY: int = 1_000  # soft-update target network every N steps
    LR_ETA_MIN: float = 1e-5             # cosine scheduler floor (was hardcoded 1e-6)
    TRAINING_STEPS: int = 200_000
    EVAL_EVERY: int = 10_000
    EVAL_EPISODES: int = 100

    HIDDEN_DIM: int = 256
    GRAD_CLIP_NORM: float = 10.0

    PER_ALPHA: float = 0.6        # priority exponent — 0 = uniform, 1 = fully prioritised
    PER_BETA_START: float = 0.4   # IS weight exponent at start of training
    PER_BETA_END: float = 1.0     # IS weight exponent at end of training (fully corrects bias)
    PER_PRIORITY_EPSILON: float = 1e-5  # small constant added to TD errors to prevent zero priority

    CHECKPOINT_DIR: str = "checkpoints/dqn"
    OUTPUT_DIR: str = "output/dqn"


def get_dqn_settings() -> DQNSettings:
    return DQNSettings()
