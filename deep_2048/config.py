from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BOARD_SIZE: int = 4
    SEED: int = 42

    GAMMA: float = 0.99
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 64
    REPLAY_CAPACITY: int = 100_000
    MIN_REPLAY_SIZE: int = 2_000

    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: int = 50_000

    TAU: float = 0.005
    TRAINING_STEPS: int = 200_000
    EVAL_EVERY: int = 10_000
    EVAL_EPISODES: int = 100

    HIDDEN_DIM: int = 256
    GRAD_CLIP_NORM: float = 10.0

    CHECKPOINT_DIR: str = "checkpoints"
    OUTPUT_DIR: str = "output"


def get_settings() -> Settings:
    return Settings()
