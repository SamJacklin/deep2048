from pydantic import BaseModel


class CheckpointMetadata(BaseModel):
    games_played: int
    learning_rate: float
