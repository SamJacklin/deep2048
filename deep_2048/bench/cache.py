"""Cache evaluation results on disk so unchanged agents aren't replayed.

Results are stored per game, so new metrics can be computed from old runs. The key
doesn't cover agent code, so pass `--rerun` to evaluate.py after changing how an agent plays.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ValidationError

from deep_2048.agents.base import Agent
from deep_2048.bench.runner import EvalResult

DEFAULT_CACHE_DIR = Path("output/benchmark_cache")


class RunKey(BaseModel):
    """Everything that determines an evaluation's results."""

    label: str
    checkpoint_sha256: str | None
    n_games: int
    seed_offset: int

    @classmethod
    def for_agent(cls, agent: Agent, n_games: int, seed_offset: int = 0) -> "RunKey":
        return cls(
            label=agent.label,
            checkpoint_sha256=_sha256(agent.checkpoint) if agent.checkpoint else None,
            n_games=n_games,
            seed_offset=seed_offset,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class CacheEntry(BaseModel):
    key: RunKey
    result: EvalResult


class ResultCache:
    """One JSON file per run, named by the digest of its key."""

    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.directory = Path(directory)

    def get(self, key: RunKey) -> EvalResult | None:
        """Return the cached result for `key`, or None if there isn't a readable one."""
        try:
            entry = CacheEntry.model_validate_json(self._path(key).read_text())
        except (FileNotFoundError, ValidationError):
            return None
        return entry.result if entry.key == key else None

    def put(self, key: RunKey, result: EvalResult) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(CacheEntry(key=key, result=result).model_dump_json(indent=2))
        tmp_path.replace(path)

    def _path(self, key: RunKey) -> Path:
        return self.directory / f"{key.digest()}.json"


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
