"""Record how each experiment was run, so its results can be reproduced and written up.

A run's folder under experiments/ holds its manifest, training history and evaluations.
Checkpoints stay in the gitignored checkpoints/ folder: training is deterministic, so the
commit, command and settings in the manifest are enough to rebuild them.
"""

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from deep_2048.bench import EvalResult, RunKey

EXPERIMENTS_DIR = Path("experiments")
MANIFEST_NAME = "manifest.json"
EVALUATIONS_NAME = "evaluations.json"
LOCK_NAME = ".lock"
RECORDED_PACKAGES = ("numpy", "numba", "torch", "pydantic")


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    """Hold a run folder for one process, refusing if another live process already holds it.

    Two processes training into the same folder would mix their histories and overwrite each
    other's checkpoints. A lock left by a process that has since died is replaced.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_NAME
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = int(path.read_text() or 0)
            if _is_alive(holder):
                raise RuntimeError(f"{directory} is already being trained by process {holder}") from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w") as f:
            f.write(str(os.getpid()))
        break
    else:
        raise RuntimeError(f"couldn't lock {directory}")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def now() -> datetime:
    return datetime.now(UTC)


class Environment(BaseModel):
    """The code and machine a result came from."""

    git_commit: str | None
    git_dirty: bool | None
    python: str
    packages: dict[str, str]
    machine: str
    cpu_count: int | None

    @classmethod
    def capture(cls) -> "Environment":
        commit = _git("rev-parse", "HEAD")
        # Results written into experiments/ during a run shouldn't count as uncommitted code
        changes = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!experiments")
        return cls(
            git_commit=commit,
            git_dirty=None if commit is None else bool(changes),
            python=platform.python_version(),
            packages={name: version(name) for name in RECORDED_PACKAGES},
            machine=_machine(),
            cpu_count=os.cpu_count(),
        )


class TrainingSession(BaseModel):
    """One invocation of a training script. A run has several if it was stopped and resumed."""

    command: list[str]
    settings: dict[str, Any]
    resumed_from: str | None
    # Checkpoints are overwritten and deleted, so the hash pins down exactly which network a branch started from
    resumed_from_sha256: str | None = None
    environment: Environment
    started_at: datetime
    started_with: dict[str, int]
    finished_at: datetime | None = None
    finished_with: dict[str, int] | None = None
    interrupted: bool | None = None
    # CPU time across all of the process's threads, for comparing training cost beyond games and wall time
    cpu_seconds: float | None = None


class RunManifest(BaseModel):
    name: str
    sessions: list[TrainingSession] = Field(default_factory=list)

    @classmethod
    def load_or_create(cls, directory: Path) -> "RunManifest":
        path = directory / MANIFEST_NAME
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return cls(name=directory.name)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MANIFEST_NAME).write_text(self.model_dump_json(indent=2) + "\n")


class EvaluationRecord(BaseModel):
    """An evaluation's per-game results. `from_cache` means the games were played by an earlier run."""

    key: RunKey
    checkpoint: str | None
    environment: Environment
    recorded_at: datetime
    from_cache: bool
    result: EvalResult


def record_evaluation(directory: Path, record: EvaluationRecord) -> None:
    """Add a record to the folder's evaluations, replacing any earlier one with the same key."""
    path = directory / EVALUATIONS_NAME
    existing = json.loads(path.read_text()) if path.exists() else []
    records = [EvaluationRecord.model_validate(r) for r in existing]
    records = [r for r in records if r.key != record.key] + [record]
    directory.mkdir(parents=True, exist_ok=True)
    payload = "[\n" + ",\n".join(r.model_dump_json(indent=2) for r in records) + "\n]\n"
    path.write_text(payload)


def command_line() -> list[str]:
    return [Path(sys.argv[0]).name, *sys.argv[1:]]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor()
    return f"{chip} ({platform.platform()})"
