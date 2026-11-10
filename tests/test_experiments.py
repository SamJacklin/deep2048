import json
import os
import subprocess
import sys

import pytest

from deep_2048.bench import EvalResult, RunKey
from deep_2048.experiments import EVALUATIONS_NAME, LOCK_NAME, Environment, EvaluationRecord, now, record_evaluation, run_lock


def make_record(label: str, scores: list[int]) -> EvaluationRecord:
    key = RunKey(label=label, checkpoint_sha256=None, n_games=len(scores), seed_offset=0)
    result = EvalResult(label=label, n_games=len(scores), scores=scores, max_tiles=[256] * len(scores))
    return EvaluationRecord(key=key, checkpoint=None, environment=Environment.capture(), recorded_at=now(), from_cache=False, result=result)


def test_environment_records_the_commit() -> None:
    environment = Environment.capture()
    assert environment.git_commit is not None and len(environment.git_commit) == 40
    assert set(environment.packages) == {"numpy", "numba", "torch", "pydantic"}


def test_record_evaluation_replaces_matching_keys(tmp_path) -> None:
    record_evaluation(tmp_path, make_record("A", [1, 2]))
    record_evaluation(tmp_path, make_record("B", [3, 4]))
    record_evaluation(tmp_path, make_record("A", [5, 6]))

    records = json.loads((tmp_path / EVALUATIONS_NAME).read_text())
    assert [(r["key"]["label"], r["result"]["scores"]) for r in records] == [("B", [3, 4]), ("A", [5, 6])]


def test_run_lock_refuses_a_second_holder(tmp_path) -> None:
    with run_lock(tmp_path):
        assert (tmp_path / LOCK_NAME).read_text() == str(os.getpid())
        with pytest.raises(RuntimeError, match="already being trained"):
            with run_lock(tmp_path):
                pass
    assert not (tmp_path / LOCK_NAME).exists()


def test_run_lock_replaces_a_lock_left_by_a_dead_process(tmp_path) -> None:
    finished = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True)
    (tmp_path / LOCK_NAME).write_text(finished.stdout.strip())
    with run_lock(tmp_path):
        assert (tmp_path / LOCK_NAME).read_text() == str(os.getpid())
