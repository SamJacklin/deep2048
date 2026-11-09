import json

from deep_2048.bench import EvalResult, RunKey
from deep_2048.experiments import EVALUATIONS_NAME, Environment, EvaluationRecord, now, record_evaluation


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
