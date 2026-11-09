export PYTHONPATH := $(PWD)
-include .env
export

SRC = deep_2048

.PHONY: lint-check
lint-check:
	uv run isort $(SRC) --check-only
	uv run ruff check $(SRC)
	uv run ruff format --check $(SRC)

.PHONY: lint-fix
lint-fix:
	uv run isort $(SRC)
	uv run ruff check $(SRC) --fix
	uv run ruff format $(SRC)

.PHONY: type-check
type-check:
	uv run mypy $(SRC) --config-file pyproject.toml

.PHONY: check
check: lint-check type-check

.PHONY: format
format: lint-fix type-check

.PHONY: test
test:
	uv run pytest $(ARGS)

.PHONY: train-dqn
train-dqn:
	uv run python scripts/train_dqn.py

.PHONY: resume-dqn
resume-dqn:
	uv run python scripts/train_dqn.py --resume $(CHECKPOINT)

# Run a single game (ARGS="--agent dqn_search", "--agent heuristic", "--render", etc.)
.PHONY: train-ntuple
train-ntuple:
	uv run python scripts/train_ntuple.py $(ARGS)

.PHONY: resume-ntuple
resume-ntuple:
	uv run python scripts/train_ntuple.py --resume $(or $(CHECKPOINT),checkpoints/ntuple/latest.npz) $(ARGS)

# Live chart of n-tuple training, redrawn from output/ntuple/training_history.csv
.PHONY: watch-ntuple
watch-ntuple:
	uv run python scripts/watch_ntuple.py $(ARGS)

.PHONY: play
play:
	uv run python scripts/play.py $(ARGS)

# Evaluate agents over N games — defaults to DQN vs heuristic side-by-side
# Override: ARGS="--agent dqn -n 200", ARGS="--agent heuristic -n 50", etc.
.PHONY: evaluate
evaluate:
	uv run python scripts/evaluate.py --agent all $(ARGS)

# Evaluate all agents and write results to frontend/public/benchmark.json
.PHONY: benchmark
benchmark:
	uv run python scripts/evaluate.py --agent all --json frontend/public/benchmark.json $(ARGS)

# Generate frontend replay JSON files
.PHONY: replays
replays:
	uv run python scripts/generate_replays.py

# Plot DQN training metrics from output/dqn/training_history.csv
.PHONY: plot
plot:
	uv run python scripts/plot.py $(ARGS)
