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

.PHONY: train
train:
	uv run python scripts/train.py

.PHONY: resume
resume:
	uv run python scripts/train.py --resume $(CHECKPOINT)

# Run a single game (ARGS="--mode drl_search", "--mode heuristic", "--render", etc.)
.PHONY: play
play:
	uv run python scripts/play.py $(ARGS)

# Evaluate agents over N games — defaults to all three side-by-side
# Override: ARGS="--mode drl -n 200", ARGS="--mode heuristic -n 50", etc.
.PHONY: evaluate
evaluate:
	uv run python scripts/evaluate.py --mode all $(ARGS)

# Evaluate all agents and write results to frontend/public/benchmark.json
.PHONY: benchmark
benchmark:
	uv run python scripts/evaluate.py --mode all --json frontend/public/benchmark.json $(ARGS)

# Generate frontend replay JSON files
.PHONY: replays
replays:
	uv run python scripts/generate_replays.py

# Plot training metrics from output/training_history.csv
.PHONY: plot
plot:
	uv run python scripts/plot.py $(ARGS)
