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

.PHONY: play
play:
	uv run python scripts/play.py $(ARGS)

.PHONY: view
view:
	uv run python scripts/view.py $(ARGS)