.PHONY: install check lint format typecheck sql-lint test build-sample eval smoke

install:
	uv sync

check: lint typecheck sql-lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

sql-lint:
	uv run sqlfluff lint src/medcover/sql

test:
	uv run python -m pytest

build-sample:
	uv run medcover build --sample

eval:
	uv run medcover eval

smoke: build-sample eval
