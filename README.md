# MedCover

Medallion RAG pipeline (**Bronze -> Silver -> Gold -> S3 WAP -> ECS Fargate FastAPI**) over the CMS Medicare Coverage Database (NCDs, LCDs, Billing/Coding Articles, and CPT/HCPCS/ICD-10 crosswalks).

See [SPEC.md](SPEC.md) for the full architecture, data contracts, and delivery design, and [AGENTS.md](AGENTS.md) for engineering guardrails.

## Quickstart

```bash
uv sync
uv run medcover --help
```

## Verification Gates

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run sqlfluff lint src/medcover/sql
uv run python -m pytest
```

Or run `make check` on environments with GNU Make installed.
