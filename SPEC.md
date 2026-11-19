# SPEC.md — MedCover: Medicare Coverage Medallion RAG Pipeline

## 1. Foundation & Architecture Overview
MedCover is a production-grade, Databricks-free Medallion data pipeline and stateless retrieval service that transforms the entire current **CMS Medicare Coverage Database (MCD)**—National Coverage Determinations (NCDs), Local Coverage Determinations (LCDs), Billing/Coding Articles, and CPT/HCPCS/ICD-10 crosswalks—into versioned, quality-gated, hybrid-searchable RAG artifacts.

Concretely, raw HTML policies and relational code tables enter through two interchangeable drivers (bulk ZIP backfill and CMS Coverage API weekly sync), pass through DuckDB SQL transforms and incremental CPU embeddings, and promote atomically to S3 via Write-Audit-Publish—where a Dockerized FastAPI service on **Amazon ECS Fargate (Express Mode)** serves keyless `/retrieve` and stateless BYO-key `/ask` via built-in Swagger `/docs`.

### Four Core Building Blocks
Every module, SQL query, and workflow in this repository recombines four primitives:
1. **Medallion Tables (Bronze -> Silver -> Gold):** Immutable raw snapshots, grain-safe policy Markdown with code arrays, and incrementally embedded chunks + pre-built DuckDB FTS index.
2. **Contracts & Write-Audit-Publish (WAP):** Pandera schemas, SQL invariants, dead-letter quarantine, and a `Recall@5` gate that block promotion to S3 `latest.json` on any regression.
3. **Hybrid SQL Retrieval:** A single DuckDB query fusing BM25 lexical rank and cosine vector rank (Reciprocal Rank Fusion) with strict `is_active` and billing-code filters.
4. **Keyless OIDC Delivery:** GitHub Actions authenticating to AWS via short-lived OIDC tokens—separating **code releases** (`ci.yml` -> ECR -> ECS Fargate) from **data releases** (`sync-mcd.yml` -> S3 WAP -> ECS rolling refresh).

### Locked-In Technical Choices
- **Language & Packaging:** Python 3.12+, `uv`, PEP 621 `pyproject.toml`, PyPA `src/` layout (`src/medcover/`).
- **Compute & Storage:** Parquet + DuckDB (`fts` extension + `array_cosine_distance` on `FLOAT[384]`).
- **Embeddings:** `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim, ONNX CPU; weights pre-baked into Docker).
- **Serving & UI:** Thin FastAPI (`/health`, `/retrieve`, stateless BYO-key `/ask` via `httpx`), exposed solely through FastAPI's built-in Swagger UI (`/docs`). Zero custom frontend code.
- **Cloud & Infra:** AWS S3, Amazon ECR, Amazon ECS on Fargate (Express Mode, 0.5 vCPU / 2 GB RAM), provisioned and torn down via plain AWS CLI scripts in `infra/`.
- **Explicit Non-Goals:** No Databricks, no orchestration servers (Airflow/Dagster), no RAG frameworks (LangChain/LlamaIndex), no vendor LLM SDKs, no server-side LLM key storage, and no external LCD PDF attachments (excluded from CMS bulk/API feeds).

---

## 2. Repository Structure
SQL files live inside `src/medcover/sql/` and load via `importlib.resources` so editable installs, built wheels, and containers behave identically.

~~~text
medcover-rag-pipeline/
├── AGENTS.md                        # Guardrails and invariants for coding agents
├── SPEC.md                          # This specification
├── Makefile                         # Human entrypoints: check, test, build-sample, eval, smoke
├── pyproject.toml                   # PEP 621 metadata, dependencies, ruff/mypy/pytest config
├── uv.lock                          # Reproducible lockfile
├── Dockerfile                       # Multi-stage non-root container with pre-baked ONNX weights
├── infra/
│   ├── bootstrap.sh                 # Idempotent AWS CLI setup (S3, ECR, OIDC role, 3 ECS IAM roles)
│   └── teardown.sh                  # Single-command cleanup of ECS, ALB, ECR, S3, and IAM roles
├── .github/workflows/
│   ├── ci.yml                       # 4 quality gates on PR/push + OIDC deploy to ECS on main
│   └── sync-mcd.yml                 # Weekly CMS API incremental sync + S3 WAP + ECS refresh
├── docs/
│   ├── architecture.md              # System diagram, grain rules, and 1:1 Databricks mapping
│   └── data_dictionary.md           # Gold Parquet & manifest schema contract for ML/GenAI peers
├── src/medcover/
│   ├── __init__.py
│   ├── config.py                    # Typed Settings (paths, S3 bucket, model, chunk limits)
│   ├── contracts.py                 # Pydantic models, Pandera schemas, and RunManifest
│   ├── cms_api.py                   # CMS Coverage API v1 client + 1-hr license token refresh
│   ├── bronze.py                    # Canonical Bronze writer (bulk ZIP + API -> Parquet)
│   ├── silver.py                    # HTML->Markdown cleaning + DuckDB silver_policies.sql
│   ├── gold.py                      # Section-aware chunking, incremental FastEmbed, index.duckdb
│   ├── quality.py                   # Pandera + dq_invariants.sql, quarantine, run_manifest.json
│   ├── publish.py                   # Staged S3/local Write-Audit-Publish & atomic latest.json
│   ├── retrieve.py                  # DuckDB hybrid search + stateless BYO-key httpx synthesis
│   ├── eval.py                      # Golden eval harness (Recall@5, MRR, active compliance)
│   ├── api.py                       # Thin FastAPI service (/health, /retrieve, /ask)
│   ├── cli.py                       # Thin CLI (ingest-zip, sync-api, build, publish, query, eval)
│   └── sql/
│       ├── __init__.py
│       ├── silver_policies.sql      # Grain-safe policy + CPT/HCPCS/ICD-10 array joins
│       ├── dq_invariants.sql        # SQL data quality assertions
│       └── hybrid_search.sql        # Metadata-filtered BM25 + cosine RRF query
└── tests/
    ├── fixtures/mcd_sample/         # ~15 real public CMS docs (incl. 1 retired, 1 malformed)
    ├── eval_queries.json            # 15-20 golden Medicare coverage queries + expected doc_ids
    ├── test_bronze_ingest.py        # Bulk ZIP + mocked API ingestion and token refresh
    ├── test_silver_parsing.py       # HTML table/heading preservation and zero code fan-out
    ├── test_gold_idempotency.py     # Repeat-run 0 re-embeds; 1-doc edit re-embeds delta only
    ├── test_quality_quarantine.py   # Malformed routing, date inversion, quarantine threshold
    ├── test_retrieval_eval.py       # Recall@5 gate + 0% retired leakage on active_only queries
    └── test_api.py                  # /health, /retrieve, mocked /ask + citation verification
~~~

---

## 3. Data Sources & Dual-Mode Ingestion (`bronze.py`, `cms_api.py`)
Both ingestion modes write the exact same Bronze Parquet schema so every downstream stage is source-agnostic.

1. **Full Backfill — Bulk ZIP (`medcover ingest-zip`):**
   - Ingests the full current CMS MCD weekly bulk downloads (Current LCDs, Current Articles, and Current NCDs, plus retired fixtures for temporal testing) by streaming their nested `_CSV.ZIP` relational tables.
   - Safely parses multiline quoted HTML cells and explicit UTF-8 / Windows-1252 encodings into Bronze Parquet.
2. **Weekly Delta — CMS Coverage API v1 (`medcover sync-api`):**
   - Base URL: `https://api.coverage.cms.gov` (keyless OpenAPI 3.0; verify exact route parameters against `/v1/coverage-api.json` during implementation).
   - Requests a 1-hour Bearer token from `GET /v1/metadata/license-agreement/` and refreshes proactively before expiry when calling AMA/ADA/AHA-licensed LCD and Article endpoints.
   - Polls the schedule and What's New endpoints to fetch only newly published or revised documents, merging into Bronze by `(doc_id, version)` and `source_sha256`.

---

## 4. Medallion Data Contracts & Grain Rules

### Bronze (`bronze/raw_documents.parquet`, `bronze/raw_code_crosswalks.parquet`)
- **Grain:** Strictly 1 row per `(doc_id, version)` in `raw_documents`.
- **Columns:** `doc_id` (e.g., `L33822`, `A56542`, `NCD220.6`), `doc_type` (`NCD` | `LCD` | `ARTICLE`), `version` (int), `title`, `raw_html`, `mac_jurisdiction` (nullable for NCD), `effective_date`, `retired_date` (nullable), `status` (`ACTIVE` | `RETIRED`), `source_mode` (`zip` | `api`), `ingested_at` (UTC), `source_sha256`.
- **Rule:** Upsert by `(doc_id, version)`; never overwrite historical versions in place.

### Silver (`silver/policies.parquet`, `silver/quarantine.parquet`)
- **Grain:** Strictly 1 row per `(doc_id, version)`.
- **Transforms:**
  1. Convert `raw_html` to clean Markdown preserving heading hierarchy (`#`, `##`, `###`) and Markdown tables (vital for coverage criteria and billing-code tables).
  2. Join CPT/HCPCS and ICD-10 crosswalks inside `sql/silver_policies.sql` using **`LIST(DISTINCT code)`** (`cpt_codes VARCHAR[]`, `icd10_codes VARCHAR[]`) so 1-to-many code links never cause row fan-out.
  3. Derive `is_active`: `status = 'ACTIVE' AND (retired_date IS NULL OR retired_date > CURRENT_DATE)`.
- **Quarantine:** Records with missing required keys, empty cleaned text, or `retired_date < effective_date` route to `silver/quarantine.parquet` with an explicit `failure_reason` and are excluded from Gold.

### Gold (`gold/chunks.parquet`, `gold/index.duckdb`, `gold/run_manifest.json`)
- **Grain:** Strictly 1 row per `(doc_id, version, chunk_index)`.
- **Columns:** `chunk_id` (`{doc_id}_v{version}_c{chunk_index:04d}`), `doc_id`, `doc_type`, `version`, `title`, `section_heading`, `chunk_text`, `token_count`, `chunk_sha256`, `cpt_codes VARCHAR[]`, `icd10_codes VARCHAR[]`, `mac_jurisdiction`, `effective_date`, `retired_date`, `is_active`, `embedding FLOAT[384]`, `embedding_model`.
- **Chunking & Incremental Embedding:**
  - Split on Markdown heading boundaries first, then apply a configurable token budget (default 450 tokens, 60-token overlap) avoiding mid-row table splits.
  - Left-join candidate chunks against prior Gold on `chunk_sha256`; reuse existing vectors for unchanged chunks and run batched `fastembed` CPU inference strictly on net-new or modified chunks.
- **Dual Gold Artifacts (Built Together):**
  - `chunks.parquet`: Canonical Lakehouse data product for downstream ML/analytics/Databricks consumers.
  - `index.duckdb`: Pre-indexed DuckDB database loaded from `chunks.parquet` with `PRAGMA create_fts_index(...)` already built at pipeline time, allowing the serving container to open it read-only in under one second with minimal memory overhead across the full 10,000+ document corpus.
- **Run Manifest (`run_manifest.json`):** `run_id`, `git_sha`, `timestamp_utc`, `source_mode`, `bronze_docs`, `silver_docs`, `quarantined_docs`, `quarantine_rate`, `gold_chunks`, `chunks_reused`, `chunks_embedded`, `token_p50`, `token_p95`, `eval_recall_at_5`, `eval_mrr`, `active_filter_violations`.

---

## 5. Write-Audit-Publish (WAP) & Serving Architecture

### Write-Audit-Publish (`publish.py`)
1. **Write:** Emit `chunks.parquet`, `index.duckdb`, `quarantine.parquet`, and `run_manifest.json` into a staged release path (`releases/<run_id>/`).
2. **Audit:** Execute Pandera contracts, `sql/dq_invariants.sql` (unique grain, zero empty text, zero null vectors, bounded token counts, `quarantine_rate <= 0.10`), and `eval.py` (`Recall@5 >= 0.85`, `active_filter_violations == 0`).
3. **Publish:** Only if Audit passes with exit code 0, upload the staged release to `s3://<bucket>/gold/releases/<run_id>/` and atomically overwrite `s3://<bucket>/gold/latest.json` (`{"run_id": "...", "chunks_uri": "...", "index_uri": "...", "manifest_uri": "..."}`). Failed runs never alter `latest.json`.

### Hybrid Retrieval (`sql/hybrid_search.sql`, `retrieve.py`)
- Pre-filters chunks by `active_only` (default `true`), optional `cpt_code` (`list_contains(cpt_codes, ?)`), `icd10_code`, `doc_type`, and `mac_jurisdiction`.
- Retrieves top lexical candidates via DuckDB `fts` BM25 and top semantic candidates via `array_cosine_distance(embedding, ?::FLOAT[384])`.
- Combines both candidate sets via **Reciprocal Rank Fusion (RRF)**: `score = 1/(60 + rank_bm25) + 1/(60 + rank_vec)`.

### Thin FastAPI Service (`api.py`)
- **Startup:** Reads `latest.json` from S3 (using the ECS Task Role, zero static credentials) or local disk, downloads `index.duckdb` + `run_manifest.json` to `/tmp/`, and opens `index.duckdb` in `read_only=True` mode alongside the pre-baked FastEmbed ONNX model.
- **`GET /health`:** Returns `200 OK` with `run_id`, `chunk_count`, `active_docs`, `retired_docs`, and `embedding_model` (wired as the ECS ALB health check path).
- **`POST /retrieve`:** Keyless. Returns ranked chunks, RRF scores, and full provenance metadata.
- **`POST /ask`:** Stateless BYO-key RAG endpoint. Accepts search parameters + `X-LLM-API-Key` header + `provider` (`openai` | `anthropic` | `gemini`, via lightweight `httpx` JSON calls). Prompts the model to answer strictly from retrieved chunks with `[doc_id]` citations, runs a deterministic verification confirming every cited `doc_id` appears in the retrieved set (`grounding_verified: bool`), and **never stores, caches, or logs** `X-LLM-API-Key`.

---

## 6. CI/CD & AWS Infrastructure (`infra/`, `.github/workflows/`)

### Minimal AWS Infrastructure (`infra/bootstrap.sh`, `infra/teardown.sh`)
Provisions only the exact resources needed, with strict least privilege:
- One S3 bucket for Bronze/Gold WAP artifacts and one Amazon ECR repository.
- GitHub Actions OIDC identity provider + deploy IAM role (scoped strictly to this repo).
- Three standard ECS IAM roles: `ecsTaskExecutionRole` (ECR pull + CloudWatch Logs), `ecsInfrastructureRoleForExpressServices` (ALB, target groups, security groups), and `medcoverTaskRole` (read-only `s3:GetObject` on the Gold bucket prefix).

### `ci.yml` — Four Quality Gates + OIDC Code Deployment
1. **Gate 1 — Static, SQL & Clean Wheel:** `ruff check`, `ruff format --check`, `mypy src`, `sqlfluff lint src/medcover/sql`, then `uv build` -> install `.whl` in a fresh venv -> `pytest`.
2. **Gate 2 — Pipeline & Idempotency:** Run `medcover build` twice on `tests/fixtures/mcd_sample`; assert 2nd run reports `chunks_embedded == 0`; mutate 1 fixture doc and assert only its delta chunks re-embed.
3. **Gate 3 — Data Contracts & Job Summary:** Run Pandera + `sql/dq_invariants.sql`; fail on violations or `quarantine_rate > 0.10`; publish a Markdown summary of `run_manifest.json` to `$GITHUB_STEP_SUMMARY`.
4. **Gate 4 — GenAI Eval & Container Smoke:** Run `medcover eval` against `tests/eval_queries.json` (`Recall@5 >= 0.85`, `active_filter_violations == 0`); build `Dockerfile`, run against sample artifacts, and assert `GET /health` + `POST /retrieve` return `200`.
5. **Deploy (`main` only, after Gates 1–4 pass):** Authenticate via **GitHub OIDC**, push `:<git-sha>` and `:latest` to ECR, deploy via `aws-actions/amazon-ecs-deploy-express-service@v1` (`cpu: 512`, `memory: 2048`, `container-port: 8000`, `health-check-path: /health`, `task-role-arn: medcoverTaskRole`), and run a live `/health` smoke check.

### `sync-mcd.yml` — Weekly Continuous Data Release
- Triggered weekly via `cron` or manually via `workflow_dispatch`.
- Authenticates via GitHub OIDC, fetches prior Bronze/Gold state from S3, runs `medcover sync-api` -> `medcover build` -> `medcover publish` (full Write-Audit-Publish including `index.duckdb` and `eval`), triggers an ECS service rolling refresh to load the new `latest.json`, and asserts live `/health` reports the new `run_id`.