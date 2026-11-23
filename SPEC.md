# SPEC.md — MedCover: Medicare Coverage Medallion RAG Pipeline

## 1. Foundation & Architecture Overview
MedCover is a self-sufficient, production-grade Medallion data pipeline and stateless retrieval service that transforms the **CMS Medicare Coverage Database (MCD)**—National Coverage Determinations (NCDs), Local Coverage Determinations (LCDs), Billing/Coding Articles, and CPT/HCPCS/ICD-10 crosswalks—into versioned, quality-gated, hybrid-searchable RAG artifacts.

Concretely, raw HTML policies and relational code tables enter through two interchangeable drivers (bulk ZIP backfill and CMS Coverage API weekly sync), pass through Arrow-typed DuckDB SQL transforms and incremental CPU embeddings, and promote atomically to S3 via Write-Audit-Publish—where a Dockerized FastAPI service on **Amazon ECS Fargate (Express Mode)** serves keyless `/retrieve` and stateless BYO-key `/ask` via built-in Swagger `/docs`.

### Four Core Building Blocks
Every module, SQL query, and workflow in this repository recombines four primitives:
1. **Medallion Tables (Bronze -> Silver -> Gold):** Immutable raw snapshots, grain-safe policy Markdown with SCD Type 2 version windows and covered-code arrays, and incrementally embedded chunks + pre-built DuckDB FTS index.
2. **Arrow + SQL Contracts & Write-Audit-Publish (WAP):** Zero-copy PyArrow schema contracts, DuckDB SQL invariants, dead-letter quarantine, and a retrieval eval gate that block promotion to S3 `latest.json` on any regression.
3. **Hybrid SQL Retrieval:** A single DuckDB query fusing BM25 lexical rank and cosine vector rank (Reciprocal Rank Fusion) with strict point-in-time `as_of_date`, HCPCS/ICD-10 billing-code, and MAC+NCD filters.
4. **Two-Lane Keyless OIDC Delivery:** GitHub Actions authenticating to AWS via short-lived OIDC tokens—separating **code verification & releases** (`ci.yml` -> wheel/pipeline/DQ/container gates -> ECR -> ECS Fargate) from **continuous data releases** (`sync-mcd.yml` -> S3 WAP -> ECS Express canary refresh).

### Locked-In Technical Choices
- **Language & Packaging:** Python 3.12+, `uv`, PEP 621 `pyproject.toml`, PyPA `src/` layout (`src/medcover/`).
- **Compute, Contracts & Storage:** Parquet + PyArrow schemas + Pydantic models + DuckDB (pinned version & explicit `STORAGE_VERSION`, `fts` extension pre-baked into Docker, `array_cosine_distance` on `FLOAT[384]`).
- **Embeddings:** `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim, ONNX CPU; weights pre-baked into Docker alongside DuckDB `fts`).
- **Serving & UI:** Thin FastAPI (`/health`, `/retrieve`, stateless BYO-key `/ask` via `httpx`), exposed solely through FastAPI's built-in Swagger UI (`/docs`). Zero custom frontend code.
- **Cloud & Infra:** AWS S3, Amazon ECR, Amazon ECS on Fargate (Express Mode; `0.5 vCPU / 2 GB RAM` baseline, stepping up to `1 vCPU / 4 GB RAM` if full-corpus `index.duckdb` + ONNX RSS exceeds 1.5 GB), provisioned and torn down via plain AWS CLI scripts in `infra/`.
- **Explicit Non-Goals:** No Databricks, no orchestration servers (Airflow/Dagster), no RAG frameworks (LangChain/LlamaIndex), no vendor LLM SDKs, no Pandas/Pandera contract duplication (PyArrow + DuckDB SQL + Pydantic own all validation), no server-side LLM key storage, and no external LCD PDF attachments (excluded from CMS bulk/API feeds).

---

## 2. Repository Structure
SQL files live inside `src/medcover/sql/` and load via `importlib.resources` so editable installs, built wheels, and containers behave identically.

~~~text
medcover-rag-pipeline/
├── AGENTS.md                        # Guardrails, invariants, and verification gates for coding agents
├── SPEC.md                          # This specification
├── Makefile                         # POSIX shorthand entrypoints: check, build-sample, eval-sample, smoke
├── pyproject.toml                   # PEP 621 metadata, dependencies, ruff/mypy/pytest config
├── uv.lock                          # Reproducible lockfile (pins DuckDB runtime/build alignment)
├── Dockerfile                       # Multi-stage non-root container with pre-baked ONNX + DuckDB fts
├── config/
│   └── eval_queries_prod.json       # Curated live-corpus CMS MCD golden queries for sync-mcd.yml WAP
├── infra/
│   ├── bootstrap.sh                 # Idempotent AWS CLI setup (S3, ECR, OIDC role, 3 ECS IAM roles)
│   └── teardown.sh                  # Single-command cleanup of ECS, ALB, ECR, S3, and IAM roles
├── .github/workflows/
│   ├── ci.yml                       # 4 pre-deploy gates (wheel, WAP, DQ, Docker smoke) + OIDC ECS deploy
│   └── sync-mcd.yml                 # Weekly/manual CMS API sync + S3 WAP + ECS Express canary refresh
├── docs/
│   ├── architecture.md              # System diagram, grain/SCD2 rules, WAP canary flow, scale-out seams
│   └── data_dictionary.md           # Gold Parquet, manifest, and /retrieve contract for ML/GenAI peers
├── src/medcover/
│   ├── __init__.py
│   ├── config.py                    # Typed Settings (paths, S3 bucket, model, chunk/memory limits)
│   ├── contracts.py                 # Canonical PyArrow schemas, Pydantic models, RunManifest, parsers
│   ├── cms_api.py                   # CMS Coverage API v1 client + 1-hr license token refresh
│   ├── bronze.py                    # Canonical Bronze writer (bulk ZIP + API -> Parquet)
│   ├── silver.py                    # HTML->Markdown cleaning, quarantine routing, silver_policies.sql
│   ├── gold.py                      # Section-aware chunking, incremental FastEmbed, index.duckdb, manifest
│   ├── quality.py                   # Arrow schema + dq_invariants.sql + index audit & Markdown summary
│   ├── publish.py                   # Staged S3/local Write-Audit-Publish & atomic latest.json
│   ├── retrieve.py                  # Shared read-only DuckDB hybrid search + hardened BYO-key /ask logic
│   ├── eval.py                      # Golden eval harness (Recall@5, MRR, temporal/forbidden checks)
│   ├── api.py                       # Thin FastAPI service (/health, /retrieve, /ask) with lifespan DB
│   ├── cli.py                       # Thin CLI (ingest-zip, sync-api, build, publish, query, eval)
│   └── sql/
│       ├── __init__.py
│       ├── silver_policies.sql      # Grain-safe code array joins + SCD2 LEAD() valid_to window
│       ├── build_index.sql          # DuckDB chunks table, FTS index, chunks_as_of macro, active_chunks
│       ├── dq_invariants.sql        # Vectorized SQL data quality assertions
│       └── hybrid_search.sql        # Metadata/temporal-filtered BM25 + cosine RRF query
└── tests/
    ├── fixtures/mcd_sample/         # ~15 real public CMS docs (incl. multi-version v1->v2, retired, malformed)
    ├── eval_queries.json            # 15-20 offline sample golden queries + temporal/supersession assertions
    ├── test_bronze_ingest.py        # Bulk ZIP + mocked API ingestion and token refresh
    ├── test_silver_parsing.py       # HTML table/heading preservation, zero fan-out, SCD2 valid_to window
    ├── test_gold_idempotency.py     # Repeat-run 0 re-embeds; 1-doc edit re-embeds delta only
    ├── test_quality_quarantine.py   # Malformed routing, date inversion, quarantine threshold, Arrow/SQL audit
    ├── test_publish_wap.py          # Staged WAP promote + failed-audit protection of latest.json
    ├── test_retrieval_eval.py       # Recall@5 gate + 0% temporal/version leakage + citation verification
    └── test_api.py                  # /health, /retrieve, mocked /ask, model regex guard, key secrecy
~~~

---

## 3. Data Sources & Dual-Mode Ingestion (`bronze.py`, `cms_api.py`)
Both ingestion modes write the exact same Bronze Parquet schema so every downstream stage is source-agnostic. Updated Bronze snapshots persist under `s3://<bucket>/bronze/` so scheduled incremental syncs operate as a closed loop.

1. **Full Backfill — Bulk ZIP (`medcover ingest-zip`):**
   - Ingests CMS MCD weekly bulk downloads (Current LCDs, Current Articles, Current NCDs, plus historical fixtures for temporal/version testing) by streaming their nested `_CSV.ZIP` relational tables.
   - Safely parses multiline quoted HTML cells and explicit UTF-8 / Windows-1252 encodings into Bronze Parquet.
   - Crosswalk ingestion normalizes `code_system` to `HCPCS` | `ICD10` (CPT is HCPCS Level I) and scopes ICD-10 strictly to **covered** diagnosis tables (never conflating `noncovered` ICD-10 codes).
2. **Weekly Delta — CMS Coverage API v1 (`medcover sync-api`):**
   - Base URL: `https://api.coverage.cms.gov` (keyless OpenAPI 3.0; verify exact route parameters against `/v1/coverage-api.json` during implementation).
   - Requests a 1-hour Bearer token from `GET /v1/metadata/license-agreement/` and refreshes proactively before expiry when calling AMA/ADA/AHA-licensed LCD and Article endpoints.
   - Polls the schedule / What's New endpoints to fetch only newly published or revised documents, upserting into Bronze by `(doc_id, version)` and `source_sha256`.

---

## 4. Medallion Data Contracts & Grain Rules

### Bronze (`bronze/raw_documents.parquet`, `bronze/raw_code_crosswalks.parquet`)
- **Grain:** Strictly 1 row per `(doc_id, version)` in `raw_documents`.
- **Columns:** `doc_id` (e.g., `L33822`, `A56542`, `NCD220.6`), `doc_type` (`NCD` | `LCD` | `ARTICLE`), `version` (int), `title`, `raw_html`, `mac_jurisdiction` (nullable for NCD), `effective_date`, `retired_date` (nullable), `status` (nullable raw CMS status `ACTIVE` | `RETIRED`, never fabricated), `source_mode` (`zip` | `api`), `ingested_at` (UTC), `source_sha256`.
- **Rule:** Upsert by `(doc_id, version)`; never overwrite historical versions in place.

### Silver (`silver/policies.parquet`, `silver/quarantine.parquet`)
- **Grain:** Strictly 1 row per `(doc_id, version)`.
- **Transforms:**
  1. Convert `raw_html` to clean Markdown preserving heading hierarchy (`#`, `##`, `###`) and Markdown tables (vital for coverage criteria and billing-code tables).
  2. Join HCPCS/CPT and covered ICD-10 crosswalks inside `sql/silver_policies.sql` using **`LIST(DISTINCT code)`** (`hcpcs_codes VARCHAR[]`, `icd10_codes VARCHAR[]`) so 1-to-many code links never cause row fan-out.
  3. Compute the timeless SCD Type 2 effective interval **`[effective_date, valid_to)`** in `sql/silver_policies.sql`—closing superseded revisions via `LEAD(effective_date) OVER (PARTITION BY doc_id ORDER BY version)` capped by `retired_date` (keeping `retired_date` for provenance, with no materialized `is_active` flag or `status` column). This guarantees a revised policy (`v12` -> `v13`) never leaks two active versions on the same `as_of_date`.
- **Quarantine:** Records with missing required keys, empty cleaned text, `retired_date < effective_date`, or contradictory `status = 'RETIRED'` with null `retired_date` route to `silver/quarantine.parquet` with an explicit `failure_reason` and are excluded from Gold.

### Gold (`gold/chunks.parquet`, `gold/index.duckdb`, `gold/run_manifest.json`)
- **Grain:** Strictly 1 row per `(doc_id, version, chunk_index)`.
- **Columns:** `chunk_id` (`{doc_id}_v{version}_c{chunk_index:04d}`), `doc_id`, `doc_type`, `version`, `chunk_index`, `title`, `section_heading`, `chunk_text`, `token_count`, `chunk_sha256`, `hcpcs_codes VARCHAR[]`, `icd10_codes VARCHAR[]`, `mac_jurisdiction`, `effective_date`, `retired_date`, `valid_to`, `embedding FLOAT[384]`, `embedding_model`.
- **Chunking & Incremental Embedding:**
  - Split on Markdown heading boundaries first, then pack blocks to a configurable **soft token budget** (default `450` tokens, `60`-token prose overlap) without splitting Markdown table rows; DQ enforces a separate **hard token cap** (default `max_chunk_tokens = 900`) so wide real CMS table rows do not falsely abort builds.
  - Define `chunk_sha256 = SHA256(embedding_model, title, section_heading, chunk_text)` so metadata-only edits (`retired_date`, `valid_to`, code arrays) reuse existing vectors without re-embedding.
  - Left-join candidate chunks against prior Gold on `chunk_sha256`; reuse unchanged vectors and run batched `fastembed` CPU inference strictly on net-new or modified chunks.
- **Dual Gold Artifacts (Built Together):**
  - `chunks.parquet`: Canonical Lakehouse data product for downstream ML/analytics consumers.
  - `index.duckdb`: Pre-indexed DuckDB database built via `sql/build_index.sql` with a pinned `STORAGE_VERSION`, `PRAGMA create_fts_index(...)`, query-time `chunks_as_of(as_of_date)` table macro (enforcing `[effective_date, valid_to)`), and `active_chunks` view—enabling sub-second read-only startup on ECS.
- **Immutable Build Manifest (`run_manifest.json`) & Eval Output (`eval_report.json`):** `gold.py` writes `run_manifest.json` as an immutable build record (`run_id`, `git_sha`, `timestamp_utc`, `source_modes`, `duckdb_version`, `duckdb_storage_version`, `bronze_docs`, `silver_docs`, `quarantined_docs`, `gold_chunks`, `chunks_reused`, `chunks_embedded`, `token_p50`, `token_p95`; `quarantine_rate` is computed on read). `eval.py` writes `eval_report.json` separately (`evaluated_at_utc`, `passed`, `recall_at_5`, `mrr`, `temporal_filter_violations`, `forbidden_doc_violations`, per-query results) and never mutates `run_manifest.json`.

---

## 5. Write-Audit-Publish (WAP) & Serving Architecture

### Write-Audit-Publish (`publish.py`)
1. **Write:** Stage `chunks.parquet`, `index.duckdb`, `quarantine.parquet`, `run_manifest.json`, and `eval_report.json` under `releases/<run_id>/`.
2. **Audit:** Verify PyArrow schema contracts, `sql/dq_invariants.sql` (unique grain, zero empty text, zero null/NaN vectors, valid date intervals, `valid_to >= effective_date`, token counts within `(0, max_chunk_tokens]`, `quarantine_rate <= 0.10`), read-only `index.duckdb` FTS/row alignment, and `eval.py` (`Recall@5 >= 0.85`, `temporal_filter_violations == 0`, `forbidden_doc_violations == 0`).
3. **Publish:** Only if Audit passes with exit code `0`, upload the staged release to `s3://<bucket>/gold/releases/<run_id>/` (or local `releases/<run_id>/`) and atomically overwrite `latest.json`:
   ```json
   {
     "run_id": "...",
     "chunks_uri": "...",
     "index_uri": "...",
     "manifest_uri": "...",
     "eval_report_uri": "...",
     "quarantine_uri": "...",
     "duckdb_storage_version": "...",
     "embedding_model": "..."
   }
   ```
   Failed audits never alter `latest.json`.

### Hybrid Retrieval (`sql/hybrid_search.sql`, `retrieve.py`)
- Pre-filters chunks by point-in-time `as_of_date` (defaults to today at request time; `null` disables temporal filtering) enforcing `effective_date <= as_of_date AND (valid_to IS NULL OR valid_to > as_of_date)`, optional `hcpcs_code` (`list_contains(hcpcs_codes, ?)`), `icd10_code`, `doc_type`, and `mac_jurisdiction` (which includes nationwide `NCD` policies alongside the specified MAC's local LCDs/Articles).
- Retrieves top lexical candidates via DuckDB `fts` BM25 and top semantic candidates via `array_cosine_distance(embedding, ?::FLOAT[384])`.
- Combines both candidate sets via **Reciprocal Rank Fusion (RRF)**: `score = 1/(60 + rank_bm25) + 1/(60 + rank_vec)`.

### Thin FastAPI Service (`api.py`)
- **Startup (`lifespan`):** Resolves `MEDCOVER_DATA_RUN_ID` if set (falling back to `latest.json` from S3 via the ECS Task Role or local disk), downloads `index.duckdb` + `run_manifest.json` to `/tmp/`, opens **one shared `read_only=True` DuckDB connection** (issuing `.cursor()` per request, with explicit `memory_limit` and `threads` caps), and loads the pre-baked FastEmbed ONNX model and pre-baked DuckDB `fts` extension with zero runtime downloads.
- **`GET /health`:** Returns `200 OK` with `run_id`, `chunk_count`, `doc_count`, live `active_docs` / `retired_docs` (evaluated via `active_chunks`), `duckdb_version`, and `embedding_model` (wired as the ECS ALB health check path).
- **`POST /retrieve`:** Keyless primary data-product endpoint. Returns ranked chunks, RRF scores, and full provenance metadata.
- **`POST /ask`:** Stateless BYO-key RAG endpoint. Accepts search parameters + `X-LLM-API-Key` header + `provider` (`openai` | `anthropic` | `gemini`, via timeout-bounded `httpx` JSON calls) + optional `model` (strictly validated against `^[A-Za-z0-9._-]{1,64}$` before URL construction). Prompts the model to answer strictly from retrieved chunks with `[doc_id]` citations, runs deterministic citation-provenance verification (`citations_verified: bool`, confirming every cited `doc_id` appeared in the retrieved set—explicitly scoped as provenance verification, not clinical accuracy, with an informational-only disclaimer), and **never stores, caches, or logs** `X-LLM-API-Key`.

---

## 6. CI/CD & AWS Infrastructure (`infra/`, `.github/workflows/`)

### Minimal AWS Infrastructure (`infra/bootstrap.sh`, `infra/teardown.sh`)
Provisions only the exact resources needed, with strict least privilege:
- One S3 bucket (`bronze/` snapshots + `gold/releases/` & `gold/latest.json`) and one Amazon ECR repository.
- GitHub Actions OIDC identity provider + deploy IAM role (scoped strictly to this repo).
- Three standard ECS IAM roles: `ecsTaskExecutionRole` (ECR pull + CloudWatch Logs), `ecsInfrastructureRoleForExpressServices` (ALB, target groups, security groups), and `medcoverTaskRole` (read-only `s3:GetObject` on the Gold bucket prefix).

### `ci.yml` — Code Verification, WAP Proof, Container Smoke & OIDC Deploy
1. **Gate 1 — Static, SQL & Clean Wheel:** `ruff check`, `ruff format --check`, `mypy src`, `sqlfluff lint src/medcover/sql`, then `uv build` -> install the built `.whl` in a fresh venv -> `python -m pytest` (proving packaged SQL resources work outside editable installs).
2. **Gate 2 — Pipeline Idempotency, Delta & WAP Protection:** Run `medcover build --sample` twice (assert 2nd run reports `chunks_embedded == 0`); mutate 1 fixture doc and assert strictly delta chunks re-embed; run `medcover publish --local` (assert `latest.json` advances) and assert a corrupted build fails audit without altering `latest.json`.
3. **Gate 3 — DQ Monitoring Scorecard & Retained Artifacts:** Run Arrow + `sql/dq_invariants.sql` audit; publish the Markdown scorecard (`run_manifest.json` + `AuditReport` + `eval_report.json`) to `$GITHUB_STEP_SUMMARY` and upload `run_manifest.json`, `eval_report.json`, and `quarantine.parquet` as workflow artifacts.
4. **Gate 4 — GenAI Eval & Live Container Smoke:** Run `medcover eval --sample` against `tests/eval_queries.json` (`Recall@5 >= 0.85`, `temporal_filter_violations == 0`, `forbidden_doc_violations == 0`); build `Dockerfile`, boot the container against the staged WAP release, and assert live `GET /health` (matching `run_id` & counts) and `POST /retrieve` (ranked chunks + `as_of_date` filtering) return `200`.
5. **Gate 5 — Deploy (`main` only, after Gates 1–4 pass):** Authenticate via **GitHub OIDC**, push `:<git-sha>` and `:latest` to ECR, deploy via `aws-actions/amazon-ecs-deploy-express-service@v1` (`cpu: 512`, `memory: 2048`, `container-port: 8000`, `health-check-path: /health`, `task-role-arn: medcoverTaskRole`), and run a post-deploy live `/health` + `/retrieve` smoke check.

### `sync-mcd.yml` — Scheduled & Manual Continuous Data Release
- Triggered weekly via `cron` or manually via `workflow_dispatch` (including a `dry_run` flag to execute full WAP audit & summary generation without mutating S3 `latest.json` or ECS).
- Authenticates via GitHub OIDC, restores prior `s3://<bucket>/bronze/` + Gold Parquet for incremental embedding reuse, runs `medcover sync-api` -> `medcover build` -> `medcover publish` (gated by Arrow schemas, `dq_invariants.sql`, and `config/eval_queries_prod.json`), persists updated Bronze back to `s3://<bucket>/bronze/`, triggers an ECS Express Mode canary rollout by updating `MEDCOVER_DATA_RUN_ID=<run_id>` (so Express Mode rolls back automatically if `/health` fails), asserts live `/health` reports the new `run_id`, and publishes the data-release scorecard + audit artifacts to `$GITHUB_STEP_SUMMARY`.

---

## 7. Known Full-Corpus Scale-Out Seams (Documented in `docs/architecture.md`)
To keep the implementation lean and inspectable while staying honest about CMS MCD production quirks, `docs/architecture.md` documents three explicit scale-out evolutions beyond v1:
1. **Many-to-Many MAC Contractors:** Real CMS bulk tables map LCDs/Articles to contractors via `lcd_x_contractor` / `article_x_contractor` (`mac_jurisdictions VARCHAR[]` by jurisdiction code/state rather than a single display string).
2. **LCD <-> Billing Article Crosswalk Propagation:** CMS places most HCPCS/ICD-10 codes on companion Articles linked via `*_related_documents`; full-corpus filtering benefits from propagating linked Article codes onto parent LCDs.
3. **High-Cardinality Code Normalization:** Real Billing Articles can carry 1,000–15,000 ICD-10 codes; at 100k+ chunks, normalizing code arrays into a `(doc_id, version)` lookup table inside `index.duckdb` prevents per-chunk memory and JSON response bloat.
