# AGENTS.md — MedCover RAG Pipeline

## 1. Mission & Classical Engineering Standard
Build and maintain **MedCover**, a clean, self-sufficient, production-grade Medallion RAG pipeline (Bronze -> Silver -> Gold -> S3 WAP -> ECS Fargate FastAPI) over the CMS Medicare Coverage Database.

Write every module, SQL query, test, and workflow as if a senior data engineer will inspect it cold in five minutes. Elegance comes from **economy of means**: establish the simplest complete foundation first, then recombine a small set of explicit building blocks rather than inventing abstractions.

### Four Core Rules
1. **Anchor the Foundation First:** Implement the working happy path end-to-end before adding configuration flags, retry tuning, or optimizations.
2. **Keep Responsibilities Cohesive:** Map responsibilities cleanly onto explicit domain modules (`contracts.py`, `cms_api.py`, `bronze.py`, `silver.py`, `gold.py`, `quality.py`, `publish.py`, `retrieve.py`, `eval.py`, `api.py`, `cli.py`) rather than sprawling hierarchies or 1,000-line catch-all files.
3. **Separate Signal from Mechanics:** Keep top-level stage functions short and readable top-to-bottom. Push HTTP token refresh, HTML parsing edge cases, and S3 I/O into small private helpers so data flow stays unmistakable.
4. **Anti-Slop Rule (Strictly Banned):**
   - NO `BasePipeline`, `AbstractIngestor`, `PipelineManager`, `ServiceFactory`, or generic `utils.py` / `helpers.py` dumping grounds.
   - NO orchestration frameworks (Airflow, Dagster), NO RAG frameworks (LangChain, LlamaIndex), NO vendor LLM SDKs (`openai`, `anthropic`, `google-genai`), NO `pandas`/`pandera` contract duplication (PyArrow schemas + DuckDB SQL + Pydantic own all validation), and NO custom frontend HTML/JS/Gradio UI (FastAPI `/docs` is the sole UI).
   - NO new dependency in `pyproject.toml` without a one-sentence justification explaining why stdlib or an existing package cannot do the job.

---

## 2. Human-Standard Conventions
- **Packaging:** Use `uv`, PEP 621 `pyproject.toml`, and `src/medcover/`. Every SQL file lives under `src/medcover/sql/` and MUST load via `importlib.resources.files("medcover.sql")` so built wheels and Docker containers never fail on relative paths.
- **SQL Discipline:** All relational joins, index-build DDL (`build_index.sql`), DQ invariants (`dq_invariants.sql`), and hybrid retrieval queries (`hybrid_search.sql`) belong in `src/medcover/sql/` and MUST pass `sqlfluff lint`. Only single-statement DuckDB control pragmas/probes (`LOAD fts`, `SET memory_limit`, `SELECT COUNT(*)`) belong in Python.
- **Typing & Docstrings:** Public functions require complete PEP 484 type annotations (`mypy src` clean) and brief Google-style docstrings stating *inputs, output grain, and invariants*—never restating argument names. Format and lint strictly with `ruff`.
- **Explicit Failure Routing:** Never swallow errors (`except Exception: pass` is banned). Malformed input records MUST route to `silver/quarantine.parquet` with an explicit `failure_reason`.

---

## 3. Non-Negotiable Data, Performance & Security Invariants
`SPEC.md` §4–§5 is the canonical schema source of truth. Verify every invariant below before finishing any change:
1. **Strict Grain & Code-Link Preservation:**
   - Bronze & Silver policies are strictly unique on `(doc_id, version)`; Gold chunks are strictly unique on `(doc_id, version, chunk_index)` with deterministic `chunk_id` and `chunk_sha256`.
   - Aggregate HCPCS/CPT (`hcpcs_codes`) and **covered** ICD-10 (`icd10_codes`) crosswalks with `LIST(DISTINCT code)`—never join 1-to-many code rows directly onto policy text and never mix non-covered ICD-10 codes into `icd10_codes`.
2. **Idempotent Incremental Embedding:**
   - Compute `chunk_sha256` strictly over `(embedding_model, title, section_heading, chunk_text)` so metadata-only changes reuse prior embeddings.
   - Re-running `medcover build` on unchanged inputs MUST produce identical Gold counts and `chunks_embedded == 0`; mutating 1 source document MUST re-embed strictly the changed `chunk_sha256` delta.
3. **Structure-Preserving HTML Parsing & Bounded Chunks:**
   - Convert HTML to Markdown preserving heading hierarchy (`#`, `##`, `###`) and Markdown tables. Never flatten tables with naive regex stripping.
   - Keep the soft chunk budget (`450` tokens) distinct from the hard DQ ceiling (`max_chunk_tokens = 900`) so intact real CMS table rows do not falsely break WAP.
4. **Build-Time Indexing & Pre-Baked, Version-Pinned Runtime:**
   - `gold.py` MUST build `index.duckdb` via `sql/build_index.sql` with a pinned DuckDB `STORAGE_VERSION`, `PRAGMA create_fts_index(...)`, `chunks_as_of(as_of_date)`, and `active_chunks`.
   - `Dockerfile` MUST pre-bake **both** the `BAAI/bge-small-en-v1.5` ONNX weights **and** DuckDB's `fts` extension so ECS starts offline with zero external downloads.
   - `api.py` MUST open **one shared `read_only=True` DuckDB connection** in FastAPI `lifespan` (using `.cursor()` per request with explicit `memory_limit` and `threads` caps) and NEVER rebuild FTS indexes at runtime.
5. **Temporal, Version & WAP Safety:**
   - Silver and Gold MUST compute and enforce the timeless SCD Type 2 interval `[effective_date, valid_to)` (`LEAD(effective_date)` capped by `retired_date`, never a materialized `is_active` column). Queries with `as_of_date` MUST achieve zero dual-version leakage and `temporal_filter_violations == 0`.
   - `run_manifest.json` MUST remain an immutable build record written by `gold.py`; `eval.py` MUST write evaluation output separately to `eval_report.json`.
   - `publish.py` MUST NEVER update `latest.json` unless PyArrow schema contracts, `dq_invariants.sql`, read-only `index.duckdb` verification, and `eval.py` all pass.
6. **Zero-Trust & Hardened `/ask`:**
   - Pass `X-LLM-API-Key` strictly in memory on the outbound `httpx` request—NEVER persist, cache, log, or echo it in logs, exceptions, or responses.
   - Validate caller-supplied `model` against `^[A-Za-z0-9._-]{1,64}$` before building provider URLs, and frame `citations_verified` strictly as citation-provenance verification (never clinical accuracy).

---

## 4. Testing, Infra & Operational Walkthroughs
- All `pytest` tests MUST run deterministically offline against `tests/fixtures/mcd_sample/` (which MUST include at least one multi-version policy `v1 -> v2`, one retired policy, and one malformed policy) and mocked HTTP responses (`respx` or `monkeypatch`). Never call live CMS or LLM endpoints in tests.
- Authenticate all AWS workflows via **GitHub OIDC** only—never commit or configure static `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
- Whenever creating or editing `src/medcover/sql/*`, `api.py`, `Dockerfile`, `infra/*`, or `.github/workflows/*`, include a concise 3-bullet operational note (*what it does, why it is shaped this way, what failure mode it prevents*) so the human owner builds direct mastery over every operational layer.

---

## 5. Mandatory Verification Gate
Before reporting any task complete, run the verification commands below and confirm zero errors (POSIX environments may run `make check && make smoke` as shorthand):
1. `uv run ruff check . && uv run ruff format --check .`
2. `uv run mypy src`
3. `uv run sqlfluff lint src/medcover/sql`
4. `uv run python -m pytest`
5. Whenever pipeline, SQL, or retrieval logic changes: `uv run medcover build --sample && uv run medcover eval --sample`

If a requirement is genuinely ambiguous, state the single simplest assumption or ask one focused question—never hedge by writing multiple speculative code paths.
