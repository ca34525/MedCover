# AGENTS.md — MedCover RAG Pipeline

## 1. Mission & Classical Engineering Standard
Build and maintain **MedCover**, a clean, human-readable Medallion RAG pipeline (Bronze -> Silver -> Gold -> S3 WAP -> ECS Fargate FastAPI) over the full CMS Medicare Coverage Database.

Write every module, SQL query, test, and workflow as if a senior data engineer will inspect it cold in five minutes. Elegance comes from **economy of means**: establish the simplest complete foundation first, then recombine a small set of explicit building blocks rather than inventing abstractions.

### Four Core Rules
1. **Anchor the Foundation First:** Implement the working happy path end-to-end before adding configuration flags, retry tuning, or optimizations.
2. **Recombine Small Building Blocks:** Map responsibilities cleanly onto existing modules (`cms_api.py`, `bronze.py`, `silver.py`, `gold.py`, `quality.py`, `publish.py`, `retrieve.py`, `eval.py`, `api.py`, `cli.py`) and explicit schemas in `contracts.py`.
3. **Separate Signal from Mechanics:** Keep top-level stage functions short and readable top-to-bottom. Push HTTP token refresh, HTML parsing edge cases, and S3 I/O into small private helpers so data flow stays unmistakable.
4. **Anti-Slop Rule (Strictly Banned):**
   - NO `BasePipeline`, `AbstractIngestor`, `PipelineManager`, `ServiceFactory`, or generic `utils.py` / `helpers.py` dumping grounds.
   - NO orchestration frameworks (Airflow, Dagster), NO RAG frameworks (LangChain, LlamaIndex), NO vendor LLM SDKs (`openai`, `anthropic`, `google-genai`), and NO custom frontend HTML/JS/Gradio UI (FastAPI `/docs` is the sole UI).
   - NO new dependency in `pyproject.toml` without a one-sentence justification explaining why stdlib or an existing package cannot do the job.

---

## 2. Human-Standard Conventions
- **Packaging:** Use `uv`, PEP 621 `pyproject.toml`, and `src/medcover/`. Every SQL file lives under `src/medcover/sql/` and MUST load via `importlib.resources.files("medcover.sql")` so built wheels and Docker containers never fail on relative paths.
- **SQL Discipline:** All relational joins, DQ invariants, and hybrid retrieval queries belong in `.sql` files (no multi-line SQL >5 lines inside `.py` files). All SQL MUST pass `sqlfluff lint`.
- **Typing & Docstrings:** Public functions require complete PEP 484 type annotations (`mypy src` clean) and brief Google-style docstrings stating *inputs, output grain, and invariants*—never restating argument names. Format and lint strictly with `ruff`.
- **Explicit Failure Routing:** Never swallow errors (`except Exception: pass` is banned). Malformed input records MUST route to `silver/quarantine.parquet` with an explicit `failure_reason`.

---

## 3. Non-Negotiable Data, Performance & Security Invariants
Verify every invariant below before finishing any change:
1. **Strict Grain Preservation:**
   - `bronze/raw_documents.parquet`: unique `(doc_id, version)`.
   - `silver/policies.parquet`: unique `(doc_id, version)`. Aggregate HCPCS/CPT (`hcpcs_codes`) and ICD-10 (`icd10_codes`) crosswalks with `LIST(DISTINCT code)`—never join 1-to-many code rows directly onto policy text.
   - `gold/chunks.parquet`: unique `(doc_id, version, chunk_index)` with deterministic `chunk_id` and `chunk_sha256`.
2. **Idempotent Incremental Embedding:**
   - Re-running `medcover build` on unchanged inputs MUST produce identical Gold counts and `chunks_embedded == 0`.
   - Mutating 1 source document MUST re-embed strictly the chunks whose `chunk_sha256` changed.
3. **Structure-Preserving HTML Parsing:**
   - Convert HTML to Markdown preserving heading hierarchy (`#`, `##`) and tables. Never flatten tables with naive regex stripping.
4. **Build-Time Indexing & Pre-Baked Runtime:**
   - `gold.py` MUST build `index.duckdb` (including `PRAGMA create_fts_index(...)` plus query-time `chunks_as_of(as_of_date)` macro and `active_chunks` view) at pipeline build time; `api.py` MUST open `index.duckdb` in `read_only=True` mode and NEVER rebuild FTS indexes during container startup.
   - `Dockerfile` MUST pre-download the `BAAI/bge-small-en-v1.5` ONNX model during `docker build` so ECS containers start with zero external model downloads.
5. **Temporal & WAP Safety:**
   - Every Gold chunk MUST store the timeless validity interval `[effective_date, retired_date)` (never a materialized `is_active` column). Queries with `as_of_date` MUST evaluate `effective_date <= as_of_date AND (retired_date IS NULL OR retired_date > as_of_date)` and achieve `temporal_filter_violations == 0`.
   - `run_manifest.json` MUST remain an immutable build record written by `gold.py`; `eval.py` MUST write evaluation output separately to `eval_report.json`.
   - `publish.py` MUST NEVER update `latest.json` unless Pandera contracts, `dq_invariants.sql`, and `eval.py` all pass.
6. **Zero-Trust BYO-Key Handling:**
   - Pass `X-LLM-API-Key` strictly in memory on the outbound `httpx` request. NEVER persist, cache, log, or echo the key in logs, exceptions, or responses.

---

## 4. Testing, Infra & CI/CD Standards
- All `pytest` tests MUST run deterministically offline against `tests/fixtures/mcd_sample/` and mocked HTTP responses (`respx` or `monkeypatch`). Never call live CMS or LLM endpoints in tests.
- Authenticate all AWS workflows via **GitHub OIDC** only—never commit or configure static `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
- Whenever editing `Dockerfile`, `infra/*`, or `.github/workflows/*`, state in 1–2 sentences what specific failure mode the change prevents so the human owner retains full operational understanding.

---

## 5. Mandatory Verification Gate
Before reporting any task complete, run `make check` and confirm zero errors:
1. `uv run ruff check . && uv run ruff format --check .`
2. `uv run mypy src`
3. `uv run sqlfluff lint src/medcover/sql`
4. `uv run pytest`
5. Whenever pipeline, SQL, or retrieval logic changes: `uv run medcover build --sample && uv run medcover eval`

If a requirement is genuinely ambiguous, state the single simplest assumption or ask one focused question—never hedge by writing multiple speculative code paths.