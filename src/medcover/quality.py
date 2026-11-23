"""Data quality audit (Pandera + dq_invariants.sql), quarantine threshold, and manifest summary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd
import pandera.pandas as pa
import pyarrow as arrow
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from medcover.config import Settings
from medcover.contracts import (
    SILVER_POLICIES_SCHEMA,
    SILVER_QUARANTINE_SCHEMA,
    RunManifest,
    build_gold_chunks_schema,
)
from medcover.sql import load_sql

if TYPE_CHECKING:
    from medcover.eval import EvalReport


class AuditCheck(BaseModel):
    """Single Pandera, SQL invariant, or artifact consistency assertion outcome."""

    name: str
    passed: bool
    violation_count: int = Field(ge=0)
    detail: str


class AuditReport(BaseModel):
    """Structured output of a Medallion data quality audit gate."""

    passed: bool
    silver_docs: int = Field(ge=0)
    quarantined_docs: int = Field(ge=0)
    gold_chunks: int = Field(ge=0)
    checks: list[AuditCheck]
    manifest: RunManifest | None = None

    @property
    def quarantine_rate(self) -> float:
        """Fraction of processed documents routed to quarantine."""
        total = self.silver_docs + self.quarantined_docs
        return (self.quarantined_docs / total) if total > 0 else 0.0

    @property
    def failed_checks(self) -> list[AuditCheck]:
        """Return only checks that failed audit verification."""
        return [check for check in self.checks if not check.passed]


def audit_medallion(settings: Settings) -> AuditReport:
    """Execute Pandera contracts, ``dq_invariants.sql``, and Gold index/manifest checks.

    Inputs: ``Settings`` pointing to built Silver and Gold artifact directories.
    Output grain: single ``AuditReport`` summarizing every contract and SQL invariant check.
    Invariants: never mutates artifacts; opens ``index.duckdb`` strictly read-only; returns
    ``passed=False`` on any schema error, SQL violation, index mismatch, or quarantine overflow.
    """
    policies_path = settings.silver_dir / "policies.parquet"
    quarantine_path = settings.silver_dir / "quarantine.parquet"
    chunks_path = settings.gold_dir / "chunks.parquet"
    index_path = settings.gold_dir / "index.duckdb"
    manifest_path = settings.gold_dir / "run_manifest.json"

    for required_path in (policies_path, quarantine_path, chunks_path, index_path):
        if not required_path.exists():
            msg = f"Required audit artifact not found: {required_path}"
            raise FileNotFoundError(msg)

    policies_table = pq.read_table(policies_path)
    quarantine_table = pq.read_table(quarantine_path)
    chunks_table = pq.read_table(chunks_path)

    silver_docs = policies_table.num_rows
    quarantined_docs = quarantine_table.num_rows
    gold_chunks = chunks_table.num_rows

    checks: list[AuditCheck] = [
        _run_pandera_check("pandera_silver_policies", SILVER_POLICIES_SCHEMA, policies_table),
        _run_pandera_check("pandera_silver_quarantine", SILVER_QUARANTINE_SCHEMA, quarantine_table),
        _run_pandera_check(
            "pandera_gold_chunks",
            build_gold_chunks_schema(settings.embedding_dim),
            chunks_table,
        ),
        _check_gold_index_sync(
            index_path=index_path,
            expected_chunks=gold_chunks,
            extension_dir=settings.resolved_duckdb_extension_dir(),
        ),
    ]
    checks.extend(
        _run_sql_invariants(
            policies_table=policies_table,
            quarantine_table=quarantine_table,
            chunks_table=chunks_table,
            settings=settings,
        )
    )

    manifest: RunManifest | None = None
    if manifest_path.exists():
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        checks.append(
            _check_manifest_alignment(
                manifest=manifest,
                silver_docs=silver_docs,
                quarantined_docs=quarantined_docs,
                gold_chunks=gold_chunks,
            )
        )

    return AuditReport(
        passed=all(check.passed for check in checks),
        silver_docs=silver_docs,
        quarantined_docs=quarantined_docs,
        gold_chunks=gold_chunks,
        checks=checks,
        manifest=manifest,
    )


def render_manifest_summary(
    manifest: RunManifest,
    audit: AuditReport | None = None,
    eval_report: EvalReport | None = None,
) -> str:
    """Format a RunManifest and optional AuditReport / EvalReport into GitHub-flavored Markdown.

    Inputs: immutable ``RunManifest`` plus optional ``AuditReport`` and ``EvalReport``.
    Output grain: single Markdown document suitable for ``$GITHUB_STEP_SUMMARY`` or logs.
    Invariants: deterministic formatting with explicit pass/fail audit status when provided.
    """
    source_modes = ", ".join(manifest.source_modes) if manifest.source_modes else "none"
    recall = f"{eval_report.recall_at_5:.3f}" if eval_report is not None else "n/a"
    mrr = f"{eval_report.mrr:.3f}" if eval_report is not None else "n/a"
    violations = str(eval_report.temporal_filter_violations) if eval_report is not None else "n/a"
    lines = [
        f"## MedCover Run Summary (`{manifest.run_id}`)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Source Modes | `{source_modes}` |",
        f"| Git SHA | `{manifest.git_sha}` |",
        f"| Bronze Docs | {manifest.bronze_docs} |",
        f"| Silver Docs | {manifest.silver_docs} |",
        f"| Quarantined Docs | {manifest.quarantined_docs} ({manifest.quarantine_rate:.2%}) |",
        f"| Gold Chunks | {manifest.gold_chunks} |",
        f"| Chunks Reused / Embedded | {manifest.chunks_reused} / {manifest.chunks_embedded} |",
        f"| Chunk Tokens (p50 / p95) | {manifest.token_p50} / {manifest.token_p95} |",
        f"| Eval Recall@5 / MRR | {recall} / {mrr} |",
        f"| Temporal Filter Violations | {violations} |",
    ]
    if audit is not None:
        status = "PASSED" if audit.passed else "FAILED"
        lines.extend(
            [
                "",
                f"### Data Quality Audit: {status}",
                "",
                "| Check | Status | Violations | Detail |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for check in audit.checks:
            badge = "PASS" if check.passed else "FAIL"
            lines.append(f"| `{check.name}` | {badge} | {check.violation_count} | {check.detail} |")
    return "\n".join(lines) + "\n"


def _run_pandera_check(
    name: str,
    schema: pa.DataFrameSchema,
    table: arrow.Table,
) -> AuditCheck:
    df: pd.DataFrame = table.to_pandas()
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failure_count = max(1, len(exc.failure_cases))
        first_case = exc.failure_cases.iloc[0].to_dict() if not exc.failure_cases.empty else {}
        detail = f"Pandera contract failure: {first_case}"
        return AuditCheck(name=name, passed=False, violation_count=failure_count, detail=detail)
    except pa.errors.SchemaError as exc:
        return AuditCheck(name=name, passed=False, violation_count=1, detail=str(exc))

    return AuditCheck(
        name=name,
        passed=True,
        violation_count=0,
        detail=f"Validated {len(df)} rows against {name}",
    )


def _check_gold_index_sync(
    *,
    index_path: Path,
    expected_chunks: int,
    extension_dir: Path | None,
) -> AuditCheck:
    config: dict[str, str | bool | int | float | list[str]] = {}
    if extension_dir is not None:
        config["extension_directory"] = str(extension_dir.resolve())

    try:
        with duckdb.connect(str(index_path), read_only=True, config=config) as con:
            con.execute("LOAD fts")
            row = con.execute("SELECT COUNT(*) FROM chunks").fetchone()
            index_count = int(row[0]) if row is not None else -1
            fts_row = con.execute(
                "SELECT COUNT(*) FROM duckdb_schemas() WHERE schema_name = 'fts_main_chunks'"
            ).fetchone()
            has_fts_index = bool(fts_row and int(fts_row[0]) > 0)
            con.execute("SELECT COUNT(*) FROM active_chunks").fetchone()
    except duckdb.Error as exc:
        return AuditCheck(
            name="gold_index_sync",
            passed=False,
            violation_count=1,
            detail=f"Failed to open index.duckdb read-only with FTS: {exc}",
        )

    if not has_fts_index:
        return AuditCheck(
            name="gold_index_sync",
            passed=False,
            violation_count=1,
            detail="Pre-built fts_main_chunks index schema is missing from index.duckdb",
        )

    if index_count != expected_chunks:
        return AuditCheck(
            name="gold_index_sync",
            passed=False,
            violation_count=abs(index_count - expected_chunks),
            detail=f"index.duckdb chunks ({index_count}) != chunks.parquet ({expected_chunks})",
        )

    return AuditCheck(
        name="gold_index_sync",
        passed=True,
        violation_count=0,
        detail=f"index.duckdb read-only FTS verified ({index_count} chunks)",
    )


def _run_sql_invariants(
    *,
    policies_table: arrow.Table,
    quarantine_table: arrow.Table,
    chunks_table: arrow.Table,
    settings: Settings,
) -> list[AuditCheck]:
    params_schema = arrow.schema(
        [
            arrow.field("embedding_dim", arrow.int64(), nullable=False),
            arrow.field("max_chunk_tokens", arrow.int64(), nullable=False),
            arrow.field("max_quarantine_rate", arrow.float64(), nullable=False),
        ]
    )
    dq_params = arrow.Table.from_pylist(
        [
            {
                "embedding_dim": settings.embedding_dim,
                "max_chunk_tokens": settings.chunk_token_limit,
                "max_quarantine_rate": settings.max_quarantine_rate,
            }
        ],
        schema=params_schema,
    )

    sql = load_sql("dq_invariants.sql")
    with duckdb.connect(":memory:") as con:
        con.register("silver_policies", policies_table)
        con.register("silver_quarantine", quarantine_table)
        con.register("gold_chunks", chunks_table)
        con.register("dq_params", dq_params)
        rows: list[dict[str, Any]] = con.sql(sql).to_arrow_table().to_pylist()

    checks: list[AuditCheck] = []
    for row in rows:
        violations = int(row["violation_count"])
        checks.append(
            AuditCheck(
                name=str(row["check_name"]),
                passed=violations == 0,
                violation_count=violations,
                detail=str(row["detail"]),
            )
        )
    return checks


def _check_manifest_alignment(
    *,
    manifest: RunManifest,
    silver_docs: int,
    quarantined_docs: int,
    gold_chunks: int,
) -> AuditCheck:
    mismatches: list[str] = []
    if manifest.silver_docs != silver_docs:
        mismatches.append(f"silver_docs={manifest.silver_docs} != {silver_docs}")
    if manifest.quarantined_docs != quarantined_docs:
        mismatches.append(f"quarantined_docs={manifest.quarantined_docs} != {quarantined_docs}")
    if manifest.gold_chunks != gold_chunks:
        mismatches.append(f"gold_chunks={manifest.gold_chunks} != {gold_chunks}")

    if mismatches:
        return AuditCheck(
            name="manifest_counts_match",
            passed=False,
            violation_count=len(mismatches),
            detail="; ".join(mismatches),
        )
    return AuditCheck(
        name="manifest_counts_match",
        passed=True,
        violation_count=0,
        detail="run_manifest.json counts match Silver and Gold Parquet tables",
    )
