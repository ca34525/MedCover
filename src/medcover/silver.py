"""Silver layer HTML-to-Markdown cleaning, quarantine routing, and DuckDB code aggregation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from medcover.config import Settings
from medcover.contracts import (
    BRONZE_CROSSWALKS_ARROW_SCHEMA,
    SILVER_POLICIES_ARROW_SCHEMA,
    SILVER_POLICIES_SCHEMA,
    SILVER_QUARANTINE_ARROW_SCHEMA,
    SILVER_QUARANTINE_SCHEMA,
    VALID_DOC_TYPES,
    VALID_SOURCE_MODES,
    VALID_STATUSES,
    parse_optional_date,
    parse_optional_int,
    parse_optional_str,
)
from medcover.sql import load_sql

_BLANK_LINES_RE = re.compile(r"\n{3,}")

_SILVER_VALID_DOCS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("version", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("markdown_text", pa.string()),
        pa.field("mac_jurisdiction", pa.string()),
        pa.field("effective_date", pa.date32()),
        pa.field("retired_date", pa.date32()),
        pa.field("source_sha256", pa.string()),
    ]
)


@dataclass(frozen=True)
class SilverResult:
    """Summary of Silver policy transformation and quarantine routing."""

    policies_path: Path
    quarantine_path: Path
    bronze_docs: int
    silver_docs: int
    quarantined_docs: int

    @property
    def quarantine_rate(self) -> float:
        """Fraction of processed documents routed to quarantine."""
        total = self.silver_docs + self.quarantined_docs
        if total == 0:
            return 0.0
        return self.quarantined_docs / total


def html_to_markdown(raw_html: str | None) -> str:
    """Convert raw CMS policy HTML into structured Markdown preserving headings and tables.

    Inputs: raw HTML string (nullable).
    Output grain: single cleaned Markdown string (empty string if input has no textual content).
    Invariants: parses the HTML DOM once, preserves ATX headings (``#``, ``##``, ``###``) and
    Markdown pipe tables, and strips ``<script>`` / ``<style>`` / ``<noscript>`` cleanly.
    """
    if raw_html is None or not raw_html.strip():
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    if not soup.get_text(strip=True):
        return ""

    rendered = str(
        MarkdownConverter(heading_style="ATX", bullets="-", strip=["img"]).convert_soup(soup)
    )
    lines = [line.rstrip() for line in rendered.replace("\r\n", "\n").splitlines()]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def build_silver(settings: Settings) -> SilverResult:
    """Transform Bronze Parquet tables into grain-safe Silver policies and quarantine records.

    Inputs: ``Settings`` locating ``bronze_dir`` and ``silver_dir``.
    Output grain: ``silver/policies.parquet`` strictly 1 row per ``(doc_id, version)`` with
    aggregated ``hcpcs_codes`` and ``icd10_codes`` arrays, plus ``silver/quarantine.parquet``.
    Invariants: malformed documents never enter ``policies.parquet`` and always record an
    explicit ``failure_reason`` in ``quarantine.parquet``.
    """
    bronze_docs_path = settings.bronze_dir / "raw_documents.parquet"
    bronze_codes_path = settings.bronze_dir / "raw_code_crosswalks.parquet"
    if not bronze_docs_path.exists():
        msg = f"Bronze documents file not found: {bronze_docs_path}"
        raise FileNotFoundError(msg)

    bronze_docs = pq.read_table(bronze_docs_path).to_pylist()
    codes_table = (
        pq.read_table(bronze_codes_path)
        .select(BRONZE_CROSSWALKS_ARROW_SCHEMA.names)
        .cast(BRONZE_CROSSWALKS_ARROW_SCHEMA)
        if bronze_codes_path.exists()
        else pa.Table.from_pylist([], schema=BRONZE_CROSSWALKS_ARROW_SCHEMA)
    )

    valid_rows, quarantined_rows = _partition_and_clean(bronze_docs)

    settings.silver_dir.mkdir(parents=True, exist_ok=True)
    policies_path = settings.silver_dir / "policies.parquet"
    quarantine_path = settings.silver_dir / "quarantine.parquet"

    quarantine_table = pa.Table.from_pylist(
        quarantined_rows,
        schema=SILVER_QUARANTINE_ARROW_SCHEMA,
    )
    SILVER_QUARANTINE_SCHEMA.validate(quarantine_table.to_pandas())
    pq.write_table(quarantine_table, quarantine_path)

    policies_table = _aggregate_silver_policies(valid_rows, codes_table)
    SILVER_POLICIES_SCHEMA.validate(policies_table.to_pandas())
    pq.write_table(policies_table, policies_path)

    return SilverResult(
        policies_path=policies_path,
        quarantine_path=quarantine_path,
        bronze_docs=len(bronze_docs),
        silver_docs=policies_table.num_rows,
        quarantined_docs=quarantine_table.num_rows,
    )


def _partition_and_clean(
    bronze_docs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    quarantined_rows: list[dict[str, Any]] = []
    quarantined_at = datetime.now(tz=UTC)

    for record in bronze_docs:
        doc_id = parse_optional_str(record.get("doc_id"))
        doc_type = parse_optional_str(record.get("doc_type"))
        version = parse_optional_int(record.get("version"))
        title = parse_optional_str(record.get("title"))
        raw_html = parse_optional_str(record.get("raw_html"))
        mac_jurisdiction = parse_optional_str(record.get("mac_jurisdiction"))
        effective_date = parse_optional_date(record.get("effective_date"))
        retired_date = parse_optional_date(record.get("retired_date"))
        status = parse_optional_str(record.get("status"))
        source_mode = parse_optional_str(record.get("source_mode"))
        source_sha256 = parse_optional_str(record.get("source_sha256"))

        markdown_text = html_to_markdown(raw_html)
        reasons = _collect_failure_reasons(
            doc_id=doc_id,
            doc_type=doc_type,
            version=version,
            title=title,
            markdown_text=markdown_text,
            effective_date=effective_date,
            retired_date=retired_date,
            status=status,
            source_mode=source_mode,
        )

        if reasons:
            quarantined_rows.append(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "version": version,
                    "title": title,
                    "effective_date": effective_date,
                    "retired_date": retired_date,
                    "status": status,
                    "source_mode": source_mode,
                    "source_sha256": source_sha256,
                    "failure_reason": "; ".join(reasons),
                    "quarantined_at": quarantined_at,
                }
            )
            continue

        valid_rows.append(
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "version": version,
                "title": title,
                "markdown_text": markdown_text,
                "mac_jurisdiction": mac_jurisdiction,
                "effective_date": effective_date,
                "retired_date": retired_date,
                "source_sha256": source_sha256,
            }
        )

    return valid_rows, quarantined_rows


def _collect_failure_reasons(
    *,
    doc_id: str | None,
    doc_type: str | None,
    version: int | None,
    title: str | None,
    markdown_text: str,
    effective_date: date | None,
    retired_date: date | None,
    status: str | None,
    source_mode: str | None,
) -> list[str]:
    reasons: list[str] = []
    if not doc_id:
        reasons.append("missing_doc_id")
    if version is None or version < 1:
        reasons.append("invalid_version")
    if not doc_type or doc_type not in VALID_DOC_TYPES:
        reasons.append("invalid_doc_type")
    if not title:
        reasons.append("missing_title")
    if effective_date is None:
        reasons.append("missing_effective_date")
    if status is not None and status not in VALID_STATUSES:
        reasons.append("invalid_status")
    if status == "RETIRED" and retired_date is None:
        reasons.append("retired_without_date")
    if not source_mode or source_mode not in VALID_SOURCE_MODES:
        reasons.append("invalid_source_mode")
    if retired_date is not None and effective_date is not None and retired_date < effective_date:
        reasons.append("retired_date_before_effective_date")
    if not markdown_text.strip():
        reasons.append("empty_cleaned_text")
    return reasons


def _aggregate_silver_policies(
    valid_rows: list[dict[str, Any]],
    codes_table: pa.Table,
) -> pa.Table:
    if not valid_rows:
        return pa.Table.from_pylist([], schema=SILVER_POLICIES_ARROW_SCHEMA)

    valid_table = pa.Table.from_pylist(valid_rows, schema=_SILVER_VALID_DOCS_SCHEMA)
    sql = load_sql("silver_policies.sql")
    with duckdb.connect(":memory:") as con:
        con.register("silver_valid_docs", valid_table)
        con.register("raw_code_crosswalks", codes_table)
        arrow_result = con.sql(sql).to_arrow_table()

    return arrow_result.select(SILVER_POLICIES_ARROW_SCHEMA.names).cast(
        SILVER_POLICIES_ARROW_SCHEMA
    )
