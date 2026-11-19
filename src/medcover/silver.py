"""Silver layer HTML-to-Markdown cleaning, quarantine routing, and DuckDB code aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from markdownify import markdownify

from medcover.config import Settings
from medcover.contracts import (
    VALID_DOC_TYPES,
    VALID_SOURCE_MODES,
    VALID_STATUSES,
    validate_silver_policies,
    validate_silver_quarantine,
)
from medcover.sql import load_sql

SILVER_QUARANTINE_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "doc_type",
    "version",
    "title",
    "effective_date",
    "retired_date",
    "status",
    "source_mode",
    "source_sha256",
    "failure_reason",
    "quarantined_at",
)

SILVER_POLICIES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("doc_type", pa.string(), nullable=False),
        pa.field("version", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("markdown_text", pa.string(), nullable=False),
        pa.field("cpt_codes", pa.list_(pa.string()), nullable=False),
        pa.field("icd10_codes", pa.list_(pa.string()), nullable=False),
        pa.field("mac_jurisdiction", pa.string(), nullable=True),
        pa.field("effective_date", pa.date32(), nullable=False),
        pa.field("retired_date", pa.date32(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
        pa.field("source_mode", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
    ]
)

_BLANK_LINES_RE = re.compile(r"\n{3,}")


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
        """Fraction of Bronze documents routed to quarantine."""
        if self.bronze_docs == 0:
            return 0.0
        return self.quarantined_docs / self.bronze_docs


def html_to_markdown(raw_html: str | None) -> str:
    """Convert raw CMS policy HTML into structured Markdown preserving headings and tables.

    Inputs: raw HTML string (nullable).
    Output grain: single cleaned Markdown string (empty string if input has no textual content).
    Invariants: preserves ATX headings (``#``, ``##``, ``###``) and Markdown pipe tables;
    strips ``<script>`` and ``<style>`` blocks cleanly without naive tag-stripping regexes.
    """
    if raw_html is None or not raw_html.strip():
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    if not soup.get_text(strip=True):
        return ""

    rendered = str(
        markdownify(
            str(soup),
            heading_style="ATX",
            bullets="-",
            strip=["img"],
        )
    )
    lines = [line.rstrip() for line in rendered.replace("\r\n", "\n").splitlines()]
    normalized = _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()
    return normalized


def build_silver(settings: Settings) -> SilverResult:
    """Transform Bronze Parquet tables into grain-safe Silver policies and quarantine records.

    Inputs: ``Settings`` locating ``bronze_dir`` and ``silver_dir``.
    Output grain: ``silver/policies.parquet`` strictly 1 row per ``(doc_id, version)`` with
    aggregated ``cpt_codes`` and ``icd10_codes`` arrays, plus ``silver/quarantine.parquet``.
    Invariants: malformed documents never enter ``policies.parquet`` and always record an
    explicit ``failure_reason`` in ``quarantine.parquet``.
    """
    bronze_docs_path = settings.bronze_dir / "raw_documents.parquet"
    bronze_codes_path = settings.bronze_dir / "raw_code_crosswalks.parquet"
    if not bronze_docs_path.exists():
        msg = f"Bronze documents file not found: {bronze_docs_path}"
        raise FileNotFoundError(msg)

    bronze_docs = pd.read_parquet(bronze_docs_path)
    bronze_codes = (
        pd.read_parquet(bronze_codes_path)
        if bronze_codes_path.exists()
        else pd.DataFrame(columns=["doc_id", "version", "code_system", "code"])
    )

    valid_rows, quarantined_rows = _partition_and_clean(bronze_docs)

    settings.silver_dir.mkdir(parents=True, exist_ok=True)
    policies_path = settings.silver_dir / "policies.parquet"
    quarantine_path = settings.silver_dir / "quarantine.parquet"

    quarantine_df = _build_quarantine_frame(quarantined_rows)
    validate_silver_quarantine(quarantine_df)
    quarantine_df.to_parquet(quarantine_path, index=False)

    policies_df = _aggregate_silver_policies(valid_rows, bronze_codes)
    validate_silver_policies(policies_df)
    _write_silver_policies_parquet(policies_df, policies_path)

    return SilverResult(
        policies_path=policies_path,
        quarantine_path=quarantine_path,
        bronze_docs=len(bronze_docs),
        silver_docs=len(policies_df),
        quarantined_docs=len(quarantine_df),
    )


def _partition_and_clean(
    bronze_docs: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    quarantined_rows: list[dict[str, Any]] = []
    quarantined_at = datetime.now(tz=UTC)

    for record in bronze_docs.to_dict(orient="records"):
        doc_id = _opt_str(record.get("doc_id"))
        doc_type = _opt_str(record.get("doc_type"))
        version = _opt_int(record.get("version"))
        title = _opt_str(record.get("title"))
        raw_html = _opt_str(record.get("raw_html"))
        mac_jurisdiction = _opt_str(record.get("mac_jurisdiction"))
        effective_date = _opt_date(record.get("effective_date"))
        retired_date = _opt_date(record.get("retired_date"))
        status = _opt_str(record.get("status"))
        source_mode = _opt_str(record.get("source_mode"))
        source_sha256 = _opt_str(record.get("source_sha256"))

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
                "status": status,
                "source_mode": source_mode,
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
    if not status or status not in VALID_STATUSES:
        reasons.append("invalid_status")
    if not source_mode or source_mode not in VALID_SOURCE_MODES:
        reasons.append("invalid_source_mode")
    if retired_date is not None and effective_date is not None and retired_date < effective_date:
        reasons.append("retired_date_before_effective_date")
    if not markdown_text.strip():
        reasons.append("empty_cleaned_text")
    return reasons


def _aggregate_silver_policies(
    valid_rows: list[dict[str, Any]],
    bronze_codes: pd.DataFrame,
) -> pd.DataFrame:
    if not valid_rows:
        return pd.DataFrame(
            columns=[field.name for field in SILVER_POLICIES_ARROW_SCHEMA],
        )

    valid_table = pa.Table.from_pylist(
        valid_rows,
        schema=pa.schema(
            [
                pa.field("doc_id", pa.string()),
                pa.field("doc_type", pa.string()),
                pa.field("version", pa.int64()),
                pa.field("title", pa.string()),
                pa.field("markdown_text", pa.string()),
                pa.field("mac_jurisdiction", pa.string()),
                pa.field("effective_date", pa.date32()),
                pa.field("retired_date", pa.date32()),
                pa.field("status", pa.string()),
                pa.field("source_mode", pa.string()),
                pa.field("source_sha256", pa.string()),
            ]
        ),
    )

    codes_table = pa.Table.from_pandas(
        bronze_codes.reindex(columns=["doc_id", "version", "code_system", "code"]),
        schema=pa.schema(
            [
                pa.field("doc_id", pa.string()),
                pa.field("version", pa.int64()),
                pa.field("code_system", pa.string()),
                pa.field("code", pa.string()),
            ]
        ),
        preserve_index=False,
    )

    sql = load_sql("silver_policies.sql")
    with duckdb.connect(":memory:") as con:
        con.register("silver_valid_docs", valid_table)
        con.register("raw_code_crosswalks", codes_table)
        result_df = con.execute(sql).df()

    result_df["cpt_codes"] = result_df["cpt_codes"].map(lambda value: list(value))
    result_df["icd10_codes"] = result_df["icd10_codes"].map(lambda value: list(value))
    result_df["mac_jurisdiction"] = result_df["mac_jurisdiction"].map(_opt_str)
    result_df["effective_date"] = result_df["effective_date"].map(_opt_date)
    result_df["retired_date"] = result_df["retired_date"].map(_opt_date)
    result_df["is_active"] = result_df["is_active"].astype(bool)
    return result_df


def _write_silver_policies_parquet(df: pd.DataFrame, output_path: Path) -> None:
    records = [
        {
            key: (None if isinstance(value, float) and pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]
    table = pa.Table.from_pylist(records, schema=SILVER_POLICIES_ARROW_SCHEMA)
    pq.write_table(table, output_path)


def _build_quarantine_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(SILVER_QUARANTINE_COLUMNS))
    return pd.DataFrame(rows).reindex(columns=list(SILVER_QUARANTINE_COLUMNS))


def _opt_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def _opt_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _opt_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _opt_str(value)
    return date.fromisoformat(text[:10]) if text else None
