"""Canonical PyArrow schema contracts, RunManifest, Arrow validators, and scalar parsers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal, get_args

import pyarrow
from pydantic import BaseModel, Field

DocType = Literal["NCD", "LCD", "ARTICLE"]
PolicyStatus = Literal["ACTIVE", "RETIRED"]
SourceMode = Literal["zip", "api"]
CodeSystem = Literal["HCPCS", "ICD10"]

VALID_DOC_TYPES: frozenset[str] = frozenset(get_args(DocType))
VALID_STATUSES: frozenset[str] = frozenset(get_args(PolicyStatus))
VALID_SOURCE_MODES: frozenset[str] = frozenset(get_args(SourceMode))
VALID_CODE_SYSTEMS: frozenset[str] = frozenset(get_args(CodeSystem))

_CMS_US_DATE_FORMATS: tuple[str, ...] = (
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
)


class RunManifest(BaseModel):
    """Immutable build provenance and lineage summary emitted alongside each Gold build."""

    run_id: str
    git_sha: str
    timestamp_utc: datetime
    source_modes: list[SourceMode] = Field(default_factory=list)
    duckdb_version: str
    duckdb_storage_version: str
    bronze_docs: int = Field(ge=0)
    silver_docs: int = Field(ge=0)
    quarantined_docs: int = Field(ge=0)
    gold_chunks: int = Field(ge=0)
    chunks_reused: int = Field(ge=0)
    chunks_embedded: int = Field(ge=0)
    token_p50: float = Field(ge=0.0)
    token_p95: float = Field(ge=0.0)

    @property
    def quarantine_rate(self) -> float:
        """Fraction of processed documents routed to quarantine."""
        total = self.silver_docs + self.quarantined_docs
        return (self.quarantined_docs / total) if total > 0 else 0.0


def parse_optional_str(value: Any) -> str | None:
    """Normalize a nullable value to a stripped non-empty string or ``None``.

    Inputs: arbitrary scalar from CSV, Parquet, or Arrow.
    Output grain: single ``str | None``.
    Invariants: treats ``None``, ``NaN``, and blank strings as ``None``.
    """
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def parse_optional_int(value: Any) -> int | None:
    """Parse a nullable integer without raising on malformed strings.

    Inputs: arbitrary scalar from CSV, Parquet, or Arrow.
    Output grain: single ``int | None``.
    Invariants: rejects booleans, non-integral floats, and non-numeric strings as ``None``.
    """
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = parse_optional_str(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_optional_date(value: Any) -> date | None:
    """Strictly parse ISO-8601 or CMS US date/datetime strings into ``datetime.date``.

    Inputs: nullable ``date``, ``datetime``, or string.
    Output grain: single ``date | None``.
    Invariants: rejects trailing garbage and malformed dates by returning ``None``.
    """
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = parse_optional_str(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in _CMS_US_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def validate_arrow_table(
    table: pyarrow.Table,
    expected_schema: pyarrow.Schema,
    *,
    unique_keys: Sequence[str] = (),
) -> None:
    """Validate an Arrow table strictly against a canonical schema, nullability, and key grain.

    Inputs: ``pyarrow.Table``, canonical ``pyarrow.Schema``, and optional composite key columns.
    Output grain: returns ``None`` on success; raises ``ValueError`` with concrete mismatch details.
    Invariants: zero Pandas conversion; enforces exact column set/order, Arrow types, non-null
    field constraints, and unique composite keys when provided.
    """
    actual_names = table.schema.names
    expected_names = expected_schema.names
    if actual_names != expected_names:
        msg = f"Schema column mismatch: expected {expected_names}, got {actual_names}"
        raise ValueError(msg)

    for expected_field in expected_schema:
        actual_field = table.schema.field(expected_field.name)
        if actual_field.type != expected_field.type:
            msg = (
                f"Column '{expected_field.name}' type mismatch: "
                f"expected {expected_field.type}, got {actual_field.type}"
            )
            raise ValueError(msg)
        if not expected_field.nullable:
            null_count = table.column(expected_field.name).null_count
            if null_count > 0:
                msg = f"Non-nullable column '{expected_field.name}' contains {null_count} null(s)"
                raise ValueError(msg)

    if unique_keys and table.num_rows > 1:
        key_columns = [table.column(col).to_pylist() for col in unique_keys]
        seen: set[tuple[Any, ...]] = set()
        for key in zip(*key_columns, strict=True):
            if key in seen:
                msg = f"Duplicate grain key on {tuple(unique_keys)}: {key!r}"
                raise ValueError(msg)
            seen.add(key)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


BRONZE_DOCUMENTS_ARROW_SCHEMA = pyarrow.schema(
    [
        pyarrow.field("doc_id", pyarrow.string(), nullable=False),
        pyarrow.field("doc_type", pyarrow.string(), nullable=True),
        pyarrow.field("version", pyarrow.int64(), nullable=False),
        pyarrow.field("title", pyarrow.string(), nullable=True),
        pyarrow.field("raw_html", pyarrow.string(), nullable=True),
        pyarrow.field("mac_jurisdiction", pyarrow.string(), nullable=True),
        pyarrow.field("effective_date", pyarrow.date32(), nullable=True),
        pyarrow.field("retired_date", pyarrow.date32(), nullable=True),
        pyarrow.field("status", pyarrow.string(), nullable=True),
        pyarrow.field("source_mode", pyarrow.string(), nullable=False),
        pyarrow.field("ingested_at", pyarrow.timestamp("us", tz="UTC"), nullable=False),
        pyarrow.field("source_sha256", pyarrow.string(), nullable=False),
    ]
)

BRONZE_CROSSWALKS_ARROW_SCHEMA = pyarrow.schema(
    [
        pyarrow.field("doc_id", pyarrow.string(), nullable=False),
        pyarrow.field("version", pyarrow.int64(), nullable=False),
        pyarrow.field("code_system", pyarrow.string(), nullable=False),
        pyarrow.field("code", pyarrow.string(), nullable=False),
    ]
)

SILVER_POLICIES_ARROW_SCHEMA = pyarrow.schema(
    [
        pyarrow.field("doc_id", pyarrow.string(), nullable=False),
        pyarrow.field("doc_type", pyarrow.string(), nullable=False),
        pyarrow.field("version", pyarrow.int64(), nullable=False),
        pyarrow.field("title", pyarrow.string(), nullable=False),
        pyarrow.field("markdown_text", pyarrow.string(), nullable=False),
        pyarrow.field("hcpcs_codes", pyarrow.list_(pyarrow.string()), nullable=False),
        pyarrow.field("icd10_codes", pyarrow.list_(pyarrow.string()), nullable=False),
        pyarrow.field("mac_jurisdiction", pyarrow.string(), nullable=True),
        pyarrow.field("effective_date", pyarrow.date32(), nullable=False),
        pyarrow.field("retired_date", pyarrow.date32(), nullable=True),
        pyarrow.field("valid_to", pyarrow.date32(), nullable=True),
        pyarrow.field("source_sha256", pyarrow.string(), nullable=False),
    ]
)

SILVER_QUARANTINE_ARROW_SCHEMA = pyarrow.schema(
    [
        pyarrow.field("doc_id", pyarrow.string(), nullable=True),
        pyarrow.field("doc_type", pyarrow.string(), nullable=True),
        pyarrow.field("version", pyarrow.int64(), nullable=True),
        pyarrow.field("title", pyarrow.string(), nullable=True),
        pyarrow.field("effective_date", pyarrow.date32(), nullable=True),
        pyarrow.field("retired_date", pyarrow.date32(), nullable=True),
        pyarrow.field("status", pyarrow.string(), nullable=True),
        pyarrow.field("source_mode", pyarrow.string(), nullable=True),
        pyarrow.field("source_sha256", pyarrow.string(), nullable=True),
        pyarrow.field("failure_reason", pyarrow.string(), nullable=False),
        pyarrow.field("quarantined_at", pyarrow.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def gold_chunks_arrow_schema(embedding_dim: int = 384) -> pyarrow.Schema:
    """Build the canonical Arrow schema for ``gold/chunks.parquet``.

    Inputs: expected fixed-size embedding vector dimension (default 384).
    Output grain: Arrow schema matching ``(doc_id, version, chunk_index)`` chunk rows.
    Invariants: fixed-size ``float32[dim]`` vectors, ``valid_to`` SCD2 bound, and list code arrays.
    """
    return pyarrow.schema(
        [
            pyarrow.field("chunk_id", pyarrow.string(), nullable=False),
            pyarrow.field("doc_id", pyarrow.string(), nullable=False),
            pyarrow.field("doc_type", pyarrow.string(), nullable=False),
            pyarrow.field("version", pyarrow.int64(), nullable=False),
            pyarrow.field("chunk_index", pyarrow.int64(), nullable=False),
            pyarrow.field("title", pyarrow.string(), nullable=False),
            pyarrow.field("section_heading", pyarrow.string(), nullable=False),
            pyarrow.field("chunk_text", pyarrow.string(), nullable=False),
            pyarrow.field("token_count", pyarrow.int64(), nullable=False),
            pyarrow.field("chunk_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("hcpcs_codes", pyarrow.list_(pyarrow.string()), nullable=False),
            pyarrow.field("icd10_codes", pyarrow.list_(pyarrow.string()), nullable=False),
            pyarrow.field("mac_jurisdiction", pyarrow.string(), nullable=True),
            pyarrow.field("effective_date", pyarrow.date32(), nullable=False),
            pyarrow.field("retired_date", pyarrow.date32(), nullable=True),
            pyarrow.field("valid_to", pyarrow.date32(), nullable=True),
            pyarrow.field(
                "embedding", pyarrow.list_(pyarrow.float32(), embedding_dim), nullable=False
            ),
            pyarrow.field("embedding_model", pyarrow.string(), nullable=False),
        ]
    )
