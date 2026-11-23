"""Canonical Arrow schemas, Pandera contracts, RunManifest, and scalar parsers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, get_args

import pandas as pd
import pandera.pandas as pa
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

    Inputs: arbitrary scalar from CSV, Parquet, Arrow, or Pandas.
    Output grain: single ``str | None``.
    Invariants: treats ``None``, ``pd.NaT``, ``NaN``, and blank strings as ``None``.
    """
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def parse_optional_int(value: Any) -> int | None:
    """Parse a nullable integer without raising on malformed strings.

    Inputs: arbitrary scalar from CSV, Parquet, Arrow, or Pandas.
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

    Inputs: nullable ``date``, ``datetime``, ``pd.Timestamp``, or string.
    Output grain: single ``date | None``.
    Invariants: rejects trailing garbage and malformed dates by returning ``None``.
    """
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
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


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, float) and bool(pd.isna(value)):
        return True
    return isinstance(value, pd.Timestamp) and bool(pd.isna(value))


def _all_non_empty_strings(series: pd.Series[Any]) -> bool:
    return bool(series.map(lambda value: isinstance(value, str) and bool(value.strip())).all())


def _all_string_lists(series: pd.Series[Any]) -> bool:
    return bool(
        series.map(
            lambda value: (
                (
                    isinstance(value, (list, tuple))
                    or (hasattr(value, "tolist") and isinstance(value.tolist(), list))
                )
                and all(isinstance(item, str) for item in list(value))
            )
        ).all()
    )


def _valid_embedding_vectors(series: pd.Series[Any], dim: int) -> bool:
    for value in series:
        vec = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(vec, list) or len(vec) != dim:
            return False
        if any(item is None or item != item for item in vec):
            return False
    return True


def _retired_not_before_effective(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    mask = df["retired_date"].notna() & df["effective_date"].notna()
    if not mask.any():
        return True
    retired = pd.to_datetime(df.loc[mask, "retired_date"]).dt.date
    effective = pd.to_datetime(df.loc[mask, "effective_date"]).dt.date
    return bool((retired >= effective).all())


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
    Invariants: fixed-size ``float32[dim]`` vectors and list-typed HCPCS/ICD-10 code arrays.
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
            pyarrow.field(
                "embedding", pyarrow.list_(pyarrow.float32(), embedding_dim), nullable=False
            ),
            pyarrow.field("embedding_model", pyarrow.string(), nullable=False),
        ]
    )


BRONZE_DOCUMENTS_SCHEMA = pa.DataFrameSchema(
    columns={
        "doc_id": pa.Column(str, nullable=False),
        "doc_type": pa.Column(None, nullable=True),
        "version": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
        "title": pa.Column(None, nullable=True),
        "raw_html": pa.Column(None, nullable=True),
        "mac_jurisdiction": pa.Column(None, nullable=True),
        "effective_date": pa.Column(None, nullable=True),
        "retired_date": pa.Column(None, nullable=True),
        "status": pa.Column(None, nullable=True),
        "source_mode": pa.Column(
            str,
            checks=pa.Check.isin(sorted(VALID_SOURCE_MODES)),
            nullable=False,
        ),
        "ingested_at": pa.Column(None, nullable=False),
        "source_sha256": pa.Column(str, nullable=False),
    },
    strict=True,
    unique=["doc_id", "version"],
)

BRONZE_CROSSWALKS_SCHEMA = pa.DataFrameSchema(
    columns={
        "doc_id": pa.Column(str, nullable=False),
        "version": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
        "code_system": pa.Column(
            str,
            checks=pa.Check.isin(sorted(VALID_CODE_SYSTEMS)),
            nullable=False,
        ),
        "code": pa.Column(str, nullable=False),
    },
    strict=True,
)

SILVER_POLICIES_SCHEMA = pa.DataFrameSchema(
    columns={
        "doc_id": pa.Column(
            str,
            checks=pa.Check(_all_non_empty_strings, name="non_empty_doc_id"),
            nullable=False,
        ),
        "doc_type": pa.Column(
            str,
            checks=pa.Check.isin(sorted(VALID_DOC_TYPES)),
            nullable=False,
        ),
        "version": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
        "title": pa.Column(
            str,
            checks=pa.Check(_all_non_empty_strings, name="non_empty_title"),
            nullable=False,
        ),
        "markdown_text": pa.Column(
            str,
            checks=pa.Check(_all_non_empty_strings, name="non_empty_markdown"),
            nullable=False,
        ),
        "hcpcs_codes": pa.Column(
            object,
            checks=pa.Check(_all_string_lists, name="hcpcs_code_list"),
            nullable=False,
        ),
        "icd10_codes": pa.Column(
            object,
            checks=pa.Check(_all_string_lists, name="icd10_code_list"),
            nullable=False,
        ),
        "mac_jurisdiction": pa.Column(None, nullable=True),
        "effective_date": pa.Column(None, nullable=False),
        "retired_date": pa.Column(None, nullable=True),
        "source_sha256": pa.Column(str, nullable=False),
    },
    checks=pa.Check(_retired_not_before_effective, name="retired_gte_effective"),
    strict=True,
    unique=["doc_id", "version"],
)

SILVER_QUARANTINE_SCHEMA = pa.DataFrameSchema(
    columns={
        "doc_id": pa.Column(None, nullable=True),
        "doc_type": pa.Column(None, nullable=True),
        "version": pa.Column(None, nullable=True),
        "title": pa.Column(None, nullable=True),
        "effective_date": pa.Column(None, nullable=True),
        "retired_date": pa.Column(None, nullable=True),
        "status": pa.Column(None, nullable=True),
        "source_mode": pa.Column(None, nullable=True),
        "source_sha256": pa.Column(None, nullable=True),
        "failure_reason": pa.Column(
            None,
            checks=pa.Check(_all_non_empty_strings, name="non_empty_failure_reason"),
            nullable=False,
        ),
        "quarantined_at": pa.Column(None, nullable=False),
    },
    strict=True,
)


def build_gold_chunks_schema(embedding_dim: int = 384) -> pa.DataFrameSchema:
    """Build the Pandera contract for ``gold/chunks.parquet``.

    Inputs: expected vector dimension (default 384 for ``BAAI/bge-small-en-v1.5``).
    Output grain: Pandera ``DataFrameSchema`` enforcing unique ``(doc_id, version, chunk_index)``.
    Invariants: non-empty chunk text, positive token count, valid date interval, fixed-dim vectors.
    """
    return pa.DataFrameSchema(
        columns={
            "chunk_id": pa.Column(
                str,
                checks=pa.Check(_all_non_empty_strings, name="non_empty_chunk_id"),
                nullable=False,
                unique=True,
            ),
            "doc_id": pa.Column(str, nullable=False),
            "doc_type": pa.Column(
                str,
                checks=pa.Check.isin(sorted(VALID_DOC_TYPES)),
                nullable=False,
            ),
            "version": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
            "chunk_index": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
            "title": pa.Column(str, nullable=False),
            "section_heading": pa.Column(str, nullable=False),
            "chunk_text": pa.Column(
                str,
                checks=pa.Check(_all_non_empty_strings, name="non_empty_chunk_text"),
                nullable=False,
            ),
            "token_count": pa.Column(int, checks=pa.Check.gt(0), nullable=False),
            "chunk_sha256": pa.Column(str, nullable=False),
            "hcpcs_codes": pa.Column(
                object,
                checks=pa.Check(_all_string_lists, name="hcpcs_code_list"),
                nullable=False,
            ),
            "icd10_codes": pa.Column(
                object,
                checks=pa.Check(_all_string_lists, name="icd10_code_list"),
                nullable=False,
            ),
            "mac_jurisdiction": pa.Column(None, nullable=True),
            "effective_date": pa.Column(None, nullable=False),
            "retired_date": pa.Column(None, nullable=True),
            "embedding": pa.Column(
                object,
                checks=pa.Check(
                    lambda series: _valid_embedding_vectors(series, embedding_dim),
                    name=f"embedding_dim_{embedding_dim}",
                ),
                nullable=False,
            ),
            "embedding_model": pa.Column(
                str,
                checks=pa.Check(_all_non_empty_strings, name="non_empty_model"),
                nullable=False,
            ),
        },
        checks=pa.Check(_retired_not_before_effective, name="retired_gte_effective"),
        strict=True,
        unique=["doc_id", "version", "chunk_index"],
    )
