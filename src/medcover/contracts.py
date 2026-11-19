"""Pydantic domain models, Pandera DataFrame contracts, and RunManifest schema."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, Field

DocType = Literal["NCD", "LCD", "ARTICLE"]
PolicyStatus = Literal["ACTIVE", "RETIRED"]
SourceMode = Literal["zip", "api"]
CodeSystem = Literal["CPT", "HCPCS", "ICD10"]

VALID_DOC_TYPES: frozenset[str] = frozenset({"NCD", "LCD", "ARTICLE"})
VALID_STATUSES: frozenset[str] = frozenset({"ACTIVE", "RETIRED"})
VALID_SOURCE_MODES: frozenset[str] = frozenset({"zip", "api"})
VALID_CODE_SYSTEMS: frozenset[str] = frozenset({"CPT", "HCPCS", "ICD10"})


class BronzeDocument(BaseModel):
    """Canonical Bronze document record prior to Silver HTML cleaning and quarantine checks."""

    doc_id: str
    doc_type: str
    version: int = Field(ge=1)
    title: str | None = None
    raw_html: str | None = None
    mac_jurisdiction: str | None = None
    effective_date: date | None = None
    retired_date: date | None = None
    status: str | None = None
    source_mode: SourceMode
    ingested_at: datetime
    source_sha256: str = Field(min_length=64, max_length=64)


class BronzeCodeCrosswalk(BaseModel):
    """Single billing or diagnosis code link associated with a Bronze policy version."""

    doc_id: str
    version: int = Field(ge=1)
    code_system: CodeSystem
    code: str = Field(min_length=1)


class RunManifest(BaseModel):
    """Audit and lineage summary emitted alongside each Gold build and WAP release."""

    run_id: str
    git_sha: str
    timestamp_utc: datetime
    source_mode: SourceMode
    bronze_docs: int = Field(ge=0)
    silver_docs: int = Field(ge=0)
    quarantined_docs: int = Field(ge=0)
    quarantine_rate: float = Field(ge=0.0, le=1.0)
    gold_chunks: int = Field(ge=0)
    chunks_reused: int = Field(ge=0)
    chunks_embedded: int = Field(ge=0)
    token_p50: float = Field(ge=0.0)
    token_p95: float = Field(ge=0.0)
    eval_recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    eval_mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    active_filter_violations: int | None = Field(default=None, ge=0)


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
        "cpt_codes": pa.Column(
            object,
            checks=pa.Check(_all_string_lists, name="cpt_code_list"),
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
        "status": pa.Column(
            str,
            checks=pa.Check.isin(sorted(VALID_STATUSES)),
            nullable=False,
        ),
        "is_active": pa.Column(bool, nullable=False),
        "source_mode": pa.Column(
            str,
            checks=pa.Check.isin(sorted(VALID_SOURCE_MODES)),
            nullable=False,
        ),
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
    Invariants: non-empty chunk text, positive token count, fixed-dim non-null embeddings.
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
            "cpt_codes": pa.Column(
                object,
                checks=pa.Check(_all_string_lists, name="cpt_code_list"),
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
            "is_active": pa.Column(bool, nullable=False),
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
        strict=True,
        unique=["doc_id", "version", "chunk_index"],
    )


def validate_bronze_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Bronze raw documents DataFrame against ``BRONZE_DOCUMENTS_SCHEMA``.

    Inputs: Bronze documents DataFrame.
    Output grain: validated DataFrame unique on ``(doc_id, version)``.
    Invariants: raises ``pandera.errors.SchemaError`` on schema or uniqueness violation.
    """
    return BRONZE_DOCUMENTS_SCHEMA.validate(df)


def validate_bronze_crosswalks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Bronze code crosswalks DataFrame against ``BRONZE_CROSSWALKS_SCHEMA``.

    Inputs: Bronze code crosswalks DataFrame.
    Output grain: validated code link rows.
    Invariants: ``code_system`` is strictly in ``{CPT, HCPCS, ICD10}``.
    """
    return BRONZE_CROSSWALKS_SCHEMA.validate(df)


def validate_silver_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Silver policies DataFrame against ``SILVER_POLICIES_SCHEMA``.

    Inputs: Silver policies DataFrame.
    Output grain: validated DataFrame unique on ``(doc_id, version)``.
    Invariants: non-empty Markdown, list-typed code arrays, ``retired_date >= effective_date``.
    """
    return SILVER_POLICIES_SCHEMA.validate(df)


def validate_silver_quarantine(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Silver quarantine DataFrame against ``SILVER_QUARANTINE_SCHEMA``.

    Inputs: Silver quarantine DataFrame.
    Output grain: validated dead-letter records with non-empty ``failure_reason``.
    Invariants: every quarantined row records why it was rejected.
    """
    return SILVER_QUARANTINE_SCHEMA.validate(df)


def validate_gold_chunks(df: pd.DataFrame, embedding_dim: int = 384) -> pd.DataFrame:
    """Validate Gold chunks DataFrame against the Gold Pandera contract.

    Inputs: Gold chunks DataFrame and expected embedding dimension.
    Output grain: validated DataFrame unique on ``(doc_id, version, chunk_index)``.
    Invariants: non-empty chunk text, unique ``chunk_id``, non-null ``FLOAT[dim]`` vectors.
    """
    schema = build_gold_chunks_schema(embedding_dim=embedding_dim)
    return schema.validate(df)
