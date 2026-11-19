"""Canonical Bronze layer ingestion from CMS bulk ZIP archives and fixture directories."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from medcover.config import Settings
from medcover.contracts import (
    SourceMode,
    validate_bronze_crosswalks,
    validate_bronze_documents,
)

# CMS HTML policy cells can exceed the default 128 KB CSV field limit.
csv.field_size_limit(min(sys.maxsize, 100_000_000))

BRONZE_DOCUMENT_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "doc_type",
    "version",
    "title",
    "raw_html",
    "mac_jurisdiction",
    "effective_date",
    "retired_date",
    "status",
    "source_mode",
    "ingested_at",
    "source_sha256",
)

BRONZE_CROSSWALK_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "version",
    "code_system",
    "code",
)


@dataclass(frozen=True)
class BronzeResult:
    """Summary of a canonical Bronze write."""

    documents_path: Path
    crosswalks_path: Path
    total_documents: int
    total_crosswalks: int


def ingest_zip_or_dir(
    input_path: Path,
    bronze_dir: Path,
    source_mode: SourceMode = "zip",
) -> BronzeResult:
    """Ingest CMS bulk ZIP archives (including nested ``_CSV.ZIP``) or CSV directories into Bronze.

    Inputs: path to a ``.zip`` archive or directory containing CMS CSV/ZIP files, plus Bronze dir.
    Output grain: ``raw_documents.parquet`` unique on ``(doc_id, version)`` and
    ``raw_code_crosswalks.parquet`` code-link rows.
    Invariants: preserves historical versions via upsert on ``(doc_id, version)`` and validates
    output tables against Bronze Pandera schemas.
    """
    ingested_at = datetime.now(tz=UTC)
    documents: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []

    for csv_name, raw_bytes in _iter_csv_payloads(input_path):
        rows = _parse_csv_bytes(raw_bytes)
        if not rows:
            continue
        headers = {key.strip().lower() for key in rows[0] if key is not None}
        if {"doc_id", "version", "code_system", "code"}.issubset(headers):
            crosswalks.extend(_normalize_crosswalk_rows(rows))
        elif "doc_id" in headers and ("raw_html" in headers or "title" in headers):
            documents.extend(
                _normalize_document_rows(
                    rows,
                    source_mode=source_mode,
                    ingested_at=ingested_at,
                    csv_name=csv_name,
                )
            )

    return write_bronze_tables(
        documents=documents,
        crosswalks=crosswalks,
        bronze_dir=bronze_dir,
    )


def write_bronze_tables(
    documents: Iterable[Mapping[str, Any]],
    crosswalks: Iterable[Mapping[str, Any]],
    bronze_dir: Path,
) -> BronzeResult:
    """Upsert normalized Bronze records into canonical Parquet files.

    Inputs: iterable of Bronze document dicts and code crosswalk dicts, plus output directory.
    Output grain: strictly 1 row per ``(doc_id, version)`` in ``raw_documents.parquet``.
    Invariants: historical ``(doc_id, version)`` rows are retained; duplicate incoming keys
    update only matching ``(doc_id, version)`` tuples.
    """
    bronze_dir.mkdir(parents=True, exist_ok=True)
    docs_path = bronze_dir / "raw_documents.parquet"
    codes_path = bronze_dir / "raw_code_crosswalks.parquet"

    incoming_docs = _frame_with_columns(list(documents), BRONZE_DOCUMENT_COLUMNS)
    if docs_path.exists():
        existing_docs = pd.read_parquet(docs_path)
        merged_docs = _upsert_documents(existing_docs, incoming_docs)
    else:
        merged_docs = _deduplicate_documents(incoming_docs)

    merged_docs = _coerce_bronze_document_types(merged_docs)
    validate_bronze_documents(merged_docs)
    merged_docs.to_parquet(docs_path, index=False)

    incoming_codes = _frame_with_columns(list(crosswalks), BRONZE_CROSSWALK_COLUMNS)
    if codes_path.exists():
        existing_codes = pd.read_parquet(codes_path)
        merged_codes = pd.concat([existing_codes, incoming_codes], ignore_index=True)
    else:
        merged_codes = incoming_codes

    merged_codes = _coerce_bronze_crosswalk_types(merged_codes)
    validate_bronze_crosswalks(merged_codes)
    merged_codes.to_parquet(codes_path, index=False)

    return BronzeResult(
        documents_path=docs_path,
        crosswalks_path=codes_path,
        total_documents=len(merged_docs),
        total_crosswalks=len(merged_codes),
    )


def ingest_sample_fixtures(settings: Settings) -> BronzeResult:
    """Ingest the offline sample fixtures into Bronze Parquet.

    Inputs: validated ``Settings`` pointing to ``sample_fixtures_dir`` and ``bronze_dir``.
    Output grain: canonical Bronze Parquet tables for the sample fixture corpus.
    Invariants: deterministic offline execution with zero network calls.
    """
    return ingest_zip_or_dir(
        input_path=settings.sample_fixtures_dir,
        bronze_dir=settings.bronze_dir,
        source_mode="zip",
    )


def _iter_csv_payloads(input_path: Path) -> Iterator[tuple[str, bytes]]:
    if not input_path.exists():
        msg = f"Bronze input path does not exist: {input_path}"
        raise FileNotFoundError(msg)

    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".zip":
            yield from _iter_zip_csv_bytes(input_path.read_bytes(), source_label=input_path.name)
            return
        if suffix == ".csv":
            yield input_path.name, input_path.read_bytes()
            return
        msg = f"Unsupported Bronze input file format: {input_path}"
        raise ValueError(msg)

    for child in sorted(input_path.rglob("*")):
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix == ".csv":
            yield child.name, child.read_bytes()
        elif suffix == ".zip":
            yield from _iter_zip_csv_bytes(child.read_bytes(), source_label=child.name)


def _iter_zip_csv_bytes(zip_bytes: bytes, source_label: str) -> Iterator[tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in sorted(archive.namelist()):
            lower = member.lower()
            if member.endswith("/"):
                continue
            if lower.endswith(".zip"):
                nested_bytes = archive.read(member)
                yield from _iter_zip_csv_bytes(
                    nested_bytes, source_label=f"{source_label}!{member}"
                )
            elif lower.endswith(".csv"):
                yield f"{source_label}!{member}", archive.read(member)


def _decode_csv_bytes(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1252")


def _parse_csv_bytes(raw_bytes: bytes) -> list[dict[str, str]]:
    text = _decode_csv_bytes(raw_bytes)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return [dict(row) for row in reader]


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = _clean_optional_str(value)
    if text is None:
        return None
    return date.fromisoformat(text[:10])


def _compute_source_sha256(record: Mapping[str, Any]) -> str:
    payload = {
        "doc_id": record["doc_id"],
        "doc_type": record["doc_type"],
        "version": record["version"],
        "title": record["title"],
        "raw_html": record["raw_html"],
        "mac_jurisdiction": record["mac_jurisdiction"],
        "effective_date": (
            record["effective_date"].isoformat()
            if isinstance(record["effective_date"], date)
            else record["effective_date"]
        ),
        "retired_date": (
            record["retired_date"].isoformat()
            if isinstance(record["retired_date"], date)
            else record["retired_date"]
        ),
        "status": record["status"],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _normalize_document_rows(
    rows: Iterable[Mapping[str, str]],
    source_mode: SourceMode,
    ingested_at: datetime,
    csv_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
        doc_id = _clean_optional_str(lowered.get("doc_id"))
        if doc_id is None:
            msg = f"Missing doc_id in Bronze source {csv_name}"
            raise ValueError(msg)
        raw_version = _clean_optional_str(lowered.get("version"))
        if raw_version is None:
            msg = f"Missing version for doc_id={doc_id} in {csv_name}"
            raise ValueError(msg)
        version = int(raw_version)

        doc_type = _clean_optional_str(lowered.get("doc_type"))
        status = _clean_optional_str(lowered.get("status"))
        record: dict[str, Any] = {
            "doc_id": doc_id,
            "doc_type": doc_type.upper() if doc_type else None,
            "version": version,
            "title": _clean_optional_str(lowered.get("title")),
            "raw_html": lowered.get("raw_html"),
            "mac_jurisdiction": _clean_optional_str(lowered.get("mac_jurisdiction")),
            "effective_date": _parse_optional_date(lowered.get("effective_date")),
            "retired_date": _parse_optional_date(lowered.get("retired_date")),
            "status": status.upper() if status else None,
            "source_mode": source_mode,
            "ingested_at": ingested_at,
        }
        record["source_sha256"] = _compute_source_sha256(record)
        normalized.append(record)
    return normalized


def _normalize_crosswalk_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
        doc_id = _clean_optional_str(lowered.get("doc_id"))
        raw_version = _clean_optional_str(lowered.get("version"))
        code_system = _clean_optional_str(lowered.get("code_system"))
        code = _clean_optional_str(lowered.get("code"))
        if not (doc_id and raw_version and code_system and code):
            continue
        normalized.append(
            {
                "doc_id": doc_id,
                "version": int(raw_version),
                "code_system": code_system.upper(),
                "code": code.upper(),
            }
        )
    return normalized


def _frame_with_columns(
    rows: Sequence[Mapping[str, Any]],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame([dict(row) for row in rows]).reindex(columns=list(columns))


def _deduplicate_documents(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=["doc_id", "version"], keep="last").reset_index(drop=True)


def _upsert_documents(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return _deduplicate_documents(incoming)
    if incoming.empty:
        return _deduplicate_documents(existing)
    combined = pd.concat([existing, incoming], ignore_index=True)
    return _deduplicate_documents(combined)


def _coerce_bronze_document_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        for col in ("doc_id", "source_mode", "source_sha256"):
            result[col] = pd.Series(dtype="string")
        result["version"] = pd.Series(dtype="int64")
        return result
    result["version"] = result["version"].astype("int64")
    result["effective_date"] = result["effective_date"].map(_parse_optional_date)
    result["retired_date"] = result["retired_date"].map(_parse_optional_date)
    result = result.sort_values(["doc_id", "version"]).reset_index(drop=True)
    return result


def _coerce_bronze_crosswalk_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        for col in ("doc_id", "code_system", "code"):
            result[col] = pd.Series(dtype="string")
        result["version"] = pd.Series(dtype="int64")
        return result
    result["version"] = result["version"].astype("int64")
    result = result.sort_values(["doc_id", "version", "code_system", "code"]).reset_index(drop=True)
    return result
