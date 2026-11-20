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
    recognized_tables = 0
    saw_crosswalk_table = False

    for csv_name, raw_bytes in _iter_csv_payloads(input_path):
        headers, rows = _parse_csv_bytes(raw_bytes)
        if not headers:
            continue
        table_kind = _classify_csv_headers(headers)
        if table_kind is None:
            continue
        recognized_tables += 1
        if table_kind.endswith("_crosswalk"):
            saw_crosswalk_table = True
            crosswalks.extend(_normalize_crosswalk_rows(rows, table_kind=table_kind))
        else:
            documents.extend(
                _normalize_document_rows(
                    rows,
                    table_kind=table_kind,
                    source_mode=source_mode,
                    ingested_at=ingested_at,
                    csv_name=csv_name,
                )
            )

    if recognized_tables == 0:
        msg = f"No recognized CMS policy or crosswalk CSV tables found in {input_path}"
        raise ValueError(msg)

    return write_bronze_tables(
        documents=documents,
        crosswalks=crosswalks,
        bronze_dir=bronze_dir,
        replace_doc_crosswalks=saw_crosswalk_table,
    )


def write_bronze_tables(
    documents: Iterable[Mapping[str, Any]],
    crosswalks: Iterable[Mapping[str, Any]],
    bronze_dir: Path,
    *,
    replace_doc_crosswalks: bool = False,
) -> BronzeResult:
    """Upsert normalized Bronze records into canonical Parquet files.

    Inputs: iterable of Bronze document dicts and code crosswalk dicts, plus output directory.
    Output grain: strictly 1 row per ``(doc_id, version)`` in ``raw_documents.parquet``.
    Invariants: historical ``(doc_id, version)`` rows are retained; duplicate incoming keys
    update only matching ``(doc_id, version)`` tuples, and refreshed versions replace prior
    code crosswalk links rather than accumulating stale associations.
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
        refreshed_doc_keys = (
            set(zip(incoming_docs["doc_id"], incoming_docs["version"], strict=True))
            if (replace_doc_crosswalks or not incoming_codes.empty) and not incoming_docs.empty
            else set()
        )
        merged_codes = _upsert_crosswalks(
            existing_codes,
            incoming_codes,
            refreshed_doc_keys=refreshed_doc_keys,
        )
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


def _parse_csv_bytes(raw_bytes: bytes) -> tuple[set[str], list[dict[str, str]]]:
    text = _decode_csv_bytes(raw_bytes)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = {name.strip().lower() for name in (reader.fieldnames or []) if name is not None}
    return headers, [dict(row) for row in reader]


def _classify_csv_headers(headers: set[str]) -> str | None:
    if {"doc_id", "version", "code_system", "code"}.issubset(headers):
        return "canonical_crosswalk"
    if {"lcd_id", "lcd_version"}.issubset(headers) and (
        "hcpc_code_id" in headers or "icd10_code_id" in headers
    ):
        return "cms_lcd_crosswalk"
    if {"article_id", "article_version"}.issubset(headers) and (
        "hcpc_code_id" in headers or "icd10_code_id" in headers
    ):
        return "cms_article_crosswalk"
    if "doc_id" in headers and ("raw_html" in headers or "title" in headers):
        return "canonical_doc"
    if {"lcd_id", "lcd_version"}.issubset(headers) and bool(
        headers & {"title", "indication", "raw_html"}
    ):
        return "cms_lcd_doc"
    if {"article_id", "article_version"}.issubset(headers) and bool(
        headers & {"title", "description", "raw_html"}
    ):
        return "cms_article_doc"
    if {"ncd_id", "ncd_vrsn_num"}.issubset(headers) and bool(
        headers & {"ncd_mnl_sect_title", "title", "indctn_lmtn", "itm_srvc_desc", "raw_html"}
    ):
        return "cms_ncd_doc"
    return None


def _clean_optional_str(value: Any) -> str | None:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


_CMS_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
)


def _parse_optional_date(value: Any) -> date | None:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return None if bool(pd.isna(value)) else value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean_optional_str(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    for fmt in _CMS_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


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


_CMS_HTML_COLUMNS: dict[str, tuple[str, ...]] = {
    "cms_lcd_doc": (
        "indication",
        "coding_guidelines",
        "doc_reqs",
        "summary_of_evidence",
        "analysis_of_evidence",
        "cms_cov_policy",
        "util_guide",
        "appendices",
    ),
    "cms_article_doc": ("description", "coding_guidelines", "other_comments"),
    "cms_ncd_doc": ("itm_srvc_desc", "indctn_lmtn"),
}

_CMS_STATUS_MAP: dict[str, str] = {
    "A": "ACTIVE",
    "ACTIVE": "ACTIVE",
    "CURRENT": "ACTIVE",
    "R": "RETIRED",
    "RETIRED": "RETIRED",
}


def _first_present(row: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _clean_optional_str(row.get(key))
        if value is not None:
            return value
    return None


def _prefix_cms_id(raw_id: str, *, prefix: str, allowed_prefixes: tuple[str, ...]) -> str:
    upper = raw_id.upper()
    return upper if upper.startswith(allowed_prefixes) else f"{prefix}{upper}"


def _extract_doc_identity(
    lowered: Mapping[str, str], *, table_kind: str
) -> tuple[str | None, str | None, str | None]:
    if table_kind == "cms_lcd_doc":
        raw_id = _clean_optional_str(lowered.get("lcd_id"))
        doc_id = (
            _prefix_cms_id(raw_id, prefix="L", allowed_prefixes=("L", "DL")) if raw_id else None
        )
        return doc_id, _clean_optional_str(lowered.get("lcd_version")), "LCD"
    if table_kind == "cms_article_doc":
        raw_id = _clean_optional_str(lowered.get("article_id"))
        doc_id = (
            _prefix_cms_id(raw_id, prefix="A", allowed_prefixes=("A", "DA")) if raw_id else None
        )
        return doc_id, _clean_optional_str(lowered.get("article_version")), "ARTICLE"
    if table_kind == "cms_ncd_doc":
        raw_id = _first_present(lowered, ("ncd_mnl_sect", "ncd_id"))
        doc_id = _prefix_cms_id(raw_id, prefix="NCD", allowed_prefixes=("NCD",)) if raw_id else None
        return doc_id, _clean_optional_str(lowered.get("ncd_vrsn_num")), "NCD"
    doc_type = _clean_optional_str(lowered.get("doc_type"))
    return (
        _clean_optional_str(lowered.get("doc_id")),
        _clean_optional_str(lowered.get("version")),
        doc_type.upper() if doc_type else None,
    )


def _normalize_document_rows(
    rows: Iterable[Mapping[str, str]],
    table_kind: str,
    source_mode: SourceMode,
    ingested_at: datetime,
    csv_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
        doc_id, raw_version, doc_type = _extract_doc_identity(lowered, table_kind=table_kind)
        if doc_id is None:
            msg = f"Missing doc_id in Bronze source {csv_name}"
            raise ValueError(msg)
        if raw_version is None:
            msg = f"Missing version for doc_id={doc_id} in {csv_name}"
            raise ValueError(msg)
        version = int(raw_version)

        if "raw_html" in lowered:
            raw_html: str | None = lowered.get("raw_html")
        else:
            parts = [
                part
                for col in _CMS_HTML_COLUMNS.get(table_kind, ())
                if (part := _clean_optional_str(lowered.get(col))) is not None
            ]
            raw_html = "\n".join(parts) if parts else None

        effective_date = _parse_optional_date(
            _first_present(
                lowered,
                (
                    "effective_date",
                    "rev_eff_date",
                    "orig_det_eff_date",
                    "orig_eff_date",
                    "ncd_efctv_dt",
                ),
            )
        )
        retired_date = _parse_optional_date(
            _first_present(
                lowered,
                ("retired_date", "ret_date", "ent_det_end_date", "rev_end_date", "ncd_trmntn_dt"),
            )
        )

        raw_status = _clean_optional_str(lowered.get("status"))
        if raw_status is not None:
            status: str | None = _CMS_STATUS_MAP.get(raw_status.upper(), raw_status.upper())
        elif table_kind.startswith("cms_"):
            status = "RETIRED" if retired_date is not None else "ACTIVE"
        else:
            status = None

        record: dict[str, Any] = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "version": version,
            "title": _first_present(lowered, ("title", "ncd_mnl_sect_title")),
            "raw_html": raw_html,
            "mac_jurisdiction": _first_present(lowered, ("mac_jurisdiction", "contractor_name")),
            "effective_date": effective_date,
            "retired_date": retired_date,
            "status": status,
            "source_mode": source_mode,
            "ingested_at": ingested_at,
        }
        record["source_sha256"] = _compute_source_sha256(record)
        normalized.append(record)
    return normalized


def _normalize_crosswalk_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    table_kind: str = "canonical_crosswalk",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
        if table_kind == "cms_lcd_crosswalk":
            raw_id = _clean_optional_str(lowered.get("lcd_id"))
            doc_id = (
                _prefix_cms_id(raw_id, prefix="L", allowed_prefixes=("L", "DL")) if raw_id else None
            )
            raw_version = _clean_optional_str(lowered.get("lcd_version"))
        elif table_kind == "cms_article_crosswalk":
            raw_id = _clean_optional_str(lowered.get("article_id"))
            doc_id = (
                _prefix_cms_id(raw_id, prefix="A", allowed_prefixes=("A", "DA")) if raw_id else None
            )
            raw_version = _clean_optional_str(lowered.get("article_version"))
        else:
            doc_id = _clean_optional_str(lowered.get("doc_id"))
            raw_version = _clean_optional_str(lowered.get("version"))

        if table_kind == "canonical_crosswalk":
            code_system = _clean_optional_str(lowered.get("code_system"))
            code = _clean_optional_str(lowered.get("code"))
        elif (code := _clean_optional_str(lowered.get("hcpc_code_id"))) is not None:
            code_system = "CPT" if code.isdigit() and len(code) == 5 else "HCPCS"
        elif (code := _clean_optional_str(lowered.get("icd10_code_id"))) is not None:
            code_system = "ICD10"
        else:
            code_system = None
            code = None

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


def _upsert_crosswalks(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    refreshed_doc_keys: set[tuple[Any, Any]],
) -> pd.DataFrame:
    if existing.empty:
        return incoming
    keys_to_replace = set(refreshed_doc_keys)
    if not incoming.empty:
        keys_to_replace.update(zip(incoming["doc_id"], incoming["version"], strict=True))
    if not keys_to_replace:
        return existing
    keep_mask = [
        (doc_id, version) not in keys_to_replace
        for doc_id, version in zip(existing["doc_id"], existing["version"], strict=True)
    ]
    retained = existing.loc[keep_mask]
    if incoming.empty:
        return retained.reset_index(drop=True)
    if retained.empty:
        return incoming.reset_index(drop=True)
    return pd.concat([retained, incoming], ignore_index=True)


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
