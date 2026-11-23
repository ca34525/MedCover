"""Canonical Bronze layer ingestion from CMS bulk ZIP archives and fixture directories."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from medcover.config import Settings
from medcover.contracts import (
    BRONZE_CROSSWALKS_ARROW_SCHEMA,
    BRONZE_CROSSWALKS_SCHEMA,
    BRONZE_DOCUMENTS_ARROW_SCHEMA,
    BRONZE_DOCUMENTS_SCHEMA,
    SourceMode,
    parse_optional_date,
    parse_optional_int,
    parse_optional_str,
)

# CMS HTML policy cells can exceed the default 128 KB CSV field limit.
csv.field_size_limit(min(sys.maxsize, 100_000_000))

_CMS_STATUS_MAP: dict[str, str] = {
    "A": "ACTIVE",
    "ACTIVE": "ACTIVE",
    "CURRENT": "ACTIVE",
    "R": "RETIRED",
    "RETIRED": "RETIRED",
}

_CODE_SYSTEM_MAP: dict[str, str] = {
    "CPT": "HCPCS",
    "HCPCS": "HCPCS",
    "ICD10": "ICD10",
}

_EFFECTIVE_DATE_KEYS: tuple[str, ...] = (
    "effective_date",
    "rev_eff_date",
    "orig_det_eff_date",
    "orig_eff_date",
    "ncd_efctv_dt",
)

_RETIRED_DATE_KEYS: tuple[str, ...] = (
    "retired_date",
    "ret_date",
    "ent_det_end_date",
    "rev_end_date",
    "ncd_trmntn_dt",
)


@dataclass(frozen=True)
class BronzeResult:
    """Summary of a canonical Bronze write."""

    documents_path: Path
    crosswalks_path: Path
    total_documents: int
    total_crosswalks: int


@dataclass(frozen=True)
class _CsvTableSpec:
    kind: Literal["doc", "crosswalk"]
    required_headers: frozenset[str]
    marker_headers: frozenset[str]
    id_columns: tuple[str, ...]
    version_column: str
    id_prefix: str = ""
    allowed_prefixes: tuple[str, ...] = ()
    doc_type: str | None = None
    html_columns: tuple[str, ...] = ()
    is_cms_native: bool = False


_CSV_TABLE_SPECS: tuple[_CsvTableSpec, ...] = (
    _CsvTableSpec(
        kind="crosswalk",
        required_headers=frozenset({"doc_id", "version", "code_system", "code"}),
        marker_headers=frozenset(),
        id_columns=("doc_id",),
        version_column="version",
    ),
    _CsvTableSpec(
        kind="crosswalk",
        required_headers=frozenset({"lcd_id", "lcd_version"}),
        marker_headers=frozenset({"hcpc_code_id", "icd10_code_id"}),
        id_columns=("lcd_id",),
        version_column="lcd_version",
        id_prefix="L",
        allowed_prefixes=("L", "DL"),
        doc_type="LCD",
        is_cms_native=True,
    ),
    _CsvTableSpec(
        kind="crosswalk",
        required_headers=frozenset({"article_id", "article_version"}),
        marker_headers=frozenset({"hcpc_code_id", "icd10_code_id"}),
        id_columns=("article_id",),
        version_column="article_version",
        id_prefix="A",
        allowed_prefixes=("A", "DA"),
        doc_type="ARTICLE",
        is_cms_native=True,
    ),
    _CsvTableSpec(
        kind="doc",
        required_headers=frozenset({"doc_id"}),
        marker_headers=frozenset({"raw_html", "title"}),
        id_columns=("doc_id",),
        version_column="version",
    ),
    _CsvTableSpec(
        kind="doc",
        required_headers=frozenset({"lcd_id", "lcd_version"}),
        marker_headers=frozenset({"title", "indication", "raw_html"}),
        id_columns=("lcd_id",),
        version_column="lcd_version",
        id_prefix="L",
        allowed_prefixes=("L", "DL"),
        doc_type="LCD",
        html_columns=(
            "indication",
            "coding_guidelines",
            "doc_reqs",
            "summary_of_evidence",
            "analysis_of_evidence",
            "cms_cov_policy",
            "util_guide",
            "appendices",
        ),
        is_cms_native=True,
    ),
    _CsvTableSpec(
        kind="doc",
        required_headers=frozenset({"article_id", "article_version"}),
        marker_headers=frozenset({"title", "description", "raw_html"}),
        id_columns=("article_id",),
        version_column="article_version",
        id_prefix="A",
        allowed_prefixes=("A", "DA"),
        doc_type="ARTICLE",
        html_columns=("description", "coding_guidelines", "other_comments"),
        is_cms_native=True,
    ),
    _CsvTableSpec(
        kind="doc",
        required_headers=frozenset({"ncd_id", "ncd_vrsn_num"}),
        marker_headers=frozenset(
            {"ncd_mnl_sect_title", "title", "indctn_lmtn", "itm_srvc_desc", "raw_html"}
        ),
        id_columns=("ncd_mnl_sect", "ncd_id"),
        version_column="ncd_vrsn_num",
        id_prefix="NCD",
        allowed_prefixes=("NCD",),
        doc_type="NCD",
        html_columns=("itm_srvc_desc", "indctn_lmtn"),
        is_cms_native=True,
    ),
)


def ingest_zip_or_dir(
    input_path: Path,
    bronze_dir: Path,
    source_mode: SourceMode = "zip",
) -> BronzeResult:
    """Ingest CMS bulk ZIP archives (including nested ``_CSV.ZIP``) or CSV directories into Bronze.

    Inputs: path to a ``.zip`` archive or directory containing CMS CSV/ZIP files, plus Bronze dir.
    Output grain: ``raw_documents.parquet`` unique on ``(doc_id, version)`` and
    ``raw_code_crosswalks.parquet`` code-link rows.
    Invariants: preserves historical versions via upsert on ``(doc_id, version)``, scopes
    crosswalk replacement to the document families whose crosswalk tables were ingested,
    and validates output tables against Bronze Pandera contracts.
    """
    ingested_at = datetime.now(tz=UTC)
    documents: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []
    recognized_tables = 0
    replace_all_doc_crosswalks = False
    refreshed_doc_types: set[str] = set()

    for csv_name, raw_bytes in _iter_csv_payloads(input_path):
        headers, rows = _parse_csv_bytes(raw_bytes)
        if not headers:
            continue
        spec = _match_table_spec(headers)
        if spec is None:
            continue
        recognized_tables += 1
        if spec.kind == "crosswalk":
            if spec.doc_type is None:
                replace_all_doc_crosswalks = True
            else:
                refreshed_doc_types.add(spec.doc_type)
            crosswalks.extend(_normalize_crosswalk_rows(rows, spec=spec))
        else:
            documents.extend(
                _normalize_document_rows(
                    rows,
                    spec=spec,
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
        replace_doc_crosswalks=replace_all_doc_crosswalks,
        refreshed_doc_types=refreshed_doc_types,
    )


def write_bronze_tables(
    documents: Iterable[Mapping[str, Any]],
    crosswalks: Iterable[Mapping[str, Any]],
    bronze_dir: Path,
    *,
    replace_doc_crosswalks: bool = False,
    refreshed_doc_types: Collection[str] | None = None,
) -> BronzeResult:
    """Upsert normalized Bronze records into canonical typed Parquet files.

    Inputs: iterables of Bronze document dicts and code crosswalk dicts, plus output directory.
    Output grain: strictly 1 row per ``(doc_id, version)`` in ``raw_documents.parquet``.
    Invariants: historical ``(doc_id, version)`` rows are retained; refreshed versions replace
    prior crosswalk links strictly for incoming crosswalk keys and matching ``refreshed_doc_types``.
    """
    bronze_dir.mkdir(parents=True, exist_ok=True)
    docs_path = bronze_dir / "raw_documents.parquet"
    codes_path = bronze_dir / "raw_code_crosswalks.parquet"

    incoming_docs = [_coerce_document_record(row) for row in documents]
    docs_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if docs_path.exists():
        for existing in pq.read_table(docs_path).to_pylist():
            docs_by_key[(str(existing["doc_id"]), int(existing["version"]))] = existing
    for doc in incoming_docs:
        docs_by_key[(str(doc["doc_id"]), int(doc["version"]))] = doc

    merged_docs = [docs_by_key[key] for key in sorted(docs_by_key)]
    docs_table = pa.Table.from_pylist(merged_docs, schema=BRONZE_DOCUMENTS_ARROW_SCHEMA)
    BRONZE_DOCUMENTS_SCHEMA.validate(docs_table.to_pandas())
    pq.write_table(docs_table, docs_path)

    incoming_codes = [_coerce_crosswalk_record(row) for row in crosswalks]
    refreshed_types = set(refreshed_doc_types) if refreshed_doc_types else set()
    keys_to_replace: set[tuple[str, int]] = {
        (str(row["doc_id"]), int(row["version"])) for row in incoming_codes
    }
    for doc in incoming_docs:
        if replace_doc_crosswalks or (doc["doc_type"] in refreshed_types):
            keys_to_replace.add((str(doc["doc_id"]), int(doc["version"])))

    retained_codes: list[dict[str, Any]] = []
    if codes_path.exists():
        for existing_code in pq.read_table(codes_path).to_pylist():
            key = (str(existing_code["doc_id"]), int(existing_code["version"]))
            if key not in keys_to_replace:
                retained_codes.append(existing_code)

    merged_codes = sorted(
        [*retained_codes, *incoming_codes],
        key=lambda row: (
            str(row["doc_id"]),
            int(row["version"]),
            str(row["code_system"]),
            str(row["code"]),
        ),
    )
    codes_table = pa.Table.from_pylist(merged_codes, schema=BRONZE_CROSSWALKS_ARROW_SCHEMA)
    BRONZE_CROSSWALKS_SCHEMA.validate(codes_table.to_pandas())
    pq.write_table(codes_table, codes_path)

    return BronzeResult(
        documents_path=docs_path,
        crosswalks_path=codes_path,
        total_documents=docs_table.num_rows,
        total_crosswalks=codes_table.num_rows,
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
            yield from _iter_zip_csv_bytes(input_path, source_label=input_path.name)
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
            yield from _iter_zip_csv_bytes(child, source_label=child.name)


def _iter_zip_csv_bytes(
    zip_source: Path | io.BytesIO,
    *,
    source_label: str,
) -> Iterator[tuple[str, bytes]]:
    with zipfile.ZipFile(zip_source) as archive:
        for member in sorted(archive.namelist()):
            lower = member.lower()
            if member.endswith("/"):
                continue
            if lower.endswith(".zip"):
                yield from _iter_zip_csv_bytes(
                    io.BytesIO(archive.read(member)),
                    source_label=f"{source_label}!{member}",
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
    return headers, list(reader)


def _match_table_spec(headers: set[str]) -> _CsvTableSpec | None:
    for spec in _CSV_TABLE_SPECS:
        if spec.required_headers.issubset(headers) and (
            not spec.marker_headers or bool(headers & spec.marker_headers)
        ):
            return spec
    return None


def _first_present(row: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = parse_optional_str(row.get(key))
        if value is not None:
            return value
    return None


def _extract_identity(
    lowered: Mapping[str, str],
    spec: _CsvTableSpec,
) -> tuple[str | None, int | None, str | None]:
    raw_id = _first_present(lowered, spec.id_columns)
    if raw_id is None:
        doc_id: str | None = None
    elif spec.id_prefix:
        upper = raw_id.upper()
        doc_id = upper if upper.startswith(spec.allowed_prefixes) else f"{spec.id_prefix}{upper}"
    else:
        doc_id = raw_id

    version = parse_optional_int(lowered.get(spec.version_column))
    if spec.doc_type is not None:
        doc_type: str | None = spec.doc_type
    else:
        raw_type = parse_optional_str(lowered.get("doc_type"))
        doc_type = raw_type.upper() if raw_type else None
    return doc_id, version, doc_type


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
    *,
    spec: _CsvTableSpec,
    source_mode: SourceMode,
    ingested_at: datetime,
    csv_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
        doc_id, version, doc_type = _extract_identity(lowered, spec)
        if doc_id is None:
            msg = f"Missing doc_id in Bronze source {csv_name}"
            raise ValueError(msg)
        if version is None:
            msg = f"Missing or invalid version for doc_id={doc_id} in {csv_name}"
            raise ValueError(msg)

        if "raw_html" in lowered:
            raw_html = parse_optional_str(lowered.get("raw_html"))
        else:
            parts = [
                part
                for col in spec.html_columns
                if (part := parse_optional_str(lowered.get(col))) is not None
            ]
            raw_html = "\n".join(parts) if parts else None

        effective_date = parse_optional_date(_first_present(lowered, _EFFECTIVE_DATE_KEYS))
        retired_date = parse_optional_date(_first_present(lowered, _RETIRED_DATE_KEYS))

        raw_status = parse_optional_str(lowered.get("status"))
        status: str | None = (
            _CMS_STATUS_MAP.get(raw_status.upper(), raw_status.upper())
            if raw_status is not None
            else None
        )

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
    spec: _CsvTableSpec,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
        doc_id, version, _ = _extract_identity(lowered, spec)

        if not spec.is_cms_native:
            raw_system = parse_optional_str(lowered.get("code_system"))
            code_system = (
                _CODE_SYSTEM_MAP.get(raw_system.upper(), raw_system.upper())
                if raw_system is not None
                else None
            )
            code = parse_optional_str(lowered.get("code"))
        elif (code := parse_optional_str(lowered.get("hcpc_code_id"))) is not None:
            code_system = "HCPCS"
        elif (code := parse_optional_str(lowered.get("icd10_code_id"))) is not None:
            code_system = "ICD10"
        else:
            code_system = None
            code = None

        if not (doc_id and version is not None and code_system and code):
            continue
        normalized.append(
            {
                "doc_id": doc_id,
                "version": version,
                "code_system": code_system.upper(),
                "code": code.upper(),
            }
        )
    return normalized


def _coerce_document_record(row: Mapping[str, Any]) -> dict[str, Any]:
    doc_id = parse_optional_str(row.get("doc_id"))
    version = parse_optional_int(row.get("version"))
    if doc_id is None or version is None:
        msg = f"Bronze document requires non-null doc_id and integer version: {row!r}"
        raise ValueError(msg)

    raw_ingested = row.get("ingested_at")
    if isinstance(raw_ingested, datetime):
        ingested_at = (
            raw_ingested if raw_ingested.tzinfo is not None else raw_ingested.replace(tzinfo=UTC)
        )
    else:
        ingested_at = datetime.now(tz=UTC)

    record: dict[str, Any] = {
        "doc_id": doc_id,
        "doc_type": parse_optional_str(row.get("doc_type")),
        "version": version,
        "title": parse_optional_str(row.get("title")),
        "raw_html": parse_optional_str(row.get("raw_html")),
        "mac_jurisdiction": parse_optional_str(row.get("mac_jurisdiction")),
        "effective_date": parse_optional_date(row.get("effective_date")),
        "retired_date": parse_optional_date(row.get("retired_date")),
        "status": parse_optional_str(row.get("status")),
        "source_mode": parse_optional_str(row.get("source_mode")) or "zip",
        "ingested_at": ingested_at,
    }
    record["source_sha256"] = parse_optional_str(row.get("source_sha256")) or (
        _compute_source_sha256(record)
    )
    return record


def _coerce_crosswalk_record(row: Mapping[str, Any]) -> dict[str, Any]:
    doc_id = parse_optional_str(row.get("doc_id"))
    version = parse_optional_int(row.get("version"))
    code_system = parse_optional_str(row.get("code_system"))
    code = parse_optional_str(row.get("code"))
    if not (doc_id and version is not None and code_system and code):
        msg = f"Bronze crosswalk requires doc_id, version, code_system, and code: {row!r}"
        raise ValueError(msg)
    normalized_system = _CODE_SYSTEM_MAP.get(code_system.upper(), code_system.upper())
    return {
        "doc_id": doc_id,
        "version": version,
        "code_system": normalized_system,
        "code": code.upper(),
    }
