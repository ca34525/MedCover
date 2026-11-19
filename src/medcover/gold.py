"""Gold layer section-aware chunking, incremental FastEmbed inference, and index.duckdb builder."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastembed import TextEmbedding

from medcover.config import Settings
from medcover.contracts import RunManifest, SourceMode, validate_gold_chunks

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class MarkdownChunk:
    """Intermediate text chunk produced before incremental embedding."""

    section_heading: str
    chunk_text: str
    token_count: int


@dataclass(frozen=True)
class GoldResult:
    """Summary of a Gold build and pre-built DuckDB FTS index."""

    chunks_path: Path
    index_path: Path
    manifest_path: Path
    manifest: RunManifest


def count_tokens(text: str) -> int:
    """Count deterministic lexical tokens for chunk budgeting and manifest telemetry.

    Inputs: UTF-8 text string.
    Output grain: non-negative integer token count.
    Invariants: deterministic offline execution with zero external tokenizer downloads.
    """
    return len(_TOKEN_RE.findall(text))


def chunk_markdown(
    markdown_text: str,
    *,
    default_heading: str,
    token_limit: int = 450,
    token_overlap: int = 60,
) -> list[MarkdownChunk]:
    """Split policy Markdown into heading-scoped, table-safe chunks within a token budget.

    Inputs: cleaned Markdown policy body, fallback heading, token budget, and overlap tokens.
    Output grain: ordered list of ``MarkdownChunk`` values covering the document.
    Invariants: splits on Markdown headings first, preserves complete Markdown table rows
    (repeating table headers if a large table spans chunks), and emits non-empty chunks.
    """
    sections = _split_markdown_sections(markdown_text, default_heading=default_heading)
    chunks: list[MarkdownChunk] = []

    for heading, body in sections:
        blocks = _split_blocks(body)
        if not blocks:
            continue
        section_texts = _pack_blocks(
            blocks,
            token_limit=token_limit,
            token_overlap=token_overlap,
        )
        for text in section_texts:
            cleaned = text.strip()
            if cleaned:
                chunks.append(
                    MarkdownChunk(
                        section_heading=heading,
                        chunk_text=cleaned,
                        token_count=max(1, count_tokens(cleaned)),
                    )
                )

    return chunks


def build_gold(
    settings: Settings,
    *,
    bronze_docs: int | None = None,
    quarantined_docs: int | None = None,
    source_mode: SourceMode = "zip",
    embed_fn: EmbedFn | None = None,
) -> GoldResult:
    """Chunk Silver policies, incrementally embed changed chunks, and build ``index.duckdb``.

    Inputs: ``Settings`` plus optional Bronze/quarantine counts and injectable ``embed_fn``.
    Output grain: ``gold/chunks.parquet`` unique on ``(doc_id, version, chunk_index)``,
    pre-indexed ``gold/index.duckdb``, and ``gold/run_manifest.json``.
    Invariants: unchanged ``chunk_sha256`` vectors are reused from prior Gold so repeat runs
    yield ``chunks_embedded == 0``; ``index.duckdb`` is built with FTS ready for read-only serving.
    """
    policies_path = settings.silver_dir / "policies.parquet"
    if not policies_path.exists():
        msg = f"Silver policies file not found: {policies_path}"
        raise FileNotFoundError(msg)

    policies_df = pd.read_parquet(policies_path)
    candidates = _build_candidate_chunks(policies_df, settings=settings)

    settings.gold_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.gold_dir / "chunks.parquet"
    index_path = settings.gold_dir / "index.duckdb"
    manifest_path = settings.gold_dir / "run_manifest.json"

    prior_vectors = _load_prior_embeddings(
        chunks_path,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    chunks_df, chunks_reused, chunks_embedded = _apply_incremental_embeddings(
        candidates,
        prior_vectors=prior_vectors,
        settings=settings,
        embed_fn=embed_fn,
    )
    validate_gold_chunks(chunks_df, embedding_dim=settings.embedding_dim)
    _write_gold_chunks_parquet(
        chunks_df,
        chunks_path,
        embedding_dim=settings.embedding_dim,
    )
    _build_duckdb_index(
        chunks_path=chunks_path,
        index_path=index_path,
        embedding_dim=settings.embedding_dim,
        extension_dir=settings.resolved_duckdb_extension_dir(),
    )

    silver_count = len(policies_df)
    quarantine_count = _resolve_quarantine_count(settings, quarantined_docs)
    bronze_count = (
        bronze_docs
        if bronze_docs is not None
        else _resolve_bronze_count(settings, fallback=silver_count + quarantine_count)
    )
    quarantine_rate = (quarantine_count / bronze_count) if bronze_count > 0 else 0.0

    token_counts = chunks_df["token_count"].to_numpy(dtype=float) if not chunks_df.empty else None
    token_p50 = float(np.percentile(token_counts, 50)) if token_counts is not None else 0.0
    token_p95 = float(np.percentile(token_counts, 95)) if token_counts is not None else 0.0

    now = datetime.now(tz=UTC)
    git_sha = _resolve_git_sha()
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{git_sha}"
    manifest = RunManifest(
        run_id=run_id,
        git_sha=git_sha,
        timestamp_utc=now,
        source_mode=source_mode,
        bronze_docs=bronze_count,
        silver_docs=silver_count,
        quarantined_docs=quarantine_count,
        quarantine_rate=round(quarantine_rate, 6),
        gold_chunks=len(chunks_df),
        chunks_reused=chunks_reused,
        chunks_embedded=chunks_embedded,
        token_p50=round(token_p50, 2),
        token_p95=round(token_p95, 2),
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    return GoldResult(
        chunks_path=chunks_path,
        index_path=index_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _split_markdown_sections(markdown_text: str, *, default_heading: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = default_heading
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    tail = "\n".join(current_lines).strip()
    if tail:
        sections.append((current_heading, tail))
    return sections


def _split_blocks(section_body: str) -> list[str]:
    blocks: list[str] = []
    lines = section_body.splitlines()
    idx = 0
    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        if lines[idx].lstrip().startswith("|"):
            table_lines: list[str] = []
            while idx < len(lines) and lines[idx].lstrip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            blocks.append("\n".join(table_lines))
            continue

        para_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() and not lines[idx].lstrip().startswith("|"):
            para_lines.append(lines[idx])
            idx += 1
        blocks.append("\n".join(para_lines))

    return blocks


def _is_markdown_table(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and lines[0].startswith("|") and lines[1].startswith("|")


def _split_large_table(table_block: str, *, token_limit: int) -> list[str]:
    lines = [line for line in table_block.splitlines() if line.strip()]
    if len(lines) <= 2 or count_tokens(table_block) <= token_limit:
        return [table_block]

    header = lines[:2]
    data_rows = lines[2:]
    pieces: list[str] = []
    current_rows: list[str] = list(header)

    for row in data_rows:
        candidate = "\n".join([*current_rows, row])
        if len(current_rows) > 2 and count_tokens(candidate) > token_limit:
            pieces.append("\n".join(current_rows))
            current_rows = [*header, row]
        else:
            current_rows.append(row)

    if len(current_rows) > 2:
        pieces.append("\n".join(current_rows))
    return pieces


def _split_large_prose(block: str, *, token_limit: int, token_overlap: int) -> list[str]:
    if count_tokens(block) <= token_limit:
        return [block]

    words = block.split()
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = start
        while end < len(words):
            candidate = " ".join(words[start : end + 1])
            if end > start and count_tokens(candidate) > token_limit:
                break
            end += 1
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        step_back = _overlap_word_count(words[start:end], token_overlap=token_overlap)
        next_start = max(start + 1, end - step_back)
        start = next_start
    return pieces


def _overlap_word_count(words: Sequence[str], *, token_overlap: int) -> int:
    if token_overlap <= 0 or not words:
        return 0
    tokens = 0
    count = 0
    for word in reversed(words):
        tokens += max(1, count_tokens(word))
        count += 1
        if tokens >= token_overlap:
            break
    return count


def _tail_overlap_text(text: str, *, token_overlap: int) -> str:
    if token_overlap <= 0 or _is_markdown_table(text):
        return ""
    words = text.split()
    keep = _overlap_word_count(words, token_overlap=token_overlap)
    return " ".join(words[-keep:]) if keep > 0 else ""


def _pack_blocks(
    blocks: Sequence[str],
    *,
    token_limit: int,
    token_overlap: int,
) -> list[str]:
    expanded: list[str] = []
    for block in blocks:
        if _is_markdown_table(block):
            expanded.extend(_split_large_table(block, token_limit=token_limit))
        else:
            expanded.extend(
                _split_large_prose(block, token_limit=token_limit, token_overlap=token_overlap)
            )

    packed: list[str] = []
    current = ""
    for piece in expanded:
        if not current:
            current = piece
            continue
        combined = f"{current}\n\n{piece}"
        if count_tokens(combined) <= token_limit:
            current = combined
            continue

        packed.append(current)
        overlap = _tail_overlap_text(current, token_overlap=token_overlap)
        if overlap and not _is_markdown_table(piece):
            with_overlap = f"{overlap}\n\n{piece}"
            current = with_overlap if count_tokens(with_overlap) <= token_limit else piece
        else:
            current = piece

    if current:
        packed.append(current)
    return packed


def _compute_chunk_sha256(
    *,
    title: str,
    section_heading: str,
    chunk_text: str,
    embedding_model: str,
) -> str:
    payload = json.dumps(
        {
            "title": title,
            "section_heading": section_heading,
            "chunk_text": chunk_text,
            "embedding_model": embedding_model,
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_candidate_chunks(
    policies_df: pd.DataFrame,
    *,
    settings: Settings,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    sorted_policies = policies_df.sort_values(["doc_id", "version"])

    for record in sorted_policies.to_dict(orient="records"):
        doc_id = str(record["doc_id"])
        version = int(record["version"])
        title = str(record["title"])
        markdown_text = str(record["markdown_text"])
        cpt_codes = [str(code) for code in list(record["cpt_codes"])]
        icd10_codes = [str(code) for code in list(record["icd10_codes"])]
        mac_raw = record.get("mac_jurisdiction")
        mac_jurisdiction = (
            None
            if mac_raw is None or (isinstance(mac_raw, float) and pd.isna(mac_raw))
            else str(mac_raw)
        )
        effective_date = _to_date(record["effective_date"])
        retired_date = _to_optional_date(record.get("retired_date"))
        is_active = bool(record["is_active"])

        doc_chunks = chunk_markdown(
            markdown_text,
            default_heading=title,
            token_limit=settings.chunk_token_limit,
            token_overlap=settings.chunk_token_overlap,
        )
        for chunk_index, chunk in enumerate(doc_chunks):
            chunk_id = f"{doc_id}_v{version}_c{chunk_index:04d}"
            chunk_sha256 = _compute_chunk_sha256(
                title=title,
                section_heading=chunk.section_heading,
                chunk_text=chunk.chunk_text,
                embedding_model=settings.embedding_model,
            )
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_type": str(record["doc_type"]),
                    "version": version,
                    "chunk_index": chunk_index,
                    "title": title,
                    "section_heading": chunk.section_heading,
                    "chunk_text": chunk.chunk_text,
                    "token_count": chunk.token_count,
                    "chunk_sha256": chunk_sha256,
                    "cpt_codes": cpt_codes,
                    "icd10_codes": icd10_codes,
                    "mac_jurisdiction": mac_jurisdiction,
                    "effective_date": effective_date,
                    "retired_date": retired_date,
                    "is_active": is_active,
                    "embedding_model": settings.embedding_model,
                }
            )

    return candidates


def _load_prior_embeddings(
    chunks_path: Path,
    *,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, list[float]]:
    if not chunks_path.exists():
        return {}
    prior_df = pd.read_parquet(
        chunks_path,
        columns=["chunk_sha256", "embedding", "embedding_model"],
    )
    matching = prior_df[prior_df["embedding_model"] == embedding_model]
    vectors: dict[str, list[float]] = {}
    for sha, vec in zip(matching["chunk_sha256"], matching["embedding"], strict=True):
        values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        if len(values) == embedding_dim:
            vectors[str(sha)] = [float(x) for x in values]
    return vectors


def _embed_with_fastembed(texts: Sequence[str], *, settings: Settings) -> list[list[float]]:
    cache_dir = settings.resolved_fastembed_cache_dir()
    model = TextEmbedding(
        model_name=settings.embedding_model,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return [[float(x) for x in vector.tolist()] for vector in model.embed(list(texts))]


def _apply_incremental_embeddings(
    candidates: list[dict[str, Any]],
    *,
    prior_vectors: dict[str, list[float]],
    settings: Settings,
    embed_fn: EmbedFn | None,
) -> tuple[pd.DataFrame, int, int]:
    if not candidates:
        columns = [
            "chunk_id",
            "doc_id",
            "doc_type",
            "version",
            "chunk_index",
            "title",
            "section_heading",
            "chunk_text",
            "token_count",
            "chunk_sha256",
            "cpt_codes",
            "icd10_codes",
            "mac_jurisdiction",
            "effective_date",
            "retired_date",
            "is_active",
            "embedding",
            "embedding_model",
        ]
        return pd.DataFrame(columns=columns), 0, 0

    missing_indices: list[int] = []
    missing_inputs: list[str] = []
    vectors: list[list[float] | None] = [None] * len(candidates)
    chunks_reused = 0

    for idx, row in enumerate(candidates):
        cached = prior_vectors.get(str(row["chunk_sha256"]))
        if cached is not None:
            vectors[idx] = cached
            chunks_reused += 1
        else:
            missing_indices.append(idx)
            missing_inputs.append(f"{row['title']}\n{row['section_heading']}\n{row['chunk_text']}")

    chunks_embedded = len(missing_indices)
    if missing_inputs:
        runner = (
            embed_fn
            if embed_fn is not None
            else lambda batch: _embed_with_fastembed(batch, settings=settings)
        )
        computed = runner(missing_inputs)
        if len(computed) != len(missing_indices):
            msg = f"Embedder returned {len(computed)} vectors for {len(missing_indices)} inputs"
            raise ValueError(msg)
        for idx, computed_vec in zip(missing_indices, computed, strict=True):
            vectors[idx] = [float(x) for x in computed_vec]

    enriched_rows: list[dict[str, Any]] = []
    for row, vector in zip(candidates, vectors, strict=True):
        if vector is None:
            msg = f"Missing embedding vector for chunk_id={row['chunk_id']}"
            raise RuntimeError(msg)
        enriched = dict(row)
        enriched["embedding"] = vector
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows), chunks_reused, chunks_embedded


def _gold_arrow_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("doc_id", pa.string(), nullable=False),
            pa.field("doc_type", pa.string(), nullable=False),
            pa.field("version", pa.int64(), nullable=False),
            pa.field("chunk_index", pa.int64(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("section_heading", pa.string(), nullable=False),
            pa.field("chunk_text", pa.string(), nullable=False),
            pa.field("token_count", pa.int64(), nullable=False),
            pa.field("chunk_sha256", pa.string(), nullable=False),
            pa.field("cpt_codes", pa.list_(pa.string()), nullable=False),
            pa.field("icd10_codes", pa.list_(pa.string()), nullable=False),
            pa.field("mac_jurisdiction", pa.string(), nullable=True),
            pa.field("effective_date", pa.date32(), nullable=False),
            pa.field("retired_date", pa.date32(), nullable=True),
            pa.field("is_active", pa.bool_(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim), nullable=False),
            pa.field("embedding_model", pa.string(), nullable=False),
        ]
    )


def _write_gold_chunks_parquet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    embedding_dim: int,
) -> None:
    records = [
        {
            key: (None if isinstance(value, float) and pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]
    table = pa.Table.from_pylist(
        records,
        schema=_gold_arrow_schema(embedding_dim),
    )
    pq.write_table(table, output_path)


def _build_duckdb_index(
    *,
    chunks_path: Path,
    index_path: Path,
    embedding_dim: int,
    extension_dir: Path | None,
) -> None:
    tmp_path = index_path.with_suffix(".duckdb.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    config: dict[str, str | bool | int | float | list[str]] = {}
    if extension_dir is not None:
        config["extension_directory"] = str(extension_dir.resolve())

    with duckdb.connect(str(tmp_path), config=config) as con:
        try:
            con.execute("LOAD fts")
        except duckdb.IOException:
            con.execute("INSTALL fts")
            con.execute("LOAD fts")

        con.execute(
            f"CREATE TABLE chunks AS "
            f"SELECT * REPLACE (CAST(embedding AS FLOAT[{embedding_dim}]) AS embedding) "
            f"FROM read_parquet(?)",
            [str(chunks_path)],
        )
        con.execute(
            "PRAGMA create_fts_index("
            "'chunks', 'chunk_id', 'title', 'section_heading', 'chunk_text', "
            "stemmer='porter', stopwords='english', overwrite=1)"
        )

    os.replace(tmp_path, index_path)


def _resolve_quarantine_count(settings: Settings, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    quarantine_path = settings.silver_dir / "quarantine.parquet"
    if quarantine_path.exists():
        return len(pd.read_parquet(quarantine_path))
    return 0


def _resolve_bronze_count(settings: Settings, *, fallback: int) -> int:
    bronze_path = settings.bronze_dir / "raw_documents.parquet"
    if bronze_path.exists():
        return len(pd.read_parquet(bronze_path))
    return fallback


def _resolve_git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = completed.stdout.strip()
        return sha if sha else "nogit"
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _to_date(value: Any) -> date:
    parsed = _to_optional_date(value)
    if parsed is None:
        msg = f"Expected non-null date, received {value!r}"
        raise ValueError(msg)
    return parsed


def _to_optional_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
