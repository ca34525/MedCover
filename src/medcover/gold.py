"""Gold layer section-aware chunking, incremental FastEmbed inference, and index.duckdb builder."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from fastembed import TextEmbedding

from medcover.config import Settings
from medcover.contracts import (
    VALID_SOURCE_MODES,
    RunManifest,
    SourceMode,
    build_gold_chunks_schema,
    gold_chunks_arrow_schema,
)

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


@dataclass(frozen=True)
class _BlockPiece:
    text: str
    token_count: int
    is_table: bool
    skip_overlap: bool


def count_tokens(text: str) -> int:
    """Count deterministic lexical tokens for chunk budgeting and manifest telemetry.

    Inputs: UTF-8 text string.
    Output grain: non-negative integer token count.
    Invariants: deterministic offline execution; strictly additive across whitespace/newlines.
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
    (repeating headers on large tables and never slicing table tails for overlap), and emits
    non-empty chunks in linear time.
    """
    sections = _split_markdown_sections(markdown_text, default_heading=default_heading)
    chunks: list[MarkdownChunk] = []

    for heading, body in sections:
        blocks = _split_blocks(body)
        if not blocks:
            continue
        for text, token_count in _pack_blocks(
            blocks,
            token_limit=token_limit,
            token_overlap=token_overlap,
        ):
            cleaned = text.strip()
            if cleaned:
                chunks.append(
                    MarkdownChunk(
                        section_heading=heading,
                        chunk_text=cleaned,
                        token_count=max(1, token_count),
                    )
                )

    return chunks


def build_gold(
    settings: Settings,
    *,
    bronze_docs: int | None = None,
    quarantined_docs: int | None = None,
    source_modes: Sequence[SourceMode] | None = None,
    embed_fn: EmbedFn | None = None,
) -> GoldResult:
    """Chunk Silver policies, incrementally embed changed chunks, and build ``index.duckdb``.

    Inputs: ``Settings`` plus optional Bronze/quarantine counts, source modes, and ``embed_fn``.
    Output grain: ``gold/chunks.parquet`` unique on ``(doc_id, version, chunk_index)``,
    pre-indexed ``gold/index.duckdb``, and immutable ``gold/run_manifest.json``.
    Invariants: unchanged ``chunk_sha256`` vectors are reused from prior Gold (and deduplicated
    within the same build) so repeat runs yield ``chunks_embedded == 0``; ``index.duckdb`` is
    built with FTS and query-time temporal views ready for read-only serving.
    """
    policies_path = settings.silver_dir / "policies.parquet"
    if not policies_path.exists():
        msg = f"Silver policies file not found: {policies_path}"
        raise FileNotFoundError(msg)

    policies_rows = pq.read_table(policies_path).to_pylist()
    candidates = _build_candidate_chunks(policies_rows, settings=settings)

    settings.gold_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.gold_dir / "chunks.parquet"
    index_path = settings.gold_dir / "index.duckdb"
    manifest_path = settings.gold_dir / "run_manifest.json"

    prior_vectors = _load_prior_embeddings(
        chunks_path,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    enriched_rows, chunks_reused, chunks_embedded = _apply_incremental_embeddings(
        candidates,
        prior_vectors=prior_vectors,
        settings=settings,
        embed_fn=embed_fn,
    )

    chunks_table = pa.Table.from_pylist(
        enriched_rows,
        schema=gold_chunks_arrow_schema(settings.embedding_dim),
    )
    chunks_df = chunks_table.to_pandas()
    build_gold_chunks_schema(settings.embedding_dim).validate(chunks_df)
    pq.write_table(chunks_table, chunks_path)

    _build_duckdb_index(
        chunks_path=chunks_path,
        index_path=index_path,
        embedding_dim=settings.embedding_dim,
        extension_dir=settings.resolved_duckdb_extension_dir(),
    )

    bronze_docs_path = settings.bronze_dir / "raw_documents.parquet"
    silver_count = len(policies_rows)
    quarantine_count = (
        quarantined_docs
        if quarantined_docs is not None
        else _parquet_row_count(settings.silver_dir / "quarantine.parquet", fallback=0)
    )
    bronze_count = (
        bronze_docs
        if bronze_docs is not None
        else _parquet_row_count(
            bronze_docs_path,
            fallback=silver_count + quarantine_count,
        )
    )

    if chunks_df.empty:
        token_p50, token_p95 = 0.0, 0.0
    else:
        quantiles = chunks_df["token_count"].quantile([0.50, 0.95])
        token_p50 = float(quantiles.loc[0.50])
        token_p95 = float(quantiles.loc[0.95])

    now = datetime.now(tz=UTC)
    git_sha = _resolve_git_sha()
    resolved_source_modes: list[SourceMode] = (
        sorted(set(source_modes))
        if source_modes is not None
        else _resolve_source_modes(bronze_docs_path)
    )
    manifest = RunManifest(
        run_id=f"{now.strftime('%Y%m%dT%H%M%SZ')}_{git_sha}",
        git_sha=git_sha,
        timestamp_utc=now,
        source_modes=resolved_source_modes,
        bronze_docs=bronze_count,
        silver_docs=silver_count,
        quarantined_docs=quarantine_count,
        gold_chunks=chunks_table.num_rows,
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


def _split_large_table(table_block: str, *, token_limit: int) -> list[tuple[str, int]]:
    lines = [line for line in table_block.splitlines() if line.strip()]
    row_tokens = [count_tokens(line) for line in lines]
    total_tokens = sum(row_tokens)
    if len(lines) <= 2 or total_tokens <= token_limit:
        return [(table_block, total_tokens)]

    header_lines = lines[:2]
    header_tokens = row_tokens[0] + row_tokens[1]
    pieces: list[tuple[str, int]] = []
    current_rows: list[str] = list(header_lines)
    current_tokens = header_tokens

    for row, tokens in zip(lines[2:], row_tokens[2:], strict=True):
        if len(current_rows) > 2 and current_tokens + tokens > token_limit:
            pieces.append(("\n".join(current_rows), current_tokens))
            current_rows = [*header_lines, row]
            current_tokens = header_tokens + tokens
        else:
            current_rows.append(row)
            current_tokens += tokens

    if len(current_rows) > 2:
        pieces.append(("\n".join(current_rows), current_tokens))
    return pieces


def _split_large_prose(
    block: str,
    *,
    token_limit: int,
    token_overlap: int,
) -> list[tuple[str, int]]:
    total_tokens = count_tokens(block)
    if total_tokens <= token_limit:
        return [(block, total_tokens)]

    words = block.split()
    word_tokens = [count_tokens(word) for word in words]
    pieces: list[tuple[str, int]] = []
    start = 0
    while start < len(words):
        end = start
        chunk_tokens = 0
        while end < len(words):
            next_tokens = chunk_tokens + word_tokens[end]
            if end > start and next_tokens > token_limit:
                break
            chunk_tokens = next_tokens
            end += 1
        pieces.append((" ".join(words[start:end]), chunk_tokens))
        if end >= len(words):
            break
        step_back = _overlap_word_count(word_tokens[start:end], token_overlap=token_overlap)
        start = max(start + 1, end - step_back)
    return pieces


def _overlap_word_count(word_tokens: Sequence[int], *, token_overlap: int) -> int:
    if token_overlap <= 0 or not word_tokens:
        return 0
    tokens = 0
    count = 0
    for tok in reversed(word_tokens):
        tokens += max(1, tok)
        count += 1
        if tokens >= token_overlap:
            break
    return count


def _tail_prose_overlap(
    parts: Sequence[_BlockPiece],
    *,
    token_overlap: int,
) -> tuple[str, int]:
    if token_overlap <= 0 or not parts or parts[-1].is_table:
        return "", 0

    trailing_prose: list[str] = []
    for part in reversed(parts):
        if part.is_table:
            break
        trailing_prose.append(part.text)
    trailing_prose.reverse()

    words = " ".join(trailing_prose).split()
    word_tokens = [count_tokens(word) for word in words]
    keep = _overlap_word_count(word_tokens, token_overlap=token_overlap)
    if keep <= 0:
        return "", 0
    return " ".join(words[-keep:]), sum(word_tokens[-keep:])


def _pack_blocks(
    blocks: Sequence[str],
    *,
    token_limit: int,
    token_overlap: int,
) -> list[tuple[str, int]]:
    expanded: list[_BlockPiece] = []
    for block in blocks:
        if _is_markdown_table(block):
            expanded.extend(
                _BlockPiece(text=text, token_count=tokens, is_table=True, skip_overlap=True)
                for text, tokens in _split_large_table(block, token_limit=token_limit)
            )
        else:
            prose_pieces = _split_large_prose(
                block, token_limit=token_limit, token_overlap=token_overlap
            )
            expanded.extend(
                _BlockPiece(text=text, token_count=tokens, is_table=False, skip_overlap=idx > 0)
                for idx, (text, tokens) in enumerate(prose_pieces)
            )

    packed: list[tuple[str, int]] = []
    current_parts: list[_BlockPiece] = []
    current_tokens = 0

    for piece in expanded:
        if not current_parts:
            current_parts = [piece]
            current_tokens = piece.token_count
            continue
        if current_tokens + piece.token_count <= token_limit:
            current_parts.append(piece)
            current_tokens += piece.token_count
            continue

        packed.append(("\n\n".join(part.text for part in current_parts), current_tokens))
        overlap_text, overlap_tokens = (
            ("", 0)
            if piece.skip_overlap
            else _tail_prose_overlap(current_parts, token_overlap=token_overlap)
        )
        if overlap_text and overlap_tokens + piece.token_count <= token_limit:
            overlap_piece = _BlockPiece(
                text=overlap_text,
                token_count=overlap_tokens,
                is_table=False,
                skip_overlap=False,
            )
            current_parts = [overlap_piece, piece]
            current_tokens = overlap_tokens + piece.token_count
        else:
            current_parts = [piece]
            current_tokens = piece.token_count

    if current_parts:
        packed.append(("\n\n".join(part.text for part in current_parts), current_tokens))
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
    policies_rows: Sequence[dict[str, Any]],
    *,
    settings: Settings,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    sorted_policies = sorted(
        policies_rows,
        key=lambda row: (str(row["doc_id"]), int(row["version"])),
    )

    for record in sorted_policies:
        doc_id = str(record["doc_id"])
        version = int(record["version"])
        title = str(record["title"])
        markdown_text = str(record["markdown_text"])
        hcpcs_codes = [str(code) for code in record["hcpcs_codes"]]
        icd10_codes = [str(code) for code in record["icd10_codes"]]

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
                    "hcpcs_codes": hcpcs_codes,
                    "icd10_codes": icd10_codes,
                    "mac_jurisdiction": record["mac_jurisdiction"],
                    "effective_date": record["effective_date"],
                    "retired_date": record["retired_date"],
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
    rows = pq.read_table(
        chunks_path,
        columns=["chunk_sha256", "embedding", "embedding_model"],
    ).to_pylist()
    vectors: dict[str, list[float]] = {}
    for row in rows:
        if row["embedding_model"] != embedding_model:
            continue
        vec = row["embedding"]
        if isinstance(vec, list) and len(vec) == embedding_dim:
            vectors[str(row["chunk_sha256"])] = vec
    return vectors


def _embed_with_fastembed(texts: Sequence[str], *, settings: Settings) -> list[list[float]]:
    cache_dir = settings.resolved_fastembed_cache_dir()
    model = TextEmbedding(
        model_name=settings.embedding_model,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return [list(vector.tolist()) for vector in model.embed(list(texts))]


def _apply_incremental_embeddings(
    candidates: list[dict[str, Any]],
    *,
    prior_vectors: dict[str, list[float]],
    settings: Settings,
    embed_fn: EmbedFn | None,
) -> tuple[list[dict[str, Any]], int, int]:
    if not candidates:
        return [], 0, 0

    known_vectors: dict[str, list[float]] = dict(prior_vectors)
    missing_by_sha: dict[str, str] = {}
    for row in candidates:
        sha = str(row["chunk_sha256"])
        if sha not in known_vectors and sha not in missing_by_sha:
            missing_by_sha[sha] = f"{row['title']}\n{row['section_heading']}\n{row['chunk_text']}"

    chunks_embedded = len(missing_by_sha)
    if missing_by_sha:
        runner = (
            embed_fn
            if embed_fn is not None
            else lambda batch: _embed_with_fastembed(batch, settings=settings)
        )
        computed = runner(list(missing_by_sha.values()))
        if len(computed) != chunks_embedded:
            msg = f"Embedder returned {len(computed)} vectors for {chunks_embedded} inputs"
            raise ValueError(msg)
        for sha, computed_vec in zip(missing_by_sha, computed, strict=True):
            known_vectors[sha] = (
                computed_vec if isinstance(computed_vec, list) else list(computed_vec)
            )

    enriched_rows: list[dict[str, Any]] = []
    for row in candidates:
        sha = str(row["chunk_sha256"])
        vector = known_vectors.get(sha)
        if vector is None:
            msg = f"Missing embedding vector for chunk_id={row['chunk_id']}"
            raise RuntimeError(msg)
        enriched_rows.append({**row, "embedding": vector})

    chunks_reused = len(candidates) - chunks_embedded
    return enriched_rows, chunks_reused, chunks_embedded


def _build_duckdb_index(
    *,
    chunks_path: Path,
    index_path: Path,
    embedding_dim: int,
    extension_dir: Path | None,
) -> None:
    tmp_path = index_path.with_suffix(".duckdb.tmp")
    for stale in (tmp_path, tmp_path.with_name(f"{tmp_path.name}.wal")):
        if stale.exists():
            stale.unlink()

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
        con.execute(
            "CREATE OR REPLACE MACRO chunks_as_of(as_of_date) AS TABLE "
            "SELECT * FROM chunks "
            "WHERE effective_date <= as_of_date "
            "AND (retired_date IS NULL OR retired_date > as_of_date)"
        )
        con.execute(
            "CREATE OR REPLACE VIEW active_chunks AS SELECT * FROM chunks_as_of(CURRENT_DATE)"
        )

    os.replace(tmp_path, index_path)


def _parquet_row_count(path: Path, *, fallback: int) -> int:
    if path.exists():
        return int(pq.read_metadata(path).num_rows)
    return fallback


def _resolve_source_modes(bronze_docs_path: Path) -> list[SourceMode]:
    if not bronze_docs_path.exists():
        return []
    rows = pq.read_table(bronze_docs_path, columns=["source_mode"]).to_pylist()
    modes: set[SourceMode] = set()
    for row in rows:
        value = row.get("source_mode")
        if isinstance(value, str) and value in VALID_SOURCE_MODES:
            modes.add("api" if value == "api" else "zip")
    return sorted(modes)


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
