"""Tests for Gold chunking, incremental embedding idempotency, and pre-built DuckDB FTS index."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from medcover.bronze import ingest_sample_fixtures, ingest_zip_or_dir
from medcover.config import Settings
from medcover.gold import build_gold, chunk_markdown
from medcover.silver import build_silver


def _deterministic_embed(texts: Sequence[str], dim: int = 384) -> list[list[float]]:
    """Fast offline deterministic 384-dim unit-like embedder for unit tests."""
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [float(digest[idx % len(digest)]) / 255.0 for idx in range(dim)]
        vectors.append(vec)
    return vectors


def test_chunk_markdown_preserves_table_rows_on_split() -> None:
    """Verify large Markdown tables split strictly on row boundaries with repeated headers."""
    rows = "\n".join(
        f"| CODE{i:03d} | Coverage criteria detail number {i} for Medicare |" for i in range(12)
    )
    md = f"## Coverage Table\n\n| Code | Description |\n| --- | --- |\n{rows}"

    chunks = chunk_markdown(md, default_heading="Test", token_limit=60, token_overlap=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.section_heading == "Coverage Table"
        assert "| Code | Description |" in chunk.chunk_text
        for line in chunk.chunk_text.splitlines():
            if line.strip():
                assert line.startswith("|") and line.endswith("|")


def test_chunk_markdown_does_not_duplicate_overlap_on_long_paragraph() -> None:
    """Verify a prose paragraph exceeding token_limit applies overlap once, not twice."""
    words = [f"word{i}" for i in range(451)]
    md = "## Criteria\n\n" + " ".join(words)

    chunks = chunk_markdown(md, default_heading="Test", token_limit=450, token_overlap=60)
    assert len(chunks) == 2
    assert chunks[0].token_count == 450
    assert chunks[1].token_count == 61
    assert chunks[1].chunk_text.split() == words[390:451]


def test_chunk_markdown_mixed_prose_and_table_never_corrupts_table_overlap() -> None:
    """Verify a chunk ending with a Markdown table never leaks pipe fragments into overlap."""
    intro = " ".join(f"intro{i}" for i in range(15))
    table = "| Code | Rule |\n| --- | --- |\n| 72148 | six weeks |\n| 72149 | contrast |"
    after = " ".join(f"after{i}" for i in range(40))
    md = f"## Criteria\n\n{intro}\n\n{table}\n\n{after}"

    chunks = chunk_markdown(md, default_heading="Test", token_limit=55, token_overlap=15)
    assert len(chunks) == 2
    assert "| 72148 | six weeks |" in chunks[0].chunk_text
    assert "|" not in chunks[1].chunk_text
    assert chunks[1].chunk_text.startswith("after0")


def test_gold_deduplicates_intra_build_chunks_and_records_provenance(tmp_path: Path) -> None:
    """Verify identical sections embed once per build and manifest records DuckDB & source modes."""
    api_csv = tmp_path / "api_versions.csv"
    body = '"<h2>Criteria</h2><p>Identical text.</p>"'
    api_csv.write_text(
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        f"L55555,LCD,1,Shared Policy,Novitas,2023-01-01,,ACTIVE,{body}\n"
        f"L55555,LCD,2,Shared Policy,Novitas,2024-01-01,,ACTIVE,{body}\n",
        encoding="utf-8",
    )

    settings = Settings(data_dir=tmp_path)
    ingest_zip_or_dir(api_csv, settings.bronze_dir, source_mode="api")
    silver_res = build_silver(settings)
    gold_res = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )

    assert gold_res.manifest.source_modes == ["api"]
    assert gold_res.manifest.duckdb_version == duckdb.__version__
    assert gold_res.manifest.duckdb_storage_version == settings.duckdb_storage_version
    assert gold_res.manifest.gold_chunks == 2
    assert gold_res.manifest.chunks_embedded == 1
    assert gold_res.manifest.chunks_reused == 1

    # Mixing a zip backfill doc alongside api docs reports both source modes honestly.
    zip_csv = tmp_path / "zip_doc.csv"
    zip_csv.write_text(
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        f"L55556,LCD,1,Backfill Policy,Novitas,2023-01-01,,ACTIVE,{body}\n",
        encoding="utf-8",
    )
    ingest_zip_or_dir(zip_csv, settings.bronze_dir, source_mode="zip")
    mixed_silver = build_silver(settings)
    mixed_gold = build_gold(
        settings,
        bronze_docs=mixed_silver.bronze_docs,
        quarantined_docs=mixed_silver.quarantined_docs,
        embed_fn=_deterministic_embed,
    )
    assert mixed_gold.manifest.source_modes == ["api", "zip"]


def test_gold_idempotency_and_delta_reembed(tmp_path: Path) -> None:
    """Verify repeat Gold run embeds 0 chunks and 1-doc edit re-embeds only its delta chunks."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)

    first = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )
    assert first.manifest.gold_chunks > 0
    assert first.manifest.chunks_embedded == first.manifest.gold_chunks
    assert first.manifest.chunks_reused == 0

    second = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )
    assert second.manifest.gold_chunks == first.manifest.gold_chunks
    assert second.manifest.chunks_embedded == 0
    assert second.manifest.chunks_reused == first.manifest.gold_chunks

    # Mutate 1 document in Bronze, rebuild Silver + Gold, and confirm strictly its chunks re-embed.
    bronze_docs_path = settings.bronze_dir / "raw_documents.parquet"
    bronze_table = pq.read_table(bronze_docs_path)
    bronze_rows = bronze_table.to_pylist()
    for row in bronze_rows:
        if row["doc_id"] == "NCD190.15":
            row["raw_html"] = (
                str(row["raw_html"])
                + "<p>Updated hematology documentation requirement for delta test.</p>"
            )
    pq.write_table(pa.Table.from_pylist(bronze_rows, schema=bronze_table.schema), bronze_docs_path)

    silver_after = build_silver(settings)
    third = build_gold(
        settings,
        bronze_docs=silver_after.bronze_docs,
        quarantined_docs=silver_after.quarantined_docs,
        embed_fn=_deterministic_embed,
    )
    chunks_table = pq.read_table(third.chunks_path)
    assert "is_active" not in chunks_table.column_names
    assert "hcpcs_codes" in chunks_table.column_names
    assert "valid_to" in chunks_table.column_names
    ncd_chunks = sum(1 for row in chunks_table.to_pylist() if row["doc_id"] == "NCD190.15")

    assert third.manifest.chunks_embedded > 0
    assert third.manifest.chunks_embedded <= ncd_chunks
    assert (
        third.manifest.chunks_reused == third.manifest.gold_chunks - third.manifest.chunks_embedded
    )


def test_gold_index_duckdb_opens_readonly_with_fts_vectors_and_scd2_views(tmp_path: Path) -> None:
    """Verify pre-built index.duckdb opens read-only with FTS, vectors, and SCD2 chunks_as_of."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)
    gold_res = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )

    config: dict[str, str] = {}
    ext_dir = settings.resolved_duckdb_extension_dir()
    if ext_dir is not None:
        config["extension_directory"] = str(ext_dir.resolve())

    query_vec = _deterministic_embed(["lumbar spine MRI radiculopathy"])[0]
    with duckdb.connect(str(gold_res.index_path), read_only=True, config=config) as con:
        con.execute("LOAD fts")
        fts_hits = con.execute(
            "WITH scored AS ("
            "SELECT chunk_id, doc_id, fts_main_chunks.match_bm25(chunk_id, ?) AS score FROM chunks"
            ") SELECT chunk_id, doc_id, score FROM scored "
            "WHERE score IS NOT NULL ORDER BY score DESC LIMIT 5",
            ["lumbar spine MRI"],
        ).fetchall()
        vec_hits = con.execute(
            "SELECT chunk_id, array_cosine_distance(embedding, ?::FLOAT[384]) AS dist "
            "FROM chunks ORDER BY dist ASC LIMIT 5",
            [query_vec],
        ).fetchall()
        docs_2021 = {
            (row[0], row[1])
            for row in con.execute(
                "SELECT DISTINCT doc_id, version FROM chunks_as_of(DATE '2021-06-01')"
            ).fetchall()
        }
        docs_2023 = {
            row[0]
            for row in con.execute(
                "SELECT DISTINCT doc_id FROM chunks_as_of(DATE '2023-06-01')"
            ).fetchall()
        }
        docs_2026 = {
            (row[0], row[1])
            for row in con.execute(
                "SELECT DISTINCT doc_id, version FROM chunks_as_of(DATE '2026-01-01')"
            ).fetchall()
        }
        active_view_docs = {
            row[0] for row in con.execute("SELECT DISTINCT doc_id FROM active_chunks").fetchall()
        }

    assert len(fts_hits) > 0
    assert any(doc_id in {"L33822", "A56542"} for _, doc_id, _ in fts_hits)
    assert len(vec_hits) == 5
    assert ("L33822", 11) in docs_2021
    assert ("L33822", 12) not in docs_2021
    assert "L33381" in docs_2023
    assert ("L33822", 12) in docs_2026
    assert ("L33822", 11) not in docs_2026
    assert all(doc_id != "L33381" for doc_id, _ in docs_2026)
    assert "L33381" not in active_view_docs
