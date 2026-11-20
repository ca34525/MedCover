"""Tests for Gold chunking, incremental embedding idempotency, and pre-built DuckDB FTS index."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from medcover.bronze import ingest_sample_fixtures
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
    bronze_df = pd.read_parquet(bronze_docs_path)
    mask = bronze_df["doc_id"] == "NCD190.15"
    bronze_df.loc[mask, "raw_html"] = (
        str(bronze_df.loc[mask, "raw_html"].iloc[0])
        + "<p>Updated hematology documentation requirement for delta test.</p>"
    )
    bronze_df.to_parquet(bronze_docs_path, index=False)

    silver_after = build_silver(settings)
    third = build_gold(
        settings,
        bronze_docs=silver_after.bronze_docs,
        quarantined_docs=silver_after.quarantined_docs,
        embed_fn=_deterministic_embed,
    )
    chunks_df = pd.read_parquet(third.chunks_path)
    active_chunks = chunks_df[chunks_df["is_active"]]
    assert active_chunks["retired_date"].isna().all()
    ncd_chunks = int((chunks_df["doc_id"] == "NCD190.15").sum())

    assert third.manifest.chunks_embedded > 0
    assert third.manifest.chunks_embedded <= ncd_chunks
    assert (
        third.manifest.chunks_reused == third.manifest.gold_chunks - third.manifest.chunks_embedded
    )


def test_gold_index_duckdb_opens_readonly_with_fts_and_vectors(tmp_path: Path) -> None:
    """Verify pre-built index.duckdb opens read-only with active FTS BM25 and cosine distance."""
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

    assert len(fts_hits) > 0
    assert any(doc_id in {"L33822", "A56542"} for _, doc_id, _ in fts_hits)
    assert len(vec_hits) == 5
