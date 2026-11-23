"""Tests for Silver quarantine routing and Gold/Silver data quality audit invariants."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from medcover.bronze import ingest_sample_fixtures, ingest_zip_or_dir
from medcover.config import Settings
from medcover.gold import build_gold
from medcover.quality import audit_medallion, render_manifest_summary
from medcover.silver import build_silver


def _deterministic_embed(texts: Sequence[str], dim: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([float(digest[idx % len(digest)]) / 255.0 for idx in range(dim)])
    return vectors


def test_malformed_fixture_routed_to_quarantine(tmp_path: Path) -> None:
    """Verify L99999 routes to quarantine with explicit reasons and never enters Silver policies."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)

    quarantine_df = pd.read_parquet(silver_res.quarantine_path)
    policies_df = pd.read_parquet(silver_res.policies_path)

    assert len(quarantine_df) == 1
    row = quarantine_df.iloc[0]
    assert row["doc_id"] == "L99999"
    assert "retired_date_before_effective_date" in str(row["failure_reason"])
    assert "empty_cleaned_text" in str(row["failure_reason"])
    assert "L99999" not in set(policies_df["doc_id"])
    assert silver_res.quarantine_rate <= settings.max_quarantine_rate


def test_missing_title_and_date_inversion_quarantined(tmp_path: Path) -> None:
    """Verify custom malformed inputs are explicitly quarantined with accurate failure reasons."""
    bad_csv = tmp_path / "bad_docs.csv"
    bad_csv.write_text(
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        'L11111,LCD,1,,Novitas,2024-01-01,,ACTIVE,"<p>Valid text</p>"\n'
        'L22222,LCD,1,Inverted Dates,Novitas,2024-06-01,2024-01-01,RETIRED,"<p>Valid text</p>"\n'
        'L33333,LCD,1,Valid Policy,Novitas,2024-01-01,,ACTIVE,"<h1>Good</h1><p>Covered.</p>"\n',
        encoding="utf-8",
    )

    settings = Settings(data_dir=tmp_path)
    ingest_zip_or_dir(bad_csv, settings.bronze_dir)
    silver_res = build_silver(settings)

    quarantine_df = pd.read_parquet(silver_res.quarantine_path)
    policies_df = pd.read_parquet(silver_res.policies_path)

    assert len(policies_df) == 1
    assert policies_df.iloc[0]["doc_id"] == "L33333"
    assert len(quarantine_df) == 2
    reasons = dict(zip(quarantine_df["doc_id"], quarantine_df["failure_reason"], strict=True))
    assert "missing_title" in reasons["L11111"]
    assert "retired_date_before_effective_date" in reasons["L22222"]
    assert silver_res.quarantine_rate > settings.max_quarantine_rate


def test_all_quarantined_batch_and_malformed_bronze_strings(tmp_path: Path) -> None:
    """Verify 0-valid-row batches pass Silver and Gold Pandera/Arrow contracts cleanly."""
    settings = Settings(data_dir=tmp_path)
    settings.bronze_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "doc_id": "L44444",
                "doc_type": "LCD",
                "version": "not-an-int",
                "title": "Bad Types",
                "raw_html": "<p>Text</p>",
                "mac_jurisdiction": None,
                "effective_date": "not-a-date",
                "retired_date": None,
                "status": "ACTIVE",
                "source_mode": "zip",
                "ingested_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                "source_sha256": "a" * 64,
            }
        ]
    ).to_parquet(settings.bronze_dir / "raw_documents.parquet", index=False)

    silver_res = build_silver(settings)
    assert silver_res.silver_docs == 0
    assert silver_res.quarantined_docs == 1

    quarantine_df = pd.read_parquet(silver_res.quarantine_path)
    reason = str(quarantine_df.iloc[0]["failure_reason"])
    assert "invalid_version" in reason
    assert "missing_effective_date" in reason

    gold_res = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )
    assert gold_res.manifest.gold_chunks == 0
    assert gold_res.manifest.chunks_embedded == 0
    assert gold_res.index_path.exists()


def test_audit_medallion_passes_on_sample_build(tmp_path: Path) -> None:
    """Verify full Pandera + dq_invariants.sql + FTS index audit passes on sample build."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)
    build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )

    report = audit_medallion(settings)
    assert report.passed is True
    assert report.failed_checks == []
    assert report.silver_docs == 14
    assert report.quarantined_docs == 1
    assert report.quarantine_rate <= settings.max_quarantine_rate
    assert report.manifest is not None

    summary = render_manifest_summary(report.manifest, report)
    assert "Data Quality Audit: PASSED" in summary
    assert "`gold_valid_embeddings`" in summary


def test_audit_medallion_fails_when_quarantine_rate_exceeds_threshold(tmp_path: Path) -> None:
    """Verify quarantine_rate > max_quarantine_rate fails the data quality audit."""
    bad_csv = tmp_path / "high_quarantine.csv"
    bad_csv.write_text(
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        'L11111,LCD,1,,Novitas,2024-01-01,,ACTIVE,"<p>Missing title</p>"\n'
        'L22222,LCD,1,Inverted,Novitas,2024-06-01,2024-01-01,RETIRED,"<p>Bad dates</p>"\n'
        'L33333,LCD,1,Valid Policy,Novitas,2024-01-01,,ACTIVE,"<h1>Good</h1><p>Covered.</p>"\n',
        encoding="utf-8",
    )

    settings = Settings(data_dir=tmp_path)
    ingest_zip_or_dir(bad_csv, settings.bronze_dir)
    silver_res = build_silver(settings)
    build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )

    report = audit_medallion(settings)
    assert report.passed is False
    failed_names = {check.name for check in report.failed_checks}
    assert "quarantine_rate_within_limit" in failed_names


def test_audit_medallion_catches_gold_sql_and_temporal_violations(tmp_path: Path) -> None:
    """Verify dq_invariants.sql catches duplicates, empty text, NaN vectors, and active leaks."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)
    gold_res = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_deterministic_embed,
    )

    table = pq.read_table(gold_res.chunks_path)
    rows = table.to_pylist()
    corrupted = dict(rows[0])
    corrupted["chunk_text"] = "   "
    corrupted["token_count"] = settings.chunk_token_limit + 50
    corrupted["retired_date"] = date(2020, 1, 1)
    corrupted["is_active"] = True
    bad_vec = list(corrupted["embedding"])
    bad_vec[0] = float("nan")
    corrupted["embedding"] = bad_vec
    rows.append(corrupted)

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), gold_res.chunks_path)

    report = audit_medallion(settings)
    assert report.passed is False
    failed_names = {check.name for check in report.failed_checks}
    expected_failures = {
        "pandera_gold_chunks",
        "gold_index_sync",
        "gold_unique_grain",
        "gold_unique_chunk_id",
        "gold_non_empty_chunk_text",
        "gold_bounded_token_count",
        "gold_valid_embeddings",
        "gold_active_flag_consistency",
        "manifest_counts_match",
    }
    assert expected_failures.issubset(failed_names)
