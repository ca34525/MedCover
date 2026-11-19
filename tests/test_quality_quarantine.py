"""Tests for Silver quarantine routing on malformed records, date inversions, and empty bodies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from medcover.bronze import ingest_sample_fixtures, ingest_zip_or_dir
from medcover.config import Settings
from medcover.silver import build_silver


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
