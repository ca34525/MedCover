"""Tests for Silver HTML-to-Markdown preservation, code array aggregation, and grain safety."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from medcover.bronze import ingest_sample_fixtures
from medcover.config import Settings
from medcover.silver import build_silver, html_to_markdown


def test_html_to_markdown_preserves_headings_and_tables() -> None:
    """Verify HTML headings and tables convert cleanly to Markdown and scripts are stripped."""
    raw_html = (
        "<script>alert('ignore')</script>"
        "<h1>Policy Header</h1>"
        "<h2>Criteria Table</h2>"
        "<table><thead><tr><th>Code</th><th>Rule</th></tr></thead>"
        "<tbody><tr><td>72148</td><td>6 weeks therapy</td></tr></tbody></table>"
    )
    md = html_to_markdown(raw_html)

    assert "alert" not in md
    assert "# Policy Header" in md
    assert "## Criteria Table" in md
    assert "| Code | Rule |" in md
    assert "| 72148 | 6 weeks therapy |" in md


def test_silver_zero_code_fanout_and_distinct_arrays(tmp_path: Path) -> None:
    """Verify 1-to-many duplicate crosswalks aggregate via LIST(DISTINCT code) without fan-out."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)

    policies_df = pd.read_parquet(silver_res.policies_path)

    # 15 Bronze docs minus 1 malformed doc (L99999) = 14 unique Silver policies.
    assert silver_res.bronze_docs == 15
    assert silver_res.silver_docs == 14
    assert silver_res.quarantined_docs == 1
    assert not policies_df.duplicated(subset=["doc_id", "version"]).any()
    assert {"is_active", "status", "source_mode"}.isdisjoint(set(policies_df.columns))

    # L33822 has duplicate 72148 in raw crosswalks; Silver must deduplicate cleanly.
    l33822 = policies_df[policies_df["doc_id"] == "L33822"].iloc[0]
    assert list(l33822["hcpcs_codes"]) == ["72148", "72149", "72158"]
    assert list(l33822["icd10_codes"]) == ["G83.4", "M51.16", "M54.50"]
    assert l33822["effective_date"] == date(2022, 4, 1)
    assert pd.isna(l33822["retired_date"])

    # L33381 carries its exact validity interval [effective_date, retired_date).
    l33381 = policies_df[policies_df["doc_id"] == "L33381"].iloc[0]
    assert l33381["effective_date"] == date(2018, 1, 1)
    assert l33381["retired_date"] == date(2024, 6, 30)
    assert "| Procedure | CPT Code | Legacy Status |" in str(l33381["markdown_text"])
