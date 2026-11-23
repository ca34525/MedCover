"""Tests for Silver HTML-to-Markdown preservation, code aggregation, and SCD2 valid_to."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

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


def test_silver_zero_code_fanout_distinct_arrays_and_scd2_valid_to(tmp_path: Path) -> None:
    """Verify LIST(DISTINCT code) anti-fan-out and SCD2 LEAD() valid_to version closing."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)

    policies_table = pq.read_table(silver_res.policies_path)
    policies = policies_table.to_pylist()

    # 16 Bronze docs minus 1 malformed doc (L99999) = 15 unique Silver policies.
    assert silver_res.bronze_docs == 16
    assert silver_res.silver_docs == 15
    assert silver_res.quarantined_docs == 1
    keys = [(row["doc_id"], row["version"]) for row in policies]
    assert len(keys) == len(set(keys))
    assert {"is_active", "status", "source_mode"}.isdisjoint(set(policies_table.column_names))

    by_key = {(row["doc_id"], row["version"]): row for row in policies}

    # L33822 v11 has retired_date=None, yet SCD2 LEAD() closes its valid_to at v12's effective_date.
    l33822_v11 = by_key[("L33822", 11)]
    assert l33822_v11["effective_date"] == date(2020, 1, 1)
    assert l33822_v11["retired_date"] is None
    assert l33822_v11["valid_to"] == date(2022, 4, 1)

    # L33822 v12 deduplicates duplicate 72148 in raw crosswalks and stays open-ended.
    l33822_v12 = by_key[("L33822", 12)]
    assert list(l33822_v12["hcpcs_codes"]) == ["72148", "72149", "72158"]
    assert list(l33822_v12["icd10_codes"]) == ["G83.4", "M51.16", "M54.50"]
    assert l33822_v12["effective_date"] == date(2022, 4, 1)
    assert l33822_v12["retired_date"] is None
    assert l33822_v12["valid_to"] is None

    # L33381 carries its exact validity interval [effective_date, valid_to) = retired_date.
    l33381 = by_key[("L33381", 5)]
    assert l33381["effective_date"] == date(2018, 1, 1)
    assert l33381["retired_date"] == date(2024, 6, 30)
    assert l33381["valid_to"] == date(2024, 6, 30)
    assert "| Procedure | CPT Code | Legacy Status |" in str(l33381["markdown_text"])
