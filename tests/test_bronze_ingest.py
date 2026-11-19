"""Tests for canonical Bronze ingestion (nested bulk ZIP, encodings, multiline HTML, upsert)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from medcover.bronze import ingest_sample_fixtures, ingest_zip_or_dir
from medcover.config import Settings


def test_ingest_sample_fixtures_grain_and_counts(tmp_path: Path) -> None:
    """Verify sample fixture directory loads into Bronze with unique (doc_id, version)."""
    settings = Settings(data_dir=tmp_path)
    result = ingest_sample_fixtures(settings)

    docs_df = pd.read_parquet(result.documents_path)
    codes_df = pd.read_parquet(result.crosswalks_path)

    assert result.total_documents == 15
    assert len(docs_df) == 15
    assert not docs_df.duplicated(subset=["doc_id", "version"]).any()
    assert set(docs_df["doc_type"].dropna()) == {"NCD", "LCD", "ARTICLE"}
    assert len(codes_df) > 0


def test_nested_csv_zip_multiline_and_windows1252(tmp_path: Path) -> None:
    """Verify nested _CSV.ZIP streaming decodes cp1252 bytes and multiline HTML cells."""
    csv_text = (
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        'L77777,LCD,1,"CP1252 \u2014 Test Policy",Novitas,2024-01-01,,ACTIVE,'
        '"<h1>Section \u00a71</h1>\n<p>Line one\nLine two \u2013 preserved.</p>"\n'
    )
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner_zip:
        inner_zip.writestr("lcd_cp1252.csv", csv_text.encode("cp1252"))

    outer_path = tmp_path / "cms_bulk_download.zip"
    with zipfile.ZipFile(outer_path, "w") as outer_zip:
        outer_zip.writestr("current_lcd_CSV.ZIP", inner_buf.getvalue())

    bronze_dir = tmp_path / "bronze"
    result = ingest_zip_or_dir(outer_path, bronze_dir)
    docs_df = pd.read_parquet(result.documents_path)

    assert result.total_documents == 1
    row = docs_df.iloc[0]
    assert row["doc_id"] == "L77777"
    assert "\u00a71" in str(row["raw_html"])
    assert "Line one\nLine two" in str(row["raw_html"])


def test_bronze_upsert_preserves_historical_versions(tmp_path: Path) -> None:
    """Verify ingesting a new version preserves prior (doc_id, version) rows."""
    v1_csv = tmp_path / "v1.csv"
    v1_csv.write_text(
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        'L88888,LCD,1,Test Policy,Novitas,2023-01-01,,ACTIVE,"<h1>V1</h1><p>Original</p>"\n',
        encoding="utf-8",
    )
    v2_csv = tmp_path / "v2.csv"
    v2_csv.write_text(
        "doc_id,doc_type,version,title,mac_jurisdiction,effective_date,retired_date,status,raw_html\n"
        'L88888,LCD,2,Test Policy,Novitas,2024-01-01,,ACTIVE,"<h1>V2</h1><p>Updated</p>"\n',
        encoding="utf-8",
    )

    bronze_dir = tmp_path / "bronze"
    ingest_zip_or_dir(v1_csv, bronze_dir)
    ingest_zip_or_dir(v2_csv, bronze_dir)
    docs_df = pd.read_parquet(bronze_dir / "raw_documents.parquet")

    assert len(docs_df) == 2
    assert list(docs_df["version"]) == [1, 2]
