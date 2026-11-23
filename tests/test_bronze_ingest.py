"""Tests for canonical Bronze ingestion (nested bulk ZIP, encodings, multiline HTML, upsert)."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

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


def test_bronze_reingest_replaces_obsolete_crosswalks(tmp_path: Path) -> None:
    """Verify re-ingestion keeps crosswalk count stable and removes dropped codes."""
    settings = Settings(data_dir=tmp_path)
    first = ingest_sample_fixtures(settings)
    second = ingest_sample_fixtures(settings)
    assert second.total_crosswalks == first.total_crosswalks

    updated_codes = tmp_path / "updated_crosswalks.csv"
    updated_codes.write_text(
        "doc_id,version,code_system,code\nL33822,12,CPT,72148\n",
        encoding="utf-8",
    )
    ingest_zip_or_dir(updated_codes, settings.bronze_dir)
    codes_df = pd.read_parquet(settings.bronze_dir / "raw_code_crosswalks.parquet")
    l33822_codes = codes_df[(codes_df["doc_id"] == "L33822") & (codes_df["version"] == 12)]
    assert list(l33822_codes["code"]) == ["72148"]


def test_cms_bulk_lcd_columns_dates_and_unrecognized_archive(tmp_path: Path) -> None:
    """Verify CMS MCD bulk headers and US dates ingest cleanly and empty/unknown CSVs fail."""
    cms_dir = tmp_path / "cms_bulk"
    cms_dir.mkdir()
    (cms_dir / "lcd.csv").write_text(
        "lcd_id,lcd_version,title,indication,orig_det_eff_date,ret_date,status\n"
        '35391,3,Lumbar MRI,"<p>Radiculopathy criteria</p>",04/01/2024,,A\n'
        '35392,1,Bad Date Policy,"<p>Body</p>",not-a-date,,A\n',
        encoding="utf-8",
    )
    (cms_dir / "lcd_x_hcpc_code.csv").write_text(
        "lcd_id,lcd_version,hcpc_code_id\n35391,3,72148\n35391,3,E0601\n",
        encoding="utf-8",
    )

    result = ingest_zip_or_dir(cms_dir, tmp_path / "bronze")
    docs_df = pd.read_parquet(result.documents_path)
    codes_df = pd.read_parquet(result.crosswalks_path)

    assert result.total_documents == 2
    assert set(docs_df["doc_id"]) == {"L35391", "L35392"}
    l35391 = docs_df[docs_df["doc_id"] == "L35391"].iloc[0]
    assert l35391["doc_type"] == "LCD"
    assert l35391["effective_date"] == date(2024, 4, 1)
    assert pd.isna(l35391["retired_date"])
    assert l35391["status"] == "ACTIVE"
    l35392 = docs_df[docs_df["doc_id"] == "L35392"].iloc[0]
    assert pd.isna(l35392["effective_date"])
    assert set(zip(codes_df["code_system"], codes_df["code"], strict=True)) == {
        ("CPT", "72148"),
        ("HCPCS", "E0601"),
    }

    unknown_csv = tmp_path / "unknown.csv"
    unknown_csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No recognized CMS policy"):
        ingest_zip_or_dir(unknown_csv, tmp_path / "bronze_empty")


def test_bronze_partial_bundle_preserves_other_family_crosswalks_and_rejects_garbage_dates(
    tmp_path: Path,
) -> None:
    """Verify LCD crosswalk updates keep NCD codes and trailing garbage dates parse to None."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)

    partial_dir = tmp_path / "partial_bundle"
    partial_dir.mkdir()
    (partial_dir / "lcd.csv").write_text(
        "lcd_id,lcd_version,title,indication,orig_det_eff_date,ret_date,status\n"
        '35399,1,Trailing Garbage Date,"<p>Body</p>",2024-01-15garbage,,A\n',
        encoding="utf-8",
    )
    (partial_dir / "ncd.csv").write_text(
        "ncd_id,ncd_mnl_sect,ncd_vrsn_num,ncd_mnl_sect_title,itm_srvc_desc,ncd_efctv_dt\n"
        '220,220.6,2,PET Scans Updated,"<p>Updated NCD body</p>",2013-06-11\n',
        encoding="utf-8",
    )
    (partial_dir / "lcd_x_hcpc_code.csv").write_text(
        "lcd_id,lcd_version,hcpc_code_id\n35399,1,72148\n",
        encoding="utf-8",
    )

    ingest_zip_or_dir(partial_dir, settings.bronze_dir)
    docs_df = pd.read_parquet(settings.bronze_dir / "raw_documents.parquet")
    codes_df = pd.read_parquet(settings.bronze_dir / "raw_code_crosswalks.parquet")

    l35399 = docs_df[docs_df["doc_id"] == "L35399"].iloc[0]
    assert pd.isna(l35399["effective_date"])

    ncd_codes = codes_df[(codes_df["doc_id"] == "NCD220.6") & (codes_df["version"] == 2)]
    assert set(ncd_codes["code"]) >= {"78815", "78816", "C34.90"}
    l35399_codes = codes_df[(codes_df["doc_id"] == "L35399") & (codes_df["version"] == 1)]
    assert list(l35399_codes["code"]) == ["72148"]
