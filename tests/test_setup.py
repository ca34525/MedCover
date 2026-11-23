"""Baseline tests verifying package layout, configuration, SQL resource loading, and CLI."""

from importlib.resources import files
from pathlib import Path

import pytest

from medcover import __version__
from medcover.cli import build_parser
from medcover.config import Settings, get_settings


def test_package_version_and_sql_resource_root() -> None:
    """Verify package metadata and importlib.resources access to medcover.sql."""
    assert __version__ == "0.1.0"
    assert files("medcover.sql").is_dir()


def test_settings_defaults_and_medallion_paths() -> None:
    """Verify default configuration matches SPEC.md invariants."""
    settings = get_settings()
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dim == 384
    assert settings.chunk_token_limit == 450
    assert settings.chunk_token_overlap == 60
    assert settings.max_chunk_tokens == 900
    assert settings.duckdb_storage_version == "v1.5.0"
    assert settings.duckdb_memory_limit == "1GB"
    assert settings.duckdb_threads == 2
    assert settings.max_quarantine_rate == 0.10
    assert settings.min_recall_at_5 == 0.85
    assert settings.bronze_dir == Path("data/bronze")
    assert settings.silver_dir == Path("data/silver")
    assert settings.gold_dir == Path("data/gold")
    assert settings.releases_dir == Path("data/releases")


def test_settings_rejects_invalid_chunk_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify invalid chunk overlap or max_chunk_tokens configuration fails fast."""
    with pytest.raises(ValueError, match="chunk_token_overlap"):
        Settings(chunk_token_limit=100, chunk_token_overlap=100)

    with pytest.raises(ValueError, match="max_chunk_tokens"):
        Settings(chunk_token_limit=450, max_chunk_tokens=400)

    monkeypatch.setenv("MEDCOVER_CHUNK_TOKEN_LIMIT", "100")
    monkeypatch.setenv("MEDCOVER_CHUNK_TOKEN_OVERLAP", "100")
    with pytest.raises(ValueError, match="chunk_token_overlap"):
        get_settings()


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify MEDCOVER_-prefixed environment variables override defaults."""
    monkeypatch.setenv("MEDCOVER_DATA_DIR", str(tmp_path))
    settings = Settings()
    assert settings.data_dir == tmp_path
    assert settings.gold_dir == tmp_path / "gold"


def test_cli_exposes_all_spec_subcommands() -> None:
    """Verify CLI parser wires all six SPEC.md commands."""
    parser = build_parser()
    for argv, expected in [
        (["ingest-zip", "sample.zip"], "ingest-zip"),
        (["sync-api"], "sync-api"),
        (["build", "--sample"], "build"),
        (["publish", "--local"], "publish"),
        (["query", "mri lumbar spine"], "query"),
        (["eval"], "eval"),
    ]:
        parsed = parser.parse_args(argv)
        assert parsed.command == expected


def test_cli_build_sample_isolates_from_primary_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify `medcover build --sample` writes to data_dir/sample without touching primary data."""
    from medcover import cli, gold

    primary_bronze = tmp_path / "bronze"
    primary_bronze.mkdir()
    sentinel = primary_bronze / "raw_documents.parquet"
    sentinel.write_bytes(b"existing-production-bronze")

    monkeypatch.setenv("MEDCOVER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        gold,
        "_embed_with_fastembed",
        lambda texts, *, settings: [[0.0] * settings.embedding_dim for _ in texts],
    )

    assert cli.main(["build", "--sample"]) == 0
    assert sentinel.read_bytes() == b"existing-production-bronze"
    assert (tmp_path / "sample" / "gold" / "chunks.parquet").exists()
