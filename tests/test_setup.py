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
    assert settings.max_quarantine_rate == 0.10
    assert settings.min_recall_at_5 == 0.85
    assert settings.bronze_dir == Path("data/bronze")
    assert settings.silver_dir == Path("data/silver")
    assert settings.gold_dir == Path("data/gold")
    assert settings.releases_dir == Path("data/releases")


def test_settings_rejects_overlap_gte_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify invalid chunk overlap configuration fails fast."""
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
