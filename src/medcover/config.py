"""Typed runtime configuration for the MedCover pipeline and serving layer."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline paths, thresholds, embedding parameters, and S3 release targets."""

    model_config = SettingsConfigDict(
        env_prefix="MEDCOVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    sample_fixtures_dir: Path = Field(default=Path("tests/fixtures/mcd_sample"))
    eval_queries_path: Path = Field(default=Path("tests/eval_queries.json"))

    duckdb_extension_dir: Path | None = Field(default=None)
    fastembed_cache_dir: Path | None = Field(default=None)

    s3_bucket: str | None = Field(default=None)
    s3_prefix: str = Field(default="gold")

    cms_api_base_url: str = Field(default="https://api.coverage.cms.gov")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384, ge=1)
    chunk_token_limit: int = Field(default=450, ge=32)
    chunk_token_overlap: int = Field(default=60, ge=0)

    max_quarantine_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    min_recall_at_5: float = Field(default=0.85, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

    @property
    def bronze_dir(self) -> Path:
        """Directory holding immutable Bronze Parquet tables."""
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        """Directory holding cleaned Silver policies and quarantine records."""
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        """Directory holding Gold chunks, pre-built index.duckdb, and manifest."""
        return self.data_dir / "gold"

    @property
    def releases_dir(self) -> Path:
        """Directory holding staged Write-Audit-Publish release directories."""
        return self.data_dir / "releases"

    def resolved_duckdb_extension_dir(self) -> Path | None:
        """Return explicit or workspace-local DuckDB extension cache directory when available."""
        if self.duckdb_extension_dir is not None:
            return self.duckdb_extension_dir
        local_cache = Path(".cache/duckdb_extensions")
        return local_cache if local_cache.exists() else None

    def resolved_fastembed_cache_dir(self) -> Path | None:
        """Return explicit or workspace-local FastEmbed cache directory when available."""
        if self.fastembed_cache_dir is not None:
            return self.fastembed_cache_dir
        local_cache = Path(".cache/fastembed")
        return local_cache if local_cache.exists() else None


def get_settings() -> Settings:
    """Load pipeline settings from environment variables and defaults.

    Inputs: environment variables prefixed with ``MEDCOVER_`` and optional ``.env``.
    Output grain: single validated ``Settings`` instance.
    Invariants: ``chunk_token_overlap`` is strictly smaller than ``chunk_token_limit``.
    """
    settings = Settings()
    if settings.chunk_token_overlap >= settings.chunk_token_limit:
        msg = (
            f"chunk_token_overlap ({settings.chunk_token_overlap}) must be less than "
            f"chunk_token_limit ({settings.chunk_token_limit})"
        )
        raise ValueError(msg)
    return settings
