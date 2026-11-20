"""Thin CLI entrypoint for MedCover ingestion, Medallion build, WAP publish, search, and eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from medcover.bronze import ingest_sample_fixtures, ingest_zip_or_dir
from medcover.config import get_settings
from medcover.gold import build_gold
from medcover.silver import build_silver


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``medcover`` CLI argument parser.

    Inputs: none.
    Output grain: configured ``argparse.ArgumentParser`` with all pipeline subcommands.
    Invariants: exposes ``ingest-zip``, ``sync-api``, ``build``, ``publish``, ``query``, ``eval``.
    """
    parser = argparse.ArgumentParser(
        prog="medcover",
        description="Medicare Coverage Medallion RAG pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_zip = subparsers.add_parser(
        "ingest-zip",
        help="Ingest CMS Medicare Coverage Database bulk ZIP archives into Bronze Parquet.",
    )
    ingest_zip.add_argument("input_path", help="Path to directory or bulk ZIP archive.")

    subparsers.add_parser(
        "sync-api",
        help="Fetch incremental CMS Coverage API v1 updates into Bronze Parquet.",
    )

    build = subparsers.add_parser(
        "build",
        help="Run Silver, Gold, incremental embedding, and DuckDB FTS indexing.",
    )
    build.add_argument(
        "--sample",
        action="store_true",
        help="Build from offline sample fixtures in tests/fixtures/mcd_sample.",
    )

    publish = subparsers.add_parser(
        "publish",
        help="Run Write-Audit-Publish checks and promote a staged release to latest.json.",
    )
    publish.add_argument(
        "--local",
        action="store_true",
        help="Publish to the local releases directory instead of S3.",
    )

    query = subparsers.add_parser(
        "query",
        help="Run hybrid BM25 + vector RRF search against Gold index.duckdb.",
    )
    query.add_argument("text", help="Natural-language coverage question.")
    query.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    query.add_argument(
        "--include-retired",
        action="store_true",
        help="Include retired policies in search results.",
    )

    eval_cmd = subparsers.add_parser(
        "eval",
        help="Run golden retrieval evaluation (Recall@5, MRR, active filter compliance).",
    )
    eval_cmd.add_argument(
        "--sample",
        action="store_true",
        help="Evaluate against isolated sample build artifacts.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI subcommands.

    Inputs: optional argument list (defaults to ``sys.argv[1:]``).
    Output grain: process exit code (0 on success).
    Invariants: keeps CLI wiring thin; stage logic lives in dedicated modules.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "ingest-zip":
        bronze_res = ingest_zip_or_dir(Path(args.input_path), settings.bronze_dir)
        print(
            json.dumps(
                {
                    "documents_path": str(bronze_res.documents_path),
                    "total_documents": bronze_res.total_documents,
                    "total_crosswalks": bronze_res.total_crosswalks,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "build":
        if args.sample:
            settings = settings.model_copy(update={"data_dir": settings.data_dir / "sample"})
            ingest_sample_fixtures(settings)
        silver_res = build_silver(settings)
        gold_res = build_gold(
            settings,
            bronze_docs=silver_res.bronze_docs,
            quarantined_docs=silver_res.quarantined_docs,
        )
        print(json.dumps(gold_res.manifest.model_dump(mode="json"), indent=2))
        return 0

    msg = f"Subcommand '{args.command}' is not implemented yet."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
