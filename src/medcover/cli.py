"""Thin CLI entrypoint for MedCover ingestion, Medallion build, WAP publish, search, and eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from medcover.bronze import ingest_sample_fixtures, ingest_zip_or_dir
from medcover.config import get_settings
from medcover.eval import evaluate_retrieval
from medcover.gold import build_gold
from medcover.retrieve import RetrieveRequest, retrieve_chunks
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
    temporal_group = query.add_mutually_exclusive_group()
    temporal_group.add_argument(
        "--as-of",
        dest="as_of_date",
        type=date.fromisoformat,
        default=None,
        help="Evaluate policy validity as of YYYY-MM-DD (defaults to today).",
    )
    temporal_group.add_argument(
        "--all-dates",
        action="store_true",
        help="Disable temporal filtering and search all active and historical policies.",
    )
    query.add_argument(
        "--hcpcs",
        dest="hcpcs_code",
        default=None,
        help="Filter by HCPCS/CPT procedure code.",
    )
    query.add_argument("--icd10", dest="icd10_code", default=None, help="Filter by ICD-10 code.")
    query.add_argument(
        "--doc-type",
        dest="doc_type",
        default=None,
        help="Filter by document type (NCD, LCD, or ARTICLE).",
    )
    query.add_argument(
        "--mac",
        dest="mac_jurisdiction",
        default=None,
        help="Filter by MAC jurisdiction (includes nationwide NCDs).",
    )
    query.add_argument(
        "--sample",
        action="store_true",
        help="Query isolated sample build artifacts.",
    )

    eval_cmd = subparsers.add_parser(
        "eval",
        help="Run golden retrieval evaluation (Recall@5, MRR, temporal filter compliance).",
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
    Output grain: process exit code (0 on success, non-zero on gate failure).
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

    if args.command == "query":
        if args.sample:
            settings = settings.model_copy(update={"data_dir": settings.data_dir / "sample"})
        resolved_as_of: date | None = (
            None
            if args.all_dates
            else (args.as_of_date if args.as_of_date is not None else date.today())
        )
        request = RetrieveRequest(
            query=args.text,
            top_k=args.top_k,
            as_of_date=resolved_as_of,
            hcpcs_code=args.hcpcs_code,
            icd10_code=args.icd10_code,
            doc_type=args.doc_type,
            mac_jurisdiction=args.mac_jurisdiction,
        )
        response = retrieve_chunks(request, settings=settings)
        print(json.dumps(response.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "eval":
        if args.sample:
            settings = settings.model_copy(update={"data_dir": settings.data_dir / "sample"})
        report = evaluate_retrieval(settings)
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0 if report.passed else 1

    msg = f"Subcommand '{args.command}' is not implemented yet."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
