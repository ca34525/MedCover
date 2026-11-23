"""Offline golden retrieval evaluation harness (Recall@5, MRR, active filter compliance)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from medcover.config import Settings
from medcover.contracts import DocType, RunManifest
from medcover.retrieve import EmbedFn, HybridRetriever, RetrieveRequest


class EvalQuery(BaseModel):
    """Single golden Medicare coverage query with expected and forbidden policy IDs."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_doc_ids: list[str] = Field(default_factory=list)
    forbidden_doc_ids: list[str] = Field(default_factory=list)
    active_only: bool = Field(default=True)
    top_k: int = Field(default=5, ge=5, le=50)
    cpt_code: str | None = Field(default=None)
    icd10_code: str | None = Field(default=None)
    doc_type: DocType | None = Field(default=None)
    mac_jurisdiction: str | None = Field(default=None)


class EvalQueryResult(BaseModel):
    """Per-query retrieval metrics and temporal compliance checks."""

    query_id: str
    query: str
    active_only: bool
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    active_filter_violations: int = Field(ge=0)
    forbidden_doc_violations: int = Field(ge=0)


class EvalReport(BaseModel):
    """Aggregate retrieval evaluation metrics and WAP gate pass/fail outcome."""

    passed: bool
    total_queries: int = Field(ge=0)
    scored_queries: int = Field(ge=0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    active_filter_violations: int = Field(ge=0)
    forbidden_doc_violations: int = Field(ge=0)
    min_recall_at_5: float = Field(ge=0.0, le=1.0)
    results: list[EvalQueryResult]


_EVAL_QUERIES_ADAPTER: TypeAdapter[list[EvalQuery]] = TypeAdapter(list[EvalQuery])


def load_eval_queries(queries_path: Path) -> list[EvalQuery]:
    """Load and validate golden Medicare evaluation queries from JSON.

    Inputs: path to JSON array of golden ``EvalQuery`` objects.
    Output grain: non-empty list of validated ``EvalQuery`` records.
    Invariants: fails fast on missing file, invalid JSON, or empty query list.
    """
    if not queries_path.exists():
        msg = f"Evaluation queries file not found: {queries_path}"
        raise FileNotFoundError(msg)
    queries = _EVAL_QUERIES_ADAPTER.validate_json(queries_path.read_text(encoding="utf-8"))
    if not queries:
        msg = f"Evaluation queries file is empty: {queries_path}"
        raise ValueError(msg)
    return queries


def evaluate_retrieval(
    settings: Settings,
    *,
    queries_path: Path | None = None,
    index_path: Path | None = None,
    embed_fn: EmbedFn | None = None,
    update_manifest: bool = True,
) -> EvalReport:
    """Evaluate hybrid retrieval against golden queries and optionally record metrics in manifest.

    Inputs: ``Settings`` plus optional overrides for ``queries_path``, ``index_path``, ``embed_fn``.
    Output grain: single ``EvalReport`` containing aggregate ``Recall@5``, ``MRR``, and violations.
    Invariants: passes strictly when ``recall_at_5 >= settings.min_recall_at_5``,
    ``active_filter_violations == 0``, and ``forbidden_doc_violations == 0``.
    """
    resolved_queries_path = queries_path if queries_path is not None else settings.eval_queries_path
    resolved_index_path = (
        index_path if index_path is not None else settings.gold_dir / "index.duckdb"
    )
    queries = load_eval_queries(resolved_queries_path)
    retriever = HybridRetriever(resolved_index_path, settings=settings, embed_fn=embed_fn)

    today = date.today()
    results: list[EvalQueryResult] = []
    recall_scores: list[float] = []
    rr_scores: list[float] = []
    total_active_violations = 0
    total_forbidden_violations = 0

    for item in queries:
        response = retriever.search(
            RetrieveRequest(
                query=item.query,
                top_k=item.top_k,
                active_only=item.active_only,
                cpt_code=item.cpt_code,
                icd10_code=item.icd10_code,
                doc_type=item.doc_type,
                mac_jurisdiction=item.mac_jurisdiction,
            )
        )
        ranked_doc_ids = _dedupe_doc_ids(chunk.doc_id for chunk in response.chunks)

        active_violations = 0
        if item.active_only:
            active_violations = sum(
                1
                for chunk in response.chunks
                if (not chunk.is_active)
                or (chunk.retired_date is not None and chunk.retired_date <= today)
            )
        forbidden_set = {doc_id.upper() for doc_id in item.forbidden_doc_ids}
        forbidden_violations = sum(
            1 for chunk in response.chunks if chunk.doc_id.upper() in forbidden_set
        )

        total_active_violations += active_violations
        total_forbidden_violations += forbidden_violations

        recall: float | None = None
        rr: float | None = None
        if item.expected_doc_ids:
            expected_set = {doc_id.upper() for doc_id in item.expected_doc_ids}
            top_5_set = {doc_id.upper() for doc_id in ranked_doc_ids[:5]}
            recall = len(expected_set & top_5_set) / len(expected_set)
            rr = _reciprocal_rank(ranked_doc_ids, expected_set)
            recall_scores.append(recall)
            rr_scores.append(rr)

        results.append(
            EvalQueryResult(
                query_id=item.query_id,
                query=item.query,
                active_only=item.active_only,
                expected_doc_ids=item.expected_doc_ids,
                retrieved_doc_ids=ranked_doc_ids,
                recall_at_5=round(recall, 4) if recall is not None else None,
                reciprocal_rank=round(rr, 4) if rr is not None else None,
                active_filter_violations=active_violations,
                forbidden_doc_violations=forbidden_violations,
            )
        )

    scored_queries = len(recall_scores)
    mean_recall = round(sum(recall_scores) / scored_queries, 4) if scored_queries > 0 else 0.0
    mean_mrr = round(sum(rr_scores) / scored_queries, 4) if scored_queries > 0 else 0.0
    passed = (
        scored_queries > 0
        and mean_recall >= settings.min_recall_at_5
        and total_active_violations == 0
        and total_forbidden_violations == 0
    )

    if update_manifest:
        _update_run_manifest(
            manifest_path=settings.gold_dir / "run_manifest.json",
            recall_at_5=mean_recall,
            mrr=mean_mrr,
            active_filter_violations=total_active_violations,
        )

    return EvalReport(
        passed=passed,
        total_queries=len(queries),
        scored_queries=scored_queries,
        recall_at_5=mean_recall,
        mrr=mean_mrr,
        active_filter_violations=total_active_violations,
        forbidden_doc_violations=total_forbidden_violations,
        min_recall_at_5=settings.min_recall_at_5,
        results=results,
    )


def _dedupe_doc_ids(doc_ids: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        key = doc_id.upper()
        if key not in seen:
            seen.add(key)
            ordered.append(doc_id)
    return ordered


def _reciprocal_rank(ranked_doc_ids: list[str], expected_upper: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id.upper() in expected_upper:
            return 1.0 / rank
    return 0.0


def _update_run_manifest(
    *,
    manifest_path: Path,
    recall_at_5: float,
    mrr: float,
    active_filter_violations: int,
) -> None:
    if not manifest_path.exists():
        return
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    updated = manifest.model_copy(
        update={
            "eval_recall_at_5": recall_at_5,
            "eval_mrr": mrr,
            "active_filter_violations": active_filter_violations,
        }
    )
    manifest_path.write_text(
        json.dumps(updated.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
