"""Offline tests for hybrid BM25+vector retrieval, golden eval, BYO-key synthesis, and CLI."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb
import httpx
import pytest
import respx

from medcover import cli, gold, retrieve
from medcover.bronze import ingest_sample_fixtures
from medcover.config import Settings
from medcover.contracts import RunManifest
from medcover.eval import evaluate_retrieval
from medcover.gold import build_gold
from medcover.retrieve import (
    HybridRetriever,
    RetrievedChunk,
    RetrieveRequest,
    retrieve_chunks,
    synthesize_answer,
    verify_citations,
)
from medcover.silver import build_silver

_WORD_RE = re.compile(r"[a-z0-9]+")


def _hashed_bow_embed(texts: Sequence[str], dim: int = 384) -> list[list[float]]:
    """Fast deterministic 384-dim hashed bag-of-words embedder for offline retrieval tests."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in _WORD_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(val * val for val in vec))
        if norm > 0.0:
            vec = [val / norm for val in vec]
        vectors.append(vec)
    return vectors


def _build_sample_index(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)
    build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=_hashed_bow_embed,
    )
    return settings


def test_hybrid_retrieval_filters_active_codes_type_and_mac(tmp_path: Path) -> None:
    """Verify RRF search enforces active_only, CPT/ICD-10 arrays, doc_type, and MAC filters."""
    settings = _build_sample_index(tmp_path)
    retriever = HybridRetriever(
        settings.gold_dir / "index.duckdb",
        settings=settings,
        embed_fn=_hashed_bow_embed,
    )

    # Active-only search must never leak retired LCD L33381.
    active_bariatric = retriever.search(
        RetrieveRequest(
            query="Roux-en-Y gastric bypass bariatric surgery morbid obesity",
            active_only=True,
        )
    )
    assert all(chunk.is_active for chunk in active_bariatric.chunks)
    assert "L33381" not in {chunk.doc_id for chunk in active_bariatric.chunks}

    # Explicit historical lookup with active_only=False returns retired L33381.
    retired_bariatric = retriever.search(
        RetrieveRequest(
            query="Roux-en-Y gastric bypass bariatric surgery 43644",
            active_only=False,
            cpt_code="43644",
        )
    )
    assert retired_bariatric.total_hits > 0
    assert retired_bariatric.chunks[0].doc_id == "L33381"
    assert retired_bariatric.chunks[0].is_active is False

    # Case-insensitive CPT + ICD-10 filter scopes to lumbar spine MRI pair.
    mri_pair = retriever.search(
        RetrieveRequest(
            query="lumbar spine MRI radiculopathy",
            cpt_code="72148",
            icd10_code="m51.16",
        )
    )
    assert {chunk.doc_id for chunk in mri_pair.chunks} == {"L33822", "A56542"}

    # doc_type filter separates LCD from companion billing Article.
    mri_lcd_only = retriever.search(
        RetrieveRequest(
            query="lumbar spine MRI radiculopathy",
            cpt_code="72148",
            doc_type="LCD",
        )
    )
    assert {chunk.doc_id for chunk in mri_lcd_only.chunks} == {"L33822"}

    # Case-insensitive MAC jurisdiction filter scopes to First Coast cataract LCD.
    cataract_mac = retriever.search(
        RetrieveRequest(
            query="cataract surgery functional impairment",
            mac_jurisdiction="first coast service options (jn)",
        )
    )
    assert {chunk.doc_id for chunk in cataract_mac.chunks} == {"L35041"}


def test_hybrid_retrieval_handles_zero_vectors_and_runtime_retired_date(tmp_path: Path) -> None:
    """Verify NaN cosine distances degrade to BM25 and past retired_date is excluded at runtime."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)
    gold_res = build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )

    # Simulate an index whose is_active flag went stale after a policy's retired_date passed.
    config: dict[str, str] = {}
    ext_dir = settings.resolved_duckdb_extension_dir()
    if ext_dir is not None:
        config["extension_directory"] = str(ext_dir.resolve())
    with duckdb.connect(str(gold_res.index_path), config=config) as con:
        con.execute("UPDATE chunks SET is_active = TRUE WHERE doc_id = 'L33381'")

    res = retrieve_chunks(
        RetrieveRequest(query="Roux-en-Y gastric bypass bariatric surgery", active_only=True),
        settings=settings,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )
    assert "L33381" not in {chunk.doc_id for chunk in res.chunks}

    mri = retrieve_chunks(
        RetrieveRequest(query="lumbar spine MRI radiculopathy"),
        settings=settings,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )
    assert mri.total_hits > 0
    assert mri.chunks[0].doc_id in {"L33822", "A56542"}
    assert mri.chunks[0].bm25_rank == 1
    assert mri.chunks[0].vec_rank is None


def test_golden_eval_passes_recall_gate_and_updates_manifest(tmp_path: Path) -> None:
    """Verify golden eval achieves Recall@5 >= 0.85, 0 active leaks, and updates run_manifest."""
    settings = _build_sample_index(tmp_path)
    report = evaluate_retrieval(
        settings,
        embed_fn=_hashed_bow_embed,
        update_manifest=True,
    )

    assert report.passed is True
    assert report.total_queries >= 15
    assert report.recall_at_5 >= settings.min_recall_at_5
    assert report.mrr >= 0.85
    assert report.active_filter_violations == 0
    assert report.forbidden_doc_violations == 0

    manifest_path = settings.gold_dir / "run_manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.eval_recall_at_5 == report.recall_at_5
    assert manifest.eval_mrr == report.mrr
    assert manifest.active_filter_violations == 0


def test_eval_fails_when_recall_drops_below_gate(tmp_path: Path) -> None:
    """Verify evaluation returns passed=False when expected documents are missing."""
    settings = _build_sample_index(tmp_path)
    failing_queries = tmp_path / "failing_queries.json"
    failing_queries.write_text(
        json.dumps(
            [
                {
                    "query_id": "q_missing",
                    "query": "lumbar spine MRI radiculopathy",
                    "expected_doc_ids": ["NCD999.99"],
                    "active_only": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_retrieval(
        settings,
        queries_path=failing_queries,
        embed_fn=_hashed_bow_embed,
        update_manifest=False,
    )
    assert report.passed is False
    assert report.recall_at_5 == 0.0
    assert report.mrr == 0.0


def test_verify_citations_detects_grounded_and_hallucinated_ids() -> None:
    """Verify deterministic citation verification accepts retrieved IDs and rejects fabrications."""
    chunk = RetrievedChunk(
        chunk_id="L33822_v12_c0000",
        doc_id="L33822",
        doc_type="LCD",
        version=12,
        chunk_index=0,
        title="Lumbar Spine MRI",
        section_heading="Coverage Indications",
        chunk_text="Requires 6 weeks of conservative therapy.",
        token_count=7,
        cpt_codes=["72148"],
        icd10_codes=["M51.16"],
        mac_jurisdiction="Novitas Solutions (JH/JL)",
        effective_date=date(2022, 4, 1),
        retired_date=None,
        is_active=True,
        rrf_score=0.032,
    )

    cited, ungrounded, verified = verify_citations(
        "Lumbar MRI requires 6 weeks conservative therapy [L33822] (note [sic]).",
        [chunk],
    )
    assert cited == ["L33822"]
    assert ungrounded == []
    assert verified is True

    cited_bad, ungrounded_bad, verified_bad = verify_citations(
        "Covered under [L33822] and fabricated [L99999].",
        [chunk],
    )
    assert cited_bad == ["L33822", "L99999"]
    assert ungrounded_bad == ["L99999"]
    assert verified_bad is False

    cited_none, ungrounded_none, verified_none = verify_citations(
        "Lumbar MRI requires 6 weeks of physical therapy.",
        [chunk],
    )
    assert cited_none == []
    assert ungrounded_none == []
    assert verified_none is False


@respx.mock
def test_byo_key_synthesis_providers_and_zero_key_leakage() -> None:
    """Verify OpenAI, Anthropic, and Gemini httpx calls verify citations and never leak API keys."""
    chunk = RetrievedChunk(
        chunk_id="L33822_v12_c0000",
        doc_id="L33822",
        doc_type="LCD",
        version=12,
        chunk_index=0,
        title="Lumbar Spine MRI",
        section_heading="Coverage Indications",
        chunk_text="Requires 6 weeks of conservative therapy unless red flags exist.",
        token_count=10,
        cpt_codes=["72148"],
        icd10_codes=["M51.16"],
        mac_jurisdiction="Novitas Solutions (JH/JL)",
        effective_date=date(2022, 4, 1),
        retired_date=None,
        is_active=True,
        rrf_score=0.032,
    )
    secret_key = "sk-super-secret-byo-test-key"

    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Covered after 6 weeks conservative therapy [L33822]."}}
                ]
            },
        )
    )
    openai_res = synthesize_answer(
        "Lumbar MRI criteria?",
        [chunk],
        provider="openai",
        api_key=secret_key,
    )
    assert openai_route.called
    assert openai_route.calls[0].request.headers["authorization"] == f"Bearer {secret_key}"
    assert openai_res.grounding_verified is True
    assert openai_res.cited_doc_ids == ["L33822"]
    assert secret_key not in openai_res.model_dump_json()

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "See [L33822] and [FAKE999]."}]},
        )
    )
    anthropic_res = synthesize_answer(
        "Lumbar MRI criteria?",
        [chunk],
        provider="anthropic",
        api_key=secret_key,
    )
    assert anthropic_route.called
    assert anthropic_route.calls[0].request.headers["x-api-key"] == secret_key
    assert anthropic_res.grounding_verified is False
    assert anthropic_res.ungrounded_doc_ids == ["FAKE999"]

    gemini_route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Covered [L33822]."}]}}]},
        )
    )
    gemini_res = synthesize_answer(
        "Lumbar MRI criteria?",
        [chunk],
        provider="gemini",
        api_key=secret_key,
    )
    assert gemini_route.called
    assert gemini_route.calls[0].request.headers["x-goog-api-key"] == secret_key
    assert secret_key not in str(gemini_route.calls[0].request.url)
    assert gemini_res.grounding_verified is True

    # Upstream 401 failure must never echo the API key in the exception message.
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": f"Invalid key {secret_key}"}})
    )
    with pytest.raises(RuntimeError) as exc_info:
        synthesize_answer("Lumbar MRI criteria?", [chunk], provider="openai", api_key=secret_key)
    assert secret_key not in str(exc_info.value)


def test_cli_query_and_eval_sample_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `medcover build --sample`, `query --sample`, and `eval --sample` end-to-end."""
    monkeypatch.setenv("MEDCOVER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        gold,
        "_embed_with_fastembed",
        lambda texts, *, settings: _hashed_bow_embed(texts, dim=settings.embedding_dim),
    )
    monkeypatch.setattr(
        retrieve.HybridRetriever,
        "_embed_query",
        lambda self, query: _hashed_bow_embed([query], dim=self._settings.embedding_dim)[0],
    )

    assert cli.main(["build", "--sample"]) == 0
    capsys.readouterr()

    assert cli.main(["query", "lumbar spine MRI", "--sample", "--cpt", "72148"]) == 0
    query_payload = json.loads(capsys.readouterr().out)
    assert query_payload["total_hits"] > 0
    assert {c["doc_id"] for c in query_payload["chunks"]} == {"L33822", "A56542"}

    assert cli.main(["eval", "--sample"]) == 0
    eval_payload = json.loads(capsys.readouterr().out)
    assert eval_payload["passed"] is True
    assert eval_payload["active_filter_violations"] == 0
