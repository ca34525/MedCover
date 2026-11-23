"""Offline tests for hybrid BM25+vector retrieval, golden eval, BYO-key synthesis, and CLI."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from medcover import cli, gold, retrieve
from medcover.bronze import ingest_sample_fixtures
from medcover.config import Settings
from medcover.eval import EvalReport, evaluate_retrieval
from medcover.gold import build_gold
from medcover.quality import audit_medallion, render_manifest_summary
from medcover.retrieve import (
    AskRequest,
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


def test_hybrid_retrieval_filters_as_of_date_codes_type_and_mac_with_ncds(
    tmp_path: Path,
) -> None:
    """Verify RRF search enforces SCD2 valid_to, HCPCS/ICD-10 arrays, doc_type, and MAC."""
    settings = _build_sample_index(tmp_path)
    with HybridRetriever(
        settings.gold_dir / "index.duckdb",
        settings=settings,
        embed_fn=_hashed_bow_embed,
    ) as retriever:
        # Current as_of_date must exclude retired LCD L33381 and superseded L33822 v11.
        current_bariatric = retriever.search(
            RetrieveRequest(
                query="Roux-en-Y gastric bypass bariatric surgery morbid obesity",
                as_of_date=date(2026, 1, 1),
            )
        )
        assert "L33381" not in {chunk.doc_id for chunk in current_bariatric.chunks}
        assert all(
            chunk.effective_date <= date(2026, 1, 1)
            and (chunk.valid_to is None or chunk.valid_to > date(2026, 1, 1))
            for chunk in current_bariatric.chunks
        )

        # Multi-version SCD2 check: 2021-06-01 returns ONLY L33822 v11; 2022-04-01+ returns v12.
        mri_2021 = retriever.search(
            RetrieveRequest(
                query="lumbar spine MRI conservative therapy",
                as_of_date=date(2021, 6, 1),
                hcpcs_code="72148",
                doc_type="LCD",
            )
        )
        assert len(mri_2021.chunks) > 0
        assert {(c.doc_id, c.version) for c in mri_2021.chunks} == {("L33822", 11)}
        assert mri_2021.chunks[0].valid_to == date(2022, 4, 1)

        mri_2022_boundary = retriever.search(
            RetrieveRequest(
                query="lumbar spine MRI conservative therapy",
                as_of_date=date(2022, 4, 1),
                hcpcs_code="72148",
                doc_type="LCD",
            )
        )
        assert {(c.doc_id, c.version) for c in mri_2022_boundary.chunks} == {("L33822", 12)}

        # Point-in-time lookup before retirement recovers L33381 and excludes post-dated LCDs.
        historical_bariatric = retriever.search(
            RetrieveRequest(
                query="Roux-en-Y gastric bypass bariatric surgery 43644",
                as_of_date=date(2023, 6, 1),
                hcpcs_code="43644",
            )
        )
        assert len(historical_bariatric.chunks) > 0
        assert historical_bariatric.chunks[0].doc_id == "L33381"
        assert historical_bariatric.chunks[0].retired_date == date(2024, 6, 30)

        # A date prior to L34998's 2024-01-15 effective_date must exclude L34998.
        before_moldx = retriever.search(
            RetrieveRequest(
                query="MolDX NGS panel EGFR ALK non-small cell lung cancer",
                as_of_date=date(2023, 12, 31),
                hcpcs_code="81445",
            )
        )
        assert before_moldx.chunks == []

        # as_of_date=None disables temporal filtering and returns retired L33381.
        all_dates_bariatric = retriever.search(
            RetrieveRequest(
                query="Roux-en-Y gastric bypass bariatric surgery 43644",
                as_of_date=None,
                hcpcs_code="43644",
            )
        )
        assert len(all_dates_bariatric.chunks) > 0
        assert all_dates_bariatric.chunks[0].doc_id == "L33381"

        # Case-insensitive HCPCS + ICD-10 filter scopes to active lumbar spine MRI pair (v12 only).
        mri_pair = retriever.search(
            RetrieveRequest(
                query="lumbar spine MRI radiculopathy",
                as_of_date=date(2026, 1, 1),
                hcpcs_code="72148",
                icd10_code="m51.16",
            )
        )
        assert {chunk.doc_id for chunk in mri_pair.chunks} == {"L33822", "A56542"}
        assert all(chunk.version == 12 for chunk in mri_pair.chunks if chunk.doc_id == "L33822")

        # doc_type filter separates LCD from companion billing Article.
        mri_lcd_only = retriever.search(
            RetrieveRequest(
                query="lumbar spine MRI radiculopathy",
                as_of_date=date(2026, 1, 1),
                hcpcs_code="72148",
                doc_type="LCD",
            )
        )
        assert {(c.doc_id, c.version) for c in mri_lcd_only.chunks} == {("L33822", 12)}

        # MAC filter returns that MAC's local policies plus nationwide NCDs, never other MACs.
        cataract_mac = retriever.search(
            RetrieveRequest(
                query="cataract surgery functional impairment",
                as_of_date=date(2026, 1, 1),
                mac_jurisdiction="first coast service options (jn)",
            )
        )
        assert cataract_mac.chunks[0].doc_id == "L35041"
        assert all(
            chunk.doc_type == "NCD"
            or (chunk.mac_jurisdiction or "").lower() == "first coast service options (jn)"
            for chunk in cataract_mac.chunks
        )

        # Nationwide NCDs remain discoverable when filtering by a regional MAC.
        pet_in_novitas = retriever.search(
            RetrieveRequest(
                query="FDG PET scans non-small cell lung cancer",
                as_of_date=date(2026, 1, 1),
                mac_jurisdiction="Novitas Solutions (JH/JL)",
            )
        )
        assert pet_in_novitas.chunks[0].doc_id == "NCD220.6"

        # Combining mac_jurisdiction + doc_type="LCD" restricts strictly to that MAC's local LCDs.
        cataract_lcd_only = retriever.search(
            RetrieveRequest(
                query="cataract surgery functional impairment",
                as_of_date=date(2026, 1, 1),
                mac_jurisdiction="First Coast Service Options (JN)",
                doc_type="LCD",
            )
        )
        assert {chunk.doc_id for chunk in cataract_lcd_only.chunks} == {"L35041"}


def test_hybrid_retrieval_handles_zero_vectors_and_retirement_boundary(tmp_path: Path) -> None:
    """Verify NaN cosine distances degrade to BM25 and exact retired_date is excluded."""
    settings = Settings(data_dir=tmp_path)
    ingest_sample_fixtures(settings)
    silver_res = build_silver(settings)
    build_gold(
        settings,
        bronze_docs=silver_res.bronze_docs,
        quarantined_docs=silver_res.quarantined_docs,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )

    # Half-open [effective_date, retired_date): L33381 (retired 2024-06-30) is excluded on that day.
    on_retire_day = retrieve_chunks(
        RetrieveRequest(
            query="Roux-en-Y gastric bypass bariatric surgery",
            as_of_date=date(2024, 6, 30),
        ),
        settings=settings,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )
    assert "L33381" not in {chunk.doc_id for chunk in on_retire_day.chunks}

    day_before_retire = retrieve_chunks(
        RetrieveRequest(
            query="Roux-en-Y gastric bypass bariatric surgery",
            as_of_date=date(2024, 6, 29),
        ),
        settings=settings,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )
    assert "L33381" in {chunk.doc_id for chunk in day_before_retire.chunks}

    mri = retrieve_chunks(
        RetrieveRequest(query="lumbar spine MRI radiculopathy", as_of_date=date(2026, 1, 1)),
        settings=settings,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
    )
    assert len(mri.chunks) > 0
    assert mri.chunks[0].doc_id in {"L33822", "A56542"}
    assert mri.chunks[0].bm25_rank == 1
    assert mri.chunks[0].vec_rank is None


def test_golden_eval_passes_recall_gate_and_writes_separate_report(tmp_path: Path) -> None:
    """Verify golden eval writes eval_report.json without mutating immutable run_manifest.json."""
    settings = _build_sample_index(tmp_path)
    manifest_path = settings.gold_dir / "run_manifest.json"
    manifest_before = manifest_path.read_text(encoding="utf-8")

    report = evaluate_retrieval(
        settings,
        embed_fn=_hashed_bow_embed,
    )

    assert report.passed is True
    assert report.total_queries >= 15
    assert report.recall_at_5 >= settings.min_recall_at_5
    assert report.mrr >= 0.85
    assert report.temporal_filter_violations == 0
    assert report.forbidden_doc_violations == 0

    # run_manifest.json remains byte-for-byte immutable; eval_report.json holds eval metrics.
    assert manifest_path.read_text(encoding="utf-8") == manifest_before
    eval_report_path = settings.gold_dir / "eval_report.json"
    assert eval_report_path.exists()
    persisted = EvalReport.model_validate_json(eval_report_path.read_text(encoding="utf-8"))
    assert persisted.recall_at_5 == report.recall_at_5
    assert persisted.temporal_filter_violations == 0

    audit = audit_medallion(settings)
    assert audit.manifest is not None
    summary = render_manifest_summary(audit.manifest, audit, eval_report=report)
    assert "Temporal Filter Violations" in summary
    assert f"{report.recall_at_5:.3f}" in summary


def test_eval_fails_when_recall_drops_and_custom_index_does_not_overwrite_gold_report(
    tmp_path: Path,
) -> None:
    """Verify eval returns passed=False on missed docs and custom index_path leaves Gold intact."""
    settings = _build_sample_index(tmp_path)
    failing_queries = tmp_path / "failing_queries.json"
    failing_queries.write_text(
        json.dumps(
            [
                {
                    "query_id": "q_missing",
                    "query": "lumbar spine MRI radiculopathy",
                    "expected_doc_ids": ["NCD999.99"],
                    "as_of_date": "2026-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_retrieval(
        settings,
        queries_path=failing_queries,
        index_path=settings.gold_dir / "index.duckdb",
        embed_fn=_hashed_bow_embed,
    )
    assert report.passed is False
    assert report.recall_at_5 == 0.0
    assert report.mrr == 0.0
    assert not (settings.gold_dir / "eval_report.json").exists()


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
        hcpcs_codes=["72148"],
        icd10_codes=["M51.16"],
        mac_jurisdiction="Novitas Solutions (JH/JL)",
        effective_date=date(2022, 4, 1),
        retired_date=None,
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
        hcpcs_codes=["72148"],
        icd10_codes=["M51.16"],
        mac_jurisdiction="Novitas Solutions (JH/JL)",
        effective_date=date(2022, 4, 1),
        retired_date=None,
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
    prompt_body = json.loads(openai_route.calls[0].request.content.decode("utf-8"))
    assert (
        "Effective: 2022-04-01, Valid To: open, HCPCS: 72148"
        in (prompt_body["messages"][1]["content"])
    )
    assert openai_res.citations_verified is True
    assert "provenance" in openai_res.disclaimer
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
    assert anthropic_res.citations_verified is False
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
    assert gemini_res.citations_verified is True

    # Invalid model names (path traversal / query injection) must fail before outbound HTTP calls.
    with pytest.raises(ValueError, match="model must match"):
        AskRequest(query="Lumbar MRI?", provider="gemini", model="../v1/files?key=leak")
    with pytest.raises(ValueError, match="model must match"):
        synthesize_answer(
            "Lumbar MRI criteria?",
            [chunk],
            provider="gemini",
            api_key=secret_key,
            model="../admin/secret",
        )

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

    assert (
        cli.main(
            ["query", "lumbar spine MRI", "--sample", "--hcpcs", "72148", "--as-of", "2026-01-01"]
        )
        == 0
    )
    query_payload = json.loads(capsys.readouterr().out)
    assert "total_hits" not in query_payload
    assert query_payload["as_of_date"] == "2026-01-01"
    assert {c["doc_id"] for c in query_payload["chunks"]} == {"L33822", "A56542"}

    assert (
        cli.main(["query", "bariatric surgery", "--sample", "--hcpcs", "43644", "--all-dates"]) == 0
    )
    all_dates_payload = json.loads(capsys.readouterr().out)
    assert all_dates_payload["as_of_date"] is None
    assert all_dates_payload["chunks"][0]["doc_id"] == "L33381"

    assert cli.main(["eval", "--sample"]) == 0
    eval_payload = json.loads(capsys.readouterr().out)
    assert eval_payload["passed"] is True
    assert eval_payload["temporal_filter_violations"] == 0
