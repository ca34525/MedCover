"""Hybrid DuckDB BM25 + cosine RRF retrieval and stateless BYO-key httpx RAG synthesis."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

import duckdb
import httpx
import pyarrow as pa
from fastembed import TextEmbedding
from pydantic import BaseModel, Field, field_validator

from medcover.config import Settings, get_settings
from medcover.contracts import DocType, parse_optional_str
from medcover.sql import load_sql

EmbedFn = Callable[[Sequence[str]], list[list[float]]]
LLMProvider = Literal["openai", "anthropic", "gemini"]

_DEFAULT_MODELS: dict[LLMProvider, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-2.5-flash",
}

_BRACKET_GROUP_RE = re.compile(r"\[([^\[\]\n]{1,80})\](?!\()")
_POLICY_ID_RE = re.compile(r"^[A-Za-z]+[0-9][A-Za-z0-9._-]*$")

_SYSTEM_PROMPT = (
    "You are a Medicare coverage policy assistant. Answer strictly from the provided "
    "CMS Medicare Coverage Database excerpts. Cite every factual claim inline with "
    "exact bracketed policy doc_id tags from the context (for example [L33822], "
    "[A56542], or [NCD220.6]). Do not cite any policy ID that is not present in the "
    "excerpts. If the excerpts do not answer the question, state that clearly."
)


class RetrieveRequest(BaseModel):
    """Typed search query and metadata filters for hybrid DuckDB retrieval."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    as_of_date: date | None = Field(default_factory=date.today)
    hcpcs_code: str | None = Field(default=None)
    icd10_code: str | None = Field(default=None)
    doc_type: DocType | None = Field(default=None)
    mac_jurisdiction: str | None = Field(default=None)

    @field_validator("query")
    @classmethod
    def _strip_non_empty_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "query must not be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("hcpcs_code", "icd10_code", mode="before")
    @classmethod
    def _normalize_code(cls, value: Any) -> str | None:
        cleaned = parse_optional_str(value)
        return cleaned.upper() if cleaned is not None else None

    @field_validator("doc_type", mode="before")
    @classmethod
    def _normalize_doc_type(cls, value: Any) -> str | None:
        cleaned = parse_optional_str(value)
        return cleaned.upper() if cleaned is not None else None

    @field_validator("mac_jurisdiction", mode="before")
    @classmethod
    def _normalize_mac(cls, value: Any) -> str | None:
        return parse_optional_str(value)


class RetrievedChunk(BaseModel):
    """Single Gold policy chunk ranked by hybrid BM25 + cosine Reciprocal Rank Fusion."""

    chunk_id: str
    doc_id: str
    doc_type: DocType
    version: int
    chunk_index: int
    title: str
    section_heading: str
    chunk_text: str
    token_count: int
    hcpcs_codes: list[str]
    icd10_codes: list[str]
    mac_jurisdiction: str | None
    effective_date: date
    retired_date: date | None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    cosine_distance: float | None = None
    vec_rank: int | None = None
    rrf_score: float


class RetrieveResponse(BaseModel):
    """Ranked hybrid retrieval result set with filter provenance."""

    query: str
    as_of_date: date | None
    chunks: list[RetrievedChunk]


class AskRequest(RetrieveRequest):
    """Request payload for stateless BYO-key RAG synthesis over retrieved policies."""

    provider: LLMProvider = Field(default="openai")
    model: str | None = Field(default=None)


class AskResponse(BaseModel):
    """Grounded RAG answer with deterministic citation verification metadata."""

    query: str
    provider: LLMProvider
    model: str
    answer: str
    cited_doc_ids: list[str]
    ungrounded_doc_ids: list[str]
    grounding_verified: bool
    chunks: list[RetrievedChunk]


class HybridRetriever:
    """Read-only DuckDB hybrid BM25 + vector RRF retriever over Gold ``index.duckdb``."""

    def __init__(
        self,
        index_path: Path,
        *,
        settings: Settings | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        """Initialize a read-only retriever bound to a pre-built Gold ``index.duckdb``.

        Inputs: path to pre-indexed ``index.duckdb``, optional ``Settings``, optional ``embed_fn``.
        Output grain: reusable retriever instance caching only the ONNX embedding model in memory.
        Invariants: never mutates ``index.duckdb`` and never rebuilds FTS indexes at query time.
        """
        if not index_path.exists():
            msg = f"Gold DuckDB index not found: {index_path}"
            raise FileNotFoundError(msg)
        self._index_path = index_path
        self._settings = settings if settings is not None else get_settings()
        self._embed_fn = embed_fn
        self._model: TextEmbedding | None = None
        self._sql = load_sql("hybrid_search.sql")

    def search(self, request: RetrieveRequest) -> RetrieveResponse:
        """Run metadata-filtered BM25 + cosine RRF retrieval against read-only ``index.duckdb``.

        Inputs: validated ``RetrieveRequest`` containing query text, ``top_k``, and filters.
        Output grain: ``RetrieveResponse`` with up to ``top_k`` ranked ``RetrievedChunk`` rows.
        Invariants: opens DuckDB with ``read_only=True``; enforces
        ``[effective_date, retired_date)`` containment whenever ``as_of_date`` is provided.
        """
        query_embedding = self._embed_query(request.query)
        candidate_limit = max(request.top_k * 4, 25)
        params_table = _build_search_params_table(
            request=request,
            query_embedding=query_embedding,
            embedding_dim=self._settings.embedding_dim,
            rrf_k=self._settings.rrf_k,
            candidate_limit=candidate_limit,
        )

        config: dict[str, str | bool | int | float | list[str]] = {}
        ext_dir = self._settings.resolved_duckdb_extension_dir()
        if ext_dir is not None:
            config["extension_directory"] = str(ext_dir.resolve())

        with duckdb.connect(str(self._index_path), read_only=True, config=config) as con:
            con.execute("LOAD fts")
            con.register("search_params", params_table)
            rows = con.sql(self._sql).to_arrow_table().to_pylist()

        chunks = [_row_to_chunk(row) for row in rows]
        return RetrieveResponse(
            query=request.query,
            as_of_date=request.as_of_date,
            chunks=chunks,
        )

    def _embed_query(self, query: str) -> list[float]:
        if self._embed_fn is not None:
            vectors = self._embed_fn([query])
        else:
            if self._model is None:
                cache_dir = self._settings.resolved_fastembed_cache_dir()
                self._model = TextEmbedding(
                    model_name=self._settings.embedding_model,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                )
            vectors = [list(vec.tolist()) for vec in self._model.embed([query])]

        if len(vectors) != 1 or len(vectors[0]) != self._settings.embedding_dim:
            msg = f"Query embedder must return 1 vector of dim {self._settings.embedding_dim}"
            raise ValueError(msg)
        return [float(val) for val in vectors[0]]


def retrieve_chunks(
    request: RetrieveRequest,
    *,
    settings: Settings | None = None,
    index_path: Path | None = None,
    embed_fn: EmbedFn | None = None,
) -> RetrieveResponse:
    """Execute a single hybrid retrieval query against Gold ``index.duckdb``.

    Inputs: ``RetrieveRequest`` plus optional ``Settings``, explicit ``index_path``, and embedder.
    Output grain: ``RetrieveResponse`` ordered by descending RRF score.
    Invariants: opens ``index.duckdb`` strictly read-only.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_index = (
        index_path if index_path is not None else resolved_settings.gold_dir / "index.duckdb"
    )
    retriever = HybridRetriever(resolved_index, settings=resolved_settings, embed_fn=embed_fn)
    return retriever.search(request)


def verify_citations(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> tuple[list[str], list[str], bool]:
    """Verify every bracketed policy citation in an LLM answer against retrieved ``doc_id`` values.

    Inputs: generated answer string and sequence of retrieved chunks provided as context.
    Output grain: ``(cited_doc_ids, ungrounded_doc_ids, grounding_verified)`` tuple.
    Invariants: deduplicates cited IDs in appearance order; requires zero ungrounded IDs and
    at least one grounded citation whenever ``retrieved_chunks`` is non-empty.
    """
    allowed = {chunk.doc_id.upper(): chunk.doc_id for chunk in retrieved_chunks}
    cited: list[str] = []
    seen: set[str] = set()

    for match in _BRACKET_GROUP_RE.finditer(answer):
        for raw_token in re.split(r"[,;]", match.group(1)):
            token = raw_token.strip()
            if not token or not _POLICY_ID_RE.match(token):
                continue
            key = token.upper()
            if key not in seen:
                seen.add(key)
                cited.append(allowed.get(key, token))

    ungrounded = [doc_id for doc_id in cited if doc_id.upper() not in allowed]
    if not retrieved_chunks:
        verified = len(ungrounded) == 0
    else:
        verified = bool(cited) and len(ungrounded) == 0
    return cited, ungrounded, verified


def synthesize_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    provider: LLMProvider,
    api_key: str,
    model: str | None = None,
    http_client: httpx.Client | None = None,
    timeout_seconds: float = 30.0,
) -> AskResponse:
    """Synthesize a grounded Medicare coverage answer via stateless BYO-key ``httpx`` request.

    Inputs: question, retrieved chunks, LLM provider, in-memory BYO API key, and optional model.
    Output grain: ``AskResponse`` with deterministic ``grounding_verified`` citation status.
    Invariants: ``api_key`` stays strictly in memory on the outbound request headers and is never
    logged, cached, persisted, or echoed in exceptions or responses.
    """
    cleaned_key = api_key.strip()
    if not cleaned_key:
        msg = "X-LLM-API-Key header must be non-empty"
        raise ValueError(msg)

    resolved_model = parse_optional_str(model) or _DEFAULT_MODELS[provider]
    user_prompt = _build_user_prompt(query, chunks)
    url, headers, payload = _build_provider_request(
        provider=provider,
        model=resolved_model,
        api_key=cleaned_key,
        user_prompt=user_prompt,
    )

    try:
        if http_client is not None:
            response = http_client.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        else:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        msg = f"LLM provider '{provider}' transport error: {type(exc).__name__}"
        raise RuntimeError(msg) from None

    if response.status_code >= 400:
        msg = f"LLM provider '{provider}' returned HTTP {response.status_code}"
        raise RuntimeError(msg)

    answer = _extract_provider_text(provider, response.json())
    cited, ungrounded, verified = verify_citations(answer, chunks)
    return AskResponse(
        query=query,
        provider=provider,
        model=resolved_model,
        answer=answer,
        cited_doc_ids=cited,
        ungrounded_doc_ids=ungrounded,
        grounding_verified=verified,
        chunks=list(chunks),
    )


def _build_search_params_table(
    *,
    request: RetrieveRequest,
    query_embedding: Sequence[float],
    embedding_dim: int,
    rrf_k: int,
    candidate_limit: int,
) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("query_text", pa.string(), nullable=False),
            pa.field("query_embedding", pa.list_(pa.float32(), embedding_dim), nullable=False),
            pa.field("as_of_date", pa.date32(), nullable=True),
            pa.field("hcpcs_code", pa.string(), nullable=True),
            pa.field("icd10_code", pa.string(), nullable=True),
            pa.field("doc_type", pa.string(), nullable=True),
            pa.field("mac_jurisdiction", pa.string(), nullable=True),
            pa.field("rrf_k", pa.int64(), nullable=False),
            pa.field("candidate_limit", pa.int64(), nullable=False),
            pa.field("top_k", pa.int64(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(
        [
            {
                "query_text": request.query,
                "query_embedding": list(query_embedding),
                "as_of_date": request.as_of_date,
                "hcpcs_code": request.hcpcs_code,
                "icd10_code": request.icd10_code,
                "doc_type": request.doc_type,
                "mac_jurisdiction": request.mac_jurisdiction,
                "rrf_k": rrf_k,
                "candidate_limit": candidate_limit,
                "top_k": request.top_k,
            }
        ],
        schema=schema,
    )


def _row_to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        doc_type=row["doc_type"],
        version=int(row["version"]),
        chunk_index=int(row["chunk_index"]),
        title=str(row["title"]),
        section_heading=str(row["section_heading"]),
        chunk_text=str(row["chunk_text"]),
        token_count=int(row["token_count"]),
        hcpcs_codes=[str(code) for code in (row["hcpcs_codes"] or ())],
        icd10_codes=[str(code) for code in (row["icd10_codes"] or ())],
        mac_jurisdiction=parse_optional_str(row["mac_jurisdiction"]),
        effective_date=row["effective_date"],
        retired_date=row["retired_date"],
        bm25_score=float(row["bm25_score"]) if row["bm25_score"] is not None else None,
        bm25_rank=int(row["bm25_rank"]) if row["bm25_rank"] is not None else None,
        cosine_distance=(
            float(row["cosine_distance"]) if row["cosine_distance"] is not None else None
        ),
        vec_rank=int(row["vec_rank"]) if row["vec_rank"] is not None else None,
        rrf_score=float(row["rrf_score"]),
    )


def _build_user_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return f"Question: {query}\n\nRetrieved Context: (no matching policy chunks)"

    blocks: list[str] = []
    for chunk in chunks:
        codes = ", ".join(chunk.hcpcs_codes[:10]) if chunk.hcpcs_codes else "none"
        retired = chunk.retired_date.isoformat() if chunk.retired_date is not None else "none"
        blocks.append(
            f"[{chunk.doc_id}] ({chunk.doc_type} v{chunk.version}, "
            f"Effective: {chunk.effective_date.isoformat()}, Retired: {retired}, HCPCS: {codes})\n"
            f"Title: {chunk.title}\n"
            f"Section: {chunk.section_heading}\n"
            f"{chunk.chunk_text}"
        )
    context = "\n\n---\n\n".join(blocks)
    return f"Question: {query}\n\nRetrieved Context:\n{context}"


def _build_provider_request(
    *,
    provider: LLMProvider,
    model: str,
    api_key: str,
    user_prompt: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    if provider == "openai":
        return (
            "https://api.openai.com/v1/chat/completions",
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            {
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
    if provider == "anthropic":
        return (
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0.0,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        {
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.0},
        },
    )


def _extract_provider_text(provider: LLMProvider, body: Any) -> str:
    if not isinstance(body, dict):
        msg = f"Unexpected non-object JSON payload from LLM provider '{provider}'"
        raise RuntimeError(msg)

    if provider == "openai":
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    elif provider == "anthropic":
        content_blocks = body.get("content")
        if isinstance(content_blocks, list):
            parts = [
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "\n".join(part for part in parts if part).strip()
            if joined:
                return joined
    else:
        candidates = body.get("candidates")
        if isinstance(candidates, list) and candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if isinstance(parts, list):
                joined = "\n".join(
                    str(part.get("text", "")) for part in parts if isinstance(part, dict)
                ).strip()
                if joined:
                    return joined

    msg = f"LLM provider '{provider}' response contained no answer text"
    raise RuntimeError(msg)
