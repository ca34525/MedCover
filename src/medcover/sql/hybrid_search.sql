WITH filtered AS (
    SELECT
        c.chunk_id,
        c.doc_id,
        c.doc_type,
        c.version,
        c.chunk_index,
        c.title,
        c.section_heading,
        c.chunk_text,
        c.token_count,
        c.cpt_codes,
        c.icd10_codes,
        c.mac_jurisdiction,
        c.effective_date,
        c.retired_date,
        c.is_active,
        c.embedding
    FROM chunks AS c
    CROSS JOIN search_params AS p
    WHERE
        (
            NOT p.active_only
            OR (
                c.is_active
                AND (c.retired_date IS NULL OR c.retired_date > CURRENT_DATE)
            )
        )
        AND (p.cpt_code IS NULL OR LIST_CONTAINS(c.cpt_codes, UPPER(p.cpt_code)))
        AND (p.icd10_code IS NULL OR LIST_CONTAINS(c.icd10_codes, UPPER(p.icd10_code)))
        AND (p.doc_type IS NULL OR c.doc_type = UPPER(p.doc_type))
        AND (
            p.mac_jurisdiction IS NULL
            OR LOWER(COALESCE(c.mac_jurisdiction, '')) = LOWER(p.mac_jurisdiction)
        )
),

bm25_scored AS (
    SELECT
        f.chunk_id,
        fts_main_chunks.MATCH_BM25(f.chunk_id, p.query_text) AS bm25_score
    FROM filtered AS f
    CROSS JOIN search_params AS p
),

bm25_ranked AS (
    SELECT
        ranked.chunk_id,
        ranked.bm25_score,
        ranked.bm25_rank
    FROM (
        SELECT
            s.chunk_id,
            s.bm25_score,
            ROW_NUMBER() OVER (ORDER BY s.bm25_score DESC, s.chunk_id ASC) AS bm25_rank
        FROM bm25_scored AS s
        WHERE s.bm25_score IS NOT NULL
    ) AS ranked
    CROSS JOIN search_params AS p
    WHERE ranked.bm25_rank <= p.candidate_limit
),

vec_scored AS (
    SELECT
        f.chunk_id,
        ARRAY_COSINE_DISTANCE(f.embedding, p.query_embedding) AS cosine_distance
    FROM filtered AS f
    CROSS JOIN search_params AS p
),

vec_ranked AS (
    SELECT
        ranked.chunk_id,
        ranked.cosine_distance,
        ranked.vec_rank
    FROM (
        SELECT
            s.chunk_id,
            s.cosine_distance,
            ROW_NUMBER() OVER (ORDER BY s.cosine_distance ASC, s.chunk_id ASC) AS vec_rank
        FROM vec_scored AS s
        WHERE
            s.cosine_distance IS NOT NULL
            AND NOT ISNAN(s.cosine_distance)
    ) AS ranked
    CROSS JOIN search_params AS p
    WHERE ranked.vec_rank <= p.candidate_limit
),

fused AS (
    SELECT
        b.bm25_score,
        b.bm25_rank,
        v.cosine_distance,
        v.vec_rank,
        COALESCE(b.chunk_id, v.chunk_id) AS chunk_id,
        COALESCE(1.0 / (p.rrf_k + b.bm25_rank), 0.0)
        + COALESCE(1.0 / (p.rrf_k + v.vec_rank), 0.0) AS rrf_score
    FROM bm25_ranked AS b
    FULL OUTER JOIN vec_ranked AS v
        ON b.chunk_id = v.chunk_id
    CROSS JOIN search_params AS p
),

final_ranked AS (
    SELECT
        f.chunk_id,
        c.doc_id,
        c.doc_type,
        c.version,
        c.chunk_index,
        c.title,
        c.section_heading,
        c.chunk_text,
        c.token_count,
        c.cpt_codes,
        c.icd10_codes,
        c.mac_jurisdiction,
        c.effective_date,
        c.retired_date,
        c.is_active,
        f.bm25_score,
        f.bm25_rank,
        f.cosine_distance,
        f.vec_rank,
        f.rrf_score,
        ROW_NUMBER() OVER (ORDER BY f.rrf_score DESC, f.chunk_id ASC) AS rrf_rank
    FROM fused AS f
    INNER JOIN filtered AS c
        ON f.chunk_id = c.chunk_id
)

SELECT
    r.chunk_id,
    r.doc_id,
    r.doc_type,
    r.version,
    r.chunk_index,
    r.title,
    r.section_heading,
    r.chunk_text,
    r.token_count,
    r.cpt_codes,
    r.icd10_codes,
    r.mac_jurisdiction,
    r.effective_date,
    r.retired_date,
    r.is_active,
    r.bm25_score,
    r.bm25_rank,
    r.cosine_distance,
    r.vec_rank,
    r.rrf_score
FROM final_ranked AS r
CROSS JOIN search_params AS p
WHERE r.rrf_rank <= p.top_k
ORDER BY
    r.rrf_rank ASC
