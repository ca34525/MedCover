WITH silver_duplicate_grain AS (
    SELECT COUNT(*) AS violation_count
    FROM (
        SELECT
            doc_id,
            version
        FROM silver_policies
        GROUP BY
            doc_id,
            version
        HAVING COUNT(*) > 1
    ) AS dupes
),

gold_duplicate_grain AS (
    SELECT COUNT(*) AS violation_count
    FROM (
        SELECT
            doc_id,
            version,
            chunk_index
        FROM gold_chunks
        GROUP BY
            doc_id,
            version,
            chunk_index
        HAVING COUNT(*) > 1
    ) AS dupes
),

gold_duplicate_chunk_id AS (
    SELECT COUNT(*) AS violation_count
    FROM (
        SELECT chunk_id
        FROM gold_chunks
        GROUP BY chunk_id
        HAVING COUNT(*) > 1
    ) AS dupes
),

bad_embedding_chunks AS (
    SELECT g.chunk_id
    FROM gold_chunks AS g
    CROSS JOIN dq_params AS p
    WHERE
        g.embedding IS NULL
        OR LEN(g.embedding) != p.embedding_dim
    UNION
    SELECT u.chunk_id
    FROM (
        SELECT
            chunk_id,
            UNNEST(CAST(embedding AS FLOAT[])) AS val
        FROM gold_chunks
        WHERE embedding IS NOT NULL
    ) AS u
    WHERE
        u.val IS NULL
        OR ISNAN(u.val)
),

quarantine_stats AS (
    SELECT
        p.max_quarantine_rate,
        (SELECT COUNT(*) FROM silver_policies) AS silver_docs,
        (SELECT COUNT(*) FROM silver_quarantine) AS quarantined_docs
    FROM dq_params AS p
)

SELECT
    'silver_unique_grain' AS check_name,
    violation_count,
    'Duplicate (doc_id, version) rows in silver/policies.parquet' AS detail
FROM silver_duplicate_grain

UNION ALL

SELECT
    'silver_non_empty_markdown' AS check_name,
    COUNT(*) AS violation_count,
    'Empty markdown_text rows in silver/policies.parquet' AS detail
FROM silver_policies
WHERE TRIM(COALESCE(markdown_text, '')) = ''

UNION ALL

SELECT
    'silver_valid_dates' AS check_name,
    COUNT(*) AS violation_count,
    'Rows with retired_date < effective_date in silver/policies.parquet' AS detail
FROM silver_policies
WHERE
    retired_date IS NOT NULL
    AND retired_date < effective_date

UNION ALL

SELECT
    'quarantine_explicit_reason' AS check_name,
    COUNT(*) AS violation_count,
    'Quarantined rows missing explicit failure_reason' AS detail
FROM silver_quarantine
WHERE TRIM(COALESCE(failure_reason, '')) = ''

UNION ALL

SELECT
    'gold_unique_grain' AS check_name,
    violation_count,
    'Duplicate (doc_id, version, chunk_index) rows in gold/chunks.parquet' AS detail
FROM gold_duplicate_grain

UNION ALL

SELECT
    'gold_unique_chunk_id' AS check_name,
    violation_count,
    'Duplicate chunk_id values in gold/chunks.parquet' AS detail
FROM gold_duplicate_chunk_id

UNION ALL

SELECT
    'gold_non_empty_chunk_text' AS check_name,
    COUNT(*) AS violation_count,
    'Empty chunk_text rows in gold/chunks.parquet' AS detail
FROM gold_chunks
WHERE TRIM(COALESCE(chunk_text, '')) = ''

UNION ALL

SELECT
    'gold_bounded_token_count' AS check_name,
    COUNT(*) AS violation_count,
    'Chunk token_count outside (0, max_chunk_tokens]' AS detail
FROM gold_chunks AS g
CROSS JOIN dq_params AS p
WHERE
    g.token_count <= 0
    OR g.token_count > p.max_chunk_tokens

UNION ALL

SELECT
    'gold_valid_embeddings' AS check_name,
    COUNT(*) AS violation_count,
    'Chunks with null, NaN, or wrong-dimension embeddings' AS detail
FROM bad_embedding_chunks

UNION ALL

SELECT
    'gold_no_quarantined_docs' AS check_name,
    COUNT(*) AS violation_count,
    'Gold chunks whose (doc_id, version) appears in silver/quarantine.parquet' AS detail
FROM gold_chunks AS g
INNER JOIN silver_quarantine AS q
    ON
        g.doc_id = q.doc_id
        AND g.version = q.version

UNION ALL

SELECT
    'gold_active_flag_consistency' AS check_name,
    COUNT(*) AS violation_count,
    'Gold chunks marked is_active with retired_date <= CURRENT_DATE or null flag' AS detail
FROM gold_chunks
WHERE
    is_active IS NULL
    OR (
        is_active
        AND retired_date IS NOT NULL
        AND retired_date <= CURRENT_DATE
    )

UNION ALL

SELECT
    'quarantine_rate_within_limit' AS check_name,
    CASE
        WHEN
            (silver_docs + quarantined_docs) > 0
            AND (
                CAST(quarantined_docs AS DOUBLE)
                / CAST(silver_docs + quarantined_docs AS DOUBLE)
            ) > max_quarantine_rate
            THEN 1
        ELSE 0
    END AS violation_count,
    'Quarantine rate exceeds max_quarantine_rate threshold' AS detail
FROM quarantine_stats
