WITH code_agg AS (
    SELECT
        doc_id,
        version,
        LIST(DISTINCT code ORDER BY code) FILTER (
            WHERE code_system = 'HCPCS'
        ) AS hcpcs_codes,
        LIST(DISTINCT code ORDER BY code) FILTER (
            WHERE code_system = 'ICD10'
        ) AS icd10_codes
    FROM raw_code_crosswalks
    GROUP BY
        doc_id,
        version
),

versioned_docs AS (
    SELECT
        doc_id,
        doc_type,
        version,
        title,
        markdown_text,
        mac_jurisdiction,
        effective_date,
        retired_date,
        source_sha256,
        LEAD(effective_date) OVER (
            PARTITION BY doc_id
            ORDER BY version ASC
        ) AS next_effective_date
    FROM silver_valid_docs
)

SELECT
    d.doc_id,
    d.doc_type,
    d.version,
    d.title,
    d.markdown_text,
    d.mac_jurisdiction,
    d.effective_date,
    d.retired_date,
    d.source_sha256,
    CASE
        WHEN d.retired_date IS NULL THEN d.next_effective_date
        WHEN d.next_effective_date IS NULL THEN d.retired_date
        ELSE LEAST(d.retired_date, d.next_effective_date)
    END AS valid_to,
    COALESCE(c.hcpcs_codes, CAST([] AS VARCHAR[])) AS hcpcs_codes,
    COALESCE(c.icd10_codes, CAST([] AS VARCHAR[])) AS icd10_codes
FROM versioned_docs AS d
LEFT JOIN code_agg AS c
    ON
        d.doc_id = c.doc_id
        AND d.version = c.version
ORDER BY
    d.doc_id,
    d.version
