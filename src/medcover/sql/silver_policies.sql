WITH cpt_agg AS (
    SELECT
        doc_id,
        version,
        LIST(DISTINCT code ORDER BY code) AS cpt_codes
    FROM raw_code_crosswalks
    WHERE code_system IN ('CPT', 'HCPCS')
    GROUP BY
        doc_id,
        version
),

icd10_agg AS (
    SELECT
        doc_id,
        version,
        LIST(DISTINCT code ORDER BY code) AS icd10_codes
    FROM raw_code_crosswalks
    WHERE code_system = 'ICD10'
    GROUP BY
        doc_id,
        version
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
    d.status,
    d.source_mode,
    d.source_sha256,
    COALESCE(c.cpt_codes, CAST([] AS VARCHAR[])) AS cpt_codes,
    COALESCE(i.icd10_codes, CAST([] AS VARCHAR[])) AS icd10_codes,
    (
        d.status = 'ACTIVE'
        AND (d.retired_date IS NULL OR d.retired_date > CURRENT_DATE)
    ) AS is_active
FROM silver_valid_docs AS d
LEFT JOIN cpt_agg AS c
    ON
        d.doc_id = c.doc_id
        AND d.version = c.version
LEFT JOIN icd10_agg AS i
    ON
        d.doc_id = i.doc_id
        AND d.version = i.version
ORDER BY
    d.doc_id,
    d.version
