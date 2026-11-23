CREATE TABLE chunks AS
SELECT
    chunk_id,
    doc_id,
    doc_type,
    version,
    chunk_index,
    title,
    section_heading,
    chunk_text,
    token_count,
    chunk_sha256,
    hcpcs_codes,
    icd10_codes,
    mac_jurisdiction,
    effective_date,
    retired_date,
    valid_to,
    embedding,
    embedding_model
FROM gold_chunks_input;

CREATE OR REPLACE MACRO CHUNKS_AS_OF(AS_OF_DATE) AS TABLE
    SELECT
        chunk_id,
        doc_id,
        doc_type,
        version,
        chunk_index,
        title,
        section_heading,
        chunk_text,
        token_count,
        chunk_sha256,
        hcpcs_codes,
        icd10_codes,
        mac_jurisdiction,
        effective_date,
        retired_date,
        valid_to,
        embedding,
        embedding_model
    FROM chunks
    WHERE
        effective_date <= as_of_date
        AND (valid_to IS NULL OR valid_to > as_of_date);

CREATE OR REPLACE VIEW active_chunks AS
SELECT
    chunk_id,
    doc_id,
    doc_type,
    version,
    chunk_index,
    title,
    section_heading,
    chunk_text,
    token_count,
    chunk_sha256,
    hcpcs_codes,
    icd10_codes,
    mac_jurisdiction,
    effective_date,
    retired_date,
    valid_to,
    embedding,
    embedding_model
FROM CHUNKS_AS_OF(CURRENT_DATE);
