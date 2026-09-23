SELECT 'documents' AS metric, count(*)::text AS value FROM documents
UNION ALL
SELECT 'chunks', count(*)::text FROM chunks
UNION ALL
SELECT 'retrieval_nodes', count(*)::text FROM retrieval_nodes
UNION ALL
SELECT 'query_logs', count(*)::text FROM query_logs;

SELECT
    count(*) AS total_chunks,
    count(embedding) AS chunks_with_embedding
FROM chunks;

SELECT
    vector_dims(embedding) AS dimensions,
    count(*) AS chunks
FROM chunks
WHERE embedding IS NOT NULL
GROUP BY vector_dims(embedding);

SELECT
    ingestion_status,
    count(*) AS documents
FROM documents
GROUP BY ingestion_status
ORDER BY ingestion_status;

SELECT
    extname,
    extversion
FROM pg_extension
ORDER BY extname;

SELECT version_num
FROM alembic_version;
