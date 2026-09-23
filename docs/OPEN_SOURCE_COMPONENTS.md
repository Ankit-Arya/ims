# Open-source components — 0.5.0 additions

0.5.0 adds PyMuPDF for lazy source-page/crop rendering. ONNX Runtime, OpenVINO and Optimum are optional inference-image dependencies used only when explicitly building an optimized reranker benchmark/runtime; PyTorch remains the default until evaluation passes.

---

## Historical / earlier-release notes


This project is designed so that the application framework, retrieval stack, document processing,
embeddings, reranking, database, queue, OCR, and local inference runtime can be self-hosted with
open-source software. The only intentionally metered external dependency in the default configuration
is the configured LLM API.

> **Important:** this file is an engineering inventory, not legal advice. Before organization-wide
> deployment, generate an SBOM from the built images and have your organization review the licenses
> of the exact resolved versions and all transitive dependencies/model artifacts.

## Primary components

| Component | Purpose | Default license / status | Project source |
|---|---|---|---|
| FastAPI | HTTP API and web backend | MIT | https://github.com/fastapi/fastapi |
| LangGraph | Stateful workflow orchestration | MIT | https://github.com/langchain-ai/langgraph |
| PostgreSQL | Relational database and lexical search | PostgreSQL License | https://www.postgresql.org/ |
| pgvector | Vector similarity search in PostgreSQL | PostgreSQL-style license | https://github.com/pgvector/pgvector |
| SQLAlchemy | ORM / SQL toolkit | MIT | https://github.com/sqlalchemy/sqlalchemy |
| Alembic | Database migrations | MIT | https://github.com/sqlalchemy/alembic |
| Celery | Background task processing | BSD-3-Clause | https://github.com/celery/celery |
| Valkey | Queue broker | BSD-3-Clause | https://github.com/valkey-io/valkey |
| Docling | PDF/document understanding | MIT codebase | https://github.com/docling-project/docling |
| Docling layout/TableFormer artifacts | layout/table understanding | permissive model terms documented by upstream (Apache-2.0 / CDLA-Permissive-2.0 depending artifact) | https://huggingface.co/docling-project/docling-models |
| Tesseract | OCR fallback | Apache-2.0 | https://github.com/tesseract-ocr/tesseract |
| Sentence Transformers | Local embeddings/reranking runtime | Apache-2.0 | https://github.com/huggingface/sentence-transformers |
| PyTorch | Local ML runtime | Open source; main project BSD-3-Clause with bundled third-party notices | https://github.com/pytorch/pytorch |
| BAAI/bge-m3 | Default embedding model | MIT (model card) | https://huggingface.co/BAAI/bge-m3 |
| BAAI/bge-reranker-v2-m3 | Default reranker | Apache-2.0 (model card) | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| OpenAI Python SDK | Default LLM API client | Apache-2.0 | https://github.com/openai/openai-python |

## What is intentionally not required

The default architecture does **not** require paid LangSmith, a hosted vector database, a hosted
embedding API, a hosted reranking API, a managed OCR API, or a commercial message broker.

The default local services perform:

- PDF conversion and OCR;
- chunk creation;
- document embeddings;
- query embeddings;
- cross-encoder reranking;
- PostgreSQL lexical/vector search;
- workflow orchestration.

The external LLM is used for query expansion when needed, grounded answer generation, selective
verification, and report synthesis. Therefore the normal variable API charge is primarily LLM token
usage. Your own CPU/RAM/storage/network/backup infrastructure still has a cost.

## Docling model caution

Docling's codebase is MIT, while its model bundle is separately licensed. The upstream model card for
the layout/TableFormer bundle currently identifies permissive Apache-2.0 and CDLA-Permissive-2.0
terms. Do not assume the Docling code license automatically covers every optional model; audit the
exact downloaded model revisions again whenever Docling or its model bundle is upgraded.

## Reproducibility and SBOM

Before organization deployment:

1. Build the exact Docker images that passed acceptance testing.
2. Tag them with an immutable application version or Git commit.
3. Generate an SBOM, for example with Syft:
   `syft <image-name> -o cyclonedx-json > sbom.json`.
4. Scan the SBOM and image with your approved security scanner.
5. Preserve the resolved Python package list from each image:
   `python -m pip freeze`.
6. Preserve the exact Hugging Face model revisions in the deployment record.
7. Re-run retrieval and answer-quality regression tests before dependency/model upgrades.

## Model replacement policy

Embedding dimension is part of the database schema in this release. The default `BAAI/bge-m3`
embedding is 1024-dimensional. Replacing it with a model of another dimension requires a new vector
migration and full re-embedding. A reranker can be replaced independently, but must be benchmarked on
the organization's golden evaluation set before rollout.


## Parser version

Release 0.4.0 retains the pin `docling==2.123.1` so the tested OCR-mode/options API does not drift between laptop acceptance and deployment. Model artifacts still have their own upstream licenses and should be reviewed separately.

## 0.3 artifact bootstrap note

Docling remains pinned to `2.123.1`. Its local PDF model artifacts are cached in the Docker `model_cache` volume through the `docling-tools models download` command. Review the upstream model artifact licenses as part of production approval in addition to the Python package license.