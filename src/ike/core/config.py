from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    app_name: str = "IMS - Incident Management System"
    organization_name: str = "Delhi Metro Rail Corporation (DMRC)"
    organization_short_name: str = "DMRC"
    app_secret_key: str = Field(min_length=32)
    session_cookie_secure: bool = False
    log_level: str = "INFO"

    database_url: str
    redis_url: str = "redis://valkey:6379/0"
    task_visibility_timeout_seconds: int = 21600
    data_root: Path = Path("/data")
    max_upload_mb: int = 200

    # 0.5.0 physically separates latency-sensitive Q&A ML from background document embedding.
    # INFERENCE_URL is kept as a compatibility fallback for older deployments.
    inference_url: str = "http://inference-query:8090"
    query_inference_url: str = "http://inference-query:8090"
    ingestion_inference_url: str = "http://inference-ingest:8090"
    internal_service_token: str = Field(min_length=24)
    # Search/runtime model metadata used by ingestion/indexing and v4 retrieval.
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    rrf_k: int = 60
    hnsw_ef_search: int = 100
    hnsw_max_scan_tuples: int = 20000
    hnsw_iterative_scan_mode: Literal["strict_order", "relaxed_order"] = "strict_order"

    # Online request admission is intentionally bounded per API replica. The local
    # inference-query service provides a second global bottleneck for expensive ML stages.
    query_max_active_per_api: int = 4
    query_max_waiting_per_api: int = 48

    # Low-level retrieval mechanics. These improve recall but never decide meaning,
    # source identity, completeness, or whether the agent should stop.
    relaxed_lexical_enabled: bool = True
    section_navigation_enabled: bool = True
    section_navigation_scan_multiplier: int = 4

    # Acceptance aid: expose a per-query forensic report to the asking user/admin.
    query_debug_download_enabled: bool = True

    # v5 planned AI research. The planner owns query understanding/decomposition;
    # the answer agent owns evidence relevance, gap detection and final synthesis.
    # Python only enforces retrieval/access/time/resource bounds.
    agentic_qa_enabled: bool = True
    agentic_qa_fallback_enabled: bool = False
    agent_search_top_k: int = 14
    agent_search_max_top_k: int = 24
    agent_search_prefilter_k: int = 28
    agent_rerank_enabled: bool = False
    agent_search_rerank_candidates: int = 80
    agent_enumeration_top_k: int = 48
    agent_enumeration_max_top_k: int = 80
    agent_enumeration_prefilter_k: int = 96
    agent_enumeration_dense_top_k: int = 32
    agent_enumeration_rerank_candidates: int = 80
    agent_document_search_limit: int = 12
    agent_structure_max_nodes: int = 180
    agent_evidence_catalog_limit: int = 120
    agent_answer_evidence_cap: int = 48
    agent_selected_evidence_cap: int = 24
    agent_max_evidence: int = 50
    agent_recent_history_count: int = 6
    agent_recent_history_answer_chars: int = 1200
    agent_primary_research_time_seconds: float = 20.0
    agent_gap_research_time_seconds: float = 12.0
    agent_max_gap_tasks: int = 6
    agent_planner_max_output_tokens: int = 1800
    agent_answer_max_output_tokens: int = 4200
    agent_planner_reasoning: Literal["low", "medium", "high"] = "medium"
    agent_answer_reasoning: Literal["low", "medium", "high"] = "medium"

    # 0.9.0 industrial retrieval fabric.  A secondary hierarchical index is built from
    # existing chunks so query-time planning can use the organisation's own vocabulary
    # and likely governing documents before expensive chunk-level search.
    retrieval_intelligence_enabled: bool = True
    retrieval_intelligence_dense_top_k: int = 18
    retrieval_intelligence_lexical_top_k: int = 24
    retrieval_intelligence_fused_top_k: int = 20
    retrieval_intelligence_terms_max: int = 10
    retrieval_intelligence_documents_max_fast: int = 8
    retrieval_intelligence_documents_max_focused: int = 16
    retrieval_intelligence_family_expansion_max_documents: int = 24
    retrieval_intelligence_section_chars: int = 6000
    # 0.9.1 stores corpus terminology on section nodes instead of creating a separate
    # vector-bearing concept row for every term. Existing chunk embeddings are reused to
    # derive document/section centroids, so no secondary document-embedding backfill is
    # required.
    retrieval_intelligence_terms_per_section: int = 12
    retrieval_intelligence_min_hint_coverage_ratio: float = 0.90

    # Corpus-aware terminology resolution.  Suggestions are search expansions only; the
    # original user query is always retained and protected technical identifiers are never
    # silently corrected.
    corpus_term_resolution_enabled: bool = True
    corpus_term_max_query_tokens: int = 8
    corpus_term_candidates_per_token: int = 3
    corpus_term_similarity_threshold: float = 0.42
    corpus_term_expansions_max: int = 6

    # Lazy visual evidence: source PDFs/canonical JSON are resolved only for retrieved
    # pages, so existing indexed documents do not need reprocessing.
    visual_evidence_enabled: bool = True
    visual_max_items: int = 2
    visual_nearby_pages: int = 1
    visual_render_dpi: int = 130
    visual_cache_dir_name: str = "visual-cache"
    visual_token_ttl_seconds: int = 3600

    chunk_max_tokens: int = 500
    docling_ocr: bool = True
    docling_ocr_languages: str = "eng"
    docling_ocr_mode: Literal[
        "default", "layout_regions", "pdf_aware_layout_regions", "full_page"
    ] = "pdf_aware_layout_regions"
    docling_table_structure: bool = True
    docling_table_mode: Literal["fast", "accurate"] = "accurate"
    docling_timeout_seconds: int = 3600
    docling_artifacts_path: Path = Path("/models/docling")
    ingestion_embedding_batch_size: int = 32
    ingest_cpu_target_percent: int = 75
    ingest_worker_concurrency: str = "1"
    ingest_max_concurrency: int = 1
    ingest_memory_gb_per_process: float = 2.5
    ingest_memory_reserve_gb: float = 3.0
    docling_num_threads_per_document: int = 2
    docling_ocr_batch_size: int = 2
    docling_layout_batch_size: int = 2
    docling_table_batch_size: int = 2

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_fast_model: str = "gpt-5.6-luna"
    llm_strong_model: str = "gpt-5.6-terra"
    llm_fast_reasoning: str = "low"
    llm_strong_reasoning: str = "high"
    llm_max_output_tokens: int = 7000
    llm_request_timeout_seconds: float = 120.0
    llm_max_retries: int = 1

    # Optional ingestion-time organisational knowledge profile enrichment.
    okf_enabled: bool = False
    okf_bundle_dir_name: str = "okf"
    okf_profile_enrichment_enabled: bool = True

    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = "change-this-before-running"
    bootstrap_admin_department: str = "platform"

    report_max_documents: int = 20
    report_pack_max_chars: int = 24000
    report_max_packs: int = 120
    report_parallelism: int = 3
    report_reduce_batch_chars: int = 60000
    report_synthesis_max_chars: int = 120000
    report_reduce_max_rounds: int = 4

    @field_validator("embedding_dim")
    @classmethod
    def fixed_schema_dimension(cls, value: int) -> int:
        if value != 1024:
            raise ValueError(
                "This release's database schema is fixed at 1024 dimensions. "
                "Rebuild the vector migration before changing EMBEDDING_DIM."
            )
        return value

    @property
    def ocr_language_list(self) -> list[str]:
        values = [part.strip() for part in self.docling_ocr_languages.split(",") if part.strip()]
        return values or ["eng"]

    @property
    def originals_dir(self) -> Path:
        return self.data_root / "originals"

    @property
    def parsed_dir(self) -> Path:
        return self.data_root / "parsed"

    @property
    def visual_cache_dir(self) -> Path:
        return self.data_root / self.visual_cache_dir_name

    @property
    def okf_bundle_dir(self) -> Path:
        return self.data_root / self.okf_bundle_dir_name


@lru_cache
def get_settings() -> Settings:
    return Settings()
