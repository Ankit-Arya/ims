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
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_max_seq_length: int = 1024
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 1024
    ml_device: str = "cpu"
    ml_batch_size: int = 8

    dense_top_k: int = 80
    lexical_top_k: int = 80
    exact_top_k: int = 30
    fused_top_k: int = 80
    rerank_top_k: int = 16
    answer_evidence_k: int = 12
    direct_dense_top_k: int = 30
    direct_lexical_top_k: int = 40
    direct_exact_top_k: int = 30
    direct_fused_top_k: int = 36
    direct_rerank_top_k: int = 12
    direct_evidence_k: int = 10
    lookup_scan_top_k: int = 240
    lookup_rerank_top_k: int = 16
    lookup_evidence_k: int = 16
    rrf_k: int = 60
    hnsw_ef_search: int = 100
    hnsw_max_scan_tuples: int = 20000
    hnsw_iterative_scan_mode: Literal["strict_order", "relaxed_order"] = "strict_order"
    rerank_score_threshold: float = 0.12

    # 0.10 retrieval-control foundation. Likely-document routing is a recall-preserving
    # boost lane, never an implicit hard scope. Only explicit document selection may
    # narrow the accessible corpus.
    routed_document_boost_enabled: bool = True
    routed_document_boost: float = 1.12
    routed_dense_top_k: int = 12
    routed_lexical_top_k: int = 16
    routed_search_max_queries: int = 2
    neighbor_radius: int = 1
    coverage_scan_top_k: int = 800
    coverage_max_documents: int = 16
    coverage_evidence_per_document: int = 2
    coverage_max_evidence_k: int = 32

    # Broad role/responsibility questions need alias resolution plus section-level
    # coverage within the same document. These bounds are API/query-side only.
    role_alias_scan_top_k: int = 120
    role_alias_fallback_scan_top_k: int = 240
    role_coverage_max_aliases: int = 4
    role_coverage_scan_top_k: int = 240
    role_coverage_max_documents: int = 24
    role_coverage_max_sections: int = 32
    role_coverage_sections_per_document: int = 8
    role_coverage_max_evidence_k: int = 32
    role_coverage_rerank_pool: int = 40
    role_coverage_rerank_top_k: int = 20
    role_alias_cache_ttl_seconds: int = 600
    role_alias_cache_max_items: int = 512

    # Online request admission is intentionally bounded per API replica. The local
    # inference-query service provides a second global bottleneck for expensive ML stages.
    query_max_active_per_api: int = 4
    query_max_waiting_per_api: int = 48

    # Answer usefulness policy. Research always plans structure; Direct stays one-call but
    # may include closely related evidence when it materially helps the user understand/use
    # the answer. This is generalized and never hard-codes topics or documents.
    helpful_context_mode: Literal["off", "relevant", "rich"] = "relevant"
    helpful_context_max_sections: int = 3
    direct_max_output_tokens: int = 1800

    # 0.6.0 operational retrieval. The generic overview lane is searched for every query;
    # its evidence is only shown when relevant. MRGR is the current configured overview book.
    # Governing references are always-considered evidence lanes (for example MRGR),
    # not priority gates.  The legacy OVERVIEW_* settings remain as compatibility
    # fallbacks during rollout.
    governing_reference_enabled: bool = True
    governing_reference_patterns: str = ""
    governing_reference_dense_top_k: int = 6
    governing_reference_lexical_top_k: int = 8
    governing_reference_evidence_k: int = 2

    overview_retrieval_enabled: bool = True
    overview_document_pattern: str = "MRGR"
    overview_dense_top_k: int = 12
    overview_lexical_top_k: int = 12
    overview_evidence_k: int = 4

    # 0.8.0 retrieval control plane.  A configured priority source is searched as a
    # real first stage, not merely mixed into global RRF.  Leave PRIORITY_DOCUMENT_PATTERNS
    # blank to inherit the legacy OVERVIEW_DOCUMENT_PATTERN during upgrade.
    priority_retrieval_enabled: bool = True
    priority_document_patterns: str = ""
    priority_max_queries: int = 8
    priority_dense_top_k: int = 56
    priority_lexical_top_k: int = 56
    priority_relaxed_lexical_top_k: int = 48
    priority_section_top_k: int = 36
    priority_exact_top_k: int = 24
    priority_fused_top_k: int = 48
    priority_rerank_top_k: int = 20
    priority_evidence_k: int = 18
    priority_recovery_enabled: bool = True
    priority_recovery_max_queries_per_goal: int = 4  # legacy compatibility
    recovery_max_queries_per_goal: int = 3

    # Separate recall-oriented lexical and section-navigation lanes.  They complement,
    # rather than replace, the existing precise websearch FTS and dense retrieval.
    relaxed_lexical_enabled: bool = True
    relaxed_lexical_top_k: int = 36
    relaxed_lexical_weight: float = 0.92
    section_navigation_enabled: bool = True
    section_navigation_top_k: int = 28
    section_navigation_weight: float = 1.30
    section_navigation_scan_multiplier: int = 4

    # Goal-local cross-encoder scoring prevents one branch of a compositional question
    # from winning a global popularity contest. Pair counts remain bounded on CPU.
    goal_local_rerank_enabled: bool = True
    interactive_research_cross_encoder_enabled: bool = False
    goal_local_rerank_candidates: int = 10
    goal_local_rerank_top_k: int = 4
    compositional_definition_evidence_per_goal: int = 6

    # Retrieval-quality gating runs before answer generation and can issue a materially
    # different search-only repair probe when a priority source is incomplete.
    retrieval_quality_gate_enabled: bool = True
    # When a first-pass answer is incomplete, allow one bounded LLM query-repair call
    # grounded in corpus headings/section vocabulary before the single recovery search.
    # Successful queries never pay this latency cost.
    adaptive_semantic_recovery_enabled: bool = True
    # Recovery also admits a small first/last-chunk sample from the top semantic corpus
    # sections. This gives a failed query an independent path out of a wrong-scope chunk
    # without adding another embedding/LLM call or domain-specific phrase rules.
    adaptive_corpus_section_recovery_enabled: bool = True
    adaptive_corpus_section_recovery_max_sections: int = 8
    adaptive_corpus_section_recovery_chunks_per_section: int = 2
    # Enumeration/overview failures often live in many sibling sections of one large
    # handbook. Re-run corpus-section discovery on at most a couple of recovery probes;
    # these are navigation hints only and never become answer evidence by themselves.
    adaptive_enumeration_section_rediscovery_enabled: bool = True
    adaptive_enumeration_section_rediscovery_max_queries: int = 2
    # Auto mode may broaden retrieval to Research without paying the strong-model latency.
    # Explicit Research still uses the strong model; this switch is an escape hatch for
    # environments that prefer maximum synthesis quality over interactive response time.
    auto_research_strong_answer_enabled: bool = False
    retrieval_repair_max_output_tokens: int = 1200

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

    # Retrieval effort follows the same broad industry pattern as agentic retrieval: easy
    # requests take a bounded path, focused procedures get hierarchical expansion, and
    # corpus-wide/compositional requests get research coverage.
    adaptive_retrieval_effort_enabled: bool = True
    fast_draft_evidence_k: int = 8
    focused_draft_evidence_k: int = 16
    research_draft_evidence_k: int = 28
    draft_evidence_per_goal_min: int = 2
    draft_evidence_per_document_max: int = 6

    # Priority sources are a cheap governing-rule probe in 0.9 instead of a second full
    # research pass.  If the probe is incomplete, routed/source-family retrieval proceeds.
    priority_probe_max_queries: int = 4
    priority_probe_dense_top_k: int = 18
    priority_probe_lexical_top_k: int = 20
    priority_probe_relaxed_top_k: int = 14
    priority_probe_section_top_k: int = 12
    priority_probe_exact_top_k: int = 12
    priority_probe_fused_top_k: int = 24
    priority_probe_rerank_top_k: int = 8
    priority_probe_evidence_k: int = 8
    priority_probe_recovery_enabled: bool = False

    # Batch goal reranking eliminates serial cross-encoder calls for independent evidence
    # goals while preserving the same goal-local ranking semantics.
    batch_goal_rerank_enabled: bool = True
    batch_goal_rerank_max_groups: int = 12
    batch_goal_rerank_max_pairs: int = 120

    # Recovery may add corpus-grounded terminology but must preserve the original evidence
    # goal.  Supported evidence is protected from later retrieval eviction.
    monotonic_evidence_enabled: bool = True
    semantic_drift_guard_enabled: bool = True

    # Technical identifiers are expanded separator-tolerantly (e.g. RS10/RS-10/RS 10)
    # without replacing the user's original wording.
    identifier_variant_expansion: bool = True
    identifier_variant_max_queries: int = 8

    # Existing Docling ingestion already stores contextual_text with repeated table headers
    # plus chunk metadata. These knobs improve table ranking without corpus reprocessing.
    table_context_boost_enabled: bool = True
    table_context_fusion_bonus: float = 0.12
    table_retrieval_top_k: int = 16
    table_retrieval_weight: float = 1.35

    # Section-aware evidence reconstruction from existing ordinal/section_path metadata.
    hierarchical_retrieval_enabled: bool = True
    hierarchical_window: int = 4
    hierarchical_max_chunks_per_anchor: int = 6

    # Keep reranker work bounded on CPU. Structural/metadata discovery should reduce the
    # pool before the expensive cross-encoder rather than globally lowering recall.
    rerank_prefilter_max_candidates: int = 40

    # Candidate-channel reservations prevent one noisy retriever from crowding out a
    # structurally important lane before the single primary cross-encoder pass.
    lane_reservation_enabled: bool = True
    lane_reserve_dense: int = 4
    lane_reserve_lexical: int = 4
    lane_reserve_exact: int = 2
    lane_reserve_table: int = 4
    lane_reserve_governing: int = 2
    lane_reserve_routed: int = 2

    # 0.7.0 compositional query planning. Complex/multi-part questions are decomposed
    # into bounded evidence goals before retrieval. These limits are deliberately small
    # so arbitrary user wording cannot create unbounded fan-out or CPU reranking work.
    compositional_planning_enabled: bool = True
    compositional_semantic_planning_enabled: bool = True
    # Even when full semantic planning is disabled, structurally lossy deterministic plans
    # may receive one bounded semantic repair pass. Troubleshooting/definition paths remain deterministic.
    compositional_semantic_repair_enabled: bool = True
    query_frame_enabled: bool = True
    # Reserved for the future fan-out/fan-in graph. Keep disabled until each branch uses
    # an independent SQLAlchemy session and tracing proves concurrency is safe.
    speculative_retrieval_enabled: bool = False
    compositional_max_goals: int = 12
    compositional_max_queries: int = 14
    compositional_queries_per_goal: int = 3
    compositional_goal_candidate_reserve: int = 2
    compositional_goal_evidence_per_goal: int = 2
    compositional_max_evidence_k: int = 32
    compositional_rerank_pool: int = 28
    compositional_rerank_top_k: int = 16
    compositional_lookup_scan_top_k: int = 160
    compositional_planner_max_output_tokens: int = 1800
    compositional_audit_enabled: bool = True
    compositional_audit_max_evidence: int = 24
    compositional_audit_max_output_tokens: int = 1800
    compositional_recovery_enabled: bool = True
    compositional_recovery_max_goals: int = 4

    # Verification is risk based. Compositional plans add goal-level risk signals.
    verification_risk_threshold: int = 3

    # Lazy visual evidence: source PDFs/canonical JSON are resolved only for retrieved
    # pages, so existing indexed documents do not need reprocessing.
    visual_evidence_enabled: bool = True
    visual_max_items: int = 2
    visual_nearby_pages: int = 1
    visual_render_dpi: int = 130
    visual_cache_dir_name: str = "visual-cache"
    visual_token_ttl_seconds: int = 3600
    selective_vision_enabled: bool = False

    chunk_max_tokens: int = 500
    docling_ocr: bool = True
    docling_ocr_languages: str = "eng"
    docling_ocr_mode: Literal["default", "layout_regions", "pdf_aware_layout_regions", "full_page"] = "pdf_aware_layout_regions"
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
    llm_strong_reasoning: str = "medium"
    llm_max_output_tokens: int = 7000
    verify_mode: Literal["off", "selective", "always"] = "selective"

    # Local OKF + Jev experiment. Disabled unless explicitly enabled in .env.
    okf_enabled: bool = False
    okf_bundle_dir_name: str = "okf"
    okf_profile_enrichment_enabled: bool = True
    okf_query_enabled: bool = True
    okf_query_max_documents: int = 12
    okf_query_fusion_bonus: float = 0.18
    okf_query_reserve_candidates: int = 4
    okf_query_targeted_reserve: int = 8

    jev_mode: Literal["off", "shadow", "apply"] = "off"

    # auto:
    #   jv_live_* -> hosted jevtypesafeai.com gateway
    #   anything else -> official TypeSafe API
    jev_provider: Literal["auto", "typesafe", "hosted"] = "auto"

    typesafe_api_key: str = ""
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    jev_timeout_seconds: float = 6.0
    jev_max_candidates: int = 16
    jev_candidate_chars: int = 2600
    jev_weight: float = 0.55
    jev_fail_open: bool = True

    # Complex/compositional Jev judging reuses IMS evidence goals instead of judging
    # every passage only against the broad parent question.
    jev_complex_enabled: bool = True
    jev_complex_candidate_chars: int = 1800
    jev_max_goal_pairs: int = 24
    jev_max_goals_per_candidate: int = 2

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
            raise ValueError("This release's database schema is fixed at 1024 dimensions. Rebuild the vector migration before changing EMBEDDING_DIM.")
        return value

    @property
    def governing_reference_pattern_list(self) -> list[str]:
        from ike.retrieval.source_policy import split_patterns

        configured = split_patterns(self.governing_reference_patterns)
        if configured:
            return configured
        # Upgrade compatibility: existing deployments already identify MRGR (or another
        # general rulebook) through OVERVIEW_DOCUMENT_PATTERN.
        return split_patterns(self.overview_document_pattern)

    @property
    def priority_pattern_list(self) -> list[str]:
        from ike.retrieval.source_policy import split_patterns

        configured = split_patterns(self.priority_document_patterns)
        if configured:
            return configured
        # Backward-compatible upgrade path: existing deployments already use the legacy
        # overview pattern to identify the general rule/source family.
        return split_patterns(self.overview_document_pattern)

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
