"""Application runtime settings loaded from environment variables.

Environment variable format:
- Prefix: MMAA_
- Source: .env (plus process environment)

Common configuration profiles:
1) Fully local fallback profile:
   - LLM provider: heuristic/auto
   - Embedding provider: deterministic
   - Multimodal provider: heuristic/auto
2) Local Ollama (OpenAI-compatible) profile:
   - LLM provider: openai + MMAA_LLM_BASE_URL=http://localhost:11434/v1
   - Embedding provider: openai + MMAA_RAG_OPENAI_BASE_URL=http://localhost:11434/v1
     (or deterministic for faster local use)
   - Multimodal provider: openai + MMAA_MULTIMODAL_BASE_URL=http://localhost:11434/v1
3) Hosted OpenAI profile:
   - LLM provider: openai + MMAA_LLM_API_KEY
   - Embedding provider: openai + MMAA_RAG_OPENAI_API_KEY
   - Multimodal provider: openai + MMAA_MULTIMODAL_API_KEY
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed runtime settings for API, RAG, agents, and model providers.

    Provider options by field:
    - rag_vector_store_provider: pgvector | qdrant | auto
    - rag_embedding_provider: deterministic | openai | sentence_transformers | auto | neural
    - rag_reranker_provider: lexical | cross_encoder
    - llm_provider: auto | heuristic | deterministic | fallback | openai
    - multimodal_provider: auto | heuristic | fallback | openai
    """

    # Core service identity and logging.
    app_name: str = "multimodal-multiagent-ai-assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database / vector infrastructure.
    database_url: str | None = None
    rag_vector_store_provider: str = "pgvector"
    rag_vector_store_mirror_writes: bool = True
    rag_table_name: str = "rag_chunks"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "rag_chunks"
    qdrant_prefer_grpc: bool = False
    qdrant_timeout_sec: float = 10.0

    # RAG embeddings and ingestion parameters.
    rag_embedding_provider: str = "deterministic"
    rag_embedding_model: str = "text-embedding-3-small"
    rag_sentence_transformer_model: str = "BAAI/bge-small-en-v1.5"
    rag_openai_api_key: str | None = None
    rag_openai_base_url: str | None = None
    rag_embedding_dimensions: int = 64
    rag_requested_embedding_dimensions: int | None = None
    rag_chunk_size: int = 2500
    rag_chunk_overlap: int = 120
    rag_max_source_bytes: int = 2_000_000

    # Retrieval and ranking budgets.
    rag_dense_top_k: int = 12
    rag_lexical_top_k: int = 12
    rag_rrf_k: int = 60
    rag_rerank_pool_size: int = 24
    rag_reranker_enabled: bool = True
    rag_reranker_provider: str = "lexical"
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM generation provider and decoding controls.
    llm_provider: str = "auto"
    llm_model: str = "gpt-4.1-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500

    # Multimodal provider and media processing budgets.
    multimodal_provider: str = "auto"
    multimodal_api_key: str | None = None
    multimodal_base_url: str | None = None
    multimodal_vision_model: str = "gpt-4o-mini"
    multimodal_video_sample_fps: float = 1.0
    multimodal_video_max_frames: int = 24
    multimodal_video_latency_budget_ms: int = 1500
    multimodal_video_estimated_ms_per_frame: float = 12.0
    multimodal_video_max_window_sec: float = 120.0
    multimodal_video_max_key_events: int = 5
    multimodal_video_remote_fetch_timeout_sec: float = 20.0
    multimodal_video_max_remote_source_bytes: int = 120_000_000
    multimodal_video_require_frame_findings: bool = True
    vision_max_source_bytes: int = 5_000_000

    # Agent orchestration guardrails.
    agent_max_steps: int = 6
    agent_tool_budget: int = 2
    agent_tool_timeout_sec: float = 2.0
    agent_tool_retries: int = 1
    agent_retrieval_top_k: int = 5
    agent_checkpoint_enabled: bool = True
    agent_resume_from_checkpoint: bool = True

    model_config = SettingsConfigDict(
        env_prefix="MMAA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
