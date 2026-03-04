from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "multimodal-multiagent-ai-assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    database_url: str | None = None
    rag_vector_store_provider: str = "pgvector"
    rag_vector_store_mirror_writes: bool = True
    rag_table_name: str = "rag_chunks"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "rag_chunks"
    qdrant_prefer_grpc: bool = False
    qdrant_timeout_sec: float = 10.0
    rag_embedding_provider: str = "deterministic"
    rag_embedding_model: str = "text-embedding-3-small"
    rag_sentence_transformer_model: str = "BAAI/bge-small-en-v1.5"
    rag_openai_api_key: str | None = None
    rag_embedding_dimensions: int = 64
    rag_requested_embedding_dimensions: int | None = None
    rag_chunk_size: int = 700
    rag_chunk_overlap: int = 120
    rag_max_source_bytes: int = 2_000_000
    rag_dense_top_k: int = 12
    rag_lexical_top_k: int = 12
    rag_rrf_k: int = 60
    rag_rerank_pool_size: int = 24
    rag_reranker_enabled: bool = True
    rag_reranker_provider: str = "lexical"
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(env_prefix="MMAA_", extra="ignore")


settings = Settings()
