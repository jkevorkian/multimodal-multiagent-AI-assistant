from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import tempfile
from pathlib import Path
from uuid import uuid4

from app.agents import (
    AgentOrchestrator,
    AnalystAgent,
    AnswerAgent,
    InMemoryCheckpointStore,
    NullCheckpointStore,
    ResearchAgent,
)
from app.agents.context_manager import AgentContextManager
from app.core.config import settings
from app.core.context_compaction import ContextCompactor
from app.core.event_bus import InMemoryEventBus, get_event_bus
from app.interfaces import (
    EmbeddingClient,
    LLMClient,
    MultimodalEmbeddingClient,
    Reranker,
    Retriever,
    Tool,
    VectorStore,
    VideoClient,
    VisionClient,
)
from app.llm import build_llm_client
from app.multimodal import build_multimodal_clients
from app.rag import (
    DocumentIngestionService,
    TextRAGRetriever,
    build_embedding_client,
    build_multimodal_embedding_client,
    build_reranker,
)
from app.storage import FallbackVectorStore, PgVectorStore, QdrantVectorStore, SQLiteChatStore
from app.tools import ToolRegistry, build_default_tools


@dataclass
class ServiceContainer:
    llm: LLMClient
    embeddings: EmbeddingClient
    multimodal_embeddings: MultimodalEmbeddingClient | None
    reranker: Reranker
    retriever: Retriever
    vector_store: VectorStore
    ingestion: DocumentIngestionService
    vision: VisionClient
    video: VideoClient
    tools: list[Tool]
    tool_registry: ToolRegistry
    orchestrator: AgentOrchestrator
    event_bus: InMemoryEventBus
    chat_store: SQLiteChatStore
    embedding_provider: str
    multimodal_embedding_provider: str
    llm_provider: str
    reranker_provider: str
    multimodal_provider: str
    vector_store_provider: str


def build_service_container(database_url: str | None = None) -> ServiceContainer:
    return _build_service_container_internal(database_url=database_url)


def _build_service_container_internal(
    database_url: str | None = None,
    *,
    embedding_provider_override: str | None = None,
    llm_provider_override: str | None = None,
    multimodal_provider_override: str | None = None,
    disable_external_api: bool = False,
) -> ServiceContainer:
    embedding_provider = embedding_provider_override or settings.rag_embedding_provider
    llm_provider = llm_provider_override or settings.llm_provider
    multimodal_provider = multimodal_provider_override or settings.multimodal_provider
    rag_openai_api_key = None if disable_external_api else settings.rag_openai_api_key
    rag_multimodal_api_key = (
        None
        if disable_external_api
        else (settings.rag_multimodal_openai_api_key or settings.rag_openai_api_key)
    )
    rag_multimodal_base_url = (
        None
        if disable_external_api
        else (settings.rag_multimodal_openai_base_url or settings.rag_openai_base_url)
    )
    llm_api_key = None if disable_external_api else (settings.llm_api_key or rag_openai_api_key)
    multimodal_api_key = None if disable_external_api else (settings.multimodal_api_key or llm_api_key or rag_openai_api_key)

    embedding_selection = build_embedding_client(
        provider=embedding_provider,
        model_name=settings.rag_embedding_model,
        sentence_transformer_model=settings.rag_sentence_transformer_model,
        api_key=rag_openai_api_key,
        base_url=None if disable_external_api else settings.rag_openai_base_url,
        deterministic_dimensions=settings.rag_embedding_dimensions,
        requested_dimensions=settings.rag_requested_embedding_dimensions,
    )
    embeddings = embedding_selection.client
    multimodal_embeddings: MultimodalEmbeddingClient | None = None
    multimodal_embedding_provider_name = "disabled"
    multimodal_embedding_dimensions = settings.rag_multimodal_embedding_dimensions
    if settings.rag_multimodal_enabled:
        multimodal_embedding_provider = settings.rag_multimodal_embedding_provider
        if disable_external_api and multimodal_embedding_provider.strip().lower().replace("-", "_") in {
            "openai",
            "qwen3_vl",
            "qwen3_vl_embedding",
            "qwen_vl_embedding",
        }:
            multimodal_embedding_provider = "deterministic"
        multimodal_embedding_selection = build_multimodal_embedding_client(
            provider=multimodal_embedding_provider,
            model_name=settings.rag_multimodal_embedding_model,
            api_key=rag_multimodal_api_key,
            base_url=rag_multimodal_base_url,
            deterministic_dimensions=settings.rag_multimodal_embedding_dimensions,
            requested_dimensions=settings.rag_requested_embedding_dimensions,
            endpoint_path=settings.rag_multimodal_embedding_endpoint,
        )
        multimodal_embeddings = multimodal_embedding_selection.client
        multimodal_embedding_provider_name = multimodal_embedding_selection.provider_name
        multimodal_embedding_dimensions = (
            multimodal_embedding_selection.dimensions or settings.rag_multimodal_embedding_dimensions
        )
    reranker_provider = settings.rag_reranker_provider
    if disable_external_api and reranker_provider.strip().lower().replace("-", "_") in {"openai", "qwen3_vl"}:
        reranker_provider = "lexical"
    reranker_openai_api_key = (
        None
        if disable_external_api
        else (settings.rag_reranker_openai_api_key or llm_api_key or rag_openai_api_key)
    )
    reranker_openai_base_url = (
        None
        if disable_external_api
        else (settings.rag_reranker_openai_base_url or settings.llm_base_url or settings.rag_openai_base_url)
    )
    reranker_selection = build_reranker(
        enabled=settings.rag_reranker_enabled,
        provider=reranker_provider,
        cross_encoder_model=settings.rag_reranker_model,
        openai_api_key=reranker_openai_api_key,
        openai_base_url=reranker_openai_base_url,
        qwen3_vl_endpoint=settings.rag_reranker_qwen3_vl_endpoint,
    )
    embedding_dimensions = embedding_selection.dimensions or settings.rag_embedding_dimensions
    pgvector_store = PgVectorStore(
        database_url=settings.database_url if database_url is None else database_url,
        table_name=settings.rag_table_name,
        embedding_dimensions=embedding_dimensions,
    )
    vector_store: VectorStore = pgvector_store
    vector_store_provider = "pgvector"
    requested_store = settings.rag_vector_store_provider.strip().lower()
    if requested_store in {"qdrant", "auto"}:
        qdrant_url = settings.qdrant_url
        if qdrant_url:
            qdrant_store = QdrantVectorStore(
                url=qdrant_url,
                api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection_name,
                embedding_dimensions=embedding_dimensions,
                multimodal_embedding_dimensions=multimodal_embedding_dimensions,
                text_vector_name=settings.rag_text_vector_name,
                multimodal_vector_name=settings.rag_multimodal_vector_name,
                enable_named_vectors=settings.rag_multimodal_enabled,
                prefer_grpc=settings.qdrant_prefer_grpc,
                timeout_sec=settings.qdrant_timeout_sec,
            )
            vector_store = FallbackVectorStore(
                primary=qdrant_store,
                secondary=pgvector_store,
                mirror_writes=settings.rag_vector_store_mirror_writes,
            )
            vector_store_provider = "qdrant+pgvector"

    multimodal_selection = build_multimodal_clients(
        provider=multimodal_provider,
        vision_model=settings.multimodal_vision_model,
        api_key=multimodal_api_key,
        base_url=None if disable_external_api else settings.multimodal_base_url,
    )

    ingestion = DocumentIngestionService(
        embedding_client=embeddings,
        vector_store=vector_store,
        multimodal_embedding_client=multimodal_embeddings,
        text_vector_name=settings.rag_text_vector_name,
        multimodal_vector_name=settings.rag_multimodal_vector_name,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        max_source_bytes=settings.rag_max_source_bytes,
        vision_client=multimodal_selection.vision,
        video_client=multimodal_selection.video,
        video_sample_fps=settings.multimodal_video_sample_fps,
        video_max_frames=settings.multimodal_video_max_frames,
        video_latency_budget_ms=settings.multimodal_video_latency_budget_ms,
        video_estimated_ms_per_frame=settings.multimodal_video_estimated_ms_per_frame,
        video_max_window_sec=settings.multimodal_video_max_window_sec,
        video_max_key_events=settings.multimodal_video_max_key_events,
        video_remote_fetch_timeout_sec=settings.multimodal_video_remote_fetch_timeout_sec,
        video_max_remote_source_bytes=settings.multimodal_video_max_remote_source_bytes,
        video_require_frame_findings=settings.multimodal_video_require_frame_findings,
        video_ingest_enrich_with_analysis=settings.multimodal_video_ingest_enrich_with_analysis,
        video_audio_transcription_enabled=settings.multimodal_video_audio_transcription_enabled,
        video_audio_transcription_model=settings.multimodal_video_audio_transcription_model,
        video_audio_transcription_language=settings.multimodal_video_audio_transcription_language,
        video_audio_transcription_max_segments=settings.multimodal_video_audio_transcription_max_segments,
    )
    retriever = TextRAGRetriever(
        embedding_client=embeddings,
        vector_store=vector_store,
        multimodal_embedding_client=multimodal_embeddings,
        reranker=reranker_selection.reranker,
        dense_top_k=settings.rag_dense_top_k,
        lexical_top_k=settings.rag_lexical_top_k,
        rrf_k=settings.rag_rrf_k,
        rerank_pool_size=settings.rag_rerank_pool_size,
        text_vector_name=settings.rag_text_vector_name,
        multimodal_vector_name=settings.rag_multimodal_vector_name,
        use_text_dense_branch=settings.rag_multimodal_fuse_text_dense,
    )
    tools: list[Tool] = build_default_tools(
        retriever=retriever,
        vector_store=vector_store,
        workspace_root=Path.cwd(),
        enable_network_tools=not disable_external_api,
        vision_client=multimodal_selection.vision,
        video_client=multimodal_selection.video,
        video_probe_sample_fps=settings.multimodal_video_sample_fps,
        video_probe_max_frames=settings.multimodal_video_max_frames,
    )
    tool_registry = ToolRegistry(tools)
    event_bus = InMemoryEventBus() if disable_external_api else get_event_bus()
    research_agent = ResearchAgent(
        retriever=retriever,
        tool_registry=tool_registry,
        retrieval_top_k=settings.agent_retrieval_top_k,
        tool_timeout_sec=settings.agent_tool_timeout_sec,
        tool_retries=settings.agent_tool_retries,
        max_tools_per_pass=settings.agent_max_tools_per_research_pass,
        event_bus=event_bus,
    )
    checkpoint_store = InMemoryCheckpointStore() if settings.agent_checkpoint_enabled else NullCheckpointStore()
    llm_selection = build_llm_client(
        provider=llm_provider,
        model_name=settings.llm_model,
        api_key=llm_api_key,
        base_url=None if disable_external_api else settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    llm = llm_selection.client
    answer_agent = AnswerAgent(llm=llm, event_bus=event_bus)
    context_manager = AgentContextManager(
        compactor=ContextCompactor(
            token_threshold=settings.context_compaction_token_threshold,
            target_token_threshold=settings.context_compaction_target_token_threshold,
        ),
        enabled=settings.context_compaction_enabled,
    )
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=answer_agent,
        checkpoint_store=checkpoint_store,
        max_steps=settings.agent_max_steps,
        max_revision_iterations=settings.agent_max_revision_iterations,
        event_bus=event_bus,
        run_timeout_sec=settings.agent_run_timeout_sec,
        context_manager=context_manager,
    )

    chat_store_path = settings.chat_store_path
    if disable_external_api:
        chat_store_path = str(Path(tempfile.gettempdir()) / f"mmaa_chat_store_{uuid4().hex}.db")

    return ServiceContainer(
        llm=llm,
        embeddings=embeddings,
        multimodal_embeddings=multimodal_embeddings,
        reranker=reranker_selection.reranker,
        retriever=retriever,
        vector_store=vector_store,
        ingestion=ingestion,
        vision=multimodal_selection.vision,
        video=multimodal_selection.video,
        tools=tools,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        event_bus=event_bus,
        chat_store=SQLiteChatStore(chat_store_path),
        embedding_provider=embedding_selection.provider_name,
        multimodal_embedding_provider=multimodal_embedding_provider_name,
        llm_provider=llm_selection.provider_name,
        reranker_provider=reranker_selection.provider_name,
        multimodal_provider=multimodal_selection.provider_name,
        vector_store_provider=vector_store_provider,
    )


@lru_cache
def get_container() -> ServiceContainer:
    return build_service_container()


def create_test_container() -> ServiceContainer:
    return _build_service_container_internal(
        database_url="",
        embedding_provider_override="deterministic",
        llm_provider_override="heuristic",
        multimodal_provider_override="heuristic",
        disable_external_api=True,
    )
