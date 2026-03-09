from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import HealthResponse, RuntimeModelEntry, RuntimeModelsResponse, Trace
from app.core.config import settings
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@router.get("/health/models", response_model=RuntimeModelsResponse)
async def health_models(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> RuntimeModelsResponse:
    transcript_provider = "whisper" if settings.multimodal_video_audio_transcription_enabled else "disabled"
    transcript_model = (
        settings.multimodal_video_audio_transcription_model if settings.multimodal_video_audio_transcription_enabled else None
    )
    mm_embedding_model = settings.rag_multimodal_embedding_model if settings.rag_multimodal_enabled else None
    reranker_model = settings.rag_reranker_model if settings.rag_reranker_enabled else None

    entries = [
        RuntimeModelEntry(
            subsystem="ingestion",
            component="text_embedding",
            provider=container.embedding_provider,
            model=settings.rag_embedding_model,
            details=[f"chunk={settings.rag_chunk_size}", f"overlap={settings.rag_chunk_overlap}"],
        ),
        RuntimeModelEntry(
            subsystem="ingestion",
            component="multimodal_embedding",
            provider=container.multimodal_embedding_provider,
            model=mm_embedding_model,
            details=[f"enabled={settings.rag_multimodal_enabled}"],
        ),
        RuntimeModelEntry(
            subsystem="ingestion",
            component="video_transcription",
            provider=transcript_provider,
            model=transcript_model,
            details=[f"enabled={settings.multimodal_video_audio_transcription_enabled}"],
        ),
        RuntimeModelEntry(
            subsystem="rag",
            component="retriever",
            provider="hybrid_dense_lexical_rrf",
            model=None,
            details=[f"dense_top_k={settings.rag_dense_top_k}", f"lexical_top_k={settings.rag_lexical_top_k}"],
        ),
        RuntimeModelEntry(
            subsystem="rag",
            component="reranker",
            provider=container.reranker_provider,
            model=reranker_model,
            details=[f"enabled={settings.rag_reranker_enabled}"],
        ),
        RuntimeModelEntry(
            subsystem="rag",
            component="vector_store",
            provider=container.vector_store_provider,
            model=settings.qdrant_collection_name if "qdrant" in container.vector_store_provider else settings.rag_table_name,
            details=[],
        ),
        RuntimeModelEntry(
            subsystem="agents",
            component="answer_llm",
            provider=container.llm_provider,
            model=settings.llm_model,
            details=[
                f"max_steps={settings.agent_max_steps}",
                f"tool_budget={settings.agent_tool_budget}",
                f"tool_timeout_sec={settings.agent_tool_timeout_sec}",
                f"retrieval_top_k={settings.agent_retrieval_top_k}",
                f"retry_on_empty_retrieval={settings.agent_retry_on_empty_retrieval}",
                f"retry_empty_retrieval_top_k={settings.agent_retry_empty_retrieval_top_k}",
            ],
        ),
        RuntimeModelEntry(
            subsystem="multimodal",
            component="vision_video_analysis",
            provider=container.multimodal_provider,
            model=settings.multimodal_vision_model,
            details=[
                f"video_sample_fps={settings.multimodal_video_sample_fps}",
                f"video_max_frames={settings.multimodal_video_max_frames}",
            ],
        ),
    ]

    return RuntimeModelsResponse(
        entries=entries,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
