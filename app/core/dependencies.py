from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.agents import (
    AgentOrchestrator,
    AnalystAgent,
    AnswerAgent,
    InMemoryCheckpointStore,
    NullCheckpointStore,
    ResearchAgent,
)
from app.core.config import settings
from app.interfaces import EmbeddingClient, LLMClient, Reranker, Retriever, Tool, VectorStore, VideoClient, VisionClient
from app.rag import DocumentIngestionService, TextRAGRetriever, build_embedding_client, build_reranker
from app.storage import FallbackVectorStore, PgVectorStore, QdrantVectorStore
from app.tools import ToolRegistry


class GroundedLLMClient:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        if not context:
            return "I do not have indexed context yet. Ingest documents first."
        top_context = " ".join(segment.strip() for segment in context[:2] if segment.strip())
        return f"Grounded answer for '{prompt}': {top_context}"


class StubVisionClient:
    async def analyze_image(self, image_uri: str, prompt: str | None = None) -> str:
        return f"Stub vision analysis for {image_uri}"


class StubVideoClient:
    async def analyze_video(
        self,
        video_uri: str,
        prompt: str | None = None,
        sample_fps: float = 1.0,
        max_frames: int = 32,
    ) -> str:
        return f"Stub video analysis for {video_uri}"


class StubTool:
    name = "stub_tool"

    async def run(self, payload: dict) -> dict:
        return {"status": "ok", "payload": payload}


@dataclass
class ServiceContainer:
    llm: LLMClient
    embeddings: EmbeddingClient
    reranker: Reranker
    retriever: Retriever
    vector_store: VectorStore
    ingestion: DocumentIngestionService
    vision: VisionClient
    video: VideoClient
    tools: list[Tool]
    tool_registry: ToolRegistry
    orchestrator: AgentOrchestrator
    embedding_provider: str
    reranker_provider: str
    vector_store_provider: str


def build_service_container(database_url: str | None = None) -> ServiceContainer:
    embedding_selection = build_embedding_client(
        provider=settings.rag_embedding_provider,
        model_name=settings.rag_embedding_model,
        sentence_transformer_model=settings.rag_sentence_transformer_model,
        api_key=settings.rag_openai_api_key,
        deterministic_dimensions=settings.rag_embedding_dimensions,
        requested_dimensions=settings.rag_requested_embedding_dimensions,
    )
    embeddings = embedding_selection.client
    reranker_selection = build_reranker(
        enabled=settings.rag_reranker_enabled,
        provider=settings.rag_reranker_provider,
        cross_encoder_model=settings.rag_reranker_model,
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
                prefer_grpc=settings.qdrant_prefer_grpc,
                timeout_sec=settings.qdrant_timeout_sec,
            )
            vector_store = FallbackVectorStore(
                primary=qdrant_store,
                secondary=pgvector_store,
                mirror_writes=settings.rag_vector_store_mirror_writes,
            )
            vector_store_provider = "qdrant+pgvector"

    ingestion = DocumentIngestionService(
        embedding_client=embeddings,
        vector_store=vector_store,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        max_source_bytes=settings.rag_max_source_bytes,
    )
    retriever = TextRAGRetriever(
        embedding_client=embeddings,
        vector_store=vector_store,
        reranker=reranker_selection.reranker,
        dense_top_k=settings.rag_dense_top_k,
        lexical_top_k=settings.rag_lexical_top_k,
        rrf_k=settings.rag_rrf_k,
        rerank_pool_size=settings.rag_rerank_pool_size,
    )
    tools: list[Tool] = [StubTool()]
    tool_registry = ToolRegistry(tools)
    research_agent = ResearchAgent(
        retriever=retriever,
        tool_registry=tool_registry,
        retrieval_top_k=settings.agent_retrieval_top_k,
        tool_timeout_sec=settings.agent_tool_timeout_sec,
        tool_retries=settings.agent_tool_retries,
    )
    checkpoint_store = InMemoryCheckpointStore() if settings.agent_checkpoint_enabled else NullCheckpointStore()
    llm = GroundedLLMClient()
    answer_agent = AnswerAgent(llm=llm)
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=answer_agent,
        checkpoint_store=checkpoint_store,
        max_steps=settings.agent_max_steps,
    )

    return ServiceContainer(
        llm=llm,
        embeddings=embeddings,
        reranker=reranker_selection.reranker,
        retriever=retriever,
        vector_store=vector_store,
        ingestion=ingestion,
        vision=StubVisionClient(),
        video=StubVideoClient(),
        tools=tools,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        embedding_provider=embedding_selection.provider_name,
        reranker_provider=reranker_selection.provider_name,
        vector_store_provider=vector_store_provider,
    )


@lru_cache
def get_container() -> ServiceContainer:
    return build_service_container()


def create_test_container() -> ServiceContainer:
    return build_service_container(database_url="")
