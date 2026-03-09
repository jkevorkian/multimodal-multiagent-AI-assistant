from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any

from app.core.dependencies import create_test_container
from app.rag import build_embedding_client, build_multimodal_embedding_client
from app.rag.embeddings import cosine_similarity
from app.rag.ingestion import DocumentIngestionService
from app.rag.retriever import TextRAGRetriever
from app.storage.qdrant_store import QdrantVectorStore


@dataclass
class _Row:
    id: str
    vectors_by_name: dict[str, list[float]]
    metadata: dict[str, Any]


class _NamedVectorMemoryStore:
    def __init__(self) -> None:
        self.rows: list[_Row] = []
        self.named_upsert_calls = 0

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        for row_id, vector, payload in zip(ids, vectors, metadata, strict=True):
            self.rows.append(_Row(id=row_id, vectors_by_name={"text_dense": vector}, metadata=payload))

    async def upsert_named(
        self,
        ids: list[str],
        vectors_by_name: dict[str, list[list[float]]],
        metadata: list[dict[str, Any]],
    ) -> None:
        self.named_upsert_calls += 1
        for row_index, row_id in enumerate(ids):
            per_row_vectors = {name: vectors[row_index] for name, vectors in vectors_by_name.items()}
            self.rows.append(_Row(id=row_id, vectors_by_name=per_row_vectors, metadata=metadata[row_index]))

    async def search(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        return await self.search_named(vector=vector, top_k=top_k, vector_name="text_dense", metadata_filter=metadata_filter)

    async def search_named(
        self,
        vector: list[float],
        top_k: int,
        vector_name: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        scored = []
        for row in self.rows:
            if not self._matches(row.metadata, metadata_filter):
                continue
            row_vector = row.vectors_by_name.get(vector_name)
            if row_vector is None:
                row_vector = next(iter(row.vectors_by_name.values()))
            scored.append(
                {
                    "id": row.id,
                    "metadata": row.metadata,
                    "score": cosine_similarity(vector, row_vector),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term]
        if not terms:
            return []
        scored: list[dict] = []
        for row in self.rows:
            if not self._matches(row.metadata, metadata_filter):
                continue
            snippet = str(row.metadata.get("snippet", "")).lower()
            source = str(row.metadata.get("source", "")).lower()
            combined = f"{source} {snippet}"
            score = float(sum(combined.count(term) for term in terms))
            if score > 0:
                scored.append({"id": row.id, "metadata": row.metadata, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def list_indexed_sources(
        self,
        limit: int = 200,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            if not self._matches(row.metadata, metadata_filter):
                continue
            source = str(row.metadata.get("source", "unknown"))
            bucket = grouped.setdefault(
                source,
                {
                    "source": source,
                    "modality": row.metadata.get("modality", "text"),
                    "chunk_count": 0,
                },
            )
            bucket["chunk_count"] += 1
        return list(grouped.values())[:limit]

    @staticmethod
    def _matches(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True


def test_multimodal_embedding_builder_auto_falls_back_to_deterministic() -> None:
    selection = build_multimodal_embedding_client(
        provider="auto",
        api_key=None,
        base_url=None,
        deterministic_dimensions=40,
    )
    assert selection.provider_name == "deterministic"
    vector = asyncio.run(selection.client.embed_query("cross modal retrieval baseline"))
    assert len(vector) == 40
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-3)


def test_m53_ingestion_writes_named_text_and_mm_vectors(tmp_path) -> None:
    source_text = tmp_path / "brief.txt"
    source_text.write_text("Control room screenshot highlights the cooling pump status.", encoding="utf-8")

    store = _NamedVectorMemoryStore()
    text_embeddings = build_embedding_client(provider="deterministic", deterministic_dimensions=32).client
    multimodal_embeddings = build_multimodal_embedding_client(
        provider="deterministic",
        deterministic_dimensions=32,
    ).client
    ingestion = DocumentIngestionService(
        embedding_client=text_embeddings,
        multimodal_embedding_client=multimodal_embeddings,
        vector_store=store,
        chunk_size=100,
        chunk_overlap=10,
    )

    summary = asyncio.run(
        ingestion.ingest(
            sources=[source_text.as_uri(), "https://assets.example.com/pump-panel.png", "https://assets.example.com/pump-tour.mp4"],
            source_type="mixed",
        )
    )
    assert summary.accepted_sources == 3
    assert summary.indexed_chunks >= 3
    assert store.named_upsert_calls == 1
    assert store.rows
    assert all("text_dense" in row.vectors_by_name for row in store.rows)
    assert all("mm_dense" in row.vectors_by_name for row in store.rows)
    modalities = {str(row.metadata.get("modality")) for row in store.rows}
    assert {"text", "image", "video"}.issubset(modalities)


def test_m53_cross_modal_retrieval_text_image_video(tmp_path) -> None:
    source_text = tmp_path / "notes.txt"
    source_text.write_text("Mission Aurora launch checklist confirms ignition sequence.", encoding="utf-8")

    store = _NamedVectorMemoryStore()
    text_embeddings = build_embedding_client(provider="deterministic", deterministic_dimensions=48).client
    multimodal_embeddings = build_multimodal_embedding_client(
        provider="deterministic",
        deterministic_dimensions=48,
    ).client
    ingestion = DocumentIngestionService(
        embedding_client=text_embeddings,
        multimodal_embedding_client=multimodal_embeddings,
        vector_store=store,
        chunk_size=120,
        chunk_overlap=20,
    )
    asyncio.run(
        ingestion.ingest(
            sources=[
                source_text.as_uri(),
                "https://assets.example.com/aurora-dashboard.png",
                "https://assets.example.com/aurora-briefing.mp4",
            ],
            source_type="mixed",
        )
    )
    retriever = TextRAGRetriever(
        embedding_client=text_embeddings,
        multimodal_embedding_client=multimodal_embeddings,
        vector_store=store,
        dense_top_k=8,
        lexical_top_k=8,
        rerank_pool_size=12,
    )

    dashboard_hits = asyncio.run(retriever.retrieve("aurora dashboard screenshot", top_k=3))
    video_hits = asyncio.run(retriever.retrieve("aurora briefing clip", top_k=3))
    launch_hits = asyncio.run(retriever.retrieve("ignition sequence checklist", top_k=3))

    assert any(hit.get("modality") == "image" for hit in dashboard_hits)
    assert any(hit.get("modality") == "video" for hit in video_hits)
    assert any(hit.get("modality") == "text" for hit in launch_hits)


def test_m53_video_ingestion_uses_direct_vl_fallback_without_frame_decode() -> None:
    container = create_test_container()
    summary = asyncio.run(
        container.ingestion.ingest(
            sources=["https://assets.example.com/aurora-briefing.mp4"],
            source_type="mixed",
        )
    )
    assert summary.accepted_sources == 1
    assert summary.indexed_chunks >= 1
    indexed = asyncio.run(container.vector_store.list_indexed_sources(limit=20))
    assert any(str(item.get("modality")) == "video" for item in indexed)


def test_m53_reranker_can_lift_target_candidate() -> None:
    class _BranchStore:
        async def search_named(  # noqa: ARG002
            self,
            vector: list[float],
            top_k: int,
            vector_name: str,
            metadata_filter: dict[str, Any] | None = None,
        ) -> list[dict]:
            if vector_name == "mm_dense":
                return [
                    {
                        "id": "wrong-first",
                        "metadata": {
                            "source": "other_doc",
                            "chunk_id": 0,
                            "offset": 0,
                            "snippet": "unrelated panel screenshot",
                            "modality": "image",
                        },
                        "score": 0.99,
                    },
                    {
                        "id": "target-second",
                        "metadata": {
                            "source": "target_doc",
                            "chunk_id": 1,
                            "offset": 10,
                            "snippet": "aurora mission dashboard",
                            "modality": "image",
                        },
                        "score": 0.98,
                    },
                ][:top_k]
            return []

        async def search(self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[dict]:  # noqa: ARG002
            return []

        async def keyword_search(self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[dict]:  # noqa: ARG002
            return []

        async def list_indexed_sources(self, limit: int = 200, metadata_filter: dict[str, Any] | None = None) -> list[dict]:  # noqa: ARG002
            return []

    class _Reranker:
        def __init__(self) -> None:
            self.called = False

        async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:  # noqa: ARG002
            self.called = True
            ordered = sorted(candidates, key=lambda item: 1 if item.get("source") == "target_doc" else 0, reverse=True)
            return ordered[:top_k]

    reranker = _Reranker()
    text_embeddings = build_embedding_client(provider="deterministic", deterministic_dimensions=32).client
    multimodal_embeddings = build_multimodal_embedding_client(
        provider="deterministic",
        deterministic_dimensions=32,
    ).client
    retriever = TextRAGRetriever(
        embedding_client=text_embeddings,
        multimodal_embedding_client=multimodal_embeddings,
        vector_store=_BranchStore(),  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        dense_top_k=4,
        lexical_top_k=2,
        rerank_pool_size=6,
        use_text_dense_branch=False,
    )

    hits = asyncio.run(retriever.retrieve("aurora dashboard", top_k=2))
    assert reranker.called is True
    assert hits
    assert hits[0]["source"] == "target_doc"


def test_m53_qdrant_named_vector_routing_contract() -> None:
    class _FakePointStruct:
        def __init__(self, id: str, vector: Any, payload: dict[str, Any]) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    class _FakeMatchValue:
        def __init__(self, value: Any) -> None:
            self.value = value

    class _FakeFieldCondition:
        def __init__(self, key: str, match: _FakeMatchValue) -> None:
            self.key = key
            self.match = match

    class _FakeFilter:
        def __init__(self, must: list[_FakeFieldCondition]) -> None:
            self.must = must

    class _FakeModels:
        PointStruct = _FakePointStruct
        MatchValue = _FakeMatchValue
        FieldCondition = _FakeFieldCondition
        Filter = _FakeFilter

    class _FakeHit:
        def __init__(self) -> None:
            self.id = "p-1"
            self.payload = {"source": "doc.txt", "snippet": "retrieved snippet", "chunk_id": 0, "offset": 0}
            self.score = 0.87

    class _FakeQueryResponse:
        def __init__(self) -> None:
            self.points = [_FakeHit()]

    class _FakeClient:
        def __init__(self) -> None:
            self.upsert_points: list[Any] = []
            self.query_kwargs: list[dict[str, Any]] = []

        def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:  # noqa: ARG002
            self.upsert_points = points

        def query_points(self, **kwargs: Any) -> _FakeQueryResponse:
            self.query_kwargs.append(kwargs)
            return _FakeQueryResponse()

    store = QdrantVectorStore(url="http://localhost:6333")
    fake_client = _FakeClient()
    store._enabled = True
    store._client = fake_client
    store._models = _FakeModels
    store._supports_named_vectors = True

    asyncio.run(
        store.upsert_named(
            ids=["p-1"],
            vectors_by_name={"text_dense": [[0.1, 0.2]], "mm_dense": [[0.3, 0.4]]},
            metadata=[{"source": "doc.txt", "chunk_id": 0, "offset": 0, "snippet": "retrieved snippet"}],
        )
    )
    assert fake_client.upsert_points
    assert isinstance(fake_client.upsert_points[0].vector, dict)
    assert {"text_dense", "mm_dense"}.issubset(fake_client.upsert_points[0].vector.keys())

    results = asyncio.run(
        store.search_named(
            vector=[0.3, 0.4],
            top_k=2,
            vector_name="mm_dense",
            metadata_filter={"source": "doc.txt"},
        )
    )
    assert results
    assert fake_client.query_kwargs
    assert fake_client.query_kwargs[-1].get("using") == "mm_dense"
