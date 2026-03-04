from __future__ import annotations

from app.interfaces.embedding import EmbeddingClient
from app.interfaces.reranker import Reranker
from app.interfaces.vector_store import VectorStore
from app.rag.reranker import NoOpReranker

class TextRAGRetriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        dense_top_k: int = 12,
        lexical_top_k: int = 12,
        rrf_k: int = 60,
        rerank_pool_size: int = 24,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._reranker = reranker or NoOpReranker()
        self._dense_top_k = dense_top_k
        self._lexical_top_k = lexical_top_k
        self._rrf_k = max(1, rrf_k)
        self._rerank_pool_size = max(1, rerank_pool_size)

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []

        dense_results: list[dict] = []
        lexical_results: list[dict] = []
        dense_limit = max(top_k, self._dense_top_k)
        lexical_limit = max(top_k, self._lexical_top_k)

        try:
            query_vector = await self._embedding_client.embed_text(query)
            if any(abs(value) > 0 for value in query_vector):
                dense_results = await self._vector_store.search(query_vector, top_k=dense_limit)
        except Exception:
            dense_results = []

        try:
            lexical_results = await self._vector_store.keyword_search(query=query, top_k=lexical_limit)
        except Exception:
            lexical_results = []

        dense_candidates = [self._normalize_result(item) for item in dense_results]
        lexical_candidates = [self._normalize_result(item) for item in lexical_results]
        fused = self._reciprocal_rank_fusion([dense_candidates, lexical_candidates], max_candidates=self._rerank_pool_size)

        if not fused:
            fallback = dense_candidates or lexical_candidates
            return fallback[:top_k]

        reranked = await self._reranker.rerank(query=query, candidates=fused, top_k=top_k)
        return [self._normalize_result(item) for item in reranked][:top_k]

    def _reciprocal_rank_fusion(self, ranked_lists: list[list[dict]], max_candidates: int) -> list[dict]:
        fused: dict[str, dict] = {}
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                key = self._candidate_key(item)
                score = 1.0 / (self._rrf_k + rank + 1)
                if key not in fused:
                    fused[key] = {**item, "score": score}
                else:
                    fused[key]["score"] = float(fused[key].get("score", 0.0)) + score

        ordered = sorted(fused.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return ordered[:max_candidates]

    @staticmethod
    def _candidate_key(item: dict) -> str:
        source = str(item.get("source", ""))
        chunk_id = str(item.get("chunk_id", ""))
        offset = str(item.get("offset", ""))
        snippet = str(item.get("snippet", ""))[:80]
        return f"{source}::{chunk_id}::{offset}::{snippet}"

    @staticmethod
    def _normalize_result(item: dict) -> dict:
        metadata = item.get("metadata", {})
        if metadata:
            source = metadata.get("source", "unknown")
            snippet = metadata.get("snippet", "")
            chunk_id = metadata.get("chunk_id", -1)
            offset = metadata.get("offset", -1)
            score = item.get("score", 0.0)
        else:
            source = item.get("source", "unknown")
            snippet = item.get("snippet", "")
            chunk_id = item.get("chunk_id", -1)
            offset = item.get("offset", -1)
            score = item.get("score", 0.0)

        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            chunk_id = -1
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            offset = -1

        return {
            "source": str(source),
            "snippet": str(snippet),
            "chunk_id": chunk_id,
            "offset": offset,
            "score": float(score),
        }
