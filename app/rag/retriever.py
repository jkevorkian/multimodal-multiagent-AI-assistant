from __future__ import annotations

from typing import Any

from app.interfaces.embedding import EmbeddingClient
from app.interfaces.multimodal_embedding import MultimodalEmbeddingClient
from app.interfaces.reranker import Reranker
from app.interfaces.vector_store import VectorStore
from app.rag.reranker import NoOpReranker


class TextRAGRetriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        multimodal_embedding_client: MultimodalEmbeddingClient | None = None,
        reranker: Reranker | None = None,
        dense_top_k: int = 12,
        lexical_top_k: int = 12,
        rrf_k: int = 60,
        rerank_pool_size: int = 24,
        text_vector_name: str = "text_dense",
        multimodal_vector_name: str = "mm_dense",
        use_text_dense_branch: bool = True,
    ) -> None:
        self._embedding_client = embedding_client
        self._multimodal_embedding_client = multimodal_embedding_client
        self._vector_store = vector_store
        self._reranker = reranker or NoOpReranker()
        self._dense_top_k = dense_top_k
        self._lexical_top_k = lexical_top_k
        self._rrf_k = max(1, rrf_k)
        self._rerank_pool_size = max(1, rerank_pool_size)
        self._text_vector_name = text_vector_name
        self._multimodal_vector_name = multimodal_vector_name
        self._use_text_dense_branch = use_text_dense_branch

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []

        dense_limit = max(top_k, self._dense_top_k)
        lexical_limit = max(top_k, self._lexical_top_k)
        ranked_dense_lists: list[list[dict]] = []

        multimodal_candidates = await self._retrieve_multimodal_dense(
            query=query,
            top_k=dense_limit,
            metadata_filter=metadata_filter,
        )
        if multimodal_candidates:
            ranked_dense_lists.append(multimodal_candidates)

        text_candidates = await self._retrieve_text_dense(
            query=query,
            top_k=dense_limit,
            metadata_filter=metadata_filter,
        )
        if text_candidates:
            should_keep_text_branch = self._use_text_dense_branch or not multimodal_candidates
            if should_keep_text_branch:
                ranked_dense_lists.append(text_candidates)

        try:
            try:
                lexical_results = await self._vector_store.keyword_search(
                    query=query,
                    top_k=lexical_limit,
                    metadata_filter=metadata_filter,
                )
            except TypeError:
                lexical_results = await self._vector_store.keyword_search(query=query, top_k=lexical_limit)
        except Exception:
            lexical_results = []

        lexical_candidates = [self._normalize_result(item) for item in lexical_results]
        ranked_lists = [*ranked_dense_lists]
        if lexical_candidates:
            ranked_lists.append(lexical_candidates)
        fused = self._reciprocal_rank_fusion(ranked_lists, max_candidates=self._rerank_pool_size)

        if not fused:
            fallback = []
            for candidate_list in ranked_lists:
                fallback.extend(candidate_list)
            return fallback[:top_k]

        reranked = await self._reranker.rerank(query=query, candidates=fused, top_k=top_k)
        return [self._normalize_result(item) for item in reranked][:top_k]

    async def _retrieve_multimodal_dense(
        self,
        *,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict]:
        if self._multimodal_embedding_client is None:
            return []
        try:
            query_vector = await self._multimodal_embedding_client.embed_query(query)
        except Exception:
            return []
        if not any(abs(value) > 0 for value in query_vector):
            return []
        dense_results = await self._search_dense(
            vector=query_vector,
            top_k=top_k,
            vector_name=self._multimodal_vector_name,
            metadata_filter=metadata_filter,
        )
        return [self._normalize_result(item) for item in dense_results]

    async def _retrieve_text_dense(
        self,
        *,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict]:
        try:
            query_vector = await self._embedding_client.embed_text(query)
        except Exception:
            return []
        if not any(abs(value) > 0 for value in query_vector):
            return []
        dense_results = await self._search_dense(
            vector=query_vector,
            top_k=top_k,
            vector_name=self._text_vector_name,
            metadata_filter=metadata_filter,
        )
        return [self._normalize_result(item) for item in dense_results]

    async def _search_dense(
        self,
        *,
        vector: list[float],
        top_k: int,
        vector_name: str,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict]:
        if hasattr(self._vector_store, "search_named"):
            try:
                return await self._vector_store.search_named(
                    vector=vector,
                    top_k=top_k,
                    vector_name=vector_name,
                    metadata_filter=metadata_filter,
                )
            except TypeError:
                try:
                    return await self._vector_store.search_named(
                        vector=vector,
                        top_k=top_k,
                        vector_name=vector_name,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        try:
            return await self._vector_store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        except TypeError:
            return await self._vector_store.search(vector=vector, top_k=top_k)
        except Exception:
            return []

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
            modality = metadata.get("modality", "text")
            timestamp_sec = metadata.get("timestamp_sec")
            frame_index = metadata.get("frame_index")
            score = item.get("score", 0.0)
        else:
            source = item.get("source", "unknown")
            snippet = item.get("snippet", "")
            chunk_id = item.get("chunk_id", -1)
            offset = item.get("offset", -1)
            modality = item.get("modality", "text")
            timestamp_sec = item.get("timestamp_sec")
            frame_index = item.get("frame_index")
            score = item.get("score", 0.0)

        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            chunk_id = -1
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            offset = -1

        normalized = {
            "source": str(source),
            "snippet": str(snippet),
            "chunk_id": chunk_id,
            "offset": offset,
            "score": float(score),
            "modality": str(modality),
        }
        if timestamp_sec is not None:
            normalized["timestamp_sec"] = timestamp_sec
        if frame_index is not None:
            normalized["frame_index"] = frame_index
        return normalized
