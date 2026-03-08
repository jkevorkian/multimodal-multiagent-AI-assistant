from __future__ import annotations

from typing import Any

from app.interfaces.vector_store import VectorStore


class FallbackVectorStore:
    def __init__(self, primary: VectorStore, secondary: VectorStore, mirror_writes: bool = True) -> None:
        self._primary = primary
        self._secondary = secondary
        self._mirror_writes = mirror_writes

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        primary_ok = False
        try:
            await self._primary.upsert(ids=ids, vectors=vectors, metadata=metadata)
            primary_ok = True
        except Exception:
            primary_ok = False

        if self._mirror_writes or not primary_ok:
            await self._secondary.upsert(ids=ids, vectors=vectors, metadata=metadata)

    async def search(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        try:
            try:
                primary_results = await self._primary.search(
                    vector=vector,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                )
            except TypeError:
                primary_results = await self._primary.search(vector=vector, top_k=top_k)
            if primary_results:
                return primary_results
        except Exception:
            pass
        try:
            return await self._secondary.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        except TypeError:
            return await self._secondary.search(vector=vector, top_k=top_k)

    async def keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        try:
            try:
                secondary_results = await self._secondary.keyword_search(
                    query=query,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                )
            except TypeError:
                secondary_results = await self._secondary.keyword_search(query=query, top_k=top_k)
            if secondary_results:
                return secondary_results
        except Exception:
            pass

        try:
            try:
                return await self._primary.keyword_search(query=query, top_k=top_k, metadata_filter=metadata_filter)
            except TypeError:
                return await self._primary.keyword_search(query=query, top_k=top_k)
        except Exception:
            return []

    async def list_indexed_sources(
        self,
        limit: int = 200,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        try:
            try:
                secondary_results = await self._secondary.list_indexed_sources(limit=limit, metadata_filter=metadata_filter)
            except TypeError:
                secondary_results = await self._secondary.list_indexed_sources(limit=limit)
            if secondary_results:
                return secondary_results
        except Exception:
            pass

        try:
            try:
                return await self._primary.list_indexed_sources(limit=limit, metadata_filter=metadata_filter)
            except TypeError:
                return await self._primary.list_indexed_sources(limit=limit)
        except Exception:
            return []
