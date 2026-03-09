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

    async def upsert_named(
        self,
        ids: list[str],
        vectors_by_name: dict[str, list[list[float]]],
        metadata: list[dict[str, Any]],
    ) -> None:
        selected_name = self._pick_named_vector(vectors_by_name)
        selected_vectors = vectors_by_name[selected_name]
        primary_ok = False
        try:
            if hasattr(self._primary, "upsert_named"):
                await self._primary.upsert_named(ids=ids, vectors_by_name=vectors_by_name, metadata=metadata)  # type: ignore[attr-defined]
            else:
                await self._primary.upsert(ids=ids, vectors=selected_vectors, metadata=metadata)
            primary_ok = True
        except Exception:
            primary_ok = False

        if self._mirror_writes or not primary_ok:
            if hasattr(self._secondary, "upsert_named"):
                await self._secondary.upsert_named(ids=ids, vectors_by_name=vectors_by_name, metadata=metadata)  # type: ignore[attr-defined]
            else:
                await self._secondary.upsert(ids=ids, vectors=selected_vectors, metadata=metadata)

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

    async def search_named(
        self,
        vector: list[float],
        top_k: int,
        vector_name: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        try:
            if hasattr(self._primary, "search_named"):
                try:
                    primary_results = await self._primary.search_named(  # type: ignore[attr-defined]
                        vector=vector,
                        top_k=top_k,
                        vector_name=vector_name,
                        metadata_filter=metadata_filter,
                    )
                except TypeError:
                    primary_results = await self._primary.search_named(  # type: ignore[attr-defined]
                        vector=vector,
                        top_k=top_k,
                        vector_name=vector_name,
                    )
            else:
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
            if hasattr(self._secondary, "search_named"):
                try:
                    return await self._secondary.search_named(  # type: ignore[attr-defined]
                        vector=vector,
                        top_k=top_k,
                        vector_name=vector_name,
                        metadata_filter=metadata_filter,
                    )
                except TypeError:
                    return await self._secondary.search_named(  # type: ignore[attr-defined]
                        vector=vector,
                        top_k=top_k,
                        vector_name=vector_name,
                    )
            try:
                return await self._secondary.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
            except TypeError:
                return await self._secondary.search(vector=vector, top_k=top_k)
        except Exception:
            return []

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

    @staticmethod
    def _pick_named_vector(vectors_by_name: dict[str, list[list[float]]]) -> str:
        if "text_dense" in vectors_by_name:
            return "text_dense"
        if "mm_dense" in vectors_by_name:
            return "mm_dense"
        return next(iter(vectors_by_name.keys()))
