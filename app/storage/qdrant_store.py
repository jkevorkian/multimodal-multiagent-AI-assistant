from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.storage.index_summary import summarize_indexed_sources


@dataclass
class _MetadataRecord:
    id: str
    metadata: dict[str, Any]


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str = "rag_chunks",
        embedding_dimensions: int = 1536,
        multimodal_embedding_dimensions: int | None = None,
        text_vector_name: str = "text_dense",
        multimodal_vector_name: str = "mm_dense",
        enable_named_vectors: bool = True,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout_sec: float = 10.0,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._multimodal_embedding_dimensions = multimodal_embedding_dimensions or embedding_dimensions
        self._text_vector_name = text_vector_name
        self._multimodal_vector_name = multimodal_vector_name
        self._enable_named_vectors = enable_named_vectors
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._timeout_sec = timeout_sec
        self._client: Any | None = None
        self._models: Any | None = None
        self._metadata_cache: dict[str, _MetadataRecord] = {}
        self._supports_named_vectors = False

        self._enabled = self._try_initialize()

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        await self.upsert_named(
            ids=ids,
            vectors_by_name={self._text_vector_name: vectors},
            metadata=metadata,
        )

    async def upsert_named(
        self,
        ids: list[str],
        vectors_by_name: dict[str, list[list[float]]],
        metadata: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not vectors_by_name:
            raise ValueError("vectors_by_name must include at least one vector name")
        if len(metadata) != len(ids):
            raise ValueError("ids and metadata length must match")
        if not self._enabled:
            raise RuntimeError("Qdrant store is not enabled")

        named_items = list(vectors_by_name.items())
        for vector_name, vectors in named_items:
            if len(vectors) != len(ids):
                raise ValueError(f"named vector batch length mismatch for '{vector_name}'")

        assert self._client is not None
        assert self._models is not None

        selected_name = self._pick_preferred_vector_name(vectors_by_name)
        points = []
        for row_index, item_id in enumerate(ids):
            payload = metadata[row_index]
            self._metadata_cache[item_id] = _MetadataRecord(id=item_id, metadata=payload)
            if self._supports_named_vectors:
                point_vector: Any = {name: vectors[row_index] for name, vectors in named_items}
            else:
                point_vector = vectors_by_name[selected_name][row_index]
            points.append(self._models.PointStruct(id=item_id, vector=point_vector, payload=payload))

        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)

    async def search(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        return await self.search_named(
            vector=vector,
            top_k=top_k,
            vector_name=self._text_vector_name,
            metadata_filter=metadata_filter,
        )

    async def search_named(
        self,
        vector: list[float],
        top_k: int,
        vector_name: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        if not self._enabled:
            raise RuntimeError("Qdrant store is not enabled")

        assert self._client is not None
        qdrant_filter = self._build_qdrant_filter(metadata_filter)
        using_name = vector_name if self._supports_named_vectors and vector_name else None
        results = self._search_qdrant(vector=vector, top_k=top_k, qdrant_filter=qdrant_filter, using_name=using_name)

        normalized: list[dict] = []
        for row in results:
            payload = dict(getattr(row, "payload", {}) or {})
            identifier = str(getattr(row, "id", ""))
            if identifier and identifier not in self._metadata_cache:
                self._metadata_cache[identifier] = _MetadataRecord(id=identifier, metadata=payload)
            normalized.append({"id": identifier, "metadata": payload, "score": float(getattr(row, "score", 0.0))})
        return normalized

    async def keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []

        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term]
        if not terms:
            return []

        tokenized_records: list[tuple[_MetadataRecord, list[str]]] = []
        df: dict[str, int] = {term: 0 for term in terms}
        for record in self._metadata_cache.values():
            if not self._metadata_matches(record.metadata, metadata_filter):
                continue
            source = str(record.metadata.get("source", ""))
            snippet = str(record.metadata.get("snippet", ""))
            tokens = re.findall(r"[a-z0-9]+", f"{source} {snippet}".lower())
            tokenized_records.append((record, tokens))
            unique_tokens = set(tokens)
            for term in terms:
                if term in unique_tokens:
                    df[term] += 1

        total_docs = max(1, len(tokenized_records))
        avg_doc_len = sum(len(tokens) for _, tokens in tokenized_records) / total_docs if tokenized_records else 1.0
        k1 = 1.2
        b = 0.75
        scored: list[dict] = []
        for record, tokens in tokenized_records:
            if not tokens:
                continue
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            score = 0.0
            doc_len = len(tokens)
            for term in terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (total_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))

            if score > 0:
                scored.append({"id": record.id, "metadata": record.metadata, "score": float(score)})

        scored.sort(key=lambda row: row["score"], reverse=True)
        return scored[:top_k]

    async def list_indexed_sources(
        self,
        limit: int = 200,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        if limit <= 0:
            return []
        if not self._enabled:
            raise RuntimeError("Qdrant store is not enabled")

        if not self._metadata_cache:
            self._warm_metadata_cache(max_points=max(2000, limit * 20))

        metadata_rows = [
            dict(record.metadata)
            for record in self._metadata_cache.values()
            if self._metadata_matches(record.metadata, metadata_filter)
        ]
        return summarize_indexed_sources(metadata_rows, limit=limit)

    def _try_initialize(self) -> bool:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except Exception:
            return False

        try:
            client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
                timeout=self._timeout_sec,
            )
            created_named_vectors = False
            if not client.collection_exists(self._collection_name):
                if self._enable_named_vectors:
                    client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config={
                            self._text_vector_name: models.VectorParams(
                                size=self._embedding_dimensions,
                                distance=models.Distance.COSINE,
                            ),
                            self._multimodal_vector_name: models.VectorParams(
                                size=self._multimodal_embedding_dimensions,
                                distance=models.Distance.COSINE,
                            ),
                        },
                    )
                    created_named_vectors = True
                else:
                    client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=models.VectorParams(size=self._embedding_dimensions, distance=models.Distance.COSINE),
                    )

            self._client = client
            self._models = models
            if self._enable_named_vectors:
                self._supports_named_vectors = created_named_vectors or self._collection_has_named_vectors()
            else:
                self._supports_named_vectors = False
            return True
        except Exception:
            self._client = None
            self._models = None
            self._supports_named_vectors = False
            return False

    def _collection_has_named_vectors(self) -> bool:
        if self._client is None:
            return False
        try:
            info = self._client.get_collection(self._collection_name)
        except Exception:
            return False
        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        if vectors is None:
            return False
        if isinstance(vectors, dict):
            return self._text_vector_name in vectors or self._multimodal_vector_name in vectors
        candidate = getattr(vectors, "__root__", None)
        if isinstance(candidate, dict):
            return self._text_vector_name in candidate or self._multimodal_vector_name in candidate
        for attr in ("text_dense", "mm_dense"):
            if getattr(vectors, attr, None) is not None:
                return True
        try:
            as_dict = dict(vectors)
        except Exception:
            return False
        return self._text_vector_name in as_dict or self._multimodal_vector_name in as_dict

    def _search_qdrant(
        self,
        *,
        vector: list[float],
        top_k: int,
        qdrant_filter: Any | None,
        using_name: str | None,
    ) -> list[Any]:
        assert self._client is not None

        if hasattr(self._client, "query_points"):
            kwargs: dict[str, Any] = {
                "collection_name": self._collection_name,
                "query": vector,
                "limit": top_k,
                "with_payload": True,
            }
            if qdrant_filter is not None:
                kwargs["query_filter"] = qdrant_filter
            if using_name:
                kwargs["using"] = using_name
            try:
                response = self._client.query_points(**kwargs)
            except TypeError:
                if "query_filter" in kwargs:
                    kwargs["filter"] = kwargs.pop("query_filter")
                try:
                    response = self._client.query_points(**kwargs)
                except TypeError:
                    kwargs.pop("using", None)
                    response = self._client.query_points(**kwargs)
            points = response.points if hasattr(response, "points") else response
            return list(points)

        query_vector: Any = vector
        if using_name:
            query_vector = (using_name, vector)
        kwargs = {
            "collection_name": self._collection_name,
            "query_vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        if qdrant_filter is not None:
            kwargs["query_filter"] = qdrant_filter
        try:
            return list(self._client.search(**kwargs))
        except TypeError:
            if "query_filter" in kwargs:
                kwargs["filter"] = kwargs.pop("query_filter")
            return list(self._client.search(**kwargs))

    def _warm_metadata_cache(self, max_points: int = 5000) -> None:
        if not self._enabled or max_points <= 0:
            return

        assert self._client is not None
        offset: Any = None
        scanned = 0

        while scanned < max_points:
            batch_limit = min(256, max_points - scanned)
            try:
                result = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=batch_limit,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
            except TypeError:
                result = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=batch_limit,
                    with_payload=True,
                    offset=offset,
                )
            except Exception:
                return

            if isinstance(result, tuple):
                points, next_offset = result
            else:
                points = getattr(result, "points", []) or []
                next_offset = getattr(result, "next_page_offset", None)

            if not points:
                break

            for point in points:
                identifier = str(getattr(point, "id", ""))
                payload = dict(getattr(point, "payload", {}) or {})
                if identifier:
                    self._metadata_cache[identifier] = _MetadataRecord(id=identifier, metadata=payload)
                scanned += 1
                if scanned >= max_points:
                    break

            if scanned >= max_points:
                break
            if not next_offset:
                break
            offset = next_offset

    def _build_qdrant_filter(self, metadata_filter: dict[str, Any] | None) -> Any | None:
        if not metadata_filter:
            return None
        assert self._models is not None
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(self._models.FieldCondition(key=key, match=self._models.MatchValue(value=value)))
        return self._models.Filter(must=conditions)

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True

    @staticmethod
    def _pick_preferred_vector_name(vectors_by_name: dict[str, list[list[float]]]) -> str:
        if "text_dense" in vectors_by_name:
            return "text_dense"
        if "mm_dense" in vectors_by_name:
            return "mm_dense"
        return next(iter(vectors_by_name.keys()))
