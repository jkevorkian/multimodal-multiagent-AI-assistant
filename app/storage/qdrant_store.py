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
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout_sec: float = 10.0,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._timeout_sec = timeout_sec
        self._client: Any | None = None
        self._models: Any | None = None
        self._metadata_cache: dict[str, _MetadataRecord] = {}

        self._enabled = self._try_initialize()

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("ids, vectors, and metadata length must match")
        if not self._enabled:
            raise RuntimeError("Qdrant store is not enabled")

        assert self._client is not None
        assert self._models is not None

        points = []
        for item_id, vector, item_metadata in zip(ids, vectors, metadata, strict=True):
            self._metadata_cache[item_id] = _MetadataRecord(id=item_id, metadata=item_metadata)
            points.append(self._models.PointStruct(id=item_id, vector=vector, payload=item_metadata))

        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        if not self._enabled:
            raise RuntimeError("Qdrant store is not enabled")

        assert self._client is not None
        results: list[Any]

        if hasattr(self._client, "query_points"):
            query_response = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            points = query_response.points if hasattr(query_response, "points") else query_response
            results = list(points)
        else:
            results = list(
                self._client.search(
                    collection_name=self._collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True,
                )
            )

        normalized: list[dict] = []
        for row in results:
            payload = dict(getattr(row, "payload", {}) or {})
            identifier = str(getattr(row, "id", ""))
            if identifier and identifier not in self._metadata_cache:
                self._metadata_cache[identifier] = _MetadataRecord(id=identifier, metadata=payload)
            normalized.append({"id": identifier, "metadata": payload, "score": float(getattr(row, "score", 0.0))})
        return normalized

    async def keyword_search(self, query: str, top_k: int) -> list[dict]:
        if top_k <= 0:
            return []

        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term]
        if not terms:
            return []

        tokenized_records: list[tuple[_MetadataRecord, list[str]]] = []
        df: dict[str, int] = {term: 0 for term in terms}
        for record in self._metadata_cache.values():
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

    async def list_indexed_sources(self, limit: int = 200) -> list[dict]:
        if limit <= 0:
            return []
        if not self._enabled:
            raise RuntimeError("Qdrant store is not enabled")

        if not self._metadata_cache:
            self._warm_metadata_cache(max_points=max(2000, limit * 20))

        metadata_rows = [dict(record.metadata) for record in self._metadata_cache.values()]
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
            if not client.collection_exists(self._collection_name):
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(size=self._embedding_dimensions, distance=models.Distance.COSINE),
                )
            self._client = client
            self._models = models
            return True
        except Exception:
            self._client = None
            self._models = None
            return False

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
