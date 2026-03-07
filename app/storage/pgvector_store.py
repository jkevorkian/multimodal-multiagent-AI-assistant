from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.rag.embeddings import cosine_similarity
from app.storage.index_summary import summarize_indexed_sources


@dataclass
class _MemoryRecord:
    id: str
    vector: list[float]
    metadata: dict[str, Any]


class PgVectorStore:
    def __init__(
        self,
        database_url: str | None = None,
        table_name: str = "rag_chunks",
        embedding_dimensions: int = 64,
    ) -> None:
        self._database_url = database_url
        self._table_name = table_name if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name) else "rag_chunks"
        self._embedding_dimensions = embedding_dimensions
        self._records: dict[str, _MemoryRecord] = {}
        self._use_postgres = False
        self._connection: Any = None
        self._psycopg = None
        self._try_enable_postgres()

    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("ids, vectors, and metadata length must match")

        if self._use_postgres:
            try:
                self._upsert_postgres(ids, vectors, metadata)
                return
            except Exception:
                self._use_postgres = False

        for item_id, vector, item_metadata in zip(ids, vectors, metadata, strict=True):
            self._records[item_id] = _MemoryRecord(id=item_id, vector=vector, metadata=item_metadata)

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        if top_k <= 0:
            return []

        if self._use_postgres:
            try:
                return self._search_postgres(vector, top_k)
            except Exception:
                self._use_postgres = False

        scored = [
            {
                "id": record.id,
                "metadata": record.metadata,
                "score": cosine_similarity(vector, record.vector),
            }
            for record in self._records.values()
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def keyword_search(self, query: str, top_k: int) -> list[dict]:
        if top_k <= 0:
            return []

        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term]
        if not terms:
            return []

        if self._use_postgres:
            try:
                return self._keyword_search_postgres(query, top_k)
            except Exception:
                self._use_postgres = False

        tokenized_records: list[tuple[_MemoryRecord, list[str]]] = []
        df: dict[str, int] = {term: 0 for term in terms}
        for record in self._records.values():
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
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def list_indexed_sources(self, limit: int = 200) -> list[dict]:
        if limit <= 0:
            return []

        metadata_rows: list[dict[str, Any]] = []
        if self._use_postgres:
            try:
                metadata_rows = self._list_metadata_postgres()
            except Exception:
                self._use_postgres = False

        if not metadata_rows:
            metadata_rows = [dict(record.metadata) for record in self._records.values()]

        return summarize_indexed_sources(metadata_rows, limit=limit)

    def _try_enable_postgres(self) -> None:
        if not self._database_url:
            return

        try:
            import psycopg  # type: ignore
        except Exception:
            return

        try:
            connection = psycopg.connect(self._database_url, autocommit=True)
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({self._embedding_dimensions}),
                        metadata JSONB NOT NULL
                    );
                    """
                )
            self._psycopg = psycopg
            self._connection = connection
            self._use_postgres = True
        except Exception:
            self._psycopg = None
            self._connection = None
            self._use_postgres = False

    def _upsert_postgres(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        assert self._connection is not None
        assert self._psycopg is not None

        statement = f"""
            INSERT INTO {self._table_name} (id, embedding, metadata)
            VALUES (%s, %s::vector, %s::jsonb)
            ON CONFLICT (id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """
        json_dumps = self._psycopg.types.json.Jsonb
        with self._connection.cursor() as cursor:
            for item_id, vector, item_metadata in zip(ids, vectors, metadata, strict=True):
                cursor.execute(statement, (item_id, self._vector_literal(vector), json_dumps(item_metadata)))

    def _search_postgres(self, vector: list[float], top_k: int) -> list[dict]:
        assert self._connection is not None

        statement = f"""
            SELECT id, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM {self._table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vector_literal = self._vector_literal(vector)
        with self._connection.cursor() as cursor:
            cursor.execute(statement, (vector_literal, vector_literal, top_k))
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "metadata": row[1] or {},
                "score": float(row[2] or 0.0),
            }
            for row in rows
        ]

    def _keyword_search_postgres(self, query: str, top_k: int) -> list[dict]:
        assert self._connection is not None

        statement = f"""
            SELECT
                id,
                metadata,
                ts_rank_cd(
                    to_tsvector('simple', coalesce(metadata->>'source', '') || ' ' || coalesce(metadata->>'snippet', '')),
                    plainto_tsquery('simple', %s)
                ) AS score
            FROM {self._table_name}
            WHERE to_tsvector('simple', coalesce(metadata->>'source', '') || ' ' || coalesce(metadata->>'snippet', ''))
                  @@ plainto_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(statement, (query, query, top_k))
            rows = cursor.fetchall()
        return [{"id": row[0], "metadata": row[1] or {}, "score": float(row[2] or 0.0)} for row in rows]

    def _list_metadata_postgres(self) -> list[dict[str, Any]]:
        assert self._connection is not None

        statement = f"SELECT metadata FROM {self._table_name}"
        with self._connection.cursor() as cursor:
            cursor.execute(statement)
            rows = cursor.fetchall()
        return [dict(row[0] or {}) for row in rows]

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
