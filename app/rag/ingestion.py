from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from app.interfaces.embedding import EmbeddingClient
from app.interfaces.vector_store import VectorStore
from app.rag.chunking import chunk_text


@dataclass(frozen=True)
class IngestionSummary:
    accepted_sources: int
    indexed_chunks: int


class DocumentIngestionService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        max_source_bytes: int = 2_000_000,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_source_bytes = max_source_bytes

    async def ingest(self, sources: list[str], source_type: str = "mixed") -> IngestionSummary:
        ids: list[str] = []
        vectors: list[list[float]] = []
        metadata: list[dict] = []
        accepted_sources = 0

        for source in sources:
            try:
                text = self._load_source_text(source, source_type)
            except Exception:
                continue
            chunks = chunk_text(
                text=text,
                source=source,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            if not chunks:
                continue

            accepted_sources += 1
            vectors_for_chunks = await self._embedding_client.embed_texts([chunk.text for chunk in chunks])
            if len(vectors_for_chunks) != len(chunks):
                raise ValueError("embedding provider returned mismatched batch size")
            for chunk, vector in zip(chunks, vectors_for_chunks, strict=True):
                ids.append(chunk.id)
                vectors.append(vector)
                metadata.append(
                    {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "offset": chunk.offset,
                        "snippet": chunk.text,
                    }
                )

        if ids:
            await self._vector_store.upsert(ids=ids, vectors=vectors, metadata=metadata)

        return IngestionSummary(accepted_sources=accepted_sources, indexed_chunks=len(ids))

    def _load_source_text(self, source: str, source_type: str) -> str:
        if source.startswith(("http://", "https://")):
            return self._load_url(source=source, source_type=source_type)
        return self._load_local_file(source=source, source_type=source_type)

    def _load_local_file(self, source: str, source_type: str) -> str:
        path_value = source
        if source.startswith("file://"):
            parsed = urlparse(source)
            path_value = unquote(parsed.path)
            if re.match(r"^/[A-Za-z]:/", path_value):
                path_value = path_value[1:]

        path = Path(path_value)
        data = path.read_bytes()
        self._validate_size(data)
        is_pdf = source_type == "pdf" or path.suffix.lower() == ".pdf"
        return self._decode_source_bytes(data, is_pdf=is_pdf)

    def _load_url(self, source: str, source_type: str) -> str:
        request = Request(source, headers={"User-Agent": "MMAA-Ingest/0.1"})
        with urlopen(request, timeout=10) as response:
            data = response.read(self._max_source_bytes + 1)
            content_type = response.headers.get("Content-Type", "")
        self._validate_size(data)
        is_pdf = source_type == "pdf" or source.lower().endswith(".pdf") or "application/pdf" in content_type
        return self._decode_source_bytes(data, is_pdf=is_pdf)

    def _validate_size(self, data: bytes) -> None:
        if len(data) > self._max_source_bytes:
            raise ValueError("source exceeds max allowed size")

    def _decode_source_bytes(self, data: bytes, is_pdf: bool) -> str:
        if is_pdf:
            pdf_text = self._extract_pdf_like_text(data)
            if pdf_text:
                return pdf_text

        for encoding in ("utf-8", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")

    def _extract_pdf_like_text(self, data: bytes) -> str:
        # Best-effort extraction without external dependencies.
        matches = re.findall(rb"\(([^()]{1,300})\)", data)
        if matches:
            return " ".join(match.decode("latin-1", errors="ignore") for match in matches)
        return data.decode("latin-1", errors="ignore")
