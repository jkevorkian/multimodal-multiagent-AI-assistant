from __future__ import annotations

from dataclasses import dataclass

#TODO: implement pluggable chunking adapters with different chunking approaches (like semantic chunking strategies, or document-structure-based, or recursive, or query-aware chunking, or adaptive chunking, etc) instead of naive fixed-size chunks as a default strategy with no way of extending in the future.


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    chunk_id: int
    offset: int


def chunk_text(text: str, source: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(normalized_text), step)):
        end = start + chunk_size
        snippet = normalized_text[start:end].strip()
        if not snippet:
            continue
        chunks.append(
            Chunk(
                id=f"{source}::chunk-{index}",
                source=source,
                text=snippet,
                chunk_id=index,
                offset=start,
            )
        )
        if end >= len(normalized_text):
            break

    return chunks
