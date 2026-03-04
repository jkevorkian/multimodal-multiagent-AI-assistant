from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    async def embed_text(self, text: str) -> list[float]:
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
