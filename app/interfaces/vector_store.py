from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    async def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        ...

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        ...

