from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ...

