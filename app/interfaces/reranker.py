from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        ...
