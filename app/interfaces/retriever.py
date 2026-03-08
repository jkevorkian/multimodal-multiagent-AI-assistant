from __future__ import annotations

from typing import Any
from typing import Protocol


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        ...
