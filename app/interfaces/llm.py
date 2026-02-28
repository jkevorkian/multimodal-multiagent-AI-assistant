from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        ...

