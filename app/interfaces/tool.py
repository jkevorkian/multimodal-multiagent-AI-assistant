from __future__ import annotations

from typing import Protocol


class Tool(Protocol):
    name: str

    async def run(self, payload: dict) -> dict:
        ...

