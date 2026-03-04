from __future__ import annotations

from collections import defaultdict
from typing import Any


class NullCheckpointStore:
    async def save(self, trace_id: str, checkpoint: dict[str, Any]) -> None:
        _ = trace_id
        _ = checkpoint

    async def latest(self, trace_id: str) -> dict[str, Any] | None:
        _ = trace_id
        return None


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, list[dict[str, Any]]] = defaultdict(list)

    async def save(self, trace_id: str, checkpoint: dict[str, Any]) -> None:
        self._checkpoints[trace_id].append(checkpoint)

    async def latest(self, trace_id: str) -> dict[str, Any] | None:
        entries = self._checkpoints.get(trace_id, [])
        if not entries:
            return None
        return entries[-1]
