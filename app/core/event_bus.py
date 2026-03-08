from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.agents.progress_mapper import ProgressMapper
from app.contracts.runtime_events import RunStatusSnapshot, RuntimeEvent, RuntimeEventType


@dataclass
class _RunEventStream:
    sequence: int = 0
    events: list[RuntimeEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    latest_status: RunStatusSnapshot | None = None


class InMemoryEventBus:
    def __init__(self, max_events_per_run: int = 500) -> None:
        self._max_events_per_run = max(max_events_per_run, 100)
        self._streams: dict[str, _RunEventStream] = {}
        self._streams_lock = asyncio.Lock()
        self._progress_mapper = ProgressMapper()

    async def emit(
        self,
        *,
        run_id: str,
        trace_id: str,
        event_type: RuntimeEventType,
        status_text: str,
        agent: str | None = None,
        tool: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        stream = await self._get_or_create_stream(run_id)
        async with stream.condition:
            stream.sequence += 1
            event = RuntimeEvent(
                run_id=run_id,
                trace_id=trace_id,
                sequence_number=stream.sequence,
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                status_text=status_text,
                agent=agent,
                tool=tool,
                metadata=metadata or {},
            )
            stream.events.append(event)
            if len(stream.events) > self._max_events_per_run:
                overflow = len(stream.events) - self._max_events_per_run
                del stream.events[:overflow]
            stream.latest_status = self._progress_mapper.to_status_snapshot(event)
            stream.condition.notify_all()
            return event

    async def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> list[RuntimeEvent]:
        stream = self._streams.get(run_id)
        if stream is None:
            return []
        async with stream.condition:
            events = [event.model_copy(deep=True) for event in stream.events if event.sequence_number > after_sequence]
            if limit > 0:
                events = events[:limit]
            return events

    async def wait_for_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        timeout_sec: float = 15.0,
        limit: int = 500,
    ) -> list[RuntimeEvent]:
        stream = await self._get_or_create_stream(run_id)
        async with stream.condition:
            current = [event.model_copy(deep=True) for event in stream.events if event.sequence_number > after_sequence]
            if current:
                return current[:limit] if limit > 0 else current
            try:
                await asyncio.wait_for(stream.condition.wait(), timeout=timeout_sec)
            except TimeoutError:
                return []
            refreshed = [event.model_copy(deep=True) for event in stream.events if event.sequence_number > after_sequence]
            if limit > 0:
                refreshed = refreshed[:limit]
            return refreshed

    async def latest_status(self, run_id: str) -> RunStatusSnapshot | None:
        stream = self._streams.get(run_id)
        if stream is None:
            return None
        async with stream.condition:
            if stream.latest_status is None:
                return None
            return stream.latest_status.model_copy(deep=True)

    async def has_run(self, run_id: str) -> bool:
        stream = self._streams.get(run_id)
        if stream is None:
            return False
        async with stream.condition:
            return bool(stream.events)

    async def _get_or_create_stream(self, run_id: str) -> _RunEventStream:
        if run_id in self._streams:
            return self._streams[run_id]
        async with self._streams_lock:
            stream = self._streams.get(run_id)
            if stream is None:
                stream = _RunEventStream()
                self._streams[run_id] = stream
            return stream


@lru_cache
def get_event_bus() -> InMemoryEventBus:
    return InMemoryEventBus()
