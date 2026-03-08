from __future__ import annotations

import re

from app.agents.state import AgentState
from app.core.event_bus import InMemoryEventBus
from app.interfaces.retriever import Retriever
from app.tools.registry import ToolRegistry


class ResearchAgent:
    def __init__(
        self,
        retriever: Retriever,
        tool_registry: ToolRegistry,
        retrieval_top_k: int = 5,
        tool_timeout_sec: float = 2.0,
        tool_retries: int = 1,
        event_bus: InMemoryEventBus | None = None,
    ) -> None:
        self._retriever = retriever
        self._tool_registry = tool_registry
        self._retrieval_top_k = retrieval_top_k
        self._tool_timeout_sec = tool_timeout_sec
        self._tool_retries = tool_retries
        self._event_bus = event_bus

    async def run(self, state: AgentState) -> AgentState:
        state.record_step("research_agent")

        try:
            state.retrieved_context = await self._retriever.retrieve(
                state.query,
                top_k=self._retrieval_top_k,
                metadata_filter=state.retrieval_filter or None,
            )
        except Exception as exc:
            state.errors.append(f"retrieval_error:{exc}")
            state.retrieved_context = []

        selected_tools = self._select_tools(state.query, state.allowed_tools)
        for tool_name in selected_tools:
            if not state.consume_tool_budget():
                break
            state.record_tool_call(tool_name)
            await self._emit_tool_event(
                state=state,
                event_type="tool.call.started",
                tool_name=tool_name,
                status_text=f"Running tool: {tool_name}",
                metadata={
                    "timeout_sec": self._tool_timeout_sec,
                    "max_retries": self._tool_retries,
                    "tool_budget_remaining": state.tool_budget,
                },
            )
            result = await self._tool_registry.run_tool(
                tool_name=tool_name,
                payload={"query": state.query, "retrieved_count": len(state.retrieved_context)},
                timeout_sec=self._tool_timeout_sec,
                max_retries=self._tool_retries,
            )
            state.tool_outputs.append(result)
            if result.get("status") == "ok":
                await self._emit_tool_event(
                    state=state,
                    event_type="tool.call.completed",
                    tool_name=tool_name,
                    status_text=f"Tool completed: {tool_name}",
                    metadata={
                        "attempts": int(result.get("attempts", 0)),
                        "latency_ms": int(result.get("latency_ms", 0)),
                    },
                )
            else:
                tool_error = str(result.get("error", "unknown"))
                state.errors.append(f"tool_error:{tool_name}:{tool_error}")
                await self._emit_tool_event(
                    state=state,
                    event_type="tool.call.failed",
                    tool_name=tool_name,
                    status_text=f"Tool failed: {tool_name}",
                    metadata={
                        "attempts": int(result.get("attempts", 0)),
                        "latency_ms": int(result.get("latency_ms", 0)),
                        "error": tool_error,
                    },
                )

        return state

    async def _emit_tool_event(
        self,
        *,
        state: AgentState,
        event_type: str,
        tool_name: str,
        status_text: str,
        metadata: dict,
    ) -> None:
        if self._event_bus is None:
            return
        run_id = state.trace.get("run_id")
        trace_id = state.trace.get("trace_id", "unknown")
        if not run_id:
            return
        await self._event_bus.emit(
            run_id=run_id,
            trace_id=trace_id,
            event_type=event_type,
            status_text=status_text,
            agent="research_agent",
            tool=tool_name,
            metadata=metadata,
        )

    def _select_tools(self, query: str, allowed_tools: list[str]) -> list[str]:
        available = [name for name in allowed_tools if self._tool_registry.has_tool(name)]
        if not available:
            return []

        lowered = query.lower()
        query_terms = set(re.findall(r"[a-z0-9_]+", lowered))
        prioritized = [
            name
            for name in available
            if any(token in name.lower() for token in query_terms)
            or any(keyword in lowered for keyword in ("search", "lookup", "api", "latest", "tool", "find"))
        ]
        if prioritized:
            return prioritized
        return available[:1]
