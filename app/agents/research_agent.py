from __future__ import annotations

import re

from app.agents.state import AgentState
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
    ) -> None:
        self._retriever = retriever
        self._tool_registry = tool_registry
        self._retrieval_top_k = retrieval_top_k
        self._tool_timeout_sec = tool_timeout_sec
        self._tool_retries = tool_retries

    async def run(self, state: AgentState) -> AgentState:
        state.record_step("research_agent")

        try:
            state.retrieved_context = await self._retriever.retrieve(state.query, top_k=self._retrieval_top_k)
        except Exception as exc:
            state.errors.append(f"retrieval_error:{exc}")
            state.retrieved_context = []

        selected_tools = self._select_tools(state.query, state.allowed_tools)
        for tool_name in selected_tools:
            if not state.consume_tool_budget():
                break
            state.record_tool_call(tool_name)
            result = await self._tool_registry.run_tool(
                tool_name=tool_name,
                payload={"query": state.query, "retrieved_count": len(state.retrieved_context)},
                timeout_sec=self._tool_timeout_sec,
                max_retries=self._tool_retries,
            )
            state.tool_outputs.append(result)
            if result.get("status") != "ok":
                state.errors.append(f"tool_error:{tool_name}:{result.get('error', 'unknown')}")

        return state

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
