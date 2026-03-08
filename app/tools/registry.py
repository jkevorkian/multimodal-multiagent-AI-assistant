from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from app.interfaces.tool import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe_tools(self) -> list[dict[str, str]]:
        descriptions: list[dict[str, str]] = []
        for name in self.list_tools():
            tool = self._tools[name]
            description = str(getattr(tool, "description", "")).strip() or "No description provided."
            descriptions.append({"name": name, "description": description})
        return descriptions

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    async def run_tool(
        self,
        tool_name: str,
        payload: dict,
        timeout_sec: float = 2.0,
        max_retries: int = 1,
    ) -> dict:
        if tool_name not in self._tools:
            return {"tool": tool_name, "status": "error", "error": "tool_not_found", "attempts": 0}

        attempts = 0
        while True:
            attempts += 1
            started_at = time.perf_counter()
            try:
                result = await asyncio.wait_for(self._tools[tool_name].run(payload), timeout=timeout_sec)
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                return {
                    "tool": tool_name,
                    "status": "ok",
                    "result": result,
                    "attempts": attempts,
                    "latency_ms": latency_ms,
                }
            except asyncio.TimeoutError:
                if attempts > max_retries:
                    latency_ms = int((time.perf_counter() - started_at) * 1000)
                    return {
                        "tool": tool_name,
                        "status": "error",
                        "error": "timeout",
                        "attempts": attempts,
                        "latency_ms": latency_ms,
                    }
                await asyncio.sleep(0.05 * attempts)
            except Exception as exc:
                if attempts > max_retries:
                    latency_ms = int((time.perf_counter() - started_at) * 1000)
                    return {
                        "tool": tool_name,
                        "status": "error",
                        "error": str(exc),
                        "attempts": attempts,
                        "latency_ms": latency_ms,
                    }
                await asyncio.sleep(0.05 * attempts)
