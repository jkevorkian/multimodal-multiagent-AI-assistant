from __future__ import annotations

import asyncio
from collections.abc import Iterable

from app.interfaces.tool import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

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
            try:
                result = await asyncio.wait_for(self._tools[tool_name].run(payload), timeout=timeout_sec)
                return {"tool": tool_name, "status": "ok", "result": result, "attempts": attempts}
            except asyncio.TimeoutError:
                if attempts > max_retries:
                    return {"tool": tool_name, "status": "error", "error": "timeout", "attempts": attempts}
                await asyncio.sleep(0.05 * attempts)
            except Exception as exc:
                if attempts > max_retries:
                    return {"tool": tool_name, "status": "error", "error": str(exc), "attempts": attempts}
                await asyncio.sleep(0.05 * attempts)
