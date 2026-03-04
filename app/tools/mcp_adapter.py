from __future__ import annotations

from dataclasses import dataclass

from app.interfaces.tool import Tool


@dataclass
class MCPToolAdapter:
    server_name: str

    def as_tool(self, tool_name: str, description: str = "") -> Tool:
        return _MCPProxyTool(name=f"mcp::{self.server_name}::{tool_name}", description=description)


@dataclass
class _MCPProxyTool:
    name: str
    description: str = ""

    async def run(self, payload: dict) -> dict:
        # Placeholder adapter: M2 introduces protocol boundary without hard dependency on a transport client.
        return {
            "status": "not_implemented",
            "tool": self.name,
            "description": self.description,
            "payload": payload,
        }
