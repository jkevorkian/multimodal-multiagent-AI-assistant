import asyncio

from app.agents import AgentOrchestrator, AnalystAgent, AnswerAgent, ResearchAgent
from app.tools.registry import ToolRegistry


class _RetrieverStub:
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:  # noqa: ARG002
        return [
            {
                "source": "doc://policy",
                "snippet": "The policy requires periodic review and control checks.",
                "chunk_id": 0,
                "offset": 0,
                "score": 0.9,
            }
        ][:top_k]


class _LLMStub:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        ctx = " ".join(context or [])
        return f"Answer for {prompt}: {ctx}"


class _WebSearchTool:
    name = "web_search"

    async def run(self, payload: dict) -> dict:
        return {"status": "ok", "source": "web", "payload": payload}


class _SlowTool:
    name = "slow_tool"

    async def run(self, payload: dict) -> dict:  # noqa: ARG002
        await asyncio.sleep(0.05)
        return {"status": "ok"}


def test_orchestrator_state_transition_order() -> None:
    registry = ToolRegistry([_WebSearchTool()])
    orchestrator = AgentOrchestrator(
        research_agent=ResearchAgent(retriever=_RetrieverStub(), tool_registry=registry),
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    state = asyncio.run(
        orchestrator.run(
            query="search compliance policy",
            trace={"request_id": "r1", "trace_id": "t1"},
            allowed_tools=["web_search"],
            tool_budget=2,
        )
    )
    assert state.steps == ["research_agent", "analyst_agent", "answer_agent"]
    assert state.final_answer


def test_tool_selection_scenario_executes_relevant_tool() -> None:
    registry = ToolRegistry([_WebSearchTool()])
    research_agent = ResearchAgent(retriever=_RetrieverStub(), tool_registry=registry)
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    state = asyncio.run(
        orchestrator.run(
            query="Please search latest governance controls",
            trace={"request_id": "r2", "trace_id": "t2"},
            allowed_tools=["web_search", "nonexistent_tool"],
            tool_budget=2,
        )
    )
    assert "web_search" in state.tool_calls
    assert all(call != "nonexistent_tool" for call in state.tool_calls)
    assert any(output.get("status") == "ok" for output in state.tool_outputs)


def test_tool_timeout_and_retry_behavior() -> None:
    registry = ToolRegistry([_SlowTool()])
    result = asyncio.run(
        registry.run_tool(
            tool_name="slow_tool",
            payload={"query": "x"},
            timeout_sec=0.01,
            max_retries=1,
        )
    )
    assert result["status"] == "error"
    assert result["error"] == "timeout"
    assert result["attempts"] == 2
