import asyncio

from app.agents import AgentOrchestrator, AnalystAgent, AnswerAgent, ResearchAgent
from app.tools.registry import ToolRegistry


class _RetrieverStub:
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,  # noqa: ARG002
    ) -> list[dict]:
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


def test_research_agent_prefers_rag_debug_when_no_intent_match() -> None:
    class _RAGDebugTool:
        name = "rag_debug"

        async def run(self, payload: dict) -> dict:  # noqa: ARG002
            return {"status": "ok", "tool": "rag_debug"}

    class _SystemMetricsTool:
        name = "system_metrics"

        async def run(self, payload: dict) -> dict:  # noqa: ARG002
            return {"status": "ok", "tool": "system_metrics"}

    registry = ToolRegistry([_RAGDebugTool(), _SystemMetricsTool()])
    research_agent = ResearchAgent(retriever=_RetrieverStub(), tool_registry=registry)
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    state = asyncio.run(
        orchestrator.run(
            query="Analyze this",
            trace={"request_id": "r3", "trace_id": "t3"},
            allowed_tools=["rag_debug", "system_metrics"],
            tool_budget=1,
        )
    )
    assert state.tool_calls == ["rag_debug"]


def test_research_agent_selects_video_probe_for_video_clarification_query() -> None:
    class _VideoRetrieverStub:
        async def retrieve(
            self,
            query: str,  # noqa: ARG002
            top_k: int = 5,  # noqa: ARG002
            metadata_filter: dict | None = None,  # noqa: ARG002
        ) -> list[dict]:
            return [
                {
                    "source": "file:///tmp/demo.mp4",
                    "snippet": "Video source metadata only.",
                    "chunk_id": 0,
                    "offset": 0,
                    "score": 0.77,
                    "modality": "video",
                    "timestamp_sec": 8.0,
                }
            ]

    class _VideoProbeTool:
        name = "video_probe"

        async def run(self, payload: dict) -> dict:
            return {
                "status": "ok",
                "selected_source": str(payload.get("retrieved_context", [{}])[0].get("source", "")),
            }

    registry = ToolRegistry([_VideoProbeTool()])
    research_agent = ResearchAgent(retriever=_VideoRetrieverStub(), tool_registry=registry)  # type: ignore[arg-type]
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    state = asyncio.run(
        orchestrator.run(
            query="In this video, what happens and what is said?",
            trace={"request_id": "r4", "trace_id": "t4"},
            allowed_tools=["video_probe"],
            tool_budget=1,
        )
    )
    assert state.tool_calls == ["video_probe"]
    assert state.tool_outputs
    first = state.tool_outputs[0]
    assert first.get("status") == "ok"


def test_research_agent_extracts_effective_user_query_for_tool_payload() -> None:
    class _PayloadEchoWebSearchTool:
        name = "web_search"

        async def run(self, payload: dict) -> dict:
            return {
                "status": "ok",
                "effective_query": str(payload.get("query", "")),
                "raw_orchestration_query": str(payload.get("orchestration_query", "")),
            }

    registry = ToolRegistry([_PayloadEchoWebSearchTool()])
    research_agent = ResearchAgent(retriever=_RetrieverStub(), tool_registry=registry)
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    composed_query = (
        "Relevant past conversation snippets:\n"
        "- I apologize, but web search failed.\n\n"
        "Conversation context (most recent turns):\n"
        "USER: Hey search online about NTT data\n"
        "ASSISTANT: I could not complete web search.\n"
        "USER: try again\n\n"
        "Current user request:\n"
        "try again"
    )
    state = asyncio.run(
        orchestrator.run(
            query=composed_query,
            trace={"request_id": "r5", "trace_id": "t5"},
            allowed_tools=["web_search"],
            tool_budget=1,
        )
    )
    assert state.tool_calls == ["web_search"]
    assert state.tool_outputs
    first = state.tool_outputs[0]
    assert first.get("status") == "ok"
    result = first.get("result", {})
    assert "ntt data" in str(result.get("effective_query", "")).lower()
    assert "current user request" not in str(result.get("effective_query", "")).lower()
