import asyncio

from app.agents import AgentOrchestrator, AgentState, AnalystAgent, AnswerAgent, ResearchAgent
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


class _CapturingLLMStub:
    def __init__(self) -> None:
        self.last_context: list[str] = []

    async def generate(self, prompt: str, context: list[str] | None = None) -> str:  # noqa: ARG002
        self.last_context = list(context or [])
        return "captured"


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


def test_research_agent_retries_empty_retrieval_for_video_text_intent() -> None:
    class _RetryingRetrieverStub:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def retrieve(
            self,
            query: str,
            top_k: int = 5,
            metadata_filter: dict | None = None,
        ) -> list[dict]:
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "metadata_filter": dict(metadata_filter or {}),
                }
            )
            if len(self.calls) == 1:
                return []
            return [
                {
                    "source": "file:///tmp/food.mp4",
                    "snippet": "Audio event: me gustan los tacos.",
                    "chunk_id": 7,
                    "offset": 70,
                    "score": 0.82,
                    "modality": "video",
                    "timestamp_sec": 1.9,
                }
            ]

    registry = ToolRegistry([])
    retriever = _RetryingRetrieverStub()
    research_agent = ResearchAgent(
        retriever=retriever,  # type: ignore[arg-type]
        tool_registry=registry,
        retry_on_empty_retrieval=True,
        retry_empty_retrieval_top_k=9,
    )
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    state = asyncio.run(
        orchestrator.run(
            query="Whats his opinion on mexican food in this video? Check the text in the video.",
            trace={"request_id": "r6", "trace_id": "t6"},
            allowed_tools=[],
            tool_budget=1,
        )
    )
    assert len(retriever.calls) >= 2
    assert retriever.calls[1]["metadata_filter"].get("modality") == "video"
    assert state.retrieved_context
    assert any("retrieval_retry_applied" in note for note in state.analysis_notes)


def test_research_agent_prioritizes_transcript_rows_in_tool_payload_context() -> None:
    class _MixedVideoRetrieverStub:
        async def retrieve(
            self,
            query: str,  # noqa: ARG002
            top_k: int = 5,
            metadata_filter: dict | None = None,  # noqa: ARG002
        ) -> list[dict]:
            rows: list[dict] = []
            for index in range(10):
                rows.append(
                    {
                        "source": "file:///tmp/mixed.mp4",
                        "snippet": f"Visual event {index}",
                        "chunk_id": index,
                        "offset": index * 50,
                        "score": 0.95 - (index * 0.01),
                        "modality": "video",
                    }
                )
            rows.append(
                {
                    "source": "file:///tmp/mixed.mp4",
                    "snippet": "Audio event: Me gustan los tacos",
                    "chunk_id": 999,
                    "offset": 9_999,
                    "score": 0.1,
                    "modality": "video",
                }
            )
            return rows[:top_k]

    class _VideoProbeEchoTool:
        name = "video_probe"

        async def run(self, payload: dict) -> dict:
            return {
                "status": "ok",
                "payload_context": list(payload.get("retrieved_context", [])),
            }

    registry = ToolRegistry([_VideoProbeEchoTool()])
    research_agent = ResearchAgent(retriever=_MixedVideoRetrieverStub(), tool_registry=registry)  # type: ignore[arg-type]
    orchestrator = AgentOrchestrator(
        research_agent=research_agent,
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLMStub()),
        max_steps=6,
    )
    state = asyncio.run(
        orchestrator.run(
            query="In this video, what is said?",
            trace={"request_id": "r7a", "trace_id": "t7a"},
            allowed_tools=["video_probe"],
            tool_budget=1,
            retrieval_top_k=12,
        )
    )
    assert state.tool_outputs
    first = state.tool_outputs[0]
    payload_context = first.get("result", {}).get("payload_context", [])
    assert isinstance(payload_context, list)
    assert payload_context
    assert str(payload_context[0].get("snippet", "")).lower().startswith("audio event:")


def test_answer_agent_prioritizes_transcript_and_tool_outputs_in_context() -> None:
    llm = _CapturingLLMStub()
    agent = AnswerAgent(llm=llm)
    state = AgentState(
        query="What does he say about tacos?",
        trace={"request_id": "r7", "trace_id": "t7"},
        allowed_tools=["web_search", "video_probe"],
    )
    state.retrieved_context = [
        {"source": "file:///tmp/video.mp4", "chunk_id": 1, "snippet": "Visual event one.", "modality": "video"},
        {"source": "file:///tmp/video.mp4", "chunk_id": 2, "snippet": "Visual event two.", "modality": "video"},
        {"source": "file:///tmp/video.mp4", "chunk_id": 3, "snippet": "Visual event three.", "modality": "video"},
        {"source": "file:///tmp/video.mp4", "chunk_id": 4, "snippet": "Visual event four.", "modality": "video"},
        {"source": "file:///tmp/video.mp4", "chunk_id": 5, "snippet": "Visual event five.", "modality": "video"},
        {
            "source": "file:///tmp/video.mp4",
            "chunk_id": 6,
            "snippet": "Audio event: Me gustan los tacos",
            "modality": "video",
        },
    ]
    state.tool_outputs = [
        {
            "tool": "web_search",
            "status": "ok",
            "result": {
                "query": "mexican food thumbs up meaning",
                "provider": "duckduckgo_html",
                "results": [
                    {
                        "title": "Thumbs up",
                        "snippet": "Thumbs up is usually a positive signal.",
                        "url": "https://example.com/thumbs-up",
                    }
                ],
            },
        }
    ]
    state.analysis_notes = ["retrieved_chunks=6", "tool_ok=['web_search']"]

    state = asyncio.run(agent.run(state))
    assert state.final_answer == "captured"
    joined = " ".join(llm.last_context).lower()
    assert "audio event: me gustan los tacos" in joined
    assert "tool:web_search" in joined
    assert "thumbs up is usually a positive signal" in joined
