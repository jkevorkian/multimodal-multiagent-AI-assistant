import asyncio

from app.agents import AgentOrchestrator, AnalystAgent, AnswerAgent, InMemoryCheckpointStore, ResearchAgent
from app.tools.registry import ToolRegistry


class _Retriever:
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:  # noqa: ARG002
        return [{"source": "doc://resume", "snippet": "checkpoint resume behavior", "chunk_id": 0, "offset": 0, "score": 1.0}]


class _LLM:
    async def generate(self, prompt: str, context: list[str] | None = None) -> str:
        return f"answer::{prompt}::{len(context or [])}"


class _CountingTool:
    name = "web_search"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, payload: dict) -> dict:  # noqa: ARG002
        self.calls += 1
        return {"ok": True, "calls": self.calls}


def test_checkpoint_resume_avoids_repeating_completed_tool_calls() -> None:
    counting_tool = _CountingTool()
    checkpoint_store = InMemoryCheckpointStore()
    orchestrator = AgentOrchestrator(
        research_agent=ResearchAgent(
            retriever=_Retriever(),
            tool_registry=ToolRegistry([counting_tool]),
            retrieval_top_k=5,
            tool_timeout_sec=1.0,
            tool_retries=0,
        ),
        analyst_agent=AnalystAgent(),
        answer_agent=AnswerAgent(llm=_LLM()),
        checkpoint_store=checkpoint_store,
        max_steps=6,
    )

    trace = {"request_id": "r-checkpoint", "trace_id": "t-checkpoint"}
    partial_state = asyncio.run(
        orchestrator.run(
            query="please search checkpoint behavior",
            trace=trace,
            allowed_tools=["web_search"],
            tool_budget=2,
            max_steps=1,
            resume_from_checkpoint=False,
        )
    )
    assert partial_state.steps == ["research_agent"]
    assert counting_tool.calls == 1

    resumed_state = asyncio.run(
        orchestrator.run(
            query="please search checkpoint behavior",
            trace=trace,
            allowed_tools=["web_search"],
            tool_budget=2,
            max_steps=6,
            resume_from_checkpoint=True,
        )
    )
    assert resumed_state.steps == ["research_agent", "analyst_agent", "answer_agent"]
    assert counting_tool.calls == 1
    assert resumed_state.final_answer.startswith("answer::")
