from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.analyst_agent import AnalystAgent
from app.agents.answer_agent import AnswerAgent
from app.agents.checkpoint_store import NullCheckpointStore
from app.agents.research_agent import ResearchAgent
from app.agents.state import AgentState


class _OrchestratorGraphState(TypedDict):
    agent_state: AgentState


class AgentOrchestrator:
    def __init__(
        self,
        research_agent: ResearchAgent,
        analyst_agent: AnalystAgent,
        answer_agent: AnswerAgent,
        checkpoint_store: NullCheckpointStore | None = None,
        max_steps: int = 6,
    ) -> None:
        self._research_agent = research_agent
        self._analyst_agent = analyst_agent
        self._answer_agent = answer_agent
        self._checkpoint_store = checkpoint_store or NullCheckpointStore()
        self._max_steps = max_steps
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_OrchestratorGraphState)
        graph.add_node("research_agent", self._run_research_node)
        graph.add_node("analyst_agent", self._run_analyst_node)
        graph.add_node("answer_agent", self._run_answer_node)
        graph.add_edge(START, "research_agent")
        graph.add_edge("research_agent", "analyst_agent")
        graph.add_edge("analyst_agent", "answer_agent")
        graph.add_edge("answer_agent", END)
        return graph.compile()

    async def _run_research_node(self, graph_state: _OrchestratorGraphState) -> _OrchestratorGraphState:
        state = graph_state["agent_state"]
        if "research_agent" in state.steps:
            return {"agent_state": state}
        if not state.can_continue():
            state.errors.append("max_steps_reached_before_research_agent")
            return {"agent_state": state}
        state = await self._research_agent.run(state)
        await self._save_checkpoint(state)
        return {"agent_state": state}

    async def _run_analyst_node(self, graph_state: _OrchestratorGraphState) -> _OrchestratorGraphState:
        state = graph_state["agent_state"]
        if "analyst_agent" in state.steps:
            return {"agent_state": state}
        if not state.can_continue():
            state.errors.append("max_steps_reached_before_analyst_agent")
            return {"agent_state": state}
        state = await self._analyst_agent.run(state)
        await self._save_checkpoint(state)
        return {"agent_state": state}

    async def _run_answer_node(self, graph_state: _OrchestratorGraphState) -> _OrchestratorGraphState:
        state = graph_state["agent_state"]
        if "answer_agent" in state.steps:
            return {"agent_state": state}
        if not state.can_continue():
            state.errors.append("max_steps_reached_before_answer_agent")
            return {"agent_state": state}
        state = await self._answer_agent.run(state)
        await self._save_checkpoint(state)
        return {"agent_state": state}

    async def _save_checkpoint(self, state: AgentState) -> None:
        trace_id = state.trace.get("trace_id", "unknown")
        await self._checkpoint_store.save(trace_id=trace_id, checkpoint=state.snapshot())

    async def run(
        self,
        query: str,
        trace: dict[str, str],
        allowed_tools: list[str],
        tool_budget: int = 2,
        max_steps: int | None = None,
        resume_from_checkpoint: bool = False,
        ) -> AgentState:
        trace_id = trace.get("trace_id", "unknown")
        resolved_max_steps = max_steps if max_steps is not None else self._max_steps
        state: AgentState
        if resume_from_checkpoint:
            latest = await self._checkpoint_store.latest(trace_id=trace_id)
            if latest and str(latest.get("query", "")) == query:
                state = AgentState.from_snapshot(latest)
            else:
                state = AgentState(
                    query=query,
                    trace=trace,
                    allowed_tools=allowed_tools,
                    tool_budget=tool_budget,
                    max_steps=resolved_max_steps,
                )
        else:
            state = AgentState(
                query=query,
                trace=trace,
                allowed_tools=allowed_tools,
                tool_budget=tool_budget,
                max_steps=resolved_max_steps,
            )

        state.max_steps = resolved_max_steps
        state.trace = trace
        state.allowed_tools = allowed_tools
        result = await self._graph.ainvoke({"agent_state": state})
        state = result["agent_state"]

        if not state.final_answer:
            state.final_answer = "Unable to produce final answer within current orchestration limits."
            state.confidence = min(state.confidence, 0.15)
            state.clamp_confidence()
        return state
