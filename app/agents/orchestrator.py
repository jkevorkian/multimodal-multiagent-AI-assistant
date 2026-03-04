from __future__ import annotations

from app.agents.analyst_agent import AnalystAgent
from app.agents.answer_agent import AnswerAgent
from app.agents.checkpoint_store import NullCheckpointStore
from app.agents.research_agent import ResearchAgent
from app.agents.state import AgentState


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

        pipeline = [
            ("research_agent", self._research_agent.run),
            ("analyst_agent", self._analyst_agent.run),
            ("answer_agent", self._answer_agent.run),
        ]
        for stage_name, stage_runner in pipeline:
            if stage_name in state.steps:
                continue
            if not state.can_continue():
                state.errors.append(f"max_steps_reached_before_{stage_name}")
                break
            state = await stage_runner(state)
            await self._checkpoint_store.save(trace_id=trace_id, checkpoint=state.snapshot())

        if not state.final_answer:
            state.final_answer = "Unable to produce final answer within current orchestration limits."
            state.confidence = min(state.confidence, 0.15)
            state.clamp_confidence()
        return state
