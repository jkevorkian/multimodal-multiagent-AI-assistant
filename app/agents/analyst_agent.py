from __future__ import annotations

from app.agents.state import AgentState


class AnalystAgent:
    async def run(self, state: AgentState) -> AgentState:
        state.record_step("analyst_agent")

        if state.retrieved_context:
            top_sources = [item.get("source", "unknown") for item in state.retrieved_context[:3]]
            state.analysis_notes.append(f"retrieval_sources={top_sources}")
            state.analysis_notes.append(f"retrieved_chunks={len(state.retrieved_context)}")
        else:
            state.analysis_notes.append("retrieved_chunks=0")

        if state.tool_outputs:
            ok_tools = [item.get("tool", "unknown") for item in state.tool_outputs if item.get("status") == "ok"]
            failed_tools = [item.get("tool", "unknown") for item in state.tool_outputs if item.get("status") != "ok"]
            state.analysis_notes.append(f"tool_ok={ok_tools}")
            if failed_tools:
                state.analysis_notes.append(f"tool_failed={failed_tools}")
        else:
            state.analysis_notes.append("tool_calls=0")

        confidence = 0.2
        confidence += min(0.45, 0.08 * len(state.retrieved_context))
        confidence += min(0.2, 0.05 * len([item for item in state.tool_outputs if item.get("status") == "ok"]))
        confidence -= min(0.2, 0.05 * len([item for item in state.tool_outputs if item.get("status") != "ok"]))
        state.confidence = confidence
        state.clamp_confidence()
        return state
