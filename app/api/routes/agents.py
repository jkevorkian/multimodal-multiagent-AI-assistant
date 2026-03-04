from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import AgentRunRequest, AgentRunResponse, Trace
from app.core.config import settings
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["agents"])


@router.post("/agents/run", response_model=AgentRunResponse)
async def run_agents(
    payload: AgentRunRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> AgentRunResponse:
    enabled_tools = payload.tools or container.tool_registry.list_tools()
    orchestration_state = await container.orchestrator.run(
        query=payload.query,
        trace={"request_id": request.state.request_id, "trace_id": request.state.trace_id},
        allowed_tools=enabled_tools,
        tool_budget=settings.agent_tool_budget,
        max_steps=settings.agent_max_steps,
        resume_from_checkpoint=settings.agent_resume_from_checkpoint,
    )
    return AgentRunResponse(
        answer=orchestration_state.final_answer,
        steps=orchestration_state.steps,
        tool_calls=orchestration_state.tool_calls,
        confidence=orchestration_state.confidence,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
