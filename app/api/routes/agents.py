from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import AgentRunRequest, AgentRunResponse, Trace
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["agents"])


@router.post("/agents/run", response_model=AgentRunResponse)
async def run_agents(
    payload: AgentRunRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> AgentRunResponse:
    enabled_tools = payload.tools or [tool.name for tool in container.tools]
    return AgentRunResponse(
        answer="Stub multi-agent answer.",
        steps=["research_agent", "analyst_agent", "answer_agent"],
        tool_calls=enabled_tools,
        confidence=0.2,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )

