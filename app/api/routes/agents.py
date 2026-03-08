from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import AgentRunRequest, AgentRunResponse, AgentToolInfo, AgentToolsResponse, SteeringApplied, Trace
from app.core.config import settings
from app.core.dependencies import ServiceContainer, get_container
from app.core.steering import (
    apply_answer_style,
    apply_tool_policy,
    enforce_grounding_policy,
    resolve_profile,
)

router = APIRouter(tags=["agents"])


@router.post("/agents/run", response_model=AgentRunResponse)
async def run_agents(
    payload: AgentRunRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> AgentRunResponse:
    run_id = payload.run_id or str(uuid4())
    steering_resolution = resolve_profile(payload.steering)
    available_tools = container.tool_registry.list_tools()
    enabled_tools, tool_policy_notes = apply_tool_policy(
        available_tools=available_tools,
        requested_tools=payload.tools,
        steering=payload.steering,
    )
    orchestration_state = await container.orchestrator.run(
        query=payload.query,
        trace={"request_id": request.state.request_id, "trace_id": request.state.trace_id},
        run_id=run_id,
        allowed_tools=enabled_tools,
        tool_budget=settings.agent_tool_budget,
        max_steps=settings.agent_max_steps,
        resume_from_checkpoint=settings.agent_resume_from_checkpoint,
    )
    citations = [
        f"{item['source']}#chunk-{item['chunk_id']}"
        for item in orchestration_state.retrieved_context
        if "source" in item and "chunk_id" in item
    ]
    grounded_answer, grounding_notes = enforce_grounding_policy(
        answer=orchestration_state.final_answer,
        citations=citations,
        profile=steering_resolution.profile,
        steering=payload.steering,
    )
    answer = apply_answer_style(grounded_answer, steering_resolution.profile)
    confidence = orchestration_state.confidence
    if grounded_answer != orchestration_state.final_answer:
        confidence = min(confidence, 0.2)
    steering_notes = [*steering_resolution.notes, *tool_policy_notes, *grounding_notes]
    return AgentRunResponse(
        run_id=run_id,
        answer=answer,
        steps=orchestration_state.steps,
        tool_calls=orchestration_state.tool_calls,
        confidence=confidence,
        steering_applied=SteeringApplied(profile=steering_resolution.profile, notes=steering_notes),
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )


@router.get("/agents/tools", response_model=AgentToolsResponse)
async def list_agent_tools(
    container: ServiceContainer = Depends(get_container),
) -> AgentToolsResponse:
    tools = [AgentToolInfo(**tool) for tool in container.tool_registry.describe_tools()]
    return AgentToolsResponse(tools=tools)
