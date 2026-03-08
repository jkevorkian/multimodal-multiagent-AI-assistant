from frontend.architecture import (
    agent_loop_guardrail_rows,
    agent_pipeline_state_rows,
    build_agents_revision_loop_dot,
    build_agents_pipeline_dot,
    build_architecture_dot,
    high_level_flow_points,
    live_status_event_rows,
)


def test_architecture_dot_contains_core_nodes() -> None:
    dot = build_architecture_dot()
    assert "Streamlit Frontend" in dot
    assert "FastAPI Backend" in dot
    assert "AgentOrchestrator" in dot
    assert "Hybrid Retriever" in dot
    assert "VisionPreprocessor" in dot
    assert "VisionFusion" in dot
    assert "Per-frame Vision Analysis" in dot
    assert "ContextCompactor" in dot
    assert "SteeringPolicy" in dot
    assert "LLM Provider Selector" in dot


def test_high_level_flow_points_present() -> None:
    points = high_level_flow_points()
    assert len(points) >= 5
    assert any("Streamlit" in point for point in points)
    assert any("RAG" in point for point in points)
    assert any("context compaction" in point.lower() for point in points)


def test_agents_pipeline_dot_contains_core_nodes() -> None:
    dot = build_agents_pipeline_dot()
    assert "ResearchAgent" in dot
    assert "AnalystAgent" in dot
    assert "AnswerAgent" in dot
    assert "ToolRegistry" in dot
    assert "START" in dot
    assert "END" in dot


def test_agents_pipeline_state_rows_shape() -> None:
    rows = agent_pipeline_state_rows()
    assert len(rows) == 3
    assert rows[0]["stage"] == "research_agent"
    assert all("state_updates" in row for row in rows)


def test_agents_revision_loop_dot_contains_guardrails() -> None:
    dot = build_agents_revision_loop_dot()
    assert "Critic / Verifier" in dot
    assert "Revision Gate" in dot
    assert "Max Step Guard" in dot
    assert "SAFE_EXIT" in dot


def test_planned_runtime_rows_are_non_empty() -> None:
    loop_rows = agent_loop_guardrail_rows()
    status_rows = live_status_event_rows()
    assert len(loop_rows) >= 4
    assert len(status_rows) >= 6
    assert any(row["guardrail"] == "max_steps" for row in loop_rows)
    assert any(row["event_type"] == "model.call.in_progress" for row in status_rows)
