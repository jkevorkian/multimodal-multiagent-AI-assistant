from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from app.agents.state import AgentState
from app.core.config import settings
from app.core.context_compaction import ContextCompactor, estimate_agent_state_tokens
from app.core.dependencies import create_test_container, get_container
from app.main import app

client = TestClient(app)


def _parse_sse_events(payload: str) -> list[dict]:
    events: list[dict] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


def test_context_compaction_triggers_and_preserves_pinned_blocks() -> None:
    state = AgentState(
        query="Summarize policy constraints and unresolved tasks.",
        trace={"request_id": "r1", "trace_id": "t1"},
        allowed_tools=["web_search"],
        max_steps=8,
        tool_budget=2,
        analysis_notes=["note " + ("x" * 400), "constraint " + ("y" * 400)],
        retrieved_context=[
            {"source": "doc://a", "chunk_id": 1, "snippet": "alpha " * 90},
            {"source": "doc://b", "chunk_id": 2, "snippet": "beta " * 90},
        ],
        tool_outputs=[{"tool": "web_search", "status": "ok", "result": {"rows": ["v1", "v2"]}}],
        errors=["missing_citation"],
    )
    previous_tokens = estimate_agent_state_tokens(state)
    compactor = ContextCompactor(token_threshold=200, target_token_threshold=100)

    result = compactor.compact_if_needed(state)

    assert result.triggered
    assert result.previous_token_estimate == previous_tokens
    assert result.compacted_token_estimate < previous_tokens
    assert state.context_checkpoints
    checkpoint = state.context_checkpoints[-1]
    pinned = checkpoint["pinned_context"]
    assert state.query in pinned["goals"]
    assert any(item.startswith("doc://") for item in pinned["citations"])
    assert "missing_citation" in pinned["open_tasks"]
    assert state.analysis_notes
    assert state.analysis_notes[0].startswith("context_compacted[")


def test_context_compaction_event_is_emitted_via_runtime_stream() -> None:
    original_threshold = settings.context_compaction_token_threshold
    original_enabled = settings.context_compaction_enabled
    settings.context_compaction_token_threshold = 10
    settings.context_compaction_enabled = True

    container = create_test_container()
    app.dependency_overrides[get_container] = lambda: container
    try:
        run_id = f"run-{uuid4().hex}"
        run_response = client.post(
            "/agents/run",
            json={"query": "Run a long synthesis with constraints and evidence.", "run_id": run_id},
        )
        assert run_response.status_code == 200

        events_response = client.get(f"/runs/{run_id}/events", params={"follow": "false"})
        assert events_response.status_code == 200
        events = _parse_sse_events(events_response.text)
        compaction_events = [
            event
            for event in events
            if event.get("event_type") == "agent.revision.requested"
            and event.get("metadata", {}).get("reason_code") == "context_compaction"
        ]
        assert compaction_events
    finally:
        settings.context_compaction_token_threshold = original_threshold
        settings.context_compaction_enabled = original_enabled
        app.dependency_overrides.clear()
